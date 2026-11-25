"""Audited local chat completions with native tool calling for GenLoc."""
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .batch import write_json
from .embeddings import digest
from .snapshot import sha256_file


def request_json(endpoint, path, payload=None, timeout=600):
    request = urllib.request.Request(endpoint.rstrip('/') + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')[:8192]
        raise RuntimeError(f'Local inference HTTP {error.code}: {detail}') from error


class LocalToolGenerator:
    """Preserve OpenAI message/tool semantics while using only a local model.

    ``response_schema`` is the JSON schema itself, without response_format or
    json_schema wrappers. The native server template and tokenizer determine
    the context budget, including tool definitions and all tool results.
    """

    def __init__(self, project, config):
        self.project = Path(project)
        self.config = dict(config)
        self.endpoint = config['inference_endpoint'].rstrip('/')
        parsed = urlparse(self.endpoint)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/')):
            raise ValueError('GenLoc inference endpoint must be a local HTTP server root')
        manifest_path = self.project / config['server_manifest']
        manifest = json.loads(manifest_path.read_text())
        props = request_json(self.endpoint, '/props', timeout=10)
        if Path(props['model_path']).resolve() != Path(manifest['gguf_path']).resolve():
            raise ValueError('GenLoc server model path mismatch')
        if sha256_file(Path(manifest['gguf_path'])) != manifest['gguf_sha256']:
            raise ValueError('GenLoc BF16 model checksum mismatch')
        for name, checksum in manifest['source_files_sha256'].items():
            if sha256_file(self.project / config['generation_model'] / name) != checksum:
                raise ValueError('GenLoc model/tokenizer source mismatch: ' + name)
        if props.get('total_slots') != 1 or manifest['slots'] != 1:
            raise ValueError('GenLoc requires its dedicated single-slot server')
        if props.get('default_generation_settings', {}).get('n_ctx') != config['max_context_tokens']:
            raise ValueError('Server context differs from the frozen GenLoc configuration')
        if not manifest.get('no_context_shift') or manifest.get('thinking') is not False:
            raise ValueError('GenLoc requires disabled context shifting and thinking')
        self.model_alias = manifest['model_alias']
        settings = {key: value for key, value in config.items() if key not in
                    ('selection_manifest', 'repositories', 'limit', 'inference_endpoint',
                     'server_manifest', 'stop_after_failures')}
        self.identity = {
            'backend': 'llama.cpp-native-chat-tools', 'model_sha256': manifest['gguf_sha256'],
            'manifest_sha256': sha256_file(manifest_path),
            'source_files_sha256': manifest['source_files_sha256'],
            'server_build': props.get('build_info'), 'settings': settings,
            'sampling': 'request-derived seed; temperature only; no top-k/top-p filtering',
            'thinking': False, 'weights': 'BF16 unquantized', 'kv_cache': 'F16',
            'context_accounting': 'native /apply-template plus /tokenize, no truncation',
            'schema_tool_adaptation': 'Final schema only with tool_choice=none; this server otherwise suppresses native tool calls',
        }
        self.server_properties = props
        self.calls = 0
        self.folder = None

    def start_case(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def _count_payload_tokens(self, payload):
        rendered = request_json(self.endpoint, '/apply-template', payload, timeout=60)
        tokenized = request_json(self.endpoint, '/tokenize',
            {'content': rendered['prompt'], 'add_special': True, 'parse_special': True}, timeout=60)
        return len(tokenized['tokens']), rendered['prompt']

    def complete(self, messages, tools, tool_choice, response_schema=None, stage='explore'):
        if self.folder is None:
            raise ValueError('Call start_case before GenLoc inference')
        if response_schema is not None and tool_choice != 'none':
            raise ValueError('GenLoc schema requires tool_choice=none; this server overrides tool calling with a schema')
        self.calls += 1
        if self.calls > self.config.get('max_calls', 9):
            raise ValueError('GenLoc per-case model call budget exceeded')
        request = {'messages': messages, 'tools': tools, 'tool_choice': tool_choice,
                   'response_schema': response_schema, 'stage': stage,
                   'temperature': self.config.get('temperature', 1.0),
                   'max_new_tokens': self.config['max_new_tokens'],
                   'identity': digest(self.identity)}
        key = digest(request)
        target = self.folder / f'call_{self.calls:03d}.json'
        if target.exists():
            cached = json.loads(target.read_text())
            if cached.get('request_sha256') != key or cached.get('request') != request:
                raise ValueError('GenLoc inference checkpoint identity mismatch')
            if cached.get('status') != 'ok':
                raise ValueError('Previous GenLoc inference failed; use an explicit new run')
            return cached['response']
        seed = (self.config['seed'] + int(key[:8], 16)) % (2 ** 32)
        payload = {'model': self.model_alias, 'messages': messages, 'tools': tools,
                   'tool_choice': tool_choice, 'parallel_tool_calls': True,
                   'max_tokens': self.config['max_new_tokens'],
                   'temperature': self.config.get('temperature', 1.0),
                   'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'repeat_penalty': 1.0,
                   'samplers': ['temperature'], 'seed': seed, 'stream': False,
                   'cache_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
        if response_schema is not None:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'genloc_ranked_files', 'strict': True, 'schema': response_schema}}
        started = time.monotonic()
        record = {'request': request, 'request_sha256': key, 'seed': seed,
                  'payload': payload, 'status': 'running'}
        write_json(target, record)
        try:
            input_tokens, prompt = self._count_payload_tokens(payload)
            record.update(input_tokens=input_tokens, rendered_prompt=prompt)
            if input_tokens + self.config['max_new_tokens'] > self.config['max_context_tokens']:
                raise ValueError('GenLoc context overflow; no silent truncation')
            reply = request_json(self.endpoint, '/v1/chat/completions', payload,
                                 timeout=self.config.get('request_timeout', 600))
            record.update(server_response=reply, usage=reply.get('usage', {}),
                          output_tokens=reply.get('usage', {}).get('completion_tokens', 0),
                          server_timings=reply.get('timings', {}))
            choices = reply.get('choices', [])
            if len(choices) != 1:
                raise ValueError('GenLoc expected exactly one assistant completion')
            choice = choices[0]
            if reply.get('truncated') or choice.get('finish_reason') not in ('stop', 'tool_calls'):
                raise ValueError('GenLoc inference truncated or exhausted its output budget')
            # The API usage is a second check against unintended template changes.
            reported_tokens = reply.get('usage', {}).get('prompt_tokens')
            if reported_tokens is not None and reported_tokens != input_tokens:
                raise ValueError('GenLoc native prompt token count mismatch')
            message = choice['message']
            if message.get('role') != 'assistant':
                raise ValueError('GenLoc received a non-assistant response')
            calls = message.get('tool_calls', [])
            if tool_choice == 'required' and not calls:
                raise ValueError('GenLoc required tool call missing')
            if tool_choice == 'none' and calls:
                raise ValueError('GenLoc received a tool call during final-only output')
            for call in calls:
                if not call.get('id') or call.get('type') != 'function':
                    raise ValueError('Malformed GenLoc native tool call')
                arguments = json.loads(call['function']['arguments'])
                if not isinstance(arguments, dict):
                    raise ValueError('GenLoc tool arguments must be a JSON object')
            if not calls:
                content = message.get('content')
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('Empty GenLoc assistant response')
                if response_schema is not None:
                    import jsonschema
                    jsonschema.validate(json.loads(content), response_schema)
            record.update(status='ok', response=message)
            return message
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            write_json(target, record)
