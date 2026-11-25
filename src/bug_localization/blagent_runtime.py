"""Shared llama.cpp BF16 inference with two independently seeded request slots."""
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlparse
from .batch import write_json
from .embeddings import digest
from .snapshot import sha256_file


def request_json(endpoint, path, payload=None, timeout=600):
    request = urllib.request.Request(endpoint.rstrip('/')+path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class ServerGenerator:
    def __init__(self, project, config):
        from transformers import AutoTokenizer
        self.config = config
        self.endpoint = config['inference_endpoint']
        if urlparse(self.endpoint).hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('BLAgent server must be local')
        self.tokenizer = AutoTokenizer.from_pretrained(project/config['generation_model'],local_files_only=True,trust_remote_code=False)
        manifest_path = project/config['server_manifest']
        manifest = json.loads(manifest_path.read_text())
        props = request_json(self.endpoint,'/props',timeout=10)
        if Path(props['model_path']).resolve() != Path(manifest['gguf_path']).resolve():
            raise ValueError('BLAgent server model path mismatch')
        if sha256_file(Path(manifest['gguf_path'])) != manifest['gguf_sha256']:
            raise ValueError('BLAgent GGUF checksum mismatch')
        for name, checksum in manifest['source_files_sha256'].items():
            if sha256_file(project/config['generation_model']/name) != checksum:
                raise ValueError('BLAgent model/tokenizer source mismatch: '+name)
        if props.get('total_slots') != config['server_slots']:
            raise ValueError('Unexpected inference slot count')
        if props.get('default_generation_settings', {}).get('n_ctx') != config['max_context_tokens']:
            raise ValueError('Inference per-slot context differs from the frozen configuration')
        # Transport/selection/output scheduling never enters the sampling key.
        settings = {k:v for k,v in config.items() if k not in
                    ('selection_manifest','repositories','limit','inference_endpoint','server_manifest','stop_after_failures')}
        self.identity = {'backend':'llama.cpp', 'manifest_sha256':sha256_file(manifest_path),
                         'model_sha256':manifest['gguf_sha256'], 'source_files_sha256':manifest['source_files_sha256'],
                         'server_build':props.get('build_info'), 'settings':settings,
                         'sampling':'request hash seed independent of worker and scheduling',
                         'thinking':False,'weights':'BF16 unquantized','kv_cache':'F16'}
        self.server_properties = props
        self.calls = 0

    def start_case(self, folder):
        self.folder = folder
        self.calls = 0

    def render(self, messages):
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)

    def count_tokens(self, messages):
        return len(self.tokenizer.encode(self.render(messages),add_special_tokens=False))

    def complete(self, messages, *, temperature=0, stop=None, stage='generation', response_schema=None):
        self.calls += 1
        if self.calls > self.config['max_calls']:
            raise ValueError('BLAgent per-case model call budget exceeded')
        request = {'messages':messages,'temperature':temperature,'stop':stop,'stage':stage,
                   'max_new_tokens':self.config['max_new_tokens'],'identity':digest(self.identity)}
        if response_schema is not None:
            request['response_schema'] = response_schema
        key = digest(request)
        target = self.folder/f'call_{self.calls:03d}.json'
        if target.exists():
            cached = json.loads(target.read_text())
            if cached['request'] != request or cached['request_sha256'] != key:
                raise ValueError('BLAgent server call checkpoint identity mismatch')
            if cached['status'] != 'ok':
                raise ValueError('Previous inference call failed; explicit new run required')
            return cached['response']
        text = self.render(messages)
        tokens = self.tokenizer.encode(text,add_special_tokens=False)
        started = time.monotonic()
        record = {'request':request,'request_sha256':key,'input_tokens':len(tokens),'status':'running'}
        write_json(target,record)
        try:
            if len(tokens)+self.config['max_new_tokens'] > self.config['max_context_tokens']:
                raise ValueError('BLAgent context overflow; no silent truncation')
            # Pass IDs to eliminate tokenizer/chat-template differences between
            # clients and server. GGUF tokenizer IDs are checked in server prep.
            payload = {'prompt':tokens,'n_predict':self.config['max_new_tokens'],
                       'temperature':temperature,'top_p':0.9,'top_k':40,'min_p':0,
                       'repeat_penalty':1.0,'samplers':['top_k','top_p','temperature'],
                       'seed':(self.config['seed']+int(key[:8],16))%(2**32),
                       'stop':stop or [],'stream':False,'cache_prompt':True}
            if response_schema is not None:
                payload['json_schema'] = response_schema
            reply = request_json(self.endpoint,'/completion',payload,timeout=self.config.get('request_timeout',600))
            record.update(server_response=reply,output_tokens=reply.get('tokens_predicted',0),
                          raw_response=reply.get('content',''),server_timings=reply.get('timings',{}),
                          seed=payload['seed'])
            if reply.get('truncated') or reply.get('stop_type') not in ('eos','word'):
                raise ValueError('Inference truncated or exhausted the output budget')
            content = reply['content'].strip()
            if not content:
                raise ValueError('Empty model response')
            record.update(status='ok',response=content,finished=True)
            return content
        except Exception as error:
            record.update(status='failed',error_type=type(error).__name__,error=str(error))
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic()-started
            write_json(target,record)

    def chat_completion(self, model_name, prompt, system_prompt, temperature):
        return {'content':self.complete([{'role':'system','content':system_prompt},
                                        {'role':'user','content':prompt}],temperature=temperature,stage='query')}
