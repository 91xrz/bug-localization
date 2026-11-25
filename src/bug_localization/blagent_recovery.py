"""Bounded recovery for the BLAgent adapter, without gold or candidate expansion."""
import json
import re
from itertools import permutations
from .batch import write_json


def inspection_required_parser(original, inspected, case):
    """Return early final answers to the same bounded ReAct loop as observations."""
    from langchain.agents import AgentOutputParser
    from langchain_core.agents import AgentFinish
    from langchain_core.exceptions import OutputParserException
    events = []
    class InspectionRequiredParser(AgentOutputParser):
        def parse(self, text):
            result = original.parse(text)
            if isinstance(result, AgentFinish) and not inspected:
                observation = ('No successful ReadFileSkeleton inspection has occurred. '
                               'Before giving a Final Answer, call ReadFileSkeleton with one exact candidate path '
                               'chosen from the bug report, then use its actual Observation to assign scores.')
                events.append({'rejected_output':text,'observation':observation})
                write_json(case/'early_final_rejections.json',events)
                raise OutputParserException(observation, observation=observation,
                                            llm_output=text, send_to_llm=True)
            return result
        def get_format_instructions(self):
            return original.get_format_instructions()
    return InspectionRequiredParser()


def json_object(text):
    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    content = match.group(1) if match else text.strip()
    if content.startswith('Final Answer:'):
        content = content[len('Final Answer:'):].strip()
    return json.loads(content)


def permutation_schema(paths):
    if not 1 <= len(paths) <= 5 or len(set(paths))!=len(paths):
        raise ValueError('EAR constraint requires one to five unique candidates')
    return {'type':'object','properties':{'ranked_files':{'enum':[list(p) for p in permutations(paths)]}},
            'required':['ranked_files'],'additionalProperties':False}


def guarded_tool(original, candidates, inspected, errors, case):
    def read(path):
        clean = path.strip().strip('"').strip("'").strip()
        if clean not in candidates:
            message = ('Tool error: the requested path is not in the candidate list. No file was read. '
                       'Use only one exact path from this list: ' + json.dumps(candidates))
            errors.append({'requested_path': path, 'error': 'outside_candidates', 'observation': message})
            write_json(case / 'tool_errors.json', errors)
            return message
        try:
            value = original(clean)
        except (ValueError, SyntaxError, KeyError) as error:
            message = f'Tool error for {clean}: {type(error).__name__}: {error}'
            errors.append({'requested_path': path, 'error': type(error).__name__, 'observation': message})
            write_json(case / 'tool_errors.json', errors)
            return message
        inspected.append({'file_path': clean, 'skeleton': value})
        write_json(case / 'tool_calls.json', inspected)
        return value
    return read


def recover_scores(scores, query, candidates, inspected, generator, case, config, validate):
    """Ask the model to correct its own output; never filter/remap scores ourselves."""
    attempts = []
    for attempt in range(config.get('score_repair_attempts', 2) + 1):
        try:
            ranking = validate(scores, candidates)
            if attempts:
                write_json(case / 'score_recovery.json', attempts)
            return scores, ranking
        except ValueError as error:
            if attempt == config.get('score_repair_attempts', 2):
                raise ValueError('SAS output remains invalid after bounded correction') from error
            if not inspected:
                raise ValueError('No successful skeleton inspection; cannot finalize SAS') from error
            observed = {item['file_path']: item['skeleton'] for item in inspected}
            # Keep the original report and all inspected file identities. Long
            # skeletons get an explicitly recorded per-file budget for this
            # exceptional finalization call, not the ordinary ReAct loop.
            skeleton_budget = config.get('recovery_skeleton_tokens', 1024)
            evidence = {p: generator.tokenizer.decode(generator.tokenizer.encode(s, add_special_tokens=False)[:skeleton_budget],
                                                     skip_special_tokens=False) for p, s in observed.items()}
            prompt = ('Finalize file relevance scores using the bug report and the actual inspected skeletons below. '
                      'The previous answer was invalid or the tool iteration budget ended. '
                      'Return only a nonempty JSON object mapping exact candidate paths to numeric relevance scores from 0 to 10. '
                      'Score up to 10 files. Do not invent paths, request tools, or add explanation. '
                      'Choose paths only from CANDIDATES. Do not simply copy invalid paths from PREVIOUS OUTPUT.\n'
                      'BUG REPORT:\n' + query + '\nCANDIDATES:\n' + json.dumps(candidates) +
                      '\nINSPECTED SKELETONS (may be shortened):\n' + json.dumps(evidence) +
                      '\nPREVIOUS OUTPUT:\n' + json.dumps(scores))
            event = {'attempt': attempt + 1, 'validation_error': str(error), 'previous_scores': scores,
                     'skeleton_token_budget': skeleton_budget,
                     'shortened_skeletons': [p for p,s in observed.items() if len(generator.tokenizer.encode(s,add_special_tokens=False)) > skeleton_budget]}
            attempts.append(event)
            write_json(case / 'score_recovery.json', attempts)
            schema = {'type':'object', 'properties':{p:{'type':'number'} for p in candidates},
                      'additionalProperties':False}
            response = generator.complete([{'role':'user','content':prompt}], temperature=0,
                                          stage='SAS_repair', response_schema=schema)
            event['raw_response'] = response
            try:
                scores = json_object(response)
            except (ValueError, TypeError):
                scores = response
            write_json(case / 'score_recovery.json', attempts)


def evidence_rerank(query, head, contexts, evidence, modules, generator, case, config, parse):
    """Normal one-shot EAR; oversized inputs use complete pairwise merge ranking.

    Every original top-five file survives. Oversized individual contexts use all
    five exact retrieved chunks plus an explicitly budgeted skeleton. No labels,
    heuristic ranking, or candidate removal are used in either mode.
    """
    system = {'role':'system','content':'You are a helpful assistant that reranks code files.'}
    events = []
    def messages(paths, representations):
        code = ''.join(f'### File: {p} ###\n{representations[p]}\n\n' for p in paths)
        prompt = modules['prompts'].EVIDENCE_ANCHORED_RERANKER_PROMPT.format(
            num_file_paths=len(paths), problem_statement=query, aggregated_code_text=code)
        return [system, {'role':'user','content':prompt}]
    def fits(msgs):
        return generator.count_tokens(msgs) + config['max_new_tokens'] <= config['max_context_tokens']
    def ask(paths, representations, mode):
        msgs = messages(paths, representations)
        if not fits(msgs):
            raise ValueError('EAR report plus evidence exceeds context even after explicit compaction')
        event = {'paths':paths, 'mode':mode, 'input_tokens':generator.count_tokens(msgs)}
        events.append(event)
        write_json(case/'ear_comparisons.json', events)
        for attempt in range(config.get('ear_repair_attempts',1)+1):
            constraint = {'response_schema':permutation_schema(paths)} if config.get('constrain_ear_permutation') else {}
            response = generator.complete(msgs, temperature=config['ear_temperature'], stage='EAR', **constraint)
            event.setdefault('responses', []).append(response)
            try:
                ranking = parse(response, paths)
                event['ranking'] = ranking
                write_json(case/'ear_comparisons.json',events)
                return ranking
            except (ValueError, KeyError, TypeError) as error:
                event['error'] = str(error)
                write_json(case/'ear_comparisons.json',events)
                if attempt == config.get('ear_repair_attempts',1):
                    raise
                msgs = messages(paths, representations)
                msgs[-1]['content'] += ('\nYour previous answer was not a permutation of these exact paths. '
                                        'Return each listed path exactly once in ranked_files: '+json.dumps(paths))
                if not fits(msgs):
                    raise ValueError('EAR correction prompt exceeds context') from error
    original_tokens = generator.count_tokens(messages(head, contexts))
    if fits(messages(head, contexts)):
        write_json(case/'ear_strategy.json', {'mode':'single_pass','original_input_tokens':original_tokens})
        return ask(head, contexts, 'single_pass')
    # Fix the representation per file before comparisons so it cannot depend on
    # model scores or which candidate happens to face it later in the merge.
    compact = {}
    changes = []
    overhead = generator.count_tokens(messages(head[:2], dict.fromkeys(head,'')))
    per_file = (config['max_context_tokens'] - config['max_new_tokens'] - overhead - 128) // 2
    if per_file <= 0:
        raise ValueError('Original bug report leaves no EAR evidence budget')
    for path in head:
        text = contexts[path]
        if len(generator.tokenizer.encode(text,add_special_tokens=False)) > per_file:
            # Context is the author's pruned Python text; skeleton extraction
            # can fail on its decorator formatting. Use source skeleton supplied
            # by caller when available, with no repository lookup here.
            skeleton = modules.get('ear_skeletons',{}).get(path, '')
            anchors = '\n\n'.join(f'# Retrieved evidence lines {c["start_line"]}-{c["end_line"]}\n{c["code"]}' for c in evidence[path])
            anchor_tokens = len(generator.tokenizer.encode(anchors,add_special_tokens=False))
            budget = min(config.get('ear_skeleton_tokens',2048), per_file-anchor_tokens-64)
            if budget < 0:
                raise ValueError('Exact EAR anchors exceed per-file budget')
            skeleton_ids = generator.tokenizer.encode(skeleton,add_special_tokens=False)
            text = '# Structural skeleton (budgeted)\n'+generator.tokenizer.decode(skeleton_ids[:budget],skip_special_tokens=False)+'\n# Exact retrieval anchors\n'+anchors
            changes.append({'file_path':path,'original_tokens':len(generator.tokenizer.encode(contexts[path],add_special_tokens=False)),
                            'new_tokens':len(generator.tokenizer.encode(text,add_special_tokens=False)),
                            'anchors_kept':len(evidence[path]),'skeleton_tokens_kept':min(budget,len(skeleton_ids))})
        compact[path] = text
    write_json(case/'ear_strategy.json', {'mode':'pairwise_merge','original_input_tokens':original_tokens,
               'per_file_token_budget':per_file,'compactions':changes,'all_files_preserved':head})
    write_json(case/'ear_effective_context.json', compact)
    def merge_sort(paths):
        if len(paths) < 2:
            return paths
        middle = len(paths)//2
        left, right = merge_sort(paths[:middle]), merge_sort(paths[middle:])
        result = []
        while left and right:
            pair = ask([left[0],right[0]], compact, 'pairwise_merge')
            result.append((left if pair[0] == left[0] else right).pop(0))
        return result + left + right
    return merge_sort(list(head))
