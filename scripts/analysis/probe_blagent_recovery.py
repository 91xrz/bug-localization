"""Validate local server token IDs and real bounded repairs on saved failed SAS outputs.

Does not read labels or evaluate accuracy. CPU probe results are not merged into
new GPU experiment results, nor used to claim GPU speed improvements.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT/p) for p in ('.runtime/blagent-deps','.runtime/rrl-deps','.runtime/igsil-deps','src')]
from bug_localization.blagent import validate_scores
from bug_localization.blagent_runtime import ServerGenerator,request_json
from bug_localization.blagent_recovery import json_object,recover_scores
from bug_localization.batch import write_json
from bug_localization.snapshot import sha256_file


def main(output):
    config=json.loads((PROJECT/'configs/experiments/blagent500_v2.json').read_text())
    generator=ServerGenerator(PROJECT,config)
    root=PROJECT/output;root.mkdir(parents=True,exist_ok=True)
    messages=[{'role':'system','content':'Return JSON.'},{'role':'user','content':'Check Python路径: foo/bar.py\nprint("é")'}]
    rendered=generator.render(messages)
    remote=request_json(generator.endpoint,'/tokenize',{'content':rendered,'add_special':False,'parse_special':True})['tokens']
    local=generator.tokenizer.encode(rendered,add_special_tokens=False)
    if local!=remote:raise ValueError('HF and GGUF tokenizer IDs differ')
    write_json(root/'tokenizer_check.json',{'status':'passed','token_count':len(local),'exact_ids_equal':True})
    started=time.monotonic()
    def repair(position):
        source=PROJECT/'results/blagent/crossproject500-20261118/cases'/f'{position:04d}'
        case=root/f'{position:04d}';case.mkdir(exist_ok=True)
        inputs=json.loads((source/'inputs.json').read_text())
        retrieved=json.loads((source/'retrieval.json').read_text())
        inspected=json.loads((source/'tool_calls.json').read_text())
        last=sorted(source.glob('call_*.json'))[-1]
        response=json.loads(last.read_text())['response']
        scores=json_object(response.split('Final Answer:',1)[-1])
        worker=copy.copy(generator);worker.start_case(case)
        scoring,ranking=recover_scores(scores,inputs['query'],retrieved['candidates'],inspected,worker,case,config,validate_scores)
        result={'original_position':position,'status':'passed','source_call_sha256':sha256_file(last),
                'scores':scoring,'ranking':ranking,'new_calls':worker.calls,'source_inputs_sha256':sha256_file(source/'inputs.json')}
        write_json(case/'probe.json',result);print('Repaired original case',position,flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(repair,[2,110]))
    write_json(root/'summary.json',{'status':'passed','scope':'CPU server and real invalid-score recovery probe, no accuracy evaluation',
               'concurrent_requests':2,'wall_seconds':time.monotonic()-started,'results':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
