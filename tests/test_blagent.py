import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.blagent import candidate_union, localize, make_structure, parse_ranking, retrieve, run, validate_scores
from bug_localization.batch import aggregate_results
from bug_localization.bm25 import ranking_metrics


class BLAgentTests(unittest.TestCase):
    def test_author_union_is_ordered_and_can_exceed_fifteen(self):
        first = [f'a{i}.py' for i in range(20)]
        second = [f'b{i}.py' for i in range(20)]
        self.assertEqual(candidate_union(first, second), first[:15] + second[:15])
        self.assertEqual(candidate_union(first, first), first[:15])

    def test_scores_reject_hallucinations_nan_booleans_and_empty_output(self):
        paths = [f'{i}.py' for i in range(10)]
        good = dict.fromkeys(paths, 5)
        self.assertEqual(validate_scores(good, paths), paths)
        for bad in ({**good, 'extra.py': 1}, {**good, paths[0]: float('nan')},
                    {**good, paths[0]: True}, {**good, paths[0]: 11}, {}):
            with self.assertRaises(ValueError):
                validate_scores(bad, paths)
        self.assertEqual(validate_scores({paths[0]:10}, paths), paths[:1])
        extended=paths+['extra.py']
        self.assertEqual(validate_scores(dict.fromkeys(extended,5), extended), extended)

    def test_ear_is_a_permutation(self):
        self.assertEqual(parse_ranking('```json\n{"ranked_files":["b.py","a.py"]}\n```', ['a.py','b.py']), ['b.py','a.py'])
        for paths in (['a.py','a.py'], ['a.py'], ['a.py','secret.py']):
            with self.assertRaises(ValueError):
                parse_ranking(json.dumps({'ranked_files': paths}), ['a.py','b.py'])

    def test_source_structure_preserves_exact_source_and_test_files(self):
        docs = [{'file_path':'pkg/a.py','content':'def f():\n    return 1\n'},
                {'file_path':'tests/test_a.py','content':'assert True\n'}]
        tree = make_structure(docs)
        self.assertEqual('\n'.join(tree['pkg']['a.py']['text']), docs[0]['content'])
        self.assertIn('test_a.py', tree['tests'])

    def test_file_pooling_and_anchors_use_actual_similarity(self):
        import numpy as np
        chunks = [{'file_path':p, 'chunk_id':str(i), 'code':str(i), 'embedding_text':str(i), 'start_line':1,'end_line':1}
                  for i,p in enumerate(['a.py','a.py','b.py'])]
        embed = types.SimpleNamespace(encode=lambda _:np.array([[1.,0.],[0.,1.],[.6,.8]]),
                                       query=lambda q:(np.array([1.,0.] if q=='structural' else [0.,1.]),{}))
        cache = types.SimpleNamespace(get=lambda d,r:([c for c in chunks if c['file_path']==d['file_path']],[]))
        rankings, evidence, _, _ = retrieve([{'file_path':'a.py'},{'file_path':'b.py'}], ['structural','behavioral'],embed,cache)
        self.assertEqual(rankings, [['a.py','b.py'],['a.py','b.py']])
        self.assertEqual(evidence[0]['a.py'][0]['code'], '0')
        self.assertEqual(evidence[1]['a.py'][0]['code'], '1')

    def test_end_to_end_preserves_phase1_tail_and_limits_tool_access(self):
        paths = [f'{i}.py' for i in range(10)]
        docs = [{'file_path': p, 'content':f'def f{i}():\n    return {i}\n'} for i,p in enumerate(paths)]
        case_data = {}
        class Agent:
            def __init__(self, structure, entry, **kwargs):
                case_data['entry'] = entry
                self.tools = [types.SimpleNamespace(func=lambda p:'def f0()')]
                self.agent = types.SimpleNamespace()
            def run(self, **kwargs):
                self.tools[0].func(case_data.get('tool_path',paths[0]))
                return dict(zip(paths,range(10,0,-1))), {}
        class Augmentor:
            def __init__(self,*args,**kwargs): pass
            def augment(self,query,**kwargs): return 'query'+str(kwargs['type'])
        modules = {'queries':types.SimpleNamespace(QueryAugmentor=Augmentor),
                   'agent':types.SimpleNamespace(SWEBenchRankerAgent=Agent),
                   'pruner':types.SimpleNamespace(CodePruner=lambda source,anchors:types.SimpleNamespace(prune=lambda:source)),
                   'prompts':types.SimpleNamespace(EVIDENCE_ANCHORED_RERANKER_PROMPT='{problem_statement}\n{aggregated_code_text}')}
        generator = types.SimpleNamespace(calls=4,complete=lambda *a,**k:json.dumps({'ranked_files':paths[:5][::-1]}))
        evidence = {p:[{'code':d['content']}] for p,d in zip(paths,docs)}
        config = {'query_temperature':.7,'agent_temperature':.5,'ear_temperature':0,'max_agent_iterations':15}
        with tempfile.TemporaryDirectory() as tmp, patch('bug_localization.blagent.retrieve', return_value=([paths,paths],[evidence,evidence],[],[])):
            result=localize('original report',docs,modules,generator,None,None,Path(tmp),config)
            self.assertEqual(result['ranking'],paths[:5][::-1]+paths[5:])
            self.assertEqual(set(case_data['entry']),{'problem_statement','retrieved_files_t0','retrieved_files_t1'})
            case_data['tool_path']='../../result.json'
            with self.assertRaisesRegex(ValueError,'outside'):
                localize('original report',docs,modules,generator,None,None,Path(tmp),config)

    def test_failed_cases_stay_in_primary_denominator(self):
        ok={'status':'ok','metrics':ranking_metrics(['a.py'],['a.py','b.py'],10),'missing_gold':[]}
        failed={'status':'failed','repository':'r','issue_id':1,'error_type':'ValueError','error':'bad output'}
        summary=aggregate_results([ok,failed],10,2)
        self.assertEqual(summary['metrics']['Top-1'],.5)
        self.assertEqual(summary['metrics']['MAP@10'],.25)

    def test_completed_run_resumes_without_inference_and_rejects_tampering(self):
        row={'repo_name':'example/repo','issue_id':1,'before_fix_sha':'a'*40}
        sample={'dataset_row':0,'report':row}
        generator=types.SimpleNamespace(identity={},calls=0,start_case=lambda _:None)
        embedder=types.SimpleNamespace(identity={},tokenizer=None,fingerprint='test',close=lambda:None)
        config={'base_config':'base.json','semantic_config':'semantic.json',
                'generation_model':'unused','limit':1,'repositories':['example/repo']}
        docs=[{'file_path':'a.py','content':'x=1\n','sha256':'test'}]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            project=Path(tmp)
            (project/'base.json').write_text((PROJECT/'configs/retrieval/bm25.json').read_text())
            (project/'semantic.json').write_text('{}')
            for name, value in {'select_reports':([sample],{'samples':[],'selection_sha256':'test'}),
                                'author_modules':({},{}),'Generator':generator,'LocalEmbedder':embedder,
                                'version':'test','get_snapshot':(None,None),'collect_files':(docs,[]),
                                'prepare_report':(row,['a.py'],'bug report')}.items():
                stack.enter_context(patch('bug_localization.blagent.'+name,return_value=value))
            stack.enter_context(patch('bug_localization.ast_chunking.RoutingChunkCache',return_value=None))
            infer=stack.enter_context(patch('bug_localization.blagent.localize',
                return_value={'ranking':['a.py'],'phase1_ranking':['a.py']}))
            folder, first=run(project,config,Path('run'))
            _, resumed=run(project,config,Path('run'))
            self.assertEqual(first,resumed)
            self.assertEqual(infer.call_count,1)
            (folder/'cases/0001/result.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                run(project,config,Path('run'))


if __name__ == '__main__':
    unittest.main()
