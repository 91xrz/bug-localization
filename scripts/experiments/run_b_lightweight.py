"""Frozen two-stage CPU study of B grounding and fixed-budget recall fusion."""
import argparse
import csv
import hashlib
import json
import math
import mmap
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from bug_localization.lightweight_hypothesis import anchors, definitions, rank_variants
from bug_localization.paired_statistics import inference

CONFIG = ROOT/'configs/experiments/b_lightweight_20261122.json'
IDENTITY = ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha', 'query_sha256', 'labels_sha256')


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def field(path, key):
    """Read one pretty-printed top-level field without parsing chunk scores."""
    marker = ('\n  "'+key+'": ').encode()
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        start = data.find(marker)
        assert start >= 0, (path, key)
        start += len(marker)
        end = data.find(b'\n  "', start)
        raw = data[start:end].rstrip().removesuffix(b',') if end >= 0 else data[start:].rstrip()[:-1].rstrip()
        return json.loads(raw)


def load_ranking(path):
    rows = field(path, 'ranking')
    assert [r['rank'] for r in rows] == list(range(1, len(rows)+1))
    assert len({r['file_path'] for r in rows}) == len(rows)
    assert all(math.isfinite(r['score']) for r in rows)
    assert all(x['score'] >= y['score'] for x,y in zip(rows, rows[1:]))
    return [{k: r[k] for k in ('file_path', 'rank', 'score')} for r in rows]


def metrics(ranking, gold):
    assert len(ranking) == len(set(ranking))
    hits = [i for i,p in enumerate(ranking[:10],1) if p in gold]
    h50 = len(set(ranking[:50]) & gold)
    return {'MAP@10': sum(n/i for n,i in enumerate(hits,1))/len(gold),
            'MRR@10': 1/hits[0] if hits else 0.,
            'Hit@1': int(bool(hits) and hits[0] == 1), 'Hit@10': int(bool(hits)),
            'MeanRecall@10': len(hits)/len(gold),
            'Hit@50': int(h50 > 0), 'MeanRecall@50': h50/len(gold),
            'PoolHit': int(bool(set(ranking) & gold)), 'PoolRecall': len(set(ranking) & gold)/len(gold),
            'gold50': h50, 'pool_size': len(ranking)}


def summarize(cases):
    names = list(cases[0]['methods'])
    return {name: {key: sum(c['methods'][name]['metrics'][key] for c in cases)/len(cases)
                   for key in cases[0]['methods'][name]['metrics']}
            for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = read(CONFIG)
    source = ROOT/config['source_run']
    out = args.output or ROOT/config['output']
    out.mkdir(parents=True, exist_ok=False)
    selection = read(source/'selection.json')
    aa, bb = [read(source/r/'samples.json') for r in ('S0','H0')]
    prior = read(ROOT/'results/top50_complementarity/20261122/cases_full.json')
    froot = ROOT/'results/reranking_full/20261119_swe2294_gpu0'
    ff = read(froot/'F/samples.json')
    assert len(aa) == len(bb) == len(selection['samples']) == len(prior) == len(ff) == 2294
    assert read(source/'verification.json')['status'] == 'passed'
    files = [CONFIG, Path(__file__), ROOT/'src/bug_localization/lightweight_hypothesis.py',
             source/'selection.json', source/'S0/samples.json', source/'H0/samples.json', source/'verification.json']
    protocol = {'config': config, 'created_unix': time.time(),
                'implementation_and_input_sha256': {str(p.resolve()): digest(p.read_bytes()) for p in files},
                'source_verification': 'Reuse passed full-source audit, validate checkpoint identities, recompute baseline metrics and compare prior Top50 audit. Do not claim rehash of full chunk-score files.',
                'inference_calls': {'generation': 0, 'embedding': 0, 'reranking': 0}}
    save(out/'protocol.json', protocol)
    save(out/'selection.json', selection)
    dev = set(selection['development_rows'])
    all_cases, symbol_cache = [], {}
    costs = {'source_bytes_read': 0, 'unique_source_contents': 0, 'source_sha_checks': 0}
    started = time.monotonic()
    chosen = None
    for phase, indices in [('development500', [i for i,x in enumerate(aa) if x['dataset_row'] in dev]),
                           ('remaining1794', [i for i,x in enumerate(aa) if x['dataset_row'] not in dev])]:
        phase_cases = []
        for step, i in enumerate(indices, 1):
            t0 = time.monotonic()
            meta, ra, rb = selection['samples'][i], aa[i], bb[i]
            assert all(meta[k] == ra[k] == rb[k] == ff[i][k] for k in IDENTITY)
            assert ra['status'] == 'ok'
            gold = set(ra['ground_truth'])
            a = load_ranking(source/ra['result_file'])
            assert gold == set(field(source/ra['result_file'], 'ground_truth'))
            ok = rb['status'] == 'ok'
            record = {**meta, 'phase': phase, 'status': 'ok' if ok else 'B_failed',
                      'ground_truth': sorted(gold), 'methods': {},
                      'source_result_sha256_from_checkpoints': {'A':ra['result_sha256'], 'B':rb.get('result_sha256')}}
            if ok:
                b = load_ranking(source/rb['result_file'])
                assert {r['file_path'] for r in a} == {r['file_path'] for r in b}
                assert gold == set(rb['ground_truth']) == set(field(source/rb['result_file'], 'ground_truth'))
                union = {r['file_path'] for rs in (a[:50],b[:50]) for r in rs}
                gen = read(source/rb['generation_file'])
                assert all(gen['sample'][k] == meta[k] for k in IDENTITY)
                body = gen['payload']['original_report']
                assert digest(json.dumps({'body':body},sort_keys=True).encode()) == meta['query_sha256']
                inventory = read(source/ra['assets']/'candidates.json')
                sha = {r['file_path']:r['sha256'] for r in inventory['files']}
                assert sha.keys() == {r['file_path'] for r in a}
                snapshot = ROOT/'repos'/meta['repository']/meta['before_fix_sha']/'source'
                symbols = {}
                for path in sorted(union):
                    expected = sha[path]
                    if expected not in symbol_cache:
                        raw = (snapshot/path).read_bytes()
                        assert digest(raw) == expected, (meta['position'], path)
                        symbol_cache[expected] = definitions(raw.decode('utf-8',errors='replace'))
                        costs['source_bytes_read'] += len(raw)
                        costs['source_sha_checks'] += 1
                    symbols[path] = symbol_cache[expected]
                feat_started = time.monotonic()
                anchor, matches = anchors(body, symbols)
                ranks = rank_variants(a,b,anchor,config['alphas'],config['anchor_weights'])
                record['postprocess_seconds'] = time.monotonic()-feat_started
                record['anchor_features'] = {p:{'score':anchor[p],**matches[p]} for p in sorted(union) if anchor[p]>0}
                assert len(set(ranks['A']) & set(ranks['B'])) == prior[i]['overlap']
                assert set(ranks['current_RRF60']) == {r['file_path'] for r in field(froot/ff[i]['result_file'], 'candidates')}
                assert ranks['current_RRF60'] == [r['file_path'] for r in field(froot/ff[i]['result_file'], 'ranking')]
                save(out/'inputs'/f'{meta["position"]:04d}.json',
                     {'sample':meta,'a':a,'b':b,'anchor':anchor,'source_content_sha256':{p:sha[p] for p in sorted(union)}})
            else:
                assert all_cases, 'First case unexpectedly failed'
                ranks = {name: ([r['file_path'] for r in a[:50]] if name=='A' else [])
                         for name in all_cases[0]['methods']}
                record['postprocess_seconds'] = 0.
                record['error'] = rb.get('error')
            for name, rank in ranks.items():
                m = metrics(rank,gold)
                assert name=='union_ceiling_Border' or len(rank) <= 50
                if name in ('A','B','current_RRF60'):
                    base = {'A':ra,'B':rb,'current_RRF60':ff[i]}[name]
                    expected = base['metrics']['average_precision_at_k'] if base['status']=='ok' else 0
                    assert abs(m['MAP@10']-expected) < 1e-12, (meta['position'],name)
                if name in ('A','B'):
                    assert m['gold50'] == prior[i][name+'50_gold']
                record['methods'][name] = {'ranking':rank,'metrics':m}
            record['elapsed_seconds'] = time.monotonic()-t0
            save(out/'cases'/f'{meta["position"]:04d}.json',record)
            phase_cases.append(record)
            all_cases.append(record)
            if step % 50 == 0:
                print(f'{phase}: {step}/{len(indices)}, elapsed {time.monotonic()-started:.1f}s',flush=True)
        summary = summarize(phase_cases)
        save(out/(phase+'_summary.json'),summary)
        print(phase,json.dumps(summary,ensure_ascii=False),flush=True)
        if phase == 'development500':
            base = summary['B']
            eligible = [name for name,m in summary.items() if name.startswith(('score_','B50_anchor_'))
                        and m['MAP@10'] >= base['MAP@10']+.005
                        and all(m[k] >= base[k]-1e-12 for k in ('MRR@10','Hit@10','Hit@50'))]
            def preference(name):
                # Prefer simpler settings when MAP is exactly equal.
                alpha = float(name.split('_a')[1].split('_')[0]) if name.startswith('score_') else 0.
                weight = float(name.split('_e')[-1])
                return (-summary[name]['MAP@10'],alpha,weight,name)
            chosen = min(eligible,key=preference) if eligible else 'B'
            save(out/'frozen_choice.json', {'method':chosen,'eligible':eligible,'created_unix':time.time(),
                                          'chosen_on':'development500 only, before validation loop',
                                          'development_metrics':summary[chosen]})
        else:
            save(out/'validation_pairs.json',[{**{k:r[k] for k in IDENTITY},
                'B':r['methods']['B']['metrics'],'selected':r['methods'][chosen]['metrics']} for r in phase_cases])
            tests = {}
            for key in ('MAP@10','Hit@10','Hit@50'):
                tests[key] = inference([r['methods'][chosen]['metrics'][key]-r['methods']['B']['metrics'][key]
                                        for r in phase_cases],[r['repository'] for r in phase_cases],
                                       seed=20261122,repetitions=5000)
            previous = 0.
            for j, (_,test) in enumerate(sorted(tests.items(),key=lambda kv:kv[1]['p_two_sided_paired_randomization'])):
                previous = min(1.,max(previous,(len(tests)-j)*test['p_two_sided_paired_randomization']))
                test['p_holm_three_primary_tests'] = previous
            save(out/'validation_inference.json', {'selected':chosen,'tests':tests})
    all_cases.sort(key=lambda r:r['position'])
    assert [r['position'] for r in all_cases] == list(range(1,2295))
    summaries = {'all2294':summarize(all_cases),
                 **{phase:read(out/(phase+'_summary.json')) for phase in ('development500','remaining1794')}}
    save(out/'summary.json',summaries)
    save(out/'project_summary.json',{p:summarize([r for r in all_cases if r['repository']==p])
                                    for p in sorted({r['repository'] for r in all_cases})})
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        rows = [{'cohort':c,'method':n,'selected_on_development':n==chosen,**m} for c,methods in summaries.items() for n,m in methods.items()]
        w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    costs.update(unique_source_contents=len(symbol_cache),total_seconds=time.monotonic()-started,
                 postprocess_seconds=sum(r['postprocess_seconds'] for r in all_cases),
                 model_calls=0,gpu_calls=0)
    save(out/'cost.json',costs)
    for path,sha in protocol['implementation_and_input_sha256'].items():
        assert digest(Path(path).read_bytes()) == sha, 'Frozen source changed during study'
    save(out/'verification.json',{'status':'passed','samples':2294,'B_failures':sum(r['status']!='ok' for r in all_cases),
                                 'checks':['original report hashes','before-fix source content hashes for cached definitions',
                                           'all A/B/F AP baselines reproduced','prior Top50 label counts and overlap matched',
                                           'all current fusion pool paths and full ordering matched',
                                           'fixed50 except explicitly named union diagnostic',
                                           'development choice frozen before validation','frozen input/source hashes unchanged']})
    print('COMPLETE',out,'SELECTED',chosen,flush=True)


if __name__ == '__main__':
    main()
