"""Audit saved inputs, successful permutations and the cost of the cached tie prior."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from pilot_perspective_rank import C3, COMMON, LENS, cohort, fuse, paired, paths_from_saved, ranking_metrics, read, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('dev27','confirmation90'), default='dev27')
    samples,archive,output = cohort(parser.parse_args().cohort)
    costs = {}
    agreements = Counter()
    tie_effect = Counter()
    prior_free = Counter()
    paired_free = []
    c3 = {r['position']:r for r in read(C3/'samples.json')}
    for s in samples:
        key = f'{s["position"]:04d}'
        case = output/'cases'/key
        result = read(case/'rankings.json')
        heads = result['heads']
        branch = {}
        for name in ('single','reverse','owner','behavior'):
            call = read(case/name/'call_001.json')
            assert call['status']=='ok'
            lens = 'ordinary' if name in ('single','reverse') else name
            assert call['request']['messages'][0]['content']==COMMON+LENS[lens]
            branch[name] = call['request']['messages'][1]['content']
            costs.setdefault(name,Counter()).update(calls=1,input_tokens=call['input_tokens'],
                                                     output_tokens=call['output_tokens'])
        assert Counter(branch[name] for name in ('single','reverse')) == Counter(branch[name] for name in ('owner','behavior'))
        assert branch['single'] != branch['reverse']
        src = c3[s['position']]
        saved = read(C3/src['result_file'])
        prior = paths_from_saved(saved)[:10]
        source = archive/'cases'/key/'inference_input.json'
        assert prior == read(source)['candidates'] and sha256_file(source)==result['input_sha256']
        tie = read(archive/'cases'/key/'result.json')['ranking'][:10] if result['tie_source']=='ok' else prior
        row = dict(repository=s['repository'],metrics={})
        for target,a,b in [('repeat','single','reverse'), ('dual','owner','behavior')]:
            agreements[target] += int(heads[a][0]==heads[b][0])
            alt = fuse(heads[a],heads[b],prior)
            assert heads[target]==fuse(heads[a],heads[b],tie)
            tie_effect[target+'_changed_top1'] += int(alt[0]!=heads[target][0])
            row['metrics'][target] = ranking_metrics(alt,saved['ground_truth'],10)
            prior_free[target] += row['metrics'][target]['average_precision_at_k']
        paired_free.append(row)
    report = dict(n=len(samples), costs=costs, same_top1=agreements, changed_if_C3_breaks_ties=tie_effect,
                  map_with_C3_as_tie_prior={k:v/len(samples) for k,v in prior_free.items()},
                  paired_with_C3_as_tie_prior=paired(paired_free,'dual','repeat'))
    write_json(output/'audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
