"""Audit saved S0/H0 file overlap and label recovery without model inference."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/recall_full/20261118_swe2294_v2'
OUT = ROOT / 'results/top50_complementarity/20261122'
IDENTITY = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha',
            'query_sha256', 'labels_sha256')


def read(path):
    return json.loads(path.read_bytes())


def ranking(path, expected=None):
    raw = path.read_bytes()
    if expected:
        assert hashlib.sha256(raw).hexdigest() == expected
    # Frozen outputs are pretty-printed JSON. Decode only the two top-level
    # fields used here, avoiding the much larger per-chunk score arrays.
    source = raw.decode('utf-8')
    decoder = json.JSONDecoder()
    data = {}
    for key in ('ranking', 'ground_truth'):
        marker = '\n  "' + key + '": '
        start = source.index(marker) + len(marker)
        assert source.find(marker, start) == -1, (path, key)
        data[key] = decoder.raw_decode(source, start)[0]
    paths = [r['file_path'] for r in data['ranking']]
    assert len(paths) == len(set(paths))
    assert [r['rank'] for r in data['ranking']] == list(range(1, len(paths) + 1))
    return paths, set(data['ground_truth'])


def mixed(primary, secondary):
    result = list(primary[:40])
    for path in secondary:
        if len(result) >= 50:
            break
        if path not in result:
            result.append(path)
    assert len(result) == min(50, len(secondary))
    return set(result)


def case(sample, a, b, gold, b_ok):
    ar, br = ({p: i for i, p in enumerate(seq, 1)} for seq in (a, b))
    aa, bb = set(a[:50]), set(b[:50])
    # Failed dependent mixed routes contribute zero, matching the frozen protocol.
    pools = {'A50': aa, 'B50': bb, 'union': aa | bb,
             'B40_Afill': mixed(b, a) if b_ok else set(),
             'A40_Bfill': mixed(a, b) if b_ok else set()}
    gained, lost = (bb - aa) & gold, (aa - bb) & gold
    row = {k: sample[k] for k in ('position', 'dataset_row', 'repository', 'issue_id')}
    row.update(b_ok=b_ok, gold_n=len(gold), inventory_n=len(a),
               missing_inventory_gold=sorted(gold - set(a)),
               overlap=len(aa & bb), a_size=len(aa), b_size=len(bb),
               b_new=len(bb - aa), union_size=len(aa | bb),
               jaccard=len(aa & bb) / len(aa | bb),
               b_new_gold=sorted(gained), a_only_gold=sorted(lost),
               b_new_gold_ranks=[{'file': p, 'A': ar.get(p), 'B': br[p]} for p in sorted(gained)],
               lost_gold_ranks=[{'file': p, 'A': ar[p], 'B': br.get(p)} for p in sorted(lost)],
               mixed_new_gold=sorted((pools['B40_Afill'] - aa) & gold),
               mixed_lost_gold=sorted((aa - pools['B40_Afill']) & gold),
               mixed_new=len(pools['B40_Afill'] - aa),
               b_new_gold_retained=sorted(gained & pools['B40_Afill']),
               b_new_gold_dropped=sorted(gained - pools['B40_Afill']),
               top10_overlap=len(set(a[:10]) & set(b[:10])))
    for name, pool in pools.items():
        hits = len(pool & gold)
        row[name + '_hit'] = int(hits > 0)
        row[name + '_gold'] = hits
        row[name + '_recall'] = hits / len(gold)
        row[name + '_allgold'] = int(hits == len(gold))
    return row


def summarize(rows):
    paired = [r for r in rows if r['b_ok']]
    n = len(rows)
    result = {'n': n, 'paired_success': len(paired), 'B_failures': n-len(paired)}
    for key in ('overlap', 'b_new', 'union_size', 'jaccard', 'top10_overlap', 'mixed_new'):
        result[key + '_mean_paired'] = mean(r[key] for r in paired)
    result['overlap_median_paired'] = median(r['overlap'] for r in paired)
    result['overlap_histogram_paired'] = dict(sorted(Counter(r['overlap'] for r in paired).items()))
    result['overlap_fraction_paired'] = sum(r['overlap'] for r in paired) / sum(r['b_size'] for r in paired)
    result['identical_top50_sets'] = sum(r['overlap'] == r['a_size'] == r['b_size'] for r in paired)
    result['overlap_at_least40'] = sum(r['overlap'] >= 40 for r in paired)
    result['overlap_at_most25'] = sum(r['overlap'] <= 25 for r in paired)
    result['B_only_hit_paired'] = sum(r['B50_hit'] and not r['A50_hit'] for r in paired)
    result['A_only_hit_paired'] = sum(r['A50_hit'] and not r['B50_hit'] for r in paired)
    result['both_hit_paired'] = sum(r['A50_hit'] and r['B50_hit'] for r in paired)
    result['neither_hit_paired'] = sum(not r['A50_hit'] and not r['B50_hit'] for r in paired)
    for key in ('b_new_gold', 'a_only_gold', 'mixed_new_gold', 'mixed_lost_gold',
                'b_new_gold_retained', 'b_new_gold_dropped'):
        result[key + '_count_paired'] = sum(len(r[key]) for r in paired)
        result[key + '_cases_paired'] = sum(bool(r[key]) for r in paired)
    result['missing_inventory_gold'] = sum(len(r['missing_inventory_gold']) for r in rows)
    result['total_gold'] = sum(r['gold_n'] for r in rows)
    for name in ('A50', 'B50', 'union', 'B40_Afill', 'A40_Bfill'):
        result[name] = {'hits': sum(r[name+'_hit'] for r in rows),
                        'hit_rate': mean(r[name+'_hit'] for r in rows),
                        'mean_recall': mean(r[name+'_recall'] for r in rows),
                        'gold_retrieved': sum(r[name+'_gold'] for r in rows),
                        'allgold_cases': sum(r[name+'_allgold'] for r in rows),
                        'rescued_vs_A': sum(r[name+'_hit'] and not r['A50_hit'] for r in rows),
                        'lost_vs_A': sum(r['A50_hit'] and not r[name+'_hit'] for r in rows)}
    return result


def historical():
    cfg = read(ROOT / 'configs/experiments/recall_baselines500.json')
    samples = read(ROOT / cfg['selection'])['samples']
    rows = []
    for s in samples:
        i = s['position']
        if i <= 180:
            ap, bp, local = ROOT/cfg['old_semantic'], ROOT/cfg['old_hypothesis'], i
        else:
            ap = ROOT/cfg['control_run']/'new320/semantic'
            bp = ROOT/cfg['control_run']/'new320/hypothesis'
            local = i-180
        a, gold = ranking(ap/'cases'/f'{local:03d}'/'q0_result.json')
        b_record = read(bp/'samples.json')['direct_hypothesis'][local-1]
        a_record = read(ap/'samples.json')['q0'][local-1]
        assert all(a_record[k] == b_record[k] == s[k] for k in IDENTITY)
        ok = b_record['status'] == 'ok'
        b = []
        if ok:
            b, bgold = ranking(bp/'cases'/f'{local:03d}'/'direct_result.json')
            assert gold == bgold and set(a) == set(b)
        rows.append(case(s, a, b, gold, ok))
    return rows


def verify_saved_pools(rows):
    run = ROOT/'results/reranking_full/20261119_swe2294_gpu0'
    records = read(run/'F/samples.json')
    full = read(run/'FULL/samples.json')
    assert len(records) == len(full) == len(rows)
    checked, rescued, dropped = 0, [], []
    for row, record, downstream in zip(rows, records, full):
        assert row['dataset_row'] == record['dataset_row'] == downstream['dataset_row']
        if not row['b_ok']:
            assert record['status'] != 'ok'
            continue
        assert record['status'] == downstream['status'] == 'ok'
        data = read(run/record['result_file'])
        paths = [r['file_path'] for r in data['candidates']]
        gold = set(data['ground_truth'])
        assert len(paths) == len(set(paths)) == min(50, row['inventory_n'])
        assert len(set(paths) & gold) == row['B40_Afill_gold']
        assert set(row['b_new_gold_retained']) <= set(paths)
        assert not set(row['b_new_gold_dropped']) & set(paths)
        checked += 1
        if row['B40_Afill_hit'] and not row['A50_hit']:
            rescued.append({'position': row['position'], 'repository': row['repository'],
                            'issue_id': row['issue_id'],
                            'F_top10': record['metrics']['top_10'],
                            'FULL_top10': downstream['metrics']['top_10']})
        if row['A50_hit'] and not row['B40_Afill_hit']:
            dropped.append(row['position'])
    result = {'checked_successful_saved_pools': checked,
              'checks': ['candidate uniqueness and size', 'gold coverage agrees with reconstructed pool',
                         'B-exclusive gold retained/dropped agrees with actual saved pool'],
              'rescued_cases': rescued, 'lost_cases_paired': dropped,
              'rescued_top10_F': sum(r['F_top10'] for r in rescued),
              'rescued_top10_FULL': sum(r['FULL_top10'] for r in rescued)}
    (OUT/'saved_pool_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def additional_diagnostics(rows):
    aa, bb = (read(RUN/route/'samples.json') for route in ('S0', 'H0'))
    rescued = [(r, b) for r, a, b in zip(rows, aa, bb)
               if b['status'] == 'ok' and not a['metrics']['top_10'] and b['metrics']['top_10']]
    categories = Counter()
    for r, b in rescued:
        new = sum(p['B'] <= 10 for p in r['b_new_gold_ranks'])
        old = b['metrics']['retrieved_gold_files'] - new
        assert old >= 0
        categories['both_old_and_new_gold' if old and new else
                   'only_gold_already_in_A50' if old else 'only_gold_outside_A50'] += 1
    added = sum(r['b_new'] for r in rows if r['b_ok'])
    result = {'B_top10_rescues_vs_A_top10': len(rescued),
              'among_rescues_A50_miss': sum(not r['A50_hit'] for r, _ in rescued),
              'among_rescues_A50_hit': sum(r['A50_hit'] for r, _ in rescued),
              'rescued_top10_gold_origin': dict(categories),
              'cases_with_fewer_than50_files': sum(r['inventory_n'] < 50 for r in rows),
              'min_inventory': min(r['inventory_n'] for r in rows),
              'B_new_file_report_pairs': added,
              'B_new_gold_fraction': sum(len(r['b_new_gold']) for r in rows) / added}
    (OUT/'additional_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    selection = read(RUN/'selection.json')
    a_records, b_records = (read(RUN/route/'samples.json') for route in ('S0', 'H0'))
    assert len(a_records) == len(b_records) == len(selection['samples']) == 2294
    rows = []
    for s, ra, rb in zip(selection['samples'], a_records, b_records):
        assert all(s[k] == ra[k] == rb[k] for k in IDENTITY)
        assert ra['status'] == 'ok'
        a, gold = ranking(RUN/ra['result_file'], ra['result_sha256'])
        assert gold == set(ra['ground_truth'])
        ok = rb['status'] == 'ok'
        b = []
        if ok:
            b, bgold = ranking(RUN/rb['result_file'], rb['result_sha256'])
            assert gold == bgold and set(a) == set(b)
            assert ra['candidate_fingerprint'] == rb['candidate_fingerprint']
        rows.append(case(s, a, b, gold, ok))
        if len(rows) % 250 == 0:
            print(f'Checked {len(rows)}/2294', flush=True)
    old = historical()
    dev = set(selection['development_rows'])
    cohorts = {'all2294': rows, 'development500_current': [r for r in rows if r['dataset_row'] in dev],
               'remaining1794': [r for r in rows if r['dataset_row'] not in dev],
               'historical500': old}
    summary = {k: summarize(v) for k, v in cohorts.items()}
    summary['projects_full'] = {p: summarize([r for r in rows if r['repository'] == p])
                                for p in sorted({r['repository'] for r in rows})}
    summary['protocol'] = {'source': str(RUN.resolve()),
        'pair_overlap_denominator': '2293 successful pairs; failures are not zero overlap',
        'effectiveness_denominator': 'All 2294; B and mixed failure zero; union retains A',
        'union': 'Diagnostic variable-size union; not a fixed-50 ranking result',
        'mixed': 'B top40 then unique A padding to50, and reverse diagnostic',
        'labels': 'All saved ground_truth labels, including absent inventory labels',
        'gold_count_unit': 'Report-file pairs; same path in different reports counted separately',
        'checks': ['full result SHA256', 'paired identities', 'equal complete inventories and gold',
                   'unique consecutive ranks', 'no model calls or experiment writes']}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in [('summary.json', summary), ('cases_full.json', rows), ('cases_historical500.json', old)]:
        (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    verify_saved_pools(rows)
    additional_diagnostics(rows)
    with (OUT/'cases_full.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k,v in row.items()})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('projects_full','protocol')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
