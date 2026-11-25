"""Frozen 2x2 completion and genuinely unused holdout; no changes to historic runs."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bug_localization.batch import write_json, aggregate_results
from bug_localization.bm25 import ranking_metrics
from bug_localization.chunk_ablation import select_evidence
from bug_localization.embeddings import digest
from bug_localization.mixed_recall import mixed_candidates
from bug_localization.paired_statistics import analyze
from bug_localization.reranking import read_json, LocalReranker, rerank
from bug_localization.snapshot import sha256_file
from run_mixed_recall500 import Sources, same_sample, now, verify_frozen

SEED = 20261112
HISTORIC = PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500'
SELECTION = PROJECT / 'configs/experiments/selections/c3_lca_py_holdout_seed20261112.json'


def verify_historical():
    import ast
    protocol = read_json(HISTORIC / 'experiment.json')
    compatibility = []
    for name, expected in protocol['files_sha256'].items():
        path = Path(name)
        if sha256_file(path) == expected:
            continue
        if path != PROJECT / 'src/bug_localization/bm25.py':
            raise ValueError('Historical input changed: ' + name)
        revision = 'b230fce'
        old = subprocess.check_output(['git', 'show', revision + ':src/bug_localization/bm25.py'], cwd=PROJECT)
        if hashlib.sha256(old).hexdigest() != expected:
            raise ValueError('Cannot recover exact historical BM25 implementation')
        before = {n.name: ast.dump(n) for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
        changed = {n for n in before.keys() | after.keys() if before.get(n) != after.get(n)}
        if changed != {'analyze_query', 'run'}:
            raise ValueError('Historical evaluation/preparation functions changed')
        compatibility.append({'file': name, 'historical_revision': revision, 'historical_sha256': expected,
            'current_sha256': sha256_file(path), 'changed_functions': sorted(changed),
            'reason': 'Only BM25 analyzer limit handling changed; semantic pipeline and ranking_metrics unchanged by AST comparison.'})
    for method in ('C2', 'C3'):
        for record in read_json(HISTORIC / method / 'samples.json'):
            if record['status'] == 'ok' and sha256_file(HISTORIC / method / record['result_file']) != record['result_sha256']:
                raise ValueError('Historical result modified')
    return compatibility


def freeze(folder, mode):
    config = read_json(PROJECT / 'configs/experiments/chunk_ablation500.json')
    if mode == 'holdout':
        config.update(new_selection=str(SELECTION.relative_to(PROJECT)), selection=str(SELECTION.relative_to(PROJECT)),
            semantic_reference=config['old_semantic'], generation_config='configs/experiments/report_hypothesis_crossproject180.json',
            delete_only_new_temporary_indexes=True)
    files = list((PROJECT / 'src/bug_localization').glob('*.py'))
    files += [Path(__file__), PROJECT / 'scripts/analysis/analyze_c3_supplement.py',
              PROJECT / 'scripts/experiments/run_reranker500.py', PROJECT / 'scripts/experiments/run_direct_hypothesis.py',
              PROJECT / 'scripts/experiments/run_mixed_recall500.py', PROJECT / config['selection'],
              PROJECT / config['reranker_config']]
    if mode == 'holdout':
        files += [PROJECT / 'datasets/lca_py_formatted/test/dataset_info.json', PROJECT / 'datasets/lca_py_formatted/lca_sources.json', PROJECT / config['generation_config'], PROJECT / config['semantic_reference'] / 'effective_config.json',
                  PROJECT / config['semantic_reference'] / 'embedding_identity.json']
    else:
        verify_historical()
        files += [HISTORIC / 'experiment.json']
    spec = {'historical_compatibility_audit': verify_historical(), 'mode': mode, 'config': config, 'files_sha256': {str(p.resolve()): sha256_file(p) for p in files},
        'primary_comparison': 'C3 - A', 'primary_metric': 'MAP@10', 'seed': SEED,
        'methods': {'A': ['original', 'original'], 'A3': ['original', 'mixed'], 'C1': ['mixed', 'original'], 'C3': ['mixed', 'mixed']},
        'holdout_policy': 'Unused LCA Python test reports after historical exposure audit; no tuning, replacements, or outcome filtering; failures zero',
        'cost_policy': 'Local inference; no API charges. Online stage sums exclude shared snapshot/chunk/vector/index preparation and model loading. Report warm cache state explicitly. No invented currency rates.'}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'protocol.json'
    if path.exists():
        if read_json(path) != spec:
            raise ValueError('Frozen protocol changed; refusing resume')
    else:
        write_json(path, spec)
        write_json(folder / 'selection.json', read_json(PROJECT / config['selection']))
        write_json(folder / 'frozen_at.json', {'created_at': now()})
    return spec


def normalized_body(body):
    return ' '.join(body.split()).casefold()


def prepare_holdout():
    if SELECTION.exists():
        return
    from datasets import load_from_disk
    data = load_from_disk(str(PROJECT / 'datasets/lca_py_formatted'))['test']
    swe = load_from_disk(str(PROJECT / 'datasets/swe_bench_formatted'))['test']
    used_bodies = {normalized_body(r['body']) for r in swe}
    used_issues = {(r['repo_name'].casefold(), str(r['issue_id'])) for r in swe}
    used_commits = {(r['repo_name'].casefold(), r['before_fix_sha']) for r in swe}
    metadata = sha256_file(PROJECT / 'datasets/lca_py_formatted/test/dataset_info.json')
    manifests = sorted(set((PROJECT / 'results').rglob('selection.json')) | set((PROJECT / 'configs/experiments/selections').glob('*.json')))
    evidence, exposed = {}, set()
    for path in manifests:
        value = read_json(path)
        if not isinstance(value, dict) or 'samples' not in value:
            continue
        evidence[str(path.relative_to(PROJECT))] = sha256_file(path)
        for row in value['samples']:
            if value.get('dataset_metadata_sha256') == metadata:
                exposed.add(row['dataset_row'])
            if 'repository' in row and 'before_fix_sha' in row:
                used_commits.add((row['repository'].casefold(), row['before_fix_sha']))
    for i in exposed:
        row = data[i]
        used_bodies.add(normalized_body(row['body']))
        used_issues.add((row['repo_name'].casefold(), row['issue_url'].rstrip('/').split('/')[-1]))
    for path in (PROJECT / 'cache/direct_report_hypothesis').rglob('*.json'):
        body = read_json(path).get('payload', {}).get('original_report')
        if body is not None:
            used_bodies.add(normalized_body(body))
    rows, excluded = [], []
    for i, row in enumerate(data):
        repo = row['repo_name'].casefold()
        issue = row['issue_url'].rstrip('/').split('/')[-1]
        reasons = []
        if i in exposed: reasons.append('prior_LCA_manifest')
        if normalized_body(row['body']) in used_bodies: reasons.append('historical_or_duplicate_body')
        if (repo, issue) in used_issues: reasons.append('historical_or_duplicate_issue')
        if (repo, row['before_fix_sha']) in used_commits: reasons.append('historical_snapshot')
        if reasons:
            excluded.append({'dataset_row': i, 'repository': repo, 'issue_url': row['issue_url'], 'reasons': reasons})
            continue
        rows.append({'position': len(rows) + 1, 'dataset_row': i, 'repository': row['repo_name'], 'issue_id': row['issue_id'],
            'before_fix_sha': row['before_fix_sha'],
            'query_sha256': hashlib.sha256(json.dumps({'body': row['body']}, sort_keys=True).encode()).hexdigest(),
            'labels_sha256': hashlib.sha256(str(row['updated_files']).encode()).hexdigest()})
        used_bodies.add(normalized_body(row['body'])); used_issues.add((repo, issue))
    if not rows:
        raise ValueError('No unused LCA Python test reports')
    write_json(SELECTION, {'samples': rows, 'selection_sha256': digest(rows),
        'dataset_metadata_sha256': metadata, 'seed': SEED, 'created_at': now(), 'dataset': 'lca_py_formatted',
        'selection_rule': 'All LCA Python test rows excluding prior LCA exposure, normalized report body, canonical issue identity and historical snapshot overlap; no outcome filtering. Body-only query preserves SWE protocol.',
        'project_counts': dict(Counter(r['repository'] for r in rows)), 'historical_manifest_sha256': evidence,
        'excluded_reports': excluded, 'historical_SWE_reports_checked': len(swe), 'raw_LCA_test_reports': len(data)})


def configure_pipeline(pipeline):
    from bug_localization.vector import select_reports
    def context(config):
        reference = read_json(PROJECT / config['semantic_reference'] / 'effective_config.json')
        semantic = {**reference['semantic'], 'selection_manifest': config['new_selection']}
        base = {**reference['base'], 'dataset': 'lca_py_formatted', 'source_provider': 'lca_git'}
        selected, selection = select_reports(PROJECT, semantic, base, max_samples=500)
        return semantic, base, selected, selection
    pipeline.context = context
    from bug_localization.snapshot import get_snapshot
    pipeline.get_snapshot = lambda project, repo, commit: get_snapshot(project, repo, commit,
        source_provider='lca_git', source_dataset='lca_py_formatted')


class HoldoutSources:
    def __init__(self, folder, config):
        self.folder = folder
        self.samples = read_json(PROJECT / config['selection'])['samples']
        self.original = folder / 'new320/semantic'
        self.hypothesis = folder / 'new320/hypothesis'
        self.records = [read_json(self.original / 'samples.json')['q0'], read_json(self.hypothesis / 'samples.json')['direct_hypothesis']]
        for records in self.records:
            if len(records) != len(self.samples):
                raise ValueError('Incomplete holdout retrieval')
            for a, b in zip(records, self.samples):
                same_sample(a, b)
        self.instruction = read_json(self.original / 'effective_config.json')['semantic']['query_instruction']

    def failures(self, sample, original_only=False):
        return [rows[sample['position'] - 1] for rows in self.records[:1 if original_only else 2]
                if rows[sample['position'] - 1]['status'] != 'ok']

    def load(self, sample, original_only=False):
        pos = f"{sample['position']:03d}"
        oc, hc = self.original / 'cases' / pos, self.hypothesis / 'cases' / pos
        paths = [oc / 'q0_retrieval.json', oc / 'q0_result.json', oc / 'chunks.json', oc / 'candidates.json']
        original, result, chunks, inventory = [read_json(p) for p in paths]
        if original_only:
            hypothesis = original
        else:
            paths += [hc / 'direct_retrieval.json', hc / 'direct_result.json']
            hypothesis, hr = [read_json(p) for p in paths[-2:]]
            if hr['ground_truth'] != result['ground_truth']:
                raise ValueError('Ground truth mismatch')
        wrapper = 'Instruct: ' + self.instruction + '\nQuery: '
        if not original['query_info']['embedding_text'].startswith(wrapper):
            raise ValueError('Unexpected query wrapper')
        for retrieval in (original, hypothesis):
            if digest(retrieval['chunk_scores']) != retrieval['chunk_scores_sha256']:
                raise ValueError('Chunk scores changed')
        return {'original': original, 'hypothesis': hypothesis, 'query': original['query_info']['embedding_text'][len(wrapper):],
            'chunks': chunks, 'gold': result['ground_truth'], 'paths': paths,
            'inventory': {r['file_path'] for r in inventory['files']}}


def run_cells(folder, spec):
    source = Sources(spec['config']) if spec['mode'] == 'factorial' else HoldoutSources(folder, spec['config'])
    settings = read_json(PROJECT / spec['config']['reranker_config'])
    started = time.perf_counter()
    model = LocalReranker(PROJECT, settings)
    reference = read_json(PROJECT / spec['config']['reference_reranker'])['model']
    if model.identity != reference:
        raise ValueError('Reranker identity differs from historical control')
    write_json(folder / 'reranker_identity.json', {'model': model.identity, 'load_seconds': time.perf_counter() - started})
    methods = ['A3'] if spec['mode'] == 'factorial' else ['A', 'C3']
    all_records = {}
    for method in methods:
        target = folder / method; target.mkdir(exist_ok=True)
        all_records[method] = read_json(target / 'samples.json') if (target / 'samples.json').exists() else []
        for sample, record in zip(source.samples, all_records[method]):
            same_sample(sample, record)
            if record['status'] == 'ok' and sha256_file(target / record['result_file']) != record['result_sha256']:
                raise ValueError('Checkpoint result modified')
        if len(all_records[method]) > len(source.samples):
            raise ValueError('Checkpoint exceeds selection')
    # Warm-up explicitly excluded from online measurements.
    model.score('Locate the reported bug.', ['def example():\n    return None'])
    for sample in source.samples:
        pos = sample['position']
        # Alternate order to reduce method/order timing confounding.
        for method in methods[::1 if pos % 2 else -1]:
            records = all_records[method]; target = folder / method
            if len(records) >= pos:
                continue
            failures = source.failures(sample) if spec['mode'] == 'factorial' else source.failures(sample, method == 'A')
            if failures:
                record = {**sample, 'status': 'failed', 'error_type': 'SourceDependencyFailure', 'error': str(failures)}
            else:
                try:
                    case = source.load(sample) if spec['mode'] == 'factorial' else source.load(sample, method == 'A')
                    import torch
                    torch.cuda.synchronize(); start = time.perf_counter()
                    candidates = (case['original']['ranking'][:50] if method in ('A', 'A3') else
                                  mixed_candidates(case['original']['ranking'], case['hypothesis']['ranking'], 'hypothesis'))
                    policy = 'original' if method == 'A' else 'mixed'
                    head, chosen = select_evidence(candidates, case['original']['ranking'], case['chunks'],
                        case['original']['chunk_scores'], case['hypothesis']['chunk_scores'], policy)
                    selection_seconds = time.perf_counter() - start
                    torch.cuda.synchronize(); start = time.perf_counter()
                    scored = model.score(case['query'], [r['embedding_text'] for r in chosen])
                    ranking = rerank(head, chosen, scored)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                    metrics = ranking_metrics([r['file_path'] for r in ranking], case['gold'], 10)
                    path = target / 'cases' / f'{pos:03d}.json'; path.parent.mkdir(exist_ok=True)
                    result = {**sample, 'query': case['query'], 'ground_truth': case['gold'], 'candidates': candidates,
                        'chunk_policy': policy, 'ranking': ranking, 'metrics': metrics,
                        'scored_chunks': [{**r, **s} for r, s in zip(chosen, scored)],
                        'source_files_sha256': {str(p): sha256_file(p) for p in case['paths']}}
                    write_json(path, result)
                    record = {**sample, 'status': 'ok', 'metrics': metrics, 'result_file': str(path.relative_to(target)),
                        'result_sha256': sha256_file(path), 'selection_seconds': selection_seconds, 'rerank_seconds': elapsed,
                        'scored_pairs': len(scored), 'reranker_input_tokens': sum(r['input_tokens'] for r in scored)}
                except Exception as error:
                    record = {**sample, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()}
            records.append(record)
            write_json(target / 'samples.json', records)
            write_json(target / 'progress.json', {'processed': len(records), 'selected': len(source.samples),
                'failures': sum(r['status'] != 'ok' for r in records), 'updated_at': now()})
            print(f'{method} {pos}/{len(source.samples)} {record["status"]}', flush=True)
    for method, records in all_records.items():
        write_json(folder / method / 'summary.json', aggregate_results(records, 10))
    return all_records


def audit_cells(folder, spec):
    source = Sources(spec['config']) if spec['mode'] == 'factorial' else HoldoutSources(folder, spec['config'])
    methods = ['A3'] if spec['mode'] == 'factorial' else ['A', 'C3']
    from collections import defaultdict
    for method in methods:
        target = folder / method; records = read_json(target / 'samples.json')
        if len(records) != len(source.samples):
            raise ValueError('Incomplete cells')
        expected, unexpected = [], []
        for sample, record in zip(source.samples, records):
            same_sample(sample, record)
            deps = source.failures(sample) if spec['mode'] == 'factorial' else source.failures(sample, method == 'A')
            if record['status'] != 'ok':
                (expected if deps else unexpected).append(sample['position']); continue
            if deps:
                raise ValueError('Unexpected fallback')
            saved = read_json(target / record['result_file'])
            if sha256_file(target / record['result_file']) != record['result_sha256']:
                raise ValueError('Result hash mismatch')
            case = source.load(sample) if spec['mode'] == 'factorial' else source.load(sample, method == 'A')
            if saved['source_files_sha256'] != {str(p): sha256_file(p) for p in case['paths']}:
                raise ValueError('Sources changed')
            paths = [r['file_path'] for r in case['original']['ranking'][:50]]
            if method == 'C3':
                paths = [r['file_path'] for r in case['hypothesis']['ranking'][:40]]
                for r in case['original']['ranking']:
                    if len(paths) >= 50:
                        break
                    if r['file_path'] not in paths:
                        paths.append(r['file_path'])
            if [r['file_path'] for r in saved['candidates']] != paths:
                raise ValueError('Candidate policy mismatch')
            grouped = {route: defaultdict(list) for route in ('original', 'hypothesis')}
            for route in grouped:
                for r in case[route]['chunk_scores']:
                    grouped[route][r['file_path']].append(r)
                for rs in grouped[route].values():
                    rs.sort(key=lambda r: (-r['cosine_similarity'], r['chunk_id']))
            ranks = {r['file_path']: r['rank'] for r in case['original']['ranking']}
            expected_ids = []
            for path in sorted(paths, key=ranks.__getitem__):
                ids = [r['chunk_id'] for r in grouped['original'][path][:3 if method == 'A' else 1]]
                if method != 'A':
                    for r in grouped['hypothesis'][path]:
                        if len(ids) >= 3:
                            break
                        if r['chunk_id'] not in ids:
                            ids.append(r['chunk_id'])
                expected_ids.extend(ids)
            scored = saved['scored_chunks']
            if [r['chunk_id'] for r in scored] != expected_ids:
                raise ValueError('Chunk selection mismatch')
            lookup = {r['chunk_id']: r for r in case['chunks']}
            best = {}
            for r in scored:
                if any(r[k] != lookup[r['chunk_id']][k] for k in ('embedding_text', 'file_path')):
                    raise ValueError('Code text mismatch')
                best[r['file_path']] = max(best.get(r['file_path'], -float('inf')), r['score'])
            ordered = sorted(paths, key=lambda p: (-best[p], ranks[p]))
            metrics = ranking_metrics(ordered, case['gold'], 10)
            if saved['query'] != case['query'] or saved['ground_truth'] != case['gold'] or metrics != record['metrics'] or metrics != saved['metrics']:
                raise ValueError('Query/label/metric mismatch')
            if ordered != [r['file_path'] for r in saved['ranking']] or any(r['score'] != best[r['file_path']] for r in saved['ranking']):
                raise ValueError('Aggregation mismatch')
        write_json(target / 'verification.json', {'verified_successes': len(records) - len(expected) - len(unexpected),
            'expected_source_failures': expected, 'unexpected_failures': unexpected})
        if unexpected:
            raise ValueError(f'{method}: unexpected failures {unexpected}')


def timed_generation(folder, config):
    import run_reranker500 as pipeline
    configure_pipeline(pipeline)
    from run_direct_hypothesis import Generator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    class TimedGenerator(Generator):
        def __init__(self, cfg):
            super().__init__(cfg)
            start = time.perf_counter()
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(self.path, local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
            torch.cuda.synchronize()
            write_json(folder / 'generator_identity.json', {'identity': self.identity, 'load_seconds': time.perf_counter() - start})
            # Separate cache guarantees a newly measured generation, not a historical cache hit.
            self.folder = folder / 'measured_generation_cache'; self.folder.mkdir(exist_ok=True)
        def generate(self, report, cache_only=False):
            key = digest({'identity': self.identity, 'payload': {'original_report': report}})
            path = folder / 'generation_timings' / (key + '.json'); path.parent.mkdir(exist_ok=True)
            if path.exists():
                return super().generate(report, cache_only)
            cache_path = self.folder / (key + '.json')
            if cache_path.exists():
                cache_path.unlink()  # Interrupted measurement: regenerate in this run's private cache.
            torch.cuda.synchronize(); start = time.perf_counter()
            try:
                value = super().generate(report, cache_only)
            except Exception as error:
                torch.cuda.synchronize()
                write_json(path, {'report_sha256': digest(report), 'seconds': time.perf_counter() - start, 'error': str(error)})
                raise
            torch.cuda.synchronize()
            write_json(path, {'report_sha256': digest(report), 'seconds': time.perf_counter() - start,
                'input_tokens': value['input_tokens'], 'output_tokens': value['output_tokens'], 'generation_error': value['generation_error']})
            return value
    pipeline.Generator = TimedGenerator
    pipeline.generate(folder, config)


def timed_retrieval(folder, config):
    import run_reranker500 as pipeline
    configure_pipeline(pipeline)
    import torch
    Base = pipeline.LocalEmbedder
    original_retrieve = pipeline.retrieve
    # Each report is identified by its query body hash; all attempts are persisted.
    class TimedEmbedder(Base):
        def query(self, body):
            torch.cuda.synchronize(); start = time.perf_counter()
            before = self.misses
            value = super().query(body)
            torch.cuda.synchronize()
            self.last_query_timing = {'query_sha256': digest(body), 'query_seconds': time.perf_counter() - start,
                                      'query_embedding_cache_misses': self.misses - before}
            return value
    encoder_ref = []
    class CapturedEmbedder(TimedEmbedder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); encoder_ref.append(self)
    def measured_retrieve(*args, **kwargs):
        # First call prepares/reuses the index. Second measures the online search with it ready.
        start = time.perf_counter(); original_retrieve(*args, **kwargs)
        prep = time.perf_counter() - start
        start = time.perf_counter(); result = original_retrieve(*args, **kwargs)
        elapsed = time.perf_counter() - start
        timing = {**encoder_ref[0].last_query_timing, 'online_retrieve_seconds': elapsed,
                  'index_preparation_plus_first_search_seconds': prep, 'index': result['index']}
        target = folder / 'retrieval_timings'; target.mkdir(exist_ok=True)
        write_json(target / (digest({'index': result['index'], 'query': timing['query_sha256']}) + '.json'), timing)
        return result
    pipeline.LocalEmbedder = CapturedEmbedder
    pipeline.retrieve = measured_retrieve
    pipeline.semantic_retrieval(folder, config)


def summarize_cost(folder, config):
    import numpy as np
    import run_reranker500 as pipeline
    configure_pipeline(pipeline)
    _, _, selected, selection = pipeline.context(config)
    generation = {r['report_sha256']: r for p in (folder / 'generation_timings').glob('*.json') for r in [read_json(p)]}
    retrieval = {}
    for p in (folder / 'retrieval_timings').glob('*.json'):
        r = read_json(p); retrieval[(r['index'], r['query_sha256'])] = r
    records = {m: read_json(folder / m / 'samples.json') for m in ('A', 'C3')}
    costs = []
    for i, (sample, meta) in enumerate(zip(selected, selection['samples'])):
        body = sample['report']['body']; g = generation.get(digest(body))
        row = {**meta, 'generation': g, 'methods': {}}
        for method in records:
            record = records[method][i]
            if record['status'] != 'ok':
                row['methods'][method] = {'status': 'failed', 'online_stage_sum_seconds': None}; continue
            times = []
            for route, stem in [('semantic', 'q0')] + ([('hypothesis', 'direct')] if method == 'C3' else []):
                result = read_json(folder / 'new320' / route / 'cases' / f'{i+1:03d}' / (stem + '_retrieval.json'))
                times.append(retrieval[(result['index'], digest(result['query']))])
            total = record['selection_seconds'] + record['rerank_seconds'] + sum(t['query_seconds'] + t['online_retrieve_seconds'] for t in times)
            if method == 'C3':
                if g is None or 'error' in g:
                    raise ValueError('Missing successful generation timing')
                total += g['seconds']
            row['methods'][method] = {'status': 'ok', 'online_stage_sum_seconds': total,
                'retrieval_stages': times, 'selection_seconds': record['selection_seconds'], 'rerank_seconds': record['rerank_seconds'],
                'reranker_input_tokens': record['reranker_input_tokens']}
        costs.append(row)
    summary = {}
    for method in records:
        ts = [r['methods'][method]['online_stage_sum_seconds'] for r in costs if r['methods'][method]['status'] == 'ok']
        summary[method] = {'successful_timed_reports': len(ts), 'failed': len(costs) - len(ts),
            'mean_seconds': float(np.mean(ts)) if ts else None, 'p50_seconds': float(np.median(ts)) if ts else None,
            'p95_seconds': float(np.quantile(ts, .95)) if ts else None,
            'local_device_wall_hours': sum(ts) / 3600, 'currency_cost': None}
    paired = [r for r in costs if all(r['methods'][m]['status'] == 'ok' for m in records)]
    delta = float(np.mean([r['methods']['C3']['online_stage_sum_seconds'] - r['methods']['A']['online_stage_sum_seconds'] for r in paired])) if paired else None
    write_json(folder / 'costs.json', {'timing_definition': 'Sum of measured sequential online stages on one GPU, not a request-server wall-clock benchmark; warm prepared index, observed query cache state; excludes shared preprocessing/model loads/file IO. First generation may include kernel warmup.',
        'summary': summary, 'paired_successes': len(paired), 'paired_mean_extra_seconds': delta,
        'currency_note': 'Local models, no API bill; currency conversion requires an explicit hardware/electricity rate.',
        'rows': costs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['factorial', 'holdout'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['generate', 'retrieve', 'cells', 'audit'])
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); folder = a.output.resolve()
    if a.mode == 'holdout':
        prepare_holdout()
    spec = freeze(folder, a.mode)
    if a.prepare_only:
        print(folder); return
    if a.stage:
        {'generate': lambda: timed_generation(folder, spec['config']),
         'retrieve': lambda: timed_retrieval(folder, spec['config']),
         'cells': lambda: run_cells(folder, spec), 'audit': lambda: audit_cells(folder, spec)}[a.stage]()
        return
    with (folder / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            stages = ['cells', 'audit'] if a.mode == 'factorial' else ['generate', 'retrieve', 'cells', 'audit']
            for stage in stages:
                marker = folder / (stage + '.completed.json')
                if marker.exists():
                    continue
                write_json(folder / 'status.json', {'status': 'running', 'stage': stage, 'updated_at': now()})
                start = time.perf_counter()
                with (folder / (stage + '.log')).open('a') as log:
                    subprocess.run([sys.executable, '-u', str(Path(__file__)), '--mode', a.mode, '--output', str(folder), '--stage', stage],
                        cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
                write_json(marker, {'completed_at': now(), 'stage_wall_seconds': time.perf_counter() - start})
            if a.mode == 'factorial':
                subprocess.run([sys.executable, str(PROJECT / 'scripts/analysis/analyze_c3_supplement.py'),
                    '--factorial', str(folder), '--output', str(folder / 'statistics')], check=True)
            else:
                methods = {m: read_json(folder / m / 'samples.json') for m in ('A', 'C3')}
                write_json(folder / 'paired_statistics.json', analyze(methods, {'C3_vs_A': {'C3': 1, 'A': -1}}))
                summarize_cost(folder, spec['config'])
            freeze(folder, a.mode)
            write_json(folder / 'status.json', {'status': 'completed', 'updated_at': now()})
        except BaseException as error:
            write_json(folder / 'status.json', {'status': 'failed', 'error': str(error), 'updated_at': now(), 'traceback': traceback.format_exc()})
            raise

if __name__ == '__main__':
    main()
