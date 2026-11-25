"""Fixed C3 candidates/evidence, replacing only the truncated reranker report.

Fresh 0.6B yes/no scores use the complete original report. No BLAgent prediction,
patch, label or generated hypothesis is supplied to the model. Oversized inputs
fail explicitly; no report cleaning, truncation, fallback or candidate expansion.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / p) for p in
               ('.runtime/blagent-deps', '.runtime/rrl-deps', '.runtime/igsil-deps', 'src')]
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.embeddings import digest
from bug_localization.model_files import required_model_files
from bug_localization.reranking import LocalReranker, PREFIX, SUFFIX, rerank
from bug_localization.snapshot import sha256_file

C3 = PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/C3'
REPORTS = PROJECT / 'results/blagent/recovery-20261118/dual-v4/full2294'
IDENTITY = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')


def read(path):
    return json.loads(path.read_text())


def key(row):
    return tuple(row[k] for k in IDENTITY)


def evidence_inputs(saved):
    """Return model-visible evidence and semantic tie order, excluding labels."""
    chunks = [{k: c[k] for k in ('file_path', 'chunk_id', 'embedding_text')}
              for c in saved['scored_chunks']]
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Duplicate C3 evidence chunk identity')
    paths = {c['file_path'] for c in chunks}
    ranking = sorted(saved['ranking'], key=lambda r: r['semantic_rank'])
    if len({r['file_path'] for r in ranking}) != len(ranking):
        raise ValueError('Duplicate C3 file identity')
    if not paths.issubset({r['file_path'] for r in ranking}):
        raise ValueError('Evidence path outside frozen C3 ranking')
    # rerank() uses this rank only for ties and preserves the unscored tail.
    head = [{'rank': r['semantic_rank'], 'file_path': r['file_path'],
             'score': r['semantic_score']} for r in ranking if r['file_path'] in paths]
    tail = [{'rank': r['semantic_rank'], 'file_path': r['file_path'],
             'score': r['semantic_score']} for r in saved['ranking'] if r['file_path'] not in paths]
    return head + tail, chunks


def score_evidence(query, original_order, chunks, model):
    """Inference boundary: only original body, frozen code and tie order."""
    scores = model.score(query, [c['embedding_text'] for c in chunks])
    ranking = rerank(original_order, chunks, scores)
    return ranking, [{**c, **s} for c, s in zip(chunks, scores)]


def input_budget_report(prepared, config):
    """Exact tokenizer preflight; does not instantiate model weights or CUDA."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROJECT / config['model_path'], local_files_only=True,
                                              trust_remote_code=False, padding_side='left')
    wrappers = len(tokenizer.encode(PREFIX, add_special_tokens=False)) + len(tokenizer.encode(SUFFIX, add_special_tokens=False))
    rows = []
    for entry in prepared:
        row = {**entry['sample']}
        if entry['source']['status'] != 'ok':
            row.update(status='source_failed', error=entry['source'].get('error'))
        else:
            query = entry['query']
            texts = [f'<Instruct>: {config["instruction"]}\n<Query>: {query}\n<Document>: {c["embedding_text"]}'
                     for c in entry['chunks']]
            counts = [len(ids) + wrappers for ids in tokenizer(texts, add_special_tokens=False,
                                                              padding=False, truncation=False)['input_ids']]
            row.update(status='ready' if max(counts) <= config['max_tokens'] else 'would_fail_context_limit',
                       query_tokens=len(tokenizer.encode(query, add_special_tokens=False)),
                       baseline_query_tokens=len(tokenizer.encode(entry['saved']['query'], add_special_tokens=False)),
                       max_input_tokens=max(counts), scored_pairs=len(counts),
                       over_budget_pairs=sum(n > config['max_tokens'] for n in counts))
        rows.append(row)
    return {'max_tokens':config['max_tokens'],
            'maximum_input_tokens':max((r.get('max_input_tokens',0) for r in rows), default=0),
            'samples_exceeding_budget':sum(r['status']=='would_fail_context_limit' for r in rows),
            'cases':rows}


def prepare(selection_path, config):
    selection = read(selection_path)
    samples = selection['samples']
    if not samples or len({key(s) for s in samples}) != len(samples):
        raise ValueError('Selection must contain unique report identities')
    if len({s['position'] for s in samples}) != len(samples):
        raise ValueError('Selection positions must be unique')
    c3_records = {key(r): r for r in read(C3 / 'samples.json')}
    report_records = {key(r): r for r in read(REPORTS / 'selection.json')['samples']}
    files = [selection_path, C3 / 'samples.json', REPORTS / 'selection.json', Path(__file__),
             PROJECT / 'src/bug_localization/reranking.py', PROJECT / 'src/bug_localization/batch.py',
             PROJECT / 'src/bug_localization/bm25.py', PROJECT / 'src/bug_localization/embeddings.py',
             PROJECT / 'src/bug_localization/snapshot.py', PROJECT / 'src/bug_localization/model_files.py']
    prepared = []
    for sample in samples:
        source = c3_records[key(sample)]
        report_record = report_records[key(sample)]
        for field in ('query_sha256', 'labels_sha256'):
            if field in sample and sample[field] != source[field]:
                raise ValueError('Selected query/label identity differs from C3')
            if source[field] != report_record[field]:
                raise ValueError('Original report and C3 query/label identities differ')
        entry = {'sample': sample, 'source': source}
        if source['status'] == 'ok':
            case_path = C3 / source['result_file']
            if sha256_file(case_path) != source['result_sha256']:
                raise ValueError('C3 source result checksum mismatch')
            inputs_path = REPORTS / 'cases' / f'{report_record["position"]:04d}' / 'inputs.json'
            inputs = read(inputs_path)
            saved = read(case_path)
            if inputs['source_commit'] != sample['before_fix_sha'] or key(saved) != key(sample):
                raise ValueError('Report/source commit differs from frozen identity')
            query = inputs['query']
            if not isinstance(query, str) or not query.strip() or not query.startswith(saved['query']):
                raise ValueError('Original complete body does not extend the saved C3 query')
            order, chunks = evidence_inputs(saved)
            entry.update(saved=saved, query=query, original_order=order, chunks=chunks,
                         source_case=case_path, source_inputs=inputs_path)
            files.extend([case_path, inputs_path])
        prepared.append(entry)
    model_path = PROJECT / config['model_path']
    files.extend(model_path / p for p in required_model_files(model_path))
    protocol = {'method': 'C3 full original report + same 0.6B chunk reranker',
                'config': config, 'selection_sha256': digest(samples),
                'query_policy': 'complete original body; preserve HTML and whitespace; no truncation',
                'candidate_policy': 'exact saved C3 pool and selected chunks; file max; semantic-rank ties',
                'evaluation_policy': 'all selected reports; failed attempts zero; no fallback',
                'files_sha256': {str(p.resolve()): sha256_file(p) for p in sorted(set(files))}}
    return selection, prepared, protocol


def run(selection_path, output, config, check_only=False):
    selection, prepared, protocol = prepare(selection_path, config)
    if check_only:
        return {'status': 'preflight_passed', 'samples': len(prepared),
                'source_failures': sum(e['source']['status'] != 'ok' for e in prepared),
                'selection_sha256': protocol['selection_sha256'],
                'input_budget': input_budget_report(prepared, config)}
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = output / 'protocol.json'
        if protocol_path.exists() and read(protocol_path) != protocol:
            raise ValueError('Resume protocol differs; use a new output directory')
        write_json(protocol_path, protocol)
        write_json(output / 'selection.json', selection)
        records = read(output / 'samples.json') if (output / 'samples.json').exists() else []
        if len(records) > len(prepared):
            raise ValueError('Resume contains too many reports')
        for record, entry in zip(records, prepared):
            if key(record) != key(entry['sample']) or record['position'] != entry['sample']['position']:
                raise ValueError('Resume report identity/order differs')
            path = output / record['result_file']
            if sha256_file(path) != record['result_sha256']:
                raise ValueError('Resume case checksum differs')
        def checkpoint():
            summary = aggregate_results(records, 10, len(prepared))
            if summary['pending_samples'] == 0 and summary['failed_samples']:
                summary['status'] = 'completed_with_failures'
            summary.update(method=protocol['method'])
            write_json(output / 'samples.json', records)
            write_json(output / 'summary.json', summary)
            return summary
        checkpoint()
        if len(records) == len(prepared):
            return checkpoint()
        model = LocalReranker(PROJECT, config)
        identity_path = output / 'model_identity.json'
        if identity_path.exists() and read(identity_path) != model.identity:
            raise ValueError('Resume model/runtime identity differs')
        write_json(identity_path, model.identity)
        cases = output / 'cases'; cases.mkdir(exist_ok=True)
        for entry in prepared[len(records):]:
            started = time.monotonic()
            sample, source = entry['sample'], entry['source']
            record = {**sample, 'source_c3_position': source['position']}
            case = {**record}
            try:
                if source['status'] != 'ok':
                    raise ValueError('Frozen C3 source failed: ' + source.get('error', 'unknown error'))
                query = entry['query']
                case.update(query=query, reranker_query_sha256=digest(query),
                            baseline_query=entry['saved']['query'],
                            baseline_query_truncated=entry['saved']['query_truncated_by_semantic_encoder'],
                            selected_chunk_ids=[c['chunk_id'] for c in entry['chunks']],
                            source_case_sha256=sha256_file(entry['source_case']),
                            source_inputs_sha256=sha256_file(entry['source_inputs']))
                ranked, scored = score_evidence(query, entry['original_order'], entry['chunks'], model)
                gold = entry['saved']['ground_truth']
                metrics = ranking_metrics([r['file_path'] for r in ranked], gold, 10)
                case.update(status='ok', ranking=ranked, scored_chunks=scored, ground_truth=gold, metrics=metrics)
                record.update(status='ok', metrics=metrics, missing_gold=source['missing_gold'], scored_pairs=len(scored))
            except (OSError,ConnectionError,TimeoutError,RuntimeError) as error:
                write_json(output/'infrastructure_error.json',{'sample':sample,'error':repr(error),'traceback':traceback.format_exc()})
                raise
            except Exception as error:
                case.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                record.update(status='failed', error_type=type(error).__name__, error=str(error))
            elapsed = time.monotonic() - started
            case['elapsed_seconds'] = elapsed
            path = cases / f'{sample["position"]:04d}.json'
            write_json(path, case)
            record.update(result_file=str(path.relative_to(output)), result_sha256=sha256_file(path), elapsed_seconds=elapsed)
            records.append(record)
            checkpoint()
            print(f'{len(records)}/{len(prepared)} {sample["repository"]} #{sample["issue_id"]}: {record["status"]} ({elapsed:.1f}s)', flush=True)
        return checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = {**read(PROJECT / 'configs/retrieval/reranker.json'), 'device': args.device,
              'batch_size': args.batch_size, 'max_tokens': args.max_tokens}
    result = run((PROJECT / args.selection).resolve(), (PROJECT / args.output).resolve(), config, args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
