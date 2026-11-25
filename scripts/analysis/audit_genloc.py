"""Independently verify GenLoc inputs, cached retrieval, tool traces and scores.

This audit uses CPU parsing and existing vector caches only. It does not load
model weights, call an inference service, or fetch missing source snapshots.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
for dependency in ("rrl-deps", "genloc-deps"):
    if (PROJECT / ".runtime" / dependency).is_dir():
        sys.path.insert(0, str(PROJECT / ".runtime" / dependency))

from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import collect_files, prepare_report, ranking_metrics
from bug_localization.embeddings import digest
from bug_localization.genloc import exploration_messages, first_query_chunk
from bug_localization.genloc_tools import AuthorCode, Toolbox, build_chunks, build_file_data, normalize_ranking
from bug_localization.snapshot import get_snapshot, sha256_file
from bug_localization.vector import select_reports


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_retrieval(project, config, identity, case, query, chunks):
    """Recompute cosine scores from checksummed, read-only cached vectors."""
    import numpy as np

    retrieval = read(case / "retrieval.json")
    text, query_audit = first_query_chunk(query)
    require(retrieval["query_text"] == text and retrieval["query_audit"] == query_audit,
            "Retrieval query differs from the original report's author-selected first chunk")
    fingerprint = digest(identity)
    cache = project / config["embedding"].get("cache_dir", "cache/genloc_embeddings") / fingerprint
    require(read(cache / "identity.json") == identity, "Embedding cache identity differs")
    database = sqlite3.connect((cache / "vectors.sqlite3").as_uri() + "?mode=ro", uri=True)
    dimension = identity["dimension"]

    def vector(text, token_audit):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        require(token_audit["text_sha256"] == key, "Embedding input text checksum differs")
        require(token_audit["original_tokens"] == token_audit["used_tokens"]
                and not token_audit["truncated"] and not token_audit["over_budget"]
                and token_audit["used_tokens"] <= token_audit["max_tokens"],
                "Embedding input was truncated or over its declared token budget")
        row = database.execute("SELECT data, checksum, tokens FROM vectors WHERE key=?", (key,)).fetchone()
        require(row is not None, "Missing cached vector; audit never reruns a model")
        raw, checksum, tokens = row
        require(hashlib.sha256(raw).hexdigest() == checksum and tokens == token_audit["used_tokens"],
                "Embedding cache checksum or token count differs")
        value = np.frombuffer(raw, dtype="<f4").copy()
        require(value.shape == (dimension,) and np.isfinite(value).all()
                and abs(float(np.linalg.norm(value)) - 1) < 1e-3, "Invalid normalized cached vector")
        return value

    try:
        docs_audit = retrieval["embedding_documents_audit"]
        require(len(docs_audit) == len(chunks), "Embedding audit lost source chunks")
        vectors = np.stack([vector(chunk["embedding_text"], audit)
                            for chunk, audit in zip(chunks, docs_audit)])
        query_vector = vector(text, retrieval["embedding_query_audit"])
        scores = vectors @ query_vector
    finally:
        database.close()
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))[:config["retrieval_chunk_limit"]]
    expected_files = list(dict.fromkeys(chunks[i]["file_path"] for i in order))[:config["candidate_limit"]]
    require(retrieval["candidates"] == expected_files and retrieval["initial_ranking"] == expected_files,
            "Initial file candidates differ from all-cache-vector cosine ranking")
    require(len(retrieval["top_chunks"]) == len(order), "Top chunk count differs")
    for saved, index in zip(retrieval["top_chunks"], order):
        require(saved["chunk_id"] == chunks[index]["chunk_id"]
                and saved["file_path"] == chunks[index]["file_path"] and saved["chunk_index"] == index
                and abs(saved["score"] - float(scores[index])) <= 1e-6,
                "Saved top-chunk ranking or similarity differs")
    require(retrieval["chunks"] == len(chunks), "Retrieval chunk inventory differs")
    return retrieval


def audit_calls(case, query, file_data, candidates, author, config, generation_identity, record, model_alias):
    """Rebuild every model input from source-only tool results and prior replies."""
    calls = sorted(case.glob("call_*.json"))
    require(len(calls) == record["model_calls"] <= config["max_calls"], "Model call count/budget differs")
    expected_paths = [case / f"call_{i:03d}.json" for i in range(1, len(calls) + 1)]
    require(calls == expected_paths, "Model call sequence has gaps")
    messages = exploration_messages(query, author)
    toolbox = Toolbox(file_data, candidates, author=author)
    events = read(case / "tool_calls.json") if (case / "tool_calls.json").exists() else []
    event_index = 0
    final_ranking = None
    usage = Counter()
    allowed = {tool["function"]["name"] for tool in author.tools}
    request_fields = {"messages", "tools", "tool_choice", "response_schema", "stage", "temperature",
                      "max_new_tokens", "identity"}
    for iteration, path in enumerate(calls, 1):
        saved = read(path)
        request = saved["request"]
        choice = "required" if iteration == 1 else ("none" if iteration == config["max_iterations"] else "auto")
        schema = author.response_format["json_schema"]["schema"] if choice == "none" else None
        require(set(request) == request_fields, "Unexpected fields sent to inference")
        require(saved["request_sha256"] == digest(request), "Model request hash differs")
        require(request["messages"] == messages, "Model messages are not the original report plus replayed code evidence")
        require(request["tools"] == author.tools and request["tool_choice"] == choice
                and request["response_schema"] == schema and request["stage"] == f"explore_{iteration}"
                and request["identity"] == digest(generation_identity)
                and request["temperature"] == config["temperature"]
                and request["max_new_tokens"] == config["max_new_tokens"], "Model request policy differs")
        seed = (config["seed"] + int(saved["request_sha256"][:8], 16)) % (2 ** 32)
        require(saved["seed"] == seed, "Per-request sampling seed differs")
        payload = saved["payload"]
        expected_payload = {
            "model": model_alias, "messages": messages, "tools": author.tools,
            "tool_choice": choice, "parallel_tool_calls": True,
            "max_tokens": config["max_new_tokens"], "temperature": config["temperature"],
            "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0,
            "samplers": ["temperature"], "seed": seed, "stream": False,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            expected_payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "genloc_ranked_files", "strict": True, "schema": schema}}
        require(payload == expected_payload,
                "Actual inference payload/model/schema/decoding differs from the frozen policy")
        require(isinstance(saved.get("elapsed_seconds"), (int, float))
                and math.isfinite(saved["elapsed_seconds"]) and saved["elapsed_seconds"] >= 0,
                "Invalid recorded inference time")
        usage["calls"] += 1
        usage["input_tokens"] += saved.get("input_tokens", 0)
        usage["output_tokens"] += saved.get("output_tokens", 0)
        usage["recorded_call_elapsed_seconds"] += saved["elapsed_seconds"]
        if "server_response" in saved:
            server_usage = saved["server_response"].get("usage", {})
            require(saved["usage"] == server_usage
                    and saved["output_tokens"] == server_usage.get("completion_tokens", 0)
                    and saved["server_timings"] == saved["server_response"].get("timings", {}),
                    "Recorded token usage or server timing differs from original response")
            if {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(server_usage):
                require(server_usage["total_tokens"] == server_usage["prompt_tokens"] + server_usage["completion_tokens"],
                        "Server total token usage is inconsistent")
        if saved["status"] == "failed":
            require(record["status"] == "failed" and iteration == len(calls), "Failed inference silently retried or counted successful")
            break
        require(saved["status"] == "ok", "Unfinished inference record")
        require(saved["input_tokens"] + config["max_new_tokens"] <= config["max_context_tokens"],
                "Successful inference exceeded context budget")
        response = saved["response"]
        reply = saved["server_response"]
        require(len(reply["choices"]) == 1 and reply["choices"][0]["message"] == response
                and reply["choices"][0]["finish_reason"] in ("stop", "tool_calls")
                and not reply.get("truncated"), "Saved model reply differs or was truncated")
        require(reply.get("usage", {}).get("prompt_tokens", saved["input_tokens"]) == saved["input_tokens"],
                "Server and native tokenizer input counts differ")
        tool_calls = response.get("tool_calls") or []
        if tool_calls:
            require(choice != "none", "Tool call in final-only round")
            messages.append(response)
            for tool_call in tool_calls:
                name = tool_call["function"].get("name")
                raw_args = tool_call["function"].get("arguments")
                args = None
                try:
                    if name not in allowed:
                        raise ValueError("Unknown exploration tool")
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    result = toolbox.execute(name, args)
                except (ValueError, TypeError, KeyError) as error:
                    result = {"error": "Invalid tool request", "detail": str(error)}
                require(event_index < len(events), "Tool call missing its execution event")
                event = events[event_index]
                expected = {"iteration": iteration, "call_id": tool_call["id"], "name": name,
                            "raw_arguments": raw_args, "arguments": args, "response": result,
                            "success": not (isinstance(result, dict) and "error" in result)}
                require({key: event.get(key) for key in expected} == expected, "Tool response fails source-only replay")
                require(event["elapsed_seconds"] >= 0, "Negative tool execution time")
                event_index += 1
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "name": name,
                                 "content": json.dumps(result, ensure_ascii=False)})
        else:
            require(choice != "required", "First round omitted its required tool call")
            require(iteration == len(calls), "Model called again after a final answer")
            require(read(case / "final_raw.json") == {"content": response["content"], "iteration": iteration},
                    "Final answer differs from original model completion")
            try:
                final_ranking, normalization = normalize_ranking(response["content"], file_data, author=author)
            except (ValueError, TypeError, KeyError):
                require(record["status"] == "failed", "Malformed final output counted successful")
            else:
                require(read(case / "normalization.json") == normalization, "Final path recovery audit differs")
                messages.append(response)
    require(event_index == len(events), "Tool execution log contains unrequested extra events")
    if (case / "conversation.json").exists():
        require(read(case / "conversation.json") == messages, "Saved conversation differs from audited trace")
    if record["status"] == "ok":
        require(final_ranking is not None and event_index > 0, "Successful case lacks final output or actual tools")
    return {"ranking": final_ranking, "usage": usage, "events": events}


def audit(project, folder):
    project, folder = Path(project).resolve(), Path(folder).resolve()
    config_file = read(folder / "effective_config.json")
    config, base = config_file["genloc"], config_file["base"]
    provenance = read(folder / "provenance.json")
    identity = provenance["identity"]
    require(provenance["identity_sha256"] == digest(identity), "Run identity checksum differs")
    require(identity["config"] == config and identity["base"] == base, "Effective configuration differs from provenance")
    for name, checksum in identity["implementation"].items():
        require(sha256_file(project / "src/bug_localization" / name) == checksum, "Implementation changed since run: " + name)
    author = AuthorCode(project / config["author_checkout"])
    require(author.provenance == identity["author"], "Author checkout identity differs")
    server_manifest_path = project / config["server_manifest"]
    server_manifest = read(server_manifest_path)
    require(sha256_file(server_manifest_path) == identity["generation"]["manifest_sha256"]
            and server_manifest["gguf_sha256"] == identity["generation"]["model_sha256"]
            and server_manifest["source_files_sha256"] == identity["generation"]["source_files_sha256"],
            "Pinned generation server/model manifest differs")
    selected, selection = select_reports(project, config, base, max_samples=50)
    selection["selection_rule"] = read(project / config["selection_manifest"])["selection_rule"]
    require(read(folder / "selection.json") == selection
            and selection["selection_sha256"] == identity["selection_sha256"], "Frozen selection differs")
    records, summary = read(folder / "samples.json"), read(folder / "summary.json")
    require(len(records) <= len(selected), "Extra samples appeared in run")
    usage, counts = Counter(), Counter()
    initial_records = []
    protocol_observations = []
    for position, (sample, record) in enumerate(zip(selected, records), 1):
        row = sample["report"]
        case = folder / "cases" / f"{position:04d}"
        expected_entry = {"position": position, "dataset_row": sample["dataset_row"],
                          "repository": row["repo_name"], "issue_id": row["issue_id"],
                          "before_fix_sha": row["before_fix_sha"]}
        require({key: record.get(key) for key in expected_entry} == expected_entry
                and read(case / "record.json") == record, "Sample identity or per-case record differs")
        require(isinstance(record.get("elapsed_seconds"), (int, float))
                and math.isfinite(record["elapsed_seconds"]) and record["elapsed_seconds"] >= 0,
                "Invalid recorded per-case time")
        usage["recorded_case_elapsed_seconds"] += record["elapsed_seconds"]
        _, gold, query = prepare_report(row, {**base, "repository": row["repo_name"], "issue_id": row["issue_id"]})
        if not (case / "inputs.json").exists():
            require(record["status"] == "failed" and (case / "error.json").exists()
                    and record["model_calls"] == 0, "Missing inputs in a non-preprocessing failure")
            continue
        require((project / "repos" / row["repo_name"] / row["before_fix_sha"] / "snapshot.json").exists(),
                "Missing historical snapshot; audit will not download one")
        source, manifest = get_snapshot(project, row["repo_name"], row["before_fix_sha"])
        documents, skipped = collect_files(source, manifest, base)
        expected_inputs = {"query": query, "source_commit": row["before_fix_sha"],
                           "candidate_files": [{"file_path": d["file_path"], "sha256": d["sha256"]} for d in documents]}
        require(read(case / "inputs.json") == expected_inputs, "Input report or historical candidate inventory differs")
        pool = {d["file_path"] for d in documents}
        file_data = build_file_data(documents, author=author)
        if (case / "file_data.json").exists():
            require(read(case / "file_data.json") == file_data, "Saved method evidence differs from historical source")
        chunks = build_chunks(file_data, config["chunk_tokens"])
        if (case / "chunks.json").exists():
            require(read(case / "chunks.json") == chunks, "Saved embedding chunks differ from author construction")
        candidates = []
        if (case / "retrieval.json").exists():
            retrieval = audit_retrieval(project, config, identity["embedding"], case, query, chunks)
            candidates = retrieval["candidates"]
            initial_metrics = ranking_metrics(candidates, gold, 10)
            require(record.get("initial_metrics") == initial_metrics, "Retrieval metrics missing/different, including failed agents")
            initial_records.append(initial_metrics)
            counts["retrievals"] += 1
        if record["model_calls"]:
            trace = audit_calls(case, query, file_data, candidates, author, config,
                                identity["generation"], record, server_manifest["model_alias"])
            usage.update(trace["usage"])
            counts["tool_calls"] += len(trace["events"])
            counts["tool_errors"] += sum(not event["success"] for event in trace["events"])
            counts["method_body_calls"] += sum(event["name"] == "get_method_body" for event in trace["events"])
        else:
            trace = {"ranking": None}
        if record["status"] == "failed":
            require((case / "error.json").exists(), "Failed sample lacks its failure record")
            continue
        result = read(case / "result.json")
        require(record["result_file"] == (case / "result.json").relative_to(folder).as_posix()
                and sha256_file(case / "result.json") == record["result_sha256"], "Result checksum or path differs")
        require(result["ranking"] == trace["ranking"] and len(result["ranking"]) <= 10
                and len(set(result["ranking"])) == len(result["ranking"])
                and set(result["ranking"]).issubset(pool), "Final ranking differs or violates the common file pool")
        require(all(result[key] == value for key, value in expected_inputs.items())
                and result["gold_files"] == gold and result["initial_ranking"] == candidates,
                "Result input or evaluation labels differ")
        metrics = ranking_metrics(result["ranking"], gold, 10)
        require(result["metrics"] == record["metrics"] == metrics
                and result["initial_metrics"] == record["initial_metrics"], "Per-case metric recomputation differs")
        require(result["normalization"] == read(case / "normalization.json")
                and result["iterations"] == record["model_calls"]
                and result["tool_calls"] == len(trace["events"])
                and result["tool_errors"] == sum(not event["success"] for event in trace["events"])
                and result["method_body_calls"] == sum(event["name"] == "get_method_body" for event in trace["events"]),
                "Result normalization or exploration counts differ from trace")
        require(result["ranking_outside_initial_candidates"] == [p for p in result["ranking"] if p not in candidates]
                and result["gold_outside_initial_found"] == sorted(set(gold) & (set(result["ranking"]) - set(candidates))),
                "Exploration beyond the initial shortlist differs")
        require(result["missing_gold"] == record["missing_gold"] == sorted(set(gold) - pool)
                and result["skipped_files"] == skipped, "Gold coverage/source-filter audit differs")
        require(result["initial_recall_at_50"] == len(set(gold) & set(candidates)) / len(set(gold)),
                "Initial Recall@50 differs")
        protocol_observations.append({
            "position": position, "repository": row["repo_name"], "issue_id": row["issue_id"],
            "raw_ranking_entries": result["normalization"]["raw_count"],
            "final_unique_ranking_entries": len(result["ranking"]),
            "raw_list_has_ten_entries": result["normalization"]["raw_count"] == 10,
            "final_list_has_ten_entries": len(result["ranking"]) == 10,
            "successful_method_body_calls": sum(event["name"] == "get_method_body" and event["success"]
                                                 for event in trace["events"]),
            "tool_errors": [{"iteration": event["iteration"], "tool": event["name"],
                             "arguments": event["arguments"], "response": event["response"]}
                            for event in trace["events"] if not event["success"]],
        })
    recalculated = aggregate_results(records, 10, len(selected))
    for field in ("selected_samples", "attempted_samples", "successful_samples", "failed_samples",
                  "pending_samples", "metrics", "metrics_successful_only", "coverage"):
        require(summary[field] == recalculated[field], "Run aggregate differs: " + field)
    metric_fields = {"MAP@10": "average_precision_at_k", "MRR@10": "reciprocal_rank_at_k",
                     "Top-1": "top_1", "Top-5": "top_5", "Top-10": "top_10"}
    initial_summary = None if recalculated["pending_samples"] else {
        name: sum(metrics[field] for metrics in initial_records) / len(selected)
        for name, field in metric_fields.items()}
    require(summary["initial_retrieval_metrics"] == initial_summary, "Initial retrieval aggregate differs")
    result = {"status": "passed", "complete": not recalculated["pending_samples"],
              "samples": len(records), "selected_samples": len(selected),
              "successful": recalculated["successful_samples"], "failed": recalculated["failed_samples"],
              "pending": recalculated["pending_samples"], "metrics": recalculated["metrics"],
              "initial_retrieval_metrics": initial_summary, "usage": dict(usage), "checks": dict(counts),
              "selection_sha256": selection["selection_sha256"],
              "identity_sha256": provenance["identity_sha256"],
              "protocol_observations": protocol_observations,
              "protocol_note": "Successful means valid nonempty output under this local adapter; it does not imply that the model returned ten files or inspected method bodies.",
              "verification": ["original body-only reports and frozen dataset identities",
                               "historical snapshot hashes and common file inventories",
                               "author parsing and exact source-derived chunks",
                               "all cached-vector checksums and CPU cosine ranking",
                               "source-only replay of tool results and model message histories",
                               "corrected unique final paths and all per-case/aggregate metrics",
                               "failed samples retained in the primary denominator"],
              "audit_inference_calls": 0}
    write_json(folder / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(PROJECT, (PROJECT / args.run).resolve()), ensure_ascii=False, indent=2))
