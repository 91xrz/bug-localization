"""Construct label-free, multi-fragment file documents for a frozen reranker."""
from collections import defaultdict


def file_packets(chunks):
    """Use all selected evidence once, in a content-identity-determined order.

    No generated hypotheses, retrieval scores, labels, or old ranks are exposed
    to the model. Repeated path/signature prefixes are retained as source data.
    """
    grouped = defaultdict(list)
    seen = set()
    for chunk in chunks:
        identity = chunk["chunk_id"]
        if identity in seen:
            raise ValueError("Duplicate chunk identity")
        seen.add(identity)
        text = chunk["embedding_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty code evidence")
        grouped[chunk["file_path"]].append((identity, text))
    return [{"file_path": path, "chunk_ids": [i for i, _ in sorted(items)],
             "document": "File: " + path + "\n\n" + "\n\n".join(
                 f"[Code fragment {n}]\n{text}" for n, (_, text) in enumerate(sorted(items), 1))}
            for path, items in sorted(grouped.items())]
