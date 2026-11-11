"""Offline conservative filtering of the first generated response."""
import json
from .embeddings import digest
from .report_enrichment import FIELDS, validate, build_queries


def cached_with_evidence_filter(enricher, report):
    key = digest({'identity': enricher.identity, 'report': report})
    saved = json.loads((enricher.folder / (key + '.json')).read_text())
    if saved['identity'] != enricher.identity or saved['report'] != report :
        raise ValueError('Invalid first-response cache')
    raw = saved['raw']
    if raw.startswith('```json') and raw.endswith('```'):
        raw = raw[7:-3].strip()
    empty = {'text': '', 'basis': 'missing', 'evidence': []}
    try:
        if not saved['finished']:
            raise ValueError('Generation truncated')
        sections = json.loads(raw)
        if not isinstance(sections, dict) or set(sections) != set(FIELDS):
            raise ValueError('Invalid first-response structure')
    except ValueError as error:
        sections = {f: dict(empty) for f in FIELDS}
        return {**saved, 'sections': sections, 'generation_error': str(error),
                'validation_policy': 'empty_enrichment_for_invalid_first_response_v1',
                'queries': build_queries(report, sections)}
    accepted, rejected = {}, []
    for name, item in sections.items():
        try:
            validate({f: item if f == name else empty for f in FIELDS}, report)
            accepted[name] = item
        except ValueError as error:
            # Remove the entire unsupported section; do not repair quotes or regenerate.
            accepted[name] = dict(empty)
            rejected.append({'field': name, 'item': item, 'reason': str(error)})
    return {**saved, 'sections': accepted, 'rejected_sections': rejected,
            'validation_policy': 'drop_invalid_sections_from_first_response_v1',
            'queries': build_queries(report, accepted)}
