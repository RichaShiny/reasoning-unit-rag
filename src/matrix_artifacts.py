"""Validate completed diagnostic artifacts before statistical comparison."""
import json
from pathlib import Path
import re
from evidence_metrics import validate_metric_records
from information_controls import CONDITIONS, PROTOCOL

COMPARISON_FIELDS = (
    'input_sha256', 'annotation_sha256', 'budget_unit', 'budget_policy',
    'fusion', 'rrf_k', 'original_question_policy', 'query_types',
    'metric_schema_version', 'evidence_level', 'primary_metric', 'evidence_interpretation',
    'source_sha256', 'fusion_source_sha256', 'metric_source_sha256', 'corpus_scope',
)
CONFIGURATION_FIELDS = ('dataset', 'top_k', 'context_budget', 'tokenizer', 'tokenizer_revision')


def load_matrix_run(path):
    path = Path(path)
    manifest = json.loads((path.parent/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Comparison requires a complete run: ' + str(path.parent))
    for field in COMPARISON_FIELDS:
        if field not in manifest:
            raise ValueError('Missing manifest field: ' + field + '; regenerate legacy runs')
    config = manifest.get('configuration')
    if not isinstance(config, dict) or any(field not in config for field in CONFIGURATION_FIELDS):
        raise ValueError('Missing comparison configuration fields')
    for field in ('top_k', 'context_budget'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError('Positive integer budget required: ' + field)
    for field in ('input_sha256', 'source_sha256', 'fusion_source_sha256', 'metric_source_sha256'):
        if not isinstance(manifest[field], str) or not re.fullmatch('[0-9a-f]{64}', manifest[field]):
            raise ValueError('Invalid manifest hash: ' + field)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    contract = validate_metric_records(records)
    if any(manifest[field] != value for field, value in contract.items()):
        raise ValueError('Manifest metric contract does not match predictions')
    ids = [record['example_id'] for record in records]
    if (type(manifest.get('processed_examples')) is not int or manifest['processed_examples'] != len(records)
            or manifest.get('example_ids') != ids):
        raise ValueError('Prediction IDs/count must exactly match the complete run manifest')
    queries = manifest['query_types']
    if (not isinstance(queries, list) or not queries or any(not isinstance(q, str) or not q for q in queries)
            or len(set(queries)) != len(queries) or 'original' not in queries):
        raise ValueError('Manifest requires unique query types including original')
    if set(queries) & set(CONDITIONS[1:]):
        if set(queries) != set(CONDITIONS) or manifest.get('information_control_protocol') != PROTOCOL:
            raise ValueError('Information controls require the complete declared protocol')
        source = manifest.get('information_control_source_sha256')
        if not isinstance(source, str) or not re.fullmatch('[0-9a-f]{64}', source):
            raise ValueError('Missing information-control source hash')
    keys = {unit+'/'+query for unit in ('passage', 'sentence', 'window') for query in queries}
    for record in records:
        if set(record['cells']) != keys:
            raise ValueError('Prediction cells do not match manifested query conditions')
        for cell in record['cells'].values():
            for field in ('fusion', 'rrf_k', 'original_question_policy'):
                if field not in cell or cell[field] != manifest[field]:
                    raise ValueError('Cell does not match manifest: ' + field)
    return manifest, records


def validate_comparison_manifests(left, right):
    if left.get('information_control_protocol') is not None or right.get('information_control_protocol') is not None:
        for field in ('information_control_protocol', 'information_control_source_sha256'):
            if not left.get(field) or left[field] != right.get(field):
                raise ValueError('Mismatch in ' + field)
    for field in COMPARISON_FIELDS:
        if left[field] != right[field]:
            raise ValueError('Mismatch in ' + field)
    for field in CONFIGURATION_FIELDS:
        if left['configuration'][field] != right['configuration'][field]:
            raise ValueError('Mismatch in ' + field)
