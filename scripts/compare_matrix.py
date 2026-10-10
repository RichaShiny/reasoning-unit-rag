"""Paired retriever interactions; require identical questions and experimental cells."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paired_analysis import paired_interval
from evidence_metrics import validate_metric_records


def compare(left, right, resamples=10000, seed=42):
    contract = validate_metric_records(left)
    if validate_metric_records(right) != contract:
        raise ValueError('Retriever comparisons require identical metric contracts')
    def keyed(rows):
        result = {r['example_id']: r for r in rows}
        if not rows or len(result) != len(rows):
            raise ValueError('Require nonempty, unique IDs')
        return result
    a, b = keyed(left), keyed(right)
    if a.keys() != b.keys():
        raise ValueError('Retriever comparisons require the exact same question set')
    keys = set(left[0]['cells'])
    if any(set(r['cells']) != keys for r in left+right):
        raise ValueError('Retriever comparisons require identical cells')
    # Query/fusion changes are interventions, not retriever-only comparisons.
    fields = ('queries', 'fusion', 'rrf_k', 'original_question_policy')
    for identifier in a:
        if a[identifier]['evidence_level'] != b[identifier]['evidence_level']:
            raise ValueError('Retriever comparisons require identical evidence levels')
        for key in keys:
            for field in fields:
                if field not in a[identifier]['cells'][key] or field not in b[identifier]['cells'][key]:
                    raise ValueError('Missing comparison provenance: ' + field + '; regenerate legacy runs')
                if a[identifier]['cells'][key][field] != b[identifier]['cells'][key][field]:
                    raise ValueError('Mismatch in ' + field + ' for ' + identifier + '/' + key)
    result = {}
    for key in sorted(keys):
        unit, query = key.split('/')
        if query == 'original':
            continue
        base = unit+'/original'
        m = contract['primary_metric']
        values = [(b[i]['cells'][key]['metrics'][m]-b[i]['cells'][base]['metrics'][m]) -
                  (a[i]['cells'][key]['metrics'][m]-a[i]['cells'][base]['metrics'][m]) for i in sorted(a)]
        result[key] = dict(metric=m, **paired_interval(values, resamples, seed))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--resamples', type=int, default=10000)
    args = p.parse_args()
    manifests = [json.loads((path.parent/'manifest.json').read_text()) for path in (args.left,args.right)]
    for field in ('input_sha256','annotation_sha256','budget_unit','budget_policy',
                  'fusion','rrf_k','original_question_policy','query_types',
                  'metric_schema_version','evidence_level','primary_metric','evidence_interpretation'):
        if manifests[0].get(field) != manifests[1].get(field):
            raise ValueError('Mismatch in '+field)
    for field in ('top_k','context_budget','tokenizer','tokenizer_revision'):
        if manifests[0]['configuration'][field] != manifests[1]['configuration'][field]:
            raise ValueError('Mismatch in '+field)
    rows = [[json.loads(l) for l in path.read_text().splitlines() if l.strip()] for path in (args.left,args.right)]
    contract = validate_metric_records(rows[0])
    for manifest in manifests:
        if any(manifest.get(field) != value for field, value in contract.items()):
            raise ValueError('Manifest metric contract does not match predictions')
    report = dict(**contract, interpretation='Query benefit in right retriever minus query benefit in left retriever',
                  effects=compare(*rows, resamples=args.resamples))
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
