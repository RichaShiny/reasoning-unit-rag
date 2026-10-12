"""Paired retriever interactions; require identical questions and experimental cells."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paired_analysis import paired_interval
from evidence_metrics import validate_metric_records
from matrix_artifacts import load_matrix_run, validate_comparison_manifests


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
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.resamples < 1:
        p.error('--resamples must be positive')
    loaded = [load_matrix_run(path) for path in (args.left, args.right)]
    manifests = [item[0] for item in loaded]
    rows = [item[1] for item in loaded]
    validate_comparison_manifests(*manifests)
    contract = validate_metric_records(rows[0])
    report = dict(**contract, interpretation='Query benefit in right retriever minus query benefit in left retriever',
                  bootstrap=dict(resamples=args.resamples, seed=args.seed),
                  runs=[dict(predictions=str(path), examples=len(records),
                             input_sha256=manifest['input_sha256'], source_sha256=manifest['source_sha256'],
                             retriever=manifest['configuration'].get('retriever'),
                             model=manifest['configuration'].get('model'),
                             model_revision=manifest['configuration'].get('model_revision'))
                        for path, (manifest, records) in zip((args.left, args.right), loaded)],
                  effects=compare(*rows, resamples=args.resamples, seed=args.seed))
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
