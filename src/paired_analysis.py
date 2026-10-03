"""Paired question-level bootstrap from saved evaluation predictions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean


def paired_interval(differences, resamples=10000, seed=42):
    """Percentile 95% interval, resampling paired question differences."""
    if not differences or resamples < 1:
        raise ValueError('Nonempty differences and positive resamples are required')
    if any(not math.isfinite(value) for value in differences):
        raise ValueError('Differences must be finite')
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(mean(differences[rng.randrange(n)] for _ in range(n))
                   for _ in range(resamples))

    def percentile(p):
        index = (len(means) - 1) * p
        lower = math.floor(index)
        upper = math.ceil(index)
        return means[lower] + (means[upper] - means[lower]) * (index - lower)

    return {'mean_difference': mean(differences),
            'percentile_95_interval': [percentile(.025), percentile(.975)],
            'examples': n, 'resamples': resamples, 'seed': seed}


def analyze(records, baseline='original_question', candidate='rule_based',
            resamples=10000, seed=42):
    ids = [r['example_id'] for r in records]
    if not ids or any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Prediction IDs must be nonempty, present, and unique')
    groups = {'all': records}
    for question_type in sorted({r['question_type'] for r in records}):
        groups['type:' + question_type] = [r for r in records if r['question_type'] == question_type]
    result = {}
    for name, rows in groups.items():
        result[name] = {}
        for metric in ['recall', 'complete_evidence', 'page_coverage']:
            values = []
            for row in rows:
                base = row['methods'][baseline]['metrics'][metric]
                cand = row['methods'][candidate]['metrics'][metric]
                if not 0 <= base <= 1 or not 0 <= cand <= 1:
                    raise ValueError(f'Invalid {metric} for {row["example_id"]}')
                values.append(cand - base)
            result[name][metric] = paired_interval(values, resamples, seed)
    return {'baseline': baseline, 'candidate': candidate, 'groups': result,
            'interpretation': 'Candidate minus baseline; selected-prefix replication is not held-out population evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--baseline', default='original_question')
    parser.add_argument('--candidate', default='rule_based')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.predictions.read_bytes()
    records = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    report = analyze(records, args.baseline, args.candidate, args.resamples, args.seed)
    report['predictions_sha256'] = hashlib.sha256(data).hexdigest()
    # Do not replace a previously reviewed analysis by accident.
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report['groups']['all'], indent=2))


if __name__ == '__main__':
    main()
