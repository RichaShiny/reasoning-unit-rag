"""Summarize saved failures; this does not rerun or repair retrieval."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def summarize(cases):
    if not cases:
        raise ValueError('Failure artifact is empty')
    missing_page_cases = 0
    missing_fact_count = 0
    recalls = []
    for case in cases:
        gold = {(f['title'], f['sentence_id']) for f in case['gold_facts']}
        retrieved = {(s['title'], s['sentence_id']) for s in case['retrieved_sentences']}
        missing = gold - retrieved
        saved_missing = {(f['title'], f['sentence_id']) for f in case['missing_gold_facts']}
        recall = len(gold & retrieved) / len(gold)
        if missing != saved_missing or abs(recall - case['sentence_recall']) > 1e-9:
            raise ValueError(f"Inconsistent saved metrics for {case['example_id']}")
        pages = {title for title, _ in retrieved}
        missing_page_cases += any(title not in pages for title, _ in missing)
        missing_fact_count += len(missing)
        recalls.append(recall)
    return {
        'saved_cases': len(cases),
        'question_types': dict(sorted(Counter(c['question_type'] for c in cases).items())),
        'mean_sentence_recall': mean(recalls),
        'missing_facts': missing_fact_count,
        'cases_with_missing_evidence_page': missing_page_cases,
        'scope': 'Selected historical failure cases; original source coordinates are unverified.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/failure_analysis.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = summarize(json.loads(args.input.read_text(encoding='utf-8')))
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
