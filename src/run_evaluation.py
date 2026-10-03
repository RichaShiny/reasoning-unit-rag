"""Reproducible HotpotQA sentence retrieval with per-example evidence outputs."""
import argparse
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
from statistics import mean

from evidence import indexed_sentences
from reasoning_units import generate_reasoning_units

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def sample_examples(dataset, count, seed, exclude_first=0, first=False):
    if count < 1 or exclude_first < 0 or exclude_first + count > len(dataset):
        raise ValueError('Sample must fit the available examples after exclusion')
    indices = list(range(exclude_first, len(dataset)))
    if not first:
        random.Random(seed).shuffle(indices)
    return [dataset[i] for i in indices[:count]]


def build_sentences(example):
    context = example['context']
    if len(context['title']) != len(context['sentences']):
        raise ValueError('Context title/sentence groups have different lengths')
    items = []
    for title, group in zip(context['title'], context['sentences']):
        for sentence_id, text in indexed_sentences(group):
            items.append({'title': title, 'sentence_id': sentence_id,
                          'text': f'{title}. {text}'})
    return items


def select_ranked(items, scores, top_k, word_budget=None):
    if top_k < 1 or (word_budget is not None and word_budget < 1):
        raise ValueError('Budgets must be positive')
    if len(items) != len(scores):
        raise ValueError('One score is required per candidate')
    # Python's stable sort breaks ties in original corpus order.
    ranking = sorted(range(len(items)), key=lambda i: -scores[i])
    selected, seen, words = [], set(), 0
    for i in ranking:
        item = items[i]
        coordinate = (item['title'], item['sentence_id'])
        size = len(item['text'].split())
        if coordinate in seen:
            continue
        if word_budget is not None and words + size > word_budget:
            continue
        selected.append({**item, 'score': float(scores[i])})
        seen.add(coordinate)
        words += size
        if len(selected) == top_k:
            break
    return selected


def evidence_metrics(example, selected):
    support = example['supporting_facts']
    if len(support['title']) != len(support['sent_id']):
        raise ValueError('Supporting-fact fields have different lengths')
    gold = set(zip(support['title'], support['sent_id']))
    if not gold:
        raise ValueError('Example has no supporting facts')
    retrieved = {(s['title'], s['sentence_id']) for s in selected}
    gold_pages = {title for title, _ in gold}
    pages = {title for title, _ in retrieved}
    missing = gold - retrieved
    return {
        'recall': len(gold & retrieved) / len(gold),
        'complete_evidence': int(not missing),
        'page_coverage': len(gold_pages & pages) / len(gold_pages),
        'retrieved_words': sum(len(s['text'].split()) for s in selected),
        'retrieved_sentences': len(retrieved),
        'missing_facts': [list(fact) for fact in sorted(missing)],
    }


def evaluate_example(example, encoder, top_k=5, word_budget=None):
    import numpy as np

    items = build_sentences(example)
    question = example['question']
    units = generate_reasoning_units(question)
    # The generator sees the question only; gold labels are used below for scoring.
    queries = list(dict.fromkeys([question, *units]))
    query_vectors = encoder.encode(queries, normalize_embeddings=True,
                                   show_progress_bar=False)
    if items:
        vectors = encoder.encode([s['text'] for s in items],
                                 normalize_embeddings=True, show_progress_bar=False)
        scores = query_vectors @ vectors.T
        baseline_scores = scores[0]
        rule_scores = np.max(scores, axis=0)
    else:
        baseline_scores = rule_scores = []
    methods = {}
    for name, method_scores in [('original_question', baseline_scores),
                                ('rule_based', rule_scores)]:
        selected = select_ranked(items, method_scores, top_k, word_budget)
        methods[name] = {'queries': [question] if name == 'original_question' else queries,
                         'retrieved': selected,
                         'metrics': evidence_metrics(example, selected)}
    return {'example_id': example.get('id', example.get('_id')),
            'question': question, 'question_type': example.get('type', 'unknown'),
            'methods': methods}


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups['all'].append(record)
        groups['type:' + record['question_type']].append(record)
    summary = {}
    for group, rows in sorted(groups.items()):
        summary[group] = {'examples': len(rows), 'methods': {}}
        for method in ['original_question', 'rule_based']:
            summary[group]['methods'][method] = {
                metric: mean(r['methods'][method]['metrics'][metric] for r in rows)
                for metric in ['recall', 'complete_evidence', 'page_coverage',
                               'retrieved_words', 'retrieved_sentences']}
        summary[group]['paired_recall_difference'] = mean(
            r['methods']['rule_based']['metrics']['recall'] -
            r['methods']['original_question']['metrics']['recall'] for r in rows)
    return summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Local JSON list in Hugging Face HotpotQA schema')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--dataset-revision')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--model-revision')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-first', type=int, default=0)
    parser.add_argument('--first', action='store_true', help='Take ordered prefix rather than seeded sample')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--word-budget', type=int)
    parser.add_argument('--output', type=Path, required=True, help='New run directory; never overwritten')
    args = parser.parse_args()
    if args.top_k < 1 or (args.word_budget is not None and args.word_budget < 1):
        parser.error('Retrieval budgets must be positive')
    if args.input:
        dataset = json.loads(args.input.read_text(encoding='utf-8'))
    else:
        from datasets import load_dataset
        dataset = load_dataset('hotpotqa/hotpot_qa', 'distractor', split=args.split,
                               revision=args.dataset_revision)
    examples = sample_examples(dataset, args.count, args.seed, args.exclude_first, args.first)
    ids = [e.get('id', e.get('_id')) for e in examples]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Sample IDs must be present and unique')
    args.output.mkdir(parents=True, exist_ok=False)
    packages = {}
    for package in ['numpy', 'sentence-transformers', 'datasets', 'torch']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    manifest = {'status': 'started', 'configuration': {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()},
                'example_ids': ids, 'packages': packages,
                'dataset_fingerprint': getattr(dataset, '_fingerprint', None),
                'revisions_pinned': bool(args.model_revision and (args.input or args.dataset_revision)),
                'budget_policy': 'Top-K unique sentences capped by optional title-inclusive word budget'}
    write_json(args.output / 'manifest.json', manifest)
    # Persist the exact sample, including evaluation labels, separately from generator inputs.
    write_json(args.output / 'examples.json', examples)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.model, revision=args.model_revision)
    records = []
    with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as output:
        for example in examples:
            record = evaluate_example(example, encoder, args.top_k, args.word_budget)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            records.append(record)
    write_json(args.output / 'summary.json', summarize(records))
    manifest['status'] = 'complete'
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(summarize(records), indent=2))


if __name__ == '__main__':
    main()
