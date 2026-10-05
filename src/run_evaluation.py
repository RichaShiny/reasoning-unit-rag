"""Reproducible HotpotQA sentence retrieval with per-example evidence outputs."""
import argparse
import hashlib
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
from statistics import mean

from evidence import indexed_sentences
from reasoning_units import generate_reasoning_units
from run_state import RunState, write_json

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def sample_examples(dataset, count, seed, exclude_first=0, first=False, exclude_ids=None):
    if count < 1 or exclude_first < 0 or exclude_first + count > len(dataset):
        raise ValueError('Sample must fit the available examples after exclusion')
    indices = list(range(exclude_first, len(dataset)))
    if exclude_ids:
        indices = [i for i in indices
                   if dataset[i].get('id', dataset[i].get('_id')) not in exclude_ids]
    if count > len(indices):
        raise ValueError('Sample does not fit after excluding prior example IDs')
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


def evaluate_example(example, encoder, top_k=5, word_budget=None, generator=None):
    import numpy as np

    items = build_sentences(example)
    question = example['question']
    rule_queries = list(dict.fromkeys([question, *generate_reasoning_units(question)]))
    method_queries = {'original_question': [question], 'rule_based': rule_queries}
    generation = None
    if generator is not None:
        # Neither gold labels, answer, nor candidate titles cross this boundary.
        generation = generator.generate(question)
        method_queries['automated'] = list(dict.fromkeys([question, *generation['queries']]))
    queries = list(dict.fromkeys(q for values in method_queries.values() for q in values))
    query_vectors = encoder.encode(queries, normalize_embeddings=True,
                                   show_progress_bar=False)
    if items:
        vectors = encoder.encode([s['text'] for s in items],
                                 normalize_embeddings=True, show_progress_bar=False)
        scores = query_vectors @ vectors.T
    methods = {}
    for name, values in method_queries.items():
        method_scores = np.max(scores[[queries.index(q) for q in values]], axis=0) if items else []
        selected = select_ranked(items, method_scores, top_k, word_budget)
        methods[name] = {'queries': values, 'retrieved': selected,
                         'metrics': evidence_metrics(example, selected)}
    record = {'example_id': example.get('id', example.get('_id')),
              'question': question, 'question_type': example.get('type', 'unknown'),
              'methods': methods}
    if generation is not None:
        record['generation'] = generation
    return record


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups['all'].append(record)
        groups['type:' + record['question_type']].append(record)
    summary = {}
    for group, rows in sorted(groups.items()):
        summary[group] = {'examples': len(rows), 'methods': {}}
        for method in rows[0]['methods']:
            summary[group]['methods'][method] = {
                metric: mean(r['methods'][method]['metrics'][metric] for r in rows)
                for metric in ['recall', 'complete_evidence', 'page_coverage',
                               'retrieved_words', 'retrieved_sentences']}
        summary[group]['paired_recall_difference'] = mean(
            r['methods']['rule_based']['metrics']['recall'] -
            r['methods']['original_question']['metrics']['recall'] for r in rows)
    if records and 'automated' in records[0]['methods']:
        for group, rows in groups.items():
            summary[group]['automated_paired_recall_difference'] = mean(
                r['methods']['automated']['metrics']['recall'] -
                r['methods']['original_question']['metrics']['recall'] for r in rows)
        generated = [r['generation'] for r in records]
        attempted = [g for g in generated if g['api_call_attempted']]
        provider_attempts = [g for g in generated if g.get('provider_call_attempted', g['api_call_attempted'])]
        summary['generation'] = {
            'examples': len(generated),
            'fallbacks': sum(g['fallback'] for g in generated),
            'fallback_rate': mean(g['fallback'] for g in generated),
            'cache_hits': sum(g['cache_hit'] for g in generated),
            'provider_call_attempts': len(provider_attempts),
            'provider_latency_seconds': sum(g['response']['latency_seconds'] for g in provider_attempts),
            'provider_input_tokens': sum((g['response'].get('usage') or {}).get('input_tokens', 0) for g in provider_attempts),
            'provider_output_tokens': sum((g['response'].get('usage') or {}).get('output_tokens', 0) for g in provider_attempts),
            'api_call_attempts': len(attempted),
            'api_attempt_latency_seconds': sum(g['response']['latency_seconds'] for g in attempted),
            'reported_input_tokens_on_new_calls': sum((g['response'].get('usage') or {}).get('input_tokens', 0) for g in attempted),
            'reported_output_tokens_on_new_calls': sum((g['response'].get('usage') or {}).get('output_tokens', 0) for g in attempted),
            'new_calls_without_usage': sum(g['response'].get('usage') is None for g in attempted),
        }
    return summary


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
    parser.add_argument('--exclude-ids', type=Path, help='JSON ID list or a prior run manifest')
    parser.add_argument('--first', action='store_true', help='Take ordered prefix rather than seeded sample')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--word-budget', type=int)
    parser.add_argument('--generator-provider', choices=['openai', 'local'], default='openai')
    parser.add_argument('--generator-revision')
    parser.add_argument('--generator-device', default='cpu')
    parser.add_argument('--generator-input-limit', type=int, default=512)
    parser.add_argument('--generator-beams', type=int, default=1)
    parser.add_argument('--generator-model', help='Explicit generator model ID; enables automated retrieval')
    parser.add_argument('--generator-cache', type=Path)
    parser.add_argument('--generator-cache-only', action='store_true')
    parser.add_argument('--generator-max-output-tokens', type=int, default=512)
    parser.add_argument('--generator-reasoning-effort')
    parser.add_argument('--output', type=Path, required=True, help='New run directory; never overwritten')
    args = parser.parse_args()
    if args.top_k < 1 or (args.word_budget is not None and args.word_budget < 1):
        parser.error('Retrieval budgets must be positive')
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    with RunState(args.output, configuration) as state:
        source_dir = Path(__file__).resolve().parent
        state.info['source_sha256'] = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                                   for name in ['run_evaluation.py', 'run_state.py', 'evidence.py',
                                                'reasoning_units.py', 'query_generator.py', 'local_generator.py']}
        state.save()
        execute(args, parser, state)


def execute(args, parser, state):
    generator = None
    state.phase('loading_generator')
    if args.generator_model:
        if not args.generator_cache:
            parser.error('--generator-cache is required with --generator-model')
        from query_generator import QuestionGenerator, make_openai_provider
        provider_configuration = {}
        provider_kind = 'openai_responses'
        if args.generator_provider == 'local':
            if args.generator_reasoning_effort:
                parser.error('Local seq2seq generation does not accept reasoning effort')
            if not args.generator_revision:
                parser.error('--generator-revision is required for local generation/replay')
            if args.generator_input_limit < 1 or args.generator_beams < 1:
                parser.error('Local input limit and beam count must be positive')
            from local_generator import make_local_provider
            provider_configuration = {'revision': args.generator_revision,
                                      'device': args.generator_device,
                                      'max_input_tokens': args.generator_input_limit,
                                      'num_beams': args.generator_beams, 'do_sample': False}
            provider_kind = 'local_seq2seq'
            provider = None if args.generator_cache_only else make_local_provider(
                args.generator_model, **{k: v for k, v in provider_configuration.items() if k != 'do_sample'})
        else:
            if args.generator_revision:
                parser.error('Use a snapshot model ID for OpenAI; --generator-revision is local-only')
            provider = None if args.generator_cache_only else make_openai_provider()
        generator = QuestionGenerator(args.generator_model, args.generator_cache, provider,
                                      args.generator_cache_only, args.generator_max_output_tokens,
                                      args.generator_reasoning_effort, provider_kind, provider_configuration)
    elif args.generator_cache or args.generator_cache_only or args.generator_reasoning_effort or args.generator_revision or args.generator_provider != 'openai':
        parser.error('Generator options require --generator-model')
    state.phase('loading_dataset')
    if args.input:
        dataset = json.loads(args.input.read_text(encoding='utf-8'))
    else:
        from datasets import load_dataset
        dataset = load_dataset('hotpotqa/hotpot_qa', 'distractor', split=args.split,
                               revision=args.dataset_revision)
    excluded_ids = []
    if args.exclude_ids:
        excluded = json.loads(args.exclude_ids.read_text(encoding='utf-8'))
        excluded_ids = excluded.get('example_ids') if isinstance(excluded, dict) else excluded
        if not isinstance(excluded_ids, list) or any(not isinstance(i, str) for i in excluded_ids):
            raise ValueError('Exclusion input must contain a list of string example IDs')
    examples = sample_examples(dataset, args.count, args.seed, args.exclude_first, args.first,
                               set(excluded_ids))
    ids = [e.get('id', e.get('_id')) for e in examples]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Sample IDs must be present and unique')
    packages = {}
    for package in ['numpy', 'sentence-transformers', 'datasets', 'torch', 'openai', 'transformers', 'tokenizers', 'sentencepiece']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    manifest = state.info
    manifest.update({
                'example_ids': ids, 'excluded_example_ids': sorted(set(excluded_ids)), 'packages': packages,
                'dataset_fingerprint': getattr(dataset, '_fingerprint', None),
                'revisions_pinned': bool(args.model_revision and (args.input or args.dataset_revision)),
                'budget_policy': 'Top-K unique sentences capped by optional title-inclusive word budget'})
    if generator is not None:
        manifest['generator'] = generator.configuration()
    write_json(args.output / 'manifest.json', manifest)
    # Persist the exact sample, including evaluation labels, separately from generator inputs.
    write_json(args.output / 'examples.json', examples)
    state.phase('loading_embedding_model')
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.model, revision=args.model_revision)
    manifest['runtime']['embedding_device'] = str(getattr(encoder, 'device', 'unknown'))
    state.phase('evaluating')
    records = []
    with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as output:
        for example in examples:
            record = evaluate_example(example, encoder, args.top_k, args.word_budget, generator)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            records.append(record)
            manifest['processed_examples'] = len(records)
            state.save()
            if len(records) % 10 == 0 or len(records) == len(examples):
                print(f'[evaluating] {len(records)}/{len(examples)}', flush=True)
    write_json(args.output / 'summary.json', summarize(records))
    print(json.dumps(summarize(records), indent=2))


if __name__ == '__main__':
    main()
