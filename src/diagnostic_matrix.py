"""Factorial retrieval-unit/query diagnostics with global K and context caps."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import string
from evidence import indexed_sentences, sentence_windows
from paired_analysis import paired_interval
from reasoning_units import generate_reasoning_units
from run_state import RunState, write_json


def normalize(row, dataset):
    if dataset in ('hotpotqa', '2wiki'):
        context = row['context']
        pairs = zip(context['title'], context['sentences']) if isinstance(context, dict) else context
        docs = [{'id': t, 'title': t, 'sentences': s} for t, s in pairs]
        facts = row['supporting_facts']
        gold = list(zip(facts['title'], facts['sent_id'])) if isinstance(facts, dict) else facts
        level, oracle = 'sentence', None
    else:
        docs = [{'id': str(p['idx']), 'title': p['title'], 'sentences':
                 re.split(r'(?<=[.!?])\s+', p['paragraph_text'])} for p in row['paragraphs']]
        gold = [[str(p['idx']), None] for p in row['paragraphs'] if p['is_supporting']]
        level = 'paragraph'
        oracle = []
        previous = {}
        for step in row['question_decomposition']:
            def resolve(match):
                key = int(match.group(1))
                if key not in previous:
                    raise ValueError('Unresolved MuSiQue dependency')
                return previous[key]
            oracle.append(re.sub(r'#(\d+)', resolve, step['question']))
            previous[step['id']] = step['answer']
    identifier = row.get('id', row.get('_id'))
    if not isinstance(identifier, str) or not identifier or not gold:
        raise ValueError('Require string ID and nonempty gold evidence')
    if len({d['id'] for d in docs}) != len(docs):
        raise ValueError('Document IDs must be unique within a question')
    gold = {tuple(f) for f in gold}
    available = {(d['id'], sid) for d in docs for sid, _ in indexed_sentences(d['sentences'])}
    if level == 'sentence' and not gold <= available:
        raise ValueError('Gold sentence absent or empty: ' + identifier)
    return dict(id=identifier, question=row['question'], answer=row['answer'],
                aliases=row.get('answer_aliases', []), documents=docs, gold=gold,
                evidence_level=level, oracle=oracle)


def candidates(docs, unit):
    result = []
    for doc in docs:
        indexed = indexed_sentences(doc['sentences'])
        if unit == 'passage':
            groups = [(tuple(i for i, _ in indexed), ' '.join(t for _, t in indexed))] if indexed else []
        elif unit == 'sentence':
            groups = [((i,), t) for i, t in indexed]
        elif unit == 'window':
            groups = [(tuple(w['sentence_ids']), w['text']) for w in sentence_windows(doc['sentences'])]
        else:
            raise ValueError('Unknown unit')
        seen = set()
        for ids, text in groups:
            if ids in seen:
                continue
            seen.add(ids)
            result.append(dict(document_id=doc['id'], sentence_ids=list(ids), text=f"{doc['title']}. {text}"))
    return result


class BM25:
    def score(self, queries, texts):
        tokenize = lambda t: re.findall(r'\w+', t.lower())
        counts = [Counter(tokenize(t)) for t in texts]
        lengths = [sum(c.values()) for c in counts]
        average = mean(lengths) if lengths else 1
        df = Counter(t for c in counts for t in c)
        return [[sum(math.log(1 + (len(counts)-df[t]+.5)/(df[t]+.5)) * c[t]*2.2 /
                     (c[t]+1.2*(.25+.75*length/(average or 1))) for t in set(tokenize(q)))
                 for c, length in zip(counts, lengths)] for q in queries]


class Dense:
    def __init__(self, model, revision, contriever=False):
        self.contriever = contriever
        if contriever:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
            self.model = AutoModel.from_pretrained(model, revision=revision).eval()
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model, revision=revision)

    def encode(self, texts, query=False):
        if not self.contriever:
            method = self.model.encode_query if query else self.model.encode_document
            return method(texts, normalize_embeddings=True, show_progress_bar=False)
        import torch
        import numpy as np
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = self.tokenizer(texts[start:start+32], padding=True, truncation=True,
                                       max_length=512, return_tensors='pt')
                hidden = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                outputs.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        return np.concatenate(outputs)

    def score(self, queries, texts):
        return (self.encode(queries, True) @ self.encode(texts).T).tolist() if texts else [[] for _ in queries]


def retrieve(items, scores, k, budget, cost):
    """Global top K after fusion; pack intact units without replacement search."""
    if k < 1 or budget < 1 or len(scores) != len(items) or any(not math.isfinite(s) for s in scores):
        raise ValueError('Positive budgets and one finite score per candidate required')
    ranking = sorted(range(len(items)), key=lambda i: -scores[i])[:k]
    selected, used = [], 0
    for i in ranking:
        total = cost('\n\n'.join([s['text'] for s in selected] + [items[i]['text']]))
        size = total - used
        if total <= budget:
            selected.append({**items[i], 'score': scores[i], 'context_cost': size})
            used += size
    return selected, len(ranking)


def metrics(example, selected):
    facts = {(i['document_id'], sid) for i in selected for sid in i['sentence_ids']}
    pages = {i['document_id'] for i in selected}
    gold = example['gold']
    hits = gold & facts if example['evidence_level'] == 'sentence' else {f for f in gold if f[0] in pages}
    return dict(evidence_recall=len(hits)/len(gold), complete_evidence=int(hits == gold),
                page_coverage=len(pages & {f[0] for f in gold})/len({f[0] for f in gold}),
                context_cost=sum(i['context_cost'] for i in selected))


def answer_metrics(prediction, answers):
    def clean(t):
        t = ''.join(c for c in t.lower() if c not in string.punctuation)
        return re.sub(r'\b(a|an|the)\b', ' ', t).split()
    pred, values = clean(prediction), []
    for answer in answers:
        gold = clean(answer)
        em = int(pred == gold)
        overlap = sum((Counter(pred) & Counter(gold)).values())
        special = ('yes', 'no', 'noanswer')
        if prediction.strip().lower() in special or answer.strip().lower() in special:
            f1 = float(em)
        else:
            f1 = 2*overlap/(len(pred)+len(gold)) if pred or gold else 1.
        values.append((em, f1))
    return dict(answer_em=max(v[0] for v in values), answer_f1=max(v[1] for v in values))


def query_list(value):
    if not isinstance(value, list) or not value or any(not isinstance(q, str) or not q.strip() for q in value):
        raise ValueError('Queries must be a nonempty list of nonempty strings')
    return list(dict.fromkeys(q.strip() for q in value))


def evaluate(example, backend, query_sets, k, budget, cost, answers=None):
    cells = {}
    for unit in ('passage', 'sentence', 'window'):
        items = candidates(example['documents'], unit)
        for query_type, queries in query_sets.items():
            matrix = backend.score(queries, [i['text'] for i in items])
            scores = [max(row[i] for row in matrix) for i in range(len(items))]
            selected, examined = retrieve(items, scores, k, budget, cost)
            key = unit + '/' + query_type
            cell = dict(queries=queries, retrieved=selected, examined_candidates=examined,
                        candidate_count=len(items), metrics=metrics(example, selected),
                        reader_input=dict(question=example['question'], context='\n\n'.join(i['text'] for i in selected)))
            if answers is not None:
                prediction = answers[example['id']][key]
                if not isinstance(prediction, str):
                    raise ValueError('Answer predictions must be strings')
                cell['answer'] = prediction
                cell['metrics'].update(answer_metrics(prediction, [example['answer'], *example['aliases']]))
            cells[key] = cell
    return dict(example_id=example['id'], evidence_level=example['evidence_level'], cells=cells)


def summary(records, resamples, seed):
    report = dict(examples=len(records), cells={}, query_effects={}, interactions={})
    keys = list(records[0]['cells'])
    if any(set(r['cells']) != set(keys) for r in records):
        raise ValueError('Conditions must cover all questions')
    for key in keys:
        report['cells'][key] = {}
        for metric in records[0]['cells'][key]['metrics']:
            values = [r['cells'][key]['metrics'][metric] for r in records]
            ci = paired_interval(values, resamples, seed)
            report['cells'][key][metric] = dict(mean=mean(values), percentile_95_interval=ci['percentile_95_interval'])
        unit, query = key.split('/')
        if query == 'original':
            continue
        base = unit + '/original'
        report['query_effects'][key] = {
            m: paired_interval([r['cells'][key]['metrics'][m]-r['cells'][base]['metrics'][m] for r in records], resamples, seed)
            for m in records[0]['cells'][key]['metrics']}
        if unit != 'passage':
            m = 'evidence_recall'
            deltas = [(r['cells'][key]['metrics'][m]-r['cells'][base]['metrics'][m]) -
                      (r['cells']['passage/'+query]['metrics'][m]-r['cells']['passage/original']['metrics'][m]) for r in records]
            report['interactions'][key] = paired_interval(deltas, resamples, seed)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='Local benchmark JSON list or JSONL')
    p.add_argument('--dataset', choices=['hotpotqa', '2wiki', 'musique'], required=True)
    p.add_argument('--retriever', choices=['bm25', 'dense', 'contriever'], default='bm25')
    p.add_argument('--model')
    p.add_argument('--model-revision')
    p.add_argument('--queries', type=Path, help='ID -> {oracle: [...], llm: [...]}')
    p.add_argument('--answers', type=Path, help='ID -> {unit/query: answer}, fixed reader on exported inputs')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--context-budget', type=int, required=True)
    p.add_argument('--tokenizer', help='Fixed reader tokenizer; omission explicitly uses words')
    p.add_argument('--tokenizer-revision')
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.top_k, a.context_budget, a.resamples) < 1:
        p.error('Budgets and resamples must be positive')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    with RunState(a.output, config) as state:
        raw = a.input.read_bytes()
        rows = json.loads(raw) if raw.lstrip().startswith(b'[') else [json.loads(l) for l in raw.splitlines() if l.strip()]
        examples = [normalize(r, a.dataset) for r in rows if r.get('answerable', True)]
        if not examples or len({e['id'] for e in examples}) != len(examples):
            raise ValueError('Require nonempty examples with unique IDs')
        annotations = json.loads(a.queries.read_text()) if a.queries else {}
        answers = json.loads(a.answers.read_text()) if a.answers else None
        sets = []
        for e in examples:
            q = dict(original=[e['question']], rule=query_list(generate_reasoning_units(e['question']) or [e['question']]))
            if e['oracle']:
                q['oracle'] = query_list(e['oracle'])
            for name, value in annotations.get(e['id'], {}).items():
                if name not in ('oracle', 'llm'):
                    raise ValueError('Only oracle and llm annotations allowed')
                q[name] = query_list(value)
            sets.append(q)
        if any(set(q) != set(sets[0]) for q in sets):
            raise ValueError('Oracle/LLM annotations must cover every question')
        cost = lambda t: len(t.split())
        if a.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, revision=a.tokenizer_revision)
            cost = lambda t: len(tokenizer.encode(t, add_special_tokens=False))
        if a.retriever == 'bm25':
            backend = BM25()
        else:
            model = a.model or ('facebook/contriever' if a.retriever == 'contriever' else None)
            if not model:
                raise ValueError('Dense retrieval requires --model')
            backend = Dense(model, a.model_revision, a.retriever == 'contriever')
        state.info.update(dict(input_sha256=hashlib.sha256(raw).hexdigest(),
                               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               example_ids=[e['id'] for e in examples], excluded_unanswerable=len(rows)-len(examples),
                               corpus_scope='per-question provided candidates; not open-domain',
                               budget_unit='tokens' if a.tokenizer else 'words',
                               budget_policy='global top K after max fusion, intact-unit context cap',
                               evidence_level=examples[0]['evidence_level'],
                               annotation_sha256=hashlib.sha256(a.queries.read_bytes()).hexdigest() if a.queries else None,
                               answers_sha256=hashlib.sha256(a.answers.read_bytes()).hexdigest() if a.answers else None))
        state.phase('evaluating')
        records = []
        with (a.output/'predictions.jsonl').open('x') as stream:
            for e, q in zip(examples, sets):
                record = evaluate(e, backend, q, a.top_k, a.context_budget, cost, answers)
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                records.append(record)
                state.info['processed_examples'] = len(records)
                state.save()
        state.phase('bootstrap')
        write_json(a.output/'summary.json', summary(records, a.resamples, a.seed))


if __name__ == '__main__':
    main()
