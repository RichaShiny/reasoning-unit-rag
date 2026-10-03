import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_evaluation import (build_sentences, evidence_metrics, evaluate_example,
                            sample_examples, select_ranked, summarize)

EXAMPLE = {'id': 'test', 'question': 'Who is older, Annie Morton or Terry Richardson?',
           'type': 'comparison',
           'context': {'title': ['Annie Morton', 'Terry Richardson'],
                       'sentences': [['', 'Born first'], ['Born second']]},
           'supporting_facts': {'title': ['Annie Morton', 'Terry Richardson'], 'sent_id': [1, 0]}}


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        import numpy as np
        self.calls.append(texts)
        return np.array([[1., 0.] if 'Annie' in t else [0., 1.] for t in texts])


class EvaluationTests(unittest.TestCase):
    def test_deterministic_sample_and_exclusion(self):
        data = list(range(30))
        a = sample_examples(data, 10, 42, exclude_first=5)
        self.assertEqual(a, sample_examples(data, 10, 42, exclude_first=5))
        self.assertFalse(set(a) & set(range(5)))
        self.assertEqual(sample_examples(data, 3, 42, first=True), [0, 1, 2])
        with self.assertRaises(ValueError):
            sample_examples(data, 30, 42, exclude_first=1)

    def test_source_ids_and_metrics(self):
        items = build_sentences(EXAMPLE)
        self.assertEqual(items[0]['sentence_id'], 1)
        metrics = evidence_metrics(EXAMPLE, items)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['complete_evidence'], 1)
        self.assertEqual(evidence_metrics(EXAMPLE, items[:1])['page_coverage'], .5)

    def test_deduplication_ties_and_words(self):
        items = [{'title': 'P', 'sentence_id': i, 'text': text}
                 for i, text in [(0, 'long sentence here'), (1, 'short'), (2, 'tiny')]]
        ranked = select_ranked([items[0], items[0], *items[1:]], [1, 1, 1, 1], 3, 2)
        self.assertEqual([s['sentence_id'] for s in ranked], [1, 2])
        ranked = select_ranked([items[0], items[0], *items[1:]], [1, 1, 1, 1], 2)
        self.assertEqual([s['sentence_id'] for s in ranked], [0, 1])

    def test_shared_budget_and_gold_isolation(self):
        encoder = Encoder()
        result = evaluate_example(EXAMPLE, encoder, top_k=1)
        self.assertTrue(all(len(m['retrieved']) == 1 for m in result['methods'].values()))
        self.assertTrue(all(isinstance(q, str) for q in encoder.calls[0]))
        changed = copy.deepcopy(EXAMPLE)
        changed['supporting_facts'] = {'title': ['Secret gold'], 'sent_id': [99]}
        changed['answer'] = 'Secret answer'
        other = Encoder()
        alternate = evaluate_example(changed, other, top_k=1)
        self.assertEqual(encoder.calls, other.calls)
        self.assertEqual(result['methods']['rule_based']['retrieved'],
                         alternate['methods']['rule_based']['retrieved'])

    def test_empty_context_reports_zero_coverage(self):
        empty = copy.deepcopy(EXAMPLE)
        empty['context'] = {'title': [], 'sentences': []}
        result = evaluate_example(empty, Encoder())
        self.assertEqual(result['methods']['original_question']['metrics']['recall'], 0)
        self.assertEqual(result['methods']['original_question']['retrieved'], [])

    def test_summary_is_paired_and_stratified(self):
        record = evaluate_example(EXAMPLE, Encoder(), top_k=2)
        report = summarize([record])
        self.assertIn('type:comparison', report)
        self.assertEqual(report['all']['paired_recall_difference'], 0)


if __name__ == '__main__':
    unittest.main()
