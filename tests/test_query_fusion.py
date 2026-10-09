import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from diagnostic_matrix import BM25, evaluate, normalize
from query_fusion import fuse_scores, prepare_queries
from compare_matrix import compare

ROW = dict(id='q', question='alpha beta', answer='A',
           context=[['T', ['alpha alpha alpha.', 'beta.']]], supporting_facts=[['T', 1]])


class QueryFusionTests(unittest.TestCase):
    def test_bm25_subset_queries_cannot_change_max_fusion(self):
        backend = BM25()
        texts = ['alpha alpha beta', 'beta beta', 'alpha', 'unrelated']
        matrix = backend.score(['alpha beta', 'beta', 'alpha'], texts)
        self.assertEqual(fuse_scores(matrix, len(texts)), matrix[0])

    def test_rrf_uses_ranks_not_score_scale(self):
        matrix = [[100, 90, 0], [0, 1, .5]]
        self.assertEqual(fuse_scores(matrix, 3), [100, 90, .5])
        scores = fuse_scores(matrix, 3, 'rrf', 60)
        self.assertEqual(max(range(3), key=scores.__getitem__), 1)
        self.assertEqual(scores, fuse_scores([[100, 90, 0], [0, 1000, 500]], 3, 'rrf', 60))
        self.assertAlmostEqual(scores[1], 1/62 + 1/61)

    def test_rrf_ties_do_not_invent_document_order_signal(self):
        scores = fuse_scores([[5, 5, 0]], 3, 'rrf', 60)
        self.assertEqual(scores[:2], [1/61, 1/61])
        self.assertEqual(scores[2], 1/63)
        self.assertEqual(fuse_scores([[0, 0, 0]], 3, 'rrf'), [1/61]*3)
        self.assertEqual(fuse_scores([[]], 0, 'rrf'), [])

    def test_invalid_scores_and_dimensions_fail_even_when_max_would_hide_them(self):
        for matrix in ([], [[1], [1, 2]], [[1, float('nan')], [2, 3]], [[float('inf'), 0]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                fuse_scores(matrix, 2)
        for constant in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                fuse_scores([[1]], 1, 'rrf', constant)
        with self.assertRaises(ValueError):
            fuse_scores([[1]], 1, 'unknown')

    def test_backend_must_score_every_effective_query(self):
        class IncompleteBackend:
            def score(self, queries, texts):
                return [[0.0] * len(texts)]
        with self.assertRaisesRegex(ValueError, 'one score row'):
            evaluate(normalize(ROW, 'hotpotqa'), IncompleteBackend(),
                     {'llm': ['alpha', 'beta']}, 1, 50, len)

    def test_original_retention_is_explicit_and_deduplicated(self):
        self.assertEqual(prepare_queries('Q', [' Q ', 'sub', 'sub']), ['Q', 'sub'])
        self.assertEqual(prepare_queries('Q', ['sub'], 'include'), ['Q', 'sub'])
        self.assertEqual(prepare_queries('Q', ['Q', 'sub'], 'exclude'), ['sub'])
        self.assertEqual(prepare_queries('Q', ['Q'], 'exclude', baseline=True), ['Q'])
        with self.assertRaisesRegex(ValueError, 'No non-original query'):
            prepare_queries('Q', ['Q'], 'exclude')
        with self.assertRaises(ValueError):
            prepare_queries('Q', ['other'], baseline=True)

    def test_evaluate_records_effective_queries_and_actual_score_work(self):
        ex = normalize(ROW, 'hotpotqa')
        record = evaluate(ex, BM25(), {'original': ['alpha beta'], 'llm': ['beta']},
                          1, 50, lambda t: len(t.split()), fusion='rrf', original_question='include')
        cell = record['cells']['sentence/llm']
        self.assertEqual(cell['queries'], ['alpha beta', 'beta'])
        self.assertTrue(cell['original_question_included'])
        self.assertEqual(cell['query_count'], 2)
        self.assertEqual(cell['scored_query_candidate_pairs'], 4)
        self.assertEqual(cell['examined_candidates'], 1)
        self.assertLessEqual(cell['metrics']['context_cost'], 50)
        self.assertEqual(compare([record], [copy.deepcopy(record)], 20)['sentence/llm']['mean_difference'], 0)
        for field, value in [('queries', ['changed']), ('fusion', 'max'), ('rrf_k', 10),
                             ('original_question_policy', 'exclude')]:
            other = copy.deepcopy(record)
            other['cells']['sentence/llm'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Mismatch'):
                compare([record], [other], 20)
        other = copy.deepcopy(record)
        del other['cells']['sentence/llm']['fusion']
        with self.assertRaisesRegex(ValueError, 'Missing comparison provenance'):
            compare([record], [other], 20)

    def test_cli_controls_manifest_and_rejects_empty_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp/'input.json').write_text(json.dumps([ROW]))
            (tmp/'queries.json').write_text(json.dumps({'q': {'llm': ['beta']}}))
            command = [sys.executable, str(ROOT/'src/diagnostic_matrix.py'), '--input', str(tmp/'input.json'),
                       '--dataset', 'hotpotqa', '--queries', str(tmp/'queries.json'), '--context-budget', '50',
                       '--resamples', '20', '--query-types', 'original', 'llm', '--fusion', 'rrf',
                       '--original-question', 'exclude']
            subprocess.run(command + ['--output', str(tmp/'run')], capture_output=True, text=True, check=True)
            manifest = json.loads((tmp/'run/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['fusion'], 'rrf')
            self.assertEqual(manifest['original_question_policy'], 'exclude')
            self.assertEqual(manifest['query_types'], ['original', 'llm'])
            record = json.loads((tmp/'run/predictions.jsonl').read_text())
            self.assertEqual(record['cells']['sentence/llm']['queries'], ['beta'])
            (tmp/'queries.json').write_text(json.dumps({'q': {'llm': ['alpha beta']}}))
            failed = subprocess.run(command + ['--output', str(tmp/'failed')], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(json.loads((tmp/'failed/manifest.json').read_text())['status'], 'failed')
            self.assertFalse((tmp/'failed/predictions.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
