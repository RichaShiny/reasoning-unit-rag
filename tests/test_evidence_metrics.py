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
from diagnostic_matrix import BM25, evaluate, normalize, summary
from evidence_metrics import metric_contract, retrieval_metrics, validate_metric_records
from compare_matrix import compare

MUSIQUE = dict(id='m', question='Where is Alpha?', answer='Boston', paragraphs=[
    dict(idx=0, title='Alpha', paragraph_text='Unrelated opening. Alpha is in Boston.', is_supporting=True),
    dict(idx=1, title='Beta', paragraph_text='A second required fact.', is_supporting=True)],
    question_decomposition=[dict(id=1, question='Where is Alpha?', answer='Boston')])
HOTPOT = dict(id='h', question='Where is Alpha?', answer='Boston',
              context=[['Alpha', ['Unrelated opening.', 'Alpha is in Boston.']]],
              supporting_facts=[['Alpha', 1]])


def record(row, dataset):
    ex = normalize(row, dataset)
    return evaluate(ex, BM25(), {'original': [ex['question']], 'oracle': ['Boston']},
                    1, 100, lambda text: len(text.split()))


class EvidenceMetricTests(unittest.TestCase):
    def test_irrelevant_sentence_only_earns_paragraph_coverage(self):
        ex = normalize(MUSIQUE, 'musique')
        irrelevant = dict(document_id='0', sentence_ids=[0], context_cost=3)
        values = retrieval_metrics(ex, [irrelevant])
        self.assertEqual(values['supporting_paragraph_coverage'], .5)
        self.assertEqual(values['all_supporting_paragraphs_touched'], 0)
        self.assertNotIn('evidence_recall', values)
        self.assertNotIn('complete_evidence', values)
        values = retrieval_metrics(ex, [irrelevant, dict(document_id='1', sentence_ids=[0], context_cost=4)])
        self.assertEqual(values['supporting_paragraph_coverage'], 1)
        self.assertEqual(values['all_supporting_paragraphs_touched'], 1)
        self.assertEqual(values['context_cost'], 7)
        self.assertEqual(retrieval_metrics(ex, [])['all_supporting_paragraphs_touched'], 0)

    def test_sentence_support_still_requires_annotated_coordinates(self):
        ex = normalize(HOTPOT, 'hotpotqa')
        irrelevant = dict(document_id='Alpha', sentence_ids=[0], context_cost=3)
        values = retrieval_metrics(ex, [irrelevant])
        self.assertEqual(values['page_coverage'], 1)
        self.assertEqual(values['evidence_recall'], 0)
        self.assertEqual(values['complete_evidence'], 0)
        values = retrieval_metrics(ex, [{**irrelevant, 'sentence_ids': [1]}])
        self.assertEqual(values['evidence_recall'], 1)
        self.assertEqual(values['complete_evidence'], 1)
        self.assertNotIn('supporting_paragraph_coverage', values)

    def test_summary_and_comparison_identify_the_paragraph_estimand(self):
        left = record(MUSIQUE, 'musique')
        right = copy.deepcopy(left)
        for key, cell in left['cells'].items():
            cell['metrics']['supporting_paragraph_coverage'] = .5
            right['cells'][key]['metrics']['supporting_paragraph_coverage'] = 1 if key == 'sentence/oracle' else .5
        report = summary([right], 20, 42)
        self.assertEqual(report['primary_metric'], 'supporting_paragraph_coverage')
        self.assertEqual(report['metric_schema_version'], 2)
        self.assertEqual(report['interactions']['sentence/oracle']['mean_difference'], .5)
        effect = compare([left], [right], 20)['sentence/oracle']
        self.assertEqual(effect['metric'], 'supporting_paragraph_coverage')
        self.assertEqual(effect['mean_difference'], .5)

    def test_mixed_levels_and_legacy_metrics_are_rejected(self):
        sentence, paragraph = record(HOTPOT, 'hotpotqa'), record(MUSIQUE, 'musique')
        with self.assertRaisesRegex(ValueError, 'metric contract'):
            summary([sentence, paragraph], 20, 42)
        for mutate in ('legacy', 'wrong_name', 'schema', 'missing_metric', 'nonfinite'):
            bad = copy.deepcopy(paragraph)
            if mutate == 'legacy':
                del bad['metric_schema_version']
            elif mutate == 'wrong_name':
                bad['cells']['sentence/oracle']['metrics']['complete_evidence'] = 1
            elif mutate == 'schema':
                bad['metric_schema_version'] = 999
            elif mutate == 'missing_metric':
                del bad['cells']['sentence/oracle']['metrics']['supporting_paragraph_coverage']
            else:
                bad['cells']['sentence/oracle']['metrics']['supporting_paragraph_coverage'] = float('nan')
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                summary([bad], 20, 42)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                compare([paragraph], [bad], 20)
        paragraph['example_id'] = sentence['example_id']
        with self.assertRaisesRegex(ValueError, 'metric contracts'):
            compare([sentence], [paragraph], 20)
        with self.assertRaises(ValueError):
            validate_metric_records([])
        with self.assertRaises(ValueError):
            summary([sentence, sentence], 20, 42)
        with self.assertRaises(ValueError):
            metric_contract('unknown')

    def test_cli_exports_consistent_contract_and_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = temp/'input.jsonl'
            source.write_text(json.dumps(MUSIQUE)+'\n')
            output = temp/'run'
            subprocess.run([sys.executable, str(ROOT/'src/diagnostic_matrix.py'), '--dataset', 'musique',
                            '--input', str(source), '--output', str(output), '--context-budget', '100',
                            '--resamples', '20'], check=True, capture_output=True, text=True)
            manifest = json.loads((output/'manifest.json').read_text())
            predictions = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
            report = json.loads((output/'summary.json').read_text())
            for artifact in (manifest, predictions[0], report):
                for field, value in metric_contract('paragraph').items():
                    self.assertEqual(artifact[field], value)
            subprocess.run([sys.executable, str(ROOT/'scripts/compare_matrix.py'),
                            str(output/'predictions.jsonl'), str(output/'predictions.jsonl'),
                            '--output', str(temp/'comparison.json'), '--resamples', '20'],
                           check=True, capture_output=True, text=True)
            comparison = json.loads((temp/'comparison.json').read_text())
            self.assertEqual(comparison['primary_metric'], 'supporting_paragraph_coverage')
            self.assertEqual(comparison['effects']['sentence/oracle']['mean_difference'], 0)
            manifest['primary_metric'] = 'evidence_recall'
            (output/'manifest.json').write_text(json.dumps(manifest))
            failed = subprocess.run([sys.executable, str(ROOT/'scripts/compare_matrix.py'),
                                     str(output/'predictions.jsonl'), str(output/'predictions.jsonl'),
                                     '--output', str(temp/'bad-comparison.json'), '--resamples', '20'],
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Manifest metric contract', failed.stderr)
            self.assertFalse((temp/'bad-comparison.json').exists())


if __name__ == '__main__':
    unittest.main()
