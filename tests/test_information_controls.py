import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from diagnostic_matrix import BM25, evaluate, normalize, summary
from information_controls import CONDITIONS, PROTOCOL, build_conditions, load_conditions
from matrix_artifacts import load_matrix_run, validate_comparison_manifests

QUESTION = 'Where was the author of The Lantern born?'
ANNOTATION = dict(question=QUESTION,
                  steps=['Who wrote The Lantern?', 'Where was {author} born?'],
                  bridges={'author': {'description': 'the author of The Lantern', 'entity': 'Mira Vale'}},
                  provenance={'author': 'synthetic-test', 'source': 'synthetic-fixture'})
ROW = dict(id='q', question=QUESTION, answer='Harbor City',
           context=[['The Lantern', ['The Lantern was written by Mira Vale.']],
                    ['Mira Vale', ['Mira Vale was born in Harbor City.']]],
           supporting_facts=[['The Lantern', 0], ['Mira Vale', 0]])
PAYLOAD = dict(schema_version=1, examples={'q': ANNOTATION})


class InformationControlTests(unittest.TestCase):
    def test_paired_construction_and_no_implicit_original_retention(self):
        queries = build_conditions(QUESTION, ANNOTATION)
        self.assertEqual(tuple(queries), CONDITIONS)
        self.assertEqual(queries['decomposition'], ['Who wrote The Lantern?',
                                                  'Where was the author of The Lantern born?'])
        self.assertEqual(queries['decomposition_bridge'], ['Who wrote The Lantern?', 'Where was Mira Vale born?'])
        self.assertNotIn('Mira Vale', ' '.join(queries['decomposition']))
        self.assertIn('Mira Vale', queries['original_bridge'][0])
        self.assertEqual(queries['original'], [QUESTION])
        self.assertNotIn(ROW['answer'], json.dumps(queries))

    def test_label_changes_do_not_change_constructed_queries(self):
        ex = normalize(ROW, 'hotpotqa')
        changed = {**ex, 'answer': 'SECRET', 'gold': {('Mira Vale', 0)}}
        self.assertEqual(load_conditions(PAYLOAD, [ex]), load_conditions(PAYLOAD, [changed]))
        other = copy.deepcopy(ANNOTATION)
        other['bridges']['author']['entity'] = 'Another Author'
        before, after = build_conditions(QUESTION, ANNOTATION), build_conditions(QUESTION, other)
        self.assertEqual(before['original'], after['original'])
        self.assertEqual(before['decomposition'], after['decomposition'])
        self.assertNotEqual(before['original_bridge'], after['original_bridge'])
        self.assertNotEqual(before['decomposition_bridge'], after['decomposition_bridge'])

    def test_missing_stale_or_malformed_annotations_fail(self):
        ex = normalize(ROW, 'hotpotqa')
        for payload in ({'schema_version': 1, 'examples': {}},
                        {'schema_version': True, 'examples': {'q': ANNOTATION}},
                        {'schema_version': 1, 'examples': {'q': ANNOTATION, 'extra': ANNOTATION}}):
            with self.assertRaises(ValueError):
                load_conditions(payload, [ex])
        for change in ('stale', 'undeclared', 'attribute', 'unused', 'duplicate', 'no_bridge', 'same'):
            bad = copy.deepcopy(ANNOTATION)
            if change == 'stale': bad['question'] += ' changed'
            elif change == 'undeclared': bad['steps'][1] = 'Where is {other}?'
            elif change == 'attribute': bad['steps'][1] = 'Where is {author.name}?'
            elif change == 'unused': bad['steps'][1] = 'Where is someone?'
            elif change == 'duplicate': bad['steps'] = ['Where is {author}?']*2
            elif change == 'no_bridge': bad['bridges'] = {}
            else: bad['bridges']['author']['entity'] = bad['bridges']['author']['description']
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_conditions(QUESTION, bad)

    def test_known_paired_contrasts_and_interaction(self):
        ex = normalize(ROW, 'hotpotqa')
        records = []
        for idx, values in enumerate(((.1, .3, .4, .9), (.2, .1, .5, .7))):
            record = evaluate(ex, BM25(), build_conditions(QUESTION, ANNOTATION), 2, 100, lambda s: len(s.split()))
            record['example_id'] = str(idx)
            for unit in ('passage', 'sentence', 'window'):
                for condition, value in zip(CONDITIONS, values):
                    record['cells'][unit+'/'+condition]['metrics']['evidence_recall'] = value
            records.append(record)
        report = summary(records, 100, 42)['information_effects']
        effects = report['units']['sentence']['evidence_recall']
        for name, expected in [('decomposition_without_bridge', .05), ('decomposition_with_bridge', .35),
                               ('bridge_without_decomposition', .3), ('bridge_with_decomposition', .6),
                               ('interaction', .3)]:
            self.assertAlmostEqual(effects[name]['mean_difference'], expected)
        for bound in effects['interaction']['percentile_95_interval']:
            self.assertAlmostEqual(bound, .3)
        self.assertEqual(report['protocol'], PROTOCOL)
        for record in records:
            for unit in ('passage', 'sentence', 'window'):
                del record['cells'][unit+'/original_bridge']
        with self.assertRaisesRegex(ValueError, 'all four'):
            summary(records, 20, 42)

    def test_cli_runs_all_four_conditions_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp/'data.json').write_text(json.dumps([ROW]))
            (temp/'annotations.json').write_text(json.dumps(PAYLOAD))
            command = [sys.executable, str(ROOT/'src/diagnostic_matrix.py'), '--dataset', 'hotpotqa',
                       '--input', str(temp/'data.json'), '--information-controls', str(temp/'annotations.json'),
                       '--context-budget', '100', '--resamples', '20']
            subprocess.run(command+['--output', str(temp/'run')], check=True, capture_output=True, text=True)
            manifest, records = load_matrix_run(temp/'run/predictions.jsonl')
            self.assertEqual(manifest['query_types'], list(CONDITIONS))
            self.assertEqual(manifest['information_control_protocol'], PROTOCOL)
            self.assertIsNotNone(manifest['annotation_sha256'])
            self.assertEqual(len(records[0]['cells']), 12)
            report = json.loads((temp/'run/summary.json').read_text())
            self.assertIn('information_effects', report)
            self.assertEqual(records[0]['cells']['sentence/decomposition']['query_count'], 2)
            self.assertEqual(records[0]['cells']['sentence/original_bridge']['query_count'], 1)
            changed = {**manifest, 'information_control_source_sha256': '0'*64}
            with self.assertRaisesRegex(ValueError, 'information_control_source'):
                validate_comparison_manifests(manifest, changed)
            failed = subprocess.run(command+['--original-question', 'include', '--output', str(temp/'bad')],
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((temp/'bad').exists())


if __name__ == '__main__':
    unittest.main()
