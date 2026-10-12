import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from matrix_artifacts import load_matrix_run, validate_comparison_manifests

ROW = dict(id='q', question='Where is Alpha?', answer='Boston',
           context=[['Alpha', ['Alpha is in Boston.']]], supporting_facts=[['Alpha', 0]])


class MatrixArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'input.json').write_text(json.dumps([ROW]))
        subprocess.run([sys.executable, str(ROOT/'src/diagnostic_matrix.py'), '--dataset', 'hotpotqa',
                        '--input', str(self.root/'input.json'), '--context-budget', '100', '--resamples', '20',
                        '--output', str(self.root/'run')], check=True, capture_output=True, text=True)
        self.path = self.root/'run/predictions.jsonl'
        self.manifest_path = self.root/'run/manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_complete_run_loads_with_exact_manifest(self):
        manifest, rows = load_matrix_run(self.path)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(rows, self.rows)

    def test_partial_status_count_and_id_mismatches_fail(self):
        for field, value in [('status', 'started'), ('status', 'failed'), ('status', 'interrupted'),
                             ('processed_examples', 0), ('processed_examples', True),
                             ('example_ids', []), ('example_ids', ['q', 'extra'])]:
            with self.subTest(field=field, value=value):
                self.manifest_path.write_text(json.dumps({**self.manifest, field: value}))
                with self.assertRaises(ValueError):
                    load_matrix_run(self.path)

    def test_missing_manifest_fields_or_source_hash_fail(self):
        for field in ('input_sha256', 'source_sha256', 'query_types', 'annotation_sha256'):
            missing = copy.deepcopy(self.manifest)
            del missing[field]
            self.manifest_path.write_text(json.dumps(missing))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Missing manifest'):
                load_matrix_run(self.path)
        self.manifest_path.write_text(json.dumps({**self.manifest, 'source_sha256': 'invalid'}))
        with self.assertRaisesRegex(ValueError, 'Invalid manifest hash'):
            load_matrix_run(self.path)

    def test_missing_duplicate_and_inconsistent_cells_fail(self):
        for mutation in ('missing_cell', 'extra_row', 'wrong_fusion', 'empty_predictions'):
            rows = copy.deepcopy(self.rows)
            if mutation == 'missing_cell':
                del rows[0]['cells']['sentence/rule']
            elif mutation == 'extra_row':
                rows.append(copy.deepcopy(rows[0]))
            elif mutation == 'wrong_fusion':
                rows[0]['cells']['sentence/rule']['fusion'] = 'rrf'
            else:
                rows = []
            self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_matrix_run(self.path)

    def test_source_drift_is_not_a_retriever_only_contrast(self):
        for field in ('source_sha256', 'fusion_source_sha256', 'metric_source_sha256', 'corpus_scope'):
            other = {**self.manifest, field: '0'*64}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_comparison_manifests(self.manifest, other)
        other = copy.deepcopy(self.manifest)
        other['configuration']['dataset'] = '2wiki'
        with self.assertRaisesRegex(ValueError, 'dataset'):
            validate_comparison_manifests(self.manifest, other)
        other = copy.deepcopy(self.manifest)
        other['configuration']['retriever'] = 'contriever'
        validate_comparison_manifests(self.manifest, other)

    def test_cli_records_seed_and_rejects_incomplete_run_without_report(self):
        command = [sys.executable, str(ROOT/'scripts/compare_matrix.py'), str(self.path), str(self.path),
                   '--resamples', '20', '--seed', '17']
        report = self.root/'comparison.json'
        subprocess.run(command + ['--output', str(report)], check=True, capture_output=True, text=True)
        saved = json.loads(report.read_text())
        self.assertEqual(saved['bootstrap'], {'resamples': 20, 'seed': 17})
        self.assertEqual(saved['runs'][0]['examples'], 1)
        self.assertEqual(saved['effects']['sentence/rule']['mean_difference'], 0)
        self.manifest_path.write_text(json.dumps({**self.manifest, 'status': 'interrupted'}))
        failed_path = self.root/'failed-comparison.json'
        failed = subprocess.run(command + ['--output', str(failed_path)], capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(failed_path.exists())

    def test_cli_rejects_different_evaluator_versions(self):
        shutil.copytree(self.root/'run', self.root/'other')
        other_manifest = self.root/'other/manifest.json'
        other_manifest.write_text(json.dumps({**self.manifest, 'source_sha256': '0'*64}))
        output = self.root/'version-comparison.json'
        failed = subprocess.run([sys.executable, str(ROOT/'scripts/compare_matrix.py'), str(self.path),
                                 str(self.root/'other/predictions.jsonl'), '--output', str(output),
                                 '--resamples', '20'], capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn('source_sha256', failed.stderr)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
