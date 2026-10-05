import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_state import RunState, write_json
from run_evaluation import main


class RunStateTests(unittest.TestCase):
    def test_startup_failure_is_durable_without_exception_message(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            with self.assertRaises(RuntimeError):
                with RunState(output, {'model': 'test'}) as state:
                    state.phase('loading_generator')
                    raise RuntimeError('SECRET provider detail')
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['phase'], 'loading_generator')
            self.assertEqual(manifest['processed_examples'], 0)
            self.assertEqual(manifest['error_type'], 'RuntimeError')
            self.assertNotIn('SECRET', json.dumps(manifest))

    def test_interrupt_keeps_progress(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            with self.assertRaises(KeyboardInterrupt):
                with RunState(output, {}) as state:
                    state.phase('evaluating')
                    state.info['processed_examples'] = 3
                    state.save()
                    raise KeyboardInterrupt()
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'interrupted')
            self.assertEqual(manifest['processed_examples'], 3)

    def test_completion_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            with RunState(output, {}) as state:
                state.info['processed_examples'] = 2
            original = (output / 'manifest.json').read_text()
            self.assertEqual(json.loads(original)['status'], 'complete')
            with self.assertRaises(FileExistsError):
                with RunState(output, {}):
                    pass
            self.assertEqual((output / 'manifest.json').read_text(), original)
            self.assertFalse(list(output.glob('*.tmp')))

    def test_cli_failure_before_model_loading_is_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            arguments = ['run_evaluation.py', '--generator-model', 'test-model',
                         '--generator-cache', str(Path(root) / 'cache'), '--output', str(output)]
            with patch.object(sys, 'argv', arguments), \
                 patch('query_generator.make_openai_provider', side_effect=RuntimeError('private')), \
                 patch('builtins.print'), self.assertRaises(RuntimeError):
                main()
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['phase'], 'loading_generator')
            self.assertEqual(manifest['configuration']['generator_model'], 'test-model')
            self.assertFalse((output / 'predictions.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
