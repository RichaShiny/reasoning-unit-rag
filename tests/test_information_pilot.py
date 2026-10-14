import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_information_pilot import verify_inputs


class InformationPilotTests(unittest.TestCase):
    def test_frozen_sample_and_source_annotations_verify(self):
        protocol = verify_inputs(ROOT/'results/information-pilot')
        self.assertEqual(protocol['candidate_count'], 12)
        self.assertEqual(protocol['included_count'], 8)
        self.assertFalse(protocol['reader_inference'])

    def test_changed_annotation_is_not_silently_a_frozen_pilot(self):
        source = ROOT/'results/information-pilot'
        protocol = json.loads((source/'protocol.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for name in ('protocol.json', *protocol['locked_inputs']):
                shutil.copyfile(source/name, temp/name)
            payload = json.loads((temp/'annotations.json').read_text())
            first = next(iter(payload['examples'].values()))
            first['steps'][0] = 'Changed after results were observed?'
            (temp/'annotations.json').write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'Frozen pilot input changed'):
                verify_inputs(temp)

    def test_different_selection_is_rejected_even_with_original_input_files(self):
        source = ROOT/'results/information-pilot'
        protocol = json.loads((source/'protocol.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for name in ('protocol.json', *protocol['locked_inputs']):
                shutil.copyfile(source/name, temp/name)
            protocol['candidate_ids'].reverse()
            (temp/'protocol.json').write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, 'Candidate selection'):
                verify_inputs(temp)


if __name__ == '__main__':
    unittest.main()
