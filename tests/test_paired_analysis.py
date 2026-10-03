import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paired_analysis import analyze, paired_interval


class PairedAnalysisTests(unittest.TestCase):
    def test_pairing_identical_methods_has_zero_interval(self):
        records = []
        for i, value in enumerate([0., .5, 1.]):
            metrics = {'recall': value, 'complete_evidence': int(value == 1), 'page_coverage': value}
            records.append({'example_id': str(i), 'question_type': 'bridge',
                            'methods': {'original_question': {'metrics': metrics},
                                        'rule_based': {'metrics': dict(metrics)}}})
        report = analyze(records, resamples=100)
        self.assertEqual(report['groups']['all']['recall']['percentile_95_interval'], [0, 0])
        self.assertIn('type:bridge', report['groups'])
        with self.assertRaises(ValueError):
            analyze([records[0], records[0]], resamples=100)

    def test_seeded_percentile_interval(self):
        a = paired_interval([-1, 0, 1], resamples=300, seed=42)
        self.assertEqual(a, paired_interval([-1, 0, 1], resamples=300, seed=42))
        self.assertEqual(a['mean_difference'], 0)
        low, high = a['percentile_95_interval']
        self.assertLess(low, 0)
        self.assertGreater(high, 0)

    def test_constant_and_invalid_inputs(self):
        self.assertEqual(paired_interval([.25])['percentile_95_interval'], [.25, .25])
        for values in [[], [float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                paired_interval(values)
        with self.assertRaises(ValueError):
            paired_interval([0], resamples=0)


if __name__ == '__main__':
    unittest.main()
