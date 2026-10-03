import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evidence import indexed_sentences, sentence_windows


class EvidenceTests(unittest.TestCase):
    def test_blank_sentences_preserve_gold_coordinates(self):
        group = [' ', ' first ', '', None, ' gold ']
        self.assertEqual(indexed_sentences(group), [(1, 'first'), (4, 'gold')])
        facts = {('Page', index) for index, _ in indexed_sentences(group)}
        self.assertIn(('Page', 4), facts)
        self.assertNotIn(('Page', 0), facts)

    def test_windows_do_not_credit_skipped_coordinates(self):
        windows = list(sentence_windows(['first', '', 'second', 'third']))
        self.assertEqual(windows[1]['sentence_ids'], [0, 2, 3])
        self.assertEqual(windows[1]['text'], 'first second third')
        self.assertNotIn(1, windows[1]['sentence_ids'])

    def test_empty_and_single_sentence(self):
        self.assertEqual(list(sentence_windows(['', None])), [])
        self.assertEqual(list(sentence_windows([' x '])),
                         [{'sentence_ids': [0], 'text': 'x'}])

    def test_contiguous_windows_match_previous_behavior(self):
        self.assertEqual([w['sentence_ids'] for w in sentence_windows(['a', 'b', 'c'])],
                         [[0, 1], [0, 1, 2], [1, 2]])

    def test_invalid_sizes(self):
        for size in [0, 2, -1]:
            with self.assertRaises(ValueError):
                list(sentence_windows(['x'], size))


if __name__ == '__main__':
    unittest.main()
