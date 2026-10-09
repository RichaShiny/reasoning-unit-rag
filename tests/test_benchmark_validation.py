import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from diagnostic_matrix import normalize


ROW = dict(_id='q', question='Where is Alpha?', answer='Boston',
           context={'title': ['Alpha', 'Beta'], 'sentences': [['', 'Boston.'], ['Elsewhere.']]},
           supporting_facts={'title': ['Alpha'], 'sent_id': [1]})
MUSIQUE = dict(id='m', question='Where?', answer='Boston', paragraphs=[
    dict(idx=0, title='Alpha', paragraph_text='Boston.', is_supporting=True)],
    question_decomposition=[dict(id=1, question='Who?', answer='Alpha'),
                            dict(id=2, question='Where is #1?', answer='Boston')])


class BenchmarkValidationTests(unittest.TestCase):
    def test_rejects_unequal_columns_in_both_directions(self):
        for dataset in ('hotpotqa', '2wiki'):
            for field, column, extra in [('context', 'title', 'Lost'),
                                         ('context', 'sentences', ['Lost.']),
                                         ('supporting_facts', 'title', 'Lost'),
                                         ('supporting_facts', 'sent_id', 0)]:
                with self.subTest(dataset=dataset, field=field, column=column):
                    row = copy.deepcopy(ROW)
                    row[field][column].append(extra)
                    with self.assertRaisesRegex(ValueError, 'equal lengths'):
                        normalize(row, dataset)

    def test_native_and_columnar_coordinates_agree(self):
        row = copy.deepcopy(ROW)
        row['context'] = [['Alpha', ['', 'Boston.']], ['Beta', ['Elsewhere.']]]
        row['supporting_facts'] = [['Alpha', 1]]
        for dataset in ('hotpotqa', '2wiki'):
            self.assertEqual(normalize(row, dataset), normalize(ROW, dataset))
            self.assertEqual(normalize(row, dataset)['gold'], {('Alpha', 1)})

    def test_rejects_malformed_pairs_and_sentence_groups(self):
        for field, value in [('context', [['Alpha']]), ('context', [['Alpha', 'Boston.']]),
                             ('context', [['Alpha', [None]]]),
                             ('supporting_facts', [['Alpha', 1, 'extra']])]:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    normalize({**ROW, field: value}, 'hotpotqa')

    def test_rejects_invalid_sentence_ids_and_missing_evidence(self):
        for sid in (True, -1, 1.0, '1', 0, 99):
            with self.subTest(sid=sid):
                with self.assertRaises(ValueError):
                    normalize({**ROW, 'supporting_facts': [['Alpha', sid]]}, 'hotpotqa')
        with self.assertRaises(ValueError):
            normalize({**ROW, 'supporting_facts': [['Missing', 0]]}, 'hotpotqa')

    def test_rejects_bad_top_level_types_and_unknown_dataset(self):
        for field, value in [('question', ''), ('answer', None), ('answer_aliases', 'Boston'),
                             ('answer_aliases', [None]), ('context', [])]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    normalize({**ROW, field: value}, 'hotpotqa')
        with self.assertRaisesRegex(ValueError, 'Unknown dataset'):
            normalize(ROW, 'typo')
        with self.assertRaises(ValueError):
            normalize([], 'hotpotqa')

    def test_musique_preserves_dependencies_and_rejects_invalid_structure(self):
        self.assertEqual(normalize(MUSIQUE, 'musique')['oracle'], ['Who?', 'Where is Alpha?'])
        for field, value in [('paragraph_text', ''), ('is_supporting', 'false'), ('idx', True)]:
            row = copy.deepcopy(MUSIQUE)
            row['paragraphs'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize(row, 'musique')
        for modify in ('duplicate_step', 'unresolved', 'duplicate_document', 'no_gold'):
            row = copy.deepcopy(MUSIQUE)
            if modify == 'duplicate_step':
                row['question_decomposition'][1]['id'] = 1
            elif modify == 'unresolved':
                row['question_decomposition'][0]['question'] = 'Where is #2?'
            elif modify == 'duplicate_document':
                row['paragraphs'].append(copy.deepcopy(row['paragraphs'][0]))
            else:
                row['paragraphs'][0]['is_supporting'] = False
            with self.subTest(modify=modify), self.assertRaises(ValueError):
                normalize(row, 'musique')


if __name__ == '__main__':
    unittest.main()
