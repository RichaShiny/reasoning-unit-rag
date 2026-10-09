import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from diagnostic_matrix import (normalize, candidates, BM25, retrieve, evaluate,
                               summary, answer_metrics)

ROW = dict(id='q', question='Which city is Alpha in?', answer='Boston',
           context=dict(title=['Alpha', 'Beta'], sentences=[['', 'Alpha is in Boston.', 'Other fact.'], ['Beta is elsewhere.']]),
           supporting_facts=dict(title=['Alpha'], sent_id=[1]))


class MatrixTests(unittest.TestCase):
    def test_global_budget_and_serialized_context(self):
        items = [dict(text='oversized item text'), dict(text='a'), dict(text='b')]
        chosen, examined = retrieve(items, [3, 2, 1], 2, 2, len)
        self.assertEqual(examined, 2)
        self.assertEqual([i['text'] for i in chosen], ['a'])
        chosen, _ = retrieve(items[1:], [2, 1], 2, 3, len)
        self.assertEqual(len(chosen), 1)  # separators consume tokenizer context too

    def test_native_coordinates_and_musique_labels(self):
        ex = normalize(ROW, 'hotpotqa')
        self.assertEqual(candidates(ex['documents'], 'sentence')[0]['sentence_ids'], [1])
        native = dict(id='m', question='Q?', answer='A', paragraphs=[
            dict(idx=0, title='T', paragraph_text='First. Second.', is_supporting=True)],
            question_decomposition=[dict(id=1, question='Sub?', answer='Bridge'), dict(id=2, question='Where is #1?', answer='A')])
        ex = normalize(native, 'musique')
        self.assertEqual(ex['evidence_level'], 'paragraph')
        self.assertEqual(ex['gold'], {('0', None)})
        self.assertEqual(ex['oracle'], ['Sub?', 'Where is Bridge?'])
        result = evaluate(ex, BM25(), {'original':['Q?']}, 1, 100, lambda t:len(t.split()))
        self.assertEqual(result['cells']['sentence/original']['metrics']['evidence_recall'], 1)

    def test_labels_do_not_change_nonoracle_rankings(self):
        ex = normalize(ROW, 'hotpotqa')
        changed = copy.deepcopy(ex)
        changed['gold'] = {('Beta', 0)}
        changed['answer'] = 'secret'
        args = (BM25(), {'original':[ex['question']], 'oracle':['Boston']}, 1, 100, lambda t:len(t.split()))
        first, second = evaluate(ex, *args), evaluate(changed, *args)
        for key in first['cells']:
            self.assertEqual(first['cells'][key]['retrieved'], second['cells'][key]['retrieved'])

    def test_bootstrap_pairing_and_interaction(self):
        ex = normalize(ROW, 'hotpotqa')
        r = evaluate(ex, BM25(), {'original':[ex['question']], 'oracle':['Boston']}, 1, 100, lambda t:len(t.split()))
        report = summary([r], 20, 42)
        self.assertEqual(report['query_effects']['sentence/oracle']['evidence_recall']['mean_difference'], 0)
        self.assertEqual(report['interactions']['sentence/oracle']['percentile_95_interval'], [0,0])

    def test_answers_and_invalid_scores(self):
        self.assertEqual(answer_metrics('The Boston', ['Boston']), {'answer_em':1, 'answer_f1':1})
        self.assertEqual(answer_metrics('yes indeed', ['yes'])['answer_f1'], 0)
        with self.assertRaises(ValueError):
            retrieve([{'text':'x'}], [float('nan')], 1, 5, len)
        with self.assertRaises(ValueError):
            normalize({**ROW, 'supporting_facts':dict(title=['Alpha'], sent_id=[99])}, 'hotpotqa')


if __name__ == '__main__':
    unittest.main()
