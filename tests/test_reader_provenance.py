import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from diagnostic_matrix import BM25, evaluate, normalize, summary
from reader_provenance import BoundAnswers, digest, reader_request, validate_reader_config
from export_reader_requests import export_requests

CONFIG = dict(model='test-reader', revision='fixture-revision',
              prompt_template='Answer from context.\n{context}\nQuestion: {question}',
              decoding={'temperature': 0, 'max_new_tokens': 32})
ROW = dict(id='q', question='Where is Alpha?', answer='Boston',
           context=[['Alpha', ['Alpha is in Boston.']], ['Beta', ['Unrelated sentence.']]],
           supporting_facts=[['Alpha', 0]])


def evaluate_fixture(config=CONFIG, answers=None):
    return evaluate(normalize(ROW, 'hotpotqa'), BM25(),
                    {'original': [ROW['question']], 'oracle': ['SECRET_ORACLE_QUERY']},
                    1, 100, lambda text: len(text.split()), answers, reader_config=config)


def answer_bundle(record, config=CONFIG):
    return dict(schema_version=1, reader_config=copy.deepcopy(config), predictions={record['example_id']: {
        key: dict(answer='Boston', reader_request_sha256=cell['reader_request_sha256'])
        for key, cell in record['cells'].items()}})


def manifest_for(record, config=CONFIG):
    fields = ('metric_schema_version', 'evidence_level', 'primary_metric', 'evidence_interpretation')
    return dict(status='complete', processed_examples=1, example_ids=[record['example_id']],
                reader_config=config, reader_config_sha256=digest(config) if config else None,
                **{field: record[field] for field in fields})


class ReaderProvenanceTests(unittest.TestCase):
    def test_hashes_bind_exact_input_prompt_model_and_decoding(self):
        inputs = {'question': 'Which city?', 'context': 'Title. Boston.\n\nSecond sentence.'}
        request = reader_request(inputs, CONFIG)
        self.assertEqual(request, reader_request(dict(reversed(list(inputs.items()))), CONFIG))
        for field in ('question', 'context'):
            self.assertNotEqual(request['reader_request_sha256'],
                                reader_request({**inputs, field: inputs[field]+' '}, CONFIG)['reader_request_sha256'])
        for field, value in [('model', 'other'), ('revision', 'other'),
                             ('prompt_template', '{question}\n{context}'), ('decoding', {'temperature': 1})]:
            self.assertNotEqual(request['reader_request_sha256'],
                                reader_request(inputs, {**CONFIG, field: value})['reader_request_sha256'])
        with self.assertRaises(ValueError):
            reader_request({**inputs, 'gold_answer': 'secret'}, CONFIG)

    def test_config_rejects_ambiguous_or_unsafe_template_fields(self):
        for prompt in ('{context}', '{question} {context} {answer}', '{question.__class__} {context}',
                       '{question!r} {context}', '{question:>10} {context}', '{'):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                validate_reader_config({**CONFIG, 'prompt_template': prompt})
        for field, value in [('revision', ''), ('model', None), ('decoding', {}),
                             ('decoding', {'temperature': float('nan')})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reader_config({**CONFIG, field: value})

    def test_bound_answers_preserve_retrieval_and_score_correctly(self):
        before = evaluate_fixture()
        answers = BoundAnswers(answer_bundle(before), CONFIG)
        answers.check_coverage({'q': set(before['cells'])})
        after = evaluate_fixture(answers=answers)
        for key, cell in after['cells'].items():
            self.assertEqual(cell['retrieved'], before['cells'][key]['retrieved'])
            self.assertEqual(cell['metrics']['answer_em'], 1)
            self.assertEqual(cell['metrics']['answer_f1'], 1)
        report = summary([after], 20, 42)
        self.assertEqual(report['reader_config_sha256'], digest(CONFIG))
        self.assertEqual(report['cells']['sentence/original']['answer_em']['mean'], 1)

    def test_legacy_wrong_config_stale_input_and_wrong_coverage_fail(self):
        base = evaluate_fixture()
        with self.assertRaisesRegex(ValueError, 'schema'):
            BoundAnswers({'q': {'sentence/original': 'Boston'}}, CONFIG)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            BoundAnswers(answer_bundle(base), {**CONFIG, 'revision': 'different'})
        payload = answer_bundle(base)
        payload['predictions']['q']['sentence/original']['reader_request_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Stale or mismatched'):
            evaluate_fixture(answers=BoundAnswers(payload, CONFIG))
        for expected in ({'extra': set(base['cells'])}, {'q': {'sentence/original'}},
                         {'q': set(base['cells']), 'extra': set(base['cells'])}):
            with self.assertRaises(ValueError):
                BoundAnswers(answer_bundle(base), CONFIG).check_coverage(expected)
        with self.assertRaisesRegex(ValueError, 'BoundAnswers'):
            evaluate_fixture(answers=answer_bundle(base))

    def test_export_contains_no_gold_labels_or_oracle_query(self):
        record = evaluate_fixture()
        exported = export_requests([record], manifest_for(record), CONFIG)
        self.assertNotIn('SECRET_ORACLE_QUERY', json.dumps(exported))
        for request in exported['requests']['q'].values():
            self.assertEqual(set(request), {'reader_input', 'prompt', 'reader_input_sha256',
                                           'reader_config_sha256', 'reader_request_sha256'})
            self.assertEqual(set(request['reader_input']), {'question', 'context'})
        unconfigured = evaluate_fixture(config=None)
        self.assertEqual(export_requests([unconfigured], manifest_for(unconfigured, None), CONFIG)['requests'],
                         exported['requests'])

    def test_export_rejects_partial_runs_tampering_and_config_changes(self):
        record = evaluate_fixture()
        for field, value in [('status', 'failed'), ('processed_examples', 0), ('example_ids', ['other']),
                             ('reader_config_sha256', 'wrong')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                export_requests([record], {**manifest_for(record), field: value}, CONFIG)
        bad = copy.deepcopy(record)
        bad['cells']['sentence/original']['reader_input']['context'] += 'tampered'
        with self.assertRaisesRegex(ValueError, 'input hash'):
            export_requests([bad], manifest_for(record), CONFIG)
        bad['cells']['sentence/original']['reader_input_sha256'] = digest(bad['cells']['sentence/original']['reader_input'])
        with self.assertRaisesRegex(ValueError, 'retrieved units'):
            export_requests([bad], manifest_for(record), CONFIG)
        with self.assertRaises(ValueError):
            export_requests([record], manifest_for(record), {**CONFIG, 'revision': 'other'})

    def test_summary_rejects_mixed_readers_and_unbound_answer_scores(self):
        first = evaluate_fixture()
        second = evaluate_fixture(config={**CONFIG, 'revision': 'other'})
        second['example_id'] = 'q2'
        with self.assertRaisesRegex(ValueError, 'fixed reader'):
            summary([first, second], 20, 42)
        unbound = evaluate_fixture(config=None)
        for cell in unbound['cells'].values():
            cell['metrics']['answer_em'] = 1
        with self.assertRaisesRegex(ValueError, 'bound reader'):
            summary([unbound], 20, 42)

    def test_cli_export_answer_round_trip_and_stale_context_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp/'input.json').write_text(json.dumps([ROW]))
            (tmp/'reader.json').write_text(json.dumps(CONFIG))
            command = [sys.executable, str(ROOT/'src/diagnostic_matrix.py'), '--dataset', 'hotpotqa',
                       '--input', str(tmp/'input.json'), '--query-types', 'original', '--context-budget', '100',
                       '--reader-config', str(tmp/'reader.json'), '--resamples', '20']
            subprocess.run(command + ['--output', str(tmp/'retrieval')], check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, str(ROOT/'scripts/export_reader_requests.py'),
                            str(tmp/'retrieval/predictions.jsonl'), '--reader-config', str(tmp/'reader.json'),
                            '--output', str(tmp/'requests.json')], check=True, capture_output=True, text=True)
            requests = json.loads((tmp/'requests.json').read_text())
            payload = dict(schema_version=1, reader_config=CONFIG, predictions={'q': {
                key: dict(answer='Boston', reader_request_sha256=item['reader_request_sha256'])
                for key, item in requests['requests']['q'].items()}})
            (tmp/'answers.json').write_text(json.dumps(payload))
            subprocess.run(command + ['--answers', str(tmp/'answers.json'), '--output', str(tmp/'answered')],
                           check=True, capture_output=True, text=True)
            report = json.loads((tmp/'answered/summary.json').read_text())
            self.assertEqual(report['cells']['sentence/original']['answer_em']['mean'], 1)
            changed = copy.deepcopy(ROW)
            changed['context'][0][1][0] += ' Extra context.'
            (tmp/'input.json').write_text(json.dumps([changed]))
            failed = subprocess.run(command + ['--answers', str(tmp/'answers.json'), '--output', str(tmp/'stale')],
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Stale or mismatched', failed.stderr)
            self.assertEqual(json.loads((tmp/'stale/manifest.json').read_text())['status'], 'failed')
            self.assertFalse((tmp/'stale/summary.json').exists())


if __name__ == '__main__':
    unittest.main()
