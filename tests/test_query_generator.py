import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from query_generator import PROMPT, QuestionGenerator, make_openai_provider, parse_queries
from run_evaluation import evaluate_example, main, summarize
from test_evaluation import EXAMPLE, Encoder


def response(text='["Annie Morton birth date", "Terry Richardson birth date"]', status='completed'):
    return {'raw_text': text, 'usage': {'input_tokens': 10, 'output_tokens': 20},
            'returned_model': 'synthetic-v1', 'response_id': 'test-response', 'response_status': status}


class GeneratorTests(unittest.TestCase):
    def test_valid_and_malformed_output(self):
        self.assertEqual(parse_queries('[" a ", "a"]'), ['a'])
        for text in ['{}', '[]', '[1]', '[" "]', '["a","b","c"]',
                     '```json\n["a"]\n```', json.dumps(['x' * 241])]:
            with self.assertRaises((ValueError, TypeError)):
                parse_queries(text)

    def test_cache_replay_never_calls_provider_and_tracks_usage(self):
        with tempfile.TemporaryDirectory() as cache:
            calls = []
            def provider(request):
                calls.append(request)
                return response()
            generator = QuestionGenerator('synthetic-v1', cache, provider)
            first = generator.generate('Question')
            replay = QuestionGenerator('synthetic-v1', cache, cache_only=True).generate('Question')
            self.assertEqual(len(calls), 1)
            self.assertFalse(first['cache_hit'])
            self.assertTrue(replay['cache_hit'])
            self.assertFalse(replay['api_call_attempted'])
            self.assertEqual(first['response'], replay['response'])
            self.assertEqual(first['queries'], replay['queries'])
            with self.assertRaises(FileNotFoundError):
                generator = QuestionGenerator('different-model', cache, cache_only=True)
                generator.generate('Question')
            with self.assertRaises(FileNotFoundError):
                QuestionGenerator('synthetic-v1', cache, cache_only=True,
                                  max_output_tokens=1024).generate('Question')
            with self.assertRaises(FileNotFoundError):
                QuestionGenerator('synthetic-v1', cache, cache_only=True,
                                  reasoning_effort='low').generate('Question')

    def test_error_and_invalid_outputs_are_cached_fallbacks(self):
        def fail(request):
            raise RuntimeError('Secret-like provider message must not be logged')
        for provider, reason in [(fail, 'provider_error'),
                                 (lambda request: response('[]'), 'invalid_output'),
                                 (lambda request: response('["valid"]', 'incomplete'), 'incomplete_response')]:
            with tempfile.TemporaryDirectory() as cache:
                result = QuestionGenerator('synthetic-v1', cache, provider).generate('Question')
                self.assertTrue(result['fallback'])
                self.assertEqual(result['queries'], ['Question'])
                self.assertEqual(result['fallback_reason'], reason)
                replay = QuestionGenerator('synthetic-v1', cache, cache_only=True).generate('Question')
                self.assertEqual(result['response'], replay['response'])
                self.assertNotIn('Secret-like', json.dumps(result))

    def test_tampered_identity_rejected(self):
        with tempfile.TemporaryDirectory() as cache:
            first = QuestionGenerator('synthetic-v1', cache, lambda request: response()).generate('Q')
            path = Path(cache) / (first['cache_key'] + '.json')
            data = json.loads(path.read_text())
            data['identity']['request']['instructions'] = 'wrong prompt'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                QuestionGenerator('synthetic-v1', cache, cache_only=True).generate('Q')

    def test_gold_isolation_and_shared_budget(self):
        with tempfile.TemporaryDirectory() as cache:
            calls = []
            def provider(request):
                calls.append(request)
                return response()
            generator = QuestionGenerator('synthetic-v1', cache, provider)
            result = evaluate_example(EXAMPLE, Encoder(), top_k=1, word_budget=10, generator=generator)
            altered = copy.deepcopy(EXAMPLE)
            altered['answer'] = 'SECRET ANSWER'
            altered['supporting_facts'] = {'title': ['SECRET TITLE'], 'sent_id': [999]}
            other = evaluate_example(altered, Encoder(), top_k=1, word_budget=10, generator=generator)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['input'], EXAMPLE['question'])
            self.assertEqual(set(calls[0]), {'model', 'instructions', 'input', 'max_output_tokens', 'store'})
            self.assertNotIn('SECRET', json.dumps(calls))
            for method in result['methods'].values():
                self.assertLessEqual(len(method['retrieved']), 1)
                self.assertLessEqual(method['metrics']['retrieved_words'], 10)
            self.assertEqual(result['methods']['automated']['retrieved'], other['methods']['automated']['retrieved'])
            original = evaluate_example(EXAMPLE, Encoder(), top_k=1, word_budget=10)
            self.assertEqual(result['methods']['original_question'], original['methods']['original_question'])
            self.assertEqual(result['methods']['rule_based'], original['methods']['rule_based'])
            stats = summarize([result, other])['generation']
            self.assertEqual(stats['api_call_attempts'], 1)
            self.assertEqual(stats['cache_hits'], 1)
            self.assertEqual(stats['reported_input_tokens_on_new_calls'], 10)

    def test_automated_queries_do_not_change_baseline_scores(self):
        class DistinctEncoder:
            def encode(self, texts, **kwargs):
                import numpy as np
                return np.array([[.8, .6] if t == 'Find a fact' else
                                 [0., 1.] if t == 'second entity' or 'Terry' in t else
                                 [1., 0.] for t in texts])
        example = copy.deepcopy(EXAMPLE)
        example['question'] = 'Find a fact'
        with tempfile.TemporaryDirectory() as cache:
            generator = QuestionGenerator('synthetic-v1', cache,
                                          lambda request: response('["second entity"]'))
            result = evaluate_example(example, DistinctEncoder(), top_k=1, generator=generator)
            self.assertEqual(result['methods']['original_question']['retrieved'][0]['title'], 'Annie Morton')
            self.assertEqual(result['methods']['rule_based']['retrieved'][0]['title'], 'Annie Morton')
            self.assertEqual(result['methods']['automated']['retrieved'][0]['title'], 'Terry Richardson')

    def test_fallback_preserves_original_question_ranking(self):
        with tempfile.TemporaryDirectory() as cache:
            generator = QuestionGenerator('synthetic-v1', cache, lambda request: response('{}'))
            result = evaluate_example(EXAMPLE, Encoder(), top_k=2, generator=generator)
            self.assertEqual(result['methods']['automated'], result['methods']['original_question'])
            self.assertEqual(summarize([result])['generation']['fallback_rate'], 1)

    def test_sdk_request_contract_without_network(self):
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(output_text='["subquery"]', model='synthetic-v1',
                                   usage=SimpleNamespace(model_dump=lambda: {'input_tokens': 3}),
                                   id='test', status='completed')
        def factory(**kwargs):
            self.assertEqual(kwargs, {'timeout': 60, 'max_retries': 0})
            return SimpleNamespace(responses=SimpleNamespace(create=create))
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'}), \
             patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=factory)}), \
             tempfile.TemporaryDirectory() as cache:
            generator = QuestionGenerator('synthetic-v1', cache, make_openai_provider(), reasoning_effort='low')
            result = generator.generate('Q')
            self.assertEqual(result['queries'], ['subquery'])
            self.assertEqual(requests[0]['instructions'], PROMPT)
            self.assertEqual(requests[0]['reasoning'], {'effort': 'low'})
            self.assertFalse(requests[0]['store'])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                make_openai_provider()

    def test_cli_cache_replay_writes_complete_run_without_sdk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'cache'
            QuestionGenerator('synthetic-v1', cache, lambda request: response()).generate(EXAMPLE['question'])
            source = root / 'examples.json'
            source.write_text(json.dumps([EXAMPLE]))
            output = root / 'run'
            arguments = ['run_evaluation.py', '--input', str(source), '--count', '1', '--first',
                         '--generator-model', 'synthetic-v1', '--generator-cache', str(cache),
                         '--generator-cache-only', '--output', str(output)]
            with patch.object(sys, 'argv', arguments), \
                 patch.dict(sys.modules, {'sentence_transformers': SimpleNamespace(SentenceTransformer=lambda *a, **k: Encoder())}), \
                 patch('builtins.print'):
                main()
            manifest = json.loads((output / 'manifest.json').read_text())
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['generator']['model'], 'synthetic-v1')
            self.assertEqual(summary['generation']['cache_hits'], 1)
            self.assertEqual(summary['generation']['api_call_attempts'], 0)
            self.assertIn('automated', summary['all']['methods'])


if __name__ == '__main__':
    unittest.main()
