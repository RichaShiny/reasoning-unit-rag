from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from local_generator import make_local_provider
from query_generator import QuestionGenerator
from run_evaluation import evaluate_example, summarize
from test_evaluation import EXAMPLE, Encoder
from test_query_generator import response


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self

    def __getitem__(self, index):
        return self.values


class LocalGeneratorTests(unittest.TestCase):
    def build_provider(self, input_length=10, eos=True, limit=512):
        calls = {'loads': [], 'generations': []}
        def tokenize(prompt, **kwargs):
            calls['prompt'] = prompt
            calls['tokenization'] = kwargs
            return {'input_ids': Tensor([1] * input_length)}
        tokenizer = SimpleNamespace(eos_token_id=1, decode=lambda *a, **k: '["subquery"]')
        class Tokenizer:
            eos_token_id = 1
            def __call__(self, prompt, **kwargs):
                return tokenize(prompt, **kwargs)
            def decode(self, *args, **kwargs):
                return tokenizer.decode(*args, **kwargs)
        def generate(**kwargs):
            calls['generations'].append(kwargs)
            return Tensor([0, 2, 1 if eos else 2])
        network = SimpleNamespace(to=lambda device: calls.update(device=device),
                                  eval=lambda: calls.update(eval=True), generate=generate)
        def load_tokenizer(model, **kwargs):
            calls['loads'].append((model, kwargs))
            return Tokenizer()
        def load_network(model, **kwargs):
            calls['loads'].append((model, kwargs))
            return network
        modules = {'torch': SimpleNamespace(inference_mode=nullcontext),
                   'transformers': SimpleNamespace(
                       AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
                       AutoModelForSeq2SeqLM=SimpleNamespace(from_pretrained=load_network))}
        with patch.dict(sys.modules, modules):
            provider = make_local_provider('test-model', 'revision-1', num_beams=2, max_input_tokens=limit)
        return provider, calls

    def test_pinned_loading_and_generation(self):
        provider, calls = self.build_provider()
        result = provider({'instructions': 'Instructions', 'input': 'Question', 'max_output_tokens': 64})
        self.assertEqual(result['returned_model'], 'test-model@revision-1')
        self.assertEqual(result['response_status'], 'completed')
        self.assertEqual(result['usage'], {'input_tokens': 10, 'output_tokens': 2})
        self.assertFalse(calls['tokenization']['truncation'])
        self.assertFalse(calls['generations'][0]['do_sample'])
        self.assertEqual(calls['generations'][0]['num_beams'], 2)
        self.assertEqual(calls['loads'][1][1]['revision'], 'revision-1')
        self.assertTrue(calls['loads'][1][1]['use_safetensors'])
        self.assertFalse(calls['loads'][1][1]['trust_remote_code'])
        self.assertTrue(calls['eval'])

    def test_overlong_prompt_and_incomplete_output(self):
        provider, calls = self.build_provider(input_length=20, limit=10)
        with self.assertRaises(ValueError):
            provider({'instructions': 'I', 'input': 'Q', 'max_output_tokens': 5})
        self.assertFalse(calls['generations'])
        provider, _ = self.build_provider(eos=False)
        self.assertEqual(provider({'instructions': 'I', 'input': 'Q', 'max_output_tokens': 5})['response_status'], 'incomplete')

    def test_local_cache_identity_and_call_accounting(self):
        config = {'revision': 'rev', 'device': 'cpu', 'num_beams': 1}
        with tempfile.TemporaryDirectory() as cache:
            local = QuestionGenerator('synthetic-v1', cache, lambda r: response(),
                                      provider_kind='local_seq2seq', provider_configuration=config)
            result = evaluate_example(EXAMPLE, Encoder(), generator=local)
            self.assertFalse(result['generation']['api_call_attempted'])
            stats = summarize([result])['generation']
            self.assertEqual(stats['api_call_attempts'], 0)
            self.assertEqual(stats['provider_call_attempts'], 1)
            self.assertEqual(stats['provider_input_tokens'], 10)
            replay = QuestionGenerator('synthetic-v1', cache, cache_only=True,
                                       provider_kind='local_seq2seq', provider_configuration=config)
            self.assertEqual(replay.generate(EXAMPLE['question'])['queries'], result['generation']['queries'])
            for kind, settings in [('openai_responses', {}), ('local_seq2seq', {**config, 'revision': 'other'})]:
                with self.assertRaises(FileNotFoundError):
                    QuestionGenerator('synthetic-v1', cache, cache_only=True, provider_kind=kind,
                                      provider_configuration=settings).generate(EXAMPLE['question'])


if __name__ == '__main__':
    unittest.main()
