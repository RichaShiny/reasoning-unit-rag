"""Bind externally generated answers to the exact declared reader request."""
import hashlib
import json
import re
from string import Formatter

READER_SCHEMA_VERSION = 1


def digest(value):
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def validate_reader_config(config):
    required = {'model', 'revision', 'prompt_template', 'decoding'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Reader config requires exactly model, revision, prompt_template and decoding')
    for field in ('model', 'revision', 'prompt_template'):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError('Reader ' + field + ' must be nonempty text')
    if not isinstance(config['decoding'], dict) or not config['decoding']:
        raise ValueError('Reader decoding must explicitly record nonempty settings')
    if any(not isinstance(key, str) or not key for key in config['decoding']):
        raise ValueError('Decoding settings require nonempty string keys')
    fields = set()
    for _, field, spec, conversion in Formatter().parse(config['prompt_template']):
        if field is not None:
            if field not in ('question', 'context') or spec or conversion:
                raise ValueError('Prompt placeholders may only be plain {question} and {context}')
            fields.add(field)
    if fields != {'question', 'context'}:
        raise ValueError('Prompt must include both {question} and {context}')
    digest(config)  # Reject non-JSON values and NaN/infinity.
    return config


def reader_request(reader_input, config):
    validate_reader_config(config)
    if not isinstance(reader_input, dict) or set(reader_input) != {'question', 'context'}:
        raise ValueError('Reader input must contain only question and context')
    if any(not isinstance(value, str) for value in reader_input.values()):
        raise ValueError('Question and context must be strings')
    prompt = config['prompt_template'].format(**reader_input)
    return dict(reader_input=reader_input, prompt=prompt,
                reader_input_sha256=digest(reader_input), reader_config_sha256=digest(config),
                reader_request_sha256=digest(dict(reader_input=reader_input, reader_config=config, prompt=prompt)))


class BoundAnswers:
    def __init__(self, payload, config):
        validate_reader_config(config)
        if (not isinstance(payload, dict) or type(payload.get('schema_version')) is not int
                or payload['schema_version'] != READER_SCHEMA_VERSION):
            raise ValueError('Answers require reader schema version 1; legacy unbound answers are unsupported')
        validate_reader_config(payload.get('reader_config'))
        if digest(payload['reader_config']) != digest(config):
            raise ValueError('Answer reader configuration does not match --reader-config')
        self.reader_config_sha256 = digest(config)
        self.predictions = payload.get('predictions')
        if not isinstance(self.predictions, dict) or not self.predictions:
            raise ValueError('Answers require nonempty predictions')
        for identifier, cells in self.predictions.items():
            if not isinstance(identifier, str) or not identifier or not isinstance(cells, dict) or not cells:
                raise ValueError('Answers require nonempty IDs and cell maps')
            for cell in cells.values():
                if not isinstance(cell, dict) or set(cell) != {'answer', 'reader_request_sha256'}:
                    raise ValueError('Each answer requires answer and reader_request_sha256')
                if not isinstance(cell['answer'], str):
                    raise ValueError('Answer predictions must be strings')
                if not isinstance(cell['reader_request_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', cell['reader_request_sha256']):
                    raise ValueError('Invalid reader request hash')

    def check_coverage(self, expected):
        if self.predictions.keys() != expected.keys():
            raise ValueError('Answer IDs must exactly match the evaluated questions')
        for identifier, keys in expected.items():
            if set(self.predictions[identifier]) != set(keys):
                raise ValueError('Answer cells must exactly match evaluated conditions for ' + identifier)

    def answer(self, identifier, key, request_hash):
        item = self.predictions[identifier][key]
        if item['reader_request_sha256'] != request_hash:
            raise ValueError('Stale or mismatched reader request for ' + identifier + '/' + key)
        return item['answer']


def validate_reader_records(records):
    """Prevent pooled answer scores from combining different reader settings."""
    config_hashes = {record.get('reader_config_sha256') for record in records}
    if len(config_hashes) != 1:
        raise ValueError('Summary requires one fixed reader configuration')
    config_hash = next(iter(config_hashes))
    for record in records:
        for cell in record['cells'].values():
            answered = 'answer' in cell or bool({'answer_em', 'answer_f1'} & cell['metrics'].keys())
            if answered and (config_hash is None or not cell.get('reader_request_sha256')):
                raise ValueError('Answer metrics require bound reader provenance')
            if cell.get('reader_config_sha256') != config_hash:
                raise ValueError('Cell reader configuration does not match record')
    return config_hash
