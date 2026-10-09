"""Validate benchmark structure before normalization can discard evidence."""


def _text(value, label, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f'{label} must be a string' + ('' if allow_empty else ' with nonempty text'))


def _list(value, label, allow_empty=False):
    if not isinstance(value, (list, tuple)) or (not allow_empty and not value):
        raise ValueError(f'{label} must be a ' + ('possibly empty ' if allow_empty else 'nonempty ') + 'list')
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')


def paired_fields(value, left, right, label):
    """Accept native pairs or columnar data, never silently truncate columns."""
    if isinstance(value, dict):
        a = _list(value.get(left), f'{label}.{left}')
        b = _list(value.get(right), f'{label}.{right}')
        if len(a) != len(b):
            raise ValueError(f'{label}.{left} and {label}.{right} must have equal lengths')
        return list(zip(a, b))
    pairs = _list(value, label)
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in pairs):
        raise ValueError(f'{label} must contain two-element pairs')
    return pairs


def validate_benchmark(row, dataset):
    """Check types and native coordinates without altering benchmark labels."""
    if dataset not in ('hotpotqa', '2wiki', 'musique'):
        raise ValueError(f'Unknown dataset: {dataset}')
    if not isinstance(row, dict):
        raise ValueError('Benchmark row must be an object')
    _text(row.get('id', row.get('_id')), 'ID')
    _text(row.get('question'), 'question')
    _text(row.get('answer'), 'answer')
    for alias in _list(row.get('answer_aliases', []), 'answer_aliases', allow_empty=True):
        _text(alias, 'answer alias')
    if dataset in ('hotpotqa', '2wiki'):
        docs = paired_fields(row.get('context'), 'title', 'sentences', 'context')
        for title, sentences in docs:
            _text(title, 'context title')
            for sentence in _list(sentences, 'sentences', allow_empty=True):
                _text(sentence, 'sentence', allow_empty=True)
        for title, sid in paired_fields(row.get('supporting_facts'), 'title', 'sent_id', 'supporting_facts'):
            _text(title, 'supporting fact title')
            _integer(sid, 'supporting sentence ID')
    else:
        for paragraph in _list(row.get('paragraphs'), 'paragraphs'):
            if not isinstance(paragraph, dict):
                raise ValueError('Each paragraph must be an object')
            _integer(paragraph.get('idx'), 'paragraph idx')
            _text(paragraph.get('title'), 'paragraph title')
            _text(paragraph.get('paragraph_text'), 'paragraph text')
            if type(paragraph.get('is_supporting')) is not bool:
                raise ValueError('is_supporting must be boolean')
        seen = set()
        for step in _list(row.get('question_decomposition'), 'question_decomposition'):
            if not isinstance(step, dict):
                raise ValueError('Each decomposition step must be an object')
            _integer(step.get('id'), 'decomposition step ID', minimum=1)
            if step['id'] in seen:
                raise ValueError('Decomposition step IDs must be unique')
            seen.add(step['id'])
            _text(step.get('question'), 'decomposition question')
            _text(step.get('answer'), 'decomposition answer')
