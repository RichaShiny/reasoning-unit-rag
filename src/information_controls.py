"""Construct and contrast decomposition with and without bridge information."""
from string import Formatter
from paired_analysis import paired_interval

CONDITIONS = ('original', 'decomposition', 'original_bridge', 'decomposition_bridge')
PROTOCOL = 'paired-bridge-substitution-v1'


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ' must be nonempty text')
    return value.strip()


def build_conditions(question, annotation):
    if not isinstance(annotation, dict) or set(annotation) != {'question', 'steps', 'bridges', 'provenance'}:
        raise ValueError('Annotation requires question, steps, bridges and provenance')
    if annotation['question'] != question:
        raise ValueError('Annotation question does not match the benchmark')
    provenance = annotation['provenance']
    if not isinstance(provenance, dict) or set(provenance) != {'author', 'source'}:
        raise ValueError('Annotation provenance requires author and source')
    for key, value in provenance.items():
        _text(value, 'provenance.' + key)
    steps, bridges = annotation['steps'], annotation['bridges']
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError('Decomposition requires at least two template steps')
    if not isinstance(bridges, dict) or not bridges:
        raise ValueError('This protocol requires at least one annotated bridge')
    descriptions, entities = {}, {}
    for name, bridge in bridges.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError('Bridge slot names must be identifiers')
        if not isinstance(bridge, dict) or set(bridge) != {'description', 'entity'}:
            raise ValueError('Each bridge requires question-only description and intermediate entity')
        descriptions[name] = _text(bridge['description'], 'bridge description')
        entities[name] = _text(bridge['entity'], 'bridge entity')
    used, plain, resolved = set(), [], []
    for step in steps:
        step = _text(step, 'step')
        for _, field, spec, conversion in Formatter().parse(step):
            if field is not None:
                if field not in bridges or spec or conversion:
                    raise ValueError('Steps may only use declared plain bridge placeholders')
                used.add(field)
        plain.append(step.format(**descriptions))
        resolved.append(step.format(**entities))
    if used != set(bridges):
        raise ValueError('Every bridge must be used by a decomposition step')
    if len(set(plain)) != len(plain) or len(set(resolved)) != len(resolved):
        raise ValueError('Duplicate rendered steps would change query counts after deduplication')
    if plain == resolved:
        raise ValueError('Bridge substitution must change at least one query')
    supplied = '\n'.join(f'{descriptions[name]}: {entities[name]}' for name in sorted(bridges))
    return dict(original=[question.strip()], decomposition=plain,
                original_bridge=[question.strip() + '\nBridge information:\n' + supplied],
                decomposition_bridge=resolved)


def load_conditions(payload, examples):
    if (not isinstance(payload, dict) or type(payload.get('schema_version')) is not int
            or payload['schema_version'] != 1 or set(payload) != {'schema_version', 'examples'}):
        raise ValueError('Information annotations require schema_version 1 and examples')
    annotations = payload['examples']
    if not isinstance(annotations, dict) or set(annotations) != {example['id'] for example in examples}:
        raise ValueError('Information annotations must exactly cover all evaluated question IDs')
    return [build_conditions(example['question'], annotations[example['id']]) for example in examples]


def information_effects(records, resamples, seed):
    """Paired question-level contrasts; these are not automatic causal claims."""
    keys = set(records[0]['cells'])
    present = {key.split('/')[1] for key in keys}
    if not (present & set(CONDITIONS[1:])):
        return None
    expected = {unit+'/'+condition for unit in ('passage', 'sentence', 'window') for condition in CONDITIONS}
    if any(set(record['cells']) != expected for record in records):
        raise ValueError('Information contrasts require all four conditions at every unit and question')
    weights = {
        'decomposition_without_bridge': (-1, 1, 0, 0),
        'decomposition_with_bridge': (0, 0, -1, 1),
        'bridge_without_decomposition': (-1, 0, 1, 0),
        'bridge_with_decomposition': (0, -1, 0, 1),
        'interaction': (1, -1, -1, 1),
    }
    result = dict(protocol=PROTOCOL, conditions=list(CONDITIONS), units={})
    for unit in ('passage', 'sentence', 'window'):
        metrics = records[0]['cells'][unit+'/original']['metrics']
        result['units'][unit] = {}
        for metric in metrics:
            values = [[record['cells'][unit+'/'+condition]['metrics'][metric] for condition in CONDITIONS]
                      for record in records]
            result['units'][unit][metric] = {
                name: paired_interval([sum(w*v for w, v in zip(coefficients, row)) for row in values], resamples, seed)
                for name, coefficients in weights.items()}
    return result
