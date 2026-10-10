"""Keep exact sentence support separate from paragraph-level coverage proxies."""
import math

METRIC_SCHEMA_VERSION = 2


def metric_contract(level):
    if level == 'sentence':
        return dict(metric_schema_version=METRIC_SCHEMA_VERSION, evidence_level=level,
                    primary_metric='evidence_recall',
                    evidence_interpretation='Coverage of annotated supporting sentence coordinates')
    if level == 'paragraph':
        return dict(metric_schema_version=METRIC_SCHEMA_VERSION, evidence_level=level,
                    primary_metric='supporting_paragraph_coverage',
                    evidence_interpretation='Supporting paragraphs touched; sentence/fact support is unverified')
    raise ValueError('Unknown evidence level: ' + str(level))


def retrieval_metrics(example, selected):
    level = example['evidence_level']
    metric_contract(level)
    gold = example['gold']
    if not gold:
        raise ValueError('Evidence metrics require nonempty gold labels')
    pages = {item['document_id'] for item in selected}
    gold_pages = {fact[0] for fact in gold}
    coverage = len(pages & gold_pages) / len(gold_pages)
    result = dict(page_coverage=coverage, context_cost=sum(item['context_cost'] for item in selected))
    if level == 'sentence':
        facts = {(item['document_id'], sid) for item in selected for sid in item['sentence_ids']}
        hits = gold & facts
        result.update(evidence_recall=len(hits)/len(gold), complete_evidence=int(hits == gold))
    else:
        result.update(supporting_paragraph_coverage=coverage,
                      all_supporting_paragraphs_touched=int(gold_pages <= pages))
    return result


def validate_metric_records(records):
    """Require a single, explicit estimand before averaging or differencing."""
    if not records:
        raise ValueError('Require nonempty records')
    ids = [record['example_id'] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Require unique example IDs')
    contract = metric_contract(records[0].get('evidence_level'))
    expected = None
    for record in records:
        for field, value in contract.items():
            if record.get(field) != value:
                raise ValueError('Incompatible or missing metric contract: ' + field + '; regenerate legacy runs and keep evidence levels separate')
        if not record['cells']:
            raise ValueError('Require nonempty experimental cells')
        for cell in record['cells'].values():
            values = cell['metrics']
            required = ({'evidence_recall', 'complete_evidence'} if contract['evidence_level'] == 'sentence'
                        else {'supporting_paragraph_coverage', 'all_supporting_paragraphs_touched'})
            forbidden = ({'supporting_paragraph_coverage', 'all_supporting_paragraphs_touched'}
                         if contract['evidence_level'] == 'sentence' else {'evidence_recall', 'complete_evidence'})
            if not required <= values.keys() or forbidden & values.keys():
                raise ValueError('Metric names do not match the evidence level')
            if expected is None:
                expected = set(values)
            if set(values) != expected:
                raise ValueError('All cells must report the same metric set')
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values.values()):
                raise ValueError('Metrics must be finite numbers')
    return contract
