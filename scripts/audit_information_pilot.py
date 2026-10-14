"""Verify the frozen development pilot and reconstruct its reported results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from diagnostic_matrix import normalize, summary
from information_controls import load_conditions
from matrix_artifacts import load_matrix_run


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(root):
    protocol = json.loads((root/'protocol.json').read_text())
    source = ROOT/protocol['input_source']
    if file_hash(source) != protocol['source_sha256']:
        raise ValueError('Pilot source snapshot changed')
    for name, expected in protocol['locked_inputs'].items():
        if file_hash(root/name) != expected:
            raise ValueError('Frozen pilot input changed: ' + name)
    rows = json.loads(source.read_text())
    candidates = [row for row in rows if row['type'] == 'bridge'][:protocol['candidate_count']]
    if [row['id'] for row in candidates] != protocol['candidate_ids']:
        raise ValueError('Candidate selection differs from the first bridge-labeled prefix')
    stage = json.loads((root/'question-stage-review.json').read_text())
    review = json.loads((root/'annotation-review.json').read_text())
    if review['question_stage_sha256'] != file_hash(root/'question-stage-review.json'):
        raise ValueError('Question-stage review identity changed')
    if [item['example_id'] for item in stage['candidates']] != protocol['candidate_ids']:
        raise ValueError('Question-stage coverage differs from selection')
    if [item['example_id'] for item in review['candidates']] != protocol['candidate_ids']:
        raise ValueError('Review must retain every selected candidate')
    included = [item for item in review['candidates'] if item['decision'] == 'include']
    if [item['example_id'] for item in included] != protocol['included_ids']:
        raise ValueError('Inclusion decisions differ from the frozen protocol')
    selected = json.loads((root/'examples.json').read_text())
    lookup = {row['id']: row for row in rows}
    if selected != [lookup[identifier] for identifier in protocol['included_ids']]:
        raise ValueError('Pilot examples differ from their source records')
    payload = json.loads((root/'annotations.json').read_text())
    query_sets = load_conditions(payload, [normalize(row, 'hotpotqa') for row in selected])
    drafts = {item['example_id']: item for item in stage['candidates']}
    for item, row, queries in zip(included, selected, query_sets):
        annotation = payload['examples'][row['id']]
        draft = drafts[row['id']]
        bridge = annotation['bridges'][draft['slot']]
        if annotation['steps'] != draft['steps'] or bridge['description'] != draft['description']:
            raise ValueError('Question-only draft changed during source annotation')
        if bridge['entity'] != item['intermediate_entity']:
            raise ValueError('Entity differs from reviewed source')
        docs = dict(zip(row['context']['title'], row['context']['sentences']))
        gold = set(zip(row['supporting_facts']['title'], row['supporting_facts']['sent_id']))
        for citation in item['source_citations']:
            title, sid = citation['title'], citation['sentence_id']
            if (title, sid) not in gold or docs[title][sid] != citation['text']:
                raise ValueError('Citation does not match supplied gold coordinates/text')
        if not any(bridge['entity'].casefold() in (ref['title']+' '+ref['text']).casefold()
                   for ref in item['source_citations']):
            raise ValueError('Entity not found in reviewed source references')
        if any(row['answer'].casefold() in query.casefold() for values in queries.values() for query in values):
            raise ValueError('Literal final answer appears in constructed query')
    return protocol


def audit(root):
    protocol = verify_inputs(root)
    result = dict(interpretation='Exploratory same-agent-annotated development pilot; no independent review or held-out claim',
                  candidate_count=protocol['candidate_count'], included_count=protocol['included_count'],
                  excluded_count=protocol['candidate_count']-protocol['included_count'],
                  reader_inference=False, runs={})
    for fusion in protocol['fusion']:
        for budget in protocol['context_budgets']:
            name = f'{fusion}-b{budget}'
            run = root/name
            manifest, records = load_matrix_run(run/'predictions.jsonl')
            expected = dict(retriever=protocol['retriever'], fusion=fusion, rrf_k=protocol['rrf_k'],
                            top_k=protocol['top_k'], context_budget=budget, seed=protocol['seed'],
                            resamples=protocol['bootstrap_resamples'])
            if any(manifest['configuration'][key] != value for key, value in expected.items()):
                raise ValueError('Run configuration differs from protocol: ' + name)
            if manifest['example_ids'] != protocol['included_ids']:
                raise ValueError('Run IDs differ from pilot sample')
            if manifest['input_sha256'] != file_hash(root/'examples.json') or manifest['annotation_sha256'] != file_hash(root/'annotations.json'):
                raise ValueError('Run input/annotation hashes differ from locked files')
            for field, source in [('source_sha256', 'diagnostic_matrix.py'), ('fusion_source_sha256', 'query_fusion.py'),
                                  ('metric_source_sha256', 'evidence_metrics.py'), ('information_control_source_sha256', 'information_controls.py')]:
                if manifest[field] != file_hash(ROOT/'src'/source):
                    raise ValueError('Run source does not match current checkout: ' + source)
            rebuilt = summary(records, protocol['bootstrap_resamples'], protocol['seed'])
            if rebuilt != json.loads((run/'summary.json').read_text()):
                raise ValueError('Saved summary does not match reconstructed paired statistics')
            for record in records:
                for cell in record['cells'].values():
                    if cell['metrics']['context_cost'] > budget or len(cell['retrieved']) > protocol['top_k']:
                        raise ValueError('Output budget exceeded')
                    if 'answer' in cell or 'answer_em' in cell['metrics']:
                        raise ValueError('This pilot does not include reader inference')
            units = {}
            for unit in ('passage', 'sentence', 'window'):
                units[unit] = dict(conditions={condition: {
                    metric: rebuilt['cells'][unit+'/'+condition][metric]['mean']
                    for metric in ('evidence_recall', 'complete_evidence', 'context_cost')}
                    for condition in rebuilt['information_effects']['conditions']},
                    effects=rebuilt['information_effects']['units'][unit]['evidence_recall'])
            result['runs'][name] = dict(examples=len(records), cells=sum(len(record['cells']) for record in records), units=units)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'results/information-pilot')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    print(json.dumps({'verified_runs': len(result['runs']), 'questions': result['included_count'],
                      'cells': sum(run['cells'] for run in result['runs'].values())}))


if __name__ == '__main__':
    main()
