import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_evaluation import build_sentences, evidence_metrics, summarize

for name in sys.argv[1:] or ['results/replication-100', 'results/validation-500']:
    path = Path(name)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'Run is incomplete: {path}')
    examples = json.loads((path/'examples.json').read_text())
    rows = [json.loads(line) for line in (path/'predictions.jsonl').read_text().splitlines()]
    assert [r['example_id'] for r in rows] == manifest['example_ids']
    assert len(rows) == len(examples)
    assert summarize(rows) == json.loads((path/'summary.json').read_text())
    diagnostics = {'examples': len(rows), 'blank_sentences': 0,
                   'old_filter_shifted_coordinates': 0,
                   'questions_with_extra_rule_queries': 0,
                   'changed_sentence_sets': 0, 'improved_recall': 0, 'reduced_recall': 0,
                   'baseline_incomplete': 0, 'baseline_missing_evidence_page': 0,
                   'baseline_within_page_only': 0}
    for e, row in zip(examples, rows):
        items = build_sentences(e)
        valid = {(s['title'],s['sentence_id']):s['text'] for s in items}
        gold = set(zip(e['supporting_facts']['title'], e['supporting_facts']['sent_id']))
        assert gold <= valid.keys()
        for group in e['context']['sentences']:
            original = [i for i,s in enumerate(group) if s and s.strip()]
            diagnostics['blank_sentences'] += len(group)-len(original)
            diagnostics['old_filter_shifted_coordinates'] += sum(i!=j for j,i in enumerate(original))
        for method in row['methods'].values():
            selected = method['retrieved']
            coordinates = [(s['title'],s['sentence_id']) for s in selected]
            assert len(set(coordinates)) == len(coordinates) <= manifest['configuration']['top_k']
            assert all(valid[coord] == s['text'] for coord,s in zip(coordinates,selected))
            assert evidence_metrics(e, selected) == method['metrics']
        base = row['methods']['original_question']
        rule = row['methods']['rule_based']
        diagnostics['questions_with_extra_rule_queries'] += len(rule['queries']) > 1
        diagnostics['changed_sentence_sets'] += {(s['title'],s['sentence_id']) for s in base['retrieved']} != {(s['title'],s['sentence_id']) for s in rule['retrieved']}
        diagnostics['improved_recall'] += rule['metrics']['recall'] > base['metrics']['recall']
        diagnostics['reduced_recall'] += rule['metrics']['recall'] < base['metrics']['recall']
        missing=base['metrics']['missing_facts']
        if missing:
            diagnostics['baseline_incomplete'] += 1
            pages={s['title'] for s in base['retrieved']}
            page_missing=any(title not in pages for title,_ in missing)
            diagnostics['baseline_missing_evidence_page'] += page_missing
            diagnostics['baseline_within_page_only'] += not page_missing
    (path/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(name, json.dumps(diagnostics))
