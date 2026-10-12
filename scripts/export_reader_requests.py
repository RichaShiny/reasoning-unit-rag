"""Export prompts from a completed diagnostic run without labels or oracle queries."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evidence_metrics import validate_metric_records
from reader_provenance import READER_SCHEMA_VERSION, digest, reader_request, validate_reader_config, validate_reader_records


def export_requests(records, manifest, config):
    validate_reader_config(config)
    contract = validate_metric_records(records)
    ids = [record['example_id'] for record in records]
    if (manifest.get('status') != 'complete' or manifest.get('processed_examples') != len(records)
            or manifest.get('example_ids') != ids):
        raise ValueError('Export requires a complete run with exactly its manifested records')
    if any(manifest.get(field) != value for field, value in contract.items()):
        raise ValueError('Manifest metric contract does not match predictions')
    keys = set(records[0]['cells'])
    if any(set(record['cells']) != keys for record in records):
        raise ValueError('Export requires the same conditions for every question')
    run_hash = validate_reader_records(records)
    if manifest.get('reader_config_sha256') != run_hash:
        raise ValueError('Manifest reader config does not match predictions')
    config_hash = digest(config)
    if run_hash is not None and (run_hash != config_hash or digest(manifest.get('reader_config')) != config_hash):
        raise ValueError('Export config does not match the run manifest')
    requests = {}
    for record in records:
        recorded_hash = record.get('reader_config_sha256')
        if recorded_hash is not None and recorded_hash != config_hash:
            raise ValueError('Export config does not match the run reader config')
        cells = {}
        for key, cell in record['cells'].items():
            request = reader_request(cell['reader_input'], config)
            if cell.get('reader_input_sha256') != request['reader_input_sha256']:
                raise ValueError('Missing or mismatched reader input hash; regenerate legacy runs')
            if cell['reader_input']['context'] != '\n\n'.join(item['text'] for item in cell['retrieved']):
                raise ValueError('Reader context does not match retrieved units')
            if cell.get('reader_config_sha256') != recorded_hash:
                raise ValueError('Cell reader config does not match record')
            if recorded_hash is not None and cell.get('reader_request_sha256') != request['reader_request_sha256']:
                raise ValueError('Reader request does not match recorded configuration')
            cells[key] = request
        requests[record['example_id']] = cells
    return dict(schema_version=READER_SCHEMA_VERSION, reader_config=config, requests=requests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--reader-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    manifest = json.loads((args.predictions.parent / 'manifest.json').read_text())
    config = json.loads(args.reader_config.read_text())
    payload = export_requests(records, manifest, config)
    with args.output.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
