"""Rerun the frozen BM25 development pilot in a fresh output directory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from audit_information_pilot import ROOT, audit, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = ROOT/'results/information-pilot'
    protocol = verify_inputs(source)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('protocol.json', *protocol['locked_inputs']):
        shutil.copyfile(source/name, args.output/name)
    for fusion in protocol['fusion']:
        for budget in protocol['context_budgets']:
            subprocess.run([sys.executable, str(ROOT/'src/diagnostic_matrix.py'), '--dataset', 'hotpotqa',
                            '--input', str(args.output/'examples.json'),
                            '--information-controls', str(args.output/'annotations.json'),
                            '--retriever', protocol['retriever'], '--fusion', fusion,
                            '--rrf-k', str(protocol['rrf_k']), '--top-k', str(protocol['top_k']),
                            '--context-budget', str(budget), '--resamples', str(protocol['bootstrap_resamples']),
                            '--seed', str(protocol['seed']), '--output', str(args.output/f'{fusion}-b{budget}')], check=True)
    result = audit(args.output)
    with (args.output/'analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print('Reproduced and verified the exploratory pilot in ' + str(args.output))


if __name__ == '__main__':
    main()
