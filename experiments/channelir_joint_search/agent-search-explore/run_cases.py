#!/usr/bin/env python3
"""Run named Goal-1 cases; all remaining arguments are passed to pilot.py."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['diagnose', 'compare'])
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dry-run', action='store_true')
    args, extra = p.parse_known_args()
    cases = {c['id']: c for c in json.loads((ROOT / 'dataset.json').read_text())['cases']}
    if len(set(args.cases)) != len(args.cases) or set(args.cases) - cases.keys():
        p.error('Case names must be unique and present in dataset.json')
    if any(a.split('=')[0] in ('--family', '--n', '--input-seed', '--output') for a in extra):
        p.error('Input identity and output are determined by the case manifest')
    for name in args.cases:
        case = cases[name]
        cmd = [sys.executable, str(ROOT / 'pilot.py'), args.mode, '--output',
               str(Path(args.output) / name), '--family', case['family'],
               '--n', str(case['n']), '--input-seed', str(case.get('seed', 0)), *extra]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
