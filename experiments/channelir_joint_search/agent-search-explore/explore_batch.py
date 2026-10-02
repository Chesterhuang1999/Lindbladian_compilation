#!/usr/bin/env python3
"""Sequential, bounded diagnostic batch with durable status and error stops."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def main():
    batch = Path(sys.argv[1]).resolve()
    plan = json.loads((batch / 'plan.json').read_text())
    status = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'running',
              'completed': [], 'active': None, 'cpu_seconds': 0.}
    status_path = batch / 'status.json'
    save(status_path, status)
    try:
        for case in plan['cases']:
            # Reserve the whole next session budget before starting it.
            if status['cpu_seconds'] + plan['cpu_per_case'] > plan['batch_cpu_limit']:
                status['status'] = 'batch_budget_stop'
                break
            status['active'] = case
            save(status_path, status)
            cmd = [sys.executable, str(ROOT / 'run_cases.py'), 'diagnose',
                   '--cases', case, '--output', str(batch / 'diagnostics'), *plan['arguments']]
            with (batch / (case + '.log')).open('w') as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=plan['wall_per_case'] + 60)
            folder = batch / 'diagnostics' / case
            state_file = folder / 'diagnostic' / 'state.json'
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            cpu = state.get('used', {}).get('cpu_seconds', 0.)
            errors = [{'operation': e['operation'], 'response': e['response']}
                      for e in state.get('events', []) if e['response'].get('status') == 'error']
            row = {'case': case, 'exit_code': result.returncode, 'cpu_seconds': cpu,
                   'evaluations': len(state.get('evaluations', {})), 'errors': errors,
                   'finished': state.get('finished', False)}
            status['cpu_seconds'] += cpu
            status['completed'].append(row)
            save(status_path, status)
            if result.returncode or errors or not row['evaluations']:
                status['status'] = 'needs_review'
                break
        else:
            status['status'] = 'completed'
    except Exception as error:
        status.update(status='needs_review', error=repr(error))
        raise
    finally:
        status['active'] = None
        status['updated_at'] = time.time()
        save(status_path, status)
    if status['completed']:
        subprocess.run([sys.executable, str(ROOT / 'analyze_results.py'),
                        str(batch / 'diagnostics')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
