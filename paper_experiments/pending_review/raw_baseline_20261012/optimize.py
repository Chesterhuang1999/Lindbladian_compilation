"""Run the existing NAM optimizer adapters with bounded tool processes."""
from pathlib import Path
import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE / 'source_snapshot/Baseline_scripts')]
from run import dump, digest, MEMORY_BYTES, PYTHON
EXTERNAL_PYTHON = Path('/Users/chester_hqf/miniconda3/envs/ChannelIR_test/bin/python')


def bounded_command(command, timeout=None, cwd=None):
    import psutil
    directory = Path(os.environ['RAW_OPT_JOB'])
    start = time.monotonic()
    peak, stop = 0, None
    with (directory / 'tool.stdout').open('w') as stdout, (directory / 'tool.stderr').open('w') as stderr:
        # Inherit the worker's process group so a parent resource stop covers
        # the optimizer as well as the adapter.
        child = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, text=True)
        dump(directory / 'tool_process.json', {'pid': child.pid, 'command': command})
        while child.poll() is None:
            try:
                peak = max(peak, psutil.Process(child.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, OSError):
                stop = 'resource_monitor_failed'
            if peak > MEMORY_BYTES:
                stop = 'memory_limit'
            elif timeout is not None and time.monotonic() - start > timeout:
                stop = 'timeout'
            if stop:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(1)
        code = child.wait()
    resource = {'returncode': code, 'elapsed_seconds': time.monotonic()-start,
                'peak_rss_bytes': peak, 'resource_stop': stop}
    dump(directory / 'tool_resources.json', resource)
    return {**resource, 'timed_out': stop == 'timeout',
            'stdout': (directory / 'tool.stdout').read_text(errors='replace'),
            'stderr': (directory / 'tool.stderr').read_text(errors='replace')}


def worker(args):
    directory = HERE / 'optimizers' / args.case / args.tool
    directory.mkdir(parents=True, exist_ok=True)
    os.environ['RAW_OPT_JOB'] = str(directory)
    os.environ['TMPDIR'] = str(HERE / '.runtime')
    adapter = importlib.import_module('run_' + args.tool + '_baseline')
    adapter.run_command = bounded_command
    parser = adapter.build_parser()
    argv = ['--input', args.input, '--output', str(directory / 'output.qasm'),
            '--summary', str(directory / 'adapter_result.json'), '--timeout', '3600']
    if args.tool == 'voqc':
        binary = ROOT / 'external_tools/VOQC/VOQC/_build/default/voqc.exe'
        argv += ['--voqc', str(binary)]
    else:
        binary = ROOT / 'external_tools/feynman/dist-newstyle/build/aarch64-osx/ghc-9.14.1/Feynman-0.1.0.0/x/feynopt/build/feynopt/feynopt'
        argv += ['--feynopt', str(binary)]
    if not binary.exists():
        raise FileNotFoundError(binary)
    result = adapter.run(parser.parse_args(argv))
    process = result['process']
    resource_path = directory / 'tool_resources.json'
    resource = json.loads(resource_path.read_text()) if resource_path.exists() else {}
    output = result.get('output_stats')
    success = (process['returncode'] == 0 and not resource.get('resource_stop')
               and output is not None and set(output['ops']).issubset({'rz', 'h', 'x', 'cx', 'reset'}))
    dump(directory / 'results.json', {'case': args.case, 'tool': args.tool,
        'status': 'ok' if success else resource.get('resource_stop') or 'failed',
        'Lambda': args.lam, 'input_G': result['input_stats']['metric_total'],
        'G': output['metric_total'] if success else None,
        'input_sha256': digest(Path(args.input)), 'tool_sha256': digest(binary),
        'tool_resources': resource, 'adapter_result': str(directory / 'adapter_result.json'),
        'validation': 'NAM output recount; no additional channel-equivalence simulation of optimizer output'})


def queue():
    import psutil
    if (HERE / 'optimizer_status.json').exists():
        raise RuntimeError('Optimizer queue already started.')
    raw = json.loads((HERE / 'results.json').read_text())
    ready = sorted((r for r in raw if r['status'] in ('ok', 'reused')), key=lambda r: r['G'])
    state = {'status': 'running', 'pid': os.getpid(), 'completed': [],
             'not_run_raw_unavailable': [r['case'] for r in raw if r['status'] not in ('ok', 'reused')]}
    for row in ready:
        for tool in ('voqc', 'feynman'):
            directory = HERE / 'optimizers' / row['case'] / tool
            directory.mkdir(parents=True, exist_ok=True)
            existing = directory / 'results.json'
            if existing.exists():
                cached = json.loads(existing.read_text())
                if cached.get('status') == 'ok' and cached.get('input_sha256') == row['qasm_sha256']:
                    state['completed'].append({**cached, 'reused_preflight': True})
                    dump(HERE / 'optimizer_status.json', state)
                    continue
                raise RuntimeError(f'Existing optimizer output is not reusable: {existing}')
            command = [str(EXTERNAL_PYTHON), '-B', str(Path(__file__)), 'worker', '--case', row['case'],
                       '--tool', tool, '--input', row['qasm'], '--lam', str(row['Lambda'])]
            start = time.monotonic()
            peak, stop = 0, None
            with (directory / 'run.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                state['current'] = {'case': row['case'], 'tool': tool, 'pid': child.pid}
                dump(HERE / 'optimizer_status.json', state)
                while child.poll() is None:
                    try:
                        peak = max(peak, psutil.Process(child.pid).memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    except (psutil.Error, OSError):
                        stop = 'resource_monitor_failed'
                    if peak > MEMORY_BYTES:
                        stop = 'memory_limit'
                    elif time.monotonic() - start > 3900:
                        stop = 'adapter_timeout'
                    if stop:
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        break
                    time.sleep(1)
                code = child.wait()
            f = directory / 'results.json'
            record = json.loads(f.read_text()) if f.exists() else {'case': row['case'], 'tool': tool}
            if code != 0 or stop or not f.exists():
                record.update(status=stop or 'failed', G=None)
            record.update(exit_code=code, adapter_peak_rss_bytes=peak, wall_s=time.monotonic()-start)
            dump(directory / 'job.json', record)
            state['completed'].append(record)
            state['current'] = None
            dump(HERE / 'optimizer_status.json', state)
    state['status'] = 'finished'
    dump(HERE / 'optimizer_status.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('worker', 'queue'))
    p.add_argument('--case')
    p.add_argument('--tool', choices=('voqc', 'feynman'))
    p.add_argument('--input')
    p.add_argument('--lam', type=float)
    args = p.parse_args()
    worker(args) if args.mode == 'worker' else queue()
