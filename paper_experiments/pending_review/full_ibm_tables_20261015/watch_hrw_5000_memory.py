"""Record memory use while the authorized HRW queue runs; never dispatch jobs."""
import json
import os
from pathlib import Path
import time

import psutil
import run as batch

BASE = Path(__file__).resolve().parent
STATUS = BASE/'remote_hrw_5000_status.json'
SNAPSHOT = BASE/'remote_hrw_5000_memory.json'
SAMPLES = BASE/'remote_hrw_5000_memory.jsonl'


def sample():
    state = batch.read(STATUS)
    jobs = []
    for job in state['jobs']:
        row = {k:job.get(k) for k in ('case','mode','status','stage','worker_pid',
                                      'elapsed_s','peak_rss_bytes','worker_rss_gib')}
        row.update(rss_bytes=0, cpu_s=0, thread_count=0)
        if job['status'] == 'running':
            try:
                root = psutil.Process(job['worker_pid'])
                # Check identity so an exited worker's reused PID is never sampled.
                assert 'remote_hrw_5000.py' in ' '.join(root.cmdline())
                assert abs(root.create_time()-job['start_wall']) < 3
                for proc in [root]+root.children(recursive=True):
                    try:
                        row['rss_bytes'] += proc.memory_info().rss
                        row['cpu_s'] += sum(proc.cpu_times()[:2])
                        row['thread_count'] += proc.num_threads()
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
        jobs.append(row)
    memory = psutil.virtual_memory()
    return dict(updated=time.time(), observer_pid=os.getpid(),
        queue_status=state['status'], coordinator_pid=state['pid'],
        coordinator_alive=psutil.pid_exists(state['pid']),
        coordinator_heartbeat_age_s=time.time()-state['updated'],
        total_bytes=memory.total, available_bytes=memory.available,
        measured_worker_rss_bytes=sum(j['rss_bytes'] for j in jobs), jobs=jobs)


def main():
    lock = BASE/'remote_hrw_5000_memory.lock'
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    while True:
        data = sample()
        batch.dump(SNAPSHOT, data)
        with SAMPLES.open('a') as stream:
            stream.write(json.dumps(data, allow_nan=False)+'\n')
        if data['queue_status'] not in ('running','waiting_for_memory'):
            break
        if not data['coordinator_alive'] or data['coordinator_heartbeat_age_s'] > 120:
            # Preserve evidence of a stalled coordinator for the follow-up.
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
