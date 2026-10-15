"""Mirror metadata and refresh requested tables until both queues terminate."""
import json
import subprocess
import sys
import time
from pathlib import Path
BASE=Path(__file__).resolve().parent
REMOTE='huangqf@192.168.20.20:/pub/data/huangqf/Lindbladian_compilation/paper_experiments/pending_review/full_ibm_tables_20261015/'
def read(path):return json.loads(path.read_text())
def write(data):
    p=BASE/'watch_status.json';t=p.with_suffix('.writing');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
while True:
    try:
        cmd=['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=10']
        for exclusion in ['source_snapshot','.venv','.runtime','inputs','manifest.json','run.py','smoke*','local*','watch*','*.qasm','*.qpy','results.json','preflight_local.json']:
            # Result records inside cases must be mirrored, unlike root results.json.
            if exclusion=='results.json':exclusion='/results.json'
            cmd.append('--exclude='+exclusion)
        p=subprocess.run(cmd+[REMOTE,str(BASE)+'/'],timeout=60)
        state={'status':'running','sync_returncode':p.returncode,'updated':time.time()}
    except subprocess.TimeoutExpired:
        state={'status':'running','sync_error':'timeout; will retry','updated':time.time()}
    result=subprocess.run([sys.executable,'-B',str(BASE/'collect_dual_basis.py')],timeout=120)
    state['collect_returncode']=result.returncode
    terminated=all((BASE/f'{h}_status.json').exists() and read(BASE/f'{h}_status.json')['status'].startswith('complete') for h in ('local','remote'))
    for name in ('remote_ours_status.json', 'remote_hrw_status.json', 'remote_opt3_retry_status.json', 'remote_toom_opt3_retry_status.json', 'remote_ours_timing_status.json', 'remote_tfim40_timing_status.json', 'remote_hrw_full_status.json'):
        if (BASE/name).exists():
            terminated = terminated and read(BASE/name)['status'].startswith('complete')
    if terminated and result.returncode==0 and state.get('sync_returncode')==0:
        state['status']='finished';write(state);break
    write(state)
    time.sleep(30)
