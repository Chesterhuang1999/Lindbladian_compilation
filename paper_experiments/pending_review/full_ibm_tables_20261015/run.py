"""Frozen IBM backend completion, raw opt3, and two seed-averaged tables."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PENDING=ROOT/'paper_experiments/pending_review'
SNAP=OUT/'source_snapshot'
IBM=['u1','u2','u3','cx']
NAM=['rz','h','x','cx']
METHODS=['Raw','Raw+Qiskit opt3','Ours']

def read(p): return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.writing');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def slug(method): return {'Raw':'raw','Ours':'ours','Raw+Qiskit opt3':'raw_opt3'}[method]
def saved(case,method): return OUT/'cases'/case/slug(method)/'results.json'
def rows_from_array(A,labels):
    return [[[str(p),float(abs(z)),float((z/abs(z)).real),float((z/abs(z)).imag)]
             for p,z in zip(labels,r) if abs(z)>1e-12] for r in A if any(abs(z)>1e-12 for z in r)]

def prepare():
    import numpy as np
    if (OUT/'manifest.json').exists(): raise RuntimeError('Refuse to overwrite frozen batch')
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/eval_redesign')]
    from run_selected_tfim_hrw import sparse_gram_check
    config=read(ROOT/'paper_experiments/configs/full_channel_suite_20261011.json')
    csv_rows=[]
    for name in ['rq1_raw_optimizer_20261014.csv','rq1_scaling_20261014.csv']:
        csv_rows+=list(csv.DictReader((ROOT/'Results'/name).open()))
    raw_index={}
    for row in csv_rows:
        if row['method']=='Raw' and (row['case'] not in raw_index or row['status']=='ok'):
            raw_index[row['case']]=row
    selected=[]
    for group in config['cases']:
        for seed in group.get('parameters',{}).get('seeds',[None]):
            case=group['id']+(f'-seed{seed}' if seed is not None else '')
            selected.append((case,group))
    assert len(selected)==52
    reused=[r for r in csv.DictReader((ROOT/'Results/rq1_resource_layout_ibm_20261014_v2.csv').open())
            if r['status']=='ok' and r['method'] in ('Raw','Raw+Qiskit opt3')]
    entries={};jobs=[]
    for case,group in selected:
        history=PENDING/'tfim_hrw_full_20261011/cases'/case
        if not (history/'input.json').exists():history=PENDING/'remaining_full_20261011/cases'/case
        if case=='HRW-8':history=ROOT/'Baseline_results/table1_scaling_ours_20261013/cases/HRW-8'
        random=case.startswith('Random-5K-')
        if random:history=PENDING/'rq2_hybrid_ablation_20261014/cases'/case
        inp=read(history/'input.json');target=inp['raw_kraus'] if random else inp['input_kraus']
        hybrid=PENDING/'support_gatecost_hybrid_20261012/cases'/case
        if case=='TFIM2-HO':
            source=PENDING/'tfim2_ho_streaming_gatecost_20261015/cases/TFIM2-HO/results.json'
            old=read(source);encoded=old['compiled_kraus'];rewrite_s=old['rewrite_time_s'];canon_s=old['canonicalization_s'];policy=old['rewrite']['method']
        elif (hybrid/'hybrid_circuit.json').exists():
            source=hybrid/'hybrid_circuit.json';old=read(source)
            with np.load(hybrid/'hybrid.npz') as a:encoded=rows_from_array(a['A_final'],a['labels'])
            rew=read(hybrid/'results.json')['methods']['hybrid'];assert rew['status']=='ok'
            rewrite_s=rew['wall_s'];canon_s=None;policy={'method':'support_primary_gatecost_polish','source':str(hybrid/'results.json')}
        elif random:
            source=history/'configs/All/results.json';old=read(source);rew=read(history/'rewrites/tree.json')
            encoded=rew['compiled_kraus'];rewrite_s=old['rewrite_s'];canon_s=old['canonicalization_s'];policy=old['settings']['rewrite']
        else:
            source=history/'results.json';old=read(source);assert old['status']=='ok'
            encoded=old['assembly']['compiled_kraus'];rewrite_s=old['exclusive_stage_times_s']['rewrite']
            canon_s=old['exclusive_stage_times_s']['canonicalization'];policy=old['effective_rewrite_options']
        check=sparse_gram_check(target,encoded);assert check['passed'],(case,check)
        ours={'encoded':encoded,'Lambda':old['Lambda'],'NAM_G':old['G'],'W':old['W'],
              'rewrite_s':rewrite_s,'historical_canonicalization_s':canon_s,'rewrite_policy':policy,
              'source':str(source),'source_sha256':digest(source),'map_check':check,
              'matrix_order_options':old.get('assembly',{}).get('matrix_order_options',{}),
              'historical_compile_s':old.get('compile_time_s',old.get('compile_s'))}
        raw_row=raw_index.get(case)
        if random:
            rp=history/'configs/Basic/results.json';rd=read(rp)
            raw={'status':'ok','encoded':inp['canonical_kraus'],'Lambda':rd['Lambda'],'NAM_G':rd['G'],'W':rd['W'],
                 'canonicalization_s':rd['canonicalization_s'],'historical_assembly_s':rd['assembly_s'],
                 'source':str(rp),'source_sha256':digest(rp)}
        elif raw_row and raw_row['status']=='ok':
            rp=Path(raw_row['source_result_path']);rd=read(rp)
            raw={'status':'ok','encoded':rd['canonical_kraus'],'Lambda':rd['Lambda'],'NAM_G':rd['metrics']['G'],
                 'W':rd['metrics']['W'],'canonicalization_s':rd['canonicalization']['time_s'],
                 'historical_assembly_s':rd['assembly']['assembly_time_s'],'source':str(rp),'source_sha256':digest(rp)}
        else:
            raw={'status':raw_row['status'] if raw_row else 'not_available',
                 'Lambda':float(raw_row['Lambda']) if raw_row else sum(sum(t[1] for t in r)**2 for r in target),
                 'source':raw_row['source_result_path'] if raw_row else None}
        if raw['status']=='ok':assert sparse_gram_check(target,raw['encoded'])['passed']
        entry={'case':case,'group':group['id'],'n':group['system_qubits'],'target':target,
               'target_source':str(history/'input.json'),'target_source_sha256':digest(history/'input.json'),
               'Ours':ours,'Raw':raw}
        dump(OUT/'inputs'/f'{case}.json',entry);entries[case]={'group':group['id'],'n':group['system_qubits'],
              'input_sha256':digest(OUT/'inputs'/f'{case}.json'),'raw_status':raw['status'],
              'raw_Lambda':raw['Lambda'],'ours_Lambda':ours['Lambda']}
        jobs.append({'case':case,'kind':'ours','host':'local','priority':1000 if case=='Toom-L6' else ours['NAM_G']})
        if raw['status']!='ok':continue
        prior={r['method']:r for r in reused if r['case']==case}
        if set(prior)=={'Raw','Raw+Qiskit opt3'}:
            for method,r in prior.items():
                assert math.isclose(float(r['Lambda']),raw['Lambda'],rel_tol=1e-12)
                assert int(r['W'])==raw['W']
                assert digest(r['qasm_path'])==r['qasm_sha256']
                target_doc=read(r['target_path']);prior_target=target_doc.get('input_kraus',target_doc.get('raw_kraus'))
                assert sparse_gram_check(target,prior_target)['passed']
                d=json.loads(r['source_record']);raw_d=json.loads(prior['Raw']['source_record'])
                base_time=raw['canonicalization_s']+raw['historical_assembly_s']+raw_d['lowering_s']
                opt=d.get('optimizer_s',0) if method!='Raw' else 0
                dump(saved(case,method),{'case':case,'method':method,'status':'ok','basis':'IBM','G':int(r['G']),
                    'W':int(r['W']),'Lambda':float(r['Lambda']),'compile_time_s':base_time+opt,
                    'backend_time_s':raw_d['lowering_s']+opt,'optimizer_s':opt,'reused':True,
                    'time_scope':'sum of archived canonicalization/assembly + measured IBM opt0 lowering + optional opt3; not a fresh end-to-end run',
                    'host':'local','source':r['source_result_path'],'source_record':d,
                    'qasm_path':r['qasm_path'],'qasm_sha256':r['qasm_sha256']})
        else:jobs.append({'case':case,'kind':'raw','host':'remote','priority':raw['NAM_G']})
    sources={}
    for folder in ('src','scripts','Baseline_scripts'):
        for p in sorted((ROOT/folder).rglob('*.py')):
            rel=p.relative_to(ROOT);dest=SNAP/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);sources[str(rel)]=digest(dest)
    dump(OUT/'manifest.json',{'cases':entries,'groups':config['cases'],'partitions':config['benchmark_partitions'],
         'jobs':jobs,'sources':sources,'runner_sha256':digest(__file__),'basis_gates':IBM,'seed':20261009,
         'policy':'Raw and Ours IBM opt0; Raw+Qiskit exact whole-circuit IBM opt3. Reuse saved rewrite; no failed Raw rebuild.',
         'budgets':{'local_rss_gib':6,'remote_rss_gib':32,'assembly_lowering_s':3600,'optimizer_s':3600},
         'aggregation':'arithmetic mean separately for Lambda/G/W/time over all five seeds; never average only successful seeds',
         'time_policy':'historical rewrite plus current canonicalization/assembly/IBM lowering; excludes imports, verification and I/O. Archived reused rows explicitly marked.'})
    result=subprocess.run([sys.executable,'-B',str(ROOT/'Baseline_scripts/smoke_block_encoding_archive.py'),'--out-dir',str(OUT)])
    assert result.returncode==0 and read(OUT/'smoke.json')['passed']
    assert all(digest(ROOT/rel)==h for rel,h in sources.items())
    print('Prepared',len(entries),'cases;',Counter(j['host'] for j in jobs),'new jobs',flush=True)
    collect()

def worker(case,kind,host):
    sys.path[:0]=[str(SNAP/p) for p in ('src','scripts/eval_redesign','Baseline_scripts')]
    from copy import deepcopy
    import qiskit
    from qiskit import qasm2,QuantumCircuit
    from block_encoding import BlockEncoding
    from channel_IR import channel_ensemble
    from channel_LCU import channel_to_LCU
    from rq1_pipeline import canonicalize,deserialize_kraus
    from experiment_protocol import source_identity,transpile_channel_circuit
    m=read(OUT/'manifest.json');entry_path=OUT/'inputs'/f'{case}.json';entry=read(entry_path)
    assert digest(entry_path)==m['cases'][case]['input_sha256']
    assert read(OUT/'smoke.json')['sources']==source_identity()
    info=entry['Ours' if kind=='ours' else 'Raw'];method='Ours' if kind=='ours' else 'Raw'
    directory=OUT/'cases'/case/kind;directory.mkdir(parents=True,exist_ok=True)
    def progress(stage):dump(directory/'progress.json',{'stage':stage,'updated':time.time(),'pid':os.getpid()})
    def lower(circuit,level,basis):return transpile_channel_circuit(circuit,basis_gates=basis,optimization_level=level,
                       seed_transpiler=20261009,approximation_degree=1.0,qubits_initially_zero=False,num_processes=1)
    def save(circuit,which,assembly_s,lowering_s,canonical_s,opt_s=0):
        counts={str(k):int(v) for k,v in circuit.count_ops().items()};assert not set(counts)-set(IBM)-{'reset','barrier'}
        if circuit.layout is not None:assert circuit.layout.final_index_layout()==list(range(circuit.num_qubits))
        assert circuit.num_qubits==info['W']
        flat=QuantumCircuit(circuit.num_qubits);flat.compose(circuit,inplace=True)
        path=saved(case,which).parent/'channel_ibm.qasm';path.parent.mkdir(parents=True,exist_ok=True)
        qasm2.dump(flat,path)
        recount=Counter();width=0
        import re
        with path.open() as stream:
            for line in stream:
                if line.startswith('qreg '):width+=int(re.search(r'\[(\d+)\]',line)[1])
                op=re.match(r'^(u1|u2|u3|cx|reset|barrier)(?:\(|\s)',line)
                if op:recount[op[1]]+=1
        assert dict(recount)==counts and width==circuit.num_qubits
        rewrite_s=info.get('rewrite_s',0)
        result={'case':case,'method':which,'status':'ok','basis':'IBM','host':host,'G':sum(counts.get(k,0) for k in IBM),
                'W':width,'Lambda':info['Lambda'],'counts':counts,'compile_time_s':canonical_s+rewrite_s+assembly_s+lowering_s+opt_s,
                'rewrite_s':rewrite_s,'canonicalization_s':canonical_s,'assembly_s':assembly_s,'lowering_s':lowering_s,
                'optimizer_s':opt_s,'backend_time_s':assembly_s+lowering_s+opt_s,
                'time_scope':'saved local rewrite timing + newly measured canonicalization and circuit assembly/IBM lowering/opt3; no import, NAM reproduction, verification or I/O',
                'timing_host_note':'rewrite from local archive; backend on '+host,'rewrite_policy':info.get('rewrite_policy'),
                'input_sha256':m['cases'][case]['input_sha256'],'original_source':info['source'],'original_source_sha256':info['source_sha256'],
                'NAM_reproduction':'passed G and W','NAM_G':info['NAM_G'],
                'qubits_initially_zero':False,'optimization_level':3 if opt_s else 0,'seed_transpiler':20261009,
                'full_choi':{'status':'not_run','scope':'basis conversion/optimizer; input Gram, source hashes, NAM reproduction, output gate set/count/width checked'},
                'qasm_path':str(path),'qasm_sha256':digest(path),'qiskit_version':qiskit.__version__}
        dump(saved(case,which),result)
    progress('assembly')
    start=time.perf_counter();_,canonical_info=canonicalize(deserialize_kraus(entry['target']));canon_s=time.perf_counter()-start
    start=time.perf_counter()
    if kind=='ours':
        backend=lambda ms:BlockEncoding(ms,placement_options={},matrix_order_options=info['matrix_order_options'])
        circuit,registers=channel_to_LCU(channel_ensemble([deserialize_kraus(info['encoded'])]),structure='tree',
                        opt='Matrix-order',fstar=True,block_encoding_cls=backend)
    else:
        circuit,registers=channel_to_LCU(channel_ensemble([deserialize_kraus(info['encoded'])]),structure='basic',opt='No',fstar=False)
    assembly_s=time.perf_counter()-start
    assert math.isclose(circuit.metadata['subnormalization_factor'],info['Lambda'],rel_tol=1e-10)
    progress('nam_reproduction')
    nam=lower(circuit,0,NAM)
    assert sum(nam.count_ops().get(k,0) for k in NAM)==info['NAM_G'],(case,method,dict(nam.count_ops()),info['NAM_G'])
    assert nam.num_qubits==info['W'];del nam
    progress('ibm_lowering');start=time.perf_counter();converted=lower(circuit,0,IBM);lower_s=time.perf_counter()-start;del circuit
    progress('export_raw' if kind=='raw' else 'export_ours');save(converted,method,assembly_s,lower_s,canon_s)
    if kind=='raw':
        progress('qiskit_opt3');start=time.perf_counter();optimized=lower(converted,3,IBM);opt_s=time.perf_counter()-start;del converted
        progress('export_opt3');save(optimized,'Raw+Qiskit opt3',assembly_s,lower_s,canon_s,opt_s)
    progress('done')

def coordinate(host):
    import psutil
    m=read(OUT/'manifest.json');assert digest(__file__)==m['runner_sha256']
    for p,h in m['sources'].items():assert digest(SNAP/p)==h
    status_path=OUT/f'{host}_status.json'
    if status_path.exists():raise RuntimeError('Refusing duplicate coordinator')
    jobs=sorted([dict(j,status='queued') for j in m['jobs'] if j['host']==host],key=lambda j:j['priority'])
    def checkpoint():dump(status_path,{'status':'running','host':host,'pid':os.getpid(),'updated':time.time(),'jobs':jobs})
    checkpoint()
    for j in jobs:
        directory=OUT/'cases'/j['case']/j['kind'];directory.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();peak=0;stop=None;stage='starting';stage_start=start
        with (directory/'run.log').open('w') as log:
            p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',j['case'],'--kind',j['kind'],'--host',host],
                               stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            j.update(status='running',worker_pid=p.pid);checkpoint()
            while p.poll() is None:
                try:
                    proc=psutil.Process(p.pid);rss=sum(x.memory_info().rss for x in [proc]+proc.children(recursive=True));peak=max(peak,rss)
                except psutil.Error:pass
                progress_path=directory/'progress.json';current=read(progress_path).get('stage',stage) if progress_path.exists() else stage
                if current!=stage:stage,stage_start=current,time.monotonic()
                limit=(6 if host=='local' else 32)*2**30
                if peak>limit:stop='memory_limit'
                if time.monotonic()-stage_start>3600:stop='timeout'
                if time.monotonic()-start>10800:stop='worker_timeout'
                j.update(stage=stage,elapsed_s=time.monotonic()-start,peak_rss_bytes=peak);checkpoint()
                if stop:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                    break
                time.sleep(1)
            code=p.wait()
        j.update(status=stop or ('ok' if code==0 else 'failed'),exit_code=code,elapsed_s=time.monotonic()-start,resource_stop=stop)
        if j['status']=='ok':
            required=['Ours'] if j['kind']=='ours' else ['Raw','Raw+Qiskit opt3']
            if any(not saved(j['case'],method).exists() for method in required):j['status']='missing_results'
        dump(directory/'resources.json',j);checkpoint()
    dump(status_path,{'status':'complete' if all(j['status']=='ok' for j in jobs) else 'complete_with_issues',
                     'host':host,'updated':time.time(),'jobs':jobs})

def collect():
    m=read(OUT/'manifest.json');records=[];states={}
    for host in ('local','remote'):
        p=OUT/f'{host}_status.json'
        if p.exists():
            for j in read(p)['jobs']:states[j['case'],j['kind']]=j
    for case,e in m['cases'].items():
        for method in METHODS:
            p=saved(case,method)
            if p.exists():r=read(p)
            else:
                kind='ours' if method=='Ours' else 'raw';job=states.get((case,kind),{})
                status=job.get('status','queued')
                if kind=='raw' and e['raw_status']!='ok':status='no_raw_'+e['raw_status']
                r={'case':case,'method':method,'status':status,'basis':'IBM',
                   'Lambda':e['ours_Lambda'] if method=='Ours' else e['raw_Lambda'],
                   'failure_stage':job.get('stage'),'resource_stop':job.get('resource_stop')}
            records.append({**r,'group':e['group']})
    dump(OUT/'results.json',{'rows':records,'updated':time.time()})
    results_dir=ROOT/'Results';results_dir.mkdir(exist_ok=True)
    fields=['case','group','method','status','basis','Lambda','G','W','compile_time_s','backend_time_s','optimizer_s','host','time_scope','qasm_path','qasm_sha256','failure_stage']
    with (results_dir/'full_ibm_per_seed_20261015.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    for part in m['partitions']:
        table=[]
        for group in part['case_group_ids']:
            g=next(g for g in m['groups'] if g['id']==group);n=len(g.get('parameters',{}).get('seeds',[0]))
            row={'case':group,'system_qubits':g['system_qubits'],'samples':n}
            for method in METHODS:
                subset=[r for r in records if r['group']==group and r['method']==method];assert len(subset)==n
                successful=[r for r in subset if r['status']=='ok'];prefix=slug(method)
                row[prefix+'_completed']=len(successful);row[prefix+'_status']='ok' if len(successful)==n else ';'.join(sorted({r['status'] for r in subset if r['status']!='ok'}))
                row[prefix+'_Lambda']=statistics.mean(r['Lambda'] for r in subset)
                for metric in ('G','W','compile_time_s','backend_time_s'):
                    row[prefix+'_'+metric]=statistics.mean(r[metric] for r in subset) if len(successful)==n and all(r.get(metric) is not None for r in subset) else None
                    row[prefix+'_'+metric+'_sd']=statistics.stdev(r[metric] for r in subset) if n>1 and row[prefix+'_'+metric] is not None else None
            table.append(row)
        assert len(table)==20
        path=results_dir/('benchmark_'+part['id']+'_ibm_20261015.csv')
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,list(table[0]));w.writeheader();w.writerows(table)
        lines=['# '+part['label']+' — IBM','',
          '门集：{u1,u2,u3,cx}。Raw/Ours 为 opt0；Raw+Qiskit 为完整 IBM 电路 opt3。reset 不计入 G。',
          '随机家族各含 seed 0–4；Λ、G、W、时间分别作算术平均，只有五个相应结果齐全才显示 G/W/时间。CSV 同时保留标准差；缺失不按零计。',
          'T 为秒。Ours 的 T 含历史 rewrite 时间＋本次 canonicalization/装配/IBM 分解；复用行的 T 是已记录阶段之和。Raw+opt3 的 T 含 Raw 编译＋opt3。不是统一主机上重新计时的端到端测量；逐例 CSV 保留主机与范围。',
          'Raw 与 Raw+opt3 共享 Λ；W 仅在实际输出验证后记录。状态为当前批次快照；原 Raw 失败不补跑。','',
          '| Case | n | Λ Raw | G Raw | W Raw | T Raw | G Raw+opt3 | W Raw+opt3 | T Raw+opt3 | Λ Ours | G Ours | W Ours | T Ours |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        def val(r,p,k):
            x=r[p+'_'+k]
            if x is None:return f"{r[p+'_status']} ({r[p+'_completed']}/{r['samples']})" if k=='G' else '—'
            return f'{x:,.0f}' if k in ('G','W') and x==int(x) else f'{x:,.3f}' if k not in ('Lambda',) else f'{x:.6g}'
        for r in table:
            cells=[r['case']+(' (mean of 5)' if r['samples']==5 else ''),str(r['system_qubits'])]
            for p,ks in [('raw',['Lambda','G','W','compile_time_s']),('raw_opt3',['G','W','compile_time_s']),('ours',['Lambda','G','W','compile_time_s'])]:
                cells.extend(val(r,p,k) for k in ks)
            lines.append('| '+' | '.join(cells)+' |')
        path.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('Collected',Counter((r['method'],r['status']) for r in records),flush=True)

def watch():
    remote='huangqf@192.168.20.20:/pub/data/huangqf/Lindbladian_compilation/paper_experiments/pending_review/full_ibm_tables_20261015/'
    while True:
        result=subprocess.run(['rsync','-a','--exclude=source_snapshot','--exclude=.venv','--exclude=.runtime','--exclude=inputs',
             '--exclude=manifest.json','--exclude=run.py','--exclude=smoke*','--exclude=local_status.json',
             '--exclude=*.qasm','--exclude=*.qpy','--exclude=local*','--exclude=watch*',remote,str(OUT)+'/'],timeout=60)
        dump(OUT/'sync_status.json',{'returncode':result.returncode,'updated':time.time()})
        collect()
        if all((OUT/f'{h}_status.json').exists() and read(OUT/f'{h}_status.json')['status'].startswith('complete') for h in ['local','remote']):break
        time.sleep(30)

if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):os.environ[key]='1'
    os.environ['QISKIT_PARALLEL']='FALSE';os.environ['MPLCONFIGDIR']=str(OUT/'.runtime/mpl');os.environ['PYTHONDONTWRITEBYTECODE']='1'
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--collect',action='store_true');parser.add_argument('--watch',action='store_true');parser.add_argument('--worker');parser.add_argument('--kind',choices=['raw','ours']);parser.add_argument('--host',choices=['local','remote'],default='local');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.collect:collect()
    elif args.watch:watch()
    elif args.worker:
        try:worker(args.worker,args.kind,args.host)
        except Exception:
            dump(OUT/'cases'/args.worker/args.kind/'failure.json',{'traceback':traceback.format_exc()});raise
    else:coordinate(args.host)
