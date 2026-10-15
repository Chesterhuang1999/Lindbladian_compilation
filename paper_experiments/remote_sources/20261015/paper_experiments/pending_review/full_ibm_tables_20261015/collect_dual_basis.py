"""Add audited Ours NAM metrics to the IBM tables without changing workers."""
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
import run as batch


def write_csv(path, rows):
    temporary = path.with_suffix('.writing')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ours_termination_issue(result_path):
    resources = Path(result_path).parent / 'resources.json'
    if not resources.exists():
        return 'awaiting_process_completion'
    record = json.loads(resources.read_text())
    if record.get('resource_stop'):
        return record['resource_stop']
    if record.get('exit_code') not in (0, None):
        return 'failed'
    if record.get('status') != 'ok':
        return record.get('status', 'failed')
    return None


def collect():
    sys.path.insert(0, str(batch.ROOT / 'scripts/eval_redesign'))
    from render_full_benchmark_tables import display_case, render
    manifest = batch.read(batch.OUT / 'manifest.json')
    original_saved = batch.saved
    def saved_with_timing_fallback(case, method):
        current = original_saved(case, method)
        archive = current.parent.parent / 'ours_local_before_remote_timing' / 'results.json'
        if method == 'Ours' and archive.exists():
            if not current.exists() or ours_termination_issue(current):
                return archive
        return current
    batch.saved = saved_with_timing_fallback
    nam = {}
    for case, entry in manifest['cases'].items():
        path = batch.OUT / 'inputs' / f'{case}.json'
        assert batch.digest(path) == entry['input_sha256'], case
        info = batch.read(path)['Ours']
        source = Path(info['source'])
        assert batch.digest(source) == info['source_sha256'], case
        archive = batch.read(source)
        assert archive.get('status', 'ok') == 'ok', case
        assert archive['G'] == info['NAM_G'] and archive['W'] == info['W'], case
        assert math.isclose(archive['Lambda'], info['Lambda'], rel_tol=1e-12), case
        assert info['map_check']['passed'], case
        if 'counts' in archive:
            assert not set(archive['counts']) - set(batch.NAM) - {'reset', 'barrier'}, case
            assert sum(archive['counts'].get(g, 0) for g in batch.NAM) == info['NAM_G'], case
        current = batch.saved(case, 'Ours')
        state = 'archived; current reproduction pending'
        if current.exists():
            result = batch.read(current)
            assert result['status'] == 'ok' and result['basis'] == 'IBM', case
            assert result['input_sha256'] == entry['input_sha256'], case
            assert result['original_source_sha256'] == info['source_sha256'], case
            assert result['NAM_reproduction'] == 'passed G and W', case
            assert result['NAM_G'] == info['NAM_G'] and result['W'] == info['W'], case
            assert math.isclose(result['Lambda'], info['Lambda'], rel_tol=1e-12), case
            issue = ours_termination_issue(current)
            state = ('archived; current process ' + issue if issue
                     else 'reproduced from same high-level circuit as IBM')
        nam[case] = {'G_NAM': info['NAM_G'], 'NAM_W': info['W'],
                     'NAM_status': state, 'NAM_source': str(source),
                     'NAM_source_sha256': info['source_sha256']}
    original_read = batch.read
    def read_with_remote_ours(path):
        data = original_read(path)
        path = Path(path)
        guarded_ours = path.parent.name == 'ours' and data.get('method') == 'Ours'
        guarded_retry = path.parent.name == 'raw_opt3' and data.get('retry_policy') is not None
        if path.name == 'results.json' and (guarded_ours or guarded_retry):
            issue = ours_termination_issue(path)
            if issue:
                # Preserve the raw artifact, but do not count a killed process as
                # a successful compilation merely because it wrote a result first.
                data = {'case':data['case'], 'method':data['method'], 'status':issue,
                        'basis':'IBM', 'Lambda':data['Lambda'],
                        'unaccepted_result_path':str(path),
                        'resource_stop':issue, 'failure_stage':'process completion'}
        if Path(path) == batch.OUT / 'local_status.json':
            for name in ('remote_ours_status.json', 'remote_hrw_status.json', 'remote_ours_timing_status.json'):
                supplemental = batch.OUT / name
                if supplemental.exists():
                    remote_jobs = original_read(supplemental)['jobs']
                    moved = {job['case'] for job in remote_jobs}
                    data = dict(data, jobs=[j for j in data['jobs'] if j['case'] not in moved] + remote_jobs)
        if path == batch.OUT / 'remote_status.json':
            for name in ('remote_opt3_retry_status.json', 'remote_toom_opt3_retry_status.json'):
                retry_status = batch.OUT / name
                if retry_status.exists():
                    retry_jobs = original_read(retry_status)['jobs']
                    replaced = {(j['case'], j['kind']) for j in retry_jobs}
                    data = dict(data, jobs=[j for j in data['jobs']
                                           if (j['case'], j['kind']) not in replaced] + retry_jobs)
        return data
    batch.read = read_with_remote_ours
    try:
        batch.collect()
    finally:
        batch.read = original_read
        batch.saved = original_saved
    results = batch.ROOT / 'Results'
    path = results / 'full_ibm_per_seed_20261015.csv'
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row.update(nam[row['case']] if row['method'] == 'Ours' else dict.fromkeys(next(iter(nam.values())), ''))
        row['display_case'] = display_case(row['case'])
        selected = saved_with_timing_fallback(row['case'], row['method'])
        resource_path = selected.parent / 'resources.json'
        resource = batch.read(resource_path) if resource_path.exists() else {}
        row['worker_elapsed_s'] = resource.get('elapsed_s', '') if row['status'] == 'ok' else ''
        row['remote_compile_stages_s'] = resource.get('remote_compile_stages_s', '') if row['status'] == 'ok' else ''
        row['selected_result_path'] = str(selected)
    write_csv(path, rows)
    for partition in manifest['partitions']:
        path = results / f"benchmark_{partition['id']}_ibm_20261015.csv"
        rows = list(csv.DictReader(path.open()))
        for row in rows:
            row['display_case'] = display_case(row['case'])
            cases = [c for c, e in manifest['cases'].items() if e['group'] == row['case']]
            assert len(cases) == int(row['samples'])
            gates = [nam[c]['G_NAM'] for c in cases]
            row['ours_G_NAM'] = statistics.mean(gates)
            row['ours_G_NAM_sd'] = statistics.stdev(gates) if len(gates) > 1 else ''
            row['ours_NAM_samples'] = len(gates)
            row['ours_NAM_reproduced'] = sum(nam[c]['NAM_status'].startswith('reproduced') for c in cases)
            row['ours_NAM_status'] = 'reproduced' if row['ours_NAM_reproduced'] == len(cases) else 'archived; reproduction pending'
        assert len(rows) == 20
        write_csv(path, rows)
        lines = path.with_suffix('.md').read_text().splitlines()
        lines[0] = lines[0].replace(' — IBM', ' — IBM comparison + Ours NAM')
        lines[2] = 'IBM 门集：{u1,u2,u3,cx}；Ours 另报 NAM：{rz,h,x,cx}。两种 Ours 输出均为 opt0；Raw 为 IBM opt0，Raw+Qiskit 为完整 IBM 电路 opt3。reset 不计入 G。'
        lines.insert(6, 'Ours 的 G NAM 来自同一已保存 rewrite 的历史完整电路；† 表示本轮尚未完成同一高层电路的 NAM/IBM 复现。无 † 的行已核对本轮 NAM G/W 与历史一致。W/T Ours 是 IBM 口径，不能将其 T 作为 NAM 分解耗时。逐例 CSV 保留 NAM 来源、哈希和复现状态。')
        row_index = 0
        for index, line in enumerate(lines):
            if not line.startswith('| '):
                continue
            cells = line.split('|')
            if cells[1].strip() == 'Case':
                cells[11] = ' G Ours IBM '
                cells[13] = ' T Ours IBM '
                cells.insert(11, ' G Ours NAM ')
            elif cells[1].strip() == '---':
                cells.insert(11, ' ---: ')
            else:
                row = rows[row_index]
                cells[1] = f" {row['display_case']} "
                value = row['ours_G_NAM']
                label = f'{value:,.0f}' if value == int(value) else f'{value:,.3f}'
                if row['ours_NAM_reproduced'] != int(row['samples']):
                    label += '†'
                cells.insert(11, f' {label} ')
                row_index += 1
            lines[index] = '|'.join(cells)
        assert row_index == 20
        temporary = path.with_suffix('.md.writing')
        temporary.write_text('\n'.join(lines) + '\n')
        temporary.replace(path.with_suffix('.md'))
    print('Ours NAM audit:', Counter(n['NAM_status'] for n in nam.values()), flush=True)
    sys.path.insert(0, str(batch.ROOT / 'scripts/eval_redesign'))
    from render_full_benchmark_tables import render
    render(compile_pdf=True)


if __name__ == '__main__':
    collect()
