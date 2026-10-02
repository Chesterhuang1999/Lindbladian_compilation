"""Repackage frozen TFIM sources with the user-requested scalable search profile."""
import hashlib
import argparse
import io
import json
from pathlib import Path
import tarfile
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New deployment package directory; must not exist.')
    args = parser.parse_args()
    out = args.out.resolve()
    original = ROOT / 'Baseline_results/tfim_tud/scalable_beam_20260911/tfim_beam8_payload.tar.gz'
    with tarfile.open(original) as stream:
        contents = {m.name: stream.extractfile(m).read() for m in stream.getmembers() if m.isfile()}
    manifest = json.loads(contents['remote_source_manifest.json'])
    runtime_names = {'adapter_channelir.py', 'remote_channel_adapter.py',
                     'tfim_tud_reference.py', 'run_remote_channel.py',
                     'experiment_plan.lock.yaml', 'placement_scalable.json'}
    contents = {name: data for name, data in contents.items()
                if name.startswith(('src/', 'remote_inputs/'))
                or (name.startswith('experiments/tfim_tud/') and Path(name).name in runtime_names)}
    # Preserve the measured backend and adapters. Only the CLI/hash bookkeeping changed.
    contents['experiments/tfim_tud/run_remote_channel.py'] = (Path(__file__).parent / 'run_remote_channel.py').read_bytes()
    for name in ('update_markdown_tables.py', 'verify_scalable_beam_trial.py'):
        contents['experiments/tfim_tud/' + name] = (Path(__file__).parent / name).read_bytes()
    profile_path = ROOT / 'experiments/tfim_tud/placement_scalable.json'
    profile = json.loads(profile_path.read_text())
    plan_path = 'experiments/tfim_tud/experiment_plan.lock.yaml'
    plan = yaml.safe_load(contents[plan_path])
    plan['freeze']['placement_environment'] = profile
    contents[plan_path] = yaml.safe_dump(plan, sort_keys=False).encode()
    contents['experiments/tfim_tud/placement_scalable.json'] = profile_path.read_bytes()
    manifest['parent_payload_sha256'] = hashlib.sha256(original.read_bytes()).hexdigest()
    manifest['destination_base'] = 'Choose a new directory; do not overwrite the measured deployment.'
    manifest['placement_environment'] = profile
    manifest['channel_protocol']['sizes'] = [3, 4, 6, 8, 12]
    manifest['channel_protocol']['executed_methods'] = ['pauli_placement']
    manifest['transfer_status'] = 'AUTHORIZED_BY_USER'
    manifest['files'] = [{'relative_path': name, 'sha256': hashlib.sha256(data).hexdigest(),
                          'bytes': len(data)} for name, data in sorted(contents.items())
                         if name != 'remote_source_manifest.json']
    contents['remote_source_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    out.mkdir(parents=True, exist_ok=False)
    archive = out / 'tfim_stable_payload.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name, data in sorted(contents.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    (out / 'remote_source_manifest.json').write_bytes(contents['remote_source_manifest.json'])
    print(json.dumps({'archive': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'profile': profile}, indent=2))


if __name__ == '__main__':
    main()
