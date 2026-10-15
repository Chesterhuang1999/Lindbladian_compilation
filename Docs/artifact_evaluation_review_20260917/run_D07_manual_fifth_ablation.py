"""Complete the eight ablations for the explicitly edited fifth input only."""
import argparse
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts/eval_redesign'),
                str(ROOT / 'Baseline_results/eval_redesign/cp4_rq3_semantic_backend_validation')]
import run_cp2_random_ablation as ab
import random_coloring_no_approx_zero_check as probe
from review_backend_adapter import install

install()
variant = ROOT / 'Baseline_results/eval_redesign/artifact_review_20260917/D07_seed20260921_K4_IZZX'
OUT = variant / 'ablation'
OUT.mkdir(parents=True, exist_ok=False)
seed = 20260921
args = argparse.Namespace(example='random-pauli', out_dir=OUT, num_vertices=4, q=4, graph='linear',
                          kraus_count=5, hilbert_dim=16, first_kraus_terms=10, other_kraus_terms=2,
                          seed=seed, seeds=[seed], beam_width=8, max_steps=50, tol=1e-12,
                          transpile_optimization_level=0)
with (variant / 'modified_raw_kraus.pkl').open('rb') as stream:
    raw = pickle.load(stream)
paths = ab.cache_paths(args, seed)
assert all(ms.size == 4 for ms in raw)
metrics = ab.channel_metrics(raw)
metrics.update(benchmark_name=ab.benchmark_name(args), generator='random-pauli-with-manual-edit',
               seed=seed, build_channel_time_s=0.0,
               model={'model': 'random_pauli_kraus_channel_user_edited', 'system_qubits': 4,
                      'hilbert_dim': 16, 'term_counts': [10, 2, 2, 2, 2],
                      'manual_edits': [{'kraus_index': 4, 'term_index': 0,
                                        'old_pauli': 'ZZZZ', 'new_pauli': 'IZZX',
                                        'coefficient_unchanged': True}]})
ab.dump_pickle(paths['raw_kraus'], raw)
ab.write_json(paths['raw_metrics'], metrics)
ab.write_json(OUT / 'public_input.json', {
    'public_tuple': [5, 4, 10, 2], 'base_seed': seed,
    'input_kind': 'manual edit after random generation; not an unmodified random sample',
    'source': str(variant / 'modified_raw_kraus.pkl'),
    'source_sha256': hashlib.sha256((variant / 'modified_raw_kraus.pkl').read_bytes()).hexdigest(),
    'residual': 'fixed', 'block_registers': '[work,system]',
    'first_four_seeds': 'reuse frozen original records 20260917-20260920', 'argv': sys.argv})
rewritten, rw = ab.ensure_rewrite_cache(args, seed)
for cfg in ab.CONFIGS:
    row = ab.run_config(args, cfg, seed)
    assert row['status'] == 'ok', row['error_msg']
ab.aggregate(args)
qc, regs, _, _ = ab.build_config_circuit_random(copy.deepcopy(rewritten), ab.CONFIG_BY_SLUG['rewrite__flat__order'])
td, fro, maxe = probe.metrics(probe.direct_density(raw, 4), probe.sim_no_approx(qc, regs))
ab.write_json(OUT / 'all_state_check.json', {
    'seed': seed, 'manual_edit': 'K4 term0 ZZZZ -> IZZX', 'trace_distance': td,
    'frobenius': fro, 'max_entry': maxe, 'passed': td < 1e-10,
    'input_state': 'zero_density', 'reference': 'modified raw Kraus map'})
assert td < 1e-10
print('Eight manual-input ablations and All state probe passed.', flush=True)
