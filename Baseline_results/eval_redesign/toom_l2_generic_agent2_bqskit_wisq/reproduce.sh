#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation"
OUTDIR="$ROOT/Baseline_results/eval_redesign/toom_l2_generic_agent2_bqskit_wisq"
INPUT="$ROOT/circuits/Table1/toom_2d_periodic_L2_dt0p1/toom_2d_periodic_L2_dt0p1_basic_raw_nam_opt0.qasm"

mkdir -p "$OUTDIR/wisq"

cd "$ROOT"
conda run -n ChannelIR_test python Baseline_scripts/run_wisq_baseline.py \
  --input "$INPUT" \
  --timeout 3600 \
  --optimization-objective TOTAL \
  --output "$OUTDIR/wisq/wisq_native_output.qasm" \
  --summary "$OUTDIR/wisq/wisq_native_summary.json" \
  > "$OUTDIR/wisq/wisq_runner_stdout.txt" \
  2> "$OUTDIR/wisq/wisq_runner_stderr.txt"
