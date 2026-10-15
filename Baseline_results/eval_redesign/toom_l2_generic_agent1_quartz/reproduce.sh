#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation"
OUTDIR="$ROOT/Baseline_results/eval_redesign/toom_l2_generic_agent1_quartz"
INPUT="$ROOT/circuits/Table1/toom_2d_periodic_L2_dt0p1/toom_2d_periodic_L2_dt0p1_basic_raw_nam_opt0.qasm"

mkdir -p "$OUTDIR/quartz"

cd "$ROOT"
conda run -n ChannelIR_test python Baseline_scripts/run_quartz_baseline.py \
  --input "$INPUT" \
  --timeout 3600 \
  --max-candidates 64 \
  --checkpoint-interval 300 \
  --checkpoint-output "$OUTDIR/quartz/quartz_native_checkpoint.qasm" \
  --checkpoint-summary "$OUTDIR/quartz/quartz_native_checkpoint.json" \
  --output "$OUTDIR/quartz/quartz_native_output.qasm" \
  --summary "$OUTDIR/quartz/quartz_native_summary.json" \
  > "$OUTDIR/quartz/quartz_native_stdout.log" \
  2> "$OUTDIR/quartz/quartz_native_stderr.log"
