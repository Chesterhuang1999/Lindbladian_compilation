#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation"
cd "$ROOT"

OUT_DIR="Baseline_results/eval_redesign/toom_l2_generic_agent3_voqc_feynman_quizx"
INPUT="circuits/Table1/toom_2d_periodic_L2_dt0p1/toom_2d_periodic_L2_dt0p1_basic_raw_nam_opt0.qasm"
TIMEOUT=3600

mkdir -p "$OUT_DIR/voqc" "$OUT_DIR/feynman" "$OUT_DIR/quizx"

conda run -n ChannelIR_test python Baseline_scripts/run_voqc_baseline.py \
  --input "$INPUT" \
  --output "$OUT_DIR/voqc/optimized.qasm" \
  --summary "$OUT_DIR/voqc/summary.json" \
  --timeout "$TIMEOUT"

conda run -n ChannelIR_test python Baseline_scripts/run_feynman_baseline.py \
  --input "$INPUT" \
  --output "$OUT_DIR/feynman/optimized.qasm" \
  --summary "$OUT_DIR/feynman/summary.json" \
  --timeout "$TIMEOUT"

conda run -n ChannelIR_test python Baseline_scripts/run_quizx_baseline.py \
  --input "$INPUT" \
  --output "$OUT_DIR/quizx/optimized.qasm" \
  --summary "$OUT_DIR/quizx/summary.json" \
  --method full \
  --timeout "$TIMEOUT"
