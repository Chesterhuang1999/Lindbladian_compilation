#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation"
cd "$ROOT"
CONDA="/Users/chester_hqf/miniconda3/bin/conda"
ENV="ChannelIR_test"
INPUT="Baseline_results/eval_redesign/repetition_dqe_raw_ours_probe/repetition_dqe_d3_eps0p25/repetition_dqe_d3_eps0p25_basic_raw_nam_opt0.qasm"
OUT="Baseline_results/eval_redesign/cp1a_repetition_dqe_d3_nam_tools_3600_20260701_115437"
mkdir -p "$OUT"/runs/{voqc,feynman,wisq,quartz,quizx}

"$CONDA" run -n "$ENV" python Baseline_scripts/run_voqc_baseline.py   --input "$INPUT"   --output "$OUT/runs/voqc/voqc_native_output.qasm"   --summary "$OUT/runs/voqc/voqc_native_summary.json"   --timeout 3600

"$CONDA" run -n "$ENV" python Baseline_scripts/run_feynman_baseline.py   --input "$INPUT"   --output "$OUT/runs/feynman/feynman_native_output.qasm"   --summary "$OUT/runs/feynman/feynman_native_summary.json"   --timeout 3600

"$CONDA" run -n "$ENV" python Baseline_scripts/run_wisq_baseline.py   --input "$INPUT"   --output "$OUT/runs/wisq/wisq_native_output.qasm"   --summary "$OUT/runs/wisq/wisq_native_summary.json"   --target-gateset NAM   --optimization-objective TOTAL   --timeout 3600

"$CONDA" run -n "$ENV" python Baseline_scripts/run_quartz_baseline.py   --input "$INPUT"   --output "$OUT/runs/quartz/quartz_native_output.qasm"   --summary "$OUT/runs/quartz/quartz_native_summary.json"   --timeout 3600   --max-candidates 64   --checkpoint-interval 300   --checkpoint-output "$OUT/runs/quartz/quartz_checkpoint_best.qasm"   --checkpoint-summary "$OUT/runs/quartz/quartz_checkpoint_best.json"   --use-available-xfers   --prune-relative-to-best

"$CONDA" run -n "$ENV" python Baseline_scripts/run_quizx_baseline.py   --input "$INPUT"   --output "$OUT/runs/quizx/quizx_native_output.qasm"   --summary "$OUT/runs/quizx/quizx_native_summary.json"   --method full   --timeout 3600
