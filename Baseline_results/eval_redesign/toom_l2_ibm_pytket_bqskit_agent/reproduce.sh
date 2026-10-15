#!/usr/bin/env bash
set -euo pipefail

cd /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation
/Users/chester_hqf/miniconda3/bin/conda run -n qiskit_simulate \
  python /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation/Baseline_results/eval_redesign/toom_l2_ibm_pytket_bqskit_agent/run_all.py --only-generate

/Users/chester_hqf/miniconda3/bin/conda run -n ChannelIR_test \
  python Baseline_scripts/run_pytket_baseline.py \
  --input /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation/Baseline_results/eval_redesign/toom_l2_ibm_pytket_bqskit_agent/ibm_input/toom_2d_periodic_L2_dt0p1_basic_raw_ibm_opt0.qasm \
  --output /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation/Baseline_results/eval_redesign/toom_l2_ibm_pytket_bqskit_agent/pytket/toom_2d_periodic_L2_dt0p1_pytket_best_of_metric_total.qasm \
  --summary /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation/Baseline_results/eval_redesign/toom_l2_ibm_pytket_bqskit_agent/pytket/summary.json \
  --strategy best-of \
  --best-of-objective metric_total

/Users/chester_hqf/miniconda3/bin/conda run -n ChannelIR_test \
  python Baseline_scripts/run_bqskit_baseline.py \
  --input /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation/Baseline_results/eval_redesign/toom_l2_ibm_pytket_bqskit_agent/ibm_input/toom_2d_periodic_L2_dt0p1_basic_raw_ibm_opt0.qasm \
  --output /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation/Baseline_results/eval_redesign/toom_l2_ibm_pytket_bqskit_agent/bqskit/toom_2d_periodic_L2_dt0p1_bqskit_level2_skip_retarget_mapping.qasm \
  --summary /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation/Baseline_results/eval_redesign/toom_l2_ibm_pytket_bqskit_agent/bqskit/summary.json \
  --gate-set ibm \
  --optimization-level 2 \
  --workflow-mode skip-retarget-mapping \
  --timeout 3600

/Users/chester_hqf/miniconda3/bin/conda run -n qiskit_simulate \
  python /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation/Baseline_results/eval_redesign/toom_l2_ibm_pytket_bqskit_agent/run_all.py --only-summarize
