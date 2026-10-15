#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation"
CONDA="/Users/chester_hqf/miniconda3/bin/conda"

cd "$REPO_ROOT"
"$CONDA" run -n qiskit_simulate \
  python Baseline_results/eval_redesign/cp1_stinespring_sanity/run_cp1c_stinespring_sanity.py
