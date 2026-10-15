#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
/Users/chester_hqf/miniconda3/bin/conda run -n qiskit_simulate python Baseline_results/eval_redesign/cp4_rq3_semantic_backend_validation/run_cp4b_semantic_backend_validation.py
