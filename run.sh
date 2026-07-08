#!/usr/bin/env bash
set -euo pipefail

make proof RUN_ARGS="$*"
echo "Proof artifact: artifacts/cuda-batch-fft-filter-proof.zip"
