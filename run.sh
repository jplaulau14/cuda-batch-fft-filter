#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  make proof RUN_ARGS="$*"
else
  make proof
fi

if [ -f artifacts/cuda-batch-fft-filter-proof.zip ]; then
  echo "Proof artifact: artifacts/cuda-batch-fft-filter-proof.zip"
else
  echo "zip not found; proof files are available under output/ and data/input/"
fi
