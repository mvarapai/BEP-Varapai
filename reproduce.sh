#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/generated

python -m scripts.01_prepare_data

echo "Done. Results written to results/generated/"