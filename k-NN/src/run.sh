#!/usr/bin/env bash
# run.sh – build everything and execute the benchmark with optional args.
# Usage examples:
#     ./run.sh                 # default sizes (100 K × 1024)
#     ./run.sh 1000000 4096    # custom N and Q

set -euo pipefail

N=${1:-100000}
Q=${2:-1024}
D=${3:-128}
K=${4:-10}

echo "=== k-NN benchmark (N=$N Q=$Q d=$D k=$K) ==="
python3 benchmark.py --N "$N" --Q "$Q" --d "$D" --k "$K"
