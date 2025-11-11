#!/usr/bin/env bash
# Usage: experiments/run_full.sh <seed>
set -euo pipefail
SEED="${1:?usage: run_full.sh <seed>}"
python scripts/train.py --variant vcara --seed "$SEED" --frames 1000000 --out outputs
