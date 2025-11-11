#!/usr/bin/env bash
# Usage: experiments/run_no_cf.sh <seed>
set -euo pipefail
SEED="${1:?usage: run_no_cf.sh <seed>}"
python scripts/train.py --variant vcara_no_cf --seed "$SEED" --frames 1000000 --out outputs
