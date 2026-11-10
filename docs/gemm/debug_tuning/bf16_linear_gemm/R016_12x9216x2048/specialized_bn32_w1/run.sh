#!/usr/bin/env bash
# R016 12x9216x2048 TT BF16 specialized benchmark (PMD, gfx946).
# Usage: bash run.sh --name NEW_NAME [--compare] [--seed 29] [--dispatches 5]
#        Each --name must be new: runs/<name>/ is created without overwriting.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-/usr/bin/python3}" -B "$HERE/reproduce.py" "$@"
