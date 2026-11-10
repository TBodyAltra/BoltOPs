#!/usr/bin/env bash
# Reproduce the generic-kernel result of one shape directory on PMD (see repro.py --help).
# Usage: bash run.sh R014_12x64x2048 [--layout TT|TN] [--rounds 2] [--dispatches 6] [--out NEW_ABS_DIR]
# The Triton / compiler environment is taken from tune_lib.py (PATH_SH, AUTOTUNE_CLANG); BOLTOPS_REPO overrides the
# BoltOPs checkout used for `import boltops...` (default: the checkout that contains this directory).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-/usr/bin/python3}" -B "$HERE/repro.py" "$@"
