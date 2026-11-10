#!/usr/bin/env bash
# Original-only 12x64x2048 TT BF16 fused split-K benchmark.
# Usage: bash /absolute/path/to/OPT/run.sh [--rounds 2 --warmup 1 --reps 5 --out NEW_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATH_SH="${TRITON_PATH_SH:-/public/home/duyubo/triton_all_version/triton_3.6/triton/path.sh}"
if [[ ! -f "$PATH_SH" ]]; then
    printf 'Missing Triton environment script: %s\n' "$PATH_SH" >&2
    exit 1
fi
source "$PATH_SH"
export TRITON_HIP_CLANG_PATH="${OPT_CLANG:-/opt/dtk/aillvm/bin/clang-18}"
export TRITON_HIP_LLD_PATH="${OPT_LLD:-/opt/dtk/aillvm/bin/ld.lld}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-/usr/bin/python3}" -B "$HERE/bench.py" "$@"
