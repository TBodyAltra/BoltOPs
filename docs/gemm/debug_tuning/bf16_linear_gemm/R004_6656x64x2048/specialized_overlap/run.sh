#!/usr/bin/env bash
# R004 (6656x64x2048 TT BF16) overlap-only package launcher.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
# Offline checks use only the standard library; a Triton installation is optional.
for arg in "$@"; do
    if [[ "$arg" == "--check-only" || "$arg" == "--help" || "$arg" == "-h" ]]; then
        exec "${PYTHON:-/usr/bin/python3}" -B "$HERE/bench.py" "$@"
    fi
done
PATH_SH="${TRITON_PATH_SH:-/public/home/duyubo/triton_all_version/triton_3.6/triton/path.sh}"
if [[ ! -f "$PATH_SH" ]]; then
    printf 'Missing Triton environment script: %s\n' "$PATH_SH" >&2
    exit 1
fi
# bench.py reads this environment afresh for each isolated worker. Never import
# torch/Triton here: PMD must initialize only inside that worker's output directory.
export TRITON_PATH_SH="$PATH_SH"
export OPT_CLANG="${OPT_CLANG:-/opt/dtk/aillvm/bin/clang-18}"
export OPT_LLD="${OPT_LLD:-/opt/dtk/aillvm/bin/ld.lld}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-/usr/bin/python3}" -B "$HERE/bench.py" "$@"
