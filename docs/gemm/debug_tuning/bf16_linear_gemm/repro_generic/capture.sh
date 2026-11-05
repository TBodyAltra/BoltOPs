#!/usr/bin/env bash
# Usage: bash capture.sh SHAPE_DIR_NAME [generic|specialized] [TT|TN] [NEW_ABSOLUTE_DEST]
# e.g.   bash capture.sh R015_12x2048x4096 generic TT
#
# Compiles the shape's current config under the PMD launcher, then keeps only the
# IR, assembly and perf/TT files in
#   <shape>/generic/runs/<layout>_ir_asm_perf   (or specialized_fused_splitk/runs/ ...)
# The full capture (caches, logs, manifest) lives in a scratch directory that is removed
# on success; set KEEP_WORK=1 to keep it, or CAPTURE_WORK=/abs/dir to choose its parent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHAPE="${1:?Usage: bash capture.sh SHAPE_DIR_NAME [generic|specialized] [TT|TN] [NEW_ABSOLUTE_DEST]}"
IMPLEMENTATION="${2:-generic}"
LAYOUT="${3:-TT}"
case "$IMPLEMENTATION/$LAYOUT" in
    generic/TT|generic/TN|specialized/TT) ;;
    *) echo 'Expected generic TT, generic TN, or specialized TT' >&2; exit 2 ;;
esac
SHAPE_DIR="$(cd "$HERE/../$SHAPE" && pwd)"
if [[ "$IMPLEMENTATION" = generic ]]; then IMPL_DIR=generic; else IMPL_DIR=specialized_fused_splitk; fi
DEST="${4:-$SHAPE_DIR/$IMPL_DIR/runs/${LAYOUT}_ir_asm_perf}"
[[ "$DEST" = /* ]] || { echo 'Destination must be an absolute path' >&2; exit 2; }
[[ ! -e "$DEST" ]] || { echo "Destination already exists: $DEST; choose a fresh path" >&2; exit 2; }

OUT="$(mktemp -d "${CAPTURE_WORK:-/tmp}/capture_${SHAPE}_${IMPLEMENTATION}_${LAYOUT}.XXXXXX")"
source "${TRITON_PATH_SH:-/public/home/duyubo/triton_all_version/triton_3.6/triton/path.sh}"
export TRITON_HIP_CLANG_PATH="${AUTOTUNE_CLANG:-/opt/dtk/aillvm/bin/clang-18}"
export TRITON_HIP_LLD_PATH=/opt/dtk/aillvm/bin/ld.lld
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export AMDGCN_USE_BUFFER_OPS=1
unset GPU_ARGS GPU_DFLAGS GPU_OUT GPU_OUT_LOG SOC_ARGS SOC_DFLAGS
mkdir -p "$OUT"/{home,cache,temp,xdg-cache,xdg-config,xdg-data,torch-cache}
# Record Git state before the private HOME hides the caller's safe.directory settings.
python3 -B "$HERE/capture_current.py" --environment > "$OUT/environment.json"
export HOME="$OUT/home" TRITON_CACHE_DIR="$OUT/cache"
export TMPDIR="$OUT/temp" TMP="$OUT/temp" TEMP="$OUT/temp"
export XDG_CACHE_HOME="$OUT/xdg-cache" XDG_CONFIG_HOME="$OUT/xdg-config" XDG_DATA_HOME="$OUT/xdg-data"
export TORCHINDUCTOR_CACHE_DIR="$OUT/torch-cache"
cd "$OUT"
export PMD_PATH=/opt/rocm/pmd
export HSA_TOOLS_LIB=/opt/rocm/lib/xprofiler/libperf_gen_helper.so
export GPU_OUT=g

/opt/rocm/pmd/scripts/run.py \
    -c sb \
    -df TT Perf \
    -e python3 "$HERE/capture_current.py" --shape "$SHAPE" \
        --implementation "$IMPLEMENTATION" --layout "$LAYOUT" \
    2>&1 | tee run.log
# Some PMD launcher versions return zero even when the Python child fails.
[[ -s manifest.json ]] || { echo "Capture failed; see $OUT/run.log" >&2; exit 1; }
bash "$HERE/keep_ir_asm_perf.sh" "$OUT" "$DEST"
cd /
[[ "${KEEP_WORK:-0}" = 1 ]] && echo "Full capture kept in $OUT" || rm -rf "$OUT"
