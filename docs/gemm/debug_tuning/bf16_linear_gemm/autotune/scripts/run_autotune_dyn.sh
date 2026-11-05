#!/usr/bin/env bash
# Autotune with the dynamic-M/N/K kernel (gemm_a16w16_generic._gemm_bf16_generic_kernel without EVEN_MNK,
# optional bias), with its own work / report directories so results of older kernels are never mixed in.
#
# Phases (all in the same WORK dir; `run` skips shapes that already have a result, `full` skips shapes that
# already have a full-size measurement, so every phase is resumable):
#
#   phase 1  autotune.py run   --shapes-json PRIO   search the priority shapes only (table 3, 27 shapes)
#   phase 2  autotune.py full                       their winners on the real M x N x K   <- priority results complete here
#   phase 3  autotune.py run                        search the remaining shapes
#   phase 4  autotune.py full                       their winners on the real M x N x K
#   report   autotune.py report                     best_per_shape.json / .md
#
# IMPORTANT: use a FRESH work dir for a new kernel version. The result cache key (M, N, K, layout, config,
# dispatches) does not include the kernel version, so an old work dir would serve stale numbers.
#
#   bash run_autotune_dyn.sh [WORK_DIR] [WORKERS]
#   PRIO="" bash run_autotune_dyn.sh ...         # no priority phase, all shapes in cost order
#   RUN_TN=1 bash run_autotune_dyn.sh ...        # also TN, seeded from the TT winners
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$(dirname "$HERE")"                         # .../autotune
WORK="${1:-/public/home/duyubo/bolt/xk/BoltOPs/tmp/autotune_20261104_dyn}"
NCPU="$(nproc --all)"
WORKERS="${2:-$(( NCPU * 40 / 100 ))}"              # default: 40% of the logical CPUs (128 -> 51); no idle-CPU probing
[ "$WORKERS" -ge 1 ] || WORKERS=1
PY=python3
# SHAOBO_BU_PRIORITY_REQUIREMENTS.md table 3 (27 shapes); set PRIO="" to disable.
PRIO="${PRIO-$OUTDIR/shapes_priority_table3.json}"
mkdir -p "$WORK"

# pipeline LAYOUT [extra args for run]
pipeline() {
  local lay="$1"; shift
  local w="$WORK/$lay"
  if [ -n "$PRIO" ]; then
    $PY -B "$HERE/autotune.py" run  --work "$w" --layout "$lay" --workers "$WORKERS" --shape-par 10 \
        --shapes-json "$PRIO" "$@" >> "$WORK/${lay}_prio.log" 2>&1
    $PY -B "$HERE/autotune.py" full --work "$w" --workers "$WORKERS" >> "$WORK/${lay}_prio_full.log" 2>&1
  fi
  $PY -B "$HERE/autotune.py" run  --work "$w" --layout "$lay" --workers "$WORKERS" --shape-par 10 "$@" >> "$WORK/$lay.log" 2>&1
  $PY -B "$HERE/autotune.py" full --work "$w" --workers "$WORKERS" >> "$WORK/${lay}_full.log" 2>&1
  $PY -B "$HERE/autotune.py" report --work "$w" --out "$OUTDIR/results_${lay}_dyn"
}

pipeline TT

if [ "${RUN_TN:-0}" = "1" ]; then
  pipeline TN --seed-from "$WORK/TT" --topk-tiles 3 --n-seeds 2 --passes 2
fi
