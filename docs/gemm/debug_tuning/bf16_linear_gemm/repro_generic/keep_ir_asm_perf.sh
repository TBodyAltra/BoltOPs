#!/usr/bin/env bash
# Usage: bash keep_ir_asm_perf.sh SRC_CAPTURE_DIR NEW_DEST_DIR
# Copy only IR (ttir/ttgir/llir), assembly (amdgcn) and the per-dispatch perf/TT
# files (g/<proc>/<dispatch>/) out of a full capture directory.
set -euo pipefail
SRC="$(cd "$1" && pwd)"
DEST="$(realpath -m "$2")"
[[ ! -e "$DEST" ]] || { echo "Destination already exists: $DEST" >&2; exit 2; }
mkdir -p "$DEST"
for kind in ttir ttgir llir amdgcn; do
    [[ -s "$SRC/kernel.$kind" ]] || { echo "Missing or empty kernel.$kind in $SRC" >&2; exit 1; }
    cp -p "$SRC/kernel.$kind" "$DEST/"
done
cd "$SRC"
count=0
for d in g/*/*/; do
    [[ -s "$d/stats_xcd0.perf" ]] || continue
    mkdir -p "$DEST/$d"
    cp -p "$d"* "$DEST/$d"
    count=$((count + 1))
done
[[ "$count" -gt 0 ]] || { echo "No perf dispatch directories found in $SRC/g" >&2; exit 1; }
echo "Kept IR, assembly and $count perf dispatch directories in $DEST"
