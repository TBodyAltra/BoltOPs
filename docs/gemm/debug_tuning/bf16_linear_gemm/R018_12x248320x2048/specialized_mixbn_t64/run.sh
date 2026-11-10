#!/usr/bin/env bash
# R018 12x248320x2048 TT BF16 mixed-BN specialization benchmark (PMD, gfx946).
# Usage: bash run.sh --name NEW_NAME [--only baseline|best] [--seeds "29 30 31"] [--dispatches 4]
#                    [--compile-only] [--jobs N]
#   Runs the baseline (shared generic kernel + generic/best_config.json) and/or the specialized kernel, one
#   independent process per (version, seed): runs/<NEW_NAME>_baseline_s<seed> and runs/<NEW_NAME>_best_s<seed>.
#   Each name must be new. --dispatches counts the warm-up: 4 = 1 warm-up + 3 timed (default 2 = 1 + 1).
#   --jobs defaults to the number of cases (all in parallel; simulated cycles should not depend on host load).
#   A full-shape PMD dispatch takes about 40 minutes of host time (original wall: 4678 s / 4748 s for 2).
#   An environment manifest (BoltOPs / Triton commits, libtriton.so and file SHA256, compiler) is written to
#   runs/<NEW_NAME>_cases/manifest.json before any simulation starts.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME=""; ONLY="both"; COMPILE_ONLY=0; JOBS=""; SEEDS="29"; DISPATCHES=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --dispatches) DISPATCHES="$2"; shift 2 ;;
        --compile-only) COMPILE_ONLY=1; shift ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$NAME" ]] || { echo "--name is required" >&2; exit 2; }
[[ "$ONLY" =~ ^(both|baseline|best)$ ]] || { echo "--only must be baseline or best" >&2; exit 2; }
[[ "$DISPATCHES" =~ ^[0-9]+$ && "$DISPATCHES" -ge 2 ]] || { echo "--dispatches must be an integer >= 2" >&2; exit 2; }
CASES_DIR="$HERE/runs/${NAME}_cases"
[[ ! -e "$CASES_DIR" ]] || { echo "already exists: $CASES_DIR" >&2; exit 1; }
mkdir -p "$CASES_DIR"
export HERE NAME ONLY COMPILE_ONLY CASES_DIR SEEDS DISPATCHES
JOBS="$(python3 -I - <<'PY'
import json, os
here, name, only = os.environ['HERE'], os.environ['NAME'], os.environ['ONLY']
seeds = [int(s) for s in os.environ['SEEDS'].split()]
disp = int(os.environ['DISPATCHES'])
extra = {'compile_only': True} if os.environ['COMPILE_ONLY'] == '1' else {}
shape = [12, 248320, 2048]
timeout = 7500 * disp                       # about 3x the expected ~40 min per full-shape dispatch
cases = []
for seed in seeds:
    common = dict(shape=shape, seed=seed, dispatches=disp, timeout=timeout, **extra)
    if only in ('both', 'baseline'):
        # default operator = shared generic kernel, default config = autotune archive best.config
        # (identical to generic/best_config.json)
        cases.append(dict(name=f'{name}_baseline_s{seed}', **common))
    if only in ('both', 'best'):
        cases.append(dict(name=f'{name}_best_s{seed}', operator='r018_mixbn_kernel.py', grid_fn=True,
                          extra_constexpr=['TAIL_BN', 'CU_MULT'],
                          config=json.load(open(f'{here}/best_config.json')), **common))
json.dump(cases, open(os.environ['CASES_DIR'] + '/cases.json', 'w'), indent=1)
print(len(cases))
PY
)" || exit 1
[[ -n "${JOBS_OVERRIDE:-}" ]] && JOBS="$JOBS_OVERRIDE"
export JOBS_N="$JOBS"
# --- environment manifest (written before the simulation starts) ---
python3 -I - <<'PY'
import datetime, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
here = Path(os.environ['HERE'])
repo = Path(os.environ.get('BOLTOPS_REPO') or here.parents[5]).resolve()
linear = repo / 'docs/gemm/debug_tuning/bf16_linear_gemm'
sys.dont_write_bytecode = True
sys.path.insert(0, str(linear / 'autotune/scripts'))
import tune_lib as T


def sh(*a, cwd=None):
    r = subprocess.run(a, cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f'unavailable: {r.stderr.strip()[:200]}'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


triton = Path(T.PATH_SH).parent
lib = next(iter(sorted(triton.glob('build/lib.*/triton/_C/libtriton.so'))), None)
files = {
    'generic_kernel': linear / 'operators/gemm_a16w16_generic.py',
    'generic_kernel_scripts_copy': linear / 'autotune/scripts/gemm_a16w16_generic.py',
    'specialized_kernel': here / 'r018_mixbn_kernel.py',
    'tune_lib': linear / 'autotune/scripts/tune_lib.py',
    'bench': here / 'bench.py',
    'run_sh': here / 'run.sh',
    'specialized_best_config': here / 'best_config.json',
    'generic_best_config': here.parent / 'generic/best_config.json',
    'autotune_archive': linear / 'autotune/results_TT_dyn/shapes/R018_12x248320x2048_TT.json',
}
env = T.path_sh_env()
manifest = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'name': os.environ['NAME'], 'host': platform.node(), 'platform': platform.platform(),
    'python': sh(T.PYTHON, '--version'),
    'seeds': [int(s) for s in os.environ['SEEDS'].split()], 'dispatches': int(os.environ['DISPATCHES']),
    'only': os.environ['ONLY'], 'compile_only': os.environ['COMPILE_ONLY'] == '1',
    'boltops': {'path': str(repo), 'head': sh('git', '-C', str(repo), 'rev-parse', 'HEAD'),
                'branch': sh('git', '-C', str(repo), 'rev-parse', '--abbrev-ref', 'HEAD'),
                'tracked_changes': len([l for l in sh('git', '-C', str(repo), 'status', '--porcelain',
                                                      '--untracked-files=no').splitlines() if l.strip()])},
    'triton': {'root': str(triton), 'head': sh('git', '-C', str(triton), 'rev-parse', 'HEAD'),
               'branch': sh('git', '-C', str(triton), 'rev-parse', '--abbrev-ref', 'HEAD'),
               'tracked_changes': len([l for l in sh('git', '-C', str(triton), 'status', '--porcelain',
                                                     '--untracked-files=no').splitlines() if l.strip()]),
               'libtriton': str(lib) if lib else None, 'libtriton_sha256': sha(lib) if lib else None},
    'compiler': {'clang': T.CLANG, 'clang_version': sh(T.CLANG, '--version').splitlines()[0],
                 'lld': T.LLD, 'path_sh': T.PATH_SH, 'path_sh_env': {k: v for k, v in env.items() if k != 'PYTHONPATH'}},
    'files_sha256': {k: sha(v) for k, v in files.items()},
    'note': 'PMD build info and --num-cu / single-die lines are in each run worker.log; aggregate.py copies them.',
}
(Path(os.environ['CASES_DIR']) / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print('manifest:', Path(os.environ['CASES_DIR']) / 'manifest.json')
PY
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-/usr/bin/python3}" -B "$HERE/bench.py" suite "$CASES_DIR/cases.json" --jobs "${JOBS_N}"
