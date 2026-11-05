#!/usr/bin/env python3
"""Reproduce the generic-kernel result of one shape directory on PMD.

Reads  <shape_dir>/generic/best_config.json  and runs it with
operators/gemm_a16w16_generic.py (the copy in autotune/scripts/ is what tune_lib loads;
keep the two identical) through autotune/scripts/tune_lib.py:

  * every round is a fresh process with its own HOME / cache / PMD working directory
  * 1 warmup dispatch + (--dispatches - 1) timed dispatches per round, seed 29 + round
  * correctness: EVERY output element vs. gold = CPU FP32 reference rounded to BF16,
    torch.isclose(got, gold, atol=0.02, rtol=0.01), plus mean |got - ref| against the
    unrounded FP32 reference <= 1.05 x BF16 rounding floor + 1e-6, finite check and
    128 unchanged sentinel values before and after the output (tune_lib.worker criterion)
  * time: PMD system.gcSimCycles of the timed dispatches (kernel only; no JIT,
    allocation or host copy), converted with gcFrequency

Usage (run through run.sh so the interpreter flags match):
  bash run.sh R014_12x64x2048                   # TT, 2 rounds x (1 warmup + 5 timed)
  bash run.sh R014_12x64x2048 --layout TN       # weight W[N,K], K contiguous (gemm_a16w16 interface layout)
  bash run.sh R014_12x64x2048 --rounds 1 --dispatches 4 --out /abs/new/dir
  bash run.sh R014_12x64x2048 --config /abs/other_config.json

Writes <out>/summary.json (config, correctness, cycles, us, resources, versions); by default <out> is
<shape_dir>/generic/runs/<layout>_<time>/. Raw per-round output stays under <out>/round_*/ and is
ignored by Git (see ../.gitignore); keep summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # .../debug_tuning/bf16_linear_gemm
sys.path.insert(0, str(ROOT / 'autotune' / 'scripts'))
import tune_lib as T  # noqa: E402

DEFAULT_REPO = ROOT.parents[3]                       # BoltOPs checkout: bf16_linear_gemm/../../../..
# Describe tune_lib.worker's literal tolerances; tune_lib does not export constants.
ACCURACY_ATOL = 0.02
ACCURACY_RTOL = 0.01


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f'unknown ({exc.__class__.__name__})'


def repo_info(repo: Path) -> dict:
    changes = git(repo, 'status', '--short', '-uno')
    return dict(path=str(repo), commit=git(repo, 'rev-parse', 'HEAD'), branch=git(repo, 'branch', '--show-current'),
                tracked_changes=len(changes.splitlines()) if not changes.startswith('unknown') else changes)


def environment(repo: Path) -> dict:
    ps = T.path_sh_env()
    triton_build = Path(ps['PYTHONPATH'].split(':')[0])          # .../triton/build/lib.linux-x86_64-3.10
    triton_src = triton_build.parents[1]
    libtriton = triton_build / 'triton' / '_C' / 'libtriton.so'
    modules = {
        'operators/gemm_a16w16_generic.py': ROOT / 'operators' / 'gemm_a16w16_generic.py',
        'autotune/scripts/gemm_a16w16_generic.py': ROOT / 'autotune' / 'scripts' / 'gemm_a16w16_generic.py'}
    for name in ('kernel_repr', 'pid_preprocessing', 'gemm_config_utils'):     # imported by the generic kernel module
        modules[f'boltops/gemm/triton/{name}.py'] = repo / 'boltops' / 'gemm' / 'triton' / f'{name}.py'
    return dict(
        boltops=repo_info(repo),
        triton=dict(source=repo_info(triton_src), build=str(triton_build), libtriton_sha256=sha256(libtriton),
                    libtriton_mtime=datetime.fromtimestamp(libtriton.stat().st_mtime).isoformat(timespec='seconds')),
        compiler=dict(clang=T.CLANG, lld=T.LLD, path_sh=T.PATH_SH),
        pmd=dict(PMD_PATH=ps.get('PMD_PATH'), GPU_CHIP=ps.get('GPU_CHIP')),
        source_sha256={k: sha256(v) for k, v in modules.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('shape_dir', help='e.g. R014_12x64x2048 (directory under bf16_linear_gemm/)')
    ap.add_argument('--layout', choices=('TT', 'TN'), default='TT',
                    help='TT: B[K,N] row-major; TN: W[N,K] with K contiguous (gemm_a16w16 interface)')
    ap.add_argument('--config', help='flat config JSON or autotune result JSON (best.config); '
                                     'default: <shape_dir>/generic/best_config.json')
    ap.add_argument('--rounds', type=int, default=2, help='fresh processes, seed 29 + round')
    ap.add_argument('--dispatches', type=int, default=6, help='per round: 1 warmup + (n-1) timed')
    ap.add_argument('--out', help='NEW absolute output directory '
                                  '(default: <shape_dir>/generic/runs/<layout>_<time>)')
    ap.add_argument('--timeout', type=int, default=7200)
    args = ap.parse_args()

    m = re.fullmatch(r'([RG]\d+)_(\d+)x(\d+)x(\d+)', args.shape_dir)
    if not m:
        ap.error('shape_dir must look like R014_12x64x2048')
    gid, (M, N, K) = m.group(1), map(int, m.groups()[1:])
    cfg_path = Path(args.config) if args.config else ROOT / args.shape_dir / 'generic' / 'best_config.json'
    data = json.loads(cfg_path.read_text())
    if isinstance(data, dict) and 'best' in data:
        if not isinstance(data['best'], dict) or not isinstance(data['best'].get('config'), dict):
            ap.error('autotune result JSON must contain best.config as an object')
        config = data['best']['config']
        expected = dict(M=M, N=N, K=K, layout=args.layout)
        disagreements = [f'{key}={data[key]!r} (requested {value!r})'
                         for key, value in expected.items() if key in data and data[key] != value]
        if disagreements:
            print(f"warning: config result {cfg_path} disagrees: {', '.join(disagreements)}; "
                  'using shape_dir M/N/K and --layout, not result metadata', file=sys.stderr)
    else:
        config = data
    if args.dispatches < 2:
        ap.error('--dispatches must be >= 2 (1 warmup + at least 1 timed)')

    repo = Path(os.environ.get('BOLTOPS_REPO', DEFAULT_REPO)).resolve()
    T.REPO = repo                                         # PYTHONPATH of the worker: boltops.* imports
    out = Path(args.out) if args.out else (
        ROOT / args.shape_dir / 'generic' / 'runs' / f'{args.layout}_{datetime.now():%Y%m%d_%H%M%S}')
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f'output directory {out} is not empty; pass a new --out')
    out.mkdir(parents=True, exist_ok=True)

    rows, samples = [], []
    for r in range(args.rounds):
        case = dict(M=M, N=N, K=K, layout=args.layout, config=config, seed=29 + r, dispatches=args.dispatches,
                    full_check=True, keep_asm=True)
        res = T.run_case(case, out / f'round_{r}', timeout=args.timeout)
        acc = res.get('accuracy', {})
        rows.append(dict(round=r, seed=case['seed'], status=res['status'], median_cycles=res.get('median_cycles'),
                         median_us=res.get('median_us'), timed_cycles=res.get('timed_cycles'),
                         completed_wgs=res.get('completed_wgs'), active_cus=res.get('active_cus'),
                         grid=acc.get('grid'), checked_points=acc.get('sampled_points'),
                         mismatches=acc.get('mismatches'), finite=acc.get('finite'), guard=acc.get('guard'),
                         mean_abs=acc.get('mean_abs'), rounding_floor=acc.get('rounding_floor'),
                         resources=res.get('resources'), error=res.get('error')))
        print(f"round {r}: {res['status']}  cycles={res.get('median_cycles')}  us={res.get('median_us')}", flush=True)
        if res.get('timed_cycles') and res.get('median_us'):
            ghz = res['median_cycles'] / res['median_us'] / 1000
            samples += [(c, c / (ghz * 1000)) for c in res['timed_cycles']]

    ok = all(x['status'] == 'PASS' for x in rows) and bool(samples)
    summary = dict(
        shape_id=gid, shape=[M, N, K], layout=args.layout, dtype='bfloat16', config=config,
        config_file=str(cfg_path), correctness='PASS' if ok else 'FAIL',
        correctness_scope=f'all output elements vs gold = CPU FP32 reference rounded to BF16, '
                          f'|got-gold| <= {ACCURACY_ATOL} + {ACCURACY_RTOL}*|gold|; '
                          'mean |got-ref| <= 1.05x mean |gold-ref| + 1e-6 (ref is unrounded FP32); '
                          'all outputs finite; 128 sentinel values before/after output unchanged',
        accuracy_rtol=ACCURACY_RTOL, accuracy_atol=ACCURACY_ATOL,
        rounds=args.rounds, dispatches_per_round=args.dispatches,
        timed_samples=len(samples),
        median_cycles=statistics.median(c for c, _ in samples) if samples else None,
        median_us=statistics.median(u for _, u in samples) if samples else None,
        min_us=min(u for _, u in samples) if samples else None,
        max_us=max(u for _, u in samples) if samples else None,
        round_results=rows,
        scope='PMD kernel-only (system.gcSimCycles); excludes JIT compilation, allocation and host copies',
        environment=environment(repo), results_directory=str(out))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({k: summary[k] for k in ('shape', 'layout', 'correctness', 'median_cycles', 'median_us',
                                              'min_us', 'max_us', 'timed_samples')}, indent=2))
    print(f'summary: {out / "summary.json"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
