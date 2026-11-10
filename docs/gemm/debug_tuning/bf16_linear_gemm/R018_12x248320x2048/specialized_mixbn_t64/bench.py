#!/usr/bin/env python3
"""R018 (M=12, N=248320, K=2048, BF16, TT) PMD bench for the mixed-BN specialization (gfx946).

Archived copy of tmp/1106/r018_opt/bench.py. Changes vs that file: paths are resolved from this package
location (override the checkout with BOLTOPS_REPO), the default operator is the shared generic kernel
operators/gemm_a16w16_generic.py, the default shape is the full R018 shape, and the unused `hook` /
`pre_import` options (compiler-level experiments) were removed. Measurement code is unchanged.

Every run gets its own directory under runs/<name> with private HOME / Triton cache / PMD output,
a snapshot of the kernel source, the compiled IR/asm, a full-output accuracy check and per-dispatch PMD
diagnostics. Nothing global (Triton install, repo operator, tuned configs) is modified.

  python3 bench.py suite cases.json [--jobs 2]        # normally launched through run.sh

Case entry keys:
  name            run directory name (must not exist)
  shape           [M, N, K]                (default [12, 248320, 2048])
  config          full config              (default: best.config of the R018 autotune archive)
  config_delta    dict merged into config;  config_remove: list of keys to drop
  operator        kernel file defining _gemm_bf16_generic_kernel (default: shared generic operator)
  extra_constexpr extra constexpr names in config for custom kernels
  grid_fn         true: launch grid comes from grid_programs() in the operator file
  dispatches      default 2 (1 warm-up + 1 timed); full_check default True; compile_only: stop after compile
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('BOLTOPS_REPO') or ROOT.parents[5]).resolve()   # <repo>/docs/gemm/debug_tuning/bf16_linear_gemm/R018_*/specialized_*
LINEAR = REPO / 'docs/gemm/debug_tuning/bf16_linear_gemm'
SCRIPTS = LINEAR / 'autotune/scripts'
sys.path.insert(0, str(SCRIPTS))
import tune_lib as T  # noqa: E402
T.REPO = REPO
BASE = SCRIPTS.parent / 'results_TT_dyn/shapes/R018_12x248320x2048_TT.json'
DEFAULT_OP = LINEAR / 'operators/gemm_a16w16_generic.py'
UMC_BYTES = 256          # one UMC read request = 256 B (sector-256 reads, TccSdRdReq256Ratio=1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostic(path):
    v = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) > 1:
            try:
                x = float(f[1])
                if x == x:
                    v[f[0]] = x
            except ValueError:
                pass

    def total(p):
        return sum(x for k, x in v.items() if re.search(p, k))
    row = T.stats_of(path)
    cyc = row['cycles']
    active = total(r'CUs\d+\.SIMD\d+\.activeTimeCounter$')
    cu = total(r'CUs\d+\.totalCycles$')
    row['no_issue_frac'] = total(r'CUs\d+\.numCyclesWithNoIssue$') / cu if cu else None
    row['issue_frac'] = {t: total(r'CUs\d+\.numCyclesWithInstrTypeIssued::' + t + '$') / cu if cu else None
                         for t in ('VALU', 'MALU', 'SCA', 'MISC', 'VMEM', 'LDS')}
    row['wait_density'] = {t: total(r'CUs\d+\.SIMD\d+\.' + t + '$') / active if active else None for t in (
        'waitVmCounter', 'waitLgkmCounter', 'barrierCounter', 'depCounter',
        'memInstVmemStallCounter', 'memInstLdsStallCounter', 'emptyBufferCounter')}
    row['lds_conflict_per_inst'] = (total(r'CUs\d+\.ldsBankConflict$') / total(r'CUs\d+\.ldsInstNum$')
                                    if total(r'CUs\d+\.ldsInstNum$') else None)
    row['ta_stall'] = {t: total(r'CUs\d+\.tatd\.TATD\.' + t + '$') / cu if cu else None
                       for t in ('taSendStallBytcp', 'taSendStallByCredit', 'taCoalQueFullStall', 'taOutQueFullStall')}
    row['gmem_credit_stall'] = total(r'CUs\d+\.GlobalMemPipeline\.cmdCreditStall$') / cu if cu else None
    row['cache'] = {k.split('.')[-1]: x for k, x in v.items() if re.match(
        r'system\.shaders\.cache_system\.(?:totalTCCRequests|missTCCRequests|TCCHitRate|totalTCPRequests|'
        r'TCPHitRate|TCPCacheLineHitRate|TccRecvRespBandwidth|TccSendRespBandwidth)$', k)}
    rd = v.get('system.umcs.numReadRequests', 0)
    row['umc_read_req'] = rd
    row['umc_write_req'] = v.get('system.umcs.numWriteRequests')
    row['hbm_read_B_per_cycle'] = rd * UMC_BYTES / cyc if cyc else None
    cus = [(v.get(f'system.shaders.CUs{i:02d}.totalCycles'), v.get(f'system.shaders.CUs{i:02d}.completedWGs'))
           for i in range(48)]
    row['cu_cycles_max'] = max(c for c, _ in cus)
    row['cu_cycles_min'] = min(c for c, _ in cus)
    row['cu_wgs_hist'] = {str(w): sum(1 for _, x in cus if x == w) for w in sorted(set(x for _, x in cus))}
    row['first_wave_tick'] = v.get('system.shaders.firstWaveStartTick')
    row['last_wave_tick'] = v.get('system.shaders.lastWaveEndTick')
    row['host_s'] = v.get('system.hostSeconds')
    return row


def worker(path):
    path = Path(path).resolve()
    case = T.read(path)
    T.OP_PATH = Path(case['operator'])
    T.CONSTEXPR_KEYS += tuple(case.get('extra_constexpr', []))
    if case.get('grid_fn'):
        # Custom launch grid (e.g. mixed-BN tail panels): the operator module owns the formula, so the
        # kernel and the host grid can never disagree. Only this process's tune_lib copy is patched.
        spec = importlib.util.spec_from_file_location('grid_op', case['operator'])
        gmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gmod)
        cfg = dict(case['config'])
        T.num_tg = lambda M, N, c: gmod.grid_programs(M=M, N=N, K=case['K'], **{**cfg, **c})
    T.worker(str(path))
    for kind in ('ttir', 'ttgir', 'llir', 'amdgcn', 'json'):
        paths = list((path.parent / 'cache').glob(f'*/_gemm_bf16_generic_kernel.{kind}'))
        if len(paths) == 1:
            (path.parent / f'kernel.{kind}').write_bytes(paths[0].read_bytes())


def run(entry):
    e = dict(entry)
    name = e.pop('name')
    dest = ROOT / 'runs' / name
    if dest.exists():
        raise ValueError(f'Run exists: {dest}')
    dest.mkdir(parents=True)
    cfg = e.pop('config', None) or dict(T.read(BASE)['best']['config'])
    cfg.update(e.pop('config_delta', {}))
    for key in e.pop('config_remove', []):
        cfg.pop(key, None)
    M, N, K = e.pop('shape', [12, 248320, 2048])
    c = dict(M=M, N=N, K=K, layout='TT', seed=29, dispatches=2, full_check=True, keep=True, keep_asm=True,
             config=cfg)
    timeout = e.pop('timeout', 4 * 3600)
    c.update(e)
    source = Path(c.get('operator') or DEFAULT_OP)
    source = (source if source.is_absolute() else ROOT / source).resolve()
    snap = dest / 'kernel_source.py'
    snap.write_bytes(source.read_bytes())
    (dest / 'bench_source.py').write_bytes(Path(__file__).read_bytes())
    c.update(operator=str(snap), source_operator=str(source), source_sha256=sha(source),
             tune_lib_sha256=sha(SCRIPTS / 'tune_lib.py'), baseline_sha256=sha(BASE))
    T.dump(dest / 'case.json', c)
    env = T.env_for(dest)
    env['PYTHONPATH'] = str(ROOT) + ':' + env['PYTHONPATH']
    env['GPU_DFLAGS'] = repr(['Perf', 'TT'] if c.get('trace') else ['Perf'])
    env['TRITON_ALWAYS_COMPILE'] = '1'
    for k, val in (c.get('env') or {}).items():
        env[k] = str(val)
    start = time.monotonic()
    print('START', name, [M, N, K], flush=True)
    with (dest / 'worker.log').open('w') as log:
        p = subprocess.Popen(['/usr/bin/python3', '-B', str(Path(__file__).resolve()), 'worker', str(dest / 'case.json')],
                             cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = -9
    row = dict(name=name, status='ERROR', returncode=rc, wall_s=time.monotonic() - start, case=c)
    try:
        assert rc == 0, f'worker exit {rc}'
        row['resources'] = T.read(dest / 'compile.json')
        if c.get('compile_only'):
            row['status'] = 'COMPILED'
        else:
            check = T.read(dest / 'check.json')
            paths = sorted((dest / 'g').glob('*/*/stats.txt'),
                           key=lambda q: tuple(map(int, q.relative_to(dest / 'g').parts[:-1])))
            assert len(paths) == c['dispatches'], (len(paths), c['dispatches'])
            raw = [diagnostic(q) for q in paths]
            timed = raw[1:] or raw
            cy = [x['cycles'] for x in timed]
            us = [x['us'] for x in timed]
            assert all(x['completed_wgs'] == row['resources']['grid'][0] for x in raw)
            row.update(status=check['status'], accuracy=check, dispatches=raw, timed_cycles=cy,
                       median_cycles=statistics.median(cy), median_us=statistics.median(us),
                       tflops=2 * M * N * K / statistics.median(us) / 1e6,
                       b_bytes_per_cycle=K * N * 2 / statistics.median(cy))
    except Exception:
        row['error'] = traceback.format_exc()
        row['log_tail'] = (dest / 'worker.log').read_text(errors='replace')[-5000:]
    T.dump(dest / 'result.json', row)
    keys = ('name', 'status', 'median_cycles', 'median_us', 'b_bytes_per_cycle', 'wall_s', 'error')
    print(json.dumps({k: row.get(k) for k in keys}, ensure_ascii=False), flush=True)
    if row['status'] == 'ERROR':
        print(row.get('log_tail', ''), flush=True)
    return row


def summary(row):
    d = (row.get('dispatches') or [{}])[-1]
    return {k: row.get(k) for k in ('name', 'status', 'median_cycles', 'median_us', 'b_bytes_per_cycle',
                                     'resources', 'timed_cycles')} | {
        'hbm_read_B_per_cycle': d.get('hbm_read_B_per_cycle'), 'no_issue_frac': d.get('no_issue_frac'),
        'wait_density': d.get('wait_density'), 'cu_wgs_hist': d.get('cu_wgs_hist'),
        'mismatches': (row.get('accuracy') or {}).get('mismatches')}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'worker':
        worker(sys.argv[2])
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['suite'])
    ap.add_argument('cases', type=Path)
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()
    entries = T.read(args.cases)
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rows = list(ex.map(run, entries))
    T.dump(args.cases.with_name(args.cases.stem + '_results.json'), [summary(r) for r in rows])
    sys.exit(0 if all(r['status'] in ('PASS', 'COMPILED') for r in rows) else 1)


if __name__ == '__main__':
    main()
