#!/usr/bin/env python3
"""Isolated PMD timing/check harness for the R016 specialization (called by reproduce.py).
The installed Triton compiler and operators are read-only; all writes go under runs/<name>/.
REPO is resolved as the BoltOPs checkout containing this package.
"""
from __future__ import annotations
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
REPO = ROOT.parents[5]
SCRIPTS = REPO / 'docs/gemm/debug_tuning/bf16_linear_gemm/autotune/scripts'
sys.path.insert(0, str(SCRIPTS))
import tune_lib as T
T.REPO = REPO
BASELINE = SCRIPTS.parent / 'results_TT_dyn/shapes/R016_12x9216x2048_TT.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected(path):
    vals = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals[parts[0]] = float(parts[1])
            except ValueError:
                pass
    row = T.stats_of(path)
    row['stats_path'] = str(path.relative_to(ROOT))
    span = (vals['system.shaders.lastWaveEndTick'] - vals['system.shaders.firstWaveStartTick']) / 455
    row.update(wave_cycles=span, outside_wave_cycles=row['cycles']-span)
    for key in ('numReadRequests', 'numWriteRequests'):
        row['umc_'+key] = vals.get('system.umcs.'+key)
    for key in ('TccSendRespBandwidth','TccRecvRespBandwidth','TCCHitRate','TCPHitRate',
                'TCPCacheLineHitRate','missTCCRequests','TccSdRdReq256Ratio'):
        row[key] = vals.get('system.shaders.cache_system.'+key)
    for key in ('numCyclesWithNoIssue','numInstrTypeExecuted::SCA','numInstrTypeExecuted::MMOP',
                'numInstrTypeExecuted::VMEM','numInstrTypeExecuted::LDS','ldsBankConflict',
                'ldsBankAccesses','GlobalMemPipeline.cmdCreditStall','LocalMemPipeline.creditStall'):
        row[key] = sum(v for k,v in vals.items() if re.fullmatch(r'system\.shaders\.CUs\d+\.'+re.escape(key),k))
    for key in ('waitVmCounter','waitLgkmCounter','barrierCounter','snopCounter'):
        row[key] = sum(v for k,v in vals.items() if re.fullmatch(r'system\.shaders\.CUs\d+\.SIMD\d+\.'+key,k))
    return row


def worker(casepath):
    c = T.read(casepath)
    dest = Path(casepath).parent
    T.OP_PATH = dest / 'operator_snapshot.py'
    if c.get('rotate_b') or c.get('checked_entry') or c.get('check_each_dispatch'):
        import runtime_checks
        runtime_checks.worker(c, T, dest)
    else:
        T.worker(str(casepath))
    for kind in ('ttir','ttgir','llir','amdgcn','json','hsaco'):
        files = list((dest / 'cache').glob(f'*/_gemm_bf16_generic_kernel.{kind}'))
        if len(files) == 1:
            (dest / f'kernel.{kind}').write_bytes(files[0].read_bytes())


def run(name, case, timeout=7200):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('invalid run name')
    dest = ROOT / 'runs' / name
    dest.mkdir(parents=True, exist_ok=False)
    c = dict(M=12,N=9216,K=2048,layout='TT',seed=29,dispatches=3,
             full_check=True,keep=True,keep_asm=True)
    c.update(case)
    cfg = T.read(BASELINE)['best']['config']
    cfg.update(c.get('overrides', {}))
    c['config'] = c.get('config', cfg)
    T.dump(dest / 'case.json', c)
    source_path = Path(c.get('operator', T.OP_PATH)).resolve()
    source = source_path.read_bytes()
    (dest / 'operator_snapshot.py').write_bytes(source)
    (dest / 'bench_snapshot.py').write_bytes(Path(__file__).read_bytes())
    provenance = dict(operator=str(source_path),operator_sha256=hashlib.sha256(source).hexdigest(),
                      baseline_sha256=sha(BASELINE),tune_lib_sha256=sha(SCRIPTS/'tune_lib.py'))
    env = T.env_for(dest)
    env['PYTHONPATH'] = f'{ROOT}:' + env['PYTHONPATH']
    env['TRITON_ALWAYS_COMPILE'] = '0'
    env['GPU_DFLAGS'] = repr(['Perf','TT'] if c.get('trace') else ['Perf'])
    env.update({str(k):str(v) for k,v in c.get('env',{}).items()})
    T.dump(dest/'provenance.json',provenance)
    started = time.monotonic()
    print('START '+name,flush=True)
    with (dest/'worker.log').open('w') as log:
        process = subprocess.Popen([T.PYTHON,'-B',str(Path(__file__).resolve()),'worker',str(dest/'case.json')],
                                   cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                   preexec_fn=lambda:T._child_setup(None))
        try:
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            rc = -9
    row = dict(name=name,status='ERROR',returncode=rc,wall_s=time.monotonic()-started,
               case=c,provenance=provenance,directory=str(dest))
    try:
        assert rc==0,f'worker exit {rc}'
        comp=T.read(dest/'compile.json')
        row['resources']=comp
        if c.get('compile_only'):
            row['status']='COMPILED'
        else:
            check=T.read(dest/'check.json')
            paths=sorted((dest/'g').glob('*/*/stats.txt'),
                         key=lambda p:tuple(map(int,p.relative_to(dest/'g').parts[:-1])))
            per=2 if c['config'].get('NUM_KSPLIT',1)>1 else 1
            assert len(paths)==c['dispatches']*per,(len(paths),c['dispatches'],per)
            raw=[selected(p) for p in paths]
            groups=[raw[i:i+per] for i in range(0,len(raw),per)]
            timed=groups[1:] or groups
            cycles=[sum(x['cycles'] for x in g) for g in timed]
            us=[sum(x['us'] for x in g) for g in timed]
            assert all(g[0]['completed_wgs']==comp['grid'][0] for g in groups)
            row.update(status=check['status'],accuracy=check,dispatches=raw,timed_cycles=cycles,
                       median_cycles=statistics.median(cycles),median_us=statistics.median(us),
                       tflops=2*c['M']*c['N']*c['K']/statistics.median(us)/1e6)
    except Exception:
        row['error']=traceback.format_exc()
        row['log_tail']=(dest/'worker.log').read_text(errors='replace')[-6000:]
    T.dump(dest/'result.json',row)
    print(json.dumps({k:row.get(k) for k in ('name','status','median_cycles','median_us','wall_s','error')},ensure_ascii=False),flush=True)
    if row['status']=='ERROR':
        print(row.get('log_tail',''),flush=True)
    return row


def main():
    # Internal entry: run() re-launches this file as a worker process per case.
    if len(sys.argv)==3 and sys.argv[1]=='worker':
        worker(Path(sys.argv[2]).resolve())
        return
    sys.exit('bench.py is a library; use run.sh (reproduce.py)')


if __name__=='__main__':
    main()
