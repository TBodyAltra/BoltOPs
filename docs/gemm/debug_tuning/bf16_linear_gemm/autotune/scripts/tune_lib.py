#!/usr/bin/env python3
"""PMD measurement worker/driver for gemm_a16w16_generic._gemm_bf16_generic_kernel.

Single case = one (M, N, K, layout, config) in its own process, HOME/cache and
cwd (PMD writes ./g/<n>/<m>/stats.txt per dispatch). 1 warmup + 3 timed
dispatches, median gcSimCycles of the timed ones. Accuracy: finite/poison check
over the full output plus a random sample of output points against a CPU FP32
reference (no full-output reference).

  python tune_lib.py worker <case_dir>/case.json     (internal)

The kernel is the dynamic-M/N/K one (no EVEN_MNK constexpr, optional bias): M, N, K, strides and
all tails are runtime values, so one config compiles once for every shape. Notes for callers:
  * SPLITK_BLOCK_SIZE (when NUM_KSPLIT > 1) must be a multiple of BLOCK_SIZE_K; empty splits are
    allowed (the kernel stores zero partials). The partial buffer is NaN-initialised so an
    unwritten partial fails the check.
  * USE_MATRIX_STORE is applied by the kernel heuristic _USE_MSTORE only when it is safe (N % BLOCK_SIZE_N == 0,
    no split-K / bias); otherwise the kernel silently uses the pointer store. The kernel heuristics
    (_M/_N/_K_ALIGNED) also pick the aligned, check-free variant automatically.
  * case['poison_pad']: A/B/W are views of NaN-padded buffers, so any read outside [M, K] / [K, N]
    that is not zero-filled turns the output into NaN (off by default: it changes the leading
    dimensions, i.e. the measured access pattern).
The previous (EVEN_MNK) version of this file and of the operator is kept in legacy_pre_dyn/.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = Path('/public/home/duyubo/bolt/xk/BoltOPs')
PATH_SH = '/public/home/duyubo/triton_all_version/triton_3.6/triton/path.sh'   # the Triton/compiler environment to use
PYTHON = '/usr/bin/python3'
# path.sh enables clang-22 (clang-18 is its commented-out alternative). With this Triton build clang-22 fails at the
# assembler step ("unknown directive .amdhsa_code_object_version") even for a trivial kernel, clang-18 works.
# Override with  AUTOTUNE_CLANG=/opt/dtk/aillvm/bin/clang-22  once that is fixed.
CLANG = os.environ.get('AUTOTUNE_CLANG', '/opt/dtk/aillvm/bin/clang-18')
LLD = '/opt/dtk/aillvm/bin/ld.lld'
OP_PATH = HERE / 'gemm_a16w16_generic.py'
PEAK_FLOPS_PER_CU_CYCLE = 4096     # BF16 dense MMAC peak per CU per cycle (from best-configs.json)
NUM_CU = 48

# Constexpr (kernel-level) keys; everything else in a config is a launch option.
CONSTEXPR_KEYS = ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M',
                  'NUM_KSPLIT', 'SPLITK_BLOCK_SIZE', 'WARP_SPECIALIZE', 'USE_MLS',
                  'USE_MATRIX_STORE', 'LOOP_UNROLL_FACTOR')


ACTIVE_PGIDS = set()     # process groups of running simulator workers (killed on hard stop)


def kill_active():
    for pg in list(ACTIVE_PGIDS):
        try:
            os.killpg(pg, signal.SIGKILL)
        except ProcessLookupError:
            pass


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.' + path.name, delete=False) as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + '\n')
        tmp = f.name
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


_PATH_SH_ENV = {}


def path_sh_env():
    """Variables exported by path.sh (PYTHONPATH, TRITON_HIP_CLANG_PATH, PMD_PATH, GPU_CHIP, ...), read once."""
    if not _PATH_SH_ENV:
        out = subprocess.run(['bash', '-c', f'env -i bash -c "source {PATH_SH}; env -0"'], capture_output=True,
                             check=True).stdout.decode()
        for kv in out.split('\0'):
            k, _, v = kv.partition('=')
            if k in ('PYTHONPATH', 'TRITON_HIP_CLANG_PATH', 'TRITON_HIP_LLD_PATH', 'PMD_PATH', 'GPU_CHIP'):
                _PATH_SH_ENV[k] = v
        assert _PATH_SH_ENV.get('PYTHONPATH') and _PATH_SH_ENV.get('TRITON_HIP_CLANG_PATH'), _PATH_SH_ENV
    return _PATH_SH_ENV


def env_for(dest: Path):
    env = os.environ.copy()
    for k in list(env):
        if k.startswith(('TRITON_', 'GPU_', 'HIP_', 'XDG_', 'PMD_', 'CUDA_')) or k in (
                'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'HSA_TOOLS_LIB'):
            env.pop(k, None)
    for sub in ('home', 'cache', 'temp', 'xdg-cache', 'xdg-config', 'xdg-data', 'torch-cache'):
        (dest / sub).mkdir(exist_ok=True)
    ps = path_sh_env()
    env.update(
        PYTHONPATH=f"{ps['PYTHONPATH'].rstrip(':')}:{REPO}", PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1', PYTHONHASHSEED='0',
        GPU_CHIP=ps.get('GPU_CHIP', 'sb'), GPU_ARGS='[]', GPU_DFLAGS="['Perf']", GPU_OUT='g',
        PMD_PATH=ps.get('PMD_PATH', '/opt/rocm/pmd'),
        AMDGCN_USE_BUFFER_OPS='1', HSA_TOOLS_LIB='/opt/rocm/lib/xprofiler/libperf_gen_helper.so',
        TRITON_HIP_CLANG_PATH=CLANG, TRITON_HIP_LLD_PATH=LLD,
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        HOME=str(dest / 'home'), TRITON_CACHE_DIR=str(dest / 'cache'), TMPDIR=str(dest / 'temp'),
        TMP=str(dest / 'temp'), TEMP=str(dest / 'temp'), XDG_CACHE_HOME=str(dest / 'xdg-cache'),
        XDG_CONFIG_HOME=str(dest / 'xdg-config'), XDG_DATA_HOME=str(dest / 'xdg-data'),
        TORCHINDUCTOR_CACHE_DIR=str(dest / 'torch-cache'))
    return env


def split_config(cfg: dict):
    const = {k: cfg[k] for k in CONSTEXPR_KEYS if k in cfg}
    opts = {k: v for k, v in cfg.items() if k not in CONSTEXPR_KEYS}
    return const, opts


def num_tg(M, N, cfg):
    return (-(-M // cfg['BLOCK_SIZE_M'])) * (-(-N // cfg['BLOCK_SIZE_N'])) * cfg.get('NUM_KSPLIT', 1)


def stats_of(path: Path):
    text = path.read_text()
    assert text.count('Begin Simulation Statistics') == 1, ('non-unique stats block', str(path))

    def number(key):
        vals = re.findall(r'^' + re.escape(key) + r'\s+([\deE.+-]+)', text, re.M)
        assert len(vals) == 1, ('stats field', key, len(vals))
        return float(vals[-1])

    cycles, freq = number('system.gcSimCycles'), number('gcFrequency')
    wg = re.findall(r'^system\.shaders\.(CUs\d+)\.completedWGs\s+(\d+)', text, re.M)
    mm = re.findall(r'^system\.shaders\.(CUs\d+)\.numInstrTypeExecuted::MMOP\s+(\d+)', text, re.M)
    return dict(cycles=cycles, freq_ghz=freq, us=cycles / (freq * 1000),
                completed_wgs=sum(int(v) for _, v in wg), wave_mmac=sum(int(v) for _, v in mm),
                cu_count=len(wg), active_cus=sum(int(v) > 0 for _, v in wg))


# ----------------------------------------------------------------------------- worker
def worker(casepath: str):
    import importlib.util
    import torch
    import triton
    from triton.runtime import driver

    c = read(casepath)
    dest = Path(casepath).parent
    M, N, K, layout = c['M'], c['N'], c['K'], c['layout']
    cfg = dict(c['config'])
    const, opts = split_config(cfg)
    from triton.backends.hcu.compiler import HIPOptions       # unknown launch options would be silently ignored
    unknown = sorted(set(opts) - set(HIPOptions.__dataclass_fields__))
    if unknown:
        raise ValueError(f'unknown launch options (would be ignored): {unknown}')
    const.setdefault('GROUP_SIZE_M', 1)
    const.setdefault('NUM_KSPLIT', 1)
    const.setdefault('USE_MLS', False)
    const.setdefault('USE_MATRIX_STORE', False)
    const.setdefault('WARP_SPECIALIZE', False)
    ksplit = const['NUM_KSPLIT']
    bm, bn, bk = const['BLOCK_SIZE_M'], const['BLOCK_SIZE_N'], const['BLOCK_SIZE_K']
    # 0: ignored when NUM_KSPLIT==1 (K stays runtime); else ceil(K/ksplit) rounded up to a BLOCK_SIZE_K multiple
    const.setdefault('SPLITK_BLOCK_SIZE', 0 if ksplit == 1 else -(-(-(-K // ksplit)) // bk) * bk)
    assert driver.active.get_current_target().arch == 'gfx946'

    spec = importlib.util.spec_from_file_location('gemm_a16w16_generic', str(OP_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    kernel = mod._gemm_bf16_generic_kernel

    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(c.get('seed', 29))
    ac = torch.randn((M, K), generator=gen).to(torch.bfloat16)
    if layout == 'TT':
        bc = torch.randn((K, N), generator=gen).to(torch.bfloat16)
    else:                                   # TN: logical B[K,N] = W[N,K].T, K contiguous
        bc = torch.randn((N, K), generator=gen).to(torch.bfloat16).T
    if c.get('poison_pad'):
        # views into NaN-padded buffers: reads outside [M, K] / [K, N] must be zero-filled by the kernel
        pad = 2 * max(bm, bn, bk)
        nan = float('nan')
        abuf = torch.full((M + pad, K + pad), nan, dtype=torch.bfloat16)
        abuf[:M, :K] = ac
        a = abuf.cuda()[:M, :K]
        if layout == 'TT':
            bbuf = torch.full((K + pad, N + pad), nan, dtype=torch.bfloat16)
            bbuf[:K, :N] = bc
            b = bbuf.cuda()[:K, :N]
        else:
            wbuf = torch.full((N + pad, K + pad), nan, dtype=torch.bfloat16)
            wbuf[:N, :K] = bc.T
            b = wbuf.cuda()[:N, :K].T
    else:
        a, b = ac.cuda(), bc.cuda()
    # output with poison guards; split-K partials live in an FP32 3-D buffer
    if ksplit == 1:
        h = torch.full((128 + M * N + 128,), 123., dtype=torch.bfloat16)
        h[128:-128] = float('nan')
        storage = h.cuda()
        out = storage[128:-128].view(M, N)
        c_args = (out, 0, out.stride(0), out.stride(1))
    else:                                   # split-K: FP32 partials (ksplit, M, N) + separate reduce kernel
        sbs = const['SPLITK_BLOCK_SIZE']
        assert sbs % bk == 0, ('SPLITK_BLOCK_SIZE must be a multiple of BLOCK_SIZE_K', sbs, bk)   # empty splits are allowed
        h = torch.full((128 + M * N + 128,), 123., dtype=torch.bfloat16)
        h[128:-128] = float('nan')
        storage = h.cuda()
        out = storage[128:-128].view(M, N)
        # host fill: no extra GPU dispatch; every split (even an empty one) must overwrite its partial
        pp = torch.full((ksplit, M, N), float('nan'), dtype=torch.float32).cuda()
        c_args = (pp, pp.stride(0), pp.stride(1), pp.stride(2))
        from boltops.gemm.triton.splitk_reduce import _gemm_splitk_reduce_kernel as reduce_kernel
        rgrid = (-(-M // 32), -(-N // 32))
    # 4th argument = bias_ptr (unused: ADD_BIAS defaults to False; `a` is a placeholder). SPLITK_BLOCK_SIZE goes in via **const.
    args = (a, b, c_args[0], a, M, N, K, *a.stride(), *b.stride(), c_args[1], c_args[2], c_args[3])
    grid = (num_tg(M, N, const),)
    t0 = time.monotonic()
    compiled = kernel.warmup(*args, grid=grid, **const, **opts)
    compiled._init_handles()
    asm = compiled.asm['amdgcn']
    if c.get('keep_asm', True):
        (dest / 'kernel.amdgcn').write_text(asm)
    norm = '\n'.join(l for l in re.sub(r';[^\n]*', '', asm).splitlines() if l.strip() and '.file' not in l)
    info = dict(grid=grid, compile_wall_s=time.monotonic() - t0, shared=compiled.metadata.shared,
                n_regs=compiled.n_regs, n_spills=compiled.n_spills,
                num_warps=compiled.metadata.num_warps,
                asm_sha=hashlib.sha256(norm.encode()).hexdigest())
    dump(dest / 'compile.json', info)
    if info['shared'] > 131072:
        raise RuntimeError(f"LDS {info['shared']} B exceeds 131072 B")
    if c.get('compile_only'):
        return
    n_disp = c.get('dispatches', 4)
    for i in range(n_disp):
        storage.copy_(h)
        torch.cuda.synchronize()
        kernel[grid](*args, **const, **opts)
        if ksplit > 1:
            reduce_kernel[rgrid](pp, out, None, M, N, pp.stride(0), pp.stride(1), pp.stride(2),
                                 out.stride(0), out.stride(1), 32, 32, ksplit, triton.next_power_of_2(ksplit),
                                 ADD_BIAS=False, activation="", use_activation=False,
                                 KERNEL_NAME="_gemm_a16w16_reduce_kernel")
        torch.cuda.synchronize()
    # --- sampled accuracy ----------------------------------------------------
    actual = out.cpu()
    finite = bool(torch.isfinite(actual).all())
    raw = storage.cpu()
    guard = bool((raw[:128] == 123).all() and (raw[-128:] == 123).all())
    g2 = torch.Generator().manual_seed(1234)
    ns = min(M, 32)
    rows = torch.randperm(M, generator=g2)[:ns]
    cols = torch.randint(0, N, (256,), generator=g2)
    cols[:min(8, N)] = torch.arange(min(8, N))
    cols[-8:] = torch.arange(N - 8, N) if N >= 8 else cols[-8:]
    if c.get('full_check'):                 # generality check on small shapes: every output element
        rows, cols = torch.arange(M), torch.arange(N)
    ref = ac[rows].float() @ bc[:, cols].float()
    got = actual[rows][:, cols]
    gold = ref.to(torch.bfloat16)
    mism = int((~torch.isclose(got, gold, atol=.02, rtol=.01)).sum())
    floor = float((gold.float() - ref).abs().mean())
    mean = float((got.float() - ref).abs().mean())
    acc_ok = finite and guard and mism == 0 and mean <= 1.05 * floor + 1e-6
    dump(dest / 'check.json', dict(status='PASS' if acc_ok else 'FAIL', finite=finite, guard=guard,
                                   sampled_points=int(got.numel()), mismatches=mism,
                                   mean_abs=mean, rounding_floor=floor, **info))


def _cleanup(dest: Path, c: dict):
    if c.get('keep'):
        return
    subs = ('g', 'home', 'temp', 'torch-cache', 'xdg-cache', 'xdg-config', 'xdg-data')
    if not c.get('keep_cache'):                 # compile-only passes keep the Triton cache for the later simulation
        subs += ('cache',)
    for sub in subs:
        shutil.rmtree(dest / sub, ignore_errors=True)


def _child_setup(cpu):
    """Runs in the worker child: die with the parent (no orphaned simulators), pin to one CPU."""
    try:
        ctypes.CDLL('libc.so.6').prctl(1, int(signal.SIGKILL))      # PR_SET_PDEATHSIG
    except Exception:
        pass
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})


def run_case(c: dict, dest: Path, timeout=7200, cpu=None):
    """Run one case in a subprocess; returns result dict (also saved as result.json)."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest / 'case.json', c)
    env = env_for(dest)
    t0 = time.monotonic()
    with (dest / 'worker.log').open('w') as log:
        p = subprocess.Popen([PYTHON, '-B', str(HERE / 'tune_lib.py'), 'worker', str(dest / 'case.json')],
                             cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                             preexec_fn=lambda: _child_setup(cpu))
        ACTIVE_PGIDS.add(p.pid)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = -9
        finally:
            ACTIVE_PGIDS.discard(p.pid)
    row = dict(case=c, returncode=rc, worker_wall_s=time.monotonic() - t0, status='ERROR')
    try:
        assert rc == 0, f'worker exit {rc}'
        if (dest / 'check.json').exists():
            chk = read(dest / 'check.json')
            row.update(accuracy=chk, status=chk['status'], resources={k: chk[k] for k in ('shared', 'n_regs', 'n_spills')})
        elif c.get('compile_only'):
            row.update(status='COMPILED', resources=read(dest / 'compile.json'))
            dump(dest / 'result.json', row)
            _cleanup(dest, c)
            return row
        paths = sorted((dest / 'g').glob('*/*/stats.txt'),
                       key=lambda f: tuple(int(x) for x in f.relative_to(dest / 'g').parts[:-1]))
        n = c.get('dispatches', 4)
        per = 2 if c['config'].get('NUM_KSPLIT', 1) > 1 else 1      # split-K: main + reduce dispatch per iteration
        assert len(paths) == n * per, (len(paths), n, per)
        raw = [stats_of(p) for p in paths]
        groups = [raw[i * per:(i + 1) * per] for i in range(n)]
        samples = [dict(g[0], cycles=sum(s['cycles'] for s in g), us=sum(s['us'] for s in g)) for g in groups]
        timed = samples[1:] or samples
        cyc = statistics.median(s['cycles'] for s in timed)
        us = statistics.median(s['us'] for s in timed)
        flops = 2 * c['M'] * c['N'] * c['K']
        logical_mmac = flops / 8192            # wave-MMAC count (as in earlier harness)
        row.update(
            timed_cycles=[s['cycles'] for s in timed], median_cycles=cyc, median_us=us,
            tflops=flops / (us * 1e-6) / 1e12,
            mmac_util=flops / cyc / (PEAK_FLOPS_PER_CU_CYCLE * NUM_CU),   # fraction of 48-CU peak
            completed_wgs=samples[-1]['completed_wgs'], active_cus=samples[-1]['active_cus'],
            wave_mmac=samples[-1]['wave_mmac'], logical_wave_mmac=logical_mmac)
        if samples[-1]['completed_wgs'] != row['accuracy']['grid'][0]:
            row['status'] = 'FAIL'
            row['note'] = 'completed workgroups != grid'
    except Exception:
        row.update(status='ERROR', error=traceback.format_exc())
        if (dest / 'worker.log').exists():
            row['worker_log_tail'] = (dest / 'worker.log').read_text()[-4000:]
    dump(dest / 'result.json', row)
    _cleanup(dest, c)
    return row


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'worker':
        worker(sys.argv[2])
    else:
        sys.exit(__doc__)
