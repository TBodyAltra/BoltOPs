"""Original-only BF16 fused split-K launcher and PMD timer (12x64x2048 TT).

Uses only the local fused_kernel.py and best_config.json plus torch/Triton/PMD.
No dependency on the previous tuning directory or its benchmark helpers.
Run via `bash run.sh`; `--check-only` needs only the Python standard library.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SHAPE = (12, 64, 2048)
SOURCE_SHA256 = 'c4d2b5e596517e666a11dbdc4792f9a05530b22619063a7ae138dd5a56e1e8eb'


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + '\n')


def check_package(source=None):
    digest = hashlib.sha256((HERE / 'fused_kernel.py').read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Local kernel differs from the archived (heuristics-based) version')
    config = json.loads((HERE / 'best_config.json').read_text())
    assert [config[x] for x in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K',
                               'NUM_KSPLIT', 'SPLITK_BLOCK_SIZE', 'num_warps')] == [16, 32, 128, 8, 256, 1]
    assert config['USE_MLS'] and not config['WARP_SPECIALIZE']
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    assert not list(HERE.glob('*dynamic*.py')), 'Unexpected kernel variant in package'
    if source:
        for name in ('fused_kernel.py', 'best_config.json'):
            assert (HERE / name).read_bytes() == (Path(source) / name).read_bytes(), name
    return digest, config


def stats_of(path):
    text = path.read_text()
    assert text.count('Begin Simulation Statistics') == 1

    def number(key):
        values = re.findall(r'^' + re.escape(key) + r'\s+([\deE.+-]+)', text, re.M)
        assert len(values) == 1, (path, key)
        return float(values[0])

    cycles, freq = number('system.gcSimCycles'), number('gcFrequency')
    completed = sum(int(x) for x in re.findall(
        r'^system\.shaders\.CUs\d+\.completedWGs\s+(\d+)', text, re.M))
    return dict(dispatch_id=int(path.parent.name), cycles=cycles, freq_ghz=freq,
                us=cycles / (freq * 1000), completed_wgs=completed,
                stats_path=str(path.resolve()))


def worker(args):
    import torch
    import triton
    from triton.runtime import driver
    from fused_kernel import _gemm_bf16_fused_splitk_kernel as kernel

    digest, config = check_package()
    if driver.active.get_current_target().arch != 'gfx946':
        raise RuntimeError('This packaged configuration targets PMD gfx946')
    torch.set_num_threads(1)
    m, n, k = SHAPE
    bm, bn, bk, splits, sbs = [config[x] for x in (
        'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'NUM_KSPLIT', 'SPLITK_BLOCK_SIZE')]
    assert k == splits * sbs and sbs % bk == 0 and n % bn == 0
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    gen = torch.Generator().manual_seed(20261104)
    # The kernel bounds-checks M/N/K itself (heuristics + MLS boundary_check),
    # so A is the exact logical (m, k) tensor with no physical padding.
    ac = (torch.randn((m, k), generator=gen) * 0.25).to(torch.bfloat16)
    bc = (torch.randn((k, n), generator=gen) * 0.25).to(torch.bfloat16)
    expected = ac.float() @ bc.float()
    a, b = ac.cuda(), bc.cuda()
    guard, sentinel = 128, 123.0

    def guarded(shape, dtype, initial):
        size = 1
        for dim in shape:
            size *= dim
        host = torch.full((guard + size + guard,), sentinel, dtype=dtype)
        host[guard:-guard] = initial
        storage = host.cuda()
        return storage, storage[guard:-guard].view(shape)

    c_storage, c = guarded((m, n), torch.bfloat16, float('nan'))
    p_storage, partials = guarded((splits, m, n), torch.float32, float('nan'))
    count_storage, counters = guarded((tiles,), torch.int32, 0)
    launch_config = dict(config)
    torch.cuda.synchronize()
    # Workspace is reused. No allocation/initialization/copy/check occurs
    # between dispatches, and the kernel must reset its own counters.
    for _ in range(args.warmup + args.reps):
        compiled = kernel[(tiles * splits,)](
            a, b, c, partials, counters, m, n, k,
            *a.stride(), *b.stride(), *partials.stride(), *c.stride(), **launch_config,
        )
        torch.cuda.synchronize()
    paths = sorted(Path('g/0').glob('*/stats.txt'), key=lambda p: int(p.parent.name))
    assert len(paths) == args.warmup + args.reps, 'Unexpected GPU dispatches'
    records = [stats_of(p) for p in paths]
    assert all(r['completed_wgs'] == tiles * splits for r in records)
    samples = records[args.warmup:]
    cpu = c_storage.cpu()
    actual = cpu[guard:-guard].view(m, n).float()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)
    partial_cpu = p_storage.cpu()
    counter_cpu = count_storage.cpu()
    for storage in (cpu, partial_cpu, counter_cpu):
        assert bool((storage[:guard] == sentinel).all()), 'Prefix guard overwritten'
        assert bool((storage[-guard:] == sentinel).all()), 'Suffix guard overwritten'
    assert bool((counter_cpu[guard:-guard] == 0).all()), 'Counters did not reset'
    reference = torch.stack([ac[:, s*sbs:(s+1)*sbs].float()
                             @ bc[s*sbs:(s+1)*sbs].float() for s in range(splits)])
    torch.testing.assert_close(partial_cpu[guard:-guard].view(splits, m, n),
                               reference, rtol=0.0001, atol=0.0001)
    for kind in ('amdgcn', 'ttir', 'ttgir', 'llir'):
        if kind in compiled.asm:
            Path('kernel.' + kind).write_text(compiled.asm[kind])
    result = dict(shape=list(SHAPE), dtype='bfloat16', layout='TT', config=launch_config,
                  source_sha256=digest, logical_a_rows=m,
                  warmup_dispatches=records[:args.warmup], timed_dispatches=samples,
                  median_cycles=statistics.median(r['cycles'] for r in samples),
                  median_us=statistics.median(r['us'] for r in samples),
                  min_us=min(r['us'] for r in samples), max_us=max(r['us'] for r in samples),
                  correctness='PASS', partials='PASS', guards='PASS', counters='PASS',
                  max_abs_error=float((actual - expected).abs().max()),
                  n_regs=getattr(compiled, 'n_regs', None), metadata=str(compiled.metadata),
                  triton_path=triton.__file__, scope='PMD kernel-only; excludes compilation, allocation and counter initialization')
    dump('result.json', result)
    print(json.dumps(result, default=str), flush=True)


def env_for(dest):
    parent = os.environ
    toolchain = dict(
        PYTHONPATH=parent.get('PYTHONPATH', ''),
        TRITON_HIP_CLANG_PATH=parent.get('TRITON_HIP_CLANG_PATH', '/opt/dtk/aillvm/bin/clang-18'),
        TRITON_HIP_LLD_PATH=parent.get('TRITON_HIP_LLD_PATH', '/opt/dtk/aillvm/bin/ld.lld'),
        PMD_PATH=parent.get('PMD_PATH', '/opt/rocm/pmd'), GPU_CHIP=parent.get('GPU_CHIP', 'sb'))
    env = {k: v for k, v in parent.items()
           if not k.startswith(('TRITON_', 'GPU_', 'HIP_', 'XDG_', 'PMD_', 'CUDA_'))
           and k not in ('HSA_TOOLS_LIB', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
    env.update(toolchain)
    for name in ('home', 'cache', 'temp', 'xdg-cache', 'xdg-config', 'xdg-data', 'torch-cache'):
        (dest / name).mkdir()
    env.update(HOME=str(dest / 'home'), TRITON_CACHE_DIR=str(dest / 'cache'),
               TMPDIR=str(dest / 'temp'), TMP=str(dest / 'temp'), TEMP=str(dest / 'temp'),
               XDG_CACHE_HOME=str(dest / 'xdg-cache'), XDG_CONFIG_HOME=str(dest / 'xdg-config'),
               XDG_DATA_HOME=str(dest / 'xdg-data'), TORCHINDUCTOR_CACHE_DIR=str(dest / 'torch-cache'),
               GPU_ARGS='[]', GPU_DFLAGS="['Perf']", GPU_OUT='g', AMDGCN_USE_BUFFER_OPS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONHASHSEED='0',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', help='New output directory; existing directories are never overwritten')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--verify-source', help='Optional source directory for byte-for-byte copy validation')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.reps) < 1:
        parser.error('rounds, warmup and reps must be positive')
    digest, config = check_package(args.verify_source)
    if args.check_only:
        print(json.dumps(dict(status='PASS', source_sha256=digest, config=config), indent=2))
        return
    if args.worker:
        worker(args)
        return
    out = Path(args.out).resolve() if args.out else HERE / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for round_id in range(args.rounds):
        dest = out / f'round_{round_id}'
        dest.mkdir()
        env = env_for(dest)
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
               '--warmup', str(args.warmup), '--reps', str(args.reps)]
        dump(dest / 'command.json', dict(argv=cmd, cwd=str(dest),
             environment={key: env[key] for key in (
                 'PYTHONPATH', 'PMD_PATH', 'GPU_CHIP', 'TRITON_HIP_CLANG_PATH', 'TRITON_HIP_LLD_PATH')}))
        print(f'Round {round_id + 1}/{args.rounds}: {dest}', flush=True)
        with (dest / 'run.log').open('w') as log:
            subprocess.run(cmd, cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((dest / 'result.json').read_text())
        results.append(result)
        print(f"  correctness=PASS, median={result['median_us']:.6f} us", flush=True)
    samples = [s for result in results for s in result['timed_dispatches']]
    summary = dict(shape=list(SHAPE), dtype='bfloat16', layout='TT', config=results[0]['config'],
                   rounds=args.rounds, warmup_per_round=args.warmup, timed_per_round=args.reps,
                   samples=len(samples), median_cycles=statistics.median(s['cycles'] for s in samples),
                   median_us=statistics.median(s['us'] for s in samples),
                   min_us=min(s['us'] for s in samples), max_us=max(s['us'] for s in samples),
                   round_medians_us=[r['median_us'] for r in results],
                   correctness='PASS', source_sha256=digest, scope=results[0]['scope'],
                   results_directory=str(out))
    dump(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {out / 'summary.json'}", flush=True)


if __name__ == '__main__':
    main()
