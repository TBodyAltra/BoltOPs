# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Single-file BF16 WASP replay: repository public API/full grid or one-TG PERF.

Reads only record.json and repository code/configuration. No extracted kernel,
patch, helper runner or snapshot is required. Historical results are not changed.
--dry-run checks the four artifacts on CPU, without importing Torch or Triton.
"""
import argparse
import ast
import builtins
from copy import deepcopy
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'boltops/gemm/config.py').is_file())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_digest(path, symbol):
    source = Path(path).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == symbol)
    start = min([node.lineno] + [n.lineno for n in node.decorator_list])
    return hashlib.sha256(''.join(source.splitlines(keepends=True)[start-1:node.end_lineno]).encode()).hexdigest()


def replay_options(record):
    config = record['config']
    data = json.loads((REPO / config['path']).read_text())
    options = deepcopy(data['profiles'][config['profile']][config['bucket']])
    for key in ('wasp', 'wdra', 'use_mls'):
        assert options.pop(key) is True, key
    options['USE_MATRIX_STORE'] = options.pop('use_matrix_store')
    options['HAS_BIAS'] = False
    return {key: tuple(value) if isinstance(value, list) else value for key, value in options.items()}


def validation_cases(seed):
    """Match R001's independent random / all-ones / random checks."""
    return [dict(index=index, seed=seed + 1000*index,
                 input_kind='ones' if index == 1 else 'uniform_random')
            for index in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=Path(__file__).with_name('record.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--single-tg-perf', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--triton-build', type=Path)
    args = parser.parse_args()
    record_path = args.record.resolve()
    record = json.loads(record_path.read_text())
    output = args.output.resolve()
    if output.exists():
        parser.error('Output directory must not already exist; archived artifacts are never overwritten.')
    for artifact in ('assembly', 'single_tg_perf'):
        item = record[artifact]
        assert digest(record_path.parent / item['path']) == item['sha256'], artifact
    assert digest(REPO / record['config']['path']) == record['config']['sha256']
    assert kernel_digest(REPO / record['source']['path'], record['source']['symbol']) == record['source']['kernel_sha256']
    options = replay_options(record)
    m, n, k = record['shape']
    bm, bn, bk = [options[f'BLOCK_SIZE_{d}'] for d in 'MNK']
    assert record['layout'] == 'TT' and record['environment']['cu_count'] == 48
    assert m % bm == n % bn == k % bk == 0
    single = args.single_tg_perf
    repeats = 3 if single else record['benchmark']['repeats']
    mode = record['benchmark']['timing_mode']
    original_triton = Path(record['environment']['triton']['path']).parent.parent
    triton_build = args.triton_build.resolve() if args.triton_build else original_triton
    plan = dict(requirement=record['requirement'], shape=[m,n,k],
        kernel=record['source']['path'], symbol=record['source']['symbol'],
        launch='repository JIT directly' if single else 'repository public gemm_a16w16, backend=triton',
        grid=[1] if single else [record['full_grid_TGs']], repeats=repeats,
        correctness_dispatches=3, validation_cases=validation_cases(record['benchmark']['seed']),
        total_dispatches=3+repeats, timing_input='historical seed, normal random BF16',
        warmup_dispatches=1, historical_timing_mode=mode,
        output=str(output), triton_build=str(triton_build),
        note='CPU-only plan; no historical measurement is rerun or overwritten.')
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    # Re-exec before any Torch/Triton/boltops import, as in the R001 benchmark.
    marker = 'BOLTOPS_WASP_REPLAY_RECORD'
    if os.environ.get(marker) != str(record_path):
        env = dict(os.environ)
        for key, value in record['environment']['environment_variables'].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        if single:
            env.update(record['single_tg_perf']['environment_overrides'])
        # Relocate private runtime state, not compiler/device flags.
        env.update(GPU_OUT='g', TRITON_CACHE_DIR=str(output/'cache'),
            HOME=str(output/'home'), TMPDIR=str(output/'temp'),
            RPY_HISTORY_LOG=str(output/'history.log'), RPY_HISTORY_SIZE='1024',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1')
        env[marker] = str(record_path)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    if not triton_build.is_dir():
        parser.error('Recorded Triton build is unavailable; pass --triton-build.')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('cache', 'home', 'temp'):
        (output / name).mkdir()
    os.chdir(output)
    # PMD history lock must be private to this output, not a shared /tmp lock.
    original_open = builtins.open
    shared_lock = '/tmp/' + getpass.getuser() + '.history.log.lock'
    def private_open(file, *positional, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file) == shared_lock:
            file = output / 'history.lock'
        return original_open(file, *positional, **kwargs)
    builtins.open = private_open
    sys.path[:0] = [str(REPO), str(triton_build)]
    import torch
    import triton
    public = importlib.import_module('boltops.gemm.gemm_a16w16')
    kernels = importlib.import_module('boltops.gemm.triton.gemm_a16w16_wasp')
    fn = getattr(kernels, record['source']['symbol'])
    target = triton.runtime.driver.active.get_current_target()
    assert target.arch == 'gfx946' and torch.cuda.get_device_properties(0).multi_processor_count == 48
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator().manual_seed(record['benchmark']['seed'])
    ac = torch.randn((m,k), generator=generator).bfloat16()
    bc = torch.randn((k,n), generator=generator).bfloat16()
    a, b = ac.cuda(), bc.cuda()
    w = b.T  # Public weights [N,K], stride=(1,N); logical B is contiguous [K,N].
    rows, cols = (bm, bn) if single else (m, n)
    guard, suffix = 128, bm*n + bm*bn
    initial = torch.full((guard + rows*n + suffix,), 123., dtype=torch.bfloat16)
    initial[guard:guard+rows*n].view(rows,n)[:, :cols] = float('nan')
    storage = initial.cuda()
    c = storage[guard:guard+rows*n].view(rows,n)
    grid = (1,) if single else (record['full_grid_TGs'],)
    positional = (a, b, c, b, m,n,k, *a.stride(), *b.stride(), n,1)
    handle = fn.warmup(*positional, grid=grid, **options)
    handle._init_handles()
    assert handle.n_spills == 0
    (output/'kernel.amdgcn').write_text(handle.asm['amdgcn'])
    if not single:
        class ObserveLaunch:
            def __getitem__(self, dispatch_grid):
                real = fn[dispatch_grid]
                def launch(*values, **kwargs):
                    result = real(*values, **kwargs)
                    assert result is handle, 'public API selected a different compiled kernel/config'
                    return result
                return launch
        setattr(kernels, record['source']['symbol'], ObserveLaunch())
    def reference(ac, bc):
        fp32 = ac[:rows].float() @ bc[:,:cols].float()
        golden = fp32.bfloat16().float()
        return fp32, golden, float((golden-fp32).abs().mean())

    def launch(index, phase, golden_fp32, golden, rounding):
        storage.copy_(initial)
        torch.cuda.synchronize()
        before = set(Path('g').glob('*/*/stats.txt'))
        if single:
            assert fn[grid](*positional, **options) is handle
        else:
            assert public.gemm_a16w16(a, w, out=c, backend='triton') is c
        torch.cuda.synchronize()
        paths = set(Path('g').glob('*/*/stats.txt')) - before
        assert len(paths) == 1, paths
        path = paths.pop()
        text = path.read_text()
        assert 'End Simulation Statistics' in text
        def stat(key):
            values = re.findall(r'^' + re.escape(key) + r'\s+([\deE.+-]+)', text, re.M)
            assert len(values) == 1, key
            return float(values[0])
        cycles, frequency = stat('system.gcSimCycles'), stat('gcFrequency')
        completed = sum(map(int, re.findall(r'^system\.shaders\.CUs\d+\.completedWGs\s+(\d+)',text,re.M)))
        mmops = sum(map(int, re.findall(r'^system\.shaders\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)',text,re.M)))
        assert completed == grid[0] and mmops == grid[0]*2*bm*bn*k//8192
        host = storage.cpu()
        view = host[guard:guard+rows*n].view(rows,n)
        actual = view[:,:cols].float()
        error = (actual-golden_fp32).abs()
        canary = bool((host[:guard]==123).all() and (host[guard+rows*n:]==123).all()
                      and (not single or (view[:,cols:]==123).all()))
        accuracy = dict(elements=rows*cols, finite=bool(torch.isfinite(actual).all()),
            canary_ok=canary, outside_tolerance=int((~torch.isclose(actual,golden,atol=.02,rtol=.01)).sum()),
            mean_abs_vs_fp32=float(error.mean()), rounding_floor=rounding, atol=.02, rtol=.01)
        assert accuracy['finite'] and canary and accuracy['outside_tolerance'] == 0, accuracy
        assert accuracy['mean_abs_vs_fp32'] <= 1.05*rounding+1e-6, accuracy
        sample = dict(index=index, phase=phase, status='PASS',
            kernel_cycles=cycles, kernel_us=cycles/(frequency*1000), completed_TGs=completed,
            stats_path=str(path), stats_sha256=digest(path), accuracy=accuracy)
        if single:
            perfs = [p for p in path.parent.rglob('*.perf') if not p.name.startswith('stats_')]
            assert len(perfs) == 1, perfs
            sample['perf_path'] = str(perfs[0])
        print(json.dumps(dict(requirement=record['requirement'], single_tg=single,
            phase=phase, index=index, status='PASS', kernel_cycles=cycles,
            kernel_us=sample['kernel_us'], accuracy=accuracy)), flush=True)
        return sample

    validation = []
    for case in validation_cases(record['benchmark']['seed']):
        gen = torch.Generator().manual_seed(case['seed'])
        va = (torch.rand((m,k), generator=gen) - .5).bfloat16()
        vb = (torch.rand((k,n), generator=gen) - .5).bfloat16()
        if case['input_kind'] == 'ones':
            va.fill_(1)
            vb.fill_(1)
        a.copy_(va)
        b.copy_(vb)
        sample = launch(case['index'], 'correctness', *reference(va, vb))
        sample.update(seed=case['seed'], input_kind=case['input_kind'])
        validation.append(sample)
        del va, vb
    # R001 separates validation from timing. Restore the historical normal-random
    # inputs rather than timing its final correctness case (uniform random).
    a.copy_(ac)
    b.copy_(bc)
    golden_fp32, golden, rounding = reference(ac, bc)
    samples = [launch(index, 'warmup' if index == 0 else 'timed',
                      golden_fp32, golden, rounding) for index in range(repeats)]
    if single:
        selected = min(samples[1:], key=lambda s:s['kernel_cycles'])
        shutil.copyfile(selected['perf_path'], output/'single_tg.perf')
        performance = dict(kernel_cycles=selected['kernel_cycles'], kernel_us=selected['kernel_us'],
                           selection='minimum of repeats 1 and 2; repeat 0 excluded')
    else:
        used = samples[1:2] if mode == 'existing_first_timed' else samples[1:]
        performance = {key: statistics.median(s[key] for s in used) for key in ('kernel_cycles','kernel_us')}
        performance.update(kernel_tops=2*m*n*k/(performance['kernel_us']*1e6),
            effective_mmac_util_percent=100*2*m*n*k/(performance['kernel_cycles']*48*4096),
            selected_indexes=[s['index'] for s in used], selection=record['performance']['selection'])
    result = dict(requirement=record['requirement'], status='PASS', performance=performance,
        validation=validation, samples=samples, single_tg=single,
        correctness_dispatches=len(validation), total_dispatches=len(validation)+len(samples),
        timing_input=dict(seed=record['benchmark']['seed'], input_kind='normal_random'),
        assembly_sha256=digest(output/'kernel.amdgcn'),
        historical_assembly_matches=digest(output/'kernel.amdgcn') == record['assembly']['sha256'],
        benchmark_sha256=digest(__file__),
        source_sha256=digest(REPO/record['source']['path']),
        note='New replay result; historical record.json was not modified.')
    if single:
        result['selected'] = dict(selected, sha256=digest(output/'single_tg.perf'))
        result['historical_perf_matches'] = result['selected']['sha256'] == record['single_tg_perf']['sha256']
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', performance=performance)), flush=True)


if __name__ == '__main__':
    main()
