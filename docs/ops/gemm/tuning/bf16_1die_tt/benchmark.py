# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Replay one cold BF16 TT dispatch through the repository public API on PMD.

The default mode measures one cold full-grid public-API dispatch with CU0 PERF.
--single-tg-perf runs the same compiled persistent kernel with grid=(1,), keeping
the original shape, strides and NUM_SMS. TG0 processes tile IDs 0, full_grid_TGs,
2*full_grid_TGs, etc. Three independent correctness cases precede three timing
repeats; repeat 0 is excluded and the faster of repeats 1 and 2 supplies the PERF.
--dry-run verifies local artifacts and configuration without Torch/GPU imports.
"""
import argparse
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
import sys

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'boltops/gemm/config.py').is_file())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def options(record):
    cfg = record['config']
    data = json.loads((REPO / cfg['path']).read_text())
    profile = cfg.get('profile', 'DTYPE=bfloat16-OUT=bfloat16')
    bucket = cfg.get('bucket', f"M_EQ_{record['shape'][0]}")
    params = deepcopy(data['profiles'][profile][bucket])
    assert params.pop('kernel') == record['source']['symbol']
    params.update(zip('MNK', record['shape']))
    params.update(B_K_CONTIGUOUS=False, FORCE_MASKED_STORE=False)
    if 'resolved_launch' in cfg:
        assert params == cfg['resolved_launch'], 'production config differs from measured launch'
    return params


def single_tg_plan(record):
    m, n, k = record['shape']
    columns = (n + 255) // 256
    tile_ids = list(range(0, ((m + 255) // 256) * columns, record['full_grid_TGs']))
    tiles = [(i // columns * 256, i % columns * 256) for i in tile_ids]
    elements = sum(min(256, m - row) * min(256, n - col) for row, col in tiles)
    return tile_ids, tiles, elements


def measure_single_tg(record, params, fn, torch, lib, output):
    """Measure TG0's original persistent work without changing its binary."""
    m, n, k = record['shape']
    tile_ids, tiles, elements = single_tg_plan(record)
    guard = record['benchmark'].get('guard_elements_each_side', 65536)
    ai = torch.full((m * k + 2 * guard,), float('nan'), dtype=torch.bfloat16)
    bi = torch.full((k * n + 2 * guard,), float('nan'), dtype=torch.bfloat16)
    ci = torch.full((m * n + 2 * guard,), 123., dtype=torch.bfloat16)
    ci[guard:-guard] = float('nan')
    ar, br, cr = ai.cuda(), bi.cuda(), ci.cuda()
    a, b, c = ar[guard:-guard].view(m, k), br[guard:-guard].view(k, n), cr[guard:-guard].view(m, n)
    handle = fn.warmup(a, b, c, grid=(1,), **params)
    handle._init_handles()
    assert handle.n_spills == 0
    (output / 'kernel.amdgcn').write_text(handle.asm['amdgcn'])
    (output / 'kernel.hsaco').write_bytes(handle.asm['hsaco'])
    assert digest(output / 'kernel.amdgcn') == record['assembly']['sha256'], 'single/full compiled assembly differs'

    def prepare(seed, kind):
        generator = torch.Generator().manual_seed(seed)
        if kind == 'ones':
            ah = torch.ones((m, k), dtype=torch.bfloat16)
            bh = torch.ones((k, n), dtype=torch.bfloat16)
        elif kind == 'uniform_random':
            ah = (torch.rand((m, k), generator=generator) - .5).bfloat16()
            bh = (torch.rand((k, n), generator=generator) - .5).bfloat16()
        else:
            ah = (torch.randn((m, k), generator=generator) * .5).bfloat16()
            bh = (torch.randn((k, n), generator=generator) * .5).bfloat16()
        ai[guard:-guard].copy_(ah.reshape(-1))
        bi[guard:-guard].copy_(bh.reshape(-1))
        ar.copy_(ai)
        br.copy_(bi)
        golden = [(ah[row:row + 256].float() @ bh[:, col:col + 256].float()).bfloat16().float()
                  for row, col in tiles]
        return golden

    def launch(index, phase, golden):
        cr.copy_(ci)
        torch.cuda.synchronize()
        before = set((output / 'g').glob('*/*/stats.txt'))
        assert fn[(1,)](a, b, c, **params) is handle
        torch.cuda.synchronize()
        paths = set((output / 'g').glob('*/*/stats.txt')) - before
        assert len(paths) == 1, paths
        path = paths.pop()
        text = path.read_text()
        assert 'End Simulation Statistics' in text
        def stat(name):
            values = re.findall(r'^' + re.escape(name) + r'\s+([\deE.+-]+)', text, re.M)
            assert len(values) == 1, name
            return float(values[0])
        groups = {int(cu): int(count) for cu, count in re.findall(
            r'^system\.shaders\.CUs(\d+)\.completedWGs\s+(\d+)', text, re.M)}
        mmops = sum(int(value) for value in re.findall(
            r'^system\.shaders\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)', text, re.M))
        assert len(groups) == 48 and sum(groups.values()) == groups[0] == 1, groups
        assert mmops == len(tile_ids) * 2 * 256 * 256 * k // 8192, mmops
        perfs = [p for p in path.parent.glob('*.perf') if not p.name.startswith('stats_')]
        assert len(perfs) == 1 and perfs[0].stat().st_size > 0, perfs
        host = cr.cpu()
        observed = host[guard:-guard].view(m, n)
        outside = nonfinite = 0
        maximum = 0.
        for (row, col), expected in zip(tiles, golden):
            actual = observed[row:row + expected.shape[0], col:col + expected.shape[1]].float()
            outside += int((~torch.isclose(actual, expected, atol=.001, rtol=.01)).sum())
            nonfinite += int((~torch.isfinite(actual)).sum())
            maximum = max(maximum, float((actual - expected).abs().max()))
            observed[row:row + expected.shape[0], col:col + expected.shape[1]] = float('nan')
        canary = bool((host[:guard] == 123).all() and (host[-guard:] == 123).all())
        unselected = bool(torch.isnan(observed).all())
        inputs = bool(torch.equal(ar.cpu().view(torch.uint8), ai.view(torch.uint8)) and
                      torch.equal(br.cpu().view(torch.uint8), bi.view(torch.uint8)))
        accuracy = dict(elements=elements, mismatches=outside, nonfinite=nonfinite, max_abs=maximum,
            canary_ok=canary, unselected_output_nan=unselected, inputs_byte_identical=inputs,
            atol=.001, rtol=.01)
        assert outside == nonfinite == 0 and canary and unselected and inputs, accuracy
        cycles, frequency = stat('system.gcSimCycles'), stat('gcFrequency')
        sample = dict(index=index, phase=phase, status='PASS', dispatch_cycles=cycles,
            kernel_us=cycles / (frequency * 1000),
            wave_interval_us=(stat('system.shaders.lastWaveEndTick') - stat('system.shaders.firstWaveStartTick')) / 1e6,
            completed_TGs=1, stats_path=str(path.relative_to(output)), stats_sha256=digest(path),
            perf_path=str(perfs[0].relative_to(output)), accuracy=accuracy)
        print(json.dumps(dict(requirement=record['requirement'], phase=phase, index=index,
            kernel_us=sample['kernel_us'], status='PASS', accuracy=accuracy)), flush=True)
        return sample

    validation = []
    for index in range(3):
        seed = record['benchmark']['seed'] + 1000 * index
        kind = 'ones' if index == 1 else 'uniform_random'
        golden = prepare(seed, kind)
        sample = launch(index, 'correctness', golden)
        sample.update(seed=seed, input_kind=kind)
        validation.append(sample)
    golden = prepare(record['benchmark']['seed'], 'normal_random')
    samples = [launch(index, 'warmup' if index == 0 else 'timed', golden) for index in range(3)]
    selected = min(samples[1:], key=lambda sample: sample['dispatch_cycles'])
    shutil.copyfile(output / selected['perf_path'], output / 'single_tg.perf')
    result = dict(requirement=record['requirement'], status='PASS', single_tg=True,
        shape=record['shape'], grid=[1], persistent_tile_ids=tile_ids,
        performance=dict(kernel_cycles=selected['dispatch_cycles'], kernel_us=selected['kernel_us']),
        validation=validation, samples=samples, selected=dict(selected, sha256=digest(output / 'single_tg.perf')),
        assembly_sha256=digest(output / 'kernel.amdgcn'), hsaco_sha256=digest(output / 'kernel.hsaco'),
        benchmark_sha256=digest(__file__), source_sha256=digest(REPO / record['source']['path']),
        config_sha256=digest(REPO / record['config']['path']), public_api_sha256=digest(REPO / record['public_api']['path']),
        runtime=dict(torch=torch.__version__, libtriton_sha256=digest(lib.__file__)))
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', performance=result['performance'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--single-tg-perf', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--current-source', action='store_true',
                        help='explicitly replay the current public path')
    parser.add_argument('--triton-build', type=Path)
    args = parser.parse_args()
    record_path = args.record.resolve()
    record = json.loads(record_path.read_text())
    output = args.output.resolve()
    if output.exists():
        parser.error('Output directory must be new; archived artifacts are never overwritten.')
    assert digest(__file__) == record['benchmark']['sha256']
    for name in ('assembly', 'single_tg_perf'):
        item = record[name]
        assert digest(record_path.parent / item['path']) == item['sha256'], name
    fingerprints = {name: digest(REPO / record[name]['path'])
                    for name in ('source', 'config', 'public_api')}
    changed = [name for name, sha256 in fingerprints.items() if sha256 != record[name]['sha256']]
    if changed and not (args.current_source or args.dry_run):
        parser.error(f"recorded source differs for {', '.join(changed)}; "
                     "pass --current-source to measure the current public path")
    params = options(record)
    m, n, k = record['shape']
    grid = min(48, ((m + 255) // 256) * ((n + 255) // 256))
    assert record['layout'] == 'TT' and record['environment']['dies'] == 1
    assert record['environment']['cu_count'] == 48 and grid == record['full_grid_TGs']
    build_path = record['environment'].get('toolchain', {}).get(
        'triton_build', record['environment']['environment_variables']['PYTHONPATH'])
    build = args.triton_build or Path(build_path)
    plan = dict(requirement=record['requirement'], shape=[m, n, k], layout='TT',
        launch='repository public gemm_a16w16, backend=' + record['public_api']['backend'], source=record['source']['path'],
        config=record['config']['path'], grid=[grid], seed=record['benchmark']['seed'],
        repeats=1, execution_warmups=0, compile_warmup_calls=0, total_gemm_dispatches=1,
        host_padding=False, full_output_validation=True,
        single_tg_scope=record['single_tg_perf']['scope'], triton_build=str(build), output=str(output),
        source_mode='current' if args.current_source else 'recorded',
        changed_fingerprints=changed,
        archived_measurement='Original experimental source/binary only; production replay is a new measurement.')
    if args.single_tg_perf:
        tile_ids, _, elements = single_tg_plan(record)
        plan.update(grid=[1], repeats=3, execution_warmups=1, correctness_dispatches=3,
            total_gemm_dispatches=6, full_output_validation=False, selected_output_elements=elements,
            persistent_tile_ids=tile_ids, single_tg_scope='TG0 original persistent work, original shape/strides/NUM_SMS; independent grid=(1,) dispatch')
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    # Configure the process before any Torch/Triton/boltops imports.
    marker = 'BOLTOPS_MASKED_ACTIVE_REPLAY_RECORD'
    if os.environ.get(marker) != str(record_path):
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(('GPU_', 'HIP_', 'HSA_', 'PMD_', 'TRITON_', 'CUDA_', 'ROCR_', 'DCU_', 'RPY_', 'RBPY_', 'SOC_')) or key == 'LD_PRELOAD':
                env.pop(key, None)
        for key, value in record['environment']['environment_variables'].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        if args.single_tg_perf:
            env.update(record['single_tg_perf']['environment_overrides'])
        env.update(GPU_OUT='g', TRITON_CACHE_DIR=str(output / 'cache'), HOME=str(output / 'HOME'),
            TMPDIR=str(output / 'temp'), RPY_HISTORY_LOG=str(output / 'history.log'),
            RPY_HISTORY_SIZE='1024', PYTHONPATH=str(build), OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1')
        env[marker] = str(record_path)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    assert build.is_dir(), 'recorded Triton build unavailable; pass --triton-build'
    output.mkdir(parents=True, exist_ok=False)
    for name in ('cache', 'HOME', 'temp'):
        (output / name).mkdir()
    os.chdir(output)
    original_open = builtins.open
    shared_lock = '/tmp/' + getpass.getuser() + '.history.log.lock'
    def private_open(file, *positional, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file) == shared_lock:
            file = output / 'history.lock'
        return original_open(file, *positional, **kwargs)
    builtins.open = private_open
    sys.path[:0] = [str(REPO), str(build)]
    import torch
    import triton
    import triton._C.libtriton as lib
    public = importlib.import_module('boltops.gemm.gemm_a16w16')
    kernels = importlib.import_module('boltops.gemm.gluon.gemm_a16w16')
    fn = getattr(kernels, record['source']['symbol'])
    target = triton.runtime.driver.active.get_current_target()
    assert target.arch == 'gfx946' and target.warp_size == 64
    assert torch.cuda.get_device_properties(0).multi_processor_count == 48
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.single_tg_perf:
        return measure_single_tg(record, params, fn, torch, lib, output)
    generator = torch.Generator().manual_seed(record['benchmark']['seed'])
    ah = (torch.randn((m, k), generator=generator) * .5).bfloat16()
    bh = (torch.randn((k, n), generator=generator) * .5).bfloat16()
    guard = record['benchmark'].get('guard_elements_each_side', 65536)
    def guarded(host):
        initial = torch.full((host.numel() + 2 * guard,), float('nan'), dtype=torch.bfloat16)
        initial[guard:-guard] = host.reshape(-1)
        raw = initial.cuda()
        return raw, raw[guard:-guard].view(host.shape), initial
    ar, a, ai = guarded(ah)
    br, b, bi = guarded(bh)
    ci = torch.full((m * n + 2 * guard,), 123., dtype=torch.bfloat16)
    ci[guard:-guard] = float('nan')
    cr = ci.cuda()
    c = cr[guard:-guard].view(m, n)
    w = b.T
    launched = []
    class ObserveLaunch:
        def __getitem__(self, dispatch_grid):
            real = fn[dispatch_grid]
            def invoke(*values, **kwargs):
                expected = {key: tuple(value) if isinstance(value, list) else value for key, value in params.items()}
                # FORCE_MASKED_STORE=False is also the kernel's default.
                actual = dict(kwargs)
                actual.setdefault('FORCE_MASKED_STORE', False)
                assert actual == expected, 'public API selected different launch options'
                resolved = dispatch_grid(actual) if callable(dispatch_grid) else dispatch_grid
                assert tuple(resolved) == (grid,)
                result = real(*values, **kwargs)
                launched.append(result)
                return result
            return invoke
    setattr(kernels, record['source']['symbol'], ObserveLaunch())
    torch.cuda.synchronize()
    assert public.gemm_a16w16(a, w, out=c, backend='auto') is c
    torch.cuda.synchronize()
    assert len(launched) == 1, 'expected exactly one production GEMM dispatch'
    compiled = launched[0]
    (output / 'kernel.amdgcn').write_text(compiled.asm['amdgcn'])
    (output / 'kernel.hsaco').write_bytes(compiled.asm['hsaco'])
    paths = list((output / 'g').glob('*/*/stats.txt'))
    assert len(paths) == 1, paths
    text = paths[0].read_text()
    assert 'End Simulation Statistics' in text
    def stat(name):
        matches = re.findall(r'^' + re.escape(name) + r'\s+([\deE.+-]+)', text, re.M)
        assert len(matches) == 1, name
        return float(matches[0])
    cycles, frequency = stat('system.gcSimCycles'), stat('gcFrequency')
    groups = dict((int(i), int(v)) for i, v in re.findall(r'^system\.shaders\.CUs(\d+)\.completedWGs\s+(\d+)', text, re.M))
    mmops = sum(int(v) for v in re.findall(r'^system\.shaders\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)', text, re.M))
    assert len(groups) == 48 and sum(groups.values()) == grid and groups[0] == 1
    perfs = [p for p in (output / 'g').rglob('*.perf') if not p.name.startswith('stats_')]
    assert len(perfs) == 1 and perfs[0].stat().st_size > 0, perfs
    shutil.copyfile(perfs[0], output / 'single_tg.perf')
    host = cr.cpu()
    actual = host[guard:-guard].view(m, n)
    outside = nonfinite = mismatch = 0
    maximum = 0.
    bf = bh.float()
    for row in range(0, m, 128):
        expected = (ah[row:row + 128].float() @ bf).bfloat16().float()
        observed = actual[row:row + 128].float()
        outside += int((~torch.isclose(observed, expected, atol=.001, rtol=.01)).sum())
        nonfinite += int((~torch.isfinite(observed)).sum())
        mismatch += int((observed != expected).sum())
        error = (observed - expected).abs()
        if bool(torch.isfinite(error).all()):
            maximum = max(maximum, float(error.max()))
    canary = bool(torch.equal(host[:guard], ci[:guard]) and torch.equal(host[-guard:], ci[-guard:]))
    inputs = bool(torch.equal(ar.cpu().view(torch.uint8), ai.view(torch.uint8)) and
                  torch.equal(br.cpu().view(torch.uint8), bi.view(torch.uint8)))
    validation = dict(status='PASS' if outside == nonfinite == 0 and canary and inputs else 'FAIL',
        elements=m * n, atol=.001, rtol=.01, outside_tolerance=outside, nonfinite=nonfinite,
        mismatch=mismatch, max_abs=maximum, canary_ok=canary, inputs_byte_identical=inputs)
    performance = dict(kernel_cycles=cycles, kernel_us=cycles / (frequency * 1000),
        kernel_tops=2 * m * n * k / (cycles / (frequency * 1000) * 1e6),
        useful_mmac_util_percent=100 * 2 * m * n * k / (cycles * 48 * 4096),
        instruction_mmac_util_percent=100 * mmops * 8192 / (cycles * 48 * 4096),
        completed_TGs=grid, samples=1, execution_warmups=0)
    result = dict(requirement=record['requirement'], status=validation['status'],
        measurement='New production replay; archived experiment record is unchanged',
        performance=performance, validation=validation, benchmark_sha256=digest(__file__),
        source_sha256=digest(REPO / record['source']['path']), config_sha256=digest(REPO / record['config']['path']),
        public_api_sha256=digest(REPO / record['public_api']['path']),
        assembly_sha256=digest(output / 'kernel.amdgcn'), hsaco_sha256=digest(output / 'kernel.hsaco'),
        assembly_matches_archived=digest(output / 'kernel.amdgcn') == record['assembly']['sha256'],
        stats_sha256=digest(paths[0]), perf_sha256=digest(output / 'single_tg.perf'),
        single_tg_scope=record['single_tg_perf']['scope'],
        runtime=dict(torch=torch.__version__, triton=triton.__version__, libtriton_sha256=digest(lib.__file__)))
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    assert validation['status'] == 'PASS', validation
    assert compiled.n_spills == 0
    print(json.dumps(dict(status='PASS', performance=performance)), flush=True)


if __name__ == '__main__':
    main()
