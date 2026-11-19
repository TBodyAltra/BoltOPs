# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Replay R029 V2 through the public GEMM entry on sbx2 (2 dies / 96 CUs).

The public entry loads the recorded 96-CU JSON with backend=auto and no explicit
config. Every call verifies the selected kernel, 96-TG grid and launch options.
The full protocol checks three inputs, then one warmup and three timed calls.
--single-tg-perf intercepts that same public launch and executes only TG0 three
times, preserving full MNK/strides/options; select a timed XCD0 repeat.
Source the recorded Triton path.sh first. --compile-only checks the public
dispatch and binary without launching GPU work; --dry-run needs no Torch.
"""
import argparse
import builtins
from copy import deepcopy
import configparser
import getpass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import shutil
import struct
import sys

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'boltops/gemm/config.py').is_file())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def executable_sections(data):
    assert data[:5] == b'\x7fELF\x02'
    offset, = struct.unpack_from('<Q', data, 40)
    size, count, names = struct.unpack_from('<HHH', data, 58)
    entries = [struct.unpack_from('<IIQQQQIIQQ', data, offset+i*size) for i in range(count)]
    strings = data[entries[names][4]:entries[names][4]+entries[names][5]]
    result = {}
    for entry in entries:
        name = strings[entry[0]:].split(b'\0', 1)[0].decode()
        if name in ('.text', '.rodata'):
            content = data[entry[4]:entry[4]+entry[5]]
            result[name] = dict(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    assert '.text' in result
    return result


def measured_source(record, output):
    for key in ('source', 'config', 'public_entry', 'default_config'):
        item = record[key]
        assert digest(REPO / item['path']) == item['sha256'], f'{key} differs from the recorded implementation'
    return REPO


def launch_options(record):
    return {key: tuple(value) if isinstance(value, list) else value
            for key, value in record['launch']['options'].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=Path(__file__).with_name('record.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--single-tg-perf', action='store_true')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    record_path = args.record.resolve()
    record = json.loads(record_path.read_text())
    if args.single_tg_perf and args.compile_only:
        parser.error('--single-tg-perf and --compile-only are mutually exclusive')
    measured_source(record, None)
    config = record['config']
    recorded_options = json.loads((REPO / config['path']).read_text())['profiles'][config['profile']][config['bucket']]
    actual_options = deepcopy(recorded_options)
    assert actual_options.pop('wasp') is True
    assert actual_options.pop('wdra') is False
    assert actual_options.pop('use_mls') is True
    actual_options['USE_MATRIX_STORE'] = actual_options.pop('use_matrix_store')
    actual_options['HAS_BIAS'] = False
    assert actual_options == record['launch']['options'], 'config and launch options differ'
    args.chip = 'sbx2'
    assert record['environment']['cu_count'] == 96 and record['environment']['dies'] == 2
    assert record['dtype']['A'] == record['dtype']['W'] == record['dtype']['output'] == 'BF16'
    params = launch_options(record)
    output = args.output.resolve()
    if output.exists():
        parser.error('output directory must be new')
    m, n, k = record['shape']
    symbol = record['source']['symbol']
    small = record['requirement'] == 'R001'
    mls = symbol == '_gemm_fp16_mls_kernel'
    tiles = math.ceil(m / params['BLOCK_SIZE_M']) * math.ceil(n / params['BLOCK_SIZE_N'])
    full_grid = min(params['NUM_SMS'], tiles) if 'NUM_SMS' in params else tiles
    assert record['full_grid_TGs'] == full_grid
    assert params.get('NUM_SMS', 96) == 96
    grid = 1 if args.single_tg_perf else full_grid
    tile_ids = list(range(0, tiles, 96)) if 'NUM_SMS' in params else [0]
    bm, bn = params['BLOCK_SIZE_M'], params['BLOCK_SIZE_N']
    columns = math.ceil(n / bn)
    selected_tiles = [(i // columns * bm, i % columns * bn) for i in tile_ids]
    repeats = 3 if args.single_tg_perf else record['benchmark']['repeats']
    aggregation = record['performance']['aggregation']
    protocol = 'cold' if repeats == 1 else 'first_timed' if aggregation.startswith('first timed repeat') else 'warm_median'
    plan = dict(requirement=record['requirement'], shape=record['shape'], chip=args.chip,
                kernel=symbol, options=params, grid=[grid], protocol=protocol, repeats=repeats,
                correctness_dispatches=0 if protocol == 'cold' else 3,
                launch='public boltops.gemm.gemm_a16w16, backend=auto, no explicit config',
                config_path=config['path'], compile_only=args.compile_only)
    plan.update(single_tg=args.single_tg_perf, full_grid=[full_grid],
                selected_tile_ids=tile_ids if args.single_tg_perf else None)
    if args.single_tg_perf:
        plan.update(protocol='single_tg_perf', correctness_dispatches=0,
                    each_dispatch_validated=True, execution_warmups=1,
                    selection='fastest timed XCD0 repeat; repeat 0 excluded')
    if args.compile_only:
        plan.update(correctness_dispatches=0, repeats=0)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    marker = 'BOLTOPS_R029_PUBLIC_REPLAY_RECORD'
    if os.environ.get(marker) != str(record_path):
        env = dict(os.environ)
        # Retain path.sh's compiler, PMD and Python paths; preserve per-kernel
        # compiler flags and PMD instrumentation from the archived experiment.
        inherited = {'PYTHONPATH', 'PMD_PATH', 'LD_LIBRARY_PATH', 'TRITON_HIP_CLANG_PATH', 'TRITON_HIP_LLD_PATH'}
        for key, value in record['environment']['environment_variables'].items():
            if key in inherited:
                continue
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        env.update(GPU_CHIP=args.chip, GPU_OUT=str(output / 'g'),
                   TRITON_CACHE_DIR=str(output / 'cache'), TRITON_HOME=str(output / 'triton-home'),
                   TMPDIR=str(output / 'temp'), RPY_HISTORY_LOG=str(output / 'history.log'),
                   RPY_HISTORY_SIZE='1024', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        if args.single_tg_perf:
            env.update(GPU_DFLAGS="['StatLog','TT0','Perf']",
                       HSA_TOOLS_LIB='/opt/rocm/lib/xprofiler/libperf_gen_helper.so')
        env.pop('BOLTOPS_GEMM_CONFIG_DIR', None)
        env[marker] = str(record_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('cache', 'triton-home', 'temp'):
        (output / folder).mkdir()
    save(output / 'plan.json', plan)
    save(output / 'config.json', record['config'])
    save(output / 'environment.json', {key: value for key, value in os.environ.items()
         if key.startswith(('GPU_', 'PMD_', 'TRITON_', 'AMDGCN_', 'OMP_', 'MKL_', 'OPENBLAS_', 'RPY_'))
         or key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'HSA_TOOLS_LIB')})
    os.chdir(output)
    original_open = builtins.open
    shared_lock = '/tmp/' + getpass.getuser() + '.history.log.lock'

    def private_open(file, *positional, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file) == shared_lock:
            file = output / 'history.lock'
        return original_open(file, *positional, **kwargs)

    builtins.open = private_open
    source_root = measured_source(record, output)
    sys.path.insert(0, str(source_root))
    import torch
    import triton
    import triton._C.libtriton as lib

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    target = triton.runtime.driver.active.get_current_target()
    assert target.arch == 'gfx946' and target.warp_size == 64
    reported_cus = torch.cuda.get_device_properties(0).multi_processor_count
    assert reported_cus == (96 if args.chip == 'sbx2' else 48), reported_cus
    module = importlib.import_module(record['source']['path'][:-3].replace('/', '.'))
    fn = getattr(module, symbol)
    guard = 256 if small else 128 if mls else 65536
    bshape = (n, k) if small else (k, n)

    def guarded(shape, fill):
        initial = torch.full((math.prod(shape) + 2 * guard,), fill, dtype=torch.bfloat16)
        raw = initial.cuda()
        return initial, raw, raw[guard:-guard].view(shape)

    ai, ar, a = guarded((m, k), float('nan'))
    bi, br, b = guarded(bshape, float('nan'))
    ci, cr, c = guarded((m, n), 123.)
    ci[guard:-guard] = float('nan')
    positional = (a, b, c, b, m, n, k, *a.stride(), *b.stride(), *c.stride()) if mls else (a, b, c)
    from boltops.gemm import gemm_a16w16

    handle = None
    compiling = True
    public_calls = []

    class PublicLaunch:
        def __getitem__(self, requested_grid):
            def run(*actual, **options):
                nonlocal handle
                requested = requested_grid(options)
                assert requested == (full_grid,), requested
                assert options == params, (options, params)
                assert actual[4:] == positional[4:]
                assert all(x.data_ptr() == y.data_ptr() and x.stride() == y.stride()
                           for x, y in zip(actual[:4], positional[:4]))
                if compiling:
                    assert handle is None
                    handle = fn.warmup(*actual, grid=requested, **options)
                else:
                    assert fn[(grid,)](*actual, **options) is handle
                public_calls.append(dict(requested_grid=list(requested),
                                         executed_grid=None if compiling else [grid],
                                         options_match=True, compiling=compiling))
            return run

    setattr(module, symbol, PublicLaunch())

    def public_launch():
        assert gemm_a16w16(a, b.T, out=c, backend='auto') is c

    public_launch()
    assert handle is not None
    compiling = False
    handle._init_handles()
    assert handle.n_spills == 0
    (output / 'kernel.amdgcn').write_text(handle.asm['amdgcn'])
    (output / 'kernel.hsaco').write_bytes(handle.asm['hsaco'])
    assert executable_sections(handle.asm['hsaco']) == record['assembly']['executable_sections'], 'compiled kernel differs from measured binary'
    assert dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared) == record['resources'], 'launch resources differ'
    public_entry = dict(name='boltops.gemm.gemm_a16w16', backend='auto', explicit_config=False,
                        source_sha256=record['public_entry']['sha256'],
                        config_path=config['path'], config_sha256=config['sha256'],
                        requested_grid=[full_grid], calls=public_calls)
    if args.compile_only:
        save(output / 'compile.json', dict(status='PASS', public_entry=public_entry,
             options=params, executable_sections=executable_sections(handle.asm['hsaco']),
             resources=record['resources'], hsaco_sha256=digest(output / 'kernel.hsaco'),
             benchmark_sha256=digest(__file__)))
        print('PASS: public auto dispatch, config, executable sections and resources match', flush=True)
        return

    def prepare(seed, kind):
        generator = torch.Generator().manual_seed(seed)
        def make(shape):
            if kind == 'ones':
                return torch.ones(shape, dtype=torch.bfloat16)
            if kind == 'uniform_random':
                return (torch.rand(shape, generator=generator) - .5).bfloat16()
            if mls:
                return torch.randn(shape, generator=generator).bfloat16()
            return (torch.randn(shape, generator=generator) * .5).bfloat16()
        ah, bh = make((m, k)), make(bshape)
        ai[guard:-guard].copy_(ah.reshape(-1))
        bi[guard:-guard].copy_(bh.reshape(-1))
        ar.copy_(ai)
        br.copy_(bi)
        return ah, bh

    def launch(index, phase, ah, bh):
        cr.copy_(ci)
        torch.cuda.synchronize()
        before = set((output / 'g').glob('*/*/stats.txt'))
        before_calls = len(public_calls)
        public_launch()
        assert len(public_calls) == before_calls + 1
        torch.cuda.synchronize()
        paths = set((output / 'g').glob('*/*/stats.txt')) - before
        assert len(paths) == 1, paths
        path = paths.pop()
        text = path.read_text()
        assert 'End Simulation Statistics' in text
        def stat(key):
            values = re.findall(r'^' + re.escape(key) + r'\s+([\deE.+-]+)', text, re.M)
            assert len(values) == 1, key
            return float(values[0])
        groups = dict(re.findall(r'^(system\d*\.shaders\d*\.CUs\d+)\.completedWGs\s+(\d+)', text, re.M))
        per_die = {}
        active_per_die = {}
        for cu, value in groups.items():
            die = cu.rsplit('.CUs', 1)[0]
            per_die[die] = per_die.get(die, 0) + int(value)
            active_per_die[die] = active_per_die.get(die, 0) + (int(value) > 0)
        assert sum(per_die.values()) == grid, per_die
        mmops = sum(map(int, re.findall(r'^system\d*\.shaders\d*\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)', text, re.M)))
        expected_mmops = (len(tile_ids) if args.single_tg_perf else tiles) * 2 * bm * bn * k // 8192
        assert mmops == expected_mmops, (mmops, expected_mmops)
        host = cr.cpu()
        actual = host[guard:-guard].view(m, n)
        bf = bh.float().T if small else bh.float()
        outside = nonfinite = 0
        maximum = error_sum = rounding_sum = 0.
        atol = .02 if mls else .001
        regions = [(row, 0, min(128, m-row), n) for row in range(0, m, 128)]
        if args.single_tg_perf:
            regions = [(row, col, min(bm, m-row), min(bn, n-col)) for row, col in selected_tiles]
        elements = sum(rows*cols for row, col, rows, cols in regions)
        for row, col, rows, cols in regions:
            fp32 = ah[row:row+rows].float() @ bf[:, col:col+cols]
            golden = fp32.bfloat16().float()
            observed = actual[row:row+rows, col:col+cols].float()
            outside += int((~torch.isclose(observed, golden, atol=atol, rtol=.01)).sum())
            nonfinite += int((~torch.isfinite(observed)).sum())
            maximum = max(maximum, float((observed - golden).abs().max()))
            error_sum += float((observed - fp32).abs().sum(dtype=torch.float64))
            rounding_sum += float((golden - fp32).abs().sum(dtype=torch.float64))
            if args.single_tg_perf:
                actual[row:row+rows, col:col+cols] = float('nan')
        unselected = bool(torch.isnan(actual).all()) if args.single_tg_perf else True
        canary = bool((host[:guard] == 123).all() and (host[-guard:] == 123).all())
        inputs = bool(torch.equal(ar.cpu().view(torch.uint8), ai.view(torch.uint8)) and
                      torch.equal(br.cpu().view(torch.uint8), bi.view(torch.uint8)))
        accuracy = dict(elements=elements, unselected_output_nan=unselected, outside_tolerance=outside, nonfinite=nonfinite,
                        max_abs=maximum, canary_ok=canary, inputs_byte_identical=inputs,
                        atol=atol, rtol=.01, mean_abs_vs_fp32=error_sum/elements, rounding_floor=rounding_sum/elements)
        assert outside == nonfinite == 0 and canary and inputs and unselected, accuracy
        if mls:
            assert accuracy['mean_abs_vs_fp32'] <= 1.05 * accuracy['rounding_floor'] + 1e-6, accuracy
        cycles, frequency = stat('system.gcSimCycles'), stat('gcFrequency')
        model = configparser.RawConfigParser()
        model.read(output / 'g/config.ini')
        shaders = [s for s in model.sections() if re.fullmatch(r'system\d*\.shaders\d*', s)]
        cus = [s for s in model.sections() if re.fullmatch(r'system\d*\.shaders\d*\.CUs\d+', s)]
        dies = 2 if args.chip == 'sbx2' else 1
        assert len(shaders) == dies and len(cus) == dies * 48
        assert len(groups) == len(cus) and len(per_die) == dies
        clocks = {int(model[s + '.clk_domain']['clock']) for s in shaders}
        assert clocks == {455}, clocks
        assert math.isclose(frequency, 1000 / 455, rel_tol=1e-6)
        gpu = model['gpu_configuration']
        assert all(gpu[key].lower() in ('false', '0') for key in ('ipc_conn', 'skip_cal', 'skip_kernel_exec'))
        us = cycles * 455 / 1e6 if small else cycles / (frequency * 1000)
        sample = dict(index=index, phase=phase, status='PASS', kernel_cycles=cycles, kernel_us=us,
                      completed_TGs=grid, completed_TGs_per_die=per_die, active_CUs_per_die=active_per_die,
                      mmop_instructions=mmops, gc_frequency_ghz=frequency,
                      stats_path=str(path.relative_to(output)), stats_sha256=digest(path), accuracy=accuracy)
        if args.single_tg_perf:
            assert sorted(per_die.values()) == [0, 1], per_die
            # This PMD rotates the starting die across dispatches.
            assert sum(int(value) for key, value in groups.items() if int(key.rsplit('CUs', 1)[1]) == 0) == 1
            perfs = [p for p in path.parent.glob('*.perf') if not p.name.startswith('stats_')]
            assert len(perfs) == 1 and perfs[0].read_bytes()[1:8] == b'HYGPERF', perfs
            sample['perf_path'] = str(perfs[0].relative_to(output))
        print(json.dumps(dict(requirement=record['requirement'], phase=phase, index=index,
                             kernel_us=us, completed_TGs_per_die=per_die, status='PASS')), flush=True)
        return sample

    seed = record['benchmark']['seed']
    validation = []
    if protocol != 'cold' and not args.single_tg_perf:
        for index in range(3):
            kind = 'ones' if index == 1 else 'uniform_random'
            ah, bh = prepare(seed + 1000*index, kind)
            sample = launch(index, 'correctness', ah, bh)
            sample.update(seed=seed + 1000*index, input_kind=kind)
            validation.append(sample)
    # R001's archived warm samples retain the last uniform-random correctness
    # input. The TT records restore normal-random inputs for timing.
    if args.single_tg_perf:
        ah, bh = prepare(seed, 'uniform_random' if small else 'normal_random')
    elif not small:
        ah, bh = prepare(seed, 'normal_random')
    samples = [launch(index, 'warmup' if args.single_tg_perf and index == 0 else 'timed' if args.single_tg_perf else 'cold' if protocol == 'cold' else 'warmup' if index == 0 else 'timed', ah, bh)
               for index in range(repeats)]
    if args.single_tg_perf:
        candidates = [s for s in samples[1:] if s['completed_TGs_per_die'].get('system.shaders0') == 1]
        assert candidates, 'a timed XCD0 sample is required for the default PERF view'
        selected = min(candidates, key=lambda sample: sample['kernel_cycles'])
        shutil.copyfile(output / selected['perf_path'], output / 'single_tg.perf')
        result = dict(requirement=record['requirement'], status='PASS', single_tg=True,
                      grid=[1], full_grid=[full_grid], tile_ids=tile_ids, options=params,
                      samples=samples, selected=dict(selected, sha256=digest(output / 'single_tg.perf')),
                      resources=record['resources'], full_and_single_text_rodata_equal=True,
                      executable_sections=executable_sections(handle.asm['hsaco']),
                      hsaco_sha256=digest(output / 'kernel.hsaco'), benchmark_sha256=digest(__file__),
                      public_entry=public_entry)
        save(output / 'result.json', result)
        print(json.dumps(dict(requirement=record['requirement'], status='PASS', single_tg=True)), flush=True)
        return
    selected = samples if protocol == 'cold' else samples[1:2] if protocol == 'first_timed' else samples[1:]
    performance = {key: statistics.median(s[key] for s in selected) for key in ('kernel_cycles', 'kernel_us')}
    performance.update(kernel_tops=2*m*n*k/(performance['kernel_us']*1e6), launched_TGs=grid,
                       samples=len(selected), aggregation=aggregation, time_domain=record['performance']['time_domain'])
    result = dict(requirement=record['requirement'], status='PASS', shape=record['shape'], dtype=record['dtype'],
                  layout=record['layout'], chip=args.chip, dies=2 if args.chip == 'sbx2' else 1,
                  cu_count=96 if args.chip == 'sbx2' else 48,
                  reported_device_cu_count=reported_cus, grid=[grid], launch_options=params,
                  performance=performance, validation=validation, samples=samples,
                  timing_input=dict(seed=seed, input_kind="normal_random"),
                  baseline_record=str(record_path), baseline_record_sha256=digest(record_path),
                  source_sha256=digest(source_root / record['source']['path']), config_sha256=record['config']['sha256'],
                  assembly_sha256=digest(output / 'kernel.amdgcn'), hsaco_sha256=digest(output / 'kernel.hsaco'),
                  assembly_matches_archived=digest(output / 'kernel.amdgcn') == record['assembly']['sha256'],
                  benchmark_sha256=digest(__file__), resources=dict(VGPR=handle.n_regs, spills=handle.n_spills,
                      LDS_bytes=handle.metadata.shared),
                  runtime=dict(torch=torch.__version__, triton=triton.__version__, triton_path=triton.__file__,
                      libtriton_sha256=digest(lib.__file__)), public_entry=public_entry)
    save(output / 'result.json', result)
    print(json.dumps(dict(requirement=record['requirement'], status='PASS', performance=performance)), flush=True)


if __name__ == '__main__':
    main()
