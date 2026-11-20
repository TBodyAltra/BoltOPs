"""Replay the integrated gfx946 96-CU BF16 TT GEMMs on PMD sbx2."""

import argparse
import builtins
import configparser
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import zlib


REPO = next(p for p in Path(__file__).resolve().parents if (p / 'boltops/gemm/config.py').is_file())
CASES = {
    'R024': ((6656, 2048, 4096), '_gemm_a16w16_balanced_tail_kernel', 'gemm_a16w16_persistent'),
    'R028': ((512, 4096, 12288), '_gemm_a16w16_bf16_tt_bm256_bn128_bk64_persistent_kernel',
             'gemm_a16w16_streamk'),
}
SEED = 29
CUS = 96
PROFILE = 'DTYPE=bfloat16-OUT=bfloat16'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def paths(output):
    return set((output / 'g').glob('*/*/stats.txt'))


def stats_order(path):
    return tuple(map(int, path.parts[-3:-1]))


def read_dispatch(output, path, label, expected_tgs, expected_mmop=None):
    text = path.read_text()
    assert 'End Simulation Statistics' in text, path

    def stat(name):
        values = re.findall(r'^' + re.escape(name) + r'\s+([\deE.+-]+)', text, re.M)
        assert len(values) == 1, (name, path)
        return float(values[0])

    groups = {(int(die), int(cu)): int(count) for die, cu, count in re.findall(
        r'^system\.shaders(\d+)\.CUs(\d+)\.completedWGs\s+(\d+)', text, re.M)}
    by_die = [sum(count for (die, _), count in groups.items() if die == d) for d in (0, 1)]
    assert len(groups) == CUS and sum(by_die) == expected_tgs, (label, by_die)
    if expected_tgs > 1:
        assert all(by_die), (label, by_die)
    mmop = sum(map(int, re.findall(r'^system\.shaders\d+\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)',
                                    text, re.M)))
    if expected_mmop is not None:
        assert mmop == expected_mmop, (label, mmop, expected_mmop)
    cycles = stat('system.gcSimCycles')
    us = cycles * 455 / 1e6
    result = dict(label=label, role=label, cycles=cycles, us=us, kernel_cycles=cycles, kernel_us=us,
                  completed_TGs=expected_tgs,
                  completed_TGs_per_die=by_die, MMOP=mmop, stats_path=str(path.relative_to(output)),
                  stats_sha256=digest(path), frequency_ghz=stat('gcFrequency'))
    perfs = [p for p in path.parent.glob('*.perf') if not p.name.startswith('stats_')]
    assert len(perfs) <= 1, perfs
    if perfs:
        result.update(perf_path=str(perfs[0].relative_to(output)), perf_sha256=digest(perfs[0]))
    if expected_tgs == 1:
        die = by_die.index(1)
        result.update(die=die, cu_cycles=stat(f'system.shaders{die}.CUs00.totalCycles'))
    return result


def topology(output):
    path = output / 'g/config.ini'
    cfg = configparser.RawConfigParser()
    cfg.read(path)
    gpu = cfg['gpu_configuration']
    clocks = [int(cfg[f'system.shaders{d}.clk_domain']['clock']) for d in (0, 1)]
    assert gpu['num_chip'] == '2' and gpu['num_cu'] == '48' and clocks == [455, 455]
    assert all(gpu[key] == 'false' for key in ('ipc_conn', 'skip_cal', 'skip_kernel_exec'))
    return dict(type='PMD simulator', chip='sbx2', arch='gfx946', dies=2, cu_count=CUS,
                cu_count_per_die=[48, 48], clock_period_ps=455, config_sha256=digest(path))


def executable_sections(code):
    """Fingerprint runtime code/data independently of linker padding/debug bytes."""
    assert code[:6] == b'\x7fELF\x02\x01'
    offset, = struct.unpack_from('<Q', code, 40)
    size, count, names = struct.unpack_from('<HHH', code, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', code, offset + i * size) for i in range(count)]
    table = sections[names]
    strings = code[table[4]:table[4] + table[5]]
    result = {}
    for entry in sections:
        name = strings[entry[0]:].split(b'\0', 1)[0]
        if name in (b'.text', b'.rodata'):
            data = code[entry[4]:entry[4] + entry[5]]
            result[name.decode()] = hashlib.sha256(data).hexdigest()
    assert '.text' in result
    return result


def archive(output, label, handle):
    prefix = 'kernel' if label == 'main' else label
    assembly = output / f'{prefix}.amdgcn'
    hsaco = output / f'{prefix}.hsaco'
    assembly.write_text(handle.asm['amdgcn'])
    hsaco.write_bytes(handle.asm['hsaco'])
    return dict(path=assembly.name, sha256=digest(assembly), hsaco=hsaco.name, hsaco_sha256=digest(hsaco),
                executable_sections=executable_sections(handle.asm['hsaco']),
                symbol=handle.metadata.name, n_regs=handle.n_regs, n_spills=handle.n_spills,
                compiler_shared_bytes=handle.metadata.shared, num_warps=handle.metadata.num_warps)


class ObserveJIT:
    def __init__(self, kernel, observe):
        self.kernel = kernel
        self.observe = observe

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.observe(launch(*args, **kwargs), grid)

        return run


def prepare(torch, shape):
    m, n, k = shape
    gen = torch.Generator().manual_seed(SEED)
    a = torch.randn((m, k), generator=gen).bfloat16()
    b = torch.randn((k, n), generator=gen).bfloat16()
    guard = 128
    initial = torch.full((2 * guard + m * n,), 123., dtype=torch.bfloat16)
    initial[guard:-guard] = float('nan')
    ar, br, cr = a.cuda(), b.cuda(), initial.cuda()
    return a, b, ar, br, cr, cr[guard:-guard].view(m, n)


def validate_full(torch, a, b, ar, br, cr, c):
    actual, raw = c.cpu(), cr.cpu()
    guard = bool((raw[:128] == 123).all() and (raw[-128:] == 123).all())
    finite = bool(torch.isfinite(actual).all())
    inputs = torch.equal(ar.cpu().view(torch.uint8), a.view(torch.uint8)) and torch.equal(
        br.cpu().view(torch.uint8), b.view(torch.uint8))
    mismatches, error, floor = 0, 0., 0.
    rhs = b.float()
    for row in range(0, a.shape[0], 64):
        ref = a[row:row + 64].float() @ rhs
        rounded, got = ref.bfloat16(), actual[row:row + 64]
        mismatches += int((~torch.isclose(got, rounded, atol=.02, rtol=.01)).sum())
        error += float((got.float() - ref).abs().sum())
        floor += float((rounded.float() - ref).abs().sum())
    assert guard and finite and inputs and mismatches == 0 and error <= 1.05 * floor + 1e-6 * actual.numel(), (
        guard, finite, inputs, mismatches, error, floor)
    return dict(status='PASS', elements=actual.numel(), checked_points=actual.numel(),
                mismatches=mismatches, finite=finite, outside_tolerance=mismatches, nonfinite=0,
                canary_ok=guard, guard=guard, inputs_byte_identical=inputs, inputs_unchanged=inputs,
                mean_abs=error / actual.numel(), rounding_floor=floor / actual.numel(), atol=.02, rtol=.01)


def verify_assembly(role, actual, recorded):
    if recorded is None:
        return
    expected = recorded['assembly' if role == 'main' else 'reduce_assembly']
    for key in ('executable_sections', 'symbol', 'n_regs', 'n_spills',
                'compiler_shared_bytes', 'num_warps'):
        if actual[key] != expected[key]:
            raise ValueError(f'{role} replay {key} differs from the archived kernel: '
                             f'{actual[key]!r} != {expected[key]!r}')


def full_run(requirement, symbol, module, output, torch, shape, recorded=None):
    public = importlib.import_module('boltops.gemm.gemm_a16w16')
    dispatch = importlib.import_module('boltops.gemm.gluon.gemm_a16w16')
    captured = {}

    def observe(label, handle, grid):
        assert label not in captured, label
        captured[label] = dict(archive(output, label, handle), grid=list(grid))
        verify_assembly(label, captured[label], recorded)
        print(f'{requirement} {label} grid={tuple(grid)}', flush=True)

    a, b, ar, br, cr, c = prepare(torch, shape)
    torch.cuda.synchronize()
    before = paths(output)
    if requirement == 'R024':
        reduction = importlib.import_module('boltops.gemm.triton.splitk_reduce')
        original_main = getattr(dispatch, symbol)
        original_reduce = reduction._gemm_splitk_reduce_kernel
        setattr(dispatch, symbol, ObserveJIT(original_main, lambda h, g: observe('main', h, g)))
        reduction._gemm_splitk_reduce_kernel = ObserveJIT(original_reduce, lambda h, g: observe('reduce', h, g))
        try:
            assert public.gemm_a16w16(ar, br.T, out=c, backend='auto') is c
            torch.cuda.synchronize()
        finally:
            setattr(dispatch, symbol, original_main)
            reduction._gemm_splitk_reduce_kernel = original_reduce
        assert captured['main']['grid'] == [CUS] and captured['reduce']['grid'] == [8, 16]
    else:
        original_compile = module._compile_kernel

        class ObservedCompiled:
            def __init__(self, handle):
                self.handle = handle

            def __getattr__(self, name):
                return getattr(self.handle, name)

            @property
            def run(self):
                launch = self.handle.run

                def run(*args):
                    assert args[:3] == (CUS, 1, 1) and args[5][2] == 98304, args[:6]
                    observe('main', self.handle, args[:3])
                    return launch(*args)

                return run

        def observe_compile(*args, **kwargs):
            assert args[0] is getattr(module, symbol) and args[4] == 3, args[:5]
            return ObservedCompiled(original_compile(*args, **kwargs))

        module._compile_kernel = observe_compile
        try:
            assert public.gemm_a16w16(ar, br.T, out=c, backend='auto') is c
            torch.cuda.synchronize()
        finally:
            module._compile_kernel = original_compile
    stats = sorted(paths(output) - before, key=stats_order)
    assert len(stats) == 2, stats
    m, n, k = shape
    main = ('main', CUS, 2 * m * n * k // 8192)
    expected = [main, ('reduce', 128, 0)] if requirement == 'R024' else [('flag_reset', 1, 0), main]
    per = [read_dispatch(output, p, *entry) for p, entry in zip(stats, expected)]
    save(output / 'dispatches.json', per)
    cycles = sum(d['kernel_cycles'] for d in per)
    us = sum(d['kernel_us'] for d in per)
    performance = dict(kernel_cycles=cycles, kernel_us=us,
                       kernel_tops=2 * m * n * k / (us * 1e6),
                       mmac_util_percent=100 * 2 * m * n * k / (cycles * CUS * 4096),
                       aggregation='one cold public auto call; includes all observed GPU dispatches')
    return dict(dispatches=per, performance=performance, assemblies=captured, public_api_verified=True,
                validation=validate_full(torch, a, b, ar, br, cr, c))


def validate_trace(path, die, mmops, cu_cycles, wave_count, hsaco_sha256=None, exclude_hsaco=None):
    """Check native SQTT and the embedded object (format as in R020's runner)."""
    data = path.read_bytes()
    assert data[:8] == b'\xe4HYGPERF' and struct.unpack_from('<2Q', data, 8) == (0, 24)
    pos, traces, objects = 24, {}, []
    compressed = {6: (24, 32, 40), 8: (32, 40, 48), 11: (48, 56, 64)}
    while pos < len(data):
        kind, offset, length = struct.unpack_from('<3Q', data, pos)
        assert offset == pos - 24 and length >= 24
        end = pos + length
        if kind in compressed:
            payload, raw_bytes, packed_bytes = (struct.unpack_from('<Q', data, pos + index)[0]
                                                for index in compressed[kind])
            assert payload == offset + length
            raw = zlib.decompress(data[end:end + packed_bytes])
            assert len(raw) == raw_bytes
            if kind == 8:
                objects.append(raw)
            elif kind == 11:
                xcd = struct.unpack_from('<Q', data, pos + 72)[0] >> 32
                assert xcd not in traces
                traces[xcd] = raw
            end += packed_bytes
        assert pos < end <= len(data)
        pos = end
    # PMD includes all code objects loaded so far, not only the object
    # belonging to the current dispatch (e.g. main plus reducer).
    assert objects and all(obj.startswith(b'\x7fELF') for obj in objects)
    object_hashes = [hashlib.sha256(obj).hexdigest() for obj in objects]
    candidates = [key for key in object_hashes if key == hsaco_sha256] if hsaco_sha256 else [
        key for key in object_hashes if key != exclude_hsaco]
    assert len(candidates) == 1, (object_hashes, hsaco_sha256, exclude_hsaco)
    assert set(traces) == {0, 1}
    raw = traces[die]
    # TT0 may leave the active XCD1 empty; selection requires nonempty XCD0.
    if die != 0:
        return None
    assert raw and not traces[1 - die] and len(raw) % 2 == 0
    native = path.parent / f'ttrace_model__xcd{die}_cu0_se0.out'
    words = struct.unpack('<' + 'H' * (len(raw) // 2), raw)
    assert words == tuple(int(word, 16) for word in native.read_text().split())
    cursor = time = instructions = matrix_ops = 0
    starts, ends, waves = [], [], set()
    while cursor < len(words):
        value = words[cursor]
        cursor += 1
        kind = value & 15
        if kind == 0:
            misc = value >> 13
            assert misc in (0, 1)
            time = time + ((value >> 4) & 255) if misc == 0 else 0
            continue
        if kind == 1:
            time = sum(words[cursor + i] << (16 * i) for i in range(3))
            cursor += 3
            continue
        time += (value >> 4) & 1
        if kind in (3, 6):
            (starts if kind == 3 else ends).append(((value >> 14) & 3, (value >> 10) & 15))
            cursor += kind == 3
        elif kind == 13:
            cursor += 1
        else:
            assert kind in (9, 10), ('unsupported TT token', cursor - 1, hex(value))
            op = value >> 11
            if kind == 9:
                op += 32
                if op in (36, 37):
                    assert words[cursor] == 0
                    cursor += 1
            elif op == 29:
                op = 48 + (words[cursor] & 31)
                cursor += 1
            instructions += 1
            matrix_ops += op == 49
            waves.add(((value >> 9) & 3, (value >> 5) & 15))
        assert cursor <= len(words), 'truncated TT token'
    assert len(starts) == len(ends) == len(waves) == wave_count and set(starts) == set(ends) == waves
    assert matrix_ops == mmops and abs(time * 4 - cu_cycles) <= 4, (matrix_ops, mmops, time, cu_cycles)
    return dict(status='PASS', die=die, CU=0, SE=0, wavefronts=wave_count, instructions=instructions,
                MMOP=matrix_ops, tt_end_cycles=time * 4, cu_active_cycles=cu_cycles,
                trace_sha256=hashlib.sha256(raw).hexdigest(), native_trace_sha256=digest(native),
                embedded_hsaco_sha256=candidates[0], embedded_objects_sha256=object_hashes,
                executable_sections=executable_sections(objects[object_hashes.index(candidates[0])]))


def capture_one(output, torch, role, action, waves, mmop, exclude_hsaco=None):
    """Run a validated one-TG operation until PMD dispatches it to XCD0."""
    samples = []
    for _ in range(4):
        launch, check = action()
        torch.cuda.synchronize()
        before = paths(output)
        handle = launch()
        torch.cuda.synchronize()
        new = paths(output) - before
        assert len(new) == 1, (role, new)
        sample = read_dispatch(output, new.pop(), role, 1, mmop)
        sample['validation'] = check()
        samples.append(sample)
        save(output / f'{role}_samples.json', samples)
        if sample['die'] != 0:
            continue
        perf = output / sample['perf_path']
        assembly = archive(output, role, handle) if handle is not None else None
        sample['wavefront_trace'] = validate_trace(
            perf, 0, mmop, sample['cu_cycles'], waves,
            hsaco_sha256=assembly['hsaco_sha256'] if assembly else None, exclude_hsaco=exclude_hsaco,
        )
        name = 'single_tg.perf' if role == 'main' else f'{role}_single_tg.perf'
        shutil.copyfile(perf, output / name)
        sample.update(archive_path=name, assembly=assembly)
        save(output / f'{role}_samples.json', samples)
        return sample, samples
    raise RuntimeError(f'{role}: no single-TG launch on XCD0')


def single_run(requirement, symbol, module, output, torch, triton, target, shape, config, recorded=None):
    m, n, k = shape
    opts = {key: tuple(value) if isinstance(value, list) else value for key, value in config.items()
            if key not in ('kernel', 'NUM_SMS', 'requires_16_byte_alignment', 'graph_fallback')}
    kernel = getattr(module, symbol)
    a, b, ar, br, cr, c = prepare(torch, shape)
    initial = cr.cpu()
    assemblies, samples, selected = {}, {}, []

    def clear_output():
        # Host copy avoids an extra device launch advancing PMD's die rotation.
        cr.copy_(initial)

    def check_io():
        raw = cr.cpu()
        assert bool((raw[:128] == 123).all() and (raw[-128:] == 123).all())
        assert torch.equal(ar.cpu(), a) and torch.equal(br.cpu(), b)

    def record(role, action, waves, mmop):
        exclude = assemblies['main']['hsaco_sha256'] if role == 'flag_reset' else None
        sample, samples[role] = capture_one(output, torch, role, action, waves, mmop, exclude)
        selected.append(sample)
        if sample['assembly'] is not None:
            verify_assembly(role, sample['assembly'], recorded)
            assemblies[role] = sample['assembly']
        elif recorded is not None:
            expected = recorded['traces'][role]['wavefront_trace']['executable_sections']
            if sample['wavefront_trace']['executable_sections'] != expected:
                raise ValueError(f'{role} replay executable sections differ from the archived kernel')

    if requirement == 'R024':
        groups, splits, first = module._balanced_tail_plan(m, n, k, CUS)
        assert (groups, splits, first) == (96, 6, 6144)
        k_end = (k // 64 // splits) * 64
        positions = ((0, 0), (3072, 0))
        scope = dict(selected_original_TG=0, full_grid=[groups], complete_tile_ids=[0, 96],
                     partial_tile_id=192, partial_split=0, partial_k_range=[0, k_end])

        def main_action():
            clear_output()
            partial = torch.full((splits, m - first, n), float('nan'), dtype=torch.float32).cuda()

            def launch():
                return kernel[(1,)](ar, br.T, c, partial, M=m, N=n, K=k, NUM_SMS=CUS,
                                    TAIL_SPLITS=splits, B_K_CONTIGUOUS=False, **opts)

            def check():
                check_io()
                actual, fragments = c.cpu(), partial.cpu()
                for row, col in positions:
                    out = actual[row:row + 256, col:col + 256]
                    ref = a[row:row + 256].float() @ b[:, col:col + 256].float()
                    torch.testing.assert_close(out, ref.bfloat16(), atol=.02, rtol=.01)
                    out.fill_(float('nan'))
                got = fragments[0, :256, :256]
                ref = a[first:first + 256, :k_end].float() @ b[:k_end, :256].float()
                torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
                got.fill_(float('nan'))
                assert bool(torch.isnan(actual).all() and torch.isnan(fragments).all())
                return dict(status='PASS', checked_points=2 * 256 * 256, partial_points=256 * 256,
                            untouched_output=True, untouched_partial=True)

            return launch, check

        record('main', main_action, 12, 2 * 256 * 256 * (2 * k + k_end) // 8192)
        reducer = importlib.import_module('boltops.gemm.triton.splitk_reduce')._gemm_splitk_reduce_kernel
        gen = torch.Generator().manual_seed(SEED)
        # Exact full-shape reducer ABI; only the launch grid is narrowed.
        ref = torch.randn((splits, m - first, n), generator=gen)

        def reduce_action():
            clear_output()
            partial = ref.cuda()

            def launch():
                return reducer[(1, 1)](partial, c[first:], None, m - first, n, *partial.stride(), *c.stride(),
                                       64, 128, splits, triton.next_power_of_2(splits), ADD_BIAS=False,
                                       activation='', use_activation=False, KERNEL_NAME='_gemm_a16w16_reduce_kernel',
                                       num_warps=4)

            def check():
                check_io()
                actual = c.cpu()
                got = actual[first:first + 64, :128]
                torch.testing.assert_close(got, ref[:, :64, :128].sum(0).bfloat16(), atol=.001, rtol=.01)
                got.fill_(float('nan'))
                assert bool(torch.isnan(actual).all()) and torch.equal(partial.cpu(), ref)
                return dict(status='PASS', checked_points=64 * 128, untouched_output=True)

            return launch, check

        record('reduce', reduce_action, 4, 0)
    else:
        from triton import knobs

        group = opts.pop('schedule_group_m')
        compact = opts.pop('compact_schedule')
        for key in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K'):
            opts.pop(key)
        table, records, slots = module._make_schedule(m, n, k, CUS, schedule_group_m=group,
                                                      compact_schedule=compact)
        assert (records, slots, len(table)) == (3, 64, CUS * 3)
        rows = [table[i:i + records] for i in range(0, len(table), records)]
        pid, tasks = next((i, row) for i, row in enumerate(rows)
                          if any(x[2] and x[3] > 0 for x in row) and all(x[3] >= 0 for x in row))
        [(tile, offset, length, tag)] = [task for task in tasks if task[2]]
        assert (pid, tile, offset, length, tag) == (2, 1, 4096, 8192, 2)
        scope = dict(selected_original_TG=pid, tasks=tasks, full_grid=[CUS], records=records, slots=slots,
                     interpretation='publisher-only fragment; no final BF16 tile or peer wait')
        save(output / 'schedule.json', dict(full_table=table, isolated_TG0=tasks))
        handle = module._compile_kernel(kernel, m, n, k, records, tuple(sorted(opts.items())), target)
        run = handle.run
        schedule = torch.tensor(tasks + table[records:], dtype=torch.int32).cuda()
        edges = [torch.empty(1, dtype=torch.bfloat16, device=c.device) for _ in range(3)]

        def main_action():
            workspace = torch.full((slots, 256, 128), float('nan'), dtype=torch.float32).cuda()
            flags = torch.zeros(slots * 8, dtype=torch.int32).cuda()
            values = (ar, br.T, c, schedule, workspace, flags, *edges, 1)
            grid = (1, 1, 1)
            stream = torch.cuda.current_stream(c.device)

            def launch():
                packed = (handle.metadata.num_warps, handle.metadata.num_ctas, 98304)
                run(*grid, stream.cuda_stream, handle.function, packed,
                    handle.launch_metadata(grid, stream.cuda_stream, *values),
                    knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *values)
                return handle

            def check():
                check_io()
                assert bool(torch.isnan(c.cpu()).all())
                fragments, got_flags = workspace.cpu(), flags.cpu()
                row, col = tile // (n // 128) * 256, tile % (n // 128) * 128
                expected = a[row:row + 256, offset:offset + length].float() @ b[offset:offset + length, col:col + 128].float()
                # Different FP32 accumulation orders over 8192 K terms can
                # differ by a few 1e-4 around cancellation points.
                got = fragments[tag - 1]
                torch.testing.assert_close(got, expected, atol=5e-4, rtol=1e-4)
                max_abs = float((got - expected).abs().max())
                got.fill_(float('nan'))
                wanted_flags = torch.zeros_like(got_flags)
                wanted_flags[(tag - 1) * 8:tag * 8] = 1
                assert bool(torch.isnan(fragments).all()) and torch.equal(got_flags, wanted_flags)
                return dict(status='PASS', checked_points=0, partial_points=256 * 128,
                            partial_atol=5e-4, partial_rtol=1e-4, partial_max_abs=max_abs,
                            validated_flags=slots * 8, untouched_output=True, untouched_partial=True)

            return launch, check

        record('main', main_action, 12, 2 * 256 * 128 * length // 8192)

        def reset_action():
            state = {}

            def launch():
                state['flags'] = torch.zeros(slots * 8, dtype=torch.int32, device=c.device)

            def check():
                assert not bool(state['flags'].cpu().any())
                return dict(status='PASS', checked_points=slots * 8, reset_value=0)

            return launch, check

        record('flag_reset', reset_action, 4, 0)
    save(output / 'dispatches.json', selected)
    return dict(scope=scope, dispatches=selected, samples=samples, assemblies=assemblies,
                validation=dict(status='PASS', by_dispatch={x['role']: x['validation'] for x in selected}))


def main(requirement):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--record', type=Path, help='optional archived record to verify against the checkout')
    parser.add_argument('--single-tg-perf', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--triton-build', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('output directory must not exist')
    shape, symbol, module_name = CASES[requirement]
    m, n, k = shape
    config_path = REPO / f'boltops/gemm/configs/gfx946/96cu/gluon/gemm_a16w16/GEMM-A16W16-N={n}-K={k}-TT.json'
    bucket = f'M_EQ_{m}'
    config = json.loads(config_path.read_text())['profiles'][PROFILE][bucket]
    assert config['kernel'] == symbol and config['NUM_SMS'] == CUS, config
    deps = [config_path, REPO / f'boltops/gemm/gluon/{module_name}.py',
            REPO / 'boltops/gemm/gluon/gemm_a16w16.py', REPO / 'boltops/gemm/gemm_a16w16.py']
    if requirement == 'R024':
        deps.append(REPO / 'boltops/gemm/triton/splitk_reduce.py')
    sources = {str(p.relative_to(REPO)): digest(p) for p in deps}
    plan = dict(requirement=requirement, shape=shape, dtype='bf16', layout='TT', seed=SEED, backend='auto',
                chip='sbx2', cu_count=CUS, config_path=str(config_path.relative_to(REPO)), config_bucket=bucket,
                config=config, source_sha256=sources, benchmark_sha256=digest(__file__), single_tg=args.single_tg_perf)
    if requirement == 'R024':
        plan['dispatch_grids'] = {'main': [1], 'reduce': [1, 1]} if args.single_tg_perf else {
            'main': [96], 'reduce': [8, 16]}
    else:
        plan['dispatch_grids'] = {'main': [1], 'flag_reset': [1]} if args.single_tg_perf else {
            'flag_reset': [1], 'main': [96]}
    plan['seeded_operands'] = {'A_strides': [k, 1], 'W_strides': [1, n], 'C_strides': [n, 1]}
    recorded = json.loads(args.record.read_text()) if args.record else None
    if recorded is not None:
        archived_plan = recorded['plan']
        for key in ('requirement', 'shape', 'config', 'source_sha256'):
            if archived_plan[key] != json.loads(json.dumps(plan[key])):
                raise ValueError(f'replay {key} differs from the archived plan')
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    marker = 'BOLTOPS_SELECTED_TT_OUTPUT'
    if os.environ.get(marker) != str(output):
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(('GPU_', 'HIP_', 'HSA_', 'ROCR_', 'SOC_', 'DCU_', 'RPY_', 'RBPY_')) or key in (
                'BOLTOPS_GEMM_CONFIG_DIR', 'TRITON_OFFLINE_BUILD', 'TRITON_BUFFER_CACHE_SWIZZLE',
                'TRITON_MLS_B_INTERLEAVE', 'TRITON_HCU_MATRIX_STORE_INSN', 'LD_PRELOAD',
            ):
                env.pop(key, None)
        assert env.get('PMD_PATH') == '/opt/rocm/pmd', 'Source the recorded Triton 3.6 path.sh first.'
        if args.triton_build:
            env['PYTHONPATH'] = str(args.triton_build.resolve())
        env.update(GPU_CHIP='sbx2', GPU_OUT='g', GPU_ARGS="['--abs-max-tick=10000000000']",
                   GPU_DFLAGS="['StatLog','TT0','Perf']" if args.single_tg_perf else "['StatLog','Perf']",
                   HSA_TOOLS_LIB='/opt/rocm/lib/xprofiler/libperf_gen_helper.so', AMDGCN_USE_BUFFER_OPS='1',
                   TRITON_CACHE_DIR=str(output / 'cache'), XDG_CACHE_HOME=str(output / 'xdg-cache'),
                   TMPDIR=str(output / 'temp'), RPY_HISTORY_LOG=str(output / 'history.log'),
                   RPY_HISTORY_SIZE='1024', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        env[marker] = str(output)
        os.execve(sys.executable, [sys.executable, '-u', str(Path(sys.argv[0]).resolve()), *sys.argv[1:]], env)
    output.mkdir(parents=True)
    for name in ('cache', 'xdg-cache', 'temp'):
        (output / name).mkdir()
    os.chdir(output)
    original_open = builtins.open
    shared_lock = '/tmp/' + getpass.getuser() + '.history.log.lock'

    def private_open(file, *positional, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file) == shared_lock:
            file = output / 'history.lock'
        return original_open(file, *positional, **kwargs)

    builtins.open = private_open
    sys.path.insert(0, str(REPO))
    import torch
    import triton

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.get_device_properties(0).multi_processor_count == CUS
    target = triton.runtime.driver.active.get_current_target()
    assert target.arch == 'gfx946', target
    module = importlib.import_module(f'boltops.gemm.gluon.{module_name}')
    plan['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    plan['environment'] = {key: os.environ.get(key) for key in ('GPU_CHIP', 'GPU_ARGS', 'GPU_DFLAGS', 'PMD_PATH',
                           'TRITON_HIP_CLANG_PATH', 'HSA_TOOLS_LIB')}
    plan.update(torch=torch.__version__, triton=triton.__version__, triton_module=triton.__file__)
    save(output / 'plan.json', plan)
    print(f'{requirement} {shape} starting PMD {"single TG" if args.single_tg_perf else "full grid"}', flush=True)
    if args.single_tg_perf:
        data = single_run(requirement, symbol, module, output, torch, triton, target, shape, config, recorded)
    else:
        data = full_run(requirement, symbol, module, output, torch, shape, recorded)
    save(output / 'result.json', dict(status='PASS', requirement=requirement, shape=shape, layout='TT',
                                     single_tg=args.single_tg_perf, plan=plan,
                                     environment=topology(output), **data))
    print(f'{requirement} PASS {data.get("performance", "single TG")}', flush=True)


if __name__ == '__main__':
    main('R024')
