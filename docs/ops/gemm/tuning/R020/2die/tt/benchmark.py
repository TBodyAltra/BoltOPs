# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Measure R020's existing BF16 TT Stream-K kernel on sbx2 (2 x 48 CU).

Source the specified Triton 3.6 path.sh first. Default: one cold public auto
GEMM, full CPU-reference validation; timings include the launcher's native flag
reset dispatch, with individual costs also recorded. --single-tg-perf isolates an
original TG with no peer waits, keeping MNK, strides, compiled instructions and
RECORDS. Its task-table row is relocated
to TG0; both complete outputs and published FP32 partials/flags are validated.
Three correctness cases precede one warmup and two timed repeats. The fastest
timed repeat on die 0 supplies the PERF so its default XCD0 view contains waves.
The embedded TT stream is decoded and checked against the simulator counters.
When the checkout differs, replay uses the frozen BoltOPs source from Git.
--current-toolchain accepts a sourced local PMD environment and reports whether
the resulting executable sections match the archived binary.
--dry-run verifies artifacts without GPU imports.
"""
import argparse
import builtins
import configparser
from dataclasses import asdict
import getpass
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import zlib

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'boltops/gemm/config.py').is_file())
SYMBOL = '_gemm_a16w16_bf16_tt_bm256_bn128_bk64_persistent_kernel'
SOURCE = 'boltops/gemm/gluon/gemm_a16w16.py'
PUBLIC = 'boltops/gemm/gemm_a16w16.py'
DEPENDENCIES = (
    'boltops/gemm/config.py',
    'boltops/utility/arch_info.py',
    'boltops/gemm/backend_interfaces.py',
    'boltops/gemm/gluon/gemm_a16w16_streamk.py',
)
ENVIRONMENT_SCRIPT = '/public/home/duyubo/triton_all_version/triton_3.6/triton/path.sh'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def frozen_checkout(record):
    items = [record[key] for key in ('source', 'config', 'public_api')]
    items.extend(record['source_dependencies'])
    if all((REPO / item['path']).is_file() and digest(REPO / item['path']) == item['sha256']
           for item in items):
        return None
    commit = record['benchmark']['replay_git_commit']
    for item in items:
        content = subprocess.check_output(['git', 'show', f"{commit}:{item['path']}"], cwd=REPO)
        assert hashlib.sha256(content).hexdigest() == item['sha256'], item['path']
    return commit


def restore_boltops(output, commit):
    if commit is None:
        return REPO
    source = output / 'source'
    source.mkdir()
    content = subprocess.check_output(['git', 'archive', commit, 'boltops'], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        for member in archive.getmembers():
            assert (source / member.name).resolve().is_relative_to(source)
            assert member.isfile() or member.isdir(), member.name
        archive.extractall(source)
    return source


def executable_sections(code):
    assert code[:5] == b'\x7fELF\x02'
    offset, = struct.unpack_from('<Q', code, 40)
    size, count, names = struct.unpack_from('<HHH', code, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', code, offset+i*size) for i in range(count)]
    strings = code[headers[names][4]:headers[names][4]+headers[names][5]]
    sections = {}
    for entry in headers:
        name = strings[entry[0]:].split(b'\0', 1)[0].decode()
        if name in ('.text', '.rodata'):
            sections[name] = hashlib.sha256(code[entry[4]:entry[4]+entry[5]]).hexdigest()
    assert set(sections) == {'.text', '.rodata'}
    return sections


def options(m, n, k, commit=None):
    path = f'boltops/gemm/configs/gfx946/96cu/gluon/gemm_a16w16/GEMM-A16W16-N={n}-K={k}-TT.json'
    content = (subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=REPO)
               if commit else (REPO / path).read_bytes())
    data = json.loads(content)
    bucket = f'M_EQ_{m}'
    params = dict(data['profiles']['DTYPE=bfloat16-OUT=bfloat16'][bucket])
    assert params.pop('kernel') == SYMBOL
    return path, bucket, params


def validate_single_tg_perf(path, die, mmops, cu_cycles):
    """Decode native HYGPERF SQTT, not just its embedded source/code object.

    Token fields follow PMD scripts/ttrace/t_token.py; one TT unit is four
    cycles. sbx2 writes one SQTT section per die, including the idle die.
    """
    data = Path(path).read_bytes()
    assert data[:8] == b'\xe4HYGPERF' and struct.unpack_from('<2Q', data, 8) == (0, 24)
    lengths = {6: 48, 7: 72, 8: 56, 9: 160, 10: 64, 11: 80, 13: 56, 15: 72}
    compressed = {6: (24, 32, 40), 8: (32, 40, 48), 11: (48, 56, 64)}
    pos, traces, objects = 24, {}, []
    while pos < len(data):
        kind, offset, length = struct.unpack_from('<3Q', data, pos)
        assert offset == pos - 24 and length == lengths[kind]
        end = pos + length
        if kind in compressed:
            po, ro, co = compressed[kind]
            assert struct.unpack_from('<Q', data, pos + po)[0] == offset + length
            raw_size = struct.unpack_from('<Q', data, pos + ro)[0]
            size = struct.unpack_from('<Q', data, pos + co)[0]
            raw = zlib.decompress(data[end:end + size])
            assert len(raw) == raw_size
            if kind == 8:
                objects.append(raw)
            elif kind == 11:
                xcd = struct.unpack_from('<Q', data, pos + 72)[0] >> 32
                assert xcd not in traces
                traces[xcd] = raw
            end += size
        assert pos < end <= len(data)
        pos = end
    assert pos == len(data) and len(objects) == 1 and objects[0].startswith(b'\x7fELF')
    assert set(traces) == {0, 1} and traces[die] and not traces[1 - die]
    raw = traces[die]
    trace_path = Path(path).parent / f'ttrace_model__xcd{die}_cu0_se0.out'
    # Check the archived binary contains the exact model-emitted trace.
    native_words = [int(word, 16) for word in trace_path.read_text().split()]
    assert raw == struct.pack('<' + 'H' * len(native_words), *native_words)
    assert len(raw) % 2 == 0
    words = struct.unpack('<' + 'H' * (len(raw) // 2), raw)
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
    expected = {(simd, wave) for simd in range(4) for wave in range(3)}
    assert len(starts) == len(ends) == 12 and set(starts) == set(ends) == waves == expected
    assert matrix_ops == mmops and abs(time * 4 - cu_cycles) <= 4
    return dict(status='PASS', die=die, CU=0, SE=0, wavefronts=12, wave_starts=len(starts),
                wave_ends=len(ends), instructions=instructions, MMOP=matrix_ops,
                tt_end_cycles=time * 4, cu_active_cycles=cu_cycles, trace_bytes=len(raw),
                trace_sha256=hashlib.sha256(raw).hexdigest(), native_trace_sha256=digest(trace_path),
                embedded_hsaco_sha256=hashlib.sha256(objects[0]).hexdigest())


def read_sample(output, before, expected_tgs):
    paths = set((output / 'g').glob('*/*/stats.txt')) - before
    assert len(paths) == 1, paths
    path = paths.pop()
    text = path.read_text()
    assert 'End Simulation Statistics' in text
    def stat(name):
        values = re.findall(r'^' + re.escape(name) + r'\s+([\deE.+-]+)', text, re.M)
        assert len(values) == 1, name
        return float(values[0])
    cfg = configparser.RawConfigParser()
    cfg.read(output / 'g/config.ini')
    gpu = cfg['gpu_configuration']
    assert gpu['num_chip'] == '2' and gpu['num_cu'] == '48'
    assert all(gpu[key] == 'false' for key in ('ipc_conn', 'skip_cal', 'skip_kernel_exec'))
    clocks = [int(cfg[f'system.shaders{die}.clk_domain']['clock']) for die in (0, 1)]
    assert clocks == [455, 455], clocks
    groups = {(int(die), int(cu)): int(count) for die, cu, count in re.findall(
        r'^system\.shaders(\d+)\.CUs(\d+)\.completedWGs\s+(\d+)', text, re.M)}
    by_die = [sum(count for (d, _), count in groups.items() if d == die) for die in (0, 1)]
    assert len(groups) == 96 and sum(by_die) == expected_tgs, (len(groups), by_die)
    if expected_tgs > 1:
        assert all(by_die), by_die
    perfs = [p for p in path.parent.glob('*.perf') if not p.name.startswith('stats_')]
    assert len(perfs) == 1 and perfs[0].stat().st_size > 0, perfs
    cycles = stat('system.gcSimCycles')
    mmops = sum(int(value) for value in re.findall(r'^system\.shaders\d+\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)', text, re.M))
    sample = dict(status='PASS', dispatch_cycles=cycles, kernel_us=cycles * clocks[0] / 1e6, MMOP=mmops,
                completed_TGs=expected_tgs, completed_TGs_per_die=by_die,
                stats_path=str(path.relative_to(output)), stats_sha256=digest(path),
                perf_path=str(perfs[0].relative_to(output)), perf_sha256=digest(perfs[0]),
                reported_gc_frequency_ghz=stat('gcFrequency'))
    if expected_tgs == 1:
        die = by_die.index(1)
        sample['wavefront_trace'] = validate_single_tg_perf(
            perfs[0], die, mmops, stat(f'system.shaders{die}.CUs00.totalCycles'))
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=Path(__file__).with_name('record.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--single-tg-perf', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--triton-build', type=Path)
    parser.add_argument('--current-toolchain', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Output directory must be new.')
    requirement = 'R020'
    m, n, k = 2048, 2048, 4096
    seed, guard = int(requirement[1:]), 65536
    grid = 96
    archived = json.loads(args.record.read_text()) if args.record.is_file() else None
    replay_commit = None
    if archived:
        assert archived['requirement'] == requirement and archived['shape'] == [m, n, k]
        assert archived['environment']['dies'] == 2 and archived['environment']['cu_count'] == 96
        assert digest(__file__) == archived['benchmark']['sha256']
        replay_commit = frozen_checkout(archived)
        for key in ('assembly', 'single_tg_perf'):
            item = archived[key]
            assert digest(args.record.parent / item['path']) == item['sha256'], key
    config_path, bucket, params = options(m, n, k, replay_commit)
    plan = dict(requirement=requirement, shape=[m, n, k], layout='TT', chip='sbx2', dies=2,
                cu_count=96, backend='auto', resolved_backend='gluon', config=config_path,
                config_bucket=bucket, launch_options=params, grid=[1 if args.single_tg_perf else grid],
                seed=seed, repeats=3 if args.single_tg_perf else 1,
                correctness_dispatches=3 if args.single_tg_perf else 0,
                execution_warmups=1 if args.single_tg_perf else 0,
                full_output_validation=not args.single_tg_perf,
                single_tg_scope='Original dependency-free TG work relocated to TG0; original MNK/strides/instructions',
                replay_git_commit=replay_commit, current_toolchain=args.current_toolchain, output=str(output))
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    marker = 'BOLTOPS_SBX2_BENCHMARK_OUTPUT'
    if os.environ.get(marker) != str(output):
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(('GPU_', 'HIP_', 'HSA_', 'ROCR_', 'SOC_', 'DCU_', 'RPY_', 'RBPY_')) or key in (
                    'BOLTOPS_GEMM_CONFIG_DIR', 'TRITON_BUFFER_CACHE_SWIZZLE',
                    'TRITON_MLS_B_INTERLEAVE', 'TRITON_HCU_MATRIX_STORE_INSN', 'LD_PRELOAD'):
                env.pop(key, None)
        if args.current_toolchain:
            for key in ('PMD_PATH', 'TRITON_HIP_CLANG_PATH'):
                if not env.get(key) or not Path(env[key]).exists():
                    parser.error(f'--current-toolchain requires an existing {key} from the PMD environment')
        else:
            env.pop('TRITON_OFFLINE_BUILD', None)
            assert env.get('PMD_PATH') == '/opt/rocm/pmd', 'Source the specified path.sh first.'
            assert env.get('TRITON_HIP_CLANG_PATH') == '/public/home/duyubo/clang/install-dtk-7.2.1/bin/clang'
        if args.triton_build:
            env['PYTHONPATH'] = str(args.triton_build.resolve())
        env.update(GPU_CHIP='sbx2', GPU_OUT='g', GPU_DFLAGS="['StatLog','TT0','Perf']",
                   GPU_ARGS="['--abs-max-tick=10000000000']", AMDGCN_USE_BUFFER_OPS='1',
                   HSA_TOOLS_LIB='/opt/rocm/lib/xprofiler/libperf_gen_helper.so',
                   TRITON_CACHE_DIR=str(output / 'cache'), XDG_CACHE_HOME=str(output / 'xdg-cache'),
                   TMPDIR=str(output / 'temp'), RPY_HISTORY_LOG=str(output / 'history.log'),
                   RPY_HISTORY_SIZE='1024', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        env[marker] = str(output)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('cache', 'xdg-cache', 'temp'):
        (output / name).mkdir()
    os.chdir(output)
    # The PMD backend otherwise shares a history lock across benchmark processes.
    original_open = builtins.open
    shared_lock = '/tmp/' + getpass.getuser() + '.history.log.lock'
    def private_open(file, *positional, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file) == shared_lock:
            file = output / 'history.lock'
        return original_open(file, *positional, **kwargs)
    builtins.open = private_open
    source_root = restore_boltops(output, replay_commit)
    sys.path.insert(0, str(source_root))
    import torch
    import triton
    from triton import knobs
    import triton._C.libtriton as lib
    public = importlib.import_module('boltops.gemm.gemm_a16w16')
    kernels = importlib.import_module('boltops.gemm.gluon.gemm_a16w16')
    streamk = importlib.import_module('boltops.gemm.gluon.gemm_a16w16_streamk')
    fn = getattr(kernels, SYMBOL)
    assert torch.cuda.get_device_properties(0).multi_processor_count == 96
    target = triton.runtime.driver.active.get_current_target()
    assert target.arch == 'gfx946'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    table, records, slots = streamk._make_schedule(m, n, k, 96)
    original_tasks = [[list(item) for item in table[pid * records:(pid + 1) * records] if item[2]]
                      for pid in range(grid)]
    selected_pid = next(pid for pid, tasks in enumerate(original_tasks)
                        if all(task[3] >= 0 for task in tasks) and any(task[3] == 0 for task in tasks))
    selected_tasks = original_tasks[selected_pid]
    plan.update(schedule_records=records, split_slots=slots, selected_original_TG=selected_pid,
                selected_original_tasks=selected_tasks)
    save(output / 'plan.json', plan)
    save(output / 'schedule.json', dict(grid=[grid], records=records, slots=slots, table=table))
    compiler_options = {key: tuple(value) if isinstance(value, list) else value
                        for key, value in params.items()
                        if key not in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'NUM_SMS')}
    original_compile = streamk._compile_kernel
    captured, setup_dispatches = [], []
    class ObservedCompiled:
        def __init__(self, handle):
            self.handle = handle
        def __getattr__(self, name):
            return getattr(self.handle, name)
        @property
        def run(self):
            real = self.handle.run
            def launch(*args):
                assert tuple(args[:3]) == (grid, 1, 1)
                assert args[5] == (self.handle.metadata.num_warps, self.handle.metadata.num_ctas, 98304)
                torch.cuda.synchronize()
                before = set((output / 'g').glob('*/*/stats.txt'))
                real(*args)
                torch.cuda.synchronize()
                sample = read_sample(output, before, grid)
                captured.append((self.handle, sample))
            return launch
    def observe_compile(kernel, cm, cn, ck, count, opts, actual_target):
        assert kernel is fn and (cm, cn, ck, count) == (m, n, k, records)
        assert dict(opts) == compiler_options
        return ObservedCompiled(original_compile(kernel, cm, cn, ck, count, opts, actual_target))
    if not args.single_tg_perf:
        streamk._compile_kernel = observe_compile

    def prepare(input_seed, kind):
        generator = torch.Generator().manual_seed(input_seed)
        def tensor(shape):
            if kind == 'ones':
                return torch.ones(shape, dtype=torch.bfloat16)
            if kind == 'uniform_random':
                return (torch.rand(shape, generator=generator) - .5).bfloat16()
            return (torch.randn(shape, generator=generator) * .5).bfloat16()
        ah, bh = tensor((m, k)), tensor((k, n))
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
        return ah, bh, ar, a, ai, br, b, bi, cr, cr[guard:-guard].view(m, n), ci

    def launch(input_seed, kind, phase, index):
        ah, bh, ar, a, ai, br, b, bi, cr, c, ci = prepare(input_seed, kind)
        before = set((output / 'g').glob('*/*/stats.txt'))
        workspace = flags = None
        if args.single_tg_perf:
            # Replay a dependency-free original TG. A publisher's partial tile
            # and flags are checked as well as the complete output tile.
            isolated_table = list(table)
            isolated_table[:records] = [tuple(item) for item in selected_tasks] + [(0, 0, 0, 0)] * (records - len(selected_tasks))
            schedule = torch.tensor(isolated_table, dtype=torch.int32).cuda()
            workspace = torch.full((max(slots, 1), 256, 128), float('nan'), dtype=torch.float32).cuda()
            flags = torch.zeros(max(slots * 8, 1), dtype=torch.int32).cuda()
            edges = [torch.empty(1, dtype=torch.bfloat16).cuda() for _ in range(3)]
            handle = original_compile(fn, m, n, k, records, tuple(sorted(compiler_options.items())), target)
            run = handle.run
            values = (a, b.T, c, schedule, workspace, flags, *edges, 1)
            stream = torch.cuda.current_stream().cuda_stream
            torch.cuda.synchronize()
            before = set((output / 'g').glob('*/*/stats.txt'))
            run(1, 1, 1, stream, handle.function,
                (handle.metadata.num_warps, handle.metadata.num_ctas, 98304),
                handle.launch_metadata((1, 1, 1), stream, *values),
                knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *values)
            torch.cuda.synchronize()
            sample = read_sample(output, before, 1)
            tasks = selected_tasks
        else:
            assert public.gemm_a16w16(a, b.T, out=c, backend='auto') is c
            torch.cuda.synchronize()
            assert len(captured) == 1, 'Expected one public-API Stream-K GEMM.'
            handle, sample = captured[0]
            all_paths = set((output / 'g').glob('*/*/stats.txt')) - before
            reset_paths = all_paths - {output / sample['stats_path']}
            for path in sorted(reset_paths):
                text = path.read_text()
                assert 'End Simulation Statistics' in text
                mmops = sum(int(value) for value in re.findall(
                    r'^system\.shaders\d+\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)', text, re.M))
                assert mmops == 0, 'Only the native flag-reset dispatch may precede the GEMM.'
                cycles = float(re.search(r'^system.gcSimCycles\s+([\deE.+-]+)', text, re.M)[1])
                setup_dispatches.append(dict(kind='streamk_flag_reset', stats_path=str(path.relative_to(output)),
                    stats_sha256=digest(path), kernel_cycles=cycles, kernel_us=cycles * 455 / 1e6, MMOP=0))
            assert len(setup_dispatches) <= 1
            tasks = None
        assembly = handle.asm['amdgcn']
        if not (output / 'kernel.amdgcn').exists():
            (output / 'kernel.amdgcn').write_text(assembly)
            (output / 'kernel.hsaco').write_bytes(handle.asm['hsaco'])
        assert hashlib.sha256(assembly.encode()).hexdigest() == digest(output / 'kernel.amdgcn')
        assert handle.n_spills == 0
        if archived and args.single_tg_perf:
            expected = archived['assembly']['allocated_hsaco_sections_sha256']
            compiled = executable_sections(handle.asm['hsaco'])
            assert compiled == {name: expected[name] for name in compiled}
        host = cr.cpu()
        actual = host[guard:-guard].view(m, n)
        outside = nonfinite = elements = 0
        maximum = 0.
        def compare(observed, expected):
            nonlocal outside, nonfinite, elements, maximum
            outside += int((~torch.isclose(observed, expected, atol=.001, rtol=.01)).sum())
            nonfinite += int((~torch.isfinite(observed)).sum())
            elements += observed.numel()
            maximum = max(maximum, float((observed - expected).abs().max()))
        if tasks is None:
            bf = bh.float()
            for row in range(0, m, 128):
                expected = (ah[row:row + 128].float() @ bf).bfloat16().float()
                compare(actual[row:row + expected.shape[0]].float(), expected)
        else:
            wh, fh = workspace.cpu(), flags.cpu()
            expected_flags = torch.zeros_like(fh)
            for tile, offset, length, tag in tasks:
                row, col = tile // (n // 128) * 256, tile % (n // 128) * 128
                expected = ah[row:row + 256, offset:offset + length].float() @ bh[offset:offset + length, col:col + 128].float()
                if tag == 0:
                    compare(actual[row:row + 256, col:col + 128].float(), expected.bfloat16().float())
                    actual[row:row + 256, col:col + 128] = float('nan')
                else:
                    compare(wh[tag - 1], expected)
                    wh[tag - 1] = float('nan')
                    expected_flags[(tag - 1) * 8:tag * 8] = 1
            assert torch.equal(fh, expected_flags) and bool(torch.isnan(wh).all())
        canary = bool(torch.equal(host[:guard], ci[:guard]) and torch.equal(host[-guard:], ci[-guard:]))
        inputs = bool(torch.equal(ar.cpu().view(torch.uint8), ai.view(torch.uint8)) and
                      torch.equal(br.cpu().view(torch.uint8), bi.view(torch.uint8)))
        unselected = not args.single_tg_perf or bool(torch.isnan(actual).all())
        accuracy = dict(status='PASS', elements=elements, atol=.001, rtol=.01,
            outside_tolerance=outside, nonfinite=nonfinite, max_abs=maximum,
            canary_ok=canary, inputs_byte_identical=inputs, unselected_output_nan=unselected)
        assert outside == nonfinite == 0 and canary and inputs and unselected, accuracy
        expected_ops = 2 * 256 * 128 * sum(task[2] for task in tasks) if tasks is not None else 2 * m * n * k
        assert sample['MMOP'] * 8192 == expected_ops, (sample['MMOP'], expected_ops)
        sample.update(index=index, phase=phase, seed=input_seed, input_kind=kind, accuracy=accuracy)
        save(output / f'{phase}-{index}.json', sample)
        print(json.dumps(dict(requirement=requirement, phase=phase, index=index, kernel_us=sample['kernel_us'],
                              completed_TGs_per_die=sample['completed_TGs_per_die'], accuracy=accuracy)), flush=True)
        return sample, handle

    validation = []
    if args.single_tg_perf:
        for index, kind in enumerate(('uniform_random', 'ones', 'uniform_random')):
            sample, _ = launch(seed + index * 1000, kind, 'correctness', index)
            validation.append(sample)
        samples = []
        for index in range(3):
            sample, handle = launch(seed, 'normal_random', 'warmup' if index == 0 else 'timed', index)
            samples.append(sample)
        # The viewer opens XCD0 by default. Do not archive an idle XCD0 trace
        # just because the independently scheduled XCD1 repeat was faster.
        candidates = [item for item in samples[1:] if item['wavefront_trace']['die'] == 0]
        assert candidates, 'No timed repeat on die 0; recapture before archiving.'
        selected = min(candidates, key=lambda item: item['dispatch_cycles'])
        assert selected['wavefront_trace']['embedded_hsaco_sha256'] == digest(output / 'kernel.hsaco')
    else:
        sample, handle = launch(seed, 'normal_random', 'cold', 0)
        samples, selected = [sample], sample
    metadata = handle.metadata._asdict()
    metadata['target'] = asdict(metadata['target'])
    resources = dict(LDS_bytes=98304, compiler_shared_bytes=handle.metadata.shared,
                     VGPR=handle.n_regs, spill=handle.n_spills, metadata=metadata)
    shutil.copyfile(output / selected['perf_path'], output / 'single_tg.perf')
    environment = {key: value for key, value in sorted(os.environ.items()) if key.startswith(
        ('GPU_', 'PMD_', 'TRITON_', 'HSA_', 'HIP_', 'ROCR_', 'AMDGCN_', 'OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_'))
        or key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'LLVM_SYSPATH', 'HCU_PIPELINE_ENV_REV')}
    flag_reset_cycles = sum(item['kernel_cycles'] for item in setup_dispatches)
    flag_reset_us = sum(item['kernel_us'] for item in setup_dispatches)
    total_cycles = selected['dispatch_cycles'] + flag_reset_cycles
    total_us = selected['kernel_us'] + flag_reset_us
    operations = 2 * m * n * k if not args.single_tg_perf else 2 * 256 * 128 * sum(task[2] for task in selected_tasks)
    sections = executable_sections(handle.asm['hsaco'])
    expected = archived['assembly']['allocated_hsaco_sections_sha256'] if archived else None
    environment_script = None if args.current_toolchain else ENVIRONMENT_SCRIPT
    result = dict(requirement=requirement, status='PASS', shape=[m, n, k], single_tg=args.single_tg_perf,
        executable_sections=sections,
        matches_measured_executable=(sections == {name: expected[name] for name in sections} if expected else None),
        performance=dict(kernel_cycles=total_cycles, kernel_us=total_us,
            kernel_tops=operations / (total_us * 1e6),
            gemm_kernel_cycles=selected['dispatch_cycles'], gemm_kernel_us=selected['kernel_us'],
            gemm_kernel_tops=operations / (selected['kernel_us'] * 1e6),
            launched_TGs=1 if args.single_tg_perf else grid, samples=len(samples),
            flag_reset_cycles=flag_reset_cycles, flag_reset_us=flag_reset_us,
            gpu_dispatch_total_us=total_us),
        validation=validation if args.single_tg_perf else selected['accuracy'], samples=samples, setup_dispatches=setup_dispatches,
        selected=selected, plan=plan, resources=resources, benchmark_sha256=digest(__file__),
        assembly_sha256=digest(output / 'kernel.amdgcn'), hsaco_sha256=digest(output / 'kernel.hsaco'),
        perf_sha256=digest(output / 'single_tg.perf'), stats_sha256=selected['stats_sha256'],
        source_sha256=digest(source_root / SOURCE), config_sha256=digest(source_root / config_path),
        public_api_sha256=digest(source_root / PUBLIC),
        source_dependencies=[dict(path=path, sha256=digest(source_root / path)) for path in DEPENDENCIES],
        runtime=dict(host=socket.gethostname(), torch=torch.__version__, triton=triton.__version__,
            triton_file=triton.__file__, libtriton_sha256=digest(lib.__file__),
            clang_sha256=digest(os.environ['TRITON_HIP_CLANG_PATH']), environment_script=environment_script,
            environment_script_sha256=digest(environment_script) if environment_script else None,
            environment_variables=environment))
    save(output / 'result.json', result)
    print(json.dumps(dict(status='PASS', requirement=requirement, performance=result['performance'])), flush=True)


if __name__ == '__main__':
    main()
