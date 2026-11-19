# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Replay a recorded dual-die BF16 TN GEMM.

Source the Triton/PMD environment in the record before running. Use --input
to point to record.json and --output to select a new output directory. The
default is one cold full-grid measurement, with required setup dispatches.
--perf isolates representative TG work using the full-grid instructions.
"""
import argparse
import builtins
import configparser
import getpass
import hashlib
import math
import importlib
import json
import os
from pathlib import Path
import re
import statistics
import struct
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sections(code):
    assert code[:5] == b'\x7fELF\x02'
    offset, = struct.unpack_from('<Q', code, 40)
    size, count, names = struct.unpack_from('<HHH', code, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', code, offset+i*size) for i in range(count)]
    strings = code[headers[names][4]:headers[names][4]+headers[names][5]]
    result = {}
    for header in headers:
        name = strings[header[0]:].split(b'\0', 1)[0].decode()
        if name in ('.text','.rodata'):
            result[name] = hashlib.sha256(code[header[4]:header[4]+header[5]]).hexdigest()
    assert '.text' in result
    return result

def measure(output, record, variant, repeats, perf, repo, *, regression=False):
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    original_open = builtins.open
    lock = '/tmp/' + getpass.getuser() + '.history.log.lock'

    def private_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file) == lock:
            file = output / 'history.lock'
        return original_open(file, *args, **kwargs)

    builtins.open = private_open
    sys.path.insert(0, str(repo))
    import torch
    import triton
    import triton._C.libtriton as libtriton

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.get_device_properties(0).multi_processor_count == 96
    assert triton.runtime.driver.active.get_current_target().arch == 'gfx946'
    public = importlib.import_module('boltops.gemm.gemm_a16w16')
    m, n, k = record['shape']
    guard = 256
    g = torch.Generator().manual_seed(record['seed'])
    ah = (torch.rand((m, k), generator=g) - .5).bfloat16()
    wh = (torch.rand((n, k), generator=g) - .5).bfloat16()

    def guarded_input(host):
        padded = torch.full((host.numel() + 2 * guard,), 123., dtype=torch.bfloat16)
        padded[guard:-guard].copy_(host.flatten())
        raw = padded.cuda()
        return raw, raw[guard:-guard].view_as(host), padded

    ar, a, ai = guarded_input(ah)
    wr, w, wi = guarded_input(wh)
    initial = torch.full((m*n + 2*guard,), 123., dtype=torch.bfloat16)
    initial[guard:-guard] = float('nan')
    raw = initial.cuda()
    c = raw[guard:-guard].view(m, n)
    assert a.stride() == w.stride() == (k, 1)
    family = variant['family']
    config = {key: tuple(value) if isinstance(value, list) else value
              for key, value in variant.get('config', {}).items()}
    resources = None
    grid = None
    if family == 'public':
        def launch():
            assert public.gemm_a16w16(a, w, out=c, **config) is c
    elif family == 'streamk':
        module = importlib.import_module(variant['module'])
        runtime = importlib.import_module('boltops.gemm.gluon.gemm_a16w16_streamk')
        kernel = getattr(module, variant['symbol'])
        config.pop('kernel')
        original_compile = runtime._compile_kernel
        resources = {}

        def capture(*args, **kwargs):
            handle = original_compile(*args, **kwargs)
            handle._init_handles()
            if not regression:
                assert sections(handle.asm['hsaco']) == record['assembly']['executable_sections']
            resources.update(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=96 * 1024)
            assert not handle.n_spills, resources
            (output / 'kernel.amdgcn').write_text(handle.asm['amdgcn'])
            (output / 'kernel.hsaco').write_bytes(handle.asm['hsaco'])
            return handle

        if perf:
            grid = 1
            _, records, _ = runtime._make_schedule(m, n, k, 96)
            table = [[0, 0, k, 0], *([[0, 0, 0, 0]] * (records - 1))]
            schedule = torch.tensor(table, dtype=torch.int32).cuda()
            workspace = torch.empty(1, dtype=torch.float32, device=a.device)
            flags = torch.empty(1, dtype=torch.int32, device=a.device)
            edges = [torch.empty(1, dtype=torch.bfloat16, device=a.device) for _ in range(3)]
            options = tuple(sorted((key, value) for key, value in config.items()
                                   if key not in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'NUM_SMS')))
            target = triton.runtime.driver.active.get_current_target()
            handle = capture(kernel, m, n, k, records, options, target, b_k_contiguous=True)
            values = (a, w, c, schedule, workspace, flags, *edges, 1)
            launch_grid = (1, 1, 1)
            stream = torch.cuda.current_stream().cuda_stream

            def launch():
                handle.run(*launch_grid, stream, handle.function,
                           (handle.metadata.num_warps, handle.metadata.num_ctas, 96 * 1024),
                           handle.launch_metadata(launch_grid, stream, *values),
                           triton.knobs.runtime.launch_enter_hook,
                           triton.knobs.runtime.launch_exit_hook, *values)
        else:
            runtime._compile_kernel = capture

            def launch():
                assert public.gemm_a16w16(a, w, out=c, backend='auto') is c
    else:
        config.pop('kernel', None)
        block_m, block_n, _ = (config['BLOCK_SIZE_' + d] for d in 'MNK')
        tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
        module = importlib.import_module(variant['module'])
        kernel = getattr(module, variant['symbol'])
        if family != 'gluon':
            raise ValueError(f'Unsupported replay family {family!r}')
        grid = min(config['NUM_SMS'], tiles)
        args = (a, w, c)
        config.update(M=m, N=n, K=k, B_K_CONTIGUOUS=True)
        full_grid = grid
        if perf:
            grid = 1
        handle = kernel.warmup(*args, grid=(grid,), **config)
        handle._init_handles()
        assert sections(handle.asm['hsaco']) == record['assembly']['executable_sections']
        resources = dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared)
        (output / 'kernel.amdgcn').write_text(handle.asm['amdgcn'])
        (output / 'kernel.hsaco').write_bytes(handle.asm['hsaco'])
        assert not handle.n_spills, resources

        def launch():
            assert kernel[(grid,)](*args, **config) is handle

    stats_root = output / 'model'
    samples = []
    wf = wh.float().T.contiguous()
    for _ in range(repeats):
        raw.copy_(initial)
        torch.cuda.synchronize()
        before = set(stats_root.glob('*/*/stats.txt'))
        launch()
        torch.cuda.synchronize()
        paths = sorted(set(stats_root.glob('*/*/stats.txt')) - before, key=lambda p: int(p.parent.name))
        assert paths, 'PMD emitted no dispatch stats'
        assert len(paths) == 1 or family in ('public', 'streamk'), paths
        dispatches = []
        for path in paths:
            stats = path.read_text()
            assert 'End Simulation Statistics' in stats
            groups = re.findall(r'^(system\d*\.shaders\d+)\.CUs\d+\.completedWGs\s+(\d+)', stats, re.M)
            per_die = {die: sum(int(tgs) for key, tgs in groups if key == die)
                       for die in sorted({key for key, _ in groups})}
            assert len(groups) == 96 and len(per_die) == 2, groups
            if grid is not None:
                assert sum(per_die.values()) == grid, per_die
            if not perf and len(paths) == 1:
                assert all(per_die.values()), per_die
            cycles = int(float(re.search(r'^system.gcSimCycles\s+(\S+)', stats, re.M)[1]))
            frequency = float(re.search(r'^gcFrequency\s+(\S+)', stats, re.M)[1])
            assert math.isclose(frequency, 1000 / 455, rel_tol=1e-6)
            mmops = sum(map(int, re.findall(r'\.numInstrTypeExecuted::MMOP\s+(\d+)', stats)))
            dispatches.append(dict(cycles=cycles, TGs_per_die=per_die, mmops=mmops,
                                   stats=str(path.relative_to(output)), sha256=digest(path)))
        if family == 'streamk' and not perf:
            assert len(dispatches) == 2, dispatches
            assert list(dispatches[-1]['TGs_per_die'].values()) == [48, 48], dispatches
        ini = configparser.RawConfigParser(strict=False)
        ini.read(stats_root / 'config.ini')
        assert ini.getint('gpu_configuration', 'num_chip') == 2
        assert all(ini.getint(die + '.clk_domain', 'clock') == 455 for die in per_die)
        assert all(ini.get('gpu_configuration', key).lower() in ('false', '0')
                   for key in ('ipc_conn', 'skip_cal', 'skip_kernel_exec'))
        host = raw.cpu()
        assert (host[:guard] == 123).all() and (host[-guard:] == 123).all()
        assert torch.equal(ar.cpu(), ai) and torch.equal(wr.cpu(), wi)
        observed = host[guard:-guard].view(m, n)
        regions = [(i, 0, min(128, m-i), n) for i in range(0, m, 128)]
        if perf and family == 'streamk':
            regions = [(0, 0, 256, 128)]
        elif perf:
            ids = range(0, tiles, full_grid) if 'NUM_SMS' in config else [0]
            nc = triton.cdiv(n, block_n)
            mt = triton.cdiv(m, block_m)
            grouping = config.get('GROUP_M', config.get('GROUP_SIZE_M', 1))
            ordered = [(row, col) for first in range(0, mt, grouping)
                       for col in range(nc) for row in range(first, min(first + grouping, mt))]
            regions = [(ordered[i][0]*block_m, ordered[i][1]*block_n, block_m, block_n)
                       for i in ids]
        elements = sum(rows*cols for _, _, rows, cols in regions)
        assert sum(d['mmops'] for d in dispatches) == 2*elements*k//8192
        max_error = 0.
        for row, col, rows, cols in regions:
            expected = (ah[row:row+rows].float() @ wf[:, col:col+cols]).bfloat16().float()
            actual = observed[row:row+rows, col:col+cols]
            torch.testing.assert_close(actual.float(), expected, atol=.001, rtol=.01)
            max_error = max(max_error, float((actual.float() - expected).abs().max()))
            if perf:
                actual.fill_(float('nan'))
        sample = dict(us=sum(d['cycles'] for d in dispatches)*.000455,
                      dispatches=dispatches, max_abs=max_error,
                      validated_elements=sum(rows*cols for _, _, rows, cols in regions))
        if perf:
            assert sorted(per_die.values()) == [0, 1]
            assert torch.isnan(observed).all()
            files = [p for p in paths[0].parent.glob('*.perf') if not p.name.startswith('stats_')]
            assert len(files) == 1 and files[0].read_bytes()[1:8] == b'HYGPERF', files
            sample['perf'] = str(files[0].relative_to(output))
        samples.append(sample)
        print(json.dumps(sample), flush=True)
    frequency = 1000 / 455
    performance = statistics.median(s['us'] for s in samples[1:] or samples)
    result = dict(status='PASS', shape=record['shape'], seed=record['seed'], layout='TN',
                  strides=[list(a.stride()), list(w.stride())], variant=variant,
                  resources=resources, samples=samples, kernel_us=performance,
                  kernel_tops=2*m*n*k/(performance*1e6), clock_period_ps=455,
                  mmac_util_percent=100*2*m*n*k/(performance*1e6)/(96*4096*frequency/1000),
                  runtime=dict(torch=torch.__version__, triton=triton.__version__,
                               compiler={key: digest(path) for key, path in
                                         dict(libtriton=libtriton.__file__, clang=os.environ['TRITON_HIP_CLANG_PATH']).items()}),
                  environment={k: v for k, v in os.environ.items() if k.startswith(
                      ('GPU_', 'TRITON_', 'AMDGCN_', 'PMD_')) or k == 'HSA_TOOLS_LIB'})
    if resources:
        result.update(assembly_sha256=digest(output/'kernel.amdgcn'), hsaco_sha256=digest(output/'kernel.hsaco'))
    (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--variant', default='winner')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repo', type=Path, default=next(
        p for p in Path(__file__).resolve().parents if (p/'boltops/gemm/config.py').is_file()))
    p.add_argument('--repeats', type=int)
    p.add_argument('--perf', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--regression', action='store_true', help='check the current public Stream-K path')
    args = p.parse_args()
    repeats = args.repeats if args.repeats is not None else (3 if args.perf else 1)
    if repeats < 1:
        p.error('--repeats must be positive')
    output = args.output.resolve()
    record = json.loads(args.input.read_text())
    variant = record['variants'][args.variant]
    if args.regression and (variant['family'] != 'streamk' or args.perf or repeats != 1):
        p.error('--regression requires one full-grid Stream-K launch')
    if args.dry_run:
        print(json.dumps(dict(shape=record['shape'], seed=record['seed'], variant=variant,
                              repeats=repeats, single_tg=args.perf, chip='sbx2', cu_count=96), indent=2))
        return
    flags = "['StatLog','TT0','Perf']" if args.perf else "['StatLog']"
    os.environ.update(GPU_CHIP='sbx2', GPU_OUT=str(output/'model'), GPU_DFLAGS=flags,
                      GPU_ARGS="['--abs-max-tick=10000000000']", RPY_HISTORY_LOG=str(output/'history.log'),
                      TRITON_CACHE_DIR=str(output/'cache'), TRITON_HOME=str(output/'triton-home'),
                      AMDGCN_USE_BUFFER_OPS='1')
    for key in ('GPU_GFLAGS', 'BOLTOPS_GEMM_CONFIG_DIR', 'HSA_TOOLS_LIB',
                'TRITON_MLS_B_INTERLEAVE', 'TRITON_BUFFER_CACHE_SWIZZLE'):
        os.environ.pop(key, None)
    if args.perf:
        assert variant['family'] != 'public'
        os.environ['HSA_TOOLS_LIB'] = '/opt/rocm/lib/xprofiler/libperf_gen_helper.so'
    measure(output, record, variant, repeats, args.perf, args.repo.resolve(), regression=args.regression)


if __name__ == '__main__':
    main()
