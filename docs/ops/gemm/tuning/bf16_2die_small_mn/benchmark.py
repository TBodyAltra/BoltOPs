# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Replay the recorded BF16 TT or TN path on sbx2."""
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
import statistics
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(output, record, variant, repeats, perf, repo, automatic=False):
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
    layout = record['layout']
    tn = layout == 'TN'
    guard = 65536
    g = torch.Generator().manual_seed(record['seed'])
    ah = (torch.rand((m, k), generator=g) - .5).bfloat16()
    wh = torch.empty((n, k) if tn else (k, n), dtype=torch.bfloat16)
    for row in range(0, wh.shape[0], 512):
        tile = wh[row:row+512]
        tile.copy_(torch.rand(tile.shape, generator=g) - .5)

    def guarded_input(host):
        padded = torch.full((host.numel() + 2 * guard,), 123., dtype=torch.bfloat16)
        padded[guard:-guard].copy_(host.flatten())
        raw = padded.cuda()
        return raw, raw[guard:-guard].view_as(host), padded

    ar, a, ai = guarded_input(ah)
    wr, weight, wi = guarded_input(wh)
    w = weight if tn else weight.T
    del wh
    weights = wi[guard:-guard].view(n,k) if tn else wi[guard:-guard].view(k,n).T
    del weight
    initial = torch.full((m*n + 2*guard,), 123., dtype=torch.bfloat16)
    initial[guard:-guard] = float('nan')
    raw = initial.cuda()
    c = raw[guard:-guard].view(m, n)
    assert a.stride() == (k, 1) and w.stride() == ((k, 1) if tn else (1, n))
    family = variant['family']
    config = {key: tuple(value) if isinstance(value, list) else value
              for key, value in variant.get('config', {}).items()}
    resources = None
    grid = None
    single_tg = perf and family != 'split'
    if family in ('public', 'split') or automatic:
        if family != 'public':
            resources = {}

            class CaptureKernel:
                def __init__(self, kernel, artifact):
                    self.kernel = kernel
                    self.artifact = artifact

                def __getitem__(self, grid_config):
                    def run(*args, **kwargs):
                        handle = self.kernel[grid_config](*args, **kwargs)
                        path = output / (self.artifact + '.hsaco')
                        if not path.exists():
                            handle._init_handles()
                            info = dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared)
                            assert not info['spills'], info
                            if self.artifact == 'kernel':
                                resources.update(info)
                            else:
                                resources['reduction'] = info
                            path.write_bytes(handle.asm['hsaco'])
                            path.with_suffix('.amdgcn').write_text(handle.asm['amdgcn'])
                        return handle
                    return run

            if family == 'gluon':
                module = importlib.import_module('boltops.gemm.gluon.gemm_a16w16')
                default = ('_gemm_a16w16_kernel' if hasattr(module, '_gemm_a16w16_kernel')
                           else '_gemm_a16w16_tt_kernel')
                kernels = [(module.__name__, variant.get('public_symbol', default), 'kernel')]
            else:
                kernels = [('boltops.gemm.triton.gemm_a16w16', '_gemm_a16_w16_kernel', 'kernel')]
                if family == 'split':
                    kernels.append(('boltops.gemm.triton.splitk_reduce', '_gemm_splitk_reduce_kernel', 'reduction'))
            for mod, attr, artifact in kernels:
                module = importlib.import_module(mod)
                setattr(module, attr, CaptureKernel(getattr(module, attr), artifact))

        if family == 'public' or automatic:
            def launch():
                assert public.gemm_a16w16(a, w, out=c) is c
        else:
            from boltops.gemm.triton.gemm_a16w16 import gemm_a16w16 as triton_gemm
            from boltops.utility.arch_info import get_device_info

            device = get_device_info(a.device)

            def launch():
                triton_gemm(a, w, c, bias=None, activation=None, device=device, config=config)
    else:
        block_m, block_n, _ = (config['BLOCK_SIZE_' + d] for d in 'MNK')
        tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
        module = importlib.import_module(variant['module'])
        kernel = getattr(module, variant['symbol'])
        if family == 'gluon':
            grid = min(config['NUM_SMS'], tiles) if 'NUM_SMS' in config else tiles
            args = (a, w, c)
            config.update(M=m, N=n, K=k, B_K_CONTIGUOUS=tn)
        else:
            assert family == 'triton'
            grid = tiles
            b = w.T
            args = (a, b, None, c, m, n, k, *a.stride(), *b.stride(), m*n, *c.stride())
            config.update(activation='', use_activation=False, ADD_BIAS=False, SKIP_REDUCE=False)
        full_grid = grid
        if single_tg:
            grid = 1
        handle = kernel.warmup(*args, grid=(grid,), **config)
        handle._init_handles()
        resources = dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared)
        (output / 'kernel.amdgcn').write_text(handle.asm['amdgcn'])
        (output / 'kernel.hsaco').write_bytes(handle.asm['hsaco'])
        assert not handle.n_spills, resources

        def launch():
            assert kernel[(grid,)](*args, **config) is handle

    stats_root = output / 'model'
    samples = []
    valid = True
    total_bad = 0
    for _ in range(repeats):
        raw.copy_(initial)
        torch.cuda.synchronize()
        before = set(stats_root.glob('*/*/stats.txt'))
        launch()
        torch.cuda.synchronize()
        paths = sorted(set(stats_root.glob('*/*/stats.txt')) - before, key=lambda p: int(p.parent.name))
        assert paths, 'PMD emitted no dispatch stats'
        assert len(paths) == 1 or family in ('public', 'split'), paths
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
            if not perf and len(paths) == 1 and sum(per_die.values()) >= 2:
                assert all(per_die.values()), per_die
            cycles = int(float(re.search(r'^system.gcSimCycles\s+(\S+)', stats, re.M)[1]))
            dispatches.append(dict(cycles=cycles, TGs_per_die=per_die,
                                   stats=str(path.relative_to(output)), sha256=digest(path)))
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
        regions = [(i, j, min(128, m-i), min(2048,n-j))
                   for i in range(0,m,128) for j in range(0,n,2048)]
        if single_tg:
            ids = range(0, tiles, full_grid) if 'NUM_SMS' in config else [0]
            nc = triton.cdiv(n, block_n)
            regions = [(i//nc*block_m, i%nc*block_n,
                        min(block_m,m-i//nc*block_m), min(block_n,n-i%nc*block_n)) for i in ids]
        max_error = 0.
        mismatches = 0
        for row, col, rows, cols in regions:
            expected = (ah[row:row+rows].float() @ weights[col:col+cols].float().T).bfloat16().float()
            actual = observed[row:row+rows, col:col+cols]
            diff = torch.abs(actual.float() - expected)
            mismatches += int((~torch.isclose(actual.float(), expected, atol=.001, rtol=.01)).sum())
            max_error = max(max_error, float(torch.nan_to_num(diff, nan=0).max()))
            if single_tg:
                actual.fill_(float('nan'))
        valid &= mismatches == 0
        total_bad += mismatches
        sample = dict(us=sum(d['cycles'] for d in dispatches)*.000455, dispatches=dispatches,
                      max_abs=max_error, mismatches=mismatches,
                      validated_elements=sum(rows*cols for _, _, rows, cols in regions))
        if single_tg:
            assert sorted(per_die.values()) == [0, 1]
            assert torch.isnan(observed).all()
        if perf:
            files = [p for p in paths[0].parent.glob('*.perf') if not p.name.startswith('stats_')]
            assert len(files) == 1 and files[0].read_bytes()[1:8] == b'HYGPERF', files
            sample['perf'] = str(files[0].relative_to(output))
        samples.append(sample)
        print(json.dumps(sample), flush=True)
    frequency = 1000 / 455
    performance = statistics.median(s['us'] for s in samples[1:] or samples)
    result = dict(status='PASS' if valid else 'NUMERICAL_FAILURE', mismatches=total_bad,
                  shape=record['shape'], seed=record['seed'], layout=layout,
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
    if family == 'split':
        result['reduction_assembly_sha256'] = digest(output/'reduction.amdgcn')
        result['reduction_hsaco_sha256'] = digest(output/'reduction.hsaco')
    (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--variant', default='winner')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repo', type=Path, default=next(
        (p for p in Path(__file__).resolve().parents if (p/'boltops/gemm/config.py').is_file()), None))
    p.add_argument('--repeats', default=1, type=int)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--auto', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--perf', action='store_true')

    args = p.parse_args()
    assert args.repeats > 0
    output = args.output.resolve()
    record = json.loads(args.input.read_text())
    variant = record['variants'][args.variant]
    if args.seed is not None:
        record['seed'] = args.seed
    if args.auto and args.perf:
        p.error('--auto uses the full grid')
    if args.dry_run:
        print(json.dumps(dict(shape=record['shape'], layout=record['layout'], variant=variant,
                              chip='sbx2', cu_count=96, seed=record['seed'], repeats=args.repeats,
                              single_tg=args.perf and variant['family'] != 'split'), indent=2))
        return
    flags = "['StatLog','TT0','Perf']" if args.perf else "['StatLog']"
    os.environ.update(GPU_CHIP='sbx2', GPU_OUT=str(output/'model'), GPU_DFLAGS=flags,
                      GPU_ARGS="['--abs-max-tick=10000000000']", RPY_HISTORY_LOG=str(output/'history.log'),
                      TRITON_CACHE_DIR=str(output/'cache'), TRITON_HOME=str(output/'triton-home'),
                      AMDGCN_USE_BUFFER_OPS='1')
    for key in ('GPU_GFLAGS', 'BOLTOPS_GEMM_CONFIG_DIR', 'HSA_TOOLS_LIB',
                'TRITON_MLS_B_INTERLEAVE', 'TRITON_BUFFER_CACHE_SWIZZLE'):
        os.environ.pop(key, None)
    if variant['family'] == 'mls':
        os.environ.update(TRITON_MLS_B_INTERLEAVE='2', TRITON_BUFFER_CACHE_SWIZZLE='1')
    if args.perf:
        assert variant['family'] in ('triton', 'gluon', 'split')
        os.environ['HSA_TOOLS_LIB'] = '/opt/rocm/lib/xprofiler/libperf_gen_helper.so'
    measure(output, record, variant, args.repeats, args.perf, args.repo.resolve(), args.auto)


if __name__ == '__main__':
    main()
