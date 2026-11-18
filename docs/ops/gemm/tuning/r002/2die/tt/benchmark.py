# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Validate and measure a complete BF16 GEMM through the public API on PMD."""
import argparse
import configparser
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import statistics
import sys

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'boltops/gemm/config.py').is_file())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--single-tg-perf', action='store_true')
    mode.add_argument('--perf', action='store_true', help='capture the full output grid')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path,
                        help='reuse an existing measured compilation cache')
    parser.add_argument('--expect-binary-sha256',
                        help='reject a different compiled binary before dispatch')
    args = parser.parse_args()
    record_path = args.record.resolve()
    record = json.loads(record_path.read_text())
    environment = record['environment']
    case = dict(shape=record['shape'], dies=environment['dies'], layout=record['layout'],
                seed=record['benchmark']['seed'], repeats=record['benchmark']['repeats'],
                validation_count=record['benchmark'].get('validation_count', 3),
                requirement=record['requirement'], config=record['config']['path'])
    if args.single_tg_perf or args.perf:
        case['repeats'] = 3
    marker = 'BOLTOPS_BF16_TT_MLS_REPLAY_RECORD'
    if os.environ.get(marker) != str(record_path):
        replay_env = dict(os.environ)
        for key, value in environment['environment_variables'].items():
            if value is None:
                replay_env.pop(key, None)
            else:
                replay_env[key] = value
        if args.single_tg_perf or args.perf:
            key = 'single_tg_perf' if args.single_tg_perf else 'perf'
            replay_env.update(record[key]['environment_overrides'])
        replay_env[marker] = str(record_path)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], replay_env)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache_dir = args.cache_dir.resolve() if args.cache_dir else output / 'cache'
    os.chdir(output)
    os.environ['TRITON_CACHE_DIR'] = str(cache_dir)
    os.environ['TRITON_HOME'] = str(output / 'triton-home')
    sys.path.insert(0, str(REPO))
    import torch
    import triton

    torch.set_num_threads(1)
    assert case['dies'] == environment['dies'] == 2
    assert torch.cuda.get_device_properties(0).multi_processor_count == environment['cu_count']
    assert case['layout'] in ('TN', 'TT')
    m, n, k = case['shape']
    public = importlib.import_module('boltops.gemm.gemm_a16w16')
    kernels = importlib.import_module('boltops.gemm.gluon.gemm_a16w16')
    config_path = REPO / case['config']
    config = json.loads(config_path.read_text())['profiles']['DTYPE=bfloat16-OUT=bfloat16'][f'M_EQ_{m}']
    options = {key: tuple(value) if isinstance(value, list) else value
               for key, value in config.items()}
    symbol = options.pop('kernel')
    options['B_K_CONTIGUOUS'] = case['layout'] == 'TN'
    fn = getattr(kernels, symbol)
    grid = triton.cdiv(m, options['BLOCK_SIZE_M']) * triton.cdiv(n, options['BLOCK_SIZE_N'])
    a = torch.empty((m, k), dtype=torch.bfloat16, device='cuda')
    w = (torch.empty((n, k), dtype=torch.bfloat16, device='cuda')
         if case['layout'] == 'TN'
         else torch.empty((k, n), dtype=torch.bfloat16, device='cuda').T)
    storage = torch.full((256 + m*n + 256,), 123., dtype=torch.bfloat16, device='cuda')
    c = storage[256:256 + m*n].view(m, n)
    handle = fn.warmup(a, w, c, M=m, N=n, K=k, grid=(grid,), **options)
    handle._init_handles()
    assert handle.n_spills == 0
    md = json.loads(json.dumps(handle.metadata._asdict(), default=str))
    assert md['wasp_enabled'] and not md['wasp_wdra']
    assert tuple(md['wasp_partition_warps']) == tuple(options['wasp_partition_warps'])
    assert md['num_warps'] == sum(options['wasp_partition_warps'])
    assert md['shared'] == record['resources'].get('shared_bytes', record['resources'].get('LDS_bytes'))
    binary_sha256 = hashlib.sha256(handle.kernel).hexdigest()
    if args.expect_binary_sha256:
        assert binary_sha256 == args.expect_binary_sha256, 'compiled binary differs from the selected measurement'
    Path('kernel.hsaco').write_bytes(handle.kernel)
    assert not re.search(r'\.sgpr_spill_count:\s*[1-9]', handle.asm['amdgcn'])
    (output / 'kernel.amdgcn').write_text('\n'.join(l.rstrip() for l in handle.asm['amdgcn'].splitlines()) + '\n')
    launched = []

    class ObserveLaunch:
        def __getitem__(self, dispatch_grid):
            real = fn[dispatch_grid]

            def invoke(*positional, **kwargs):
                result = real(*positional, **kwargs)
                assert result is handle, 'public API selected a different kernel/config'
                launched.append(result)
                return result
            return invoke

    setattr(kernels, symbol, ObserveLaunch())

    def stats():
        paths = sorted(Path('m5out/0').glob('*/stats.txt'), key=lambda p: int(p.parent.name))
        path = paths[-1]
        text = path.read_text()
        values = dict(re.findall(r'^(system\.(?:gcSimCycles|shaders\d*\.firstWaveStartTick|shaders\d*\.lastWaveEndTick))\s+([\d.eE+-]+)', text, re.M))
        model = configparser.RawConfigParser(strict=False)
        model.read('m5out/config.ini')
        roots = [s for s in model.sections() if re.fullmatch(r'system\.shaders\d*', s)]
        assert len(roots) == environment['dies']
        assert model.getint('gpu_configuration', 'num_chip') == environment['dies']
        periods = {int(model.get(root + '.clk_domain', 'clock')) for root in roots}
        assert periods == {environment['clock_period_ps']['shader']}
        period = periods.pop()
        completed, active, per_die, starts, ends = 0, 0, {}, [], []
        for root in roots:
            counts = list(map(int, re.findall('^' + re.escape(root) + r'\.CUs\d+\.completedWGs\s+(\d+)', text, re.M)))
            assert len(counts) == environment['cu_count'] // environment['dies']
            tgs, cus = sum(counts), sum(x > 0 for x in counts)
            per_die[root] = dict(completed_TGs=tgs, active_CUs=cus)
            completed += tgs
            active += cus
            if tgs:
                starts.append(float(values[root + '.firstWaveStartTick']))
                ends.append(float(values[root + '.lastWaveEndTick']))
        assert completed == (1 if args.single_tg_perf else grid), (completed, grid)
        cycles = float(values['system.gcSimCycles'])
        duration = cycles * period / 1e6
        window = (max(ends) - min(starts)) / 1e6
        assert duration >= window, (duration, window)
        return {'dispatch_cycles': cycles, 'kernel_us': duration,
                'wave_interval_us': window, 'completed_TGs': completed,
                'stats_path': str(path), 'active_CUs': active, 'per_die': per_die}

    def launch():
        before = len(launched)
        torch.cuda.synchronize()
        if args.single_tg_perf:
            result = fn[(1,)](a, w, c, M=m, N=n, K=k, **options)
            assert result is handle
        else:
            result = public.gemm_a16w16(a, w, out=c, backend='auto')
            assert result is c and len(launched) == before + 1
        torch.cuda.synchronize()
        assert hashlib.sha256(handle.kernel).hexdigest() == binary_sha256
        sample = stats()
        if args.single_tg_perf or args.perf:
            directory = Path(sample['stats_path']).parent
            files = [p for p in directory.glob('*.perf') if re.match(r'^\d+_', p.name)]
            assert len(files) == 1, files
            sample['perf_path'] = str(files[0])
        return sample

    def check(golden):
        actual = c.cpu().float()
        if args.single_tg_perf:
            untouched = actual.clone()
            untouched[:options['BLOCK_SIZE_M'], :options['BLOCK_SIZE_N']] = float('nan')
            assert bool(torch.isnan(untouched).all()), 'writes outside selected tile'
            actual = actual[:options['BLOCK_SIZE_M'], :options['BLOCK_SIZE_N']]
            golden = golden[:options['BLOCK_SIZE_M'], :options['BLOCK_SIZE_N']]
        diff = (actual - golden).abs()
        bad = (~torch.isfinite(actual)) | (diff > .001 + .01*golden.abs())
        assert bool((storage[:256].cpu() == 123).all())
        assert bool((storage[-256:].cpu() == 123).all())
        accuracy = {'elements': actual.numel(), 'mismatches': int(bad.sum()),
                    'nonfinite': int((~torch.isfinite(actual)).sum()), 'max_abs': float(diff.max())}
        if bad.any():
            torch.save({'a': a.cpu(), 'w': w.cpu(), 'actual': actual, 'golden': golden}, 'failure.pt')
            raise RuntimeError(str(accuracy))
        return accuracy

    validation = []
    for i in range(1 if (args.single_tg_perf or args.perf) else case['validation_count']):
        gen = torch.Generator().manual_seed(case['seed'] + 1000*i)
        ac = (torch.rand((m, k), generator=gen) - .5).bfloat16()
        bc = (torch.rand((k, n), generator=gen) - .5).bfloat16()
        if i == 1:
            ac.fill_(1)
            bc.fill_(1)
        a.copy_(ac)
        if case['layout'] == 'TN':
            w.copy_(bc.T.contiguous())
        else:
            w.T.copy_(bc)
        c.fill_(float('nan'))
        golden = (ac.float() @ bc.float()).bfloat16().float()
        sample = launch()
        sample['accuracy'] = check(golden)
        validation.append(sample)
    samples = []
    for _ in range(case['repeats']):
        sample = launch()
        sample['accuracy'] = check(golden)
        samples.append(sample)
    warm = samples
    assert warm, 'need timing repeats after validation'
    performance = dict(kernel_us=statistics.median(s['kernel_us'] for s in warm),
                       kernel_cycles=statistics.median(s['dispatch_cycles'] for s in warm),
                       shader_us=statistics.median(s['wave_interval_us'] for s in warm))
    performance['kernel_tops'] = 2*m*n*k/(performance['kernel_us']*1e6) if not args.single_tg_perf else None
    native = lambda key: int(re.search(r'\.' + key + r':\s*(\d+)', handle.asm['amdgcn'])[1])
    native_resources = dict(SGPR=native('sgpr_count'), VGPR=native('vgpr_count'),
                            SGPR_spills=native('sgpr_spill_count'), VGPR_spills=native('vgpr_spill_count'),
                            shared_bytes=md['shared'], waves_per_TG=md['num_warps'])
    assert native_resources['SGPR_spills'] == native_resources['VGPR_spills'] == 0
    result = dict(native_resources=native_resources, requirement=case['requirement'], status='PASS', single_tg=args.single_tg_perf,
                  performance=performance, validation=validation, samples=samples,
                  assembly_sha256=digest('kernel.amdgcn'), benchmark_sha256=digest(__file__),
                  binary_sha256=binary_sha256, metadata=md, VGPR=handle.n_regs, spills=handle.n_spills,
                  public_api_verified=not args.single_tg_perf)
    if args.single_tg_perf or args.perf:
        best = samples[-1]
        import shutil
        if args.perf:
            best = samples[-1]
        filename = 'single_tg.perf' if args.single_tg_perf else 'full_grid.perf'
        shutil.copyfile(best['perf_path'], filename)
        assert Path(filename).read_bytes()[1:8] == b'HYGPERF'
        result['selected'] = dict(best, sha256=digest(filename))
    Path('result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'performance': result['performance']}), flush=True)


if __name__ == '__main__':
    main()
