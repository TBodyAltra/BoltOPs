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
    parser.add_argument('--single-tg-perf', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record_path = args.record.resolve()
    record = json.loads(record_path.read_text())
    environment = record['environment']
    case = dict(shape=record['shape'], dies=environment['dies'], layout=record['layout'],
                seed=record['benchmark']['seed'], repeats=record['benchmark']['repeats'],
                requirement=record['requirement'], config=record['config']['path'])
    if args.single_tg_perf:
        case['repeats'] = 3
    marker = 'BOLTOPS_R001_REPLAY_RECORD'
    if os.environ.get(marker) != str(record_path):
        replay_env = dict(os.environ)
        for key, value in environment['environment_variables'].items():
            if value is None:
                replay_env.pop(key, None)
            else:
                replay_env[key] = value
        if args.single_tg_perf:
            replay_env.update(record['single_tg_perf']['environment_overrides'])
        replay_env[marker] = str(record_path)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], replay_env)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    os.environ['TRITON_HOME'] = str(output / 'triton-home')
    sys.path.insert(0, str(REPO))
    import torch
    import triton

    torch.set_num_threads(1)
    assert case['dies'] == environment['dies'] == 1
    assert case['layout'] == 'TN' and case['shape'] == [64, 64, 2048]
    m, n, k = case['shape']
    public = importlib.import_module('boltops.gemm.gemm_a16w16')
    kernels = importlib.import_module('boltops.gemm.gluon.gemm_a16w16')
    config_path = REPO / case['config']
    config = json.loads(config_path.read_text())['profiles']['DTYPE=bfloat16-OUT=bfloat16']['M_EQ_64']
    options = dict(config)
    symbol = options.pop('kernel')
    fn = getattr(kernels, symbol)
    grid = triton.cdiv(m, options['BLOCK_SIZE_M']) * triton.cdiv(n, options['BLOCK_SIZE_N'])
    a = torch.empty((m, k), dtype=torch.bfloat16, device='cuda')
    w = torch.empty((n, k), dtype=torch.bfloat16, device='cuda')
    storage = torch.full((256 + m*n + 256,), 123., dtype=torch.bfloat16, device='cuda')
    c = storage[256:256 + m*n].view(m, n)
    handle = fn.warmup(a, w, c, M=m, N=n, K=k, grid=(grid,), **options)
    handle._init_handles()
    assert handle.n_spills == 0
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
        values = dict(re.findall(r'^(system\.(?:gcSimCycles|shaders\.firstWaveStartTick|shaders\.lastWaveEndTick))\s+([\d.eE+-]+)', text, re.M))
        completed = sum(map(int, re.findall(r'^system\.shaders\.CUs\d+\.completedWGs\s+(\d+)', text, re.M)))
        assert completed == (1 if args.single_tg_perf else grid), (completed, grid)
        model = configparser.RawConfigParser(strict=False)
        model.read('m5out/config.ini')
        period = int(model.get('system.shaders.clk_domain', 'clock'))
        assert period == environment['clock_period_ps']['shader']
        cycles = float(values['system.gcSimCycles'])
        return {'dispatch_cycles': cycles, 'kernel_us': cycles*period/1e6,
                'wave_interval_us': (float(values['system.shaders.lastWaveEndTick']) - float(values['system.shaders.firstWaveStartTick']))/1e6,
                'completed_TGs': completed, 'stats_path': str(path)}

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
        sample = stats()
        if args.single_tg_perf:
            directory = Path(sample['stats_path']).parent
            files = [p for p in directory.glob('*.perf') if re.match(r'^\d+_', p.name)]
            assert len(files) == 1, files
            sample['perf_path'] = str(files[0])
        return sample

    def check(golden):
        actual = c.cpu().float()
        if args.single_tg_perf:
            untouched = actual.clone()
            untouched[:16, :16] = float('nan')
            assert bool(torch.isnan(untouched).all()), 'writes outside selected tile'
            actual = actual[:16, :16]
            golden = golden[:16, :16]
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
    for i in range(3):
        gen = torch.Generator().manual_seed(case['seed'] + 1000*i)
        ac = (torch.rand((m, k), generator=gen) - .5).bfloat16()
        wc = (torch.rand((n, k), generator=gen) - .5).bfloat16()
        if i == 1:
            ac.fill_(1)
            wc.fill_(1)
        a.copy_(ac)
        w.copy_(wc)
        c.fill_(float('nan'))
        golden = (ac.float() @ wc.float().T).bfloat16().float()
        sample = launch()
        sample['accuracy'] = check(golden)
        validation.append(sample)
    samples = []
    for _ in range(case['repeats']):
        sample = launch()
        sample['accuracy'] = check(golden)
        samples.append(sample)
    warm = samples[1:]
    assert warm, 'need at least two repeats'
    performance = dict(kernel_us=statistics.median(s['kernel_us'] for s in warm),
                       kernel_cycles=statistics.median(s['dispatch_cycles'] for s in warm))
    performance['kernel_tops'] = 2*m*n*k/(performance['kernel_us']*1e6) if not args.single_tg_perf else None
    result = dict(requirement=case['requirement'], status='PASS', single_tg=args.single_tg_perf,
                  performance=performance, validation=validation, samples=samples,
                  assembly_sha256=digest('kernel.amdgcn'), benchmark_sha256=digest(__file__))
    if args.single_tg_perf:
        best = min(samples[1:], key=lambda s: s['dispatch_cycles'])
        import shutil
        shutil.copyfile(best['perf_path'], 'single_tg.perf')
        result['selected'] = dict(best, sha256=digest('single_tg.perf'))
    Path('result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'performance': result['performance']}), flush=True)


if __name__ == '__main__':
    main()
