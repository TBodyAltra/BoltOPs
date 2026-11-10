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
import time

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    environment_path = args.environment.resolve()
    environment = json.loads(environment_path.read_text())
    for key, value in environment['environment_variables'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
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
    (output / 'kernel.amdgcn').write_text(handle.asm['amdgcn'])
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
        assert completed == grid, (completed, grid)
        model = configparser.RawConfigParser(strict=False)
        model.read('m5out/config.ini')
        period = int(model.get('system.shaders.clk_domain', 'clock'))
        assert period == environment['clock_period_ps']['shader']
        cycles = float(values['system.gcSimCycles'])
        return {'dispatch_cycles': cycles, 'dispatch_us': cycles*period/1e6,
                'kernel_us': (float(values['system.shaders.lastWaveEndTick']) - float(values['system.shaders.firstWaveStartTick']))/1e6,
                'completed_TGs': completed, 'stats_path': str(path)}

    def launch():
        before = len(launched)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = public.gemm_a16w16(a, w, out=c, backend=case.get('backend', 'auto'))
        torch.cuda.synchronize()
        wall_us = (time.perf_counter_ns() - start)/1000
        assert result is c and len(launched) == before + 1
        record = stats()
        record['api_end_to_end_us'] = wall_us
        return record

    def check(golden):
        actual = c.cpu().float()
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
    result = {'requirement': case['requirement'], 'record_key': {'requirement': case['requirement'], 'dies': 1, 'layout': 'TN'},
              'shape': case['shape'], 'full_grid_TGs': grid, 'status': 'PASS',
              'performance': {name: statistics.median(s[name] for s in warm) for name in ('api_end_to_end_us', 'kernel_us', 'dispatch_us', 'dispatch_cycles')},
              'time_domains': {'api_end_to_end_us': 'host wall time through public API including PMD simulation; warmed JIT, preallocated output',
                              'kernel_us': 'PMD first wave start to last wave end', 'dispatch_us': 'PMD full dispatch gcSimCycles'},
              'source': {'path': 'boltops/gemm/gluon/gemm_a16w16.py', 'sha256': digest(REPO/'boltops/gemm/gluon/gemm_a16w16.py'), 'symbol': symbol},
              'config': {'path': case['config'], 'sha256': digest(config_path)},
              'assembly': {'path': 'kernel.amdgcn', 'sha256': digest('kernel.amdgcn')},
              'environment': {'path': str(environment_path), 'sha256': digest(environment_path)},
              'benchmark': {'path': 'scripts/perf/bench_a16w16_pmd.py', 'sha256': digest(__file__)},
              'resources': {'LDS_bytes': handle.metadata.shared, 'VGPR': handle.n_regs, 'spill': handle.n_spills},
              'validation': validation, 'samples': samples, 'warm_samples': len(warm)}
    Path('result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'performance': result['performance']}), flush=True)


if __name__ == '__main__':
    main()
