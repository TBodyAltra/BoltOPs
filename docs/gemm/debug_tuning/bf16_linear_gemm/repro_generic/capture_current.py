#!/usr/bin/env python3
"""Capture current shape/config IR, assembly and PMD TT/Perf without changing tuning results.

Run through capture.sh, which launches it under the PMD run.py launcher in a fresh
working directory with a private TRITON_CACHE_DIR.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
from types import SimpleNamespace

from repro import ROOT, DEFAULT_REPO, environment, sha256
import tune_lib as T


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', required=True, help='shape directory name, e.g. R015_12x2048x4096')
    parser.add_argument('--implementation', choices=('generic', 'specialized'), default='generic')
    parser.add_argument('--layout', choices=('TT', 'TN'), default='TT')
    args = parser.parse_args()
    shape_dir = (Path(__file__).resolve().parent.parent / args.shape).resolve()
    if not shape_dir.is_dir():
        parser.error(f'unknown shape directory: {shape_dir}')
    if args.implementation == 'specialized' and (args.layout != 'TT' or
            not (shape_dir / 'specialized_fused_splitk').is_dir()):
        parser.error('specialized is available only for R014 TT')
    sys.path.insert(0, str(DEFAULT_REPO))
    out = Path.cwd().resolve()
    if shape_dir == out or shape_dir in out.parents:
        parser.error('run in a scratch directory outside the shape directory; use capture.sh')
    if (out / 'g').exists() or (out / 'manifest.json').exists():
        parser.error('capture directory already contains output; use a fresh directory')
    flags = set(ast.literal_eval(os.environ.get('GPU_DFLAGS', '[]')))
    if not {'TT', 'Perf'} <= flags:
        parser.error('launch with /opt/rocm/pmd/scripts/run.py -c sb -df TT Perf -e python3 ...')
    cache = Path(os.environ['TRITON_CACHE_DIR']).resolve()
    if cache != out / 'cache':
        parser.error('TRITON_CACHE_DIR must be <working-directory>/cache')

    m, n, k = map(int, shape_dir.name.split('_', 1)[1].split('x'))
    implementation_dir = shape_dir / ('generic' if args.implementation == 'generic'
                                      else 'specialized_fused_splitk')
    config_path = implementation_dir / 'best_config.json'
    config = json.loads(config_path.read_text())
    operator = (ROOT / 'operators/gemm_a16w16_generic.py' if args.implementation == 'generic'
                else implementation_dir / 'fused_kernel.py')
    hashes = {str(p): sha256(p) for p in (config_path, operator)}
    dump('config.json', config)
    # A snapshot ties the archived IR/Perf to an exact source, even after later edits.
    shutil.copyfile(operator, out / 'kernel_source.py')

    if args.implementation == 'generic':
        if sha256(operator) != sha256(T.OP_PATH):
            raise RuntimeError('operators/ and autotune/scripts/ generic kernels differ')
        T.OP_PATH = operator
        case = dict(M=m, N=n, K=k, layout=args.layout, config=config,
                    seed=29, dispatches=6, full_check=True, keep_asm=True)
        dump('case.json', case)
        T.worker(str(out / 'case.json'))
        check = json.loads((out / 'check.json').read_text())
        if check['status'] != 'PASS':
            raise RuntimeError(f'correctness failed: {check}')
        # Only this kernel is compiled in the fresh cache. Export every IR stage
        # from that exact cache, and verify assembly against worker output.
        for kind in ('ttir', 'ttgir', 'llir', 'amdgcn', 'json', 'hsaco'):
            matches = list(cache.glob(f'*/_gemm_bf16_generic_kernel.{kind}'))
            if kind == 'hsaco' and not matches:
                continue
            if len(matches) != 1:
                raise RuntimeError(f'expected one cached {kind}, found {matches}')
            target = out / ('kernel.metadata.json' if kind == 'json' else f'kernel.{kind}')
            if kind == 'amdgcn' and target.read_bytes() != matches[0].read_bytes():
                raise RuntimeError('cache assembly differs from the launched kernel')
            shutil.copyfile(matches[0], target)
        resources = {key: check[key] for key in ('shared', 'n_regs', 'n_spills')}
        expected_wgs = T.num_tg(m, n, config)
    else:
        sys.path.insert(0, str(implementation_dir))
        spec = importlib.util.spec_from_file_location('capture_specialized_bench', implementation_dir / 'bench.py')
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        bench.worker(SimpleNamespace(warmup=1, reps=5))
        check = json.loads((out / 'result.json').read_text())
        if any(check[key] != 'PASS' for key in ('correctness', 'partials', 'guards', 'counters')):
            raise RuntimeError(f'specialized correctness failed: {check}')
        resources = dict(n_regs=check['n_regs'], metadata=check['metadata'])
        expected_wgs = T.num_tg(m, n, config)
        # This PMD backend caches assembly directly and may not emit HSACO.
        for kind in ('json', 'hsaco'):
            matches = list(cache.glob(f'*/_gemm_bf16_fused_splitk_kernel.{kind}'))
            if kind == 'hsaco' and not matches:
                continue
            if len(matches) != 1:
                raise RuntimeError(f'expected one cached {kind}, found {matches}')
            shutil.copyfile(matches[0], out / ('kernel.metadata.json' if kind == 'json' else f'kernel.{kind}'))

    stats_paths = sorted((out / 'g').glob('*/*/stats.txt'),
                         key=lambda p: tuple(map(int, p.relative_to(out / 'g').parts[:-1])))
    if len(stats_paths) != 6:
        raise RuntimeError(f'expected six GEMM dispatches, got {len(stats_paths)}')
    records = []
    for path in stats_paths:
        perf = list(path.parent.glob('stats_xcd*.perf'))
        if not perf or any(p.stat().st_size == 0 for p in perf):
            raise RuntimeError(f'missing or empty perf file at {path.parent}')
        record = T.stats_of(path)
        if record['completed_wgs'] != expected_wgs:
            raise RuntimeError(f'completed workgroups != {expected_wgs}: {path}')
        record.update(stats_path=str(path.relative_to(out)),
                      perf_paths=[str(p.relative_to(out)) for p in perf])
        records.append(record)
    for kind in ('ttir', 'ttgir', 'llir', 'amdgcn'):
        if (out / f'kernel.{kind}').stat().st_size == 0:
            raise RuntimeError(f'empty kernel.{kind}')
    if hashes != {p: sha256(Path(p)) for p in hashes}:
        raise RuntimeError('source/config changed during capture')
    timed = records[1:]
    manifest = dict(
        shape_id=shape_dir.name.split('_')[0], shape=[m, n, k], layout=args.layout,
        implementation=args.implementation, config=config, config_file=str(config_path),
        operator_file=str(operator), source_sha256=hashes, correctness='PASS',
        warmup_dispatches=1, timed_dispatches=5,
        median_cycles=statistics.median(r['cycles'] for r in timed),
        median_us=statistics.median(r['us'] for r in timed), resources=resources,
        dispatches=records, environment=json.loads((out / 'environment.json').read_text()),
        process_environment={key: os.environ.get(key) for key in (
            'PYTHONPATH', 'TRITON_HIP_CLANG_PATH', 'TRITON_HIP_LLD_PATH',
            'PMD_PATH', 'GPU_CHIP', 'GPU_DFLAGS', 'GPU_ARGS', 'HSA_TOOLS_LIB',
            'AMDGCN_USE_BUFFER_OPS', 'TRITON_CACHE_DIR')},
        command=['/opt/rocm/pmd/scripts/run.py', '-c', 'sb', '-df', 'TT', 'Perf',
                 '-e', 'python3', str(Path(__file__).resolve()), '--shape', args.shape,
                 '--implementation', args.implementation, '--layout', args.layout],
        results_directory=str(out),
        scope='PMD kernel-only, system.gcSimCycles; one process, 1 warmup + 5 timed; TT and Perf enabled')
    manifest['artifacts'] = {
        str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(out.glob('kernel.*')) if p.is_file()}
    manifest['dispatch_artifacts'] = {
        str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha256(p))
        for stats in stats_paths for p in sorted(stats.parent.iterdir()) if p.is_file()}
    dump('manifest.json', manifest)
    print(json.dumps({key: manifest[key] for key in (
        'shape', 'implementation', 'layout', 'correctness', 'median_cycles', 'median_us')}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--environment']:
        print(json.dumps(environment(DEFAULT_REPO), indent=2, sort_keys=True))
    else:
        main()
