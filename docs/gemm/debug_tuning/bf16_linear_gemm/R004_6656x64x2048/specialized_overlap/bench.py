#!/usr/bin/env python3
"""Standalone R004 overlap GEMM: 6656x64x2048, TT BF16, PMD gfx946.

Only kernel.py and best_config.json are needed at runtime. --check-only uses
only the standard library, launches no simulator and creates no files. Historical
results are explicitly separate from results of new package executions.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SHAPE = (6656, 64, 2048)
SOURCE_SHA256 = 'be5ab01dda606e8f611c23bf7b1823e9c42a309c4470251073251eabe2d0f915'
HISTORY = HERE / 'runs/verification_v2final_recorded/summary.json'
DEFAULT_PATH_SH = '/public/home/duyubo/triton_all_version/triton_3.6/triton/path.sh'
EXPECTED_CONFIG = {
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
    'NUM_PROGRAMS': 144, 'SK_SPLIT': 16, 'SK_FIRST': False,
    'USE_MLS': True, 'WARP_SPECIALIZE': True, 'num_warps': 4, 'num_stages': 2,
    'mmac_layout_force': -1, 'enable_v_mmac_cluster': 2,
    'hcu_use_gfx946_sched_model_v2': True, 'misched_regpressure': True,
    'wasp_wdra': False, 'wasp_partition_warps': [1, 4], 'empty_arrive_after_mmac': True,
}
CONSTEXPR_KEYS = (
    'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'SK_SPLIT', 'SK_FIRST',
    'WARP_SPECIALIZE', 'USE_MLS', 'LOOP_UNROLL_FACTOR', 'ATOMIC_SEM', 'PARTIAL_FP32',
)
SCOPE = ('PMD gfx946 kernel-only: gcSimCycles / (gcFrequency_GHz * 1000); '
         'excludes compilation, allocation, copies and CPU validation')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), label)


def check_row_ownership(config):
    """Prove that every output row has one owner and every MLS tile is in bounds."""
    m, n, k = SHAPE
    bm, bn, bk = (config[x] for x in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K'))
    align, programs = config['SK_SPLIT'], config['NUM_PROGRAMS']
    require(m % align == n % bn == k % bk == 0, 'Unaligned specialized shape')
    require(programs % (n // bn) == 0, 'Invalid program grid')
    p = programs // (n // bn)
    ownership = [0] * m
    sizes = []
    for pr in range(p):
        r0 = (pr * (m // align) // p) * align
        r1 = ((pr + 1) * (m // align) // p) * align
        om = min(r0, m - bm)
        require(0 <= om <= r0 < r1 <= om + bm <= m, 'Out-of-bounds MLS row tile')
        sizes.append(r1 - r0)
        for row in range(r0, r1):
            ownership[row] += 1
    require(all(count == 1 for count in ownership), 'Output row ownership has holes/overlap')
    return dict(rows_checked=m, output_elements=m*n, grid=[programs],
                row_ownership='exactly_once', owned_rows_per_program=sorted(set(sizes)),
                matrix_loads_in_bounds=True)


def stats_of(path):
    text = Path(path).read_text()
    require(text.count('Begin Simulation Statistics') == 1, f'Non-unique stats block: {path}')

    def number(key):
        matches = re.findall(r'^' + re.escape(key) + r'\s+([\deE.+-]+)', text, re.M)
        require(len(matches) == 1, f'Missing/duplicate stats field {key}: {path}')
        return float(matches[0])

    cycles, freq = number('system.gcSimCycles'), number('gcFrequency')
    require(math.isfinite(cycles) and cycles > 0 and math.isfinite(freq) and freq > 0,
            f'Invalid PMD timing: {path}')
    wgs = re.findall(r'^system\.shaders\.(CUs\d+)\.completedWGs\s+(\d+)', text, re.M)
    require(wgs and len({cu for cu, _ in wgs}) == len(wgs), f'Invalid CU statistics: {path}')
    return dict(cycles=cycles, freq_ghz=freq, us=cycles/(freq*1000),
                completed_wgs=sum(int(value) for _, value in wgs), cu_count=len(wgs),
                active_cus=sum(int(value) > 0 for _, value in wgs),
                stats_path=str(Path(path).resolve()))


def check_history(config):
    history = read_json(HISTORY)
    require(history['record_type'] == 'recorded_existing_results'
            and history['status'] == 'RECORDED'
            and history['results_were_executed_by_this_package'] is False,
            'History must be labelled as existing results, never as a fresh package run')
    require(history['source_sha256'] == SOURCE_SHA256 and history['config'] == config,
            'Historical source/config mismatch')
    require(history['shape'] == list(SHAPE), 'Historical shape mismatch')
    runs = {row['name']: row for row in history['recorded_runs']}
    require(len(runs) == 6, 'Expected six archived source runs')
    for row in runs.values():
        require(row['status'] == row['accuracy']['status'] == 'PASS', 'Historical accuracy failed')
        require(row['accuracy']['sampled_points'] == SHAPE[0]*SHAPE[1]
                and row['accuracy']['mismatches'] == 0, 'History is not a full-output check')
        close(statistics.median(row['timed_cycles']), row['median_cycles'], row['name'])
        close(row['median_cycles'] / (row['freq_ghz'] * 1000), row['median_us'], row['name'])
    samples = [cy for name in history['primary_run_names'] for cy in runs[name]['timed_cycles']]
    times = [cy/(runs[name]['freq_ghz']*1000) for name in history['primary_run_names']
             for cy in runs[name]['timed_cycles']]
    require(samples == history['timed_cycles'] and len(samples) == history['samples'] == 4,
            'Historical pooled samples mismatch')
    for name, value in (('median_cycles', statistics.median(samples)),
                        ('median_us', statistics.median(times)),
                        ('min_us', min(times)), ('max_us', max(times))):
        close(history[name], value, 'Historical ' + name)
    require(history['round_medians_us'] == [runs[name]['median_us']
            for name in history['primary_run_names']], 'Historical round medians mismatch')
    return history


def verify_source(source, config, history):
    """Optional source audit; normal check-only and real runs never read the old tree."""
    source = Path(source).resolve()
    require((HERE/'kernel.py').read_bytes() == (source/'best_kernel.py').read_bytes(),
            'kernel.py is not a byte-for-byte copy of best_kernel.py')
    suites = {name: {row['name']: row for row in read_json(source/name)}
              for name in history['provenance']['config_suites']}
    for name in ('final_overlap_m64_k64_seed29', 'final_overlap_m64_k64_seed143'):
        require(suites['final_cases.json'][name]['config'] == config, name + ' suite config')
    for name in history['primary_run_names']:
        require(suites['v2final_cases.json'][name]['config'] == config, name + ' suite config')
    raw_dispatches = 0
    result_hashes = {}
    for archived in history['recorded_runs']:
        result_path = source / archived['source_result_path']
        original = read_json(result_path)
        case = original['case']
        require(original['name'] == archived['name'], 'Source run name mismatch')
        require(original['status'] == archived['status'] and original['returncode'] == 0,
                'Source run was not successful')
        require([case[x] for x in ('M', 'N', 'K')] == list(SHAPE)
                and case['layout'] == 'TT' and case['full_check'], 'Wrong source shape/check scope')
        require(case['seed'] == archived['seed'], 'Source seed mismatch')
        require(case['config'] == archived.get('config', config), 'Source configuration mismatch')
        require(sha(result_path.parent/'kernel_source.py') == case['source_sha256']
                == archived['source_sha256'], 'Source snapshot hash mismatch')
        require(case['tune_lib_sha256'] == history['provenance']['archived_tune_lib_sha256'],
                'Archived measurement helper hash mismatch')
        require(case['baseline_sha256'] == history['provenance']['baseline_archive_sha256'],
                'Archived baseline hash mismatch')
        for group in ('accuracy', 'resources'):
            require(all(original[group][key] == value for key, value in archived[group].items()),
                    archived['name'] + ': transcribed ' + group + ' mismatch')
        require(original['timed_cycles'] == archived['timed_cycles'], 'Source samples mismatch')
        close(original['median_cycles'], archived['median_cycles'], 'Source median cycles')
        close(original['median_us'], archived['median_us'], 'Source median us')
        cycles = archived['warmup_cycles'] + archived['timed_cycles']
        require([row['cycles'] for row in original['dispatches']] == cycles
                and len(cycles) == case['dispatches'], 'Source dispatch count mismatch')
        for dispatch in original['dispatches']:
            raw = stats_of(source / dispatch['path'])
            for key in ('cycles', 'freq_ghz', 'us', 'completed_wgs', 'cu_count', 'active_cus'):
                close(raw[key], dispatch[key], 'Raw PMD ' + key)
            require(raw['cu_count'] == raw['active_cus'] == 48
                    and raw['completed_wgs'] == archived['resources']['grid'][0],
                    'Source CU/workgroup count mismatch')
            close(raw['freq_ghz'], archived['freq_ghz'], 'Source frequency')
            raw_dispatches += 1
        result_hashes[archived['name']] = sha(result_path)
    archive_path = Path(history['provenance']['baseline_archive'])
    require(sha(archive_path) == history['provenance']['baseline_archive_sha256'],
            'Baseline archive has changed')
    baseline = read_json(archive_path)['best']
    for key, original_key in (('median_cycles', 'cycles'), ('median_us', 'us'), ('status', 'status')):
        require(history['original_autotune_baseline'][key] == baseline['val'][original_key],
                'Original autotune validation mismatch')
    return dict(status='PASS', source_directory=str(source), source_runs=len(result_hashes),
                raw_dispatches_checked=raw_dispatches, source_result_sha256=result_hashes)


def check_package(source=None):
    for name in ('kernel.py', 'bench.py', 'best_config.json', 'run.sh', 'README.md', '.gitignore'):
        require((HERE/name).is_file(), 'Missing package file: ' + name)
    digest = sha(HERE/'kernel.py')
    require(digest == SOURCE_SHA256, 'Local kernel differs from the archived R004 kernel')
    config = read_json(HERE/'best_config.json')
    require(config == EXPECTED_CONFIG, 'Unexpected specialized launch configuration')
    for name in ('bench.py', 'kernel.py'):
        ast.parse((HERE/name).read_text(), filename=str(HERE/name))
    history = check_history(config)
    report = dict(status='PASS', record_type='offline_package_check', simulator_executed=False,
                  source_sha256=digest, config=config, geometry=check_row_ownership(config),
                  historical_runs_checked=len(history['recorded_runs']))
    if source:
        report['source_verification'] = verify_source(source, config, history)
    return digest, config, report


def toolchain_env():
    path_sh = Path(os.environ.get('TRITON_PATH_SH', DEFAULT_PATH_SH)).resolve()
    require(path_sh.is_file(), f'Missing Triton environment script: {path_sh}')
    # Source in an empty environment so inherited tmp/repository PYTHONPATH entries
    # cannot affect the standalone package. This subprocess imports no GPU runtime.
    completed = subprocess.run(['bash', '-c', 'source "$1"; env -0', 'bash', str(path_sh)],
                               env={'PATH': '/usr/bin:/bin'}, stdout=subprocess.PIPE, check=True)
    allowed = {'PYTHONPATH', 'PMD_PATH', 'GPU_CHIP'}
    exported = {}
    for entry in completed.stdout.decode().split('\0'):
        key, sep, value = entry.partition('=')
        if sep and key in allowed:
            exported[key] = value.rstrip(':') if key == 'PYTHONPATH' else value
    require(exported.get('PYTHONPATH'), 'path.sh did not export the Triton build PYTHONPATH')
    exported.update(TRITON_HIP_CLANG_PATH=os.environ.get('OPT_CLANG', '/opt/dtk/aillvm/bin/clang-18'),
                    TRITON_HIP_LLD_PATH=os.environ.get('OPT_LLD', '/opt/dtk/aillvm/bin/ld.lld'),
                    HSA_TOOLS_LIB=os.environ.get('OPT_HSA_TOOLS_LIB',
                                               '/opt/rocm/lib/xprofiler/libperf_gen_helper.so'))
    return exported


def env_for(dest):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('TRITON_', 'GPU_', 'HIP_', 'XDG_', 'PMD_', 'CUDA_'))
           and key not in ('HSA_TOOLS_LIB', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
    env.update(toolchain_env())
    for sub in ('home', 'cache', 'temp', 'xdg-cache', 'xdg-config', 'xdg-data', 'torch-cache'):
        (dest/sub).mkdir()
    env.update(HOME=str(dest/'home'), TRITON_CACHE_DIR=str(dest/'cache'),
               TMPDIR=str(dest/'temp'), TMP=str(dest/'temp'), TEMP=str(dest/'temp'),
               XDG_CACHE_HOME=str(dest/'xdg-cache'), XDG_CONFIG_HOME=str(dest/'xdg-config'),
               XDG_DATA_HOME=str(dest/'xdg-data'), TORCHINDUCTOR_CACHE_DIR=str(dest/'torch-cache'),
               GPU_CHIP=env.get('GPU_CHIP', 'sb'), PMD_PATH=env.get('PMD_PATH', '/opt/rocm/pmd'),
               GPU_ARGS='[]', GPU_DFLAGS="['Perf']", GPU_OUT='g', AMDGCN_USE_BUFFER_OPS='1',
               TRITON_ALWAYS_COMPILE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               R004_WORKER_DIR=str(dest))
    return env


def worker(args, digest, config):
    require(os.environ.get('R004_WORKER_DIR') == str(Path.cwd().resolve()),
            'Internal worker requires a freshly isolated output directory; invoke without --worker')
    import torch
    import triton
    from triton.runtime import driver
    from triton.backends.hcu.compiler import HIPOptions
    from kernel import _gemm_bf16_hsk_kernel as kernel

    target = driver.active.get_current_target()
    require(target.arch == 'gfx946', 'This configuration targets PMD gfx946')
    torch.set_num_threads(1)
    m, n, k = SHAPE
    cfg = dict(config)
    programs = cfg.pop('NUM_PROGRAMS')
    const = {key: cfg.pop(key) for key in CONSTEXPR_KEYS if key in cfg}
    unknown = sorted(set(cfg) - set(HIPOptions.__dataclass_fields__))
    require(not unknown, f'Unknown launch options (would be silently ignored): {unknown}')
    grid = (programs,)
    gen = torch.Generator().manual_seed(args.seed)
    ac = torch.randn((m, k), generator=gen).to(torch.bfloat16)
    bc = torch.randn((k, n), generator=gen).to(torch.bfloat16)
    reference = ac.float() @ bc.float()
    gold = reference.to(torch.bfloat16)
    rounding_floor = float((gold.float() - reference).abs().mean())
    a, b = ac.cuda(), bc.cuda()
    guard, sentinel = 128, 123.0

    def guarded(shape, dtype, initial):
        host = torch.full((guard + math.prod(shape) + guard,), sentinel, dtype=dtype)
        host[guard:-guard] = initial
        storage = host.cuda()
        return host, storage, storage[guard:-guard].view(shape)

    ch, cs, out = guarded((m, n), torch.bfloat16, float('nan'))
    # Retain the historical interface. This overlap kernel never accesses these
    # buffers, has no split-K and needs no counter protocol or reduction dispatch.
    wh, ws, workspace = guarded((2*const['BLOCK_SIZE_M']*const['BLOCK_SIZE_N'],),
                                torch.bfloat16, float('nan'))
    cth, cts, counters = guarded((1,), torch.int32, 0)
    launch_args = (a, b, out, workspace, counters, m, n, k,
                   *a.stride(), *b.stride(), *out.stride(), programs)
    start = time.monotonic()
    # Match r004_opt/bench.py: NUM_PROGRAMS is grid/positional NUM_DP, never a
    # compiler option; constexpr arguments and HIPOptions are separated explicitly.
    compiled = kernel.warmup(*launch_args, grid=grid, **const, **cfg)
    compiled._init_handles()
    resources = dict(grid=list(grid), compile_wall_s=time.monotonic()-start,
                     shared=compiled.metadata.shared, n_regs=compiled.n_regs,
                     n_spills=compiled.n_spills, num_warps=compiled.metadata.num_warps)
    dump('compile.json', resources)
    require(resources['shared'] <= 131072, 'LDS exceeds 131072 bytes')
    for kind in ('amdgcn', 'ttir', 'ttgir', 'llir'):
        if kind in compiled.asm:
            Path('kernel.' + kind).write_text(compiled.asm[kind])
    Path('kernel_source.py').write_bytes((HERE/'kernel.py').read_bytes())
    dump('environment.json', dict(python=sys.version, torch=torch.__version__,
         triton=triton.__version__, triton_path=triton.__file__, target=str(target),
         source_sha256=digest, bench_sha256=sha(HERE/'bench.py'),
         config_sha256=sha(HERE/'best_config.json'), environment={key: os.environ.get(key)
         for key in ('PYTHONPATH', 'PMD_PATH', 'GPU_CHIP', 'GPU_ARGS', 'GPU_DFLAGS', 'GPU_OUT',
                     'HSA_TOOLS_LIB', 'TRITON_HIP_CLANG_PATH', 'TRITON_HIP_LLD_PATH',
                     'HOME', 'TRITON_CACHE_DIR', 'TMPDIR')}))
    checks = []
    for index in range(args.warmup + args.reps):
        cs.copy_(ch)  # Poison every output before every launch; no stale-output pass.
        torch.cuda.synchronize()
        kernel[grid](*launch_args, **const, **cfg)
        torch.cuda.synchronize()
        raw = cs.cpu()
        actual = raw[guard:-guard].view(m, n)
        finite = bool(torch.isfinite(actual).all())
        guards = bool((raw[:guard] == sentinel).all() and (raw[-guard:] == sentinel).all())
        workspace_untouched = bool(torch.allclose(ws.cpu(), wh, rtol=0, atol=0, equal_nan=True))
        counters_untouched = bool(torch.equal(cts.cpu(), cth))
        mismatches = int((~torch.isclose(actual, gold, atol=0.02, rtol=0.01)).sum())
        mean_abs = float((actual.float() - reference).abs().mean())
        ok = (finite and guards and workspace_untouched and counters_untouched
              and mismatches == 0 and mean_abs <= 1.05*rounding_floor + 1e-6)
        check = dict(dispatch=index, phase='warmup' if index < args.warmup else 'timed',
                     status='PASS' if ok else 'FAIL', finite=finite, guards=guards,
                     workspace_untouched=workspace_untouched, counters_untouched=counters_untouched,
                     checked_points=actual.numel(), mismatches=mismatches,
                     mean_abs=mean_abs if math.isfinite(mean_abs) else None,
                     rounding_floor=rounding_floor, atol=0.02, rtol=0.01)
        checks.append(check)
        dump('checks.json', checks)
        require(ok, f'Full-output validation failed for dispatch {index}; see checks.json')
    paths = sorted(Path('g').glob('*/*/stats.txt'),
                   key=lambda path: tuple(int(part) for part in path.relative_to('g').parts[:-1]))
    require(len(paths) == args.warmup + args.reps, 'Unexpected GPU dispatches (including helper kernels)')
    records = [dict(dispatch_id=index, **stats_of(path)) for index, path in enumerate(paths)]
    require(all(row['completed_wgs'] == programs and row['cu_count'] == row['active_cus'] == 48
                for row in records), 'Unexpected PMD workgroup/CU counts')
    timed = records[args.warmup:]
    result = dict(record_type='fresh_package_run', status='PASS', correctness='PASS',
                  shape=list(SHAPE), dtype='bfloat16', layout='TT', seed=args.seed,
                  config=config, source_sha256=digest, resources=resources, checks=checks,
                  every_dispatch_fully_checked=True, warmup_dispatches=records[:args.warmup],
                  timed_dispatches=timed, median_cycles=statistics.median(row['cycles'] for row in timed),
                  median_us=statistics.median(row['us'] for row in timed), scope=SCOPE)
    dump('result.json', result)
    print(json.dumps(result), flush=True)


def launch_worker(cmd, dest, env, timeout):
    with (dest/'run.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=dest, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
        require(returncode == 0, f'Worker exit {returncode}; see {dest / "run.log"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', help='NEW output directory; never overwrite an existing directory')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--timeout', type=int, default=1800, help='Worker timeout in seconds')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--verify-source', help='Optional original r004_opt directory for provenance audit')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.reps, args.timeout) < 1:
        parser.error('rounds, warmup, reps and timeout must be positive')
    digest, config, check = check_package(args.verify_source)
    if args.check_only:
        print(json.dumps(check, indent=2, sort_keys=True))
        return
    if args.worker:
        worker(args, digest, config)
        return
    out = (Path(args.out).resolve() if args.out else
           HERE/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    results = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for round_id in range(args.rounds):
            dest = out / f'round_{round_id}'
            dest.mkdir()
            env = env_for(dest)
            cmd = [sys.executable, '-B', str(HERE/'bench.py'), '--worker',
                   '--warmup', str(args.warmup), '--reps', str(args.reps), '--seed', str(args.seed)]
            dump(dest/'command.json', dict(argv=cmd, cwd=str(dest),
                 environment={key: env[key] for key in ('PYTHONPATH', 'PMD_PATH', 'GPU_CHIP',
                  'GPU_ARGS', 'GPU_DFLAGS', 'HSA_TOOLS_LIB', 'TRITON_HIP_CLANG_PATH',
                  'TRITON_HIP_LLD_PATH', 'HOME', 'TRITON_CACHE_DIR', 'TMPDIR')}))
            print(f'Round {round_id+1}/{args.rounds}: {dest}', flush=True)
            launch_worker(cmd, dest, env, args.timeout)
            result = read_json(dest/'result.json')
            require(result['status'] == 'PASS', 'Worker did not pass correctness')
            results.append(result)
            print(f"  full-output correctness=PASS, median={result['median_us']:.6f} us", flush=True)
    except BaseException as exc:
        dump(out/'summary.json', dict(record_type='fresh_package_run', status='ERROR',
             correctness='NOT_ESTABLISHED', started_at_utc=started,
             completed_rounds=len(results), error=f'{type(exc).__name__}: {exc}',
             source_sha256=digest, results_directory=str(out)))
        raise
    timed = [sample for result in results for sample in result['timed_dispatches']]
    summary = dict(record_type='fresh_package_run', status='PASS', correctness='PASS',
                   results_were_executed_by_this_package=True, started_at_utc=started,
                   finished_at_utc=datetime.now(timezone.utc).isoformat(),
                   shape=list(SHAPE), dtype='bfloat16', layout='TT', seed=args.seed, config=config,
                   source_sha256=digest, bench_sha256=sha(HERE/'bench.py'),
                   config_sha256=sha(HERE/'best_config.json'), rounds=args.rounds,
                   warmup_per_round=args.warmup, timed_per_round=args.reps, samples=len(timed),
                   timed_cycles=[sample['cycles'] for sample in timed],
                   median_cycles=statistics.median(sample['cycles'] for sample in timed),
                   median_us=statistics.median(sample['us'] for sample in timed),
                   min_us=min(sample['us'] for sample in timed), max_us=max(sample['us'] for sample in timed),
                   round_medians_us=[result['median_us'] for result in results],
                   every_dispatch_fully_checked=True, checked_points_per_dispatch=SHAPE[0]*SHAPE[1],
                   scope=SCOPE, results_directory=str(out))
    dump(out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved: {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
