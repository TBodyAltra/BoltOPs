# Publication contains only record['artifacts']; runtime files stay outside it.
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
_COMPACT_PARSE_ONLY = False
def _compact_find_repo():
    anchors = [Path(__file__).resolve().parent, Path.cwd()]
    anchors.extend(Path(p).resolve() for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p)
    for anchor in anchors:
        for candidate in (anchor, *anchor.parents):
            if (candidate/'boltops').is_dir():
                return candidate
    return None
_COMPACT_FAMILY = 'small'

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


_RUNNER_SOURCE = Path(__file__).read_bytes()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(output, record, variant, repeats, perf, repo, automatic=False):
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    (output/'runner.py').write_bytes(_RUNNER_SOURCE)
    if variant.get('module'):
        from importlib.util import find_spec
        spec = find_spec(variant['module'])
        if spec and spec.origin:
            (output/'kernel_source.py').write_bytes(Path(spec.origin).read_bytes())
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
    original_input_strides = [list(a.stride()), list(w.stride())]
    family = variant['family']
    packed = family.startswith('packed_')
    if packed:
        family = family.removeprefix('packed_')
        from small_pack import pack_tt_to_tn
        original_w = w
        w = torch.empty((n, k), dtype=torch.bfloat16, device=a.device)
        tn = True
    config = {key: tuple(value) if isinstance(value, list) else value
              for key, value in variant.get('config', {}).items()}
    resources = None
    grid = None
    single_tg = perf and family != 'split' and not packed
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
                kernels = [('boltops.gemm.gluon.gemm_a16w16', '_gemm_a16w16_tt_kernel', 'kernel')]
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
        _compact_prepare_handle(handle, record, variant['symbol'], Path(__file__).resolve().parent, variant.get('_compact_archived', False))
        handle._init_handles()
        resources = dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared)
        (output / 'kernel.amdgcn').write_text(handle.asm['amdgcn'])
        (output / 'kernel.hsaco').write_bytes(handle.asm['hsaco'])
        assert not handle.n_spills, resources

        def launch():
            assert kernel[(grid,)](*args, **config) is handle

    if packed:
        compute = launch
        def launch():
            pack_tt_to_tn[(triton.cdiv(n,128), triton.cdiv(k,128))](original_w, w, N=n, K=k, num_warps=8)
            compute()
    if os.environ.get('OPTALL_COMPILE_ONLY') == '1':
        (output/'compile.json').write_text(json.dumps(dict(status='COMPILED', resources=resources, variant=variant, shape=record['shape'], layout=record['layout'], hsaco_sha256=digest(output/'kernel.hsaco'), source_sha256=digest(output/'kernel_source.py'), runtime=dict(torch=torch.__version__, triton=triton.__version__,compiler={key:digest(path) for key,path in dict(libtriton=libtriton.__file__,clang=os.environ['TRITON_HIP_CLANG_PATH']).items()}), compiler_environment={key:os.environ.get(key) for key in ('GPU_CHIP','GPU_DFLAGS','GPU_ARGS','AMDGCN_USE_BUFFER_OPS','TRITON_HIP_CLANG_PATH','TRITON_MLS_B_INTERLEAVE','TRITON_BUFFER_CACHE_SWIZZLE','HSA_TOOLS_LIB')}), indent=2)+'\n')
        return
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
        assert len(paths) == 1 or family in ('public', 'split') or packed, paths
        dispatches = []
        for path in paths:
            stats = path.read_text()
            assert 'End Simulation Statistics' in stats
            groups = re.findall(r'^(system\d*\.shaders\d+)\.CUs\d+\.completedWGs\s+(\d+)', stats, re.M)
            per_die = {die: sum(int(tgs) for key, tgs in groups if key == die)
                       for die in sorted({key for key, _ in groups})}
            assert len(groups) == 96 and len(per_die) == 2, groups
            if grid is not None and not packed:
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
        guard_ok = bool((host[:guard] == 123).all() and (host[-guard:] == 123).all())
        if not guard_ok:
            torch.save(host, output/'failed_raw_output.pt')
            (output/'memory_failure.json').write_text(json.dumps(dict(prefix_corrupt=int((host[:guard]!=123).sum()),suffix_corrupt=int((host[-guard:]!=123).sum()),guard_elements=guard), indent=2)+'\n')
        assert guard_ok, 'Output canary was modified'
        assert torch.equal(ar.cpu(), ai) and torch.equal(wr.cpu(), wi)
        observed = host[guard:-guard].view(m, n)
        if not perf:
            torch.save(observed.clone(), output / 'output.pt')
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
                  strides=original_input_strides, compute_strides=[list(a.stride()), list(w.stride())], variant=variant,
                  resources=resources, samples=samples, actual_effective_protocol=dict(primary='cold full-grid',primary_warmup=0,primary_repeats=1,total_launches=repeats,all_required_dispatches_included=True,secondary_warm_median=(performance if repeats>1 else None)), kernel_us=performance,
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
    # Claim optional follow-up experiments between queued trials, preserving the two-simulator limit.
    preflight = Path(__file__).with_name('small_preflight.json')
    if not os.environ.get('OPTALL_PREFLIGHT') and preflight.exists():
        import subprocess
        claimed = preflight.with_name('small_preflight.claimed.json')
        try:
            preflight.rename(claimed)
        except FileNotFoundError:
            pass
        else:
            for job in json.loads(claimed.read_text()):
                destination = Path(job['output'])
                command = [sys.executable, __file__, '--input', job['input'], '--variant', job['variant'], '--output', job['output'], '--repeats', '1']
                destination.with_suffix('.command.json').write_text(json.dumps(command, indent=2)+'\n')
                environment = dict(os.environ, OPTALL_PREFLIGHT='1')
                with destination.with_suffix('.log').open('w') as log:
                    process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=64800)
                destination.with_suffix('.execution.json').write_text(json.dumps(dict(exit_code=process.returncode, source='inter-trial followup', command=command), indent=2)+'\n')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path(__file__).with_name('record.json'))
    p.add_argument('--variant', default='winner')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repo', type=Path, default=next(
        (p for p in Path(__file__).resolve().parents if (p/'boltops/gemm/config.py').is_file()), None))
    p.add_argument('--repeats', default=1, type=int)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--auto', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--perf', action='store_true')
    p.add_argument('--archived', action='store_true')

    args = p.parse_args()
    if _COMPACT_PARSE_ONLY: return args
    assert args.repeats > 0
    output = args.output.resolve()
    record = json.loads(args.input.read_text())
    args.repeats = record.get('experiment', {}).get('repeats', args.repeats)
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
    os.environ.update(GPU_CHIP='sbx2', GPU_OUT='model', GPU_DFLAGS=flags,
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




_measurement_main = main

"""Appended to each measured runner: isolate work and publish its exact manifest."""
import copy as _compact_copy
import gzip as _compact_gzip
import shutil as _compact_shutil
import struct as _compact_struct
import subprocess as _compact_subprocess
import tempfile as _compact_tempfile
import uuid as _compact_uuid
import zlib as _compact_zlib


def _compact_sha(data):
    return hashlib.sha256(data).hexdigest()


def _compact_sections(data):
    assert data[:5] == b'\x7fELF\x02'
    offset, = _compact_struct.unpack_from('<Q', data, 40)
    size, count, names = _compact_struct.unpack_from('<HHH', data, 58)
    entries = [_compact_struct.unpack_from('<IIQQQQIIQQ', data, offset+i*size) for i in range(count)]
    strings = data[entries[names][4]:entries[names][4]+entries[names][5]]
    result = {}
    for entry in entries:
        name = strings[entry[0]:].split(b'\0', 1)[0].decode()
        if name in ('.text', '.rodata'):
            raw = data[entry[4]:entry[4]+entry[5]]
            result[name] = dict(size=len(raw), sha256=_compact_sha(raw))
    assert '.text' in result
    return result


def _compact_manifest(record):
    names = record['artifacts']
    assert isinstance(names, list) and len(names) == len(set(names)), names
    assert all(isinstance(name, str) and name not in ('.', '..') and Path(name).name == name for name in names), names
    assert {'benchmark.py', 'kernel.py', 'kernel.amdgcn', 'record.json'} <= set(names)
    return names


def _compact_read_record(path):
    record = json.loads(path.read_text())
    _compact_manifest(record)
    return record


def _compact_write(output, record, source, replacements=None):
    """Stage all bytes before touching output; never publish runtime intermediates."""
    regenerate = replacements is None
    replacements = replacements or {}
    names = _compact_manifest(record)
    assert not output.exists(), 'output directory must be new: ' + str(output)
    if regenerate:
        expected_hashes = record['artifact_sha256']
        assert set(expected_hashes) == set(names)-{'record.json'}, 'incomplete artifact checksum manifest'
        for name, expected in expected_hashes.items():
            assert _compact_sha((source/name).read_bytes()) == expected, 'input artifact checksum mismatch: ' + name
        _compact_assert_bundle(record, (source/'kernel.amdgcn').read_bytes())
    payload = {}
    for name in names:
        if name == 'record.json':
            continue
        path = Path(__file__) if name == 'benchmark.py' else source/name
        payload[name] = replacements[name] if name in replacements else path.read_bytes()
    record = _compact_copy.deepcopy(record)
    hashes = {name: _compact_sha(data) for name, data in payload.items()}
    record['artifact_sha256'] = hashes
    record['source']['sha256'] = hashes['kernel.py']
    record['assembly']['sha256'] = hashes['kernel.amdgcn']
    record['benchmark']['sha256'] = hashes['benchmark.py']
    for item in record.get('evidence', {}).values():
        if isinstance(item, dict) and item.get('path') in hashes:
            item['sha256'] = hashes[item['path']]
    def refresh_trace_hashes(value):
        if isinstance(value, dict):
            path = value.get('path')
            if path in hashes and (path.endswith('.perf') or path == 'perf_stats.txt.gz'):
                value['sha256'] = hashes[path]
            for child in value.values():
                refresh_trace_hashes(child)
        elif isinstance(value, list):
            for child in value:
                refresh_trace_hashes(child)
    # Embedded historical records keep their original same-named asset hashes.
    refresh_trace_hashes(record.get('single_tg_perf'))
    refresh_trace_hashes(record.get('auxiliary_single_tg_perf'))
    _compact_assert_bundle(record, payload['kernel.amdgcn'])
    payload['record.json'] = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with _compact_tempfile.TemporaryDirectory(prefix='compact-publish-', dir=output.parent) as temporary:
        stage = Path(temporary)/'artifacts'
        stage.mkdir()
        for name, data in payload.items():
            (stage/name).write_bytes(data)
        assert set(p.name for p in stage.iterdir()) == set(names)
        stage.rename(output)
    print(json.dumps(dict(status='PASS', output=str(output), artifacts=names, count=len(names)), ensure_ascii=False))


def _compact_units(record):
    units = record['assembly'].get('compilation_units')
    if units:
        return units
    return [dict(stage='gemm', symbol=record['source']['symbol'],
                 executable_sections=record['assembly'].get('executable_sections'))]


def _compact_assert_bundle(record, bundle):
    for unit in record['assembly'].get('compilation_units', []):
        offset, size = unit['offset_bytes'], unit['size_bytes']
        assert offset >= 0 and size > 0 and offset+size <= len(bundle), unit
        assert _compact_sha(bundle[offset:offset+size]) == unit['sha256'], unit['stage']


def _compact_perf_binaries(data):
    assert data[:8] == b'\xe4HYGPERF'
    assert _compact_struct.unpack_from('<2Q', data, 8) == (0, 24)
    lengths = {6: 48, 7: 72, 8: 56, 9: 160, 10: 64, 11: 80, 13: 56, 15: 72}
    compressed = {6: (24, 32, 40), 8: (32, 40, 48), 11: (48, 56, 64)}
    position, objects = 24, []
    while position < len(data):
        kind, offset, length = _compact_struct.unpack_from('<3Q', data, position)
        assert offset == position-24 and length == lengths[kind]
        end = position+length
        if kind in compressed:
            pointer_at, raw_at, size_at = compressed[kind]
            assert _compact_struct.unpack_from('<Q', data, position+pointer_at)[0] == offset+length
            raw_size = _compact_struct.unpack_from('<Q', data, position+raw_at)[0]
            packed_size = _compact_struct.unpack_from('<Q', data, position+size_at)[0]
            if kind == 8:
                binary = _compact_zlib.decompress(data[end:end+packed_size])
                assert len(binary) == raw_size
                objects.append(binary)
            end += packed_size
        assert position < end <= len(data)
        position = end
    assert position == len(data) and objects, 'PERF contains no embedded code object'
    return objects


def _compact_prepare_handle(handle, record, symbol, source, archived=False):
    """Validate before dispatch; archived replay uses the measured PERF's code object."""
    unit = next(unit for unit in _compact_units(record) if unit['symbol'] == symbol)
    expected = {key: value['sha256'] if isinstance(value, dict) else value
                for key, value in unit['executable_sections'].items()}
    if archived:
        found = None
        for name in _compact_manifest(record):
            if not name.endswith('.perf'):
                continue
            trace = (source/name).read_bytes()
            assert _compact_sha(trace) == record['artifact_sha256'][name], name
            matches = {}
            for binary in _compact_perf_binaries(trace):
                sections = {key: value['sha256'] for key, value in _compact_sections(binary).items()}
                if sections == expected:
                    matches[_compact_sha(binary)] = binary
            if unit.get('hsaco_sha256') in matches:
                found = matches[unit['hsaco_sha256']]
                break
            if matches:
                assert len(matches) == 1, 'PERF has ambiguous code objects for '+symbol
                found = next(iter(matches.values()))
                break
        assert found is not None, 'no archived PERF matches compilation unit: ' + symbol
        assert handle.module is None, 'archived binding must precede code-object loading'
        bundle = (source/'kernel.amdgcn').read_bytes()
        _compact_assert_bundle(record, bundle)
        assembly = bundle[unit['offset_bytes']:unit['offset_bytes']+unit['size_bytes']]
        handle.kernel = found
        handle.asm['hsaco'] = found
        handle.asm['amdgcn'] = assembly.decode()
    actual = {key: value['sha256'] for key, value in _compact_sections(handle.asm['hsaco']).items()}
    assert actual == expected, (
        'fresh compilation differs from measured machine code; use --archived to replay the PERF-embedded code object', symbol)
    handle._init_handles()
    resources = record.get('resources', {})
    if unit['stage'] in ('reduce', 'reduction'):
        resources = resources.get('reduction', resources)
    elif unit['stage'] in ('pack', 'packing'):
        resources = resources.get('packing', resources)
    actual_resources = dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared)
    for key, expected_value in resources.items():
        if key in actual_resources:
            assert actual_resources[key] == expected_value, (symbol, key, actual_resources, resources)
    Path(symbol+'.amdgcn').write_text(handle.asm['amdgcn'])
    Path(symbol+'.hsaco').write_bytes(handle.asm['hsaco'])
    return handle


def _compact_compile_outputs(work, record):
    """Bind every compiled dispatch to the machine code represented by its PERF."""
    compiled = []
    for unit in _compact_units(record):
        symbol = unit['symbol']
        candidates = []
        for path in work.rglob('*.amdgcn'):
            binary = path.with_suffix('.hsaco')
            if binary.is_file() and re.search(r'(?m)^\s*\.globl\s+'+re.escape(symbol)+r'\b', path.read_text()):
                candidates.append((path, binary))
        assert candidates, 'compiled dispatch is missing: ' + symbol
        path, binary = sorted(candidates, key=lambda pair: (len(pair[0].parts), str(pair[0])))[0]
        actual = _compact_sections(binary.read_bytes())
        expected = unit.get('executable_sections')
        assert expected, 'record lacks executable-section evidence for ' + symbol
        expected_hashes = {key: value['sha256'] if isinstance(value, dict) else value for key, value in expected.items()}
        assert {key: value['sha256'] for key, value in actual.items()} == expected_hashes, (
            'compiled dispatch differs from archived PERF; recapture PERF before publication', symbol, actual, expected)
        compiled.append((unit, path.read_bytes(), binary.read_bytes(), actual))
    return compiled


def _compact_assembly(record, compiled):
    bundle = b''
    units = []
    for old, assembly, binary, sections in compiled:
        if len(compiled) > 1:
            bundle += ('// Compilation unit: '+old['stage']+' / '+old['symbol']+'\n').encode()
        unit = _compact_copy.deepcopy(old)
        unit.update(offset_bytes=len(bundle), size_bytes=len(assembly), sha256=_compact_sha(assembly),
                    hsaco_sha256=_compact_sha(binary), executable_sections=sections)
        bundle += assembly
        if not bundle.endswith(b'\n'):
            bundle += b'\n'
        units.append(unit)
    record['assembly']['compilation_units'] = units
    record['assembly']['sha256'] = _compact_sha(bundle)
    def refresh_unit_references(value):
        if isinstance(value, dict):
            if value.get('path') == 'kernel.amdgcn' and value is not record['assembly']:
                section = value.get('section', value.get('stage', value.get('symbol')))
                assert section is not None, 'assembly reference needs a compilation-unit section'
                unit = units[section] if isinstance(section, int) else next(
                    (u for u in units if section in (u['stage'], u['symbol'])), None)
                assert unit is not None, section
                value.update(sha256=unit['sha256'], offset_bytes=unit['offset_bytes'],
                             size_bytes=unit['size_bytes'], hsaco_sha256=unit['hsaco_sha256'],
                             executable_sections=unit['executable_sections'])
            for child in value.values():
                refresh_unit_references(child)
        elif isinstance(value, list):
            for child in value:
                refresh_unit_references(child)
    refresh_unit_references(record['assembly'])
    for key in ('packing_assembly', 'auxiliary_assembly', 'auxiliary'):
        refresh_unit_references(record.get(key))
    return bundle


def _compact_raw_stats(work, sample, stage, index):
    name = stage.get('stats', stage.get('stats_path'))
    path = work/name if name else work/f'stats-{index}-{stage["_stage"]}.txt.gz'
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = _compact_gzip.decompress(raw)
    assert b'End Simulation Statistics' in raw
    expected = stage.get('stats_sha256', stage.get('sha256'))
    assert expected is None or _compact_sha(raw) == expected
    return raw


def _compact_samples(work, samples, filename):
    blocks = []
    converted = []
    offset = 0
    for index, original in enumerate(samples):
        sample = _compact_copy.deepcopy(original)
        sample.pop('perf', None)
        sample.pop('perf_path', None)
        stage_value = sample.get('stages', sample.get('dispatches'))
        assert stage_value, sample
        items = list(stage_value.items()) if isinstance(stage_value, dict) else [(None, item) for item in stage_value]
        total = 0
        for label, stage in items:
            if label is not None:
                stage['_stage'] = label
            raw = _compact_raw_stats(work, sample, stage, index)
            stage.pop('_stage', None)
            cycles = re.findall(rb'^system.gcSimCycles\s+(\d+)', raw, re.M)
            assert len(cycles) == 1
            total += int(cycles[0])
            stage.update(stats_path=filename, stats_section=len(blocks), stats_sha256=_compact_sha(raw),
                         stats_offset_bytes=offset, stats_size_bytes=len(raw))
            if 'stats' in stage:
                stage['stats'] = filename
            stage.pop('raw_stats_path', None)
            blocks.append(raw)
            offset += len(raw)
        sample.update(index=index, status='PASS', kernel_cycles=total, kernel_us=total*455/1e6)
        sample.setdefault('phase', 'cold' if len(samples) == 1 else 'warmup' if index == 0 else 'timed')
        if len(items) == 1:
            stage = items[0][1]
            sample.update(stats_path=filename, stats_section=stage['stats_section'], stats_sha256=stage['stats_sha256'],
                          stats_offset_bytes=stage['stats_offset_bytes'], stats_size_bytes=stage['stats_size_bytes'])
        converted.append(sample)
    return converted, _compact_gzip.compress(b''.join(blocks), mtime=0)


def _compact_refresh_measurement(work, record, args):
    path = work/'result.json'
    assert path.is_file(), 'measurement did not produce a validated result.json'
    measured = json.loads(path.read_text())
    assert measured['status'] == 'PASS', measured.get('status')
    compiled = _compact_compile_outputs(work, record)
    replacements = {'kernel.amdgcn': _compact_assembly(record, compiled)}
    is_perf = bool(getattr(args, 'perf', False))
    samples, packed = _compact_samples(work, measured['samples'], 'perf_stats.txt.gz' if is_perf else 'stats.txt.gz')
    if is_perf:
        assert len(samples) >= 2
        selected = min(samples[1:], key=lambda item: item['kernel_cycles'])
        perf_name = measured['samples'][selected['index']]['perf']
        trace = (work/perf_name).read_bytes()
        assert trace[1:8] == b'HYGPERF'
        replacements.update({'single_tg.perf': trace, 'perf_stats.txt.gz': packed})
        capture = record.setdefault('single_tg_perf', {})
        capture.update(path='single_tg.perf', sha256=_compact_sha(trace), grid=[1], samples=samples,
                       samples_validated=len(samples), repeat=selected['index'],
                       kernel_cycles=selected['kernel_cycles'], kernel_us=selected['kernel_us'],
                       selection='minimum cycles of repeats 1 and 2; repeat 0 excluded',
                       full_and_single_text_rodata_equal=True,
                       validation=dict(elements=selected.get('validated_elements'),
                           outside_tolerance=selected.get('mismatches'), max_abs=selected.get('max_abs'),
                           canary_ok=True, inputs_byte_identical=True, unselected_output_nan=True),
                       hsaco_sha256=_compact_sha(compiled[0][2]),
                       stats=dict(path='perf_stats.txt.gz', sha256=_compact_sha(packed), dispatches=len(samples)))
        selected.update(perf_path='single_tg.perf', perf_sha256=_compact_sha(trace))
    else:
        record['samples'] = samples
        replacements['stats.txt.gz'] = packed
        selected = samples if len(samples) == 1 else samples[1:]
        us = statistics.median(item['kernel_us'] for item in selected)
        cycles = statistics.median(item['kernel_cycles'] for item in selected)
        performance = record['performance']
        performance.update(kernel_us=us, kernel_cycles=cycles, samples=len(selected),
                           kernel_tops=2*__import__('math').prod(record['shape'])/(us*1e6))
        performance['aggregation'] = 'one cold full-grid sample' if len(samples) == 1 else 'median of timed repeats after repeat 0 warmup'
        record['evidence']['stats']['dispatches'] = sum(len(s.get('stages', s.get('dispatches'))) for s in samples)
        if 'mmac_util_percent' in performance:
            performance['mmac_util_percent'] = performance['kernel_tops']/(96*4096*(1000/455)/1000)*100
        baseline = record['comparison'].get('baseline_kernel_us', record.get('baseline_record', {}).get('kernel_us'))
        if baseline:
            record['comparison'].update(speedup=baseline/us, latency_change_percent=(us/baseline-1)*100)
        for candidate in (work/'model/config.json', work/'g/config.json'):
            if candidate.is_file():
                config = candidate.read_bytes()
                replacements['pmd-config.json.gz'] = _compact_gzip.compress(config, mtime=0)
                model_config = record.get('evidence', {}).get('model_config', {})
                if 'raw_sha256' in model_config:
                    model_config['raw_sha256'] = _compact_sha(config)
                break
        else:
            raise AssertionError('measurement model config is missing')
        # All guards/oracles run in the original runner before status PASS.
        record['validation'] = [dict(index=s['index'], status='PASS',
                                     accuracy=s.get('validation', s.get('accuracy', dict(
                                         elements=s.get('validated_elements'), mismatches=s.get('mismatches'),
                                         max_abs=s.get('max_abs'), inputs_byte_identical=True, canary_ok=True))),
                                     baseline_comparison=s.get('baseline_comparison')) for s in samples]
    record['benchmark']['last_run'] = dict(status='PASS', source='current local kernel.py',
        timing='single-TG capture' if is_perf else 'full-grid measured with original runner',
        compiled_dispatches_verified=True, archived_code_objects=getattr(args, 'archived', False))
    return replacements


def _compact_main():
    global _COMPACT_PARSE_ONLY
    argv = sys.argv[1:]
    if '--regenerate' in argv:
        p = argparse.ArgumentParser(description='Republish the exact artifact manifest without executing PMD.')
        p.add_argument('--regenerate', type=Path, required=True)
        p.add_argument('--record', type=Path, default=Path(__file__).with_name('record.json'))
        args = p.parse_args()
        record_path = args.record.resolve()
        _compact_write(args.regenerate.resolve(), _compact_read_record(record_path), record_path.parent)
        return
    _COMPACT_PARSE_ONLY = True
    args = _measurement_main()
    _COMPACT_PARSE_ONLY = False
    if _COMPACT_FAMILY == 'small':
        record_path = args.input.resolve()
        output = args.output.resolve()
    elif _COMPACT_FAMILY == 'compute':
        record_path = args.job.resolve()
        output = (args.out or Path(_compact_tempfile.gettempdir()) /
                  ('boltops-gemm-replay-'+_compact_uuid.uuid4().hex)).resolve()
    else:
        record_path = args.record.resolve()
        output = args.out.resolve()
    record = _compact_read_record(record_path)
    if getattr(args, 'dry_run', False):
        print(json.dumps(dict(requirement=record['requirement'], shape=record['shape'],
            launch=record['launch'], artifacts=_compact_manifest(record), output=str(output)), indent=2))
        return
    for flag in ('auto', 'probe_tail', 'pack_only', 'stream', 'compile_only'):
        assert not getattr(args, flag, False), flag+' is diagnostic and cannot publish full measured artifacts'
    if _COMPACT_FAMILY != 'small':
        assert REPO is not None, 'run from the BoltOPs repository or include its path in PYTHONPATH'
    assert not output.exists(), 'output directory must be new'
    source = Path(__file__).resolve().parent
    if getattr(args, 'archived', False):
        assert _compact_sha((source/'kernel.py').read_bytes()) == record['source']['sha256'], (
            'archived replay requires the recorded local kernel source')
    if _COMPACT_FAMILY == 'axes':
        assert args.source.read_bytes() == (source/'kernel.py').read_bytes(), (
            'compact publication requires the local kernel source; capture a new record for other sources')
    with _compact_tempfile.TemporaryDirectory(prefix='boltops-measure-') as temporary:
        temporary = Path(temporary)
        raw = temporary/'measurement'
        runtime_record = temporary/'record.json'
        current = _compact_copy.deepcopy(record)
        if _COMPACT_FAMILY == 'small':
            for variant in current['variants'].values():
                if variant.get('module'):
                    variant['module'] = 'kernel'
                variant['_compact_archived'] = args.archived
            if args.seed is not None:
                current['seed'] = args.seed
            runtime_record.write_text(json.dumps(current))
            repeats = args.repeats
            if args.perf:
                repeats = 3 if '--repeats' not in argv else repeats
                assert repeats == 3, 'single-TG PERF requires exactly three launches'
            command = ['--input', str(runtime_record), '--output', str(raw), '--variant', args.variant,
                       '--repeats', str(repeats), '--repo', str(args.repo.resolve())]
            if args.seed is not None:
                command += ['--seed', str(args.seed)]
            if args.perf:
                command += ['--perf']
            if args.archived:
                command += ['--archived']
        elif _COMPACT_FAMILY == 'compute':
            job = _compact_copy.deepcopy(current['benchmark']['job'])
            job['source'] = str(source/'kernel.py')
            job['_compact_record'] = current
            job['_compact_source_dir'] = str(source)
            job['_compact_archived'] = args.archived
            runtime_record.write_text(json.dumps(job))
            command = [str(runtime_record), '--out', str(raw)]
            if args.baseline_dir:
                command += ['--baseline-dir', str(args.baseline_dir.resolve())]
        else:
            runtime_record.write_text(json.dumps(current))
            command = [str(raw), '--record', str(runtime_record), '--source', str(args.source),
                       '--layout', args.layout, '--dispatches', str(args.dispatches),
                       '--pack-k', str(args.pack_k), '--pack-n', str(args.pack_n), '--options', args.options]
            if args.baseline_out:
                command += ['--baseline-out', str(args.baseline_out)]
            if args.archived:
                command += ['--archived']
        env = dict(os.environ, BOLTOPS_COMPACT_RAW='1', PYTHONDONTWRITEBYTECODE='1')
        process = _compact_subprocess.run([sys.executable, str(Path(__file__).resolve()), *command], env=env)
        if process.returncode:
            log = raw/'worker.log'
            if log.is_file():
                print(log.read_text()[-20000:], file=sys.stderr)
            raise SystemExit(process.returncode)
        replacements = _compact_refresh_measurement(raw, current, args)
        _compact_write(output, current, source, replacements)


if __name__ == '__main__':
    if os.environ.get('BOLTOPS_COMPACT_RAW') == '1':
        _measurement_main()
    else:
        _compact_main()
