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
_COMPACT_FAMILY = 'axes'

"""Run an isolated full-output BF16 TT PMD comparison; all layouts start from identical TT values."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

SOURCE_DIR = Path(__file__).resolve().parent
REPO = _compact_find_repo()
HERE = SOURCE_DIR
RUNNER_SOURCE = Path(__file__).read_bytes()
import configparser

import math

import struct

def executable_sections(data):
    assert data[:5] == b'\x7fELF\x02'
    offset, = struct.unpack_from('<Q', data, 40)
    size, count, names = struct.unpack_from('<HHH', data, 58)
    entries = [struct.unpack_from('<IIQQQQIIQQ', data, offset + index*size)
               for index in range(count)]
    symbols = data[entries[names][4]:entries[names][4] + entries[names][5]]
    sections = {}
    for entry in entries:
        name = symbols[entry[0]:].split(b'\x00', 1)[0].decode()
        if name in ('.text', '.rodata'):
            sections[name] = sha(data[entry[4]:entry[4]+entry[5]])
    assert '.text' in sections
    return sections

def collect_dispatch(stats_path, grid, tiles, options, k):
    raw = stats_path.read_bytes()
    content = raw.decode()
    assert 'End Simulation Statistics' in content
    groups = re.findall(r'^(system\d*\.shaders\d*)\.CUs\d+\.completedWGs\s+(\d+)', content, re.M)
    assert len(groups) == 96 and len({die for die, _ in groups}) == 2, len(groups)
    per_die = {die: sum(int(v) for d, v in groups if d == die) for die, _ in groups}
    assert sum(per_die.values()) == grid and min(per_die.values()) > 0, per_die
    mmops = sum(map(int, re.findall(r'^system\d*\.shaders\d*\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)', content, re.M)))
    block_k = options['BLOCK_SIZE_K']
    padded_k = math.ceil(k / block_k) * block_k
    expected = tiles * 2 * options['BLOCK_SIZE_M'] * options['BLOCK_SIZE_N'] * padded_k // 8192
    assert mmops == expected, (mmops, expected)
    model = configparser.RawConfigParser()
    model.read('g/config.ini')
    assert {int(model[die + '.clk_domain']['clock']) for die in per_die} == {455}
    for key in ('skip_cal', 'skip_kernel_exec', 'ipc_conn'):
        assert model['gpu_configuration'][key].lower() in ('0', 'false'), key
    cycles = re.findall(r'^system.gcSimCycles\s+(\d+)', content, re.M)
    assert len(cycles) == 1, cycles
    active = {die: sum(int(v) > 0 for d, v in groups if d == die) for die in per_die}
    return dict(cycles=int(cycles[0]), us=int(cycles[0])*455/1e6, completed_TGs=per_die,
                active_CUs=active, mmop_instructions=mmops, stats_sha256=sha(raw)), raw


M, N, K = 22736, 1152, 1536
CU, BM, BN, BK = 96, 256, 256, 64
TILES = ((M + BM - 1) // BM) * ((N + BN - 1) // BN)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    Path(name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_source(path):
    spec = importlib.util.spec_from_file_location('boltops.gemm.gluon.optall_axes_source', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(args):
    save('worker-process.json', dict(pid=os.getpid(), ppid=os.getppid(), sid=os.getsid(0), argv=sys.argv,
         lease=os.environ.get('OPTALL_PMD_LEASE'), preflight=os.environ.get('OPTALL_PREFLIGHT')))
    Path('bench.snapshot.py').write_bytes(RUNNER_SOURCE)
    original_source = args.source or SOURCE_DIR/'kernel.py'
    frozen_source = Path.cwd()/'candidate.snapshot.py'
    source_text = original_source.read_text()
    if args.probe_tail:
        old = 'range(gl.program_id(0), NUM_OUTPUT_TILES, NUM_THREAD_GROUPS)'
        assert source_text.count(old) == 2
        # Eliminate the persistent tile-loop backedge entirely. A one-tile
        # range otherwise folds pid+96<5 into signed CMPK -91 on this backend.
        source_text = source_text.replace('for tile_id in '+old+':',
            'if gl.program_id(0) == 4:\n        tile_id = gl.program_id(0)')
    frozen_source.write_text(source_text)
    args.source = frozen_source
    import builtins
    import getpass
    shared_lock = f'/tmp/{getpass.getuser()}.history.log.lock'
    original_open = builtins.open

    def private_open(file, *positional, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file) == shared_lock:
            file = Path.cwd() / 'history.lock'
        return original_open(file, *positional, **kwargs)

    builtins.open = private_open
    import torch
    import triton
    import triton._C.libtriton as lib

    sys.path.insert(0, str(REPO))
    assert os.environ['GPU_CHIP'] == 'sbx2'
    assert triton.runtime.driver.active.get_current_target().arch == 'gfx946'
    assert torch.cuda.get_device_properties(0).multi_processor_count == CU
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    save('environment.json', dict(environment={key:os.environ.get(key) for key in
         ['GPU_CHIP','GPU_ARGS','GPU_DFLAGS','AMDGCN_USE_BUFFER_OPS','TRITON_BUFFER_CACHE_SWIZZLE',
          'TRITON_MLS_B_INTERLEAVE','PMD_PATH','PYTHONPATH','LD_LIBRARY_PATH','TRITON_HIP_CLANG_PATH',
          'GPU_OUT','TRITON_CACHE_DIR','TRITON_HOME','RPY_HISTORY_LOG','OMP_NUM_THREADS','MKL_NUM_THREADS',
          'OPENBLAS_NUM_THREADS','OPTALL_PMD_LEASE','OPTALL_PREFLIGHT','OPTALL_COMPILE_ONLY']},
          python=sys.executable, torch=torch.__version__, triton=triton.__version__))

    tn = args.layout != 'tt'
    global M, N, K, TILES, BM, BN, BK
    record = json.loads(args.record.read_text())
    Path('record.snapshot.json').write_bytes(args.record.read_bytes())
    M, N, K = record['shape']
    assert (M,N,K) == (22736,1152,1536)
    ref = args.record.parent
    options = {key: tuple(value) if isinstance(value, list) else value
               for key, value in record['replay_job']['options'].items()}
    options.update(json.loads(args.options))
    BM, BN, BK = (options['BLOCK_SIZE_'+dim] for dim in 'MNK')
    TILES = 1 if args.probe_tail else ((M+BM-1)//BM)*((N+BN-1)//BN)
    options.update(M=M, N=N, K=K, B_K_CONTIGUOUS=tn, FORCE_MASKED_STORE=False)
    if args.source:
        source = args.source
    else:
        source = SOURCE_DIR / 'kernel.py'
    module = load_source(source)
    pack_b = getattr(module, "pack_b", None)
    dependencies = {}
    for imported in list(sys.modules.values()):
        file = getattr(imported, '__file__', None)
        if file and Path(file).is_file() and Path(file).resolve().is_relative_to(REPO/'boltops'):
            path = Path(file).resolve()
            relative = path.relative_to(REPO)
            destination = Path.cwd()/'source-dependencies'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(path.read_bytes())
            dependencies[str(relative)] = sha(destination.read_bytes())
    save('source-snapshot.json',dict(source_sha256=sha(source.read_bytes()),
                                   original_source=str(original_source),dependencies=dependencies))
    kernel = module._gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel
    guard = 65536

    def guarded(shape, fill):
        count = 1
        for dim in shape:
            count *= dim
        initial = torch.full((count + 2 * guard,), fill, dtype=torch.bfloat16)
        raw = initial.cuda()
        return initial, raw, raw[guard:-guard].view(shape)

    ai, ar, a = guarded((M, K), float('nan'))
    bi, br, b = guarded((K, N), float('nan'))
    ci, cr, c = guarded((M, N), 123)
    ci[guard:-guard] = float('nan')
    if tn:
        pi, pr, packed = guarded((N, K), 123)
        w = packed.T
    else:
        w = b
    assert w.stride() == ((1,K) if tn else (N,1))

    positional = (a, w, c)
    handle = kernel.warmup(*positional, grid=(CU,), **options)
    fresh = executable_sections(handle.asm['hsaco'])
    _compact_prepare_handle(handle, record, record['source']['symbol'], SOURCE_DIR, args.archived)
    if args.layout == 'pack':
        pack_handle = pack_b.warmup(b, w, K, N, args.pack_k, args.pack_n, num_warps=4,
                                   grid=(triton.cdiv(K,args.pack_k), triton.cdiv(N,args.pack_n)))
        _compact_prepare_handle(pack_handle, record, 'pack_b', SOURCE_DIR, args.archived)
    handle._init_handles()
    resources = dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared)
    Path('kernel.amdgcn').write_text(handle.asm['amdgcn'])
    Path('kernel.hsaco').write_bytes(handle.asm['hsaco'])
    for kind in ('ttgir', 'llir'):
        if kind in handle.asm:
            Path(f'kernel.{kind}').write_text(handle.asm[kind])
    negative_cmpk = []
    for instruction in re.finditer(r'\bs_cmpk_(\w+)_i32\s+[^,]+,\s*(-?0x[0-9a-fA-F]+|-?\d+)',handle.asm['amdgcn']):
        immediate = int(instruction.group(2),0)
        if immediate < 0 or 0x8000 <= immediate <= 0xffff:
            negative_cmpk.append(dict(instruction=instruction.group(0),line=handle.asm['amdgcn'][:instruction.start()].count('\n')+1))
    save('signed-cmpk-audit.json',dict(negative_signed_cmpk=negative_cmpk,
                                     scope='Compiled candidate before the first GPU dispatch'))
    assert not negative_cmpk, 'Negative signed CMPK needs explicit semantic-equivalent simulator workaround before running'
    save('compile.json', dict(source=str(source), original_source=str(original_source), source_sha256=sha(source.read_bytes()),
         runner_sha256=sha(RUNNER_SOURCE), helper_sha256=None,
         grid=CU, strides=dict(A=a.stride(), B=w.stride(), C=c.stride()), options=options,
         resources=resources, fresh_sections=fresh, dispatched_sections=executable_sections(handle.asm['hsaco']),
         toolchain=dict(triton=triton.__version__, torch=torch.__version__, libtriton_sha256=sha(Path(lib.__file__).read_bytes()),
                        clang_sha256=sha(Path(os.environ['TRITON_HIP_CLANG_PATH']).read_bytes()))))
    assert not resources['spills'] and resources['LDS_bytes'] <= 128*1024, resources
    if args.compile_only:
        return

    def measure(index, stage, launch, main=False):
        before = set(Path('g').glob('*/*/stats.txt'))
        launch()
        torch.cuda.synchronize()
        files = set(Path('g').glob('*/*/stats.txt')) - before
        assert len(files) == 1, files
        path = files.pop()
        if main:
            # In the diagnostic branch the compiler proves that tile4's
            # second 128-column accumulator is entirely masked and removes
            # its MMACs. Both full-size producer MLS loads remain emitted.
            counted_options = dict(options, BLOCK_SIZE_N=128) if args.probe_tail else options
            sample, raw = collect_dispatch(path, CU, TILES, counted_options, K)
        else:
            raw = path.read_bytes()
            content = raw.decode()
            cycles = re.findall(r'^system.gcSimCycles\s+(\d+)', content, re.M)
            assert len(cycles) == 1 and 'End Simulation Statistics' in content
            sample = dict(cycles=int(cycles[0]), us=int(cycles[0])*455/1e6, stats_sha256=sha(raw))
        with gzip.open(f'stats-{index}-{stage}.txt.gz', 'wb', compresslevel=4) as archive:
            archive.write(raw)
        print('timing', index, stage, sample, flush=True)
        return sample

    def validate_pack(bh):
        packed_host = pr.cpu()
        assert (packed_host[:guard] == 123).all() and (packed_host[-guard:] == 123).all(), 'pack canaries'
        assert torch.equal(packed_host[guard:-guard].view(N,K).T, bh), 'packed values differ'

    samples = []
    for index in range(args.dispatches):
        seed = 2059 if index == 0 else 59
        if index < 2:
            generator = torch.Generator().manual_seed(seed)
            for host, device, shape in ((ai, ar, (M,K)), (bi, br, (K,N))):
                for rows in host[guard:-guard].view(shape).split(1024):
                    rows.copy_((torch.randn(rows.shape, generator=generator) * .5).bfloat16())
                device.copy_(host)
            ah = ai[guard:-guard].view(M,K).float()
            bh = bi[guard:-guard].view(K,N)
            bf = bh.float()
            if args.layout == 'prepacked':
                pi[guard:-guard].view(N,K).copy_(bh.T)
                pr.copy_(pi)
        cr.copy_(ci)
        torch.cuda.synchronize()
        stages = {}
        if args.layout == 'pack':
            assert not args.stream
            stages['pack'] = measure(index, 'pack', lambda: pack_b[(triton.cdiv(K,args.pack_k),triton.cdiv(N,args.pack_n))](
                b, w, K, N, args.pack_k, args.pack_n, num_warps=4))
            if args.pack_only:
                validate_pack(bh)
                assert torch.equal(br.cpu().view(torch.uint8), bi.view(torch.uint8)), 'B modified'
                sample = dict(seed=seed, stages=stages, total_us=stages['pack']['us'],
                              validation=dict(elements=K*N, mismatch=0, nonfinite=0,
                                              pack_canaries=True, input_unchanged=True))
                save(f'sample-{index}.json', sample)
                samples.append(sample)
                continue
        stages['gemm'] = measure(index, 'gemm', lambda: kernel[(CU,)](*positional, **options), True)
        host = cr.cpu()
        # Preserve complete output even if a subsequent guard or input check fails.
        torch.save(host[guard:-guard].view(M,N).clone(), f'output-{index}-seed-{seed}.pt')
        assert (host[:guard] == 123).all() and (host[-guard:] == 123).all(), 'output canaries'
        assert torch.equal(ar.cpu().view(torch.uint8), ai.view(torch.uint8)), 'A modified'
        assert torch.equal(br.cpu().view(torch.uint8), bi.view(torch.uint8)), 'B modified'
        if tn:
            validate_pack(bh)
        actual = host[guard:-guard].view(M,N)
        input_hashes = {name:sha(tensor.view(torch.uint8).numpy().tobytes())
                        for name,tensor in [('A',ai),('B',bi)]}
        save(f'input-{index}-seed-{seed}.json',input_hashes)
        if args.probe_tail:
            assert actual[256:].isnan().all() and actual[:256,:1024].isnan().all(), 'probe wrote outside its one output tile'
            actual = actual[:256,1024:1152]
        oracle_m, oracle_n = actual.shape
        baseline_path = args.baseline_out / f'output-seed-{seed}.pt' if args.baseline_out else None
        baseline_mismatch = None
        if baseline_path and args.out == args.baseline_out:
            torch.save(actual.clone(), baseline_path)
        elif baseline_path:
            baseline = torch.load(baseline_path, weights_only=True)
            baseline_mismatch = int((~torch.isclose(actual.float(), baseline.float(), atol=.02, rtol=.01)).sum())
            del baseline
        mismatch = nonfinite = 0
        max_abs = abs_err = round_err = 0.
        bad_coordinates=[]
        bad_rows=[]
        bad_columns=torch.zeros(oracle_n,dtype=torch.int64)
        for row in range(0, oracle_m, 64):
            fp32 = ah[row:row+64] @ (bf[:,1024:1152] if args.probe_tail else bf)
            golden = fp32.bfloat16().float()
            observed = actual[row:row+64].float()
            nonfinite += int((~observed.isfinite()).sum())
            bad=~torch.isclose(observed, golden, atol=.02, rtol=.01)
            mismatch += int(bad.sum())
            row_counts=bad.sum(dim=1)
            bad_rows.extend([dict(row=row+j,count=int(count)) for j,count in enumerate(row_counts) if count])
            bad_columns += bad.sum(dim=0)
            if len(bad_coordinates)<64:
                for rr,cc in bad.nonzero().tolist()[:64-len(bad_coordinates)]:
                    bad_coordinates.append(dict(row=row+rr,col=cc,actual=float(observed[rr,cc]),oracle=float(golden[rr,cc])))
            max_abs = max(max_abs, float((observed-golden).abs().max()))
            abs_err += float((observed-fp32).abs().sum(dtype=torch.float64))
            round_err += float((golden-fp32).abs().sum(dtype=torch.float64))
        sample = dict(seed=seed, stages=stages, total_us=sum(s['us'] for s in stages.values()),
                      validation=dict(elements=oracle_m*oracle_n, mismatch=mismatch, nonfinite=nonfinite,
                                      max_abs=None if nonfinite else max_abs, atol=.02, rtol=.01, baseline_mismatch=baseline_mismatch,
                                      canaries=True, inputs_unchanged=True,
                                      input_sha256=input_hashes,
                                      mean_abs_vs_fp32=None if nonfinite else abs_err/(oracle_m*oracle_n), rounding_floor=round_err/(oracle_m*oracle_n)))
        save(f'sample-{index}.json', sample)
        save(f'mismatch-coordinates-{index}.json',dict(examples=bad_coordinates,rows=bad_rows,
             columns=[dict(col=j,count=int(count)) for j,count in enumerate(bad_columns) if count]))
        assert mismatch == nonfinite == 0 and baseline_mismatch in (None,0) and abs_err <= 1.05*round_err + 1e-6*oracle_m*oracle_n, sample
        samples.append(sample)
    result = dict(status='PASS', samples=samples,
                  total_us=statistics.median(s['total_us'] for s in samples[1:] or samples))
    save('result.json', result)
    print('PASS', result['total_us'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=lambda value: Path(value).resolve())
    p.add_argument('--source', type=lambda value: Path(value).resolve(), default=SOURCE_DIR/'kernel.py')
    p.add_argument('--record', type=lambda value: Path(value).resolve(), default=SOURCE_DIR/'record.json')
    p.add_argument('--baseline-out', type=lambda value: Path(value).resolve())
    p.add_argument('--options', default='{}')
    p.add_argument('--layout', choices=('tt', 'prepacked', 'pack'), default='tt')
    p.add_argument('--dispatches', type=int, default=3)
    p.add_argument('--pack-k', type=int, default=32)
    p.add_argument('--pack-n', type=int, default=64)
    p.add_argument('--pack-only', action='store_true')
    p.add_argument('--archived', action='store_true')
    p.add_argument('--stream', action='store_true')
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--probe-tail', action='store_true', help='Diagnostic only: full logical dimensions, compute tile 4 only')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if _COMPACT_PARSE_ONLY: return args
    if args.worker:
        worker(args)
        return
    assert not args.stream or (args.source and not args.archived and args.layout != 'pack')
    assert not args.pack_only or args.layout == 'pack'
    assert args.dispatches > 0
    assert args.dispatches == 3 or (args.probe_tail and args.dispatches == 1), 'R043 requires one warmup plus two timed dispatches'
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / 'g').mkdir()
    record = json.loads(args.record.read_text())
    env = dict(os.environ, **record['environment']['environment_variables'])
    env.update(GPU_ARGS="['--abs-max-tick=10000000000']", TRITON_CACHE_DIR=str(out/'cache'),
               TRITON_HOME=str(out/'triton-home'), RPY_HISTORY_LOG=str(out/'history.log'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('HSA_TOOLS_LIB', None)
    with tempfile.TemporaryDirectory(prefix='r043-') as directory:
        model = Path(directory)/'g'
        model.symlink_to(out/'g', target_is_directory=True)
        env['GPU_OUT'] = str(model)
        command = [sys.executable, str(Path(__file__).resolve()), str(out), '--worker',
                   '--layout', args.layout, '--dispatches', str(args.dispatches),
                   '--pack-k', str(args.pack_k), '--pack-n', str(args.pack_n),
                   '--record', str(args.record), '--options', args.options]
        if args.probe_tail:
            command.append('--probe-tail')
        if args.baseline_out:
            command += ['--baseline-out', str(args.baseline_out)]
        if args.source:
            command += ['--source', str(args.source)]
        for enabled, option in zip((args.archived, args.stream, args.compile_only, args.pack_only),
                                   ('--archived', '--stream', '--compile-only', '--pack-only')):
            if enabled:
                command.append(option)
        with (out/'worker.log').open('w') as log:
            process=subprocess.Popen(command, env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT)
            save(out/'launch-process.json',dict(pid=process.pid,ppid=os.getpid(),sid=os.getsid(0),command=command,
                 lease=env.get('OPTALL_PMD_LEASE'),preflight=env.get('OPTALL_PREFLIGHT')))
            rc=process.wait()
            save(out/'exit.json',dict(returncode=rc))
            if rc:raise subprocess.CalledProcessError(rc,command)




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
