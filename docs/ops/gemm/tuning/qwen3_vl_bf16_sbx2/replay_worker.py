"""Replay a recorded BF16 GEMM kernel on dual-die PMD."""
import argparse
import builtins
import configparser
import getpass
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import sys

from benchmark import elf_sections, executable_sections


def write_json(name, value):
    Path(name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def code_object_abi(data):
    """Return the AMDHSA kernel ABI that must agree with the fresh launcher."""
    import msgpack

    for _, section_type, section in elf_sections(data):
        if section_type != 7:  # SHT_NOTE
            continue
        offset = 0
        while offset + 12 <= len(section):
            name_size, desc_size, note_type = struct.unpack_from('<III', section, offset)
            name = section[offset + 12:offset + 12 + name_size].rstrip(b'\x00')
            start = offset + 12 + ((name_size + 3) & ~3)
            offset = start + ((desc_size + 3) & ~3)
            if name != b'AMDGPU' or note_type != 32:  # NT_AMDGPU_METADATA
                continue
            metadata = msgpack.unpackb(section[start:start + desc_size], raw=False)
            keys = ('.name', '.symbol', '.args', '.kernarg_segment_size',
                    '.kernarg_segment_align', '.group_segment_fixed_size',
                    '.private_segment_fixed_size', '.wavefront_size',
                    '.max_flat_workgroup_size', '.reqd_workgroup_size',
                    '.uniform_work_group_size', '.workgroup_processor_mode')
            kernels = [{key: kernel.get(key) for key in keys}
                       for kernel in metadata['amdhsa.kernels']]
            return (data[7:9], struct.unpack_from('<I', data, 48)[0],
                    metadata['amdhsa.version'], metadata.get('amdhsa.target'), kernels)
    raise ValueError('code object has no AMDHSA metadata')


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    job = json.loads(args.job.read_text())
    original_open = builtins.open
    shared_lock = '/tmp/' + getpass.getuser() + '.history.log.lock'

    def private_open(file, *positional, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fsdecode(file) == shared_lock:
            file = Path.cwd() / 'history.lock'
        return original_open(file, *positional, **kwargs)

    builtins.open = private_open
    import torch
    import triton
    import triton._C.libtriton as lib
    from boltops.gemm.config import get_gemm_config
    from boltops.gemm.triton.gemm_config_utils import compute_splitk_params

    toolchain = dict(triton=triton.__version__, torch=torch.__version__,
                     libtriton_sha256=sha(Path(lib.__file__).read_bytes()),
                     clang_sha256=sha(Path(os.environ['TRITON_HIP_CLANG_PATH']).read_bytes()))
    expected_toolchain = job['expected_toolchain']
    for key, expected in expected_toolchain.items():
        if toolchain.get(key) != expected:
            raise ValueError(f'replay toolchain differs for {key}: expected {expected}, got {toolchain.get(key)}')

    assert os.environ['GPU_CHIP'] == 'sbx2'
    assert triton.runtime.driver.active.get_current_target().arch == 'gfx946'
    assert torch.cuda.get_device_properties(0).multi_processor_count == 96
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    m, n, k = job['shape']
    family = job['family']
    is_tn = job.get('layout', 'TN') == 'TN'
    assert is_tn or job.get('layout') == 'TT'
    options = {key: tuple(value) if isinstance(value, list) else value
               for key, value in job.get('options', {}).items()}
    if family in ('baseline', 'triton'):
        overrides = options
        options, _ = get_gemm_config('GEMM-A16W16', m, None, None, backend='triton',
                                     arch='gfx946', cu_num=96)
        options.update(overrides)
        options = compute_splitk_params(options, k)
        assert options['NUM_KSPLIT'] == 1
    paths = {'baseline': ('triton.gemm_a16w16', '_gemm_a16_w16_kernel'),
             'triton': ('triton.gemm_a16w16', '_gemm_a16_w16_kernel'),
             'wasp': ('triton.gemm_a16w16_wasp', '_gemm_fp16_mls_kernel'),
             'persistent': ('gluon.gemm_a16w16', '_gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel')}
    module_name, symbol = paths[family]
    module = importlib.import_module('boltops.gemm.' + module_name)
    if job.get('masked_mls'):
        module = importlib.import_module('wasp_edges')
        symbol = 'gemm_mls_masked'
        options.update(MASK_M=m % options['BLOCK_SIZE_M'] != 0,
                       MASK_N=n % options['BLOCK_SIZE_N'] != 0)
        if 'MASK_K' in module.gemm_mls_masked.arg_names:
            options['MASK_K'] = k % options['BLOCK_SIZE_K'] != 0
        else:
            assert k % options['BLOCK_SIZE_K'] == 0
    kernel = getattr(module, symbol)
    tiles = math.ceil(m/options['BLOCK_SIZE_M'])*math.ceil(n/options['BLOCK_SIZE_N'])
    grid = min(tiles, options.get('NUM_SMS', tiles))
    guard = 65536
    ah = torch.empty((m, k), dtype=torch.bfloat16)
    bshape = (n, k) if is_tn else (k, n)
    bh = torch.empty(bshape, dtype=torch.bfloat16)

    def guarded(shape, fill):
        initial = torch.full((math.prod(shape) + 2*guard,), fill, dtype=torch.bfloat16)
        raw = initial.cuda()
        return initial, raw, raw[guard:-guard].view(shape)

    ai, ar, a = guarded((m,k), float('nan'))
    bi, br, b_physical = guarded(bshape, float('nan'))
    ci, cr, c = guarded((m,n), 123)
    ci[guard:-guard] = float('nan')
    b = b_physical.T if is_tn else b_physical
    assert a.stride() == (k,1) and c.stride() == (n,1)
    assert b.stride() == ((1,k) if is_tn else (n,1))
    if family == 'persistent':
        positional = (a, b, c)
        options.update(M=m, N=n, K=k, B_K_CONTIGUOUS=is_tn)
    elif family == 'wasp':
        assert job.get('masked_mls') or all(size % options['BLOCK_SIZE_'+dim] == 0 for dim, size in zip('MNK', (m,n,k)))
        positional = (a, b, c, b, m, n, k, *a.stride(), *b.stride(), *c.stride())
    else:
        positional = (a, b, None, c, m, n, k, *a.stride(), *b.stride(), m*n, *c.stride())
        options.update(activation="", use_activation=False, ADD_BIAS=False, SKIP_REDUCE=False)
    handle = kernel.warmup(*positional, grid=(grid,), **options)
    expected_sections = job['expected_executable_sections']
    fresh_binary = handle.asm['hsaco']
    fresh_sections = executable_sections(fresh_binary)
    fresh_matches_measured = fresh_sections == expected_sections
    code_origin = 'fresh'
    archived_abi_verified = False
    if 'archived_code_object' in job:
        binary = Path(job['archived_code_object']).read_bytes()
        assembly = Path(job['archived_assembly']).read_bytes()
        if sha(binary) != job['archived_code_sha256'] or sha(assembly) != job['archived_assembly_sha256']:
            raise ValueError('archived code object or assembly differs from the record')
        if executable_sections(binary) != expected_sections:
            raise ValueError('archived executable sections differ from the record')
        if code_object_abi(fresh_binary) != code_object_abi(binary):
            raise ValueError('archived code object is incompatible with the fresh launcher ABI')
        archived_abi_verified = True
        if not job.get('compile_only'):
            handle.kernel = binary
            handle.asm['hsaco'] = binary
            handle.asm['amdgcn'] = assembly.decode()
            code_origin = 'archived'
    elif not fresh_matches_measured:
        raise ValueError('fresh executable sections differ from the measured code object')
    handle._init_handles()
    resources = dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared)
    Path('kernel.amdgcn').write_text(handle.asm['amdgcn'])
    Path('kernel.hsaco').write_bytes(handle.asm['hsaco'])
    write_json('compile.json', dict(job=job, grid=grid, logical_tiles=tiles, strides=dict(A=a.stride(), B=b.stride(), C=c.stride()),
               symbol=symbol, options=options, resources=resources, source_sha256=sha(Path(module.__file__).read_bytes()),
               toolchain=toolchain, code_origin=code_origin, archived_abi_verified=archived_abi_verified,
               fresh_hsaco_sha256=sha(fresh_binary), fresh_executable_sections=fresh_sections,
               fresh_matches_measured=fresh_matches_measured, hsaco_sha256=sha(handle.asm['hsaco'])))
    assert resources['spills'] == 0 or (family == 'baseline' and not job.get('fallback_baseline')), resources
    if job.get('compile_only'):
        return
    assert resources['LDS_bytes'] <= 128*1024, f'gfx946 LDS limit exceeded: {resources}'
    def prepare(seed, kind):
        generator = torch.Generator().manual_seed(seed)
        if kind == 'ones':
            ah.fill_(1)
            bh.fill_(1)
        else:
            for matrix in (ah, bh):
                # Bound the temporary fp32 allocation for very large weights.
                for rows in matrix.split(1024):
                    rows.copy_((torch.randn(rows.shape, generator=generator)*.5).bfloat16())
        ai[guard:-guard].copy_(ah.view(-1))
        bi[guard:-guard].copy_(bh.view(-1))
        ar.copy_(ai)
        br.copy_(bi)
        bf = bh.float()
        return ah.float(), bf.T if is_tn else bf

    seed = job.get('seed', 59)
    input_kind = job.get('input', 'random')
    ah32, b32 = prepare(seed, input_kind)
    samples = []
    for index in range(job.get('dispatches', 1)):
        if index == 1 and 'timed_seed' in job:
            del ah32, b32
            ah32, b32 = prepare(job['timed_seed'], 'random')
            seed, input_kind = job['timed_seed'], 'random'
        cr.copy_(ci)
        torch.cuda.synchronize()
        before = set(Path('g').glob('*/*/stats.txt'))
        assert kernel[(grid,)](*positional, **options) is handle
        torch.cuda.synchronize()
        files = set(Path('g').glob('*/*/stats.txt')) - before
        assert len(files) == 1, files
        sample, raw = collect_dispatch(files.pop(), grid, tiles, options, k)
        with gzip.open(f'stats-{index}.txt.gz', 'wb', compresslevel=4) as archive:
            archive.write(raw)
        write_json(f'timing-{index}.json', sample)
        print('timing', index, sample, flush=True)
        host = cr.cpu()
        actual = host[guard:-guard].view(m,n)
        assert (host[:guard] == 123).all() and (host[-guard:] == 123).all(), 'output canaries'
        assert torch.equal(ar.cpu().view(torch.uint8), ai.view(torch.uint8)), 'A modified'
        assert torch.equal(br.cpu().view(torch.uint8), bi.view(torch.uint8)), 'B modified'
        mismatch = nonfinite = 0
        max_abs = absolute_errors = rounding_errors = 0.
        for row in range(0, m, 64):
            fp32 = ah32[row:row+64] @ b32
            golden = fp32.bfloat16().float()
            observed = actual[row:row+64].float()
            nonfinite += int((~observed.isfinite()).sum())
            mismatch += int((~torch.isclose(observed, golden, atol=.02, rtol=.01)).sum())
            max_abs = max(max_abs, float((observed-golden).abs().max()))
            absolute_errors += float((observed-fp32).abs().sum(dtype=torch.float64))
            rounding_errors += float((golden-fp32).abs().sum(dtype=torch.float64))
        sample.update(seed=seed, input_kind=input_kind)
        sample['validation'] = dict(elements=m*n, mismatch=mismatch, nonfinite=nonfinite, max_abs=max_abs if math.isfinite(max_abs) and not nonfinite else None,
                                    atol=.02, rtol=.01, output_canaries=True, inputs_unchanged=True,
                                    mean_abs_vs_fp32=absolute_errors/(m*n) if math.isfinite(absolute_errors) else None,
                                    rounding_floor=rounding_errors/(m*n))
        write_json(f'sample-{index}.json', sample)
        assert mismatch == nonfinite == 0 and absolute_errors <= 1.05*rounding_errors + 1e-6*m*n, sample
        samples.append(sample)
    measured = samples[1:] if len(samples) > 1 else samples
    us = statistics.median(s['us'] for s in measured)
    write_json('result.json', dict(status='PASS', job=job, samples=samples, kernel_us=us,
                                  tops=2*m*n*k/(us*1e6), logical_gbs=2*(m*k+n*k+m*n)/(us*1e3),
                                  benchmark_sha256=sha(Path(__file__).read_bytes())))


if __name__ == '__main__':
    main()
