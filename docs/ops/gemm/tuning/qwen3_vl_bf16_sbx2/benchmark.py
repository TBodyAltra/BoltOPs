# SPDX-License-Identifier: MIT
"""Replay a Qwen3-VL BF16 dual-die record with its measured source and launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / 'boltops').is_dir())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def elf_sections(data):
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected a little-endian ELF64 code object')
    offset, = struct.unpack_from('<Q', data, 40)
    size, count, names = struct.unpack_from('<HHH', data, 58)
    entries = [struct.unpack_from('<IIQQQQIIQQ', data, offset + index*size)
               for index in range(count)]
    symbols = data[entries[names][4]:entries[names][4] + entries[names][5]]
    for entry in entries:
        name = symbols[entry[0]:].split(b'\x00', 1)[0].decode()
        yield name, entry[1], data[entry[4]:entry[4]+entry[5]]


def executable_sections(data):
    sections = {name: sha(contents) for name, _, contents in elf_sections(data)
                if name in ('.text', '.rodata')}
    if '.text' not in sections:
        raise ValueError('code object has no executable section')
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--current-toolchain', action='store_true',
                        help='measure freshly compiled code using the sourced PMD toolchain')
    parser.add_argument('--public-api', action='store_true',
                        help='replay a configured persistent kernel through gemm_a16w16')
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    job = dict(record['replay_job'], expected_toolchain=record['environment']['toolchain'],
               expected_executable_sections=record['assembly']['executable_sections'],
               current_toolchain=args.current_toolchain)
    if args.compile_only:
        job['compile_only'] = True
    if args.public_api:
        if not args.current_toolchain or job['family'] != 'persistent':
            parser.error('--public-api requires --current-toolchain and a persistent record')
        job['public_api'] = True
    source = record['source']
    assert record['environment']['chip'] == 'sbx2'
    assert record['layout'] == job.get('layout', 'TN')
    assert record['shape'] == job['shape']
    source_archive = (args.record.parent / source['archive']).resolve()
    if args.dry_run:
        plan = dict(record=record['record_key'], shape=job['shape'],
                              grid=[record['full_grid_TGs']], strides=record['config']['strides'],
                              options=record['config']['parameters'], source_sha256=source['sha256'],
                              dispatches=job['dispatches'], seed=job['seed'], timed_seed=job['timed_seed'],
                              current_toolchain=args.current_toolchain)
        if args.public_api:
            plan['public_api'] = True
        print(json.dumps(plan, indent=2))
        return
    if args.output is None:
        parser.error('--output is required for replay')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_root = REPO if args.public_api else out/'source'
    if not args.public_api:
        shutil.copytree(REPO/'boltops', source_root/'boltops')
        with tarfile.open(source_archive, 'r:gz') as archive:
            for item in [source, *source.get('dependencies', [])]:
                data = archive.extractfile(item['member']).read()
                assert sha(data) == item['sha256'], item['member']
                path = (source_root / item['path']).resolve()
                assert path.is_relative_to(source_root)
                path.write_bytes(data)
    archived = record['assembly'].get('archived_code_object')
    if archived and not args.current_toolchain:
        record_dir = args.record.resolve().parent
        job.update(archived_code_object=str(record_dir/archived),
                   archived_code_sha256=record['assembly']['hsaco_sha256'],
                   archived_assembly=str(record_dir/record['assembly']['path']),
                   archived_assembly_sha256=record['assembly']['sha256'])
    inputs = out/'job.json'
    inputs.write_text(json.dumps(job, indent=2)+'\n')
    env = dict(os.environ, **record['environment']['environment_variables'])
    env.update(GPU_ARGS="['--abs-max-tick=10000000000']", TRITON_CACHE_DIR=str(out/'cache'),
               TRITON_HOME=str(out/'triton-home'), RPY_HISTORY_LOG=str(out/'history.log'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('HSA_TOOLS_LIB', None)
    env.pop('BOLTOPS_GEMM_CONFIG_DIR', None)
    (out/'g').mkdir()
    with tempfile.TemporaryDirectory(prefix='qvl-gpu-') as directory:
        model = Path(directory)/'g'
        model.symlink_to(out/'g', target_is_directory=True)
        env['GPU_OUT'] = str(model)
        subprocess.run([sys.executable, str(HERE/'replay_worker.py'), str(inputs), '--repo', str(source_root)],
                       cwd=out, env=env, check=True)
    compiled = json.loads((out/'compile.json').read_text())
    if args.public_api:
        assert compiled['public_api_verified']
    else:
        assert compiled['source_sha256'] == source['sha256']
    assert compiled['grid'] == record['full_grid_TGs'] and compiled['strides'] == record['config']['strides']
    if not args.current_toolchain:
        assert compiled['resources'] == record['resources']
    sections = executable_sections((out/'kernel.hsaco').read_bytes())
    if args.compile_only or args.current_toolchain:
        assert compiled['code_origin'] == 'fresh'
        assert sections == compiled['fresh_executable_sections']
        print(json.dumps(dict(code_origin='fresh',
                              matches_measured=compiled['fresh_matches_measured'],
                              archived_abi_verified=compiled['archived_abi_verified'])))
    else:
        assert sections == record['assembly']['executable_sections']


if __name__ == '__main__':
    main()
