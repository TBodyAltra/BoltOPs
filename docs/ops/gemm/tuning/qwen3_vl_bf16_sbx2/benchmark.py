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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    job = dict(record['replay_job'])
    if args.compile_only:
        job['compile_only'] = True
    source = record['source']
    assert record['environment']['chip'] == 'sbx2'
    assert record['layout'] == job.get('layout', 'TN')
    assert record['shape'] == job['shape']
    source_archive = (args.record.parent / source['archive']).resolve()
    if args.dry_run:
        print(json.dumps(dict(record=record['record_key'], shape=job['shape'],
                              grid=[record['full_grid_TGs']], strides=record['config']['strides'],
                              options=record['config']['parameters'], source_sha256=source['sha256'],
                              dispatches=job['dispatches'], seed=job['seed'], timed_seed=job['timed_seed']), indent=2))
        return
    if args.output is None:
        parser.error('--output is required for replay')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    overlay = out/'source'
    shutil.copytree(REPO/'boltops', overlay/'boltops')
    with tarfile.open(source_archive, 'r:gz') as archive:
        for item in [source, *source.get('dependencies', [])]:
            data = archive.extractfile(item['member']).read()
            assert sha(data) == item['sha256'], item['member']
            path = (overlay / item['path']).resolve()
            assert path.is_relative_to(overlay)
            path.write_bytes(data)
    if archived := record['assembly'].get('archived_code_object'):
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
        subprocess.run([sys.executable, str(HERE/'replay_worker.py'), str(inputs), '--repo', str(overlay)],
                       cwd=out, env=env, check=True)
    compiled = json.loads((out/'compile.json').read_text())
    assert compiled['source_sha256'] == source['sha256']
    assert compiled['grid'] == record['full_grid_TGs'] and compiled['strides'] == record['config']['strides']
    assert compiled['resources'] == record['resources']
    assert executable_sections((out/'kernel.hsaco').read_bytes()) == record['assembly']['executable_sections']


if __name__ == '__main__':
    main()
