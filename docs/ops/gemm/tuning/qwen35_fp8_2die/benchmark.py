"""Replay one published Qwen3.5 dual-die record using its measured source."""
import argparse
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def check_toolchain(record):
    environment = record['environment']
    if record['replay']['worker']=='bf16':
        hashes = environment['runtime']['compiler']
    else:
        hashes = dict(libtriton=environment['triton']['libtriton_sha256'],
                      clang=environment['clang']['sha256'], lld=environment['clang']['lld_sha256'])
    spec = find_spec('triton')
    if spec is None:
        raise SystemExit('Load the record’s Triton runtime before replay')
    files = dict(libtriton=Path(spec.origin).parent/'_C/libtriton.so',
                 clang=os.environ.get('TRITON_HIP_CLANG_PATH'),
                 lld=os.environ.get('TRITON_HIP_LLD_PATH'))
    for name, expected in hashes.items():
        path = files[name]
        if not path or not Path(path).is_file():
            raise SystemExit(f'Missing measured toolchain component: {name}')
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Toolchain SHA-256 mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--repo', type=Path, default=HERE.parents[4])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    source = record['source']
    repo = args.repo.resolve()
    commit = source['commit']
    subprocess.run(['git', '-C', repo, 'cat-file', '-e', commit+'^{commit}'], check=True)
    patches = [HERE/name for name in source.get('patches', [])]
    for patch in patches:
        if not patch.is_file():
            parser.error(f'Missing measured source patch: {patch}')
    if args.dry_run:
        print(json.dumps(record['replay'], indent=2))
        return
    if args.output is None:
        parser.error('--output must name a new directory for the replay')
    check_toolchain(record)
    output = args.output.resolve()
    output.mkdir()
    checkout = output/'source'
    checkout.mkdir()
    archive = subprocess.check_output(['git', '-C', repo, 'archive', commit, 'boltops'])
    subprocess.run(['tar', '-xf', '-', '-C', checkout], input=archive, check=True)
    for patch in patches:
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)], cwd=checkout, check=True)
    replay = record['replay']
    job = output/'job.json'
    job.write_text(json.dumps(replay['input'], indent=2)+'\n')
    env = os.environ.copy()
    run = output/'run'
    if replay['worker']=='bf16':
        command = [sys.executable, str(HERE/'bf16.py'), '--input', str(job),
                   '--variant', replay['variant'], '--repo', str(checkout),
                   '--output', str(run), '--repeats', str(replay['repeats'])]
        subprocess.run(command, env=env, check=True)
    else:
        run.mkdir()
        previous_path = env.get('PYTHONPATH')
        env['PYTHONPATH'] = os.pathsep.join([str(checkout)] + ([previous_path] if previous_path else []))
        env.update(GPU_CHIP='sbx2', GPU_DFLAGS="['StatLog']",
                   GPU_ARGS="['--abs-max-tick=10000000000']", GPU_OUT='m5out',
                   TRITON_HOME=str(run/'triton-home'), TRITON_CACHE_DIR=str(run/'cache'),
                   AMDGCN_USE_BUFFER_OPS='1')
        for key in ('HSA_TOOLS_LIB', 'GPU_GFLAGS', 'BOLTOPS_GEMM_CONFIG_DIR'):
            env.pop(key, None)
        subprocess.run([sys.executable, str(HERE/'fp8.py'), str(job)], cwd=run, env=env, check=True)
    result = json.loads((run/'result.json').read_text())
    if result['status']!='PASS':
        raise SystemExit('The full-grid replay failed')
    print(json.dumps(dict(kernel_us=result['kernel_us'], kernel_tops=result['kernel_tops']), indent=2))


if __name__ == '__main__':
    main()
