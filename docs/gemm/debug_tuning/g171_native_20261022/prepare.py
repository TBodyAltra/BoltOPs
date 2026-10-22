"""Prepare the historical node60 experiment outside git; do not launch kernels.

Requires the external IR snapshots and pinned PMD/Triton environment in manifest.json.
This is a retained experiment, not an operator entry point or a general autotuner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

from scale_product import transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'manifest.json').read_text())
    source = args.evidence_root.resolve()
    output = args.output_root.resolve()
    session_root = Path(manifest['external_dependencies'][0]).parents[1]
    if output.parent != session_root or output.exists():
        parser.error('output-root must be a NEW direct child of ' + str(session_root))
    # An empty .git sentinel exists at /xukang/code; reject actual repositories/worktrees.
    if any((p / '.git').is_file() or (p / '.git/HEAD').is_file()
           for p in [output, *output.parents]):
        parser.error('All generated files must be outside git')
    snapshots = {}
    for name, record in manifest['inputs'].items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            parser.error('Input snapshot SHA256 mismatch: ' + name)
        snapshots[name] = data
    output.mkdir()
    for name, data in snapshots.items():
        (output / name).write_bytes(data)
    (output / 'scale-product.ttgir').write_text(transform(snapshots['control.ttgir'].decode()))
    for name in ['bench-worker.py', 'native-transform.py', 'capture-env.py',
                 'perf_metrics_20261019.py', 'trial_mls_class_aligned_dotfirst.py', 'env.sh']:
        shutil.copy2(here / name, output / name)
    configs = json.loads((here / 'configs.json').read_text())
    for name, config in configs.items():
        case = output / name
        case.mkdir()
        (case / 'request.json').write_text(json.dumps(config, indent=2) + '\n')
        quote = shlex.quote
        launch = (
            f'source {quote(str(output / "env.sh"))}\n'
            'export GPU_CHIP=sb GPU_ARGS="[]" GPU_DFLAGS="[\'StatLog\']"\n'
            'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
            f'export TRITON_CACHE_DIR={quote(str(output / "cache"))}\n'
            'unset HSA_TOOLS_LIB\n'
            f'cd {quote(str(case))}\n'
            f'python {quote(str(output / "capture-env.py"))}\n'
            f'python {quote(str(output / "bench-worker.py"))} "$(cat request.json)" "$@"\n'
        )
        (case / 'launch.sh').write_text(launch)
    print('Prepared only; no process or queue started: ' + str(output))
    print('Explicit per-case execution: bash <case>/launch.sh --measure')


if __name__ == '__main__':
    main()
