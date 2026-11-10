"""Prepare this pinned node60 experiment outside Git; never launch a kernel or queue."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'manifest.json').read_text())
    source, output = args.evidence_root.resolve(), args.output_root.resolve()
    session_root = Path(manifest['external_root']).parent
    if output.parent != session_root or output.exists():
        parser.error('output-root must be a NEW direct child of ' + str(session_root))
    # /xukang/code has a .git sentinel; only real repositories/worktrees are rejected.
    if any((p / '.git').is_file() or (p / '.git/HEAD').is_file()
           for p in [output, *output.parents]):
        parser.error('Generated artifacts must stay outside Git')
    payload = {}
    for name, record in manifest['replay_files'].items():
        path = source / name if record['location'] == 'external' else here / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            parser.error('SHA256 mismatch: ' + str(path))
        payload[name] = data
    output.mkdir()
    for name, data in payload.items():
        (output / name).write_bytes(data)
    shutil.copy2(here / 'manifest.json', output / 'manifest.json')
    print('Prepared only; no process or validation queue started: ' + str(output))
    print('Run baseline first: python ' + str(output / 'run_one.py') + ' P01-production-w8')
    print('Then candidate: python ' + str(output / 'run_one.py') + ' P11-aligned-load-builtinflags')


if __name__ == '__main__':
    main()
