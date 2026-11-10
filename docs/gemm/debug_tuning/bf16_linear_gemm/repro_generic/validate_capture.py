#!/usr/bin/env python3
"""Validate saved current-config captures; optionally fill Git state after capture.

Usage: python3 -B validate_capture.py [--finalize-environment] /abs/capture [...]
The optional finalization records its time explicitly; it does not replace the
compiler/library hashes observed inside the original capture process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from repro import DEFAULT_REPO, environment, sha256
import tune_lib as T


def validate(out, finalize=False):
    path = out / 'manifest.json'
    data = json.loads(path.read_text())
    assert data['correctness'] == 'PASS', path
    cfg = json.loads(Path(data['config_file']).read_text())
    assert cfg == data['config'] == json.loads((out / 'config.json').read_text())
    for source, digest in data['source_sha256'].items():
        assert sha256(Path(source)) == digest, source
    assert sha256(out / 'kernel_source.py') == sha256(Path(data['operator_file']))
    meta = json.loads((out / 'kernel.metadata.json').read_text())
    assert meta['target']['arch'] == 'gfx946', meta['target']
    for key, value in T.split_config(cfg)[1].items():
        assert meta[key] == value, (key, meta[key], value)
    for kind in ('ttir', 'ttgir', 'llir', 'amdgcn'):
        artifact = out / f'kernel.{kind}'
        assert artifact.stat().st_size > 0, artifact
        assert sha256(artifact) == data['artifacts'][artifact.name]['sha256'], artifact
    expected_wgs = T.num_tg(*data['shape'][:2], cfg)
    stats_paths = sorted((out / 'g').glob('*/*/stats.txt'), key=lambda p: int(p.parent.name))
    assert len(stats_paths) == len(data['dispatches']) == 6
    all_artifacts = {}
    for stats, record in zip(stats_paths, data['dispatches']):
        observed = T.stats_of(stats)
        assert observed['completed_wgs'] == expected_wgs
        assert observed['cu_count'] == 48
        assert observed['cycles'] == record['cycles']
        assert observed['freq_ghz'] == record['freq_ghz']
        assert len(list(stats.parent.glob('*.kinfo'))) == 1
        assert list(stats.parent.glob('ttrace_model__*.out')), stats
        for rel in record['perf_paths']:
            perf = out / rel
            assert perf.stat().st_size > 0, perf
        for artifact in stats.parent.iterdir():
            if artifact.is_file():
                all_artifacts[str(artifact.relative_to(out))] = dict(
                    bytes=artifact.stat().st_size, sha256=sha256(artifact))
    if 'dispatch_artifacts' in data:
        assert all_artifacts == data['dispatch_artifacts'], 'dispatch artifact size/hash mismatch'
    assert data['median_cycles'] == statistics.median(r['cycles'] for r in data['dispatches'][1:])
    if finalize:
        current = environment(DEFAULT_REPO)
        # Confirm the code and compiler did not change before filling missing Git info.
        for key in ('libtriton_sha256', 'build'):
            assert data['environment']['triton'][key] == current['triton'][key]
        assert data['environment']['source_sha256'] == current['source_sha256']
        data['environment']['boltops'] = current['boltops']
        data['environment']['triton']['source'] = current['triton']['source']
        data['environment']['git_state_verified_after_capture_at'] = datetime.now(timezone.utc).isoformat()
        data['dispatch_artifacts'] = all_artifacts
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(f"PASS {out}: {data['implementation']} {data['layout']}, "
          f"6 perf dispatches, {data['median_us']:.6f} us; "
          f"{sum(a['bytes'] for a in all_artifacts.values())} bytes of raw dispatch artifacts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize-environment', action='store_true')
    parser.add_argument('captures', type=Path, nargs='+')
    args = parser.parse_args()
    for capture in args.captures:
        validate(capture.resolve(), args.finalize_environment)


if __name__ == '__main__':
    main()
