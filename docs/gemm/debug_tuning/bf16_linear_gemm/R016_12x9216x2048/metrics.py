#!/usr/bin/env python3
"""R016 (12x9216x2048 BF16) MMAC utilization / effective bandwidth, one definition for all rows.

Common definition, identical to ../README.md "性能指标公共口径" and the R014 / R017 READMEs:

* useful FLOP        = 2*M*N*K  (no tile padding, no split-K reduction)
* MMAC utilization   = (useful FLOP / kernel time) / 432 TFLOPS
* logical bytes D    = 2*(M*K + K*N + M*N)  (BF16; A, B read once, C written once)
* effective bandwidth= D / kernel time       (logical, not measured external-memory traffic)
* bandwidth achieved = effective bandwidth / 1.7996485 TB/s
* roofline           = min(432 TFLOPS, (2MNK/D) * 1.7996485 TB/s)

`kernel time` is always the un-rounded median_us of the *full kernel* (PMD gcSimCycles / gcFrequency),
never a SIMD-active-only or wave-span-only time. Hardware-counter "MMAC busy" figures
(tmp/.../bandwidth_mmac.json) use different denominators and are NOT comparable.

Usage:
    python3 -I metrics.py            # print the table and verify the metrics stored in every summary.json
    python3 -I metrics.py 25.2231    # derive the metrics for an arbitrary kernel time in microseconds
"""
import json
from pathlib import Path
import sys

M, N, K = 12, 9216, 2048
PEAK_TFLOPS = 432.0
UMC_TB_S = 1.7996485          # configured effective external-memory bandwidth, TB/s
HERE = Path(__file__).resolve().parent

FLOPS = 2 * M * N * K
BYTES_D = 2 * (M * K + K * N + M * N)
ARITH_INTENSITY = FLOPS / BYTES_D
ROOFLINE_TFLOPS = min(PEAK_TFLOPS, ARITH_INTENSITY * UMC_TB_S)
ROOFLINE_US = FLOPS / (ROOFLINE_TFLOPS * 1e12) * 1e6
BANDWIDTH_ONLY_US = BYTES_D / (UMC_TB_S * 1e12) * 1e6


def derive(median_us):
    sec = median_us * 1e-6
    tflops = FLOPS / sec / 1e12
    bw = BYTES_D / sec / 1e12
    return dict(
        median_us=median_us,
        tflops=tflops,
        mmac_utilization_pct=100 * tflops / PEAK_TFLOPS,
        effective_bandwidth_TB_s=bw,
        bandwidth_achieved_pct=100 * bw / UMC_TB_S,
        roofline_attained_pct=100 * tflops / ROOFLINE_TFLOPS,
    )


def constants():
    return dict(M=M, N=N, K=K, useful_flops=FLOPS, logical_bytes_D=BYTES_D,
                A_bytes=2 * M * K, B_bytes=2 * K * N, C_bytes=2 * M * N,
                arithmetic_intensity_flop_per_byte=ARITH_INTENSITY,
                peak_tflops=PEAK_TFLOPS, configured_bandwidth_TB_s=UMC_TB_S,
                roofline_tflops=ROOFLINE_TFLOPS, roofline_time_us=ROOFLINE_US)


def summaries():
    return sorted(HERE.glob('*/runs/*/summary.json'))


def _walk(node, path=''):
    if isinstance(node, dict):
        if 'metrics' in node and 'median_us' in node:
            yield path, node
        for key, value in node.items():
            yield from _walk(value, f'{path}/{key}')
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f'{path}[{i}]')


def main(argv):
    if len(argv) == 2:
        print(json.dumps(derive(float(argv[1])), indent=1))
        return 0
    c = constants()
    print(f"shape {M}x{N}x{K}: useful FLOP {FLOPS:,}; D = {BYTES_D:,} B; "
          f"AI = {ARITH_INTENSITY:.4f} FLOP/B; roofline {ROOFLINE_TFLOPS:.4f} TFLOPS "
          f"({ROOFLINE_US:.3f} us); D / 1.7996485 TB/s = {BANDWIDTH_ONLY_US:.3f} us")
    bad = 0
    checked = 0
    for path in summaries():
        data = json.loads(path.read_text())
        for where, node in _walk(data):
            checked += 1
            want = derive(node['median_us'])
            have = node['metrics']
            ok = all(abs(have[k] - want[k]) <= 1e-9 * max(1.0, abs(want[k])) for k in want)
            rel = path.relative_to(HERE)
            print(f"{'OK ' if ok else 'BAD'} {rel}{where}: {want['median_us']:.6f} us  "
                  f"MMAC {want['mmac_utilization_pct']:.4f}%  {want['tflops']:.4f} TFLOPS  "
                  f"BW {want['effective_bandwidth_TB_s']:.4f} TB/s ({want['bandwidth_achieved_pct']:.2f}%)")
            bad += not ok
    print(f'{checked} metric blocks checked, {bad} mismatches')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
