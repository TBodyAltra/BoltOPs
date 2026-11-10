#!/usr/bin/env python3
"""Full-output generality check of gemm_a16w16_generic._gemm_bf16_generic_kernel on PMD.

Every case compares EVERY output element against a CPU FP32 reference rounded to
BF16 (full_check) and checks poison guards/finite values. Small shapes only.

  python verify_generality.py <out_dir> [workers]
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_lib as T  # noqa: E402

B = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2)


def cfg(**kw):
    return {**B, **kw}


MLS = dict(USE_MLS=True)
WASP = dict(WARP_SPECIALIZE=True, wasp_wdra=False)
CASES = []


def add(name, M, N, K, layout, **kw):
    CASES.append((name, dict(M=M, N=N, K=K, layout=layout, config=cfg(**kw), full_check=True, dispatches=2)))


for lay in ('TT', 'TN'):
    for tag, extra in (('plain', {}), ('mls', MLS)):
        add(f'{tag}_aligned_{lay}', 64, 128, 1024, lay, **extra)
        add(f'{tag}_Mtail_{lay}', 50, 128, 1024, lay, **extra)
        add(f'{tag}_Ntail_{lay}', 64, 100, 1024, lay, **extra)
        add(f'{tag}_MNtail_multiTG_{lay}', 101, 197, 1024, lay, **extra)
        add(f'{tag}_Mlt_BM_{lay}', 7, 128, 1024, lay, **extra)
        add(f'{tag}_K_odd_blocks_{lay}', 64, 128, 128 * 17, lay, **extra)
add('plain_Ktail_TT', 64, 128, 1000, 'TT')
add('plain_Ktail_TN', 64, 128, 1000, 'TN')
add('plain_MNKtail_TT', 45, 99, 1000, 'TT')
add('plain_MNKtail_TN', 45, 99, 1000, 'TN')
add('mls_Ktail_TT_expected_unsupported', 64, 128, 1000, 'TT', **MLS)   # expected FAIL -> rule: no MLS if K%BK
# tile shapes
add('mls_BM16', 33, 128, 1024, 'TT', BLOCK_SIZE_M=16, **MLS)
add('mls_BN32', 64, 70, 1024, 'TT', BLOCK_SIZE_N=32, **MLS)
add('mls_BM64_BN128_BK64_w8', 100, 300, 1024, 'TT', BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, num_warps=8, **MLS)
add('mls_BM128_BN128_BK64', 200, 300, 1024, 'TN', BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, **MLS)
add('plain_BM64_BN128_BK64_w8_tail', 100, 300, 1000, 'TT', BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, num_warps=8)
add('mls_group2', 160, 256, 1024, 'TT', GROUP_SIZE_M=2, **MLS)
add('plain_group4_tail', 150, 200, 1000, 'TT', GROUP_SIZE_M=4)
# pipeline features
for lay in ('TT', 'TN'):
    add(f'mls_stages3_{lay}', 70, 150, 1024, lay, num_stages=3, **MLS)
    add(f'mls_unroll2_tail_{lay}', 70, 150, 1024, lay, LOOP_UNROLL_FACTOR=2, **MLS)
    add(f'mls_unroll4_tail_{lay}', 70, 150, 1024, lay, LOOP_UNROLL_FACTOR=4, **MLS)
    add(f'plain_unroll2_Ktail_{lay}', 70, 150, 1000, lay, LOOP_UNROLL_FACTOR=2)
    add(f'wasp14_mls_tail_{lay}', 70, 150, 1024, lay, wasp_partition_warps=[1, 4], **MLS, **WASP)
    add(f'wasp44_mls_tail_{lay}', 70, 150, 1024, lay, wasp_partition_warps=[4, 4], **MLS, **WASP)
    add(f'wdra44_mls_tail_{lay}', 70, 150, 1024, lay, wasp_partition_warps=[4, 4], **MLS,
        **{**WASP, 'wasp_wdra': True})
    add(f'wasp44_unroll2_mls_{lay}', 70, 150, 1024, lay, wasp_partition_warps=[4, 4], LOOP_UNROLL_FACTOR=2, **MLS, **WASP)
    add(f'wasp44_plain_Ktail_{lay}', 70, 150, 1000, lay, wasp_partition_warps=[4, 4], **WASP)
    add(f'wasp444_mls_{lay}', 70, 150, 1024, lay, wasp_partition_warps=[4, 4, 4], **MLS, **WASP)
    add(f'wdra444_mls_{lay}', 70, 150, 1024, lay, wasp_partition_warps=[4, 4, 4], **MLS,
        **{**WASP, 'wasp_wdra': True})
    add(f'async_copy_plain_tail_{lay}', 70, 150, 1000, lay, use_async_copy=True)
    add(f'pingpong_mls_tail_{lay}', 70, 150, 1024, lay, use_block_pingpong=True, **MLS)
    add(f'utc_warmup_mls_{lay}', 70, 150, 1024, lay, utc_warmup=True, **MLS)
    for l in (1, 2, 3, 4):
        add(f'mls_layout{l}_{lay}', 70, 150, 1024, lay, mmac_layout_force=l, **MLS)


# split-K (FP32 partials + reduce kernel); main + reduce dispatches are summed by tune_lib
for lay in ('TT', 'TN'):
    for ks, sbs in ((2, 512), (4, 256), (3, 384)):
        add(f'splitk{ks}_plain_Mtail_{lay}', 50, 100, 1024 if ks != 3 else 1152, lay, NUM_KSPLIT=ks, SPLITK_BLOCK_SIZE=sbs, BLOCK_SIZE_K=128 if sbs % 128 == 0 else 64)
        add(f'splitk{ks}_mls_Mtail_{lay}', 50, 100, 1024 if ks != 3 else 1152, lay, NUM_KSPLIT=ks, SPLITK_BLOCK_SIZE=sbs, BLOCK_SIZE_K=128 if sbs % 128 == 0 else 64, **MLS)
    add(f'splitk4_wasp44_mls_{lay}', 70, 150, 1024, lay, NUM_KSPLIT=4, SPLITK_BLOCK_SIZE=256, wasp_partition_warps=[4, 4], **MLS, **WASP)
    add(f'splitk8_bm16_mls_{lay}', 12, 130, 1024, lay, NUM_KSPLIT=8, SPLITK_BLOCK_SIZE=128, BLOCK_SIZE_M=16, **MLS)


# BM=16 / small tiles (used for M<=16 shapes; not in the BM>=32 map) and BK=32 tiles
for lay in ('TT', 'TN'):
    add(f'bm16_plain_tail_{lay}', 13, 100, 1024, lay, BLOCK_SIZE_M=16)
    add(f'bm16_mls_tail_{lay}', 13, 100, 1024, lay, BLOCK_SIZE_M=16, **MLS)
    add(f'bm16_bn32_nw2_mls_{lay}', 5, 70, 1024, lay, BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, num_warps=2, **MLS)
    add(f'bm16_bn128_nw8_mls_{lay}', 9, 200, 1024, lay, BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, num_warps=8, **MLS)
    add(f'bm16_wasp44_mls_{lay}', 13, 100, 1024, lay, BLOCK_SIZE_M=16, wasp_partition_warps=[4, 4], **MLS, **WASP)
    add(f'bk32_mls_tail_{lay}', 70, 150, 1024, lay, BLOCK_SIZE_K=32, **MLS)
    add(f'bk32_plain_tail_{lay}', 70, 150, 1000, lay, BLOCK_SIZE_K=32)
    add(f'bm256_bn128_bk64_mls_{lay}', 300, 200, 1024, lay, BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
        num_warps=8, **MLS)
    add(f'splitk_bm16_plain_{lay}', 6, 70, 1024, lay, BLOCK_SIZE_M=16, NUM_KSPLIT=4, SPLITK_BLOCK_SIZE=256)


def main():
    out = Path(sys.argv[1]).resolve()
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    only = sys.argv[3] if len(sys.argv) > 3 else ''
    cpus = sorted(os.sched_getaffinity(0))
    out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in only.split(',') if k]
    jobs = [(i, n, c) for i, (n, c) in enumerate(CASES) if not keys or any(k in n for k in keys)]

    def one(job):
        i, n, c = job
        res = T.run_case(c, out / n, timeout=1800, cpu=cpus[(i * 3 + 1) % len(cpus)])
        acc = res.get('accuracy', {})
        line = dict(name=n, status=res['status'], mismatches=acc.get('mismatches'), points=acc.get('sampled_points'),
                    finite=acc.get('finite'), guard=acc.get('guard'), shared=(res.get('resources') or {}).get('shared'),
                    n_spills=(res.get('resources') or {}).get('n_spills'),
                    err=(res.get('worker_log_tail') or res.get('error') or '')[-300:] if res['status'] == 'ERROR' else '')
        print(json.dumps(line, ensure_ascii=False), flush=True)
        return line

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs))
    T.dump(out / 'verify_summary.json', rows)
    bad = [r for r in rows if r['status'] != 'PASS']
    print(f'\n{len(rows) - len(bad)}/{len(rows)} PASS')
    for r in bad:
        print('NOT PASS:', r['name'], r['status'], r['mismatches'], r['err'][-150:].replace('\n', ' '))


if __name__ == '__main__':
    main()
