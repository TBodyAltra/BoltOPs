#!/usr/bin/env python3
"""Reproduce the final R016 implementation in fresh isolated PMD processes.
Use --compare for a baseline pair. Each case refuses to overwrite prior runs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import bench
ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True)
    ap.add_argument('--compare',action='store_true')
    ap.add_argument('--seed',type=int,default=29)
    ap.add_argument('--dispatches',type=int,default=5,help='one warmup plus the remaining timed dispatches')
    ap.add_argument('--small-k',type=int)
    ap.add_argument('--rotate-b',action='store_true',help='unique B allocation and data for each dispatch; no explicit cache flush')
    ap.add_argument('--poison-pad',action='store_true',help='NaN padded inputs; changes strides, correctness only')
    args=ap.parse_args()
    if args.dispatches<2:
        ap.error('at least two dispatches required')
    if args.rotate_b and args.poison_pad:
        ap.error('rotate-b and poison-pad are distinct protocols')
    cfg=json.loads((ROOT/'best_config.json').read_text())
    k=args.small_k or 2048
    if k<=0 or k%cfg['BLOCK_SIZE_K']:
        ap.error('K must be a positive multiple of BLOCK_SIZE_K')
    base=dict(K=k,seed=args.seed,dispatches=args.dispatches,rotate_b=args.rotate_b,
              poison_pad=args.poison_pad,check_each_dispatch=not args.poison_pad)
    best=dict(base,operator=str(ROOT/'r016_kernel.py'),config=cfg,
              checked_entry=not args.small_k and not args.poison_pad)
    cases=[(args.name+'_best',best)]
    if args.compare:
        cases.insert(0,(args.name+'_baseline',base))
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        rows=list(pool.map(lambda item:bench.run(*item),cases))
    if any(row['status']!='PASS' for row in rows):
        sys.exit(1)
    if len(rows)==2:
        b,o=rows
        print(json.dumps(dict(baseline_us=b['median_us'],optimized_us=o['median_us'],
            reduction_pct=100*(1-o['median_us']/b['median_us']),speedup=b['median_us']/o['median_us']),indent=1))

if __name__=='__main__':
    main()
