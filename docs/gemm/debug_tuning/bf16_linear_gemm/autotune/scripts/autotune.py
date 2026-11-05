#!/usr/bin/env python3
"""Per-shape (operator + config) search on the PMD gfx946 simulator.

Kernel: gemm_a16w16_generic._gemm_bf16_generic_kernel (one kernel, features switched by options).
Measurement: tune_lib.run_case (1 warmup + timed dispatches, median gcSimCycles, sampled accuracy check).

Per (shape, layout):
  A1  tile screen        every feasible BMxBNxBK x 2 base recipes on a proxy problem
  A2  recipe screen      best tiles x ~11 recipes (plain/MLS, pingpong+async copy, WASP/WDRA, unroll, ...)
  B   fine tuning        coordinate descent over every knob (+ neighbouring tiles, split-K) from the top seeds
  C   validation         finalists re-measured on a larger problem (more TGs), winner chosen there

Proxy problem: M, N shrunk so the grid has ~proxy_tg thread groups for the candidate tile (the last
partial tile is kept, so M/N tails stay in), K capped at ~proxy_k (larger for shapes with few TGs).
Score = cycles / (2*M*N*K) of the measured problem (lower is better); comparable across tiles.

  autotune.py run    --work W [--layout TT] [--workers 51] ...     (resumable; Ctrl-C safe)
  autotune.py full   --work W                                        (winner re-run on the real full-size shape)
  autotune.py report --work W --out DIR                              (best_per_shape.json / .md)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_lib as T  # noqa: E402

SHAPES_JSON = T.REPO / 'docs/gemm/gemm_a16w16/shape/linear_bf16xbf16_to_bf16_shapes.json'
INF = float('inf')
STOP = threading.Event()
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) * 40 // 100)      # 40% of all logical CPUs (128 -> 51)


# ----------------------------------------------------------------------------- search space
def _pp(g, l=1, copy=True, pp=True):
    d = {}
    if copy:
        d['use_async_copy'] = True
    if pp:
        d['use_block_pingpong'] = True
        d['block_pingpong_async_global_prefetch'] = g
        d['block_pingpong_async_local_prefetch'] = l
    return d


def _wasp(parts, wdra=False):
    return {'WARP_SPECIALIZE': True, 'wasp_wdra': wdra, 'wasp_partition_warps': list(parts)}


# knob -> [(label, config patch)]; index 0 is the default (empty patch unless stated)
KNOBS = {
    'mls': [('off', {}), ('on', {'USE_MLS': True})],
    'pingpong': [('off', {}), ('async3/1', _pp(3, 1)), ('async4/1', _pp(4, 1)), ('async8/1', _pp(8, 1)),
                 ('async3/3', _pp(3, 3)), ('copy_only', {'use_async_copy': True}),
                 ('pp_only', {'use_block_pingpong': True})],
    'wasp': [('off', {}), ('1-4', _wasp([1, 4])), ('4-4', _wasp([4, 4])), ('4-4-4', _wasp([4, 4, 4])),
             ('2-4', _wasp([2, 4])), ('4-4+wdra', _wasp([4, 4], True)), ('4-4-4+wdra', _wasp([4, 4, 4], True)),
             ('1-4+wdra', _wasp([1, 4], True))],
    'unroll': [('1', {}), ('2', {'LOOP_UNROLL_FACTOR': 2}), ('4', {'LOOP_UNROLL_FACTOR': 4})],
    'warps': [('4', {'num_warps': 4}), ('8', {'num_warps': 8}), ('2', {'num_warps': 2})],
    'stages': [('2', {'num_stages': 2}), ('3', {'num_stages': 3}), ('4', {'num_stages': 4})],
    'mmac_layout': [('3', {'mmac_layout_force': 3}), ('2', {'mmac_layout_force': 2}), ('1', {'mmac_layout_force': 1}),
                    ('4', {'mmac_layout_force': 4}), ('auto', {'mmac_layout_force': -1})],
    'cluster': [('0', {}), ('1', {'enable_v_mmac_cluster': 1}), ('2', {'enable_v_mmac_cluster': 2})],
    'sched_hint': [('none', {}), ('local-prefetch', {'schedule_hint': 'local-prefetch'})],
    'sched_v2': [('on', {}), ('off', {'hcu_use_gfx946_sched_model_v2': False})],
    'max_ilp': [('off', {}), ('on', {'amdgpu_enable_max_ilp_scheduling_strategy': True})],
    'swizzle': [('off', {}), ('on', {'buffer_cache_swizzle': True})],
    'regpress': [('on', {}), ('off', {'misched_regpressure': False})],
    'mem_cluster': [('none', {}), ('32', {'max_memory_cluster_dwords': 32})],
    'empty_arrive': [('off', {}), ('on', {'empty_arrive_after_mmac': True})],
    'utc': [('off', {}), ('on', {'utc_warmup': True})],
    'group': [('1', {}), ('2', {'GROUP_SIZE_M': 2}), ('4', {'GROUP_SIZE_M': 4}), ('8', {'GROUP_SIZE_M': 8})],
    'ksplit': [(str(k), {'_ks': k}) for k in (1, 2, 3, 4, 6, 8, 12, 16)],
}
ORDER = list(KNOBS)
# coordinate-descent sweep order (structural features first, scheduling details later)
SWEEP = ['mls', 'pingpong', 'wasp', 'warps', 'stages', 'unroll', 'mmac_layout', 'cluster', 'sched_hint', 'sched_v2',
         'max_ilp', 'swizzle', 'regpress', 'mem_cluster', 'empty_arrive', 'group', 'ksplit', 'utc']

MAP_RECIPE = dict(pingpong='async3/1', warps='8', cluster='1', max_ilp='on', sched_v2='off', mem_cluster='32',
                  regpress='off')
RECIPES = {            # label -> knob labels (unlisted knobs = default)
    'plain': {},
    'mls': dict(mls='on'),
    'map': dict(MAP_RECIPE),
    'map_nw4': dict(MAP_RECIPE, warps='4'),
    'map_mls': dict(MAP_RECIPE, mls='on'),
    'map_swz': dict(MAP_RECIPE, swizzle='on'),
    'mls_nw8': dict(mls='on', warps='8'),
    'mls_unroll2': dict(mls='on', unroll='2'),
    'mls_st3': dict(mls='on', stages='3'),
    'wasp44': dict(mls='on', wasp='4-4'),
    'wasp14': dict(mls='on', wasp='1-4'),
    'wasp444': dict(mls='on', wasp='4-4-4'),
    'wdra44': dict(mls='on', wasp='4-4+wdra'),
}
A1_RECIPES = ['mls', 'map']


def state_of(labels: dict):
    idx = []
    for k in ORDER:
        want = labels.get(k)
        names = [n for n, _ in KNOBS[k]]
        idx.append(0 if want is None else names.index(want))
    return tuple(idx)


def labels_of(state):
    return {k: KNOBS[k][i][0] for k, i in zip(ORDER, state)}


def next_pow2(x):
    return 1 << max(0, (x - 1).bit_length())


def tiles_for(M, N, K):
    mcap, ncap = max(16, next_pow2(M)), max(32, next_pow2(N))
    out = []
    for bm, bn, bk in itertools.product((16, 32, 64, 128, 256), (32, 64, 128, 256), (32, 64, 128)):
        if bm > mcap or bn > ncap or K % bk or (bm + bn) * bk * 2 > 131072:
            continue
        if bm == 16 and M >= 512 or bm <= 32 and M >= 2048:
            continue
        out.append((bm, bn, bk))
    return out


class Problem:
    """Shape + layout + proxy/validation geometry; builds configs and measured dims."""

    def __init__(self, M, N, K, layout, proxy_tg, val_tg, proxy_k):
        self.M, self.N, self.K, self.layout = M, N, K, layout
        self.proxy_tg, self.val_tg = proxy_tg, val_tg
        tg_ref = math.ceil(M / 16) * math.ceil(N / 64)
        kcap = int(proxy_k * proxy_tg / min(proxy_tg, tg_ref)) // 256 * 256
        self.kcap = max(min(K, max(kcap, proxy_k)), 256)
        self.Kp = min(K, self.kcap)

    def dims(self, tile, tg):
        bm, bn, _ = tile
        tm, tn = math.ceil(self.M / bm), math.ceil(self.N / bn)
        tm_p = min(tm, tg)
        tn_p = min(tn, math.ceil(tg / tm_p))
        Mp = self.M if tm <= tm_p else (tm_p - 1) * bm + (self.M - (tm - 1) * bm)
        Np = self.N if tn <= tn_p else (tn_p - 1) * bn + (self.N - (tn - 1) * bn)
        return Mp, Np, self.Kp, tm_p * tn_p

    def build(self, tile, state, K):
        """Config dict for a problem with reduction length K, or None if the combination is invalid."""
        lab = labels_of(state)
        bm, bn, bk = tile
        if lab['wasp'] != 'off' and (lab['warps'] != '4' or lab['pingpong'] != 'off' or lab['unroll'] != '1'):
            return None
        cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, num_warps=4, num_stages=2)
        ks = 1
        for k, i in zip(ORDER, state):
            patch = dict(KNOBS[k][i][1])
            ks = patch.pop('_ks', ks)
            cfg.update(patch)
        if ks > 1:
            if lab['group'] != '1':
                return None
            sbs = -(-(-(-K // ks)) // bk) * bk
            if -(-K // sbs) != ks:
                return None
            cfg.update(NUM_KSPLIT=ks, SPLITK_BLOCK_SIZE=sbs)
        return cfg


# ----------------------------------------------------------------------------- evaluator
class Evaluator:
    def __init__(self, work: Path, workers: int, timeout: int, cpu_offset: int = 0):
        self.work, self.timeout = work, timeout
        (work / 'cases').mkdir(parents=True, exist_ok=True)
        self.log = work / 'results.jsonl'
        self.cache, self.inflight = {}, {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(workers)
        # cpu_offset: skip the first N CPUs so a second autotune can run next to a first one without sharing cores
        cpus = sorted(os.sched_getaffinity(0))[cpu_offset:]
        assert len(cpus) >= workers, f'cpu_offset {cpu_offset} + workers {workers} exceeds the available CPUs'
        self.cpus = queue.Queue()
        for c in cpus[1::3][:workers] if len(cpus) >= 3 * workers else cpus[:workers]:
            self.cpus.put(c)
        self.n_run = self.n_hit = 0
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r['key']] = r
                except Exception:
                    pass

    @staticmethod
    def key_of(M, N, K, layout, cfg, disp):
        import hashlib
        s = json.dumps([M, N, K, layout, cfg, disp], sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:16]

    def submit(self, M, N, K, layout, cfg, dispatches, keep_asm=False, timeout=None) -> Future:
        key = self.key_of(M, N, K, layout, cfg, dispatches)
        with self.lock:
            if key in self.cache:
                self.n_hit += 1
                f = Future()
                f.set_result(self.cache[key])
                return f
            if key in self.inflight:
                return self.inflight[key]
            case = dict(M=M, N=N, K=K, layout=layout, config=cfg, seed=29, dispatches=dispatches, keep_asm=keep_asm)
            f = self.pool.submit(self._run, key, case, timeout or self.timeout)
            self.inflight[key] = f
            return f

    def _run(self, key, case, timeout):
        if STOP.is_set():
            raise RuntimeError('stopped')
        cpu = self.cpus.get()
        try:
            res = T.run_case(case, self.work / 'cases' / key, timeout=timeout, cpu=cpu)
        finally:
            self.cpus.put(cpu)
        acc = res.get('accuracy') or {}
        row = dict(key=key, M=case['M'], N=case['N'], K=case['K'], layout=case['layout'], config=case['config'],
                   dispatches=case['dispatches'], status=res['status'], median_cycles=res.get('median_cycles'),
                   median_us=res.get('median_us'), tflops=res.get('tflops'), mmac_util=res.get('mmac_util'),
                   active_cus=res.get('active_cus'), resources=res.get('resources'),
                   mismatches=acc.get('mismatches'), wall_s=round(res['worker_wall_s'], 1),
                   error=(res.get('worker_log_tail') or res.get('error') or '')[-400:] if res['status'] == 'ERROR' else '')
        if STOP.is_set():
            raise RuntimeError('stopped')
        with self.lock:
            self.cache[key] = row
            self.n_run += 1
            self.inflight.pop(key, None)
            with self.log.open('a') as f:
                f.write(json.dumps(row) + '\n')
        return row


def score(row, dims):
    if not row or row['status'] != 'PASS' or not row.get('median_cycles'):
        return INF
    Mp, Np, Kp = dims[:3]
    return row['median_cycles'] / (2.0 * Mp * Np * Kp)


# ----------------------------------------------------------------------------- per-shape search
class ShapeTuner:
    def __init__(self, ev: Evaluator, prob: Problem, a):
        self.ev, self.p, self.a = ev, prob, a
        self.tiles = tiles_for(prob.M, prob.N, prob.K)
        self.hist = {}                                        # (tile, state) -> (score, row, dims)
        self.name = f'{prob.M}x{prob.N}x{prob.K}_{prob.layout}'

    def log(self, msg):
        print(f'[{time.strftime("%m-%d %H:%M:%S")}] {self.name}: {msg}', flush=True)

    def measure(self, items, tg, dispatches, keep_asm=False, timeout=None):
        """items: [(tile, state)] -> list of (score, row, dims) (INF for invalid)."""
        futs = []
        for tile, state in items:
            cfg = self.p.build(tile, state, self.p.Kp)
            dims = self.p.dims(tile, tg)
            futs.append((None, dims) if cfg is None else
                        (self.ev.submit(dims[0], dims[1], dims[2], self.p.layout, cfg, dispatches, keep_asm, timeout), dims))
        out = []
        for f, dims in futs:
            row = f.result() if f is not None else None
            out.append((score(row, dims), row, dims))
        return out

    def proxy(self, items):
        items = [it for it in dict.fromkeys(items) if it not in self.hist]
        for it, r in zip(items, self.measure(items, self.p.proxy_tg, self.a.dispatches)):
            self.hist[it] = r
        return items

    def best_of(self, items):
        return min(items, key=lambda it: self.hist[it][0]) if items else None

    # -- stages
    def screen(self):
        recs = [state_of(RECIPES[r]) for r in A1_RECIPES]
        tiles = self.tiles
        if self.a.max_tiles and len(tiles) > self.a.max_tiles:
            tiles = sorted(tiles, key=lambda t: (abs(math.log2(t[0] * t[1] * t[2] / 262144)), t))[:self.a.max_tiles]
        self.proxy([(t, s) for t in tiles for s in recs])
        per_tile = {}
        for (t, s), (sc, _, _) in self.hist.items():
            per_tile[t] = min(per_tile.get(t, INF), sc)
        top = [t for t, sc in sorted(per_tile.items(), key=lambda kv: kv[1]) if sc < INF][:self.a.topk_tiles]
        self.log(f'A1 tiles={len(tiles)} top={top}')
        names = list(RECIPES)[:self.a.n_recipes] if self.a.n_recipes else list(RECIPES)
        self.proxy([(t, state_of(RECIPES[r])) for t in top for r in names])
        seeds = self.rank(self.a.n_seeds)
        self.log(f'A2 evals={len(self.hist)} seeds={[(t, self.hist[(t, s)][0] * 1e3) for t, s in seeds]}')
        return seeds

    def rank(self, n):
        """best state per tile, then the n best tiles (diversity over near-duplicates)."""
        per = {}
        for (t, s), (sc, _, _) in self.hist.items():
            if sc < INF and (t not in per or sc < self.hist[(t, per[t])][0]):
                per[t] = s
        return [(t, per[t]) for t in sorted(per, key=lambda t: self.hist[(t, per[t])][0])[:n]]

    def neighbors(self, cur, knob):
        tile, state = cur
        if knob == 'tile':
            bm, bn, bk = tile
            alts = [(bm * 2, bn, bk), (bm // 2, bn, bk), (bm, bn * 2, bk), (bm, bn // 2, bk), (bm, bn, bk * 2), (bm, bn, bk // 2)]
            return [(t, state) for t in alts if t in self.tiles]
        i = ORDER.index(knob)
        alts = []
        for v in range(len(KNOBS[knob])):
            if v == state[i]:
                continue
            if knob == 'ksplit':
                tm_tn = math.ceil(self.p.M / tile[0]) * math.ceil(self.p.N / tile[1])
                if tm_tn >= 96 or tm_tn * int(KNOBS[knob][v][0]) > 384:
                    continue
            if knob == 'group' and (math.ceil(self.p.M / tile[0]) < 2 or math.ceil(self.p.N / tile[1]) < 2):
                continue
            alts.append((tile, state[:i] + (v,) + state[i + 1:]))
        return alts

    def refine(self, seed, passes):
        cur = seed
        self.proxy([cur])
        for ps in range(passes):
            moved = False
            for knob in ['tile'] + SWEEP:
                alts = self.proxy(self.neighbors(cur, knob))
                alts = [x for x in self.neighbors(cur, knob) if x in self.hist]
                b = self.best_of(alts)
                if b is not None and self.hist[b][0] < self.hist[cur][0] * (1 - self.a.thr):
                    cur, moved = b, True
            self.log(f'B pass {ps + 1} cur={cur[0]} {self.hist[cur][0] * 1e3:.4f} {labels_of(cur[1])}')
            if not moved or STOP.is_set():
                break
        return cur

    def run(self, seeds_in=None):
        t0 = time.time()
        if seeds_in:
            seeds = [(tuple(s['tile']), state_of(s['labels'])) for s in seeds_in if tuple(s['tile']) in self.tiles]
            self.proxy(seeds)
            seeds = [s for s in seeds if self.hist[s][0] < INF] or self.screen()
        else:
            seeds = self.screen()
        with ThreadPoolExecutor(max(1, len(seeds))) as ex:
            refined = list(ex.map(lambda s: self.refine(s, self.a.passes), seeds))
        finals = self.rank(self.a.n_final)
        vitems = [(t, s) for t, s in finals]
        vres = self.measure(vitems, self.p.val_tg, self.a.val_dispatches, keep_asm=True, timeout=self.a.val_timeout)
        out = []
        for (t, s), (vs, vrow, vdims) in zip(vitems, vres):
            psc, prow, pdims = self.hist[(t, s)]
            out.append(dict(tile=list(t), labels=labels_of(s), config=self.p.build(t, s, self.p.K),
                            proxy=self.pack(prow, pdims, psc), val=self.pack(vrow, vdims, vs)))
        out.sort(key=lambda d: (d['val']['score'] if d['val']['score'] is not None else INF,
                                d['proxy']['score'] or INF))
        best = out[0] if out else None
        flops_real = 2 * self.p.M * self.p.N * self.p.K
        if best and best['val']['cycles']:
            v = best['val']
            f = flops_real / (2 * v['dims'][0] * v['dims'][1] * v['dims'][2])
            best['est_real'] = dict(cycles=v['cycles'] * f, us=v['us'] * f, tflops=v['tflops'], note=(
                'exact' if v['dims'][:3] == [self.p.M, self.p.N, self.p.K] else 'linear extrapolation from measured dims'))
        res = dict(M=self.p.M, N=self.p.N, K=self.p.K, layout=self.p.layout, kcap=self.p.Kp, proxy_tg=self.p.proxy_tg,
                   val_tg=self.p.val_tg, n_tiles=len(self.tiles), n_evals=len(self.hist), best=best, finalists=out,
                   wall_s=round(time.time() - t0), finished=time.strftime('%F %T'))
        self.log(f'DONE best={best and (best["tile"], best["val"]["score"])} evals={len(self.hist)} {res["wall_s"]}s')
        return res

    @staticmethod
    def pack(row, dims, sc):
        if not row or sc == INF:
            return dict(status=row['status'] if row else 'INVALID', score=None, cycles=None, us=None, tflops=None,
                        mmac_util=None, dims=list(dims), tg=dims[3] if len(dims) > 3 else None)
        return dict(status=row['status'], score=sc, cycles=row['median_cycles'], us=row['median_us'],
                    tflops=row['tflops'], mmac_util=row['mmac_util'], dims=list(dims[:3]), tg=dims[3],
                    shared=(row.get('resources') or {}).get('shared'), n_regs=(row.get('resources') or {}).get('n_regs'),
                    n_spills=(row.get('resources') or {}).get('n_spills'))


# ----------------------------------------------------------------------------- driver
def install_signals():
    def stop(*_):
        STOP.set()
        T.kill_active()
        print('stopped; finished shapes and all simulations are kept, rerun the same command to resume', flush=True)
        os._exit(130)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def cmd_full(a):
    """Stage D: re-measure each shape's winner on the REAL M x N x K (no proxy). Falls back to the next finalist
    if the winner fails accuracy or errors. Shapes whose validation already ran on the full dims are reused."""
    work = Path(a.work).resolve()
    files = sorted(work.glob('shapes/*.json'))
    docs = [(f, json.loads(f.read_text())) for f in files]
    docs = [(f, d) for f, d in docs if not d.get('full') and d.get('finalists')]
    docs.sort(key=lambda fd: math.ceil(fd[1]['M'] / 32) * math.ceil(fd[1]['N'] / 64) * fd[1]['K'])
    ev = Evaluator(work, a.workers, a.full_timeout, a.cpu_offset)
    install_signals()
    print(f'{len(docs)} shapes need a full-size run, workers {a.workers}, parallel shapes {a.shape_par}', flush=True)
    t0 = time.time()
    done = [0]

    def status():
        while not STOP.is_set():
            T.dump(work / 'status_full.json', dict(shapes_todo=len(docs), shapes_done=done[0], sims_run=ev.n_run,
                                                   inflight=len(ev.inflight),
                                                   elapsed_h=round((time.time() - t0) / 3600, 2), updated=time.strftime('%F %T')))
            time.sleep(30)
    threading.Thread(target=status, daemon=True).start()

    def one(fd):
        f, d = fd
        M, N, K, lay = d['M'], d['N'], d['K'], d['layout']
        try:
            for rank, fin in enumerate(d['finalists']):
                cfg = fin.get('config')
                if not cfg:
                    continue
                v = fin['val']
                if v['status'] == 'PASS' and v['dims'][:3] == [M, N, K]:       # validation already ran full size
                    full = dict(v, source='validation run (already full size)')
                else:
                    print(f'[{time.strftime("%m-%d %H:%M:%S")}] {M}x{N}x{K}: full run, finalist #{rank + 1} '
                          f'tile={fin["tile"]} grid={T.num_tg(M, N, cfg)}', flush=True)
                    row = ev.submit(M, N, K, lay, cfg, a.full_dispatches, False, a.full_timeout).result()
                    if row['status'] != 'PASS' or not row.get('median_cycles'):
                        print(f'  {M}x{N}x{K}: finalist #{rank + 1} {row["status"]} {row.get("error", "")[-200:]}', flush=True)
                        continue
                    full = ShapeTuner.pack(row, (M, N, K, T.num_tg(M, N, cfg)), row['median_cycles'] / (2.0 * M * N * K))
                    full['source'] = f'full-size run, {a.full_dispatches} dispatches (1 warmup + timed median)'
                d['full'] = full
                d['best'] = dict(fin, est_real=None)
                d['best_is_proxy_winner'] = rank == 0
                break
            else:
                d['full'] = dict(status='NO_VALID_FINALIST')
            T.dump(f, d)
            done[0] += 1
            print(f'[{time.strftime("%m-%d %H:%M:%S")}] {M}x{N}x{K}: full {d["full"].get("cycles")} cycles '
                  f'{d["full"].get("us") and round(d["full"]["us"], 1)} us', flush=True)
        except Exception:
            print(f'FAILED full {M}x{N}x{K}\n{traceback.format_exc()}', flush=True)

    with ThreadPoolExecutor(a.shape_par) as ex:
        list(ex.map(one, docs))
    STOP.set()
    print(f'full runs done in {(time.time() - t0) / 3600:.2f} h', flush=True)
    os._exit(0)


def cmd_run(a):
    work = Path(a.work).resolve()
    (work / 'shapes').mkdir(parents=True, exist_ok=True)
    shapes = [tuple(s) for s in json.loads(Path(a.shapes_json).read_text())['shapes']]
    if a.only:
        want = {tuple(int(x) for x in s.split('x')) for s in a.only.split(',')}
        shapes = [s for s in shapes if s in want]
    if a.idx:
        lo, hi = (int(x) for x in a.idx.split(':'))
        shapes = shapes[lo:hi]
    if a.order == 'cost':
        shapes.sort(key=lambda s: math.ceil(s[0] / 32) * math.ceil(s[1] / 64) * s[2])
    if a.priority_json:        # these shapes first (cost order inside each group), the rest afterwards
        prio = {tuple(x) for x in json.loads(Path(a.priority_json).read_text())['shapes']}
        shapes.sort(key=lambda s: s not in prio)          # stable sort
        print(f'{sum(s in prio for s in shapes)} priority shapes run first', flush=True)
    seeds = {}
    if a.seed_from:
        for f in Path(a.seed_from).glob('shapes/*.json'):
            d = json.loads(f.read_text())
            if d.get('finalists'):
                seeds[(d['M'], d['N'], d['K'])] = d['finalists']
    ev = Evaluator(work, a.workers, a.timeout, a.cpu_offset)

    install_signals()

    todo = [s for s in shapes if not (work / 'shapes' / f'{s[0]}x{s[1]}x{s[2]}_{a.layout}.json').exists()]
    print(f'{len(shapes)} shapes, {len(todo)} to do, layout {a.layout}, workers {a.workers}, shape-par {a.shape_par}', flush=True)
    done = [len(shapes) - len(todo)]
    t0 = time.time()

    def status():
        while not STOP.is_set():
            T.dump(work / 'status.json', dict(layout=a.layout, shapes_total=len(shapes), shapes_done=done[0],
                                              sims_run=ev.n_run, cache_hits=ev.n_hit, inflight=len(ev.inflight),
                                              elapsed_h=round((time.time() - t0) / 3600, 2), updated=time.strftime('%F %T')))
            time.sleep(30)
    threading.Thread(target=status, daemon=True).start()

    def one(s):
        M, N, K = s
        try:
            prob = Problem(M, N, K, a.layout, a.proxy_tg, a.val_tg, a.proxy_k)
            res = ShapeTuner(ev, prob, a).run(seeds.get(s))
            T.dump(work / 'shapes' / f'{M}x{N}x{K}_{a.layout}.json', res)
            done[0] += 1
        except Exception:
            print(f'FAILED {s}\n{traceback.format_exc()}', flush=True)

    with ThreadPoolExecutor(a.shape_par) as ex:
        list(ex.map(one, todo))
    STOP.set()
    print(f'all done in {(time.time() - t0) / 3600:.2f} h; sims run {ev.n_run}, cache hits {ev.n_hit}', flush=True)
    os._exit(0)


def short(lab):
    keys = ('mls', 'pingpong', 'wasp', 'warps', 'stages', 'unroll', 'ksplit', 'group', 'mmac_layout', 'cluster')
    dflt = {k: KNOBS[k][0][0] for k in KNOBS}
    return ' '.join(f'{k}={lab[k]}' for k in keys if lab[k] != dflt[k] or k in ('mls', 'warps', 'stages'))


def cmd_report(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(work.glob('shapes/*.json')):
        d = json.loads(f.read_text())
        rows.append(d)
    rows.sort(key=lambda d: (d['layout'], d['M'], d['N'], d['K']))
    T.dump(out / 'best_per_shape.json', dict(
        generated=time.strftime('%F %T'),
        kernel='gemm_a16w16_generic._gemm_bf16_generic_kernel (+ _gemm_splitk_reduce_kernel when NUM_KSPLIT>1)',
        note=('config = kernel constexpr + launch options for the REAL shape. `full` (when present) is the winner '
              'measured on the real M x N x K; otherwise best.val was measured on a reduced problem (best.est_real '
              'is a linear extrapolation).'),
        shapes=rows))
    md = ['# 每个 shape 的最优 config (PMD gfx946 仿真)', '',
          '| M | N | K | 布局 | tile (BM×BN×BK) | 主要选项 | 实测 dims (M×N×K, TG) | cycles | us | TFLOPS | MMAC 利用率 | 数据来源 / 整 shape 估算 us |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for d in rows:
        b = d.get('best')
        v = d.get('full') if d.get('full', {}).get('cycles') else (b or {}).get('val')
        if not b or not v or not v.get('cycles'):
            md.append(f'| {d["M"]} | {d["N"]} | {d["K"]} | {d["layout"]} | 无有效结果 | | | | | | | |')
            continue
        est = '实测(完整 shape)' if d.get('full', {}).get('cycles') else \
            f'{b["est_real"]["us"]:.0f} (外推)' if b.get('est_real') else ''
        md.append(f'| {d["M"]} | {d["N"]} | {d["K"]} | {d["layout"]} | {"×".join(map(str, b["tile"]))} | {short(b["labels"])} | '
                  f'{"×".join(map(str, v["dims"]))}, {v["tg"]} | {v["cycles"]:.0f} | {v["us"]:.1f} | {v["tflops"]:.2f} | '
                  f'{v["mmac_util"] * 100:.1f}% | {est} |')
    (out / 'best_per_shape.md').write_text('\n'.join(md) + '\n')
    print(f'{len(rows)} shape results -> {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--work', required=True)
    r.add_argument('--layout', default='TT', choices=['TT', 'TN'])
    r.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                   help='concurrent simulator processes (default: 40%% of all logical CPUs)')
    r.add_argument('--shape-par', type=int, default=10, help='shapes searched concurrently (feeds the workers)')
    r.add_argument('--shapes-json', default=str(SHAPES_JSON), help='shape list ({"shapes": [[M, N, K], ...]})')
    r.add_argument('--cpu-offset', type=int, default=0, help='pin workers to CPUs starting at this index')
    r.add_argument('--priority-json', default='', help='shape list ({"shapes": ...}) that is searched before all other shapes')
    r.add_argument('--proxy-tg', type=int, default=96)
    r.add_argument('--val-tg', type=int, default=480)
    r.add_argument('--proxy-k', type=int, default=2048)
    r.add_argument('--dispatches', type=int, default=3, help='1 warmup + (n-1) timed, screening/tuning')
    r.add_argument('--val-dispatches', type=int, default=4)
    r.add_argument('--timeout', type=int, default=3 * 3600)
    r.add_argument('--val-timeout', type=int, default=8 * 3600)
    r.add_argument('--topk-tiles', type=int, default=5)
    r.add_argument('--max-tiles', type=int, default=0, help='limit A1 tile list (smoke tests)')
    r.add_argument('--n-recipes', type=int, default=0, help='limit A2 recipes (smoke tests)')
    r.add_argument('--n-seeds', type=int, default=3)
    r.add_argument('--n-final', type=int, default=3)
    r.add_argument('--passes', type=int, default=3, help='coordinate-descent passes')
    r.add_argument('--thr', type=float, default=0.01, help='min relative gain to accept a move')
    r.add_argument('--only', default='', help='comma list of MxNxK')
    r.add_argument('--idx', default='', help='slice a:b of the shape list')
    r.add_argument('--order', default='cost', choices=['cost', 'file'])
    r.add_argument('--seed-from', default='', help='work dir of a finished run (other layout) to seed from')
    f = sub.add_parser('full', help='stage D: winner on the real full-size shape')
    f.add_argument('--work', required=True)
    f.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                   help='concurrent simulator processes (default: 40%% of all logical CPUs)')
    f.add_argument('--shape-par', type=int, default=DEFAULT_WORKERS, help='shapes run concurrently (one simulation each)')
    f.add_argument('--cpu-offset', type=int, default=0, help='pin workers to CPUs starting at this index')
    f.add_argument('--full-dispatches', type=int, default=2, help='1 warmup + (n-1) timed')
    f.add_argument('--full-timeout', type=int, default=7 * 24 * 3600)
    p = sub.add_parser('report')
    p.add_argument('--work', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    {'run': cmd_run, 'full': cmd_full, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
