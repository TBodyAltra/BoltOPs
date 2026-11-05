#!/usr/bin/env python3
"""End-to-end check of gemm_a16w16_generic._gemm_bf16_generic_kernel (dynamic M/N/K,
K tails, split-K tails, empty splits, bias, matrix_store boundary_check) on the PMD
simulator, using the same environment as the autotune workers (tune_lib.env_for).

    python verify_dynamic_mnk.py [--only SUBSTR] [--list] [--diag]   # launcher, re-runs itself in the PMD env
    python verify_dynamic_mnk.py worker [--only SUBSTR]     # (internal)

Poisoning makes boundary bugs visible:
  * A/B are views into larger buffers whose slack is NaN, so any unzeroed read
    past K / M / N turns the result into NaN.
  * C is a view [:M, :N] of a NaN-filled buffer with padding of one tile on every
    far edge (rows and columns); any out-of-bounds write (e.g. an unchecked
    matrix_store) changes the padding.
  * The split-K partial buffer is NaN-initialised, so an empty split that does not
    store zeros is detected.
Prints one `RESULT {json}` line per case and a final `SUMMARY` line.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cdiv(a, b):
    return -(-a // b)


# name, M, N, K, BM, BN, BK, KSPLIT, layout, mls, matrix_store, bias, dtype
def _cases(diag=False):
    C = []

    def add(tag, M, N, K, ks=1, layout='TT', mls=False, ms=False, bias=False, dtype='bf16',
            bm=16, bn=32, bk=128, opts=None):
        name = f"{tag}|{'mls' if mls else 'ptr'}{'+mstore' if ms else ''}|{layout}|{dtype}|ks{ks}|{'bias' if bias else 'nobias'}|{M}x{N}x{K}|{bm}x{bn}x{bk}"
        C.append(dict(name=name, M=M, N=N, K=K, bm=bm, bn=bn, bk=bk, ks=ks, layout=layout,
                      mls=mls, ms=ms, bias=bias, dtype=dtype, opts=opts or {}))

    for mls in (False, True):
        add('aligned', 16, 64, 256, mls=mls)
        add('unaligned', 13, 70, 300, mls=mls, bias=True)
        add('splitk', 13, 70, 300, ks=3, mls=mls, bias=True)
        add('empty_split', 5, 40, 300, ks=4, mls=mls)
        add('tiny', 1, 33, 50, mls=mls, bias=True)
    add('unaligned', 13, 70, 300, layout='TN', mls=True, bias=True)
    add('splitk', 13, 70, 300, ks=3, layout='TN', mls=True, bias=True)
    # ---- tl.matrix_store (fp16 output: the BF16 matrix_store needs the +bf16mstore build; tl.matrix_store needs
    # mmac_layout_force=3). Launch options of the tuned fp16 MLS+matrix_store recipe (C042 in
    # fp16-single-tg-all-block-best-configs): 32x256x64, 4 warps, 2 stages. Small tiles such as 16x32x128 have
    # no MLS tile candidate -> compile-time assertion.
    ms_opts = dict(num_warps=4, num_stages=2, mmac_layout_force=3, enable_v_mmac_cluster=1,
                   optimize_epilogue=True, empty_arrive_after_mmac=False)
    ms = dict(mls=True, ms=True, dtype='fp16', bm=32, bn=256, bk=64, opts=ms_opts)
    ms_load = dict(ms, ms=False)                            # same tile/options, plain tl.store epilogue
    # contract: N % BLOCK_SIZE_N == 0 -> only M / K tails
    add('aligned', 32, 256, 256, **ms)
    add('m_tail', 13, 256, 256, **ms)                       # rows >= M must not be written
    add('m1', 1, 256, 256, **ms)                            # single valid row
    add('m_k_tail', 45, 256, 300, **ms)                     # M tail + K tail
    add('m_tail', 13, 256, 256, layout='TN', **ms)
    # bias disables matrix_store inside the kernel (pointer-store epilogue), any M/N/K tail is fine
    add('bias_fallback', 32, 256, 256, bias=True, **ms)
    add('bias_fallback_mnk_tail', 45, 300, 300, bias=True, **ms)
    add('bias_fallback_tiny', 1, 33, 50, bias=True, **ms)
    add('bias_loadonly', 32, 256, 256, bias=True, **ms_load)
    # heuristic _USE_MSTORE: a partial N tile (N % BN != 0) quietly takes the pointer store instead of panicking
    add('n_tail_fallback', 32, 300, 256, **ms)
    add('mn_k_tail_fallback', 45, 300, 300, **ms)
    add('n_tail_fallback_bn128', 32, 100, 256, **dict(ms, bn=128))
    if diag:
        # known-bad (contract violated: partial N tile with matrix_store). The simulator panics, which kills the
        # worker -> run each with --only.
        add('isoM_m_tail_bn128', 13, 256, 256, **dict(ms, bn=128))   # M tail, BN=128: panics ('non major filter')
        add('isoN_loadonly', 32, 100, 256, **ms_load)       # same shape without matrix_store: fine
        add('isoN_n100_bn128', 32, 100, 256, **dict(ms, bn=128))   # passes (valid width 100 >= 96)
        for n in (33, 60, 64, 65, 96, 100, 112, 120, 127, 128, 129):
            add(f'sweepN_n{n}', 32, n, 256, **dict(ms, bn=128))
        for n in (100, 130, 200, 300):
            add(f'sweepN256_n{n}', 32, n, 256, **ms)
    return C


def launcher(argv):
    sys.path.insert(0, str(HERE))
    import tune_lib
    dest = Path(tempfile.mkdtemp(prefix='verify_dyn_', dir=str(tune_lib.REPO / 'tmp')))
    env = tune_lib.env_for(dest)
    print('work dir:', dest, flush=True)
    rc = subprocess.call([tune_lib.PYTHON, '-B', str(Path(__file__).resolve()), 'worker', *argv],
                         cwd=dest, env=env)
    if rc == 0:
        shutil.rmtree(dest, ignore_errors=True)
    return rc


def worker(argv):
    import importlib.util
    import traceback
    import torch
    import triton
    from triton.runtime import driver
    assert driver.active.get_current_target().arch == 'gfx946'

    only = argv[argv.index('--only') + 1] if '--only' in argv else None
    spec = importlib.util.spec_from_file_location('gemm_a16w16_generic',
                                                  os.environ.get('VERIFY_OP_PATH') or str(HERE / 'gemm_a16w16_generic.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    kernel = mod._gemm_bf16_generic_kernel
    from boltops.gemm.triton.splitk_reduce import _gemm_splitk_reduce_kernel as reduce_kernel
    dev = 'cuda'

    def run_case(cs):
        M, N, K, bm, bn, bk, ks = cs['M'], cs['N'], cs['K'], cs['bm'], cs['bn'], cs['bk'], cs['ks']
        dt = torch.bfloat16 if cs['dtype'] == 'bf16' else torch.float16
        g = torch.Generator().manual_seed(11)
        pad = 2 * max(bm, bn, bk)
        nan = float('nan')
        a_buf = torch.full((M + pad, K + pad), nan, dtype=dt)
        a_ref = torch.randn((M, K), generator=g).to(dt)
        a_buf[:M, :K] = a_ref
        a = a_buf.to(dev)[:M, :K]
        if cs['layout'] == 'TT':
            b_buf = torch.full((K + pad, N + pad), nan, dtype=dt)
            b_ref = torch.randn((K, N), generator=g).to(dt)
            b_buf[:K, :N] = b_ref
            b = b_buf.to(dev)[:K, :N]
        else:
            w_buf = torch.full((N + pad, K + pad), nan, dtype=dt)
            w_ref = torch.randn((N, K), generator=g).to(dt)
            w_buf[:N, :K] = w_ref
            b = w_buf.to(dev)[:N, :K].T
            b_ref = w_ref.T
        bias_cpu = torch.randn((N,), generator=g).to(dt) if cs['bias'] else None
        ref = a_ref.float() @ b_ref.float()
        if bias_cpu is not None:
            ref = ref + bias_cpu.float()
        bias = bias_cpu.to(dev) if bias_cpu is not None else None

        # C: [:M, :N] view of a NaN buffer with one tile of padding on the far row/col edges
        ldc = cdiv(N + bn, 8) * 8
        c_buf = torch.full((M + bm, ldc), nan, dtype=dt)
        c_dev = c_buf.to(dev)
        out = c_dev[:M, :N]

        if ks == 1:
            sbs = 0
            c, sck, scm, scn = out, 0, out.stride(0), out.stride(1)
        else:
            sbs = cdiv(cdiv(K, ks), bk) * bk
            pp = torch.full((ks, M, N), nan, dtype=torch.float32, device=dev)
            c, sck, scm, scn = pp, pp.stride(0), pp.stride(1), pp.stride(2)
        const = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                     NUM_KSPLIT=ks, SPLITK_BLOCK_SIZE=sbs, WARP_SPECIALIZE=False,
                     USE_MLS=cs['mls'], USE_MATRIX_STORE=cs['ms'] and ks == 1,
                     ADD_BIAS=bias is not None)
        grid = (cdiv(M, bm) * cdiv(N, bn) * ks,)
        kernel[grid](a, b, c, bias if bias is not None else a, M, N, K,
                     *a.stride(), *b.stride(), sck, scm, scn, **const, **({'num_warps': 4} | cs['opts']))
        if ks > 1:
            reduce_kernel[(cdiv(M, 32), cdiv(N, 32))](
                pp, out, bias, M, N, pp.stride(0), pp.stride(1), pp.stride(2),
                out.stride(0), out.stride(1), 32, 32, ks, triton.next_power_of_2(ks),
                ADD_BIAS=bias is not None, activation="", use_activation=False,
                KERNEL_NAME="_gemm_a16w16_reduce_kernel")
        torch.cuda.synchronize()

        full = c_dev.cpu()
        got = full[:M, :N].float()
        pad_rows_ok = bool(torch.isnan(full[M:, :]).all())            # rows >= M untouched
        pad_cols_ok = bool(torch.isnan(full[:M, N:]).all())           # cols >= N untouched (incl. ld slack)
        finite = bool(torch.isfinite(got).all())
        close = bool(torch.isclose(got, ref, rtol=1.5e-2, atol=0.05).all()) if finite else False
        err = float((got - ref).abs().max()) if finite else float('inf')
        diag = {}
        if finite and bias_cpu is not None and not close:
            # what did the kernel actually add? delta = got - (A@B); compare with the bias vector
            delta = got - (ref - bias_cpu.float())
            bf = bias_cpu.float()
            diag = dict(
                bias_dropped=float((delta).abs().max()),                       # ~0  => bias never added
                bias_exact=float((delta - bf[None, :]).abs().max()),            # ~0  => bias correct
                bias_row_const=float((delta - delta[:1]).abs().max()),          # ~0  => same per-column shift on every row
                bias_perm=bool(torch.allclose(delta[0].sort().values, bf.sort().values, atol=0.05)),
                bias_first_cols=[round(float(x), 3) for x in delta[0, :6]], bias_ref_cols=[round(float(x), 3) for x in bf[:6]])
        return dict(ok=pad_rows_ok and pad_cols_ok and finite and close, finite=finite, close=close,
                    pad_rows_untouched=pad_rows_ok, pad_cols_untouched=pad_cols_ok, max_abs_err=err, **diag)

    cases = [c for c in _cases('--diag' in argv) if not only or only in c['name']]
    n_ok = 0
    for cs in cases:
        try:
            res = run_case(cs)
        except Exception:
            res = dict(ok=False, error=traceback.format_exc()[-1500:])
        n_ok += bool(res['ok'])
        print('RESULT ' + json.dumps(dict(name=cs['name'], **res)), flush=True)
    print(f'SUMMARY {n_ok}/{len(cases)} passed', flush=True)
    sys.exit(0 if n_ok == len(cases) else 1)


if __name__ == '__main__':
    if '--list' in sys.argv:
        print('\n'.join(c['name'] for c in _cases('--diag' in sys.argv)))
    elif len(sys.argv) >= 2 and sys.argv[1] == 'worker':
        worker(sys.argv[2:])
    else:
        sys.exit(launcher(sys.argv[1:]))
