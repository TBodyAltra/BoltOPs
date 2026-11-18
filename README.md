# BoltOPs

Backend Optimized Low-latency Triton/TileLang Operators

**English** | [中文](README.zh-CN.md)

## Introduction

**BoltOPs** (*Backend Optimized Low-latency Triton/TileLang Operators*) is the AI Compiler team's high-performance operator library for Hygon HCU. The name also nods to the **lightning bolt** — low latency and lightning-fast execution by design. It delivers production-ready, benchmark-driven kernels written in Triton, TileLang, and other DSLs behind clean Python APIs, giving inference frameworks and model-side callers tuned HCU throughput without backend-specific glue.

The library focuses on latency-critical fused operators, correctness-first validation, and deployment-friendly packaging: fast paths where they matter, minimal friction where callers integrate.

![BoltOp is Fastest](docs/boltop_is_fastest.jpg)

|            | Name              |
| ---------- | ----------------- |
| Repository | `BoltOPs`       |
| DAS        | `boltops`       |
| PyPI / pip | `boltops`       |
| Wheel      | `boltops-*.whl` |
| Import     | `boltops`       |

## Install

```bash
pip install -e ".[triton]"          # common
pip install -e ".[triton,dev]"      # develop / test
pip install -e ".[tune]"            # MoE autotune / benchmark tools
pip install -e ".[all]"             # Triton + TileLang
./build_wheel.sh                    # pure-Python wheel → dist/
# MoE autotune / benchmark entry (shipped with the package)
# boltops-fused-moe-tune-cli <device_name> <test_type> [--benchmark]
```

Dependencies: `torch`; Triton ops also need a working Triton install (Hygon build). For MoE tuning see
`[docs/ops/fused_moe/fused_moe.md](docs/ops/fused_moe/fused_moe.md)` and
`[tune_moe_triton_user_guide.md](boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md)`.

## Usage

Callers should import public APIs from the corresponding operator package.
The root `boltops` package exposes version metadata only:

```python
from boltops.generic import (
    attn_res,
    causal_conv1d_fn,
    causal_conv1d_update,
    fused_mamba_state_scatter_with_mask,
)
from boltops.fused_moe import (
    fused_experts,
    moe_activation,
)
from boltops.gemm import (
    matmul_persistent,
    gemm_a16w16,
    w8a8_triton_block_scaled_mm,
    triton_scaled_mm,
    awq_gemm_triton,
    gemm_a8w8_blockscale,
    gemm_afp4wfp4,
    batched_gemm_a16wfp4,
    w8a8_block_int8_matmul,
)
from boltops.mla import (
    decode_attention_fwd,
    compute_num_kv_splits,
)
from boltops.fa import (
    extend_attention_fwd,
    extend_attention_fwd_unified,
    dense_prefill_attention_fwd,
    dense_prefill_attention_fwd_4d,
    dense_prefill_attention_fwd_segmented,
    decode_attention_fwd_4d,
    build_unified_kv_indices,
    can_use_dense_prefill_fp8,
    unified_attention,
    reshape_and_cache_int4,
    triangle_attention_fwd,
    KVQuantMode,
)
from boltops.sparse_gqa import (
    qwen_sparse_kv_extraction_compact_hcu_fa_triton,
    sparse_gqa_packed_decode_triton,
)
from boltops.mqa_logits import (
    triton_mqa_logits,
    triton_fp8_paged_mqa_logits,
)
from boltops.qsa import (
    qsa_mqa_paged,
    qsa_select_paged_tokens,
    expand_qsa_block_indices,
    qsa_sparse_paged_attention,
)
from boltops.step4_dsa import (
    triton_indexer_logits,
    triton_sparse_attention_prefill,
    triton_sparse_attention_decode,
)
from boltops.fla.kda import (
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_intra,
    chunk_kda_fwd_intra_token_parallel,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
    recompute_w_u_fwd as recompute_w_u_fwd_kda,
)
from boltops.fla.gdn import (
    chunk_fwd_o,
    chunk_gated_delta_rule_fwd_h,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
    recompute_w_u_fwd as recompute_w_u_fwd_gdn,
)

mixed = attn_res(prefix, blocks, norm_weight, qk_weight, num_blocks, eps)

# GDN / SGLang WY recomputation
w, u = recompute_w_u_fwd_gdn(
    k=k, v=v, beta=beta, g_cumsum=g_cumsum, A=A, cu_seqlens=cu_seqlens
)

# KDA / vLLM WY recomputation
w, u, qg, kg = recompute_w_u_fwd_kda(k=k, v=v, beta=beta, A=A, gk=gk)

fused_mamba_state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw)

out = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)

moe_activation(activated, ffn1_out, activation="silu")

# Gemm family (layouts match vLLM / aiter call sites)
c = matmul_persistent(a, b, bias)         # vLLM Triton persistent; B[K,N]
y = gemm_a16w16(x, weight, bias)          # aiter A16W16; weight [N,K]
c = w8a8_triton_block_scaled_mm(A, B, As, Bs, [128, 128])
c = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float16)
c = awq_gemm_triton(x, qweight, scales, qzeros, split_k_iters)
y = gemm_a8w8_blockscale(x, w, x_scale, w_scale)
y = gemm_afp4wfp4(x, w, x_scales, w_scales)          # needs FP4 GPU
y = batched_gemm_a16wfp4(x, w, w_scales)             # needs FP4 GPU
c = w8a8_block_int8_matmul(A, B, As, Bs, [128, 128]) # SGLang INT8 block

# MLA decode (aligned with vLLM TRITON_MLA; pass is_mla=True for absorb MLA)
splits = compute_num_kv_splits(max_seq_len)
decode_attention_fwd(
    q, k_buffer, v_buffer, o, lse,
    block_table, seq_lens, attn_logits, splits, sm_scale,
    page_size=64, is_mla=True,
)

# Triangle Attention forward; BF16 q/k/v are [B, H, N, D], q scaled by 1/sqrt(D)
out = triangle_attention_fwd(q, k, v, [mask_bias, triangle_bias])

# Step-4 DSA (vendor contract step4-dsa-ops-v1)
logits = triton_indexer_logits(index_q, weights, index_k)
out, lse = triton_sparse_attention_decode(
    q, k_cache, v_cache, packed, counts, kv_seqlens,
    num_kv_groups=1,
)
```

For A/B of backends, import the backend module directly — do not go around the public API:

```python
from boltops.generic.triton.attn_res import attn_res as triton_impl
```

## Public ops (+ planned)

| Category                                                                             | API                                                                                                                                                                                                                                                      | DSLs                  | Maintainers                            | Default backend                         | Intro & docs                                                                                                                                                                   |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | -------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Generic                                                                              | `attn_res` `fused_mamba_state_scatter_with_mask`<br>`causal_conv1d_fn`<br>`causal_conv1d_update`                                                                                                                                                               | Triton ✅ TileLang ❌ | Triton @all                            | Triton                                  | [docs](docs/ops/generic/generic.md)                                                                                                                                             |
| FLA/KDA family                                                                       | `chunk_gla_fwd_o_gk` `chunk_kda_fwd_intra` `chunk_kda_fwd_intra_token_parallel` `fused_kda_gate` `fused_kda_gate_chunk_cumsum` `fused_recurrent_kda` `fused_recurrent_kda_fwd` `fused_recurrent_kda_packed_decode` `recompute_w_u_fwd` | Triton ✅ TileLang ❌ | Triton @liji                           | Triton                                  | [docs](docs/ops/fla/kda.md)                                                                                                                                                     |
| FLA/GDN family                                                                       |                                                                                                                                                                                                                                                          |                       | Triton @liji<br />Tilelang @huangteng |                                         |                                                                                                                                                                                |
| MoE family                                                                           | `fused_experts` `moe_activation`                                                                                                                                                                                                                     | Triton ✅ TileLang ❌ | Triton @zhenxin TileLang @xxx          | Triton (`fused_experts` mixed+policy) | [docs](docs/ops/fused_moe/fused_moe.md) [performance](docs/ops/fused_moe/performance_report.md) [tuning guide](boltops/tools/fused_moe_triton_tune/tune_moe_triton_user_guide.md) |
| FA family (UnifiedAttention / PagedAttention / ExtendAttention /...) | `extend_attention_fwd` `extend_attention_fwd_unified` `dense_prefill_attention_fwd` `dense_prefill_attention_fwd_4d` `dense_prefill_attention_fwd_segmented` `decode_attention_fwd_4d` `build_unified_kv_indices` `can_use_dense_prefill_fp8`<br>`unified_attention` `reshape_and_cache_int4` (Quant type: `KVQuantMode`) | Triton ✅ TileLang ❌ | Triton @huangshuang @sunchunhao | Triton | [UnifiedAttention](docs/ops/fa/unified_attention.md) [ExtendAttention](docs/ops/fa/extend_attention.md) [DecodeAttention](docs/ops/fa/decode_attention.md) |
| Triangle Attention | `triangle_attention_fwd` | Triton ✅ TileLang ❌ | Triton @sunchunhao | Triton | [docs](docs/ops/fa/triangle_attention.md) |
| MLA / FlashMLA / Sparse MLA                                                          | `decode_attention_fwd` `decode_attention_fwd_grouped` `decode_attention_fwd_normal` `compute_num_kv_splits`                                                                                                                                      | Triton ✅ TileLang ❌ | Triton @huangshuang @sunchunhao       | Triton                                  | [docs](docs/ops/mla.md) (dense MLA decode now; FlashMLA / Sparse TBD)                                                                                                           |
| Qwen Sparse GQA                                                                      | `qwen_sparse_kv_extraction_compact_hcu_fa_triton` `sparse_gqa_packed_decode_triton`                                                                                                                                                                  | Triton ✅ TileLang ❌ | Triton @huangshuang @sunchunhao       | Triton                                  | [docs](docs/ops/sparse_gqa.md)                                                                                                                                                 |
| Step-4 DSA                                                                           | `triton_build_rope_cache` `triton_indexer_norm_rope` `triton_csa_*` `triton_indexer_logits` `triton_region_topk_ids` `triton_topk_select_and_pack` `triton_*_sparse_meta` `triton_sparse_attention_*` `triton_merge_split_states`      | Triton ✅ TileLang ❌ | Triton @xukang @duyubo                 | Triton                                  | [docs](docs/ops/step4_dsa.md)                                                                                                                                                   |
| MQA Logits                                                                           | `triton_mqa_logits` (fp8 w/ scales; fp16/bf16 w/o)<br>`triton_fp8_paged_mqa_logits` / `_stage1`                                                                                                                                                        | Triton ✅ TileLang ❌ | Triton @fangzx                         | Triton                                  | [docs](docs/ops/mqa_logits_triton.md) (prefill + FP8 paged)                                                                                                                            |
| QSA (paged scoring / selection / sparse attention)                                              | `qsa_mqa_paged` `qsa_select_paged_tokens` `expand_qsa_block_indices` `qsa_sparse_paged_attention`                                                                                                                                | Triton ✅ TileLang ❌ |                                       | Triton              | [docs](docs/ops/qsa.md) (4-step paged pipeline; TileLang scoring entries live in `boltops.mqa_logits`)  |
| MHC                                                                                  | `mhc_pre_big_fuse` `pre_big_fuse_tilelang` `mhc_post_fwd` `hc_split_sinkhorn` `mhc_fused_tilelang`                                                                                                                                             | TileLang ✅           | TileLang @zhouweiwang                  | TileLang                                | [docs](docs/ops/mhc.md)                                                                                                                                                         |
| IHC                                                                                  | `ihc_pre` `ihc_post` `ihc_head`                                                                                                                                                                                                                    | TileLang ✅           | TileLang @zhouweiwang                  | TileLang                                | [docs](docs/ops/ihc.md)                                                                                                                                                         |
| Gemm/AwqGemm family                                                                  | `gemm_a16w16` `gemm_a32w32` `gemm_a8w8` `gemm_a8w8_blockscale` `gemm_a16w4` `gemm_w4a4` `batched_gemm_bf16` `batched_gemm_a16w4`                                            | Triton ✅ Gluon ✅ (FP8 blockscale) TileLang ❌ | Triton @xukang @duyubo                 | Triton                                  | [docs](docs/gemm.md)                                                                                                                                                        |
| DeepGemm family                                                                      |                                                                                                                                                                                                                                                          |                       | Triton @zhenxin Tilelang @zhouweiwang  |                                         |                                                                                                                                                                                |

## Layout

```text
boltops/
  fla/               # Flash Linear Attention umbrella
    kda/             # FLA/KDA family (Triton @liji)
    gdn/             # FLA/GDN family (Triton @huangteng)
    common/          # helpers shared by FLA sub-families
  <op>/              # one directory per op
    __init__.py      # public signature + ship-time backend (lazy-import)
    triton/          # Triton impl (incl. configs)
    tilelang/        # TileLang impl (placeholder)
  step4_dsa/         # Step-4 DSA numerical Triton ops (vendor contract v1)
    geometry.py      # model constants (not a kernel)
    triton/          # kernels + launch helpers
    tilelang/        # placeholder
  tools/             # shipped developer tools (incl. MoE autotune)
  utility/           # shared runtime helpers
docs/ops/            # op docs
test/                # correctness + optional --bench; not in wheel
build_wheel.sh       # build wheel
```

## Development

1. **Positioning of BoltOPs operators**
   As a DSL-built operator library, we pursue performance on specific sizes **on top of generality** — that is the design intent. For a given op, different kernels + DSLs(mix) may target specific sizes or scenarios when chasing peak performance.
2. **Triton and TileLang**
   Both are Python-like DSLs and, today, complementary / collaborative — BoltOPs should reflect that:

|           | Triton                                                                                                            | TileLang                                                                                                                       |
| --------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Strengths | Strong ecosystem & compatibility; fast performance uplift; light edits can reach ~80%+ of hand-written HIPC / asm | Rising DSL with warp / thread-level fine-grained APIs; fast to develop; scene-specific hand-tuned kernels can match HIPC / asm |
| Limits    | Harder to push further toward peak                                                                                | Ecosystem and generality lag Triton                                                                                            |

   The same op may therefore ship both Triton and TileLang backends; the default backend is chosen after bake-off (or a mixed policy).
3. **API definition**
  Prefer aligning public signatures with latest vLLM, SGLang, and similar open-source frameworks for easier integration. If no matching API exists, align with HIPC / CUDA C counterparts.
4. **Testing**
  Each op has one test file / directory, with `test` and `bench` together: `test` covers interface functionality / generality; `bench` focuses on real-network shapes. They complement each other.

```bash
pip install -e ".[triton,dev]"
pytest test                  # correctness (functional / case coverage)
pytest test --bench -s       # correctness + real model-size performance
# single op:
# pytest test/fla/test_gdn_recompute_w_u.py
# pytest test/fla/test_gdn_recompute_w_u.py --bench -s
```

5. **optest**
  `optest` is the unified operator test entry used by CI; one run covers accuracy (pytest over `test/`) and performance (`scripts/run_operator_benchmarks.py`):

```bash
pip install optest -i http://10.16.1.201:9929/nightly/dtk2604/+simple/ --trusted-host 10.16.1.201  --force-reinstall
optest --op_name boltops --test_dir . --test_mode all      # accuracy + performance
optest --op_name boltops --test_dir . --test_mode acc      # accuracy only
optest --op_name boltops --test_dir . --test_mode per      # benchmarks only
optest --op_name boltops --test_dir . --test_mode all --gpus 0,2  # pin specific HCUs
optest --display                                            # HTML report from latest results
```

  Results land in `/op_test_output/boltops/<branch>_<commit>_<timestamp>/`
  (`.optest-results.json` + per-kernel CSVs) with logs under `/op_test_log/boltops/`.
  CI (`performance:optest`) runs the same entry (`--test_mode per`) for every
  MR into `main` and converts the results to JUnit via
  `ci_script/optest_to_junit.py`; the accuracy suite still has pre-existing
  failures, so acc is not gated yet (upgrade to `--test_mode all` once fixed).
  To add a performance test, write a normal `@pytest.mark.bench` test
  under `test/` -- no perf-specific code, no registration: the perf
  entry (`scripts/run_operator_benchmarks.py`) discovers it
  automatically.  While it runs, the collection layer
  (`test/conftest.py`) observes each timing call and writes one unified
  CSV per op. Full conventions (collection, CSV schema, optional
  enrichment): `docs/optest_perf.md`.
  Run a subset locally with
  `python scripts/run_operator_benchmarks.py --ops causal_conv1d`.
  Tuning scripts that search for and write configs live under
  `scripts/tune/` and are not wired into optest.

**Adding a new op:**

1. Create `boltops/<op>/` with `triton/` and `tilelang/` impls; put correctness in `test/test_<op>.py` and model-size benches under `@pytest.mark.bench`, then use `--bench` to pick the winner.
2. Prefer public signatures aligned with vLLM / SGLang (or HIPC / CUDA C); wire ship-time backend selection in `boltops/<op>/__init__.py` and export.
3. Re-export from `boltops/__init__.py` and add `docs/ops/`.

## BF16 GEMM measurements on PMD

2026-11-17 · node60 / `kxu_pmd_8569_fp8_20261017` · gfx946; 1 DIE / 48 CU unless the row specifies 2 DIE / 96 CU.
BF16 inputs/output, FP32 accumulation, complete K. Environment, configs, assembly, raw measurements and PERF: [R001 TN](docs/ops/gemm/tuning/r001/1die/tn/record.json), [R001 TT](docs/ops/gemm/tuning/r001/1die/tt/record.json), [R002 TT](docs/ops/gemm/tuning/r002/1die/tt/record.json), [R002 TT (2 DIE / 96 CU)](docs/ops/gemm/tuning/r002/2die/tt/record.json), [R003 TT](docs/ops/gemm/tuning/r003/1die/tt/record.json), [R004 TT](docs/ops/gemm/tuning/r004/1die/tt/record.json).

| Case / layout | M × N × K | Full dispatch (µs) | MMAC utilization | Validation / timing |
|---|---|---:|---:|---|
| R001 TN | 64 × 64 × 2048 | 5.324 | 0.73% | Random / ones / random; median of 9 warm samples |
| R001 TT | 64 × 64 × 2048 | 5.744 | 0.68% | Random / ones / random; median of 9 warm samples |
| R002 TT | 64 × 248320 × 2048 | 596.044 | 25.27% | Full random output; median of 2 warm samples |
| R002 TT (2 DIE / 96 CU) | 64 × 248320 × 2048 | 310.224 | 24.28% | Full random output; median of 2 warm samples |
| R003 TT | 2560 × 64 × 2048 | 10.565 | 14.70% | Random / ones / random; median of 9 warm samples |
| R004 TT | 6656 × 64 × 2048 | 25.327 | 15.94% | Random / ones / random; median of 9 warm samples |

All full-grid outputs have zero mismatches/nonfinite values at abs0.001+rel0.01; output guards and complete TG counts passed. SGPR/VGPR spills are zero. Current dispatch/native checks on sb/48 CU: 61 passed, 4 persistent cases deselected; configuration checks: 32 passed. Two known persistent TT tail compilation failures were reproduced before the refactor and are not counted as passes. PERF uses the same HSACO as the measured full grid. These are PMD results; detailed sampling scopes and hashes are linked above.

R002 on `sbx2` uses 2 DIE / 96 CU: 310.224 µs, 209.83 TOPS, 3.382 TB/s effective read+write bandwidth, 1.921× the single-DIE performance. Each DIE completes 485 TGs on 48 active CUs. The same kernel and tile configuration are used; the sbx2 preset also configures its memory and cross-DIE fabric.

R002 uses three LDS slots and register prefetch to overlap the next operand reads with the current MMAC batch. R002 single-DIE effective read+write bandwidth is 1.760 TB/s, including one logical read of A/B and one write of C; its formula and measured environment are in the record.

## FP8 R010 GEMM measurement on PMD

2026-11-17 · node60 / `kxu_pmd_8569_fp8_20261017` · gfx946, 1 DIE /48 CU, TN.

| Case | M × N × K | Kernel (µs /TOPS) | Pipeline | Validation |
|---|---|---|---|---|
| R010 | 2560 ×12288 ×2048 | 244.128 /527.79 | 48 persistent TG ×20 tiles | All 31,457,280 outputs match; guards passed |

One complete cold dispatch, GC455ps. E4M3FN inputs, BF16 output; B scale BF16 values are promoted to FP32 on CPU before upload, excluded from kernel timing. Latency is 16.57% below the historical 292.620 µs baseline; this includes the R009 pipeline and toolchain migration together. LDS114944 bytes, consumer VGPR235, no spills. Configuration/dispatch checks:34 passed. [Environment, raw statistics, assembly, benchmark and matched single-TG PERF](docs/ops/gemm/tuning/r010/1die/tn/record.json).

## FP8 R010 on dual-die PMD sbx2

2026-11-17 · node60 / `kxu_pmd_8569_fp8_20261017` · gfx946, 2 DIE ×48 CU, GC455ps, TN2560×12288×2048. Same Triton/Clang and input/scale convention as the single-die R010 measurement above.

| Chip / grid | Kernel (µs / TOPS) | TGs per die | Speedup vs single die |
|---|---|---|---|
| sbx2 /96 TG ×10 tiles | 129.222 /997.12 | {'0': 48, '1': 48} | 1.8892× |

One complete cold public-API dispatch; all31,457,280 outputs match, guards pass, and both dies execute. Time is complete-system gcSimCycles; per-die times are not summed. The96-TG dispatch uses all96 CU. Single-TG PERF covers10 tiles with the same .text/.rodata as the full96-TG kernel; all327,680 selected outputs and uncovered output guards passed in three captures. [Dual-die record](docs/ops/gemm/tuning/r010/2die/tn/record.json).

## FP8 R011 on dual-die PMD sbx2

2026-11-17 · node60 / `kxu_pmd_8569_fp8_20261017` · gfx946, 2 DIE ×48 CU, GC455ps, TN6656×2048×4096. Triton `release/3.6.x@f8cbbee18498`, Clang22 `dtk-llvm-7.2.1@264542e7e37f` with the recorded MMAC C no-sink guard.

| Chip / grid | Kernel (µs / TOPS) | TGs per die | Speedup vs nonpersistent |
|---|---|---|---|
| sbx2 /96 TG, 4–5 tiles each | 128.180 /871.19 | 48 /48 | 1.1367× |

One complete cold public-API dispatch; all 13,631,488 outputs match and guards pass on all 96 CU. Latency is 12.03% below the same-environment nonpersistent 145.708 µs baseline. Reuses the R010 continuous pipeline with two complete A-scale banks and consumer FP32 B-scale loads, fitting 131072 B LDS without spills. Single-TG PERF covers 5 tiles with the same .text/.rodata as the full 96-TG kernel; all 163,840 selected outputs and uncovered output guards passed in three captures. R009/R010 48-TG and 96-TG executable sections remain unchanged. [Dual-die record](docs/ops/gemm/tuning/r011/2die/tn/record.json).

## License

BoltOPs original code is licensed under the [MIT License](LICENSE). Portions
adapted from third-party projects retain their original licenses and copyright
notices; see the source file headers and the module source notes in
[gemm](boltops/gemm/triton/SOURCE.md), [MLA](boltops/mla/triton/SOURCE.md),
[MQA logits](boltops/mqa_logits/triton/SOURCE.md), and
[Step-4 DSA](boltops/step4_dsa/triton/SOURCE.md).
