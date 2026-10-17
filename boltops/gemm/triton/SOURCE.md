Vendored Triton GEMM sources (flat layout, no vendor subpackages):

FP16 / BF16
- `matmul_persistent.py` — vLLM / SGLang `batch_invariant`
- `gemm_a16w16.py` / `gemm_a16w16_op.py` — ROCm/aiter GEMM-A16W16
- `batched_gemm_bf16.py` — ROCm/aiter batched A16W16 (`batched_gemm_bf16`)

Quantized (vLLM / SGLang) — public
- `w8a8_triton_block_scaled_mm.py` — vLLM `fp8_utils`
- `w8a8_block_int8_matmul.py` — SGLang `int8_kernel.py` (HCU OK)
- `triton_scaled_mm.py` — vLLM / SGLang compressed-tensors path
- `awq_gemm_triton.py` — vLLM / SGLang `awq_triton.py`

Quantized (aiter) — public
- `gemm_a8w8_blockscale.py` / `_op.py`
- `gemm_afp4wfp4.py` / `_op.py`
- `batched_gemm_a16wfp4.py` / `_op.py`
- `_quant/quant.py`

Internal (vendored, not in `boltops` / `boltops.gemm` `__all__`)
- `gemm_fp16.py` — HCU WASP/WDRA experiments

Configs (on disk, HCU only where available)
- `configs/awq_gemm_triton/` — vLLM-style `AWQ_{N}_{K}_{K100_AI|BW200}.json`
- `configs/w8a8_block_int8_matmul/` — hygon aiter `block_w8a8` (`BW200` / `gfx938`)
- `configs/batched_gemm_bf16/` — optional `BATCHED_BF16-N={N}-K={K}_{bf16|fp16}_{device}.json`

Note: hygon aiter `awq_w4a16/` tables target aiter `gemm_a16w4` (different
schema) and are **not** wired to `awq_gemm_triton`. Other kernels use
`builtin_gemm_configs.py`. Gluon not vendored.

FP8 x FP4 block-scaled GEMM — public
- gemm_a8wfp4.py: boltOps implementation, ordinary FP8 tl.dot per K32 then
  A/B scale multiplication; dense and expert-grouped/ragged entry points.
- E2M1 nibble order and value mapping checked against ROCm/aiter commit
  5c9f6431ecc34f28a2ab0acfbe248e31922e7c05, aiter/utility/fp4_utils.py.
- Reference native Triton kernels and grouping conventions:
  https://github.com/ROCm/aiter/blob/5c9f6431ecc34f28a2ab0acfbe248e31922e7c05/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8wfp4.py
  https://github.com/ROCm/aiter/blob/5c9f6431ecc34f28a2ab0acfbe248e31922e7c05/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py
- Native aiter dot_scaled is not used: this API also accepts numeric FP32
  activation scales per K32/K128, matching boltOps requirements G472–G505.
