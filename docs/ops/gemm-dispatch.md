# GEMM interfaces and independent backends

Each public operation has one entry point, with its own JSON configuration
under `boltops/gemm/configs/`. All providers are independent (`triton`,
`gluon`, `tilelang`). There are no public aliases named after backend kernels.

| Public interface | Computation / input contract | Previous public entry points |
| --- | --- | --- |
| `gemm_dense` | Native A[M,K] @ B[N,K].T, optional bias | `matmul_persistent`, `gemm_a16w16` |
| `gemm_scaled` | Native block-scale or whole-K tensor/row/channel scale; FP8, INT8, FP16/BF16 where registered | `gemm_a8w8_blockscale`, `w8a8_triton_block_scaled_mm`, `triton_scaled_mm`, `w8a8_block_int8_matmul` |
| `gemm_batched` | Exact native A[B,M,K] @ B[B,N,K].T, optional [B,1,N] bias | `batched_gemm_bf16` |
| `gemm_grouped` | Group-addressed GEMM; equal groups or explicitly registered addressing modes | Independent grouped contracts |
| `gemm_awq` | A16 + packed INT4, explicit zero points, AWQ nibble order | `awq_gemm_triton` |
| `gemm_mxfp4` | Prequantized packed E2M1 A/B and E8M0 K32 scales | `gemm_afp4wfp4` |
| `gemm_batched_a16wfp4` | Existing online MXFP4 activation quantization + packed FP4 B | `batched_gemm_a16wfp4` |

The last three computations are intentionally separate: AWQ has zero points
and a different storage contract; MXFP4 inputs are already quantized;
A16WFP4 performs lossy online activation quantization. The latter is not an
exact unquantized A16 x FP4 GEMM. Registering an adapter does not establish
hardware or simulator support. Missing native instructions fail unchanged;
there is no emulation fallback.

Old public wrappers are removed without compatibility aliases. Existing
implementation functions remain in `boltops.gemm.triton.*` for backend-specific
configuration, partial split-K results, specialized output layouts and fused
activation experiments. These implementation controls are not new public
GEMM operations. NN callers of `matmul_persistent(a,b)` migrate to
`gemm_dense(a,b.T)`. `triton_scaled_mm` callers also transpose B and supply an
explicit scale recipe. Plain batched does not broadcast batch dimensions.

`gemm_awq` takes A[M,K], packed B[K,N/8], scales[K/G,N] and zeros[K/G,N/8],
all contiguous, plus explicit `group_size=G`. Its output dtype is the scale
dtype. `gemm_mxfp4` takes uint8 packed A[M,K/2], B[N,K/2] and uint8 E8M0
scales [M|N,K/32]. `gemm_batched_a16wfp4` takes A[B,M,K], packed uint8
B[B,N,K/2] and E8M0 scales[B,N,K/32]. The packed paths require K divisible by 32.

All interfaces accept `backend="auto"`, `"triton"`, `"gluon"`, `"tilelang"`.
Automatic selection queries the selected operation's own configuration file
before defaults. Explicit backend selection never retries another backend on
compile/execute failure. Unsupported combinations raise NotImplementedError;
a registered backend missing its runtime raises ImportError. Gluon retains the optimized E4M3FN/FP32 scale recipe. TileLang remains an empty registration point.

The seven files are `gemm_dense.json`, `gemm_scaled.json`, `gemm_batched.json`,
`gemm_grouped.json`, `gemm_awq.json`, `gemm_mxfp4.json`, and
`gemm_batched_a16wfp4.json`. Each carries its interface identifier. Schema 3
separates `match` (arch, num_cus, MNK, dtype/scale/layout contracts) from
`config` (backend, implementation, tile and optional existing config file).
Tile is not a matching condition. No measurements, frequency, commit IDs or
evidence paths are stored. Current tuned entries reuse existing gfx946 48-CU
FP16 TT configurations. Source kernels and default configurations are unchanged.
