# GEMM interfaces and independent backends

Each public operation has one entry point, with its own JSON configuration
under `boltops/gemm/configs/`. Backends are independent (`triton`,
`gluon`, `tilelang`). There are no public aliases named after backend kernels.

| Public interface | Computation / input contract | Previous public entry points |
| --- | --- | --- |
| `gemm_dense` | Native A[M,K] @ B[N,K].T, optional bias | `matmul_persistent`, `gemm_a16w16` |
| `gemm_scaled` | Native block-scale or whole-K tensor/row/channel scale; FP8, INT8, FP16/BF16 where registered | `gemm_a8w8_blockscale`, `w8a8_triton_block_scaled_mm`, `triton_scaled_mm`, `w8a8_block_int8_matmul` |
| `gemm_batched` | Exact native A[B,M,K] @ B[B,N,K].T, optional [B,1,N] bias | `batched_gemm_bf16` |
| `gemm_grouped` | Group-addressed GEMM; equal groups or explicitly registered addressing modes | Independent grouped contracts |
| `gemm_scaled(..., b_format="awq_int4")` | A16 + packed INT4, optional zero points (default zero), AWQ nibble order | `awq_gemm_triton` |
| `gemm_scaled(..., a_format="fp4_e2m1", b_format="fp4_e2m1")` | Prequantized packed E2M1 A/B and E8M0 K32 scales | `gemm_afp4wfp4` |
| `gemm_batched_a16wfp4_online_quant` | Existing online MXFP4 activation quantization + packed FP4 B | `batched_gemm_a16wfp4` |

AWQ and prequantized MXFP4 are formats/recipes of `gemm_scaled`, not separate
public operations. Online A16WFP4 remains separate because it performs lossy
activation quantization inside the GEMM K loop. Each row/K32 group derives
its dynamic E8M0 scale and FP4 values, then immediately calls native scaled
dot; it does not materialize a whole quantized A in global memory. This is not
exact unquantized A16 x FP4 multiplication. Missing native support still fails
without emulation or backend retries.

Old public wrappers are removed without compatibility aliases. Existing
implementation functions remain in `boltops.gemm.triton.*` for backend-specific
configuration, partial split-K results, specialized output layouts and fused
activation experiments. These implementation controls are not new public
GEMM operations. NN callers of `matmul_persistent(a,b)` migrate to
`gemm_dense(a,b.T)`. `triton_scaled_mm` callers also transpose B and supply an
explicit scale recipe. Plain batched does not broadcast batch dimensions.

Scaled format examples:

```python
# AWQ storage is explicitly [K,N/8], [K/G,N], [K/G,N/8], all contiguous.
# A has no scale; its recipe entry is unused. Missing zero points mean zero.
y = gemm_scaled(a16, packed_b, b_scale=scales, b_zero_points=zeros,
                b_format="awq_int4", scale_recipe=((1, 1), (1, G)))

# E2M1 inputs store two values per byte: A[M,K/2], B[N,K/2].
# Both scales are uint8 E8M0 with shapes [M,K/32] and [N,K/32].
y = gemm_scaled(a4, b4, a_scale, b_scale,
                a_format="fp4_e2m1", b_format="fp4_e2m1",
                scale_recipe=((1, 32), (1, 32)), scale_encodings=("e8m0", "e8m0"))
```

AWQ output dtype equals its scale dtype. MXFP4 output defaults to BF16.
`gemm_batched_a16wfp4_online_quant` takes A[B,M,K], packed B[B,N,K/2], E8M0 B scales
[B,N,K/32]; A scales are generated in the kernel. Both sides use recipe
(1,32), and packed paths require K divisible by 32.

All interfaces accept `backend="auto"`, `"triton"`, `"gluon"`, `"tilelang"`.
Automatic selection queries the selected operation's own configuration file
before defaults. Explicit backend selection never retries another backend on
compile/execute failure. Unsupported combinations raise NotImplementedError;
a registered backend missing its runtime raises ImportError. Gluon retains the optimized E4M3FN/FP32 scale recipe. TileLang currently has no implementation rows.

The five files are `gemm_dense.json`, `gemm_scaled.json`, `gemm_batched.json`,
`gemm_grouped.json`, and `gemm_batched_a16wfp4_online_quant.json`. Each carries its interface identifier. Schema 5
separates `match` (arch, num_cus, MNK, dtype/scale/layout contracts) from
`config` (backend, implementation, tile and optional existing config file).
Tile is not a matching condition. No measurements, frequency, commit IDs or
evidence paths are stored. Current tuned entries reuse existing gfx946 48-CU
FP16 TT configurations. Source kernels and default configurations are unchanged.


## One table, one dispatcher

The call chain is public input validation -> interface JSON -> implementation
name lookup -> launcher. `catalog.py`, `registry.py`, per-backend `dispatch.py`,
`INTERFACES` registries and the `Implementation` class have been removed.

`boltops/gemm/dispatch.py` owns the only implementation table. For example:

```python
IMPLEMENTATIONS = {
    "triton.batched.native16": {
        "backend": "triton",
        "interface": "batched",
        "module": "boltops.gemm.triton.batched",
        "launcher": "launch_native",
        "configs": "_native_configs",
    },
}
```

Rows use module/function names to keep optional DSLs unloaded until needed.
`configs(problem)` supplies legal default configurations. An optional
`supports(problem, config)` function can validate a configuration independently;
when omitted, membership in the declared configurations is checked. An optional
`tuning_configs` names a larger explicitly requested tuning pool.

A new pipeline adds a row and its host launcher/configuration functions. A
backend with no implementation needs no empty table or forwarding module.
Backend-specific files (`triton/dense.py`, `scaled.py`, `batched.py`,
`grouped.py`, `quantized.py`) contain ABI/configuration helpers only; they do
not register or dispatch implementations. The selected plan contains an
implementation name string and its configuration, not a registration object.

## Winner selection and default policy

Schema 5 configuration files explicitly declare a default backend order:

```json
{
  "schema_version": 5,
  "interface": "gemm_batched",
  "default_backends": ["triton", "gluon", "tilelang"],
  "entries": []
}
```

Every entry names one measured winner (`config.backend`, `implementation`,
block size and optional source config). An entry's integer `priority` defaults
to zero; larger values override lower-priority matching records. This is an
explicit routing priority, not a timing measurement or a heuristic for tile
size. Exact-shape exceptions to a broader range should use a higher priority.

`auto` gathers matching records across available backends before choosing an
implementation. There must be exactly one highest-priority match. Equal-ranked
overlapping records, including duplicate copies, raise a configuration error;
JSON order and provider registration order cannot choose a winner. A selected
record whose implementation/configuration is stale or unsupported also raises
an error, rather than disguising it as a default selection.

A record for an unavailable backend cannot be used. With no usable record,
selection follows the interface's `default_backends` list, skipping unavailable
or unsupported implementations. Within a backend, the larger `default_priority` in the implementation table
wins, then implementation name provides a stable tie order. Candidate config order is owned by that implementation.
An explicit `backend=` filters records and defaults to only that backend.
Compilation/execution failures of a normal launch still propagate unchanged.

Recorded matches can describe dense, batched, grouped and packed/scaled
operations: exact shapes, batch/group metadata, operand packing/strides,
scale recipe/encoding, quantization, bias and alignment. Existing native-dense
range contracts remain supported. No implicit specificity score is used;
conflicting exact/range rules need explicit priorities.

## Explicit autotune utility

Normal `auto` dispatch never benchmarks. A separate utility is opt-in:

```python
import torch
from boltops.gemm.autotune import autotune
from boltops.gemm.dispatch import launch

# p is a complete GemmProblem; tensors contains the launch arguments.
# reference is independently computed. device_bench measures warmed device
# execution (or provide a PMD modeled-cycle measurement callback).
def validate(plan):
    launch(plan, p, **tensors)  # compile, warm up and execute this candidate
    torch.cuda.synchronize()
    torch.testing.assert_close(tensors["out"], reference)

def benchmark(plan):
    return device_bench(lambda: launch(plan, p, **tensors))

result = autotune(
    p, validate=validate, benchmark=benchmark,
    output_path=user_config_dir / f"gemm_{p.interface}.json",
)
```

The lower positive finite metric wins. Validation is mandatory and runs before
measurement. Failed candidates retain their error in `result.trials` and cannot
win; all-failed tuning raises without publishing a record. This failure loop
exists only in explicit tuning, not normal dispatch. Metric ties follow the
explicit default policy. The utility compares declared legal candidates; it
does not generate a full tile scan. Table rows may name a separate
`tuning_configs` function when their tuning pool differs from defaults;
callers may also explicitly supply a `plans` list.

Omit `output_path` for an in-memory result only. Same-host, same-user writers
use an external lock to preserve concurrent updates. Publishing atomically updates
one exact-shape winner, preserves unrelated records, and assigns enough
priority to override overlapping broader rules. Repeat tuning replaces the
same exact match without accumulating duplicates. Measurements/errors stay in
the result and are never written into the JSON.

Set `BOLTOPS_GEMM_CONFIG_DIR` to `user_config_dir` to use generated records in
later normal calls. Missing interface files in that directory fall back to
the packaged file. A successful publish clears the local config cache; after
manual external edits, call `performance_map.clear_cache()` or restart the
process. Tests for this utility use synthetic metrics; no broad PMD autotune
scan is run as part of ordinary regression validation.


## Generic match conditions

Schema 5 matches the attributes of `GemmProblem` directly. Nested objects use
partial matching; omitted fields are unconstrained. Arrays and null are exact
values. Dtypes use names without the `torch.` prefix. For example:

```json
{
  "arch": "gfx946",
  "num_cus": 48,
  "a": {"storage_dtype": "float8_e4m3fn", "format": "native"},
  "b": {"storage_dtype": "float8_e4m3fn", "format": "native"},
  "out_dtype": "bfloat16",
  "a_scale": {"dtype": "float32", "block": [1, 128], "encoding": "numeric"},
  "b_scale": {"dtype": "float32", "block": [128, 128], "encoding": "numeric"},
  "m": {"multiple_of": 128},
  "n": {"min": 256, "multiple_of": 256},
  "k": {"min": 256, "max": 6144, "multiple_of": 256}
}
```

All specified conditions must hold. Inclusive `min`/`max` and positive integer
`multiple_of` work on any numeric attribute, including batch/group counts and
CU count. Boolean values are distinct from numbers. Unknown fields/operators,
invalid divisors and reversed ranges raise errors, even if another condition
would not match. Each JSON rule must explicitly include `arch` and `num_cus`.

Attributes and the valid field schema are derived recursively from the data
classes, including optional scale/group objects. Adding a data-class field does
not require another matcher branch. `a_layout`/`b_layout` describe dense operand
contiguity; exact recorded winners also contain operand strides. Exact winner
publication uses this same attribute dictionary.

Schema 4 is rejected. Packaged rules have been migrated with their formerly
implicit native-format, no-bias, alignment, batch/group and layout restrictions
written explicitly; no hidden defaults are applied by the generic matcher.
