# 少伯 BU GEMM 支持需求

来源：`/xukang/GEMM支持清单.html`（生成日期 2026-10-12）。按算子类型及输入、输出精度分为 9 张表，保留 511 行需求和 2504 条算子实例；表 7–9 从来源抽取为单次 grouped GEMM 需求，保留 M/N/K、分组数、dtype 和 scale。

原 HTML SHA256：`187673895b820bfb89d6b2890ffb35fa378a183e8922238a8f96df2ff738d11c`。

本清单用于逐项记录支持和优化进度。状态已按截至 2026-10-20 的接口 smoke 证据更新，见 [smoke 验证依据](#smoke-evidence)。**“接口 smoke 通过”表示该算子族 / dtype 接口已有小尺寸 PASS，不表示本行完整 M/N/K、batch/分组数、scale 和分组寻址已逐项运行，也不表示性能已调优。** 实验详情与记录规范见 [调优记录](debug_tuning/TUNING.md)。

## 记录口径

- M/N/K 沿用 HTML 中的激活行数、输出通道数和规约维，不据此推断矩阵 stride。
- batch / 分组数 E 的有效值沿用源数据；bmm 表使用 batch 列，grouped_gemm 表将原 experts 列记为分组数 E，linear 表原本全空的该列省略。M/N/K 原值不变，不以 M×E 替换 M，也不据来源 M 假定各组等长；ragged 各组 M 与 offsets 须由调用方提供。
- 输出精度指写回显存的精度，与累加精度分开验证。
- 有量化的表保留 scale 原始信息；在整张表全空或不适用的列直接省略。含推断值的条目接入模型时需核实。
- 状态分为接口级「接口 smoke 通过」「待 grouped GEMM smoke」「接口 smoke 受阻」与需求尺寸级「待验证」「编译通过」「完整尺寸正确性通过」「已调优」；不得把接口 smoke 或编译通过升级为本行完整尺寸 PASS。
- BoltOps 函数 / 实现列链接实际源码定义；它表示当前可对接的接口或内部 kernel，不替代模型布局、量化格式与分组寻址适配。表 7–9 只关注单次 grouped GEMM，输入量化值、scale 和分组索引由测试预先准备；不把在线量化、路由生成、激活或结果合并纳入验收。
- ResNet50 INT8 隐式 GEMM 卷积 24 类和末端 FC 不在这 511 行中，仍以原 HTML 及当前支持说明为准。

模型 / 配置列使用 [C 编号](#model-config-index) 引用文末的完整组合，缩短主表宽度；点击编号可查看模型、精度及序列/规模配置。

## 分类索引

| 表 | 算子 | A × B → 输出 | 需求行数 | 实例数 |
| --- | --- | --- | ---: | ---: |
| [表 1](#table-1) | bmm | BF16 × BF16 → BF16 | 29 | 194 |
| [表 2](#table-2) | linear | FP8E4M3 × FP8E4M3 → BF16 | 207 | 1206 |
| [表 3](#table-3) | linear | BF16 × BF16 → BF16 | 133 | 326 |
| [表 4](#table-4) | linear | FP32 × FP32 → FP32 | 35 | 218 |
| [表 5](#table-5) | linear | BF16 × BF16 → FP32 | 27 | 172 |
| [表 6](#table-6) | linear | BF16 × INT4 → BF16 | 4 | 4 |
| [表 7](#table-7) | grouped_gemm | FP8E4M3 × FP8E4M3 → BF16 | 36 | 70 |
| [表 8](#table-8) | grouped_gemm | FP8E4M3 × FP4E2M1 → BF16 | 34 | 304 |
| [表 9](#table-9) | grouped_gemm | BF16 × INT4 → BF16 | 6 | 10 |

<a id="smoke-evidence"></a>

## Smoke 验证依据（2026-10-20）

实测环境为 node60 / PMD gfx946 / Triton release/3.6.x。按“每接口一个小 size”的验收口径记录；同一证据会被同族多条需求引用，**469 行显示接口 smoke 通过不代表执行了 469 个需求尺寸测试**。表 7 的 36 行、表 9 的 6 行尚待直接 grouped GEMM smoke。原有完整尺寸任务保持暂停，本次只整理既有记录。

| 证据 | 接口与实际测试范围 | 最终结果 / 条件 |
| --- | --- | --- |
| <a id="smoke-bmm"></a>S24 | `batched_gemm_bf16`，BF16→BF16，B×M×N×K=2×16×128×32 | PASS，max_abs=0、mismatch=0 |
| <a id="smoke-fp8"></a>FP8 | `gemm_a8w8_blockscale` 15/15、`w8a8_triton_block_scaled_mm` 17/17；包含 FP32/E8M0、A K128/整行、K 尾块、前导维和默认配置 | 32/32 PASS，max_abs=0、mismatch=0；另有 `triton_scaled_mm` 14/14 兼容 smoke，但其 W scale 不沿 K 分组，不映射本表需求 |
| <a id="smoke-bf16"></a>S05 | `gemm_a16w16`，BF16→BF16，M×N×K=64×128×64 | PASS，max_abs=0、mismatch=0 |
| <a id="smoke-fp32"></a>S03 修复后 | `matmul_persistent`，FP32→FP32，64×64×64 | PASS，4096 元素 mismatch=0；相对 FP64 golden 的 max_abs≈1.05e-7。依赖 S03 LEGACY 补丁；旧结果的 2906/4096 FAIL 已被此结果替代 |
| <a id="smoke-bf16-fp32"></a>S06 | `gemm_a16w16`，BF16→FP32，64×128×64 | PASS，max_abs≈5.96e-8、mismatch=0 |
| <a id="smoke-awq"></a>S18 | `awq_gemm_triton`，BF16×INT4→BF16，16×64×64 | PASS，max_abs=0、mismatch=0；仅验证该测试的 AWQ 打包及 scale/zero-point 配置 |
| <a id="smoke-grouped-fp8"></a>grouped FP8 | 单次 FP8×FP8 grouped GEMM；预先准备 FP8 A、权重、scale 及分组寻址元数据 | 待直接 smoke；接口适配及 A/W scale 组合需单独验证 |
| <a id="smoke-fp8fp4"></a>S28 / S31 | `grouped_gemm_a8wfp4`，FP8×FP4→BF16；等长 E×M×N×K=3×5×35×130，ragged E=4/T=57/N=35/K=97、含空组 | PASS，复用已审计的原回归；K32/K128 与浮点/E8M0 scale 组合见原 15 项回归。FP16/FP32 输出另有 S29/S30/S32/S33 PASS |
| <a id="smoke-grouped-int4"></a>grouped INT4 | 单次 BF16×INT4 grouped GEMM；预先准备打包权重、scale/zero-point 及分组寻址元数据 | 待直接 smoke；打包格式、分组粒度和寻址需单独验证 |

原始记录均在 Git 工作区之外，统一前缀为 `/xukang/code/pmd-triton-sessions/`：

- S24/S05/S06/S18/S28/S31：`fp8-live-records-20261018/smoke-acceptance-20261018/results.json`；复用项的原路径和 SHA256 均记录在对应 evidence 中。
- S03 当前证据：`fp8-live-records-20261018/blocked-diagnosis-20261018/s03-after-relink/S03/result.json`，不是旧 smoke 总表中的 FAIL。
- FP8 两个接口：`fp8-dense-smoke-20261020/final-results.json/csv`，实现对应候选 `0cc243e`（统一分支中等价重放为 `be858e0`）；汇总见 [当前支持能力](README.md)。可选流水优化及资源限制另见 [调优记录](debug_tuning/TUNING.md)，不覆盖或撤销这里普通路径的接口 smoke。

<a id="table-1"></a>

## 表 1：bmm · BF16 × BF16 → BF16

29 行需求，累计 194 条实例。编号在本清单中全局唯一。

本表仅记录 BF16 BMM；函数使用 `B[batch,N,K]` 权重布局，模型调用需匹配 stride。

| 编号 | M × N × K | batch | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| G001 | 6 × 256 × 512 | 64 | decode | 1 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G002 | 6 × 512 × 192 | 64 | decode | 1 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G003 | 12 × 256 × 512 | 64 | decode | 1 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G004 | 12 × 512 × 192 | 64 | decode | 1 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G005 | 16 × 128 × 512 | 64 | decode | 2 | [C003](#c003) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G006 | 16 × 512 × 128 | 64 | decode | 2 | [C003](#c003) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G007 | 24 × 256 × 512 | 64 | decode | 2 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G008 | 24 × 512 × 192 | 64 | decode | 2 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G009 | 48 × 256 × 512 | 64 | decode | 2 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G010 | 48 × 512 × 192 | 64 | decode | 2 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G011 | 80 × 1024 × 4096 | 16 | decode | 6 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G012 | 96 × 1024 × 4096 | 16 | decode | 8 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G013 | 160 × 1024 × 4096 | 8 | decode | 3 | [C005](#c005) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G014 | 192 × 1024 × 4096 | 8 | decode | 4 | [C005](#c005) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G015 | 240 × 1024 × 4096 | 8 | decode | 9 | [C006](#c006) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G016 | 288 × 1024 × 4096 | 8 | decode | 12 | [C006](#c006) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G017 | 2560 × 256 × 512 | 64 | prefill | 4 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G018 | 2560 × 512 × 192 | 64 | prefill | 4 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G019 | 2560 × 1024 × 4096 | 16 | prefill | 12 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G020 | 2560 × 1024 × 4096 | 8 | prefill | 4 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G021 | 3584 × 1024 × 4096 | 8 | prefill | 16 | [C007](#c007) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G022 | 4608 × 256 × 512 | 64 | prefill | 4 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G023 | 4608 × 512 × 192 | 64 | prefill | 4 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G024 | 4608 × 1024 × 4096 | 16 | prefill | 4 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G025 | 5120 × 1024 × 4096 | 8 | prefill | 16 | [C008](#c008) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G026 | 6656 × 256 × 512 | 64 | prefill | 8 | [C009](#c009) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G027 | 6656 × 512 × 192 | 64 | prefill | 8 | [C009](#c009) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G028 | 6656 × 1024 × 4096 | 8 | prefill | 36 | [C010](#c010) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |
| G029 | 6656 × 1024 × 4096 | 16 | prefill | 16 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 | [S24](#smoke-bmm) |

<a id="table-2"></a>

## 表 2：linear · FP8E4M3 × FP8E4M3 → BF16

207 行需求，累计 1206 条实例。编号在本清单中全局唯一。

两个函数均已覆盖 A 的 K128 / 整行 scale；W 仍按每个 N、K128 分组。FP32/E8M0 scale 均有小尺寸证据，原表中含推断的 scale 类型仍需接入时确认。

| 编号 | M × N × K | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| G030 | 4 × 3072 × 4096 | decode | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G031 | 4 × 4096 × 1536 | decode | 1 | FP32@1×1536 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G032 | 4 × 4096 × 8192 | decode | 1 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G033 | 4 × 10240 × 4096 | decode | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G034 | 6 × 128 × 6144 | decode | 1 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G035 | 6 × 2624 × 6144 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G036 | 6 × 4096 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G037 | 6 × 4096 × 6144 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G038 | 6 × 6144 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G039 | 6 × 6144 × 16384 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G040 | 6 × 16384 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G041 | 12 × 128 × 6144 | decode | 1 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G042 | 12 × 2624 × 6144 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G043 | 12 × 4096 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G044 | 12 × 4096 × 6144 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G045 | 12 × 6144 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G046 | 12 × 6144 × 16384 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G047 | 12 × 16384 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G048 | 16 × 3072 × 4096 | decode | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G049 | 16 × 4096 × 1536 | decode | 1 | FP32@1×1536 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G050 | 16 × 4096 × 8192 | decode | 2 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G051 | 16 × 4096 × 13312 | decode | 1 | FP32@1×13312 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G052 | 16 × 10240 × 4096 | decode | 2 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G053 | 16 × 26624 × 4096 | decode | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G054 | 24 × 128 × 6144 | decode | 2 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G055 | 24 × 2624 × 6144 | decode | 2 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G056 | 24 × 4096 × 2048 | decode | 2 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G057 | 24 × 4096 × 6144 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G058 | 24 × 6144 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G059 | 24 × 6144 × 12288 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G060 | 24 × 6144 × 16384 | decode | 2 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G061 | 24 × 16384 × 2048 | decode | 2 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G062 | 24 × 24576 × 6144 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G063 | 48 × 128 × 6144 | decode | 2 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G064 | 48 × 2048 × 4096 | decode | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G065 | 48 × 2624 × 6144 | decode | 2 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G066 | 48 × 4096 × 1024 | decode | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G067 | 48 × 4096 × 2048 | decode | 2 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G068 | 48 × 4096 × 4096 | decode | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G069 | 48 × 4096 × 6144 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G070 | 48 × 6144 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G071 | 48 × 6144 × 12288 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G072 | 48 × 6144 × 16384 | decode | 2 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G073 | 48 × 8704 × 4096 | decode | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G074 | 48 × 16384 × 2048 | decode | 2 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G075 | 48 × 24576 × 6144 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G076 | 64 × 1024 × 2048 | decode | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G077 | 64 × 2048 × 512 | decode | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G078 | 64 × 2048 × 4096 | decode | 2 | FP32@1×128(1) / FP8E8M0/FP32@1×128 (推断)(1) | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G079 | 64 × 9216 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G080 | 64 × 12288 × 2048 | decode | 1 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G081 | 80 × 512 × 7168 | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G082 | 80 × 1536 × 7168 | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G083 | 80 × 6144 × 7168 | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G084 | 80 × 7168 × 3072 | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G085 | 80 × 7168 × 16384 | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G086 | 80 × 65536 × 1536 | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G087 | 96 × 512 × 7168 | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G088 | 96 × 1536 × 7168 | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G089 | 96 × 6144 × 7168 | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G090 | 96 × 7168 × 3072 | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G091 | 96 × 7168 × 16384 | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G092 | 96 × 8192 × 1536 | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G093 | 96 × 65536 × 1536 | decode | 8 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G094 | 160 × 512 × 4096 | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G095 | 160 × 1024 × 4096 | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G096 | 160 × 4096 × 2048 | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G097 | 160 × 4096 × 4096 | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G098 | 160 × 4096 × 8192 | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G099 | 160 × 32768 × 1024 | decode | 3 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G100 | 192 × 512 × 4096 | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G101 | 192 × 1024 × 4096 | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G102 | 192 × 2048 × 4096 | decode | 2 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G103 | 192 × 4096 × 1024 | decode | 2 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G104 | 192 × 4096 × 2048 | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G105 | 192 × 4096 × 4096 | decode | 6 | FP8E8M0@1×128(4) / FP32@1×128(2) | 离线 1×128 | [C014](#c014) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G106 | 192 × 4096 × 8192 | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G107 | 192 × 8192 × 1024 | decode | 2 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G108 | 192 × 8704 × 4096 | decode | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G109 | 192 × 10240 × 4096 | decode | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G110 | 192 × 32768 × 1024 | decode | 4 | FP8E8M0@1×128 | 离线 1×128 | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G111 | 240 × 512 × 4096 | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G112 | 240 × 1024 × 4096 | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G113 | 240 × 4096 × 2048 | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G114 | 240 × 4096 × 4096 | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G115 | 240 × 4096 × 8192 | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G116 | 240 × 32768 × 1024 | decode | 9 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G117 | 288 × 512 × 4096 | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G118 | 288 × 1024 × 4096 | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G119 | 288 × 4096 × 2048 | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G120 | 288 × 4096 × 4096 | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G121 | 288 × 4096 × 8192 | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G122 | 288 × 8192 × 1024 | decode | 6 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G123 | 288 × 32768 × 1024 | decode | 12 | FP8E8M0@1×128 | 离线 1×128 | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G124 | 512 × 384 × 4096 | prefill | 2 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G125 | 512 × 3328 × 4096 | prefill | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G126 | 512 × 4096 × 192 | prefill | 2 | FP32@1×192 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G127 | 512 × 4096 × 1664 | prefill | 1 | FP32@1×1664 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G128 | 512 × 4096 × 8192 | prefill | 4 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G129 | 512 × 10240 × 4096 | prefill | 4 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G130 | 2560 × 128 × 6144 | prefill | 4 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G131 | 2560 × 512 × 6144 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G132 | 2560 × 512 × 7168 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G133 | 2560 × 768 × 7168 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G134 | 2560 × 1024 × 2048 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G135 | 2560 × 1536 × 7168 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G136 | 2560 × 2048 × 512 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G137 | 2560 × 2048 × 4096 | prefill | 2 | FP32@1×128(1) / FP8E8M0/FP32@1×128 (推断)(1) | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G138 | 2560 × 2624 × 6144 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G139 | 2560 × 3072 × 6144 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G140 | 2560 × 4096 × 2048 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G141 | 2560 × 6144 × 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G142 | 2560 × 6144 × 1536 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G143 | 2560 × 6144 × 16384 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G144 | 2560 × 7168 × 384 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G145 | 2560 × 7168 × 8192 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G146 | 2560 × 7168 × 16384 | prefill | 12 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G147 | 2560 × 8192 × 1536 | prefill | 8 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G148 | 2560 × 9216 × 2048 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G149 | 2560 × 12288 × 2048 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G150 | 2560 × 16384 × 2048 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G151 | 2560 × 32768 × 1536 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G152 | 2560 × 65536 × 1536 | prefill | 12 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G153 | 3584 × 256 × 4096 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G154 | 3584 × 512 × 4096 | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G155 | 3584 × 1024 × 4096 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G156 | 3584 × 4096 × 128 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G157 | 3584 × 4096 × 256 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G158 | 3584 × 4096 × 8192 | prefill | 19 | FP8E8M0@1×128(16) / FP8E8M0/FP32@1×128 (推断)(2) / FP32@1×128(1) | 离线 1×128 | [C015](#c015) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G159 | 3584 × 8192 × 1024 | prefill | 8 | FP8E8M0@1×128 | 离线 1×128 | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G160 | 3584 × 17408 × 4096 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G161 | 3584 × 20480 × 4096 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G162 | 3584 × 32768 × 1024 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G163 | 4096 × 3328 × 4096 | prefill | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G164 | 4096 × 4096 × 1664 | prefill | 1 | FP32@1×1664 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G165 | 4608 × 128 × 6144 | prefill | 4 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G166 | 4608 × 512 × 6144 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G167 | 4608 × 512 × 7168 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G168 | 4608 × 768 × 7168 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G169 | 4608 × 1536 × 7168 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G170 | 4608 × 2624 × 6144 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G171 | 4608 × 3072 × 6144 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G172 | 4608 × 4096 × 2048 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G173 | 4608 × 6144 × 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G174 | 4608 × 6144 × 1536 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G175 | 4608 × 6144 × 16384 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G176 | 4608 × 7168 × 384 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G177 | 4608 × 7168 × 16384 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G178 | 4608 × 8192 × 1536 | prefill | 2 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G179 | 4608 × 16384 × 2048 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G180 | 4608 × 65536 × 1536 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G181 | 5120 × 512 × 4096 | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G182 | 5120 × 1024 × 4096 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G183 | 5120 × 4096 × 256 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G184 | 5120 × 4096 × 8192 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G185 | 5120 × 8192 × 1024 | prefill | 8 | FP8E8M0@1×128 | 离线 1×128 | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G186 | 5120 × 32768 × 1024 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G187 | 6656 × 128 × 6144 | prefill | 8 | FP8E8M0/FP32@1×128 (推断) | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G188 | 6656 × 256 × 4096 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G189 | 6656 × 384 × 4096 | prefill | 2 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G190 | 6656 × 512 × 4096 | prefill | 64 | FP8E8M0@1×128 | 离线 1×128 | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G191 | 6656 × 512 × 6144 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G192 | 6656 × 512 × 7168 | prefill | 20 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G193 | 6656 × 768 × 7168 | prefill | 20 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G194 | 6656 × 1024 × 2048 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G195 | 6656 × 1024 × 4096 | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G196 | 6656 × 1536 × 7168 | prefill | 20 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G197 | 6656 × 2048 × 512 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G198 | 6656 × 2048 × 4096 | prefill | 2 | FP32@1×128(1) / FP8E8M0/FP32@1×128 (推断)(1) | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G199 | 6656 × 2624 × 6144 | prefill | 8 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G200 | 6656 × 3072 × 6144 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G201 | 6656 × 3328 × 4096 | prefill | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G202 | 6656 × 4096 × 128 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G203 | 6656 × 4096 × 192 | prefill | 2 | FP32@1×192 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G204 | 6656 × 4096 × 256 | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G205 | 6656 × 4096 × 1664 | prefill | 1 | FP32@1×1664 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G206 | 6656 × 4096 × 2048 | prefill | 8 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G207 | 6656 × 4096 × 8192 | prefill | 39 | FP8E8M0@1×128(32) / FP8E8M0/FP32@1×128 (推断)(6) / FP32@1×128(1) | 离线 1×128 | [C017](#c017) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G208 | 6656 × 6144 × 256 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G209 | 6656 × 6144 × 1536 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G210 | 6656 × 6144 × 16384 | prefill | 8 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G211 | 6656 × 7168 × 384 | prefill | 20 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G212 | 6656 × 7168 × 8192 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G213 | 6656 × 7168 × 16384 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G214 | 6656 × 8192 × 1024 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G215 | 6656 × 8192 × 1536 | prefill | 10 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G216 | 6656 × 9216 × 2048 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G217 | 6656 × 10240 × 4096 | prefill | 4 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G218 | 6656 × 12288 × 2048 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G219 | 6656 × 16384 × 2048 | prefill | 8 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G220 | 6656 × 17408 × 4096 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G221 | 6656 × 20480 × 4096 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G222 | 6656 × 32768 × 1024 | prefill | 32 | FP8E8M0@1×128 | 离线 1×128 | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G223 | 6656 × 32768 × 1536 | prefill | 4 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G224 | 6656 × 65536 × 1536 | prefill | 16 | FP8E8M0@1×128 | 离线 1×128 | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G225 | 20480 × 3072 × 6144 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G226 | 20480 × 6144 × 1536 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G227 | 28672 × 2560 × 4096 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G228 | 28672 × 4096 × 1024 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G229 | 36864 × 3072 × 6144 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G230 | 36864 × 6144 × 1536 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G231 | 53248 × 2560 × 4096 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G232 | 53248 × 3072 × 6144 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G233 | 53248 × 3328 × 4096 | prefill | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G234 | 53248 × 4096 × 1024 | prefill | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G235 | 53248 × 4096 × 1664 | prefill | 1 | FP32@1×1664 | 离线 1×128 | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |
| G236 | 53248 × 6144 × 1536 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 | [FP8 32/32](#smoke-fp8) |

<a id="table-3"></a>

## 表 3：linear · BF16 × BF16 → BF16

133 行需求，累计 326 条实例。编号在本清单中全局唯一。

本表对应 BF16 输入、BF16 输出；`gemm_a16w16` 的权重布局为 `[N,K]`。

| 编号 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| G237 | 1 × 154880 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G238 | 2 × 64640 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G239 | 2 × 129280 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G240 | 3 × 120832 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G241 | 4 × 192 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G242 | 4 × 4096 × 8192 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G243 | 4 × 7168 × 21504 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G244 | 4 × 120832 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G245 | 4 × 154880 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G246 | 4 × 163840 × 7168 | decode,prefill | 3 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G247 | 6 × 32 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G248 | 6 × 256 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G249 | 6 × 6144 × 12288 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G250 | 6 × 154880 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G251 | 8 × 16160 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G252 | 8 × 19360 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G253 | 8 × 129280 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G254 | 8 × 248320 × 4096 | prefill | 2 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G255 | 12 × 32 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G256 | 12 × 64 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G257 | 12 × 256 × 2048 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G258 | 12 × 256 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G259 | 12 × 1024 × 2048 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G260 | 12 × 2048 × 512 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G261 | 12 × 2048 × 4096 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G262 | 12 × 6144 × 12288 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G263 | 12 × 9216 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G264 | 12 × 12288 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G265 | 12 × 154880 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G266 | 12 × 248320 × 2048 | decode,prefill | 3 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G267 | 16 × 192 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G268 | 16 × 384 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G269 | 16 × 2112 × 7168 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G270 | 16 × 4096 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G271 | 16 × 7168 × 2048 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G272 | 16 × 7168 × 8192 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G273 | 16 × 7168 × 18432 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G274 | 16 × 12288 × 1536 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G275 | 16 × 16160 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G276 | 16 × 36864 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G277 | 16 × 120832 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G278 | 16 × 163840 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G279 | 24 × 32 × 6144 | decode | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G280 | 24 × 256 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G281 | 24 × 15104 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G282 | 24 × 154880 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G283 | 32 × 19360 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G284 | 32 × 20480 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G285 | 48 × 32 × 6144 | decode | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G286 | 48 × 256 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G287 | 48 × 512 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G288 | 48 × 4096 × 8192 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G289 | 48 × 124160 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G290 | 48 × 154880 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G291 | 64 × 64 × 2048 | decode | 1 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G292 | 64 × 256 × 2048 | decode | 2 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G293 | 64 × 16160 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G294 | 64 × 31040 × 4096 | prefill | 2 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G295 | 64 × 248320 × 2048 | decode,prefill | 3 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G296 | 80 × 129280 × 7168 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G297 | 96 × 64 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G298 | 96 × 512 × 7168 | decode | 6 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G299 | 96 × 7168 × 21504 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G300 | 96 × 129280 × 7168 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G301 | 160 × 129280 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G302 | 192 × 64 × 4096 | decode | 3 | [C014](#c014) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G303 | 192 × 512 × 4096 | decode | 5 | [C014](#c014) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G304 | 192 × 4096 × 12288 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G305 | 192 × 124160 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G306 | 192 × 129280 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G307 | 240 × 129280 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G308 | 288 × 64 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G309 | 288 × 512 × 4096 | decode | 9 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G310 | 288 × 4096 × 12288 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G311 | 288 × 129280 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G312 | 512 × 192 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G313 | 2048 × 64 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G314 | 2048 × 256 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G315 | 2048 × 1024 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G316 | 2048 × 2048 × 512 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G317 | 2048 × 2048 × 4096 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G318 | 2048 × 9216 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G319 | 2048 × 12288 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G320 | 2560 × 32 × 6144 | prefill | 4 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G321 | 2560 × 64 × 2048 | prefill | 1 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G322 | 2560 × 64 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G323 | 2560 × 256 × 2048 | prefill | 2 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G324 | 2560 × 256 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G325 | 3584 × 64 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G326 | 3584 × 128 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G327 | 3584 × 512 × 4096 | prefill | 4 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G328 | 4608 × 32 × 6144 | prefill | 4 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G329 | 4608 × 64 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G330 | 4608 × 256 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G331 | 5120 × 64 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G332 | 5120 × 384 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G333 | 5120 × 512 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G334 | 5120 × 2112 × 7168 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G335 | 5120 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G336 | 5120 × 7168 × 256 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G337 | 5120 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G338 | 5120 × 7168 × 8192 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G339 | 5120 × 12288 × 1536 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G340 | 5120 × 16384 × 512 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G341 | 6656 × 32 × 6144 | prefill | 8 | [C009](#c009) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G342 | 6656 × 64 × 2048 | prefill | 2 | [C019](#c019) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G343 | 6656 × 64 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G344 | 6656 × 64 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G345 | 6656 × 128 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G346 | 6656 × 192 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G347 | 6656 × 256 × 2048 | prefill | 4 | [C019](#c019) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G348 | 6656 × 256 × 6144 | prefill | 4 | [C009](#c009) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G349 | 6656 × 384 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G350 | 6656 × 512 × 4096 | prefill | 4 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G351 | 6656 × 512 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G352 | 6656 × 1024 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G353 | 6656 × 2048 × 512 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G354 | 6656 × 2048 × 4096 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G355 | 6656 × 2112 × 7168 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G356 | 6656 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G357 | 6656 × 7168 × 256 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G358 | 6656 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G359 | 6656 × 7168 × 8192 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G360 | 6656 × 9216 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G361 | 6656 × 12288 × 1536 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G362 | 6656 × 12288 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G363 | 6656 × 16384 × 512 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G364 | 28672 × 16 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G365 | 40960 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G366 | 40960 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G367 | 53248 × 16 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G368 | 53248 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |
| G369 | 53248 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S05](#smoke-bf16) |

<a id="table-4"></a>

## 表 4：linear · FP32 × FP32 → FP32

35 行需求，累计 218 条实例。编号在本清单中全局唯一。

本表的接口级 smoke PASS 依赖当前 Triton 的 S03 LEGACY 补丁；不能推广到未带该补丁的环境。

| 编号 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| G370 | 16 × 129280 × 512 | decode | 10 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G371 | 32 × 129280 × 256 | decode | 5 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G372 | 48 × 129280 × 256 | decode | 15 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G373 | 80 × 1 × 7680 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G374 | 80 × 4 × 28672 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G375 | 80 × 384 × 7168 | decode | 6 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G376 | 96 × 4 × 28672 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G377 | 96 × 384 × 7168 | decode | 8 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G378 | 160 × 1 × 4352 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G379 | 160 × 4 × 16384 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G380 | 160 × 256 × 4096 | decode | 3 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G381 | 192 × 4 × 16384 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G382 | 192 × 256 × 4096 | decode | 4 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G383 | 240 × 1 × 4352 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G384 | 240 × 4 × 16384 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G385 | 240 × 256 × 4096 | decode | 9 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G386 | 288 × 4 × 16384 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G387 | 288 × 256 × 4096 | decode | 12 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G388 | 2560 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G389 | 2560 × 384 × 7168 | prefill | 16 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G390 | 3584 × 4 × 16384 | prefill | 2 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G391 | 3584 × 256 × 4096 | prefill | 16 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G392 | 4608 × 384 × 7168 | prefill | 4 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G393 | 5120 × 4 × 16384 | prefill | 2 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G394 | 5120 × 256 × 4096 | prefill | 16 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G395 | 6656 × 4 × 16384 | prefill | 4 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G396 | 6656 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G397 | 6656 × 256 × 4096 | prefill | 32 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G398 | 6656 × 384 × 7168 | prefill | 20 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G399 | 20480 × 4 × 28672 | prefill | 1 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G400 | 28672 × 4 × 16384 | prefill | 2 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G401 | 36864 × 4 × 28672 | prefill | 1 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G402 | 40960 × 4 × 16384 | prefill | 2 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G403 | 53248 × 4 × 16384 | prefill | 4 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |
| G404 | 53248 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） | [S03 修复后](#smoke-fp32) |

<a id="table-5"></a>

## 表 5：linear · BF16 × BF16 → FP32

27 行需求，累计 172 条实例。编号在本清单中全局唯一。

本表使用 `gemm_a16w16(..., dtype=torch.float32)`，保留 BF16 输入、FP32 输出语义。

| 编号 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| G405 | 96 × 512 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G406 | 96 × 1024 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G407 | 96 × 2048 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G408 | 192 × 512 × 4096 | decode | 2 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G409 | 192 × 1024 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G410 | 192 × 2048 × 4096 | decode | 2 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G411 | 288 × 512 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G412 | 288 × 1024 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G413 | 288 × 2048 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G414 | 2560 × 512 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G415 | 2560 × 1024 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G416 | 2560 × 2048 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G417 | 3584 × 512 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G418 | 3584 × 1024 × 4096 | prefill | 4 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G419 | 3584 × 2048 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G420 | 4608 × 512 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G421 | 4608 × 1024 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G422 | 4608 × 2048 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G423 | 5120 × 512 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G424 | 5120 × 1024 × 4096 | prefill | 4 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G425 | 5120 × 2048 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G426 | 6656 × 512 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G427 | 6656 × 512 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G428 | 6656 × 1024 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G429 | 6656 × 1024 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G430 | 6656 × 2048 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |
| G431 | 6656 × 2048 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 | [S06](#smoke-bf16-fp32) |

<a id="table-6"></a>

## 表 6：linear · BF16 × INT4 → BF16

4 行需求，累计 4 条实例。编号在本清单中全局唯一。

现有 AWQ 入口需要匹配打包权重、scale 和 zero-point。S18 只证明 BF16×INT4 小尺寸接口通过；需求未给出的 group size / zero-point / 打包格式不据此补写。

| 编号 | M × N × K | 阶段 | 频次 | 权重 scale | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| G432 | 4 × 7168 × 8192 | decode | 1 | 离线 per-group | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 | [S18](#smoke-awq) |
| G433 | 4 × 7168 × 12288 | decode | 1 | 离线 per-group | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 | [S18](#smoke-awq) |
| G434 | 4 × 24576 × 7168 | decode | 1 | 离线 per-group | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 | [S18](#smoke-awq) |
| G435 | 4 × 24576 × 14336 | decode | 1 | 离线 per-group | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 | [S18](#smoke-awq) |

<a id="table-7"></a>

## 表 7：grouped_gemm · FP8E4M3 × FP8E4M3 → BF16

36 行需求，累计 70 条实例。编号在本清单中全局唯一。

本表抽取单次 FP8×FP8 grouped GEMM 的 M/N/K、分组数 E、A/W scale 需求。实现列只链接现有内部 grouped kernel，尚无已验证的独立公开包装接口；测试需直接提供预量化 A/W、scale 与 sorted-token 等固定分组寻址元数据。A 整行 scale / W K128 等组合仍待适配与直接 smoke，状态不变。

| 编号 | M × N × K | 分组数 E | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G436 | 3 × 1024 × 2048 | 221 | decode | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G437 | 3 × 2048 × 512 | 221 | decode | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G438 | 3 × 3072 × 4096 | 11 | decode | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G439 | 3 × 4096 × 1536 | 11 | decode | 1 | FP32@1×1536 | 离线 1×128 | [C011](#c011) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G440 | 4 × 2048 × 4096 | 63 | decode | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G441 | 4 × 4096 × 1024 | 63 | decode | 1 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G442 | 4 × 4096 × 6144 | 15 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G443 | 4 × 6144 × 2048 | 15 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G444 | 6 × 4096 × 6144 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G445 | 6 × 6144 × 2048 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G446 | 11 × 3072 × 4096 | 12 | decode | 1 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G447 | 11 × 4096 × 1536 | 12 | decode | 1 | FP32@1×1536 | 离线 1×128 | [C011](#c011) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G448 | 12 × 4096 × 6144 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G449 | 12 × 6144 × 2048 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G450 | 15 × 2048 × 4096 | 64 | decode | 2 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G451 | 15 × 4096 × 1024 | 64 | decode | 2 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G452 | 24 × 4096 × 6144 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G453 | 24 × 6144 × 2048 | 16 | decode | 1 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G454 | 80 × 1024 × 2048 | 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G455 | 80 × 2048 × 512 | 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G456 | 171 × 3072 × 4096 | 24 | prefill | 2 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G457 | 171 × 4096 × 1536 | 24 | prefill | 2 | FP32@1×1536 | 离线 1×128 | [C011](#c011) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G458 | 208 × 1024 × 2048 | 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G459 | 208 × 2048 × 512 | 256 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C013](#c013) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G460 | 560 × 2048 × 4096 | 64 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G461 | 560 × 4096 × 1024 | 64 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G462 | 640 × 4096 × 6144 | 32 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G463 | 640 × 6144 × 2048 | 32 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C002](#c002) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G464 | 1040 × 2048 × 4096 | 64 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G465 | 1040 × 4096 × 1024 | 64 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C012](#c012) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G466 | 1152 × 4096 × 6144 | 32 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G467 | 1152 × 6144 × 2048 | 32 | prefill | 2 | FP32@1×128 | 离线 1×128 | [C001](#c001) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G468 | 1664 × 4096 × 6144 | 32 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G469 | 1664 × 6144 × 2048 | 32 | prefill | 4 | FP32@1×128 | 离线 1×128 | [C009](#c009) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G470 | 2219 × 3072 × 4096 | 24 | prefill | 2 | FP32@1×4096 | 离线 1×128 | [C011](#c011) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |
| G471 | 2219 × 4096 × 1536 | 24 | prefill | 2 | FP32@1×1536 | 离线 1×128 | [C011](#c011) | [FP8 grouped kernel（内部）][fn-grouped-fp8-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-fp8) |

<a id="table-8"></a>

## 表 8：grouped_gemm · FP8E4M3 × FP4E2M1 → BF16

34 行需求，累计 304 条实例。编号在本清单中全局唯一。

`grouped_gemm_a8wfp4` 支持等长及 ragged（含空组）；A scale 可为 K32/K128，W scale 为 K32。原完整需求的 34/34 编译通过不等于 34/34 精度通过；这里更新的是接口 smoke 状态。

| 编号 | M × N × K | 分组数 E | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| G472 | 2 × 1024 × 2048 | 80 | decode | 2 | FP8E8M0@1×32 | 离线 1×32 (MXFP4) | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G473 | 2 × 2048 × 512 | 80 | decode | 2 | FP32@1×32 | 离线 1×32 (MXFP4) | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G474 | 20 × 6144 × 7168 | 24 | decode | 6 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G475 | 20 × 7168 × 3072 | 24 | decode | 6 | FP32@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G476 | 24 × 6144 × 7168 | 24 | decode | 8 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G477 | 24 × 7168 × 3072 | 24 | decode | 8 | FP32@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G478 | 30 × 4096 × 2048 | 32 | decode | 3 | FP32@1×128 | 离线 1×32 (MXFP4) | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G479 | 30 × 4096 × 4096 | 32 | decode | 3 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G480 | 36 × 4096 × 2048 | 32 | decode | 4 | FP32@1×128 | 离线 1×32 (MXFP4) | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G481 | 36 × 4096 × 4096 | 32 | decode | 4 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G482 | 45 × 4096 × 2048 | 32 | decode | 9 | FP32@1×128 | 离线 1×32 (MXFP4) | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G483 | 45 × 4096 × 4096 | 32 | decode | 9 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G484 | 54 × 4096 × 2048 | 32 | decode | 12 | FP32@1×128 | 离线 1×32 (MXFP4) | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G485 | 54 × 4096 × 4096 | 32 | decode | 12 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G486 | 64 × 1024 × 2048 | 256 | prefill | 2 | FP8E8M0@1×32 | 离线 1×32 (MXFP4) | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G487 | 64 × 2048 × 512 | 256 | prefill | 2 | FP32@1×32 | 离线 1×32 (MXFP4) | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G488 | 160 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G489 | 160 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G490 | 208 × 1024 × 2048 | 256 | prefill | 2 | FP8E8M0@1×32 | 离线 1×32 (MXFP4) | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G491 | 208 × 2048 × 512 | 256 | prefill | 2 | FP32@1×32 | 离线 1×32 (MXFP4) | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G492 | 320 × 6144 × 7168 | 48 | prefill | 12 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G493 | 320 × 7168 × 3072 | 48 | prefill | 12 | FP32@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G494 | 416 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G495 | 416 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G496 | 576 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G497 | 576 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G498 | 672 × 4096 × 2048 | 32 | prefill | 16 | FP32@1×128 | 离线 1×32 (MXFP4) | [C007](#c007) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G499 | 672 × 4096 × 4096 | 32 | prefill | 16 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C007](#c007) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G500 | 832 × 6144 × 7168 | 48 | prefill | 16 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G501 | 832 × 7168 × 3072 | 48 | prefill | 16 | FP32@1×128 | 离线 1×32 (MXFP4) | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G502 | 960 × 4096 × 2048 | 32 | prefill | 16 | FP32@1×128 | 离线 1×32 (MXFP4) | [C008](#c008) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G503 | 960 × 4096 × 4096 | 32 | prefill | 16 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C008](#c008) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G504 | 1248 × 4096 × 2048 | 32 | prefill | 32 | FP32@1×128 | 离线 1×32 (MXFP4) | [C016](#c016) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |
| G505 | 1248 × 4096 × 4096 | 32 | prefill | 32 | FP8E8M0@1×128 | 离线 1×32 (MXFP4) | [C016](#c016) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 | [S28、S31](#smoke-fp8fp4) |

<a id="table-9"></a>

## 表 9：grouped_gemm · BF16 × INT4 → BF16

6 行需求，累计 10 条实例。编号在本清单中全局唯一。

本表抽取单次 BF16×INT4 grouped GEMM 的 M/N/K、分组数 E 和权重 scale 需求。实现列链接内部 W4A16 grouped kernel；独立接口与 smoke 待补。打包权重、scale/zero-point 与固定分组寻址元数据由调用方提供，来源未明确的 per-group 粒度、打包格式不得推断为已支持。

| 编号 | M × N × K | 分组数 E | 阶段 | 频次 | 权重 scale | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 | 验证 / 优化记录 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- |
| G506 | 6 × 4096 × 7168 | 24 | decode | 1 | 离线 per-group | [C003](#c003) | [W4A16 grouped kernel（内部）][fn-grouped-int4-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-int4) |
| G507 | 6 × 7168 × 2048 | 24 | decode | 1 | 离线 per-group | [C003](#c003) | [W4A16 grouped kernel（内部）][fn-grouped-int4-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-int4) |
| G508 | 854 × 4096 × 7168 | 48 | prefill | 2 | 离线 per-group | [C003](#c003) | [W4A16 grouped kernel（内部）][fn-grouped-int4-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-int4) |
| G509 | 854 × 7168 × 2048 | 48 | prefill | 2 | 离线 per-group | [C003](#c003) | [W4A16 grouped kernel（内部）][fn-grouped-int4-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-int4) |
| G510 | 1110 × 4096 × 7168 | 48 | prefill | 2 | 离线 per-group | [C003](#c003) | [W4A16 grouped kernel（内部）][fn-grouped-int4-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-int4) |
| G511 | 1110 × 7168 × 2048 | 48 | prefill | 2 | 离线 per-group | [C003](#c003) | [W4A16 grouped kernel（内部）][fn-grouped-int4-kernel] | 待 grouped GEMM smoke | [独立 grouped 待验证](#smoke-grouped-int4) |

<a id="model-config-index"></a>

## 模型与配置索引

以下编号对应主表的「模型 / 配置」列。同一完整组合共用一个编号；原始模型名称、精度和方括号内的配置原样保留。

### C001

GLM-5.1/FP8[128k/1k/s2000]

### C002

GLM-5.1/FP8[64k/1k/s2000]

### C003

Kimi-2.6/W4A16[128k/1k/s2000]

### C004

DeepSeek-V4-Pro/FP8[128k/1k/s2000]

### C005

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150]

### C006

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000]

### C007

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000]

### C008

DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000]

### C009

GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000]

### C010

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000]; DeepSeek-V4-Pro/FP8[128k/1k/s2000]

### C011

Hunyuan-V3/FP8[128k/1k/s2000]

### C012

Qwen3.5-397B-A17B/FP8[128k/1k/s2000]

### C013

Qwen3.5-35B-A3B/FP8[4k/1k/s150]

### C014

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000]

### C015

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000]

### C016

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000]

### C017

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000]; Hunyuan-V3/FP8[128k/1k/s2000]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000]

### C018

Qwen3.5-35B-A3B/MXFP4[128k/1k/s150]

### C019

Qwen3.5-35B-A3B/FP8[4k/1k/s150]; Qwen3.5-35B-A3B/MXFP4[128k/1k/s150]

<!-- BoltOps function definitions: links remain on the current branch. -->
[fn-bmm]: ../../boltops/gemm/triton/batched_gemm_bf16.py#L323
[fn-a16]: ../../boltops/gemm/triton/gemm_a16w16.py#L33
[fn-persistent]: ../../boltops/gemm/triton/matmul_persistent.py#L205
[fn-fp8-aiter]: ../../boltops/gemm/triton/gemm_a8w8_blockscale.py#L29
[fn-fp8-vllm]: ../../boltops/gemm/triton/w8a8_triton_block_scaled_mm.py#L127
[fn-awq]: ../../boltops/gemm/triton/awq_gemm_triton.py#L351
[fn-fp8fp4]: ../../boltops/gemm/triton/gemm_a8wfp4.py#L83
[fn-grouped-fp8-kernel]: ../../boltops/fused_moe/triton/moe_op.py#L1292
[fn-grouped-int4-kernel]: ../../boltops/fused_moe/triton/moe_op.py#L307
