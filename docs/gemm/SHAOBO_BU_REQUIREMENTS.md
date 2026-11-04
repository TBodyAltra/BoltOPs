# 少伯 BU GEMM 支持需求

原编号来源：`/xukang/GEMM支持清单.html`（2026-10-12）；本次于 2026-11-04 按新版目录 `shaobob0_shapes_1102/` 的 14 份 XLSX、39 个 sheet 更新。原 G/C 编号保留，不填补历史删除留下的编号空洞。原 HTML SHA256：`187673895b820bfb89d6b2890ffb35fa378a183e8922238a8f96df2ff738d11c`。

新版 Excel 共 2526 条 GEMM/BMM/grouped 来源：489 个已编号需求覆盖 2446 条，另有 42 个未编号 grouped shape 覆盖 80 条。按用户要求另保留表 8 的 34 个 FP8×FP4 兼容需求（304 条对应来源），因此主表共 **523 行支持需求**，表内频次合计 2750；兼容表与新版 FP4×FP4 表复用同一来源谱系，不能把 2750 当作新版 Excel 独立行数。

新增表 9 的 FP4×FP4 为 G506–G539；新增 Qwen3-VL-8B BF16 linear 为 G540–G559。所有 scale、量化粒度和动态/静态标签以新版 Excel 显式字段为准；原 FP8×FP4 需求与编号保留，其 scale 同步采用新版 E8M0/K32。

接口级 smoke 沿用历史证据；**本次未运行 GPU 测试，新增 FP4 grouped、新 BF16 shape，以及变更后的 scale 组合均未因此获得 PASS。** “接口 smoke 通过”不表示本行完整 MNK/E/scale 已验证或已调优。实验记录规范见 [调优记录](debug_tuning/TUNING.md)。逐行来源、原始量化标签与校验见 [本次核对记录](debug_tuning/requirements_audit_20261104/README.md) / [编号映射](debug_tuning/requirements_audit_20261104/case-provenance.csv)。

## P0 快速索引

当前 P0：Qwen3.5-35B-A3B 的 FP8、MXFP4 两版，以及 Qwen3-VL-8B BF16。优先级按模型来源覆盖其全部需求，含保留的 FP8×FP4 兼容项；其他模型暂记“未指定”。点击数量可跳到对应首条，展开下方模型编号清单可逐项跳转。

| 模型 / 配置 | 优先级 | BF16 dense | FP8 dense | FP4×FP4 grouped | FP8×FP4 兼容 | 未编号 grouped |
| --- | --- | --- | --- | --- | --- | --- |
| [Qwen3.5-35B-A3B FP8](#c013) | P0 | [7 项](#g291) | [15 项](#g076) | — | — | [6 项](#gap-001) |
| [Qwen3.5-35B-A3B MXFP4](#c018) | P0 | [22 项](#g256) | — | [6 项](#g506) | [6 项](#g472) | — |
| [Qwen3-VL-8B BF16](#c020) | P0 | [20 项](#g540) | — | — | — | — |

合计 **74 条已编号 P0 需求**（其中 47 条 BF16 dense、15 条 FP8 dense、6 条新版 FP4×FP4、6 条 FP8×FP4 兼容），另有 **6 个 P0 grouped 缺口**。G342/G347 为两种 Qwen35 配置共享，主表各只列一次；快速索引的逐模型数量不直接相加。

<a id="p0-q35-fp8"></a>
<details>
<summary>Qwen3.5-35B-A3B FP8：P0 需求编号</summary>

- BF16 dense：[G291](#g291)、[G292](#g292)、[G295](#g295)、[G321](#g321)、[G323](#g323)、[G342](#g342)、[G347](#g347)

- FP8 dense：[G076](#g076)、[G077](#g077)、[G078](#g078)、[G079](#g079)、[G080](#g080)、[G134](#g134)、[G136](#g136)、[G137](#g137)、[G148](#g148)、[G149](#g149)、[G194](#g194)、[G197](#g197)、[G198](#g198)、[G216](#g216)、[G218](#g218)

- 未编号 grouped（待范围 / 接口确认）：[3×1024×2048 / E221](#gap-001)、[3×2048×512 / E221](#gap-002)、[80×1024×2048 / E256](#gap-021)、[80×2048×512 / E256](#gap-022)、[208×1024×2048 / E256](#gap-025)、[208×2048×512 / E256](#gap-026)

</details>

<a id="p0-q35-mxfp4"></a>
<details>
<summary>Qwen3.5-35B-A3B MXFP4：P0 需求编号</summary>

- BF16 dense：[G256](#g256)、[G257](#g257)、[G259](#g259)、[G260](#g260)、[G261](#g261)、[G263](#g263)、[G264](#g264)、[G266](#g266)、[G313](#g313)、[G314](#g314)、[G315](#g315)、[G316](#g316)、[G317](#g317)、[G318](#g318)、[G319](#g319)、[G342](#g342)、[G347](#g347)、[G352](#g352)、[G353](#g353)、[G354](#g354)、[G360](#g360)、[G362](#g362)

- FP4×FP4 grouped：[G506](#g506)、[G507](#g507)、[G520](#g520)、[G521](#g521)、[G524](#g524)、[G525](#g525)

- FP8×FP4 grouped（兼容）：[G472](#g472)、[G473](#g473)、[G486](#g486)、[G487](#g487)、[G490](#g490)、[G491](#g491)

</details>

<a id="p0-qwen3-vl-8b"></a>
<details>
<summary>Qwen3-VL-8B BF16：P0 需求编号</summary>

- BF16 dense：[G540](#g540)、[G541](#g541)、[G542](#g542)、[G543](#g543)、[G544](#g544)、[G545](#g545)、[G546](#g546)、[G547](#g547)、[G548](#g548)、[G549](#g549)、[G550](#g550)、[G551](#g551)、[G552](#g552)、[G553](#g553)、[G554](#g554)、[G555](#g555)、[G556](#g556)、[G557](#g557)、[G558](#g558)、[G559](#g559)

</details>

## 记录口径

- 旧编号的 M/N/K 保留，新增行按新版 Excel 的逻辑 shape 记录激活行数、输出通道数和规约维；不据此推断矩阵 stride。
- batch / 分组数 E 的有效值沿用源数据；bmm 表使用 batch 列，grouped_gemm 表将原 experts 列记为分组数 E，linear 表原本全空的该列省略。M/N/K 原值不变，不以 M×E 替换 M，也不据来源 M 假定各组等长；ragged 各组 M 与 offsets 须由调用方提供。
- 输出精度指写回显存的精度，与累加精度分开验证。
- 激活/权重 scale、group size、动态性采用新版 Excel 的 act_quant / weight_quant；已明确字段不再标作“网补”。FP32*/BF16* 的星号原样保留，Excel 未解释其含义；不擅自改为无星号 dtype，也不据此确定物理存储/转换格式。仅缺失字段可参考历史固定 revision 的公开补充，须标注来源。
- “频次 / 实例数”是 Excel 算子记录出现次数，不是 `layer_count` 展开后的 kernel launch 次数；例如 G171 频次为 1，其来源 `layer_count=3`，两者不冲突。
- 状态分为接口级「接口 smoke 通过」「待 grouped GEMM smoke」「接口 smoke 受阻」与需求尺寸级「待验证」「编译通过」「完整尺寸正确性通过」「已调优」；不得把接口 smoke 或编译通过升级为本行完整尺寸 PASS。
- BoltOps 函数 / 实现列链接实际源码定义；它表示当前可对接的接口，不替代模型布局、量化格式与分组寻址适配。表 8/9 只关注单次 grouped GEMM，输入量化值、scale 和分组索引由测试预先准备；不把在线量化、路由生成、激活或结果合并纳入验收。
- ResNet50 INT8 隐式 GEMM 卷积 24 类和末端 FC 不在这 523 行中，仍以原 HTML 及当前支持说明为准。

模型 / 配置列使用 [C 编号](#model-config-index) 引用文末的完整组合，缩短主表宽度；点击编号可查看模型、精度及序列/规模配置。

## 分类索引

| 表 | 算子 | A × B → 输出 | 需求行数 | 来源次数 |
| --- | --- | --- | ---: | ---: |
| [表 1](#table-1) | bmm | BF16 × BF16 → BF16 | 29 | 194 |
| [表 2](#table-2) | linear | FP8E4M3 × FP8E4M3 → BF16 | 207 | 1206 |
| [表 3](#table-3) | linear | BF16 × BF16 → BF16 | 153 | 348 |
| [表 4](#table-4) | linear | FP32 × FP32 → FP32 | 35 | 218 |
| [表 5](#table-5) | linear | BF16 × BF16 → FP32 | 27 | 172 |
| [表 6](#table-6) | linear | BF16 × INT4 → BF16 | 4 | 4 |
| [表 8](#table-8) | grouped_gemm（兼容保留） | FP8E4M3 × FP4E2M1 → BF16 | 34 | 304 |
| [表 9](#table-9) | grouped_gemm（新版） | FP4E2M1 × FP4E2M1 → BF16 | 34 | 304 |

表 8 的 304 次为兼容来源复用，表 9 的 304 次属于新版 Excel 实际来源。

<a id="table-1"></a>

## 表 1：bmm · BF16 × BF16 → BF16

29 行需求，累计 194 条实例。编号在本清单中全局唯一。

本表仅记录 BF16 BMM；函数使用 `B[batch,N,K]` 权重布局，模型调用需匹配 stride。

| 编号 | 优先级 | M × N × K | batch | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| G001<a id="g001"></a> | 未指定 | 6 × 256 × 512 | 64 | decode | 1 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G002<a id="g002"></a> | 未指定 | 6 × 512 × 192 | 64 | decode | 1 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G003<a id="g003"></a> | 未指定 | 12 × 256 × 512 | 64 | decode | 1 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G004<a id="g004"></a> | 未指定 | 12 × 512 × 192 | 64 | decode | 1 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G005<a id="g005"></a> | 未指定 | 16 × 128 × 512 | 64 | decode | 2 | [C003](#c003) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G006<a id="g006"></a> | 未指定 | 16 × 512 × 128 | 64 | decode | 2 | [C003](#c003) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G007<a id="g007"></a> | 未指定 | 24 × 256 × 512 | 64 | decode | 2 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G008<a id="g008"></a> | 未指定 | 24 × 512 × 192 | 64 | decode | 2 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G009<a id="g009"></a> | 未指定 | 48 × 256 × 512 | 64 | decode | 2 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G010<a id="g010"></a> | 未指定 | 48 × 512 × 192 | 64 | decode | 2 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G011<a id="g011"></a> | 未指定 | 80 × 1024 × 4096 | 16 | decode | 6 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G012<a id="g012"></a> | 未指定 | 96 × 1024 × 4096 | 16 | decode | 8 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G013<a id="g013"></a> | 未指定 | 160 × 1024 × 4096 | 8 | decode | 3 | [C005](#c005) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G014<a id="g014"></a> | 未指定 | 192 × 1024 × 4096 | 8 | decode | 4 | [C005](#c005) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G015<a id="g015"></a> | 未指定 | 240 × 1024 × 4096 | 8 | decode | 9 | [C006](#c006) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G016<a id="g016"></a> | 未指定 | 288 × 1024 × 4096 | 8 | decode | 12 | [C006](#c006) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G017<a id="g017"></a> | 未指定 | 2560 × 256 × 512 | 64 | prefill | 4 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G018<a id="g018"></a> | 未指定 | 2560 × 512 × 192 | 64 | prefill | 4 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G019<a id="g019"></a> | 未指定 | 2560 × 1024 × 4096 | 16 | prefill | 12 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G020<a id="g020"></a> | 未指定 | 2560 × 1024 × 4096 | 8 | prefill | 4 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G021<a id="g021"></a> | 未指定 | 3584 × 1024 × 4096 | 8 | prefill | 16 | [C007](#c007) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G022<a id="g022"></a> | 未指定 | 4608 × 256 × 512 | 64 | prefill | 4 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G023<a id="g023"></a> | 未指定 | 4608 × 512 × 192 | 64 | prefill | 4 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G024<a id="g024"></a> | 未指定 | 4608 × 1024 × 4096 | 16 | prefill | 4 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G025<a id="g025"></a> | 未指定 | 5120 × 1024 × 4096 | 8 | prefill | 16 | [C008](#c008) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G026<a id="g026"></a> | 未指定 | 6656 × 256 × 512 | 64 | prefill | 8 | [C009](#c009) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G027<a id="g027"></a> | 未指定 | 6656 × 512 × 192 | 64 | prefill | 8 | [C009](#c009) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G028<a id="g028"></a> | 未指定 | 6656 × 1024 × 4096 | 8 | prefill | 36 | [C010](#c010) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G029<a id="g029"></a> | 未指定 | 6656 × 1024 × 4096 | 16 | prefill | 16 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |

<a id="table-2"></a>

## 表 2：linear · FP8E4M3 × FP8E4M3 → BF16

207 行需求，1206 条新版 Excel 来源，原 G 编号不变。A/B scale 均采用新版显式标签；逐行精确 act_quant/weight_quant 配对保存在本次编号映射中。

DeepSeek dense A 改为 FP32/K128/动态；B 仍为 E8M0/N128/K128。GLM B 为 FP32*/N128/K128；Qwen FP8 B 为 BF16/N128/K128。Hy3 A 为 FP32/逐 token 动态，B 为 FP32*/channel，替换历史公开候选的 BF16 scalar/static 口径；channel 的物理轴/布局及星号含义仍待确认。

同一 G 混合不同模型时，B scale 按 X 来源分别列出。标注来自 Excel 不代表 scale 物理布局/stride、量化连线或新组合已通过 GPU 测试；尤其 Hy3 的 channel scale 需单独核对接口适配。

### 按 A scale dtype 与 block size 分组

| 分组 | A scale | 需求行数 | 来源次数 |
| --- | --- | ---: | ---: |
| [2.1](#table-2-scale-1) | FP32@1×128，动态 | 181 | 1127 |
| [2.3](#table-2-scale-3) | FP32@1×K，逐 token 动态 | 25 | 40 |
| [2.5](#table-2-scale-5) | 同一编号含 K128 与 token 两种来源 | 1 | 39 |

<a id="table-2-scale-2"></a>
<a id="table-2-scale-4"></a>

旧 2.2 的 E8M0/K128 已依新版归入 2.1；旧 2.4 的 per-tensor 来源已依新版 token/dyn 归入 2.3。保留旧锚点用于历史引用。

<a id="table-2-scale-1"></a>

### 2.1 A scale FP32，block size 1×128

181 行需求，累计 1127 条新版来源。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G034<a id="g034"></a> | 未指定 | 6 × 128 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G035<a id="g035"></a> | 未指定 | 6 × 2624 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G036<a id="g036"></a> | 未指定 | 6 × 4096 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G037<a id="g037"></a> | 未指定 | 6 × 4096 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G038<a id="g038"></a> | 未指定 | 6 × 6144 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G039<a id="g039"></a> | 未指定 | 6 × 6144 × 16384 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G040<a id="g040"></a> | 未指定 | 6 × 16384 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G041<a id="g041"></a> | 未指定 | 12 × 128 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G042<a id="g042"></a> | 未指定 | 12 × 2624 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G043<a id="g043"></a> | 未指定 | 12 × 4096 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G044<a id="g044"></a> | 未指定 | 12 × 4096 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G045<a id="g045"></a> | 未指定 | 12 × 6144 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G046<a id="g046"></a> | 未指定 | 12 × 6144 × 16384 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G047<a id="g047"></a> | 未指定 | 12 × 16384 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G054<a id="g054"></a> | 未指定 | 24 × 128 × 6144 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G055<a id="g055"></a> | 未指定 | 24 × 2624 × 6144 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G056<a id="g056"></a> | 未指定 | 24 × 4096 × 2048 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G057<a id="g057"></a> | 未指定 | 24 × 4096 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G058<a id="g058"></a> | 未指定 | 24 × 6144 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G059<a id="g059"></a> | 未指定 | 24 × 6144 × 12288 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G060<a id="g060"></a> | 未指定 | 24 × 6144 × 16384 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G061<a id="g061"></a> | 未指定 | 24 × 16384 × 2048 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G062<a id="g062"></a> | 未指定 | 24 × 24576 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G063<a id="g063"></a> | 未指定 | 48 × 128 × 6144 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G064<a id="g064"></a> | 未指定 | 48 × 2048 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G065<a id="g065"></a> | 未指定 | 48 × 2624 × 6144 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G066<a id="g066"></a> | 未指定 | 48 × 4096 × 1024 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G067<a id="g067"></a> | 未指定 | 48 × 4096 × 2048 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G068<a id="g068"></a> | 未指定 | 48 × 4096 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G069<a id="g069"></a> | 未指定 | 48 × 4096 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G070<a id="g070"></a> | 未指定 | 48 × 6144 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G071<a id="g071"></a> | 未指定 | 48 × 6144 × 12288 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G072<a id="g072"></a> | 未指定 | 48 × 6144 × 16384 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G073<a id="g073"></a> | 未指定 | 48 × 8704 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G074<a id="g074"></a> | 未指定 | 48 × 16384 × 2048 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G075<a id="g075"></a> | 未指定 | 48 × 24576 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G076<a id="g076"></a> | P0 | 64 × 1024 × 2048 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G077<a id="g077"></a> | P0 | 64 × 2048 × 512 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G078<a id="g078"></a> | P0 | 64 × 2048 × 4096 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G079<a id="g079"></a> | P0 | 64 × 9216 × 2048 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G080<a id="g080"></a> | P0 | 64 × 12288 × 2048 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G081<a id="g081"></a> | 未指定 | 80 × 512 × 7168 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G082<a id="g082"></a> | 未指定 | 80 × 1536 × 7168 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G083<a id="g083"></a> | 未指定 | 80 × 6144 × 7168 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G084<a id="g084"></a> | 未指定 | 80 × 7168 × 3072 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G085<a id="g085"></a> | 未指定 | 80 × 7168 × 16384 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G086<a id="g086"></a> | 未指定 | 80 × 65536 × 1536 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G087<a id="g087"></a> | 未指定 | 96 × 512 × 7168 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G088<a id="g088"></a> | 未指定 | 96 × 1536 × 7168 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G089<a id="g089"></a> | 未指定 | 96 × 6144 × 7168 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G090<a id="g090"></a> | 未指定 | 96 × 7168 × 3072 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G091<a id="g091"></a> | 未指定 | 96 × 7168 × 16384 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G092<a id="g092"></a> | 未指定 | 96 × 8192 × 1536 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G093<a id="g093"></a> | 未指定 | 96 × 65536 × 1536 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G094<a id="g094"></a> | 未指定 | 160 × 512 × 4096 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G095<a id="g095"></a> | 未指定 | 160 × 1024 × 4096 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G096<a id="g096"></a> | 未指定 | 160 × 4096 × 2048 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G097<a id="g097"></a> | 未指定 | 160 × 4096 × 4096 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G098<a id="g098"></a> | 未指定 | 160 × 4096 × 8192 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G099<a id="g099"></a> | 未指定 | 160 × 32768 × 1024 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G100<a id="g100"></a> | 未指定 | 192 × 512 × 4096 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G101<a id="g101"></a> | 未指定 | 192 × 1024 × 4096 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G102<a id="g102"></a> | 未指定 | 192 × 2048 × 4096 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G103<a id="g103"></a> | 未指定 | 192 × 4096 × 1024 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G104<a id="g104"></a> | 未指定 | 192 × 4096 × 2048 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G105<a id="g105"></a> | 未指定 | 192 × 4096 × 4096 | decode | 6 | FP32@1×128（动态） | X03：E8M0@128×128（4条）<br>X13：BF16@128×128（2条） | [C014](#c014) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G106<a id="g106"></a> | 未指定 | 192 × 4096 × 8192 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G107<a id="g107"></a> | 未指定 | 192 × 8192 × 1024 | decode | 2 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G108<a id="g108"></a> | 未指定 | 192 × 8704 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G109<a id="g109"></a> | 未指定 | 192 × 10240 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G110<a id="g110"></a> | 未指定 | 192 × 32768 × 1024 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G111<a id="g111"></a> | 未指定 | 240 × 512 × 4096 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G112<a id="g112"></a> | 未指定 | 240 × 1024 × 4096 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G113<a id="g113"></a> | 未指定 | 240 × 4096 × 2048 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G114<a id="g114"></a> | 未指定 | 240 × 4096 × 4096 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G115<a id="g115"></a> | 未指定 | 240 × 4096 × 8192 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G116<a id="g116"></a> | 未指定 | 240 × 32768 × 1024 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G117<a id="g117"></a> | 未指定 | 288 × 512 × 4096 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G118<a id="g118"></a> | 未指定 | 288 × 1024 × 4096 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G119<a id="g119"></a> | 未指定 | 288 × 4096 × 2048 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G120<a id="g120"></a> | 未指定 | 288 × 4096 × 4096 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G121<a id="g121"></a> | 未指定 | 288 × 4096 × 8192 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G122<a id="g122"></a> | 未指定 | 288 × 8192 × 1024 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G123<a id="g123"></a> | 未指定 | 288 × 32768 × 1024 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G130<a id="g130"></a> | 未指定 | 2560 × 128 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G131<a id="g131"></a> | 未指定 | 2560 × 512 × 6144 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G132<a id="g132"></a> | 未指定 | 2560 × 512 × 7168 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G133<a id="g133"></a> | 未指定 | 2560 × 768 × 7168 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G134<a id="g134"></a> | P0 | 2560 × 1024 × 2048 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G135<a id="g135"></a> | 未指定 | 2560 × 1536 × 7168 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G136<a id="g136"></a> | P0 | 2560 × 2048 × 512 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G137<a id="g137"></a> | P0 | 2560 × 2048 × 4096 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G138<a id="g138"></a> | 未指定 | 2560 × 2624 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G139<a id="g139"></a> | 未指定 | 2560 × 3072 × 6144 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G140<a id="g140"></a> | 未指定 | 2560 × 4096 × 2048 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G141<a id="g141"></a> | 未指定 | 2560 × 6144 × 256 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G142<a id="g142"></a> | 未指定 | 2560 × 6144 × 1536 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G143<a id="g143"></a> | 未指定 | 2560 × 6144 × 16384 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G144<a id="g144"></a> | 未指定 | 2560 × 7168 × 384 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G145<a id="g145"></a> | 未指定 | 2560 × 7168 × 8192 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G146<a id="g146"></a> | 未指定 | 2560 × 7168 × 16384 | prefill | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G147<a id="g147"></a> | 未指定 | 2560 × 8192 × 1536 | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G148<a id="g148"></a> | P0 | 2560 × 9216 × 2048 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G149<a id="g149"></a> | P0 | 2560 × 12288 × 2048 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G150<a id="g150"></a> | 未指定 | 2560 × 16384 × 2048 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G151<a id="g151"></a> | 未指定 | 2560 × 32768 × 1536 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G152<a id="g152"></a> | 未指定 | 2560 × 65536 × 1536 | prefill | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G153<a id="g153"></a> | 未指定 | 3584 × 256 × 4096 | prefill | 4 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G154<a id="g154"></a> | 未指定 | 3584 × 512 × 4096 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G155<a id="g155"></a> | 未指定 | 3584 × 1024 × 4096 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G156<a id="g156"></a> | 未指定 | 3584 × 4096 × 128 | prefill | 4 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G157<a id="g157"></a> | 未指定 | 3584 × 4096 × 256 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G158<a id="g158"></a> | 未指定 | 3584 × 4096 × 8192 | prefill | 19 | FP32@1×128（动态） | X03：E8M0@128×128（8条）<br>X04：E8M0@128×128（8条）<br>X13：BF16@128×128（3条） | [C015](#c015) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G159<a id="g159"></a> | 未指定 | 3584 × 8192 × 1024 | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G160<a id="g160"></a> | 未指定 | 3584 × 17408 × 4096 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G161<a id="g161"></a> | 未指定 | 3584 × 20480 × 4096 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G162<a id="g162"></a> | 未指定 | 3584 × 32768 × 1024 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G165<a id="g165"></a> | 未指定 | 4608 × 128 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G166<a id="g166"></a> | 未指定 | 4608 × 512 × 6144 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G167<a id="g167"></a> | 未指定 | 4608 × 512 × 7168 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G168<a id="g168"></a> | 未指定 | 4608 × 768 × 7168 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G169<a id="g169"></a> | 未指定 | 4608 × 1536 × 7168 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G170<a id="g170"></a> | 未指定 | 4608 × 2624 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G171<a id="g171"></a> | 未指定 | 4608 × 3072 × 6144 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G172<a id="g172"></a> | 未指定 | 4608 × 4096 × 2048 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G173<a id="g173"></a> | 未指定 | 4608 × 6144 × 256 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G174<a id="g174"></a> | 未指定 | 4608 × 6144 × 1536 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G175<a id="g175"></a> | 未指定 | 4608 × 6144 × 16384 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G176<a id="g176"></a> | 未指定 | 4608 × 7168 × 384 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G177<a id="g177"></a> | 未指定 | 4608 × 7168 × 16384 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G178<a id="g178"></a> | 未指定 | 4608 × 8192 × 1536 | prefill | 2 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G179<a id="g179"></a> | 未指定 | 4608 × 16384 × 2048 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G180<a id="g180"></a> | 未指定 | 4608 × 65536 × 1536 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G181<a id="g181"></a> | 未指定 | 5120 × 512 × 4096 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G182<a id="g182"></a> | 未指定 | 5120 × 1024 × 4096 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G183<a id="g183"></a> | 未指定 | 5120 × 4096 × 256 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G184<a id="g184"></a> | 未指定 | 5120 × 4096 × 8192 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G185<a id="g185"></a> | 未指定 | 5120 × 8192 × 1024 | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G186<a id="g186"></a> | 未指定 | 5120 × 32768 × 1024 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G187<a id="g187"></a> | 未指定 | 6656 × 128 × 6144 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G188<a id="g188"></a> | 未指定 | 6656 × 256 × 4096 | prefill | 4 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G190<a id="g190"></a> | 未指定 | 6656 × 512 × 4096 | prefill | 64 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G191<a id="g191"></a> | 未指定 | 6656 × 512 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G192<a id="g192"></a> | 未指定 | 6656 × 512 × 7168 | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G193<a id="g193"></a> | 未指定 | 6656 × 768 × 7168 | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G194<a id="g194"></a> | P0 | 6656 × 1024 × 2048 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G195<a id="g195"></a> | 未指定 | 6656 × 1024 × 4096 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G196<a id="g196"></a> | 未指定 | 6656 × 1536 × 7168 | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G197<a id="g197"></a> | P0 | 6656 × 2048 × 512 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G198<a id="g198"></a> | P0 | 6656 × 2048 × 4096 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G199<a id="g199"></a> | 未指定 | 6656 × 2624 × 6144 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G200<a id="g200"></a> | 未指定 | 6656 × 3072 × 6144 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G202<a id="g202"></a> | 未指定 | 6656 × 4096 × 128 | prefill | 4 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G204<a id="g204"></a> | 未指定 | 6656 × 4096 × 256 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G206<a id="g206"></a> | 未指定 | 6656 × 4096 × 2048 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G208<a id="g208"></a> | 未指定 | 6656 × 6144 × 256 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G209<a id="g209"></a> | 未指定 | 6656 × 6144 × 1536 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G210<a id="g210"></a> | 未指定 | 6656 × 6144 × 16384 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G211<a id="g211"></a> | 未指定 | 6656 × 7168 × 384 | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G212<a id="g212"></a> | 未指定 | 6656 × 7168 × 8192 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G213<a id="g213"></a> | 未指定 | 6656 × 7168 × 16384 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G214<a id="g214"></a> | 未指定 | 6656 × 8192 × 1024 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G215<a id="g215"></a> | 未指定 | 6656 × 8192 × 1536 | prefill | 10 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G216<a id="g216"></a> | P0 | 6656 × 9216 × 2048 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G218<a id="g218"></a> | P0 | 6656 × 12288 × 2048 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G219<a id="g219"></a> | 未指定 | 6656 × 16384 × 2048 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G220<a id="g220"></a> | 未指定 | 6656 × 17408 × 4096 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G221<a id="g221"></a> | 未指定 | 6656 × 20480 × 4096 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G222<a id="g222"></a> | 未指定 | 6656 × 32768 × 1024 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G223<a id="g223"></a> | 未指定 | 6656 × 32768 × 1536 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G224<a id="g224"></a> | 未指定 | 6656 × 65536 × 1536 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G225<a id="g225"></a> | 未指定 | 20480 × 3072 × 6144 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G226<a id="g226"></a> | 未指定 | 20480 × 6144 × 1536 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G227<a id="g227"></a> | 未指定 | 28672 × 2560 × 4096 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G228<a id="g228"></a> | 未指定 | 28672 × 4096 × 1024 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G229<a id="g229"></a> | 未指定 | 36864 × 3072 × 6144 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G230<a id="g230"></a> | 未指定 | 36864 × 6144 × 1536 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G231<a id="g231"></a> | 未指定 | 53248 × 2560 × 4096 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G232<a id="g232"></a> | 未指定 | 53248 × 3072 × 6144 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G234<a id="g234"></a> | 未指定 | 53248 × 4096 × 1024 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G236<a id="g236"></a> | 未指定 | 53248 × 6144 × 1536 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a id="table-2-scale-3"></a>

### 2.3 A scale FP32，逐 token 的整行 K

25 行需求，累计 40 条新版来源。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G030<a id="g030"></a> | 未指定 | 4 × 3072 × 4096 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G031<a id="g031"></a> | 未指定 | 4 × 4096 × 1536 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G032<a id="g032"></a> | 未指定 | 4 × 4096 × 8192 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G033<a id="g033"></a> | 未指定 | 4 × 10240 × 4096 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G048<a id="g048"></a> | 未指定 | 16 × 3072 × 4096 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G049<a id="g049"></a> | 未指定 | 16 × 4096 × 1536 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G050<a id="g050"></a> | 未指定 | 16 × 4096 × 8192 | decode | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G051<a id="g051"></a> | 未指定 | 16 × 4096 × 13312 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G052<a id="g052"></a> | 未指定 | 16 × 10240 × 4096 | decode | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G053<a id="g053"></a> | 未指定 | 16 × 26624 × 4096 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G124<a id="g124"></a> | 未指定 | 512 × 384 × 4096 | prefill | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G125<a id="g125"></a> | 未指定 | 512 × 3328 × 4096 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G126<a id="g126"></a> | 未指定 | 512 × 4096 × 192 | prefill | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G127<a id="g127"></a> | 未指定 | 512 × 4096 × 1664 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G128<a id="g128"></a> | 未指定 | 512 × 4096 × 8192 | prefill | 4 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G129<a id="g129"></a> | 未指定 | 512 × 10240 × 4096 | prefill | 4 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G163<a id="g163"></a> | 未指定 | 4096 × 3328 × 4096 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G164<a id="g164"></a> | 未指定 | 4096 × 4096 × 1664 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G189<a id="g189"></a> | 未指定 | 6656 × 384 × 4096 | prefill | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G201<a id="g201"></a> | 未指定 | 6656 × 3328 × 4096 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G203<a id="g203"></a> | 未指定 | 6656 × 4096 × 192 | prefill | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G205<a id="g205"></a> | 未指定 | 6656 × 4096 × 1664 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G217<a id="g217"></a> | 未指定 | 6656 × 10240 × 4096 | prefill | 4 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G233<a id="g233"></a> | 未指定 | 53248 × 3328 × 4096 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G235<a id="g235"></a> | 未指定 | 53248 × 4096 × 1664 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a id="table-2-scale-5"></a>

### 2.5 同一编号含多种 A scale

1 行需求，累计 39 条新版来源。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G207<a id="g207"></a> | 未指定 | 6656 × 4096 × 8192 | prefill | 39 | FP32@1×128（动态） × 35条 / FP32@1×K（逐 token 动态） × 4条 | X01：E8M0@128×128（8条）<br>X02：E8M0@128×128（8条）<br>X03：E8M0@128×128（8条）<br>X04：E8M0@128×128（8条）<br>X12：FP32*@channel（4条）<br>X13：BF16@128×128（3条） | [C017](#c017) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a id="table-3"></a>

## 表 3：linear · BF16 × BF16 → BF16

153 行需求，累计 348 条来源（新增 Qwen3-VL-8B 20 行 / 22 条来源）。编号在本清单中全局唯一。

本表对应 BF16 输入、BF16 输出；`gemm_a16w16` 的权重布局为 `[N,K]`。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| G237<a id="g237"></a> | 未指定 | 1 × 154880 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G238<a id="g238"></a> | 未指定 | 2 × 64640 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G239<a id="g239"></a> | 未指定 | 2 × 129280 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G240<a id="g240"></a> | 未指定 | 3 × 120832 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G241<a id="g241"></a> | 未指定 | 4 × 192 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G242<a id="g242"></a> | 未指定 | 4 × 4096 × 8192 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G243<a id="g243"></a> | 未指定 | 4 × 7168 × 21504 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G244<a id="g244"></a> | 未指定 | 4 × 120832 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G245<a id="g245"></a> | 未指定 | 4 × 154880 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G246<a id="g246"></a> | 未指定 | 4 × 163840 × 7168 | decode,prefill | 3 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G247<a id="g247"></a> | 未指定 | 6 × 32 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G248<a id="g248"></a> | 未指定 | 6 × 256 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G249<a id="g249"></a> | 未指定 | 6 × 6144 × 12288 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G250<a id="g250"></a> | 未指定 | 6 × 154880 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G251<a id="g251"></a> | 未指定 | 8 × 16160 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G252<a id="g252"></a> | 未指定 | 8 × 19360 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G253<a id="g253"></a> | 未指定 | 8 × 129280 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G254<a id="g254"></a> | 未指定 | 8 × 248320 × 4096 | prefill | 2 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G255<a id="g255"></a> | 未指定 | 12 × 32 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G256<a id="g256"></a> | P0 | 12 × 64 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G257<a id="g257"></a> | P0 | 12 × 256 × 2048 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G258<a id="g258"></a> | 未指定 | 12 × 256 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G259<a id="g259"></a> | P0 | 12 × 1024 × 2048 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G260<a id="g260"></a> | P0 | 12 × 2048 × 512 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G261<a id="g261"></a> | P0 | 12 × 2048 × 4096 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G262<a id="g262"></a> | 未指定 | 12 × 6144 × 12288 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G263<a id="g263"></a> | P0 | 12 × 9216 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G264<a id="g264"></a> | P0 | 12 × 12288 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G265<a id="g265"></a> | 未指定 | 12 × 154880 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G266<a id="g266"></a> | P0 | 12 × 248320 × 2048 | decode,prefill | 3 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G267<a id="g267"></a> | 未指定 | 16 × 192 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G268<a id="g268"></a> | 未指定 | 16 × 384 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G269<a id="g269"></a> | 未指定 | 16 × 2112 × 7168 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G270<a id="g270"></a> | 未指定 | 16 × 4096 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G271<a id="g271"></a> | 未指定 | 16 × 7168 × 2048 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G272<a id="g272"></a> | 未指定 | 16 × 7168 × 8192 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G273<a id="g273"></a> | 未指定 | 16 × 7168 × 18432 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G274<a id="g274"></a> | 未指定 | 16 × 12288 × 1536 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G275<a id="g275"></a> | 未指定 | 16 × 16160 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G276<a id="g276"></a> | 未指定 | 16 × 36864 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G277<a id="g277"></a> | 未指定 | 16 × 120832 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G278<a id="g278"></a> | 未指定 | 16 × 163840 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G279<a id="g279"></a> | 未指定 | 24 × 32 × 6144 | decode | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G280<a id="g280"></a> | 未指定 | 24 × 256 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G281<a id="g281"></a> | 未指定 | 24 × 15104 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G282<a id="g282"></a> | 未指定 | 24 × 154880 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G283<a id="g283"></a> | 未指定 | 32 × 19360 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G284<a id="g284"></a> | 未指定 | 32 × 20480 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G285<a id="g285"></a> | 未指定 | 48 × 32 × 6144 | decode | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G286<a id="g286"></a> | 未指定 | 48 × 256 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G287<a id="g287"></a> | 未指定 | 48 × 512 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G288<a id="g288"></a> | 未指定 | 48 × 4096 × 8192 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G289<a id="g289"></a> | 未指定 | 48 × 124160 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G290<a id="g290"></a> | 未指定 | 48 × 154880 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G291<a id="g291"></a> | P0 | 64 × 64 × 2048 | decode | 1 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G292<a id="g292"></a> | P0 | 64 × 256 × 2048 | decode | 2 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G293<a id="g293"></a> | 未指定 | 64 × 16160 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G294<a id="g294"></a> | 未指定 | 64 × 31040 × 4096 | prefill | 2 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G295<a id="g295"></a> | P0 | 64 × 248320 × 2048 | decode,prefill | 3 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G296<a id="g296"></a> | 未指定 | 80 × 129280 × 7168 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G297<a id="g297"></a> | 未指定 | 96 × 64 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G298<a id="g298"></a> | 未指定 | 96 × 512 × 7168 | decode | 6 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G299<a id="g299"></a> | 未指定 | 96 × 7168 × 21504 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G300<a id="g300"></a> | 未指定 | 96 × 129280 × 7168 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G301<a id="g301"></a> | 未指定 | 160 × 129280 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G302<a id="g302"></a> | 未指定 | 192 × 64 × 4096 | decode | 3 | [C014](#c014) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G303<a id="g303"></a> | 未指定 | 192 × 512 × 4096 | decode | 5 | [C014](#c014) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G304<a id="g304"></a> | 未指定 | 192 × 4096 × 12288 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G305<a id="g305"></a> | 未指定 | 192 × 124160 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G306<a id="g306"></a> | 未指定 | 192 × 129280 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G307<a id="g307"></a> | 未指定 | 240 × 129280 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G308<a id="g308"></a> | 未指定 | 288 × 64 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G309<a id="g309"></a> | 未指定 | 288 × 512 × 4096 | decode | 9 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G310<a id="g310"></a> | 未指定 | 288 × 4096 × 12288 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G311<a id="g311"></a> | 未指定 | 288 × 129280 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G312<a id="g312"></a> | 未指定 | 512 × 192 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G313<a id="g313"></a> | P0 | 2048 × 64 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G314<a id="g314"></a> | P0 | 2048 × 256 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G315<a id="g315"></a> | P0 | 2048 × 1024 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G316<a id="g316"></a> | P0 | 2048 × 2048 × 512 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G317<a id="g317"></a> | P0 | 2048 × 2048 × 4096 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G318<a id="g318"></a> | P0 | 2048 × 9216 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G319<a id="g319"></a> | P0 | 2048 × 12288 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G320<a id="g320"></a> | 未指定 | 2560 × 32 × 6144 | prefill | 4 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G321<a id="g321"></a> | P0 | 2560 × 64 × 2048 | prefill | 1 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G322<a id="g322"></a> | 未指定 | 2560 × 64 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G323<a id="g323"></a> | P0 | 2560 × 256 × 2048 | prefill | 2 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G324<a id="g324"></a> | 未指定 | 2560 × 256 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G325<a id="g325"></a> | 未指定 | 3584 × 64 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G326<a id="g326"></a> | 未指定 | 3584 × 128 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G327<a id="g327"></a> | 未指定 | 3584 × 512 × 4096 | prefill | 4 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G328<a id="g328"></a> | 未指定 | 4608 × 32 × 6144 | prefill | 4 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G329<a id="g329"></a> | 未指定 | 4608 × 64 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G330<a id="g330"></a> | 未指定 | 4608 × 256 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G331<a id="g331"></a> | 未指定 | 5120 × 64 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G332<a id="g332"></a> | 未指定 | 5120 × 384 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G333<a id="g333"></a> | 未指定 | 5120 × 512 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G334<a id="g334"></a> | 未指定 | 5120 × 2112 × 7168 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G335<a id="g335"></a> | 未指定 | 5120 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G336<a id="g336"></a> | 未指定 | 5120 × 7168 × 256 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G337<a id="g337"></a> | 未指定 | 5120 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G338<a id="g338"></a> | 未指定 | 5120 × 7168 × 8192 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G339<a id="g339"></a> | 未指定 | 5120 × 12288 × 1536 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G340<a id="g340"></a> | 未指定 | 5120 × 16384 × 512 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G341<a id="g341"></a> | 未指定 | 6656 × 32 × 6144 | prefill | 8 | [C009](#c009) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G342<a id="g342"></a> | P0 | 6656 × 64 × 2048 | prefill | 2 | [C019](#c019) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G343<a id="g343"></a> | 未指定 | 6656 × 64 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G344<a id="g344"></a> | 未指定 | 6656 × 64 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G345<a id="g345"></a> | 未指定 | 6656 × 128 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G346<a id="g346"></a> | 未指定 | 6656 × 192 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G347<a id="g347"></a> | P0 | 6656 × 256 × 2048 | prefill | 4 | [C019](#c019) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G348<a id="g348"></a> | 未指定 | 6656 × 256 × 6144 | prefill | 4 | [C009](#c009) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G349<a id="g349"></a> | 未指定 | 6656 × 384 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G350<a id="g350"></a> | 未指定 | 6656 × 512 × 4096 | prefill | 4 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G351<a id="g351"></a> | 未指定 | 6656 × 512 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G352<a id="g352"></a> | P0 | 6656 × 1024 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G353<a id="g353"></a> | P0 | 6656 × 2048 × 512 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G354<a id="g354"></a> | P0 | 6656 × 2048 × 4096 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G355<a id="g355"></a> | 未指定 | 6656 × 2112 × 7168 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G356<a id="g356"></a> | 未指定 | 6656 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G357<a id="g357"></a> | 未指定 | 6656 × 7168 × 256 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G358<a id="g358"></a> | 未指定 | 6656 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G359<a id="g359"></a> | 未指定 | 6656 × 7168 × 8192 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G360<a id="g360"></a> | P0 | 6656 × 9216 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G361<a id="g361"></a> | 未指定 | 6656 × 12288 × 1536 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G362<a id="g362"></a> | P0 | 6656 × 12288 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G363<a id="g363"></a> | 未指定 | 6656 × 16384 × 512 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G364<a id="g364"></a> | 未指定 | 28672 × 16 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G365<a id="g365"></a> | 未指定 | 40960 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G366<a id="g366"></a> | 未指定 | 40960 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G367<a id="g367"></a> | 未指定 | 53248 × 16 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G368<a id="g368"></a> | 未指定 | 53248 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G369<a id="g369"></a> | 未指定 | 53248 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G540<a id="g540"></a> | P0 | 512 × 4096 × 4096 | decode | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G541<a id="g541"></a> | P0 | 512 × 4096 × 12288 | decode | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G542<a id="g542"></a> | P0 | 512 × 6144 × 4096 | decode | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G543<a id="g543"></a> | P0 | 512 × 24576 × 4096 | decode | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G544<a id="g544"></a> | P0 | 512 × 151936 × 4096 | decode/prefill | 3 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G545<a id="g545"></a> | P0 | 2560 × 4096 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G546<a id="g546"></a> | P0 | 2560 × 4096 × 12288 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G547<a id="g547"></a> | P0 | 2560 × 6144 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G548<a id="g548"></a> | P0 | 2560 × 24576 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G549<a id="g549"></a> | P0 | 5684 × 4096 × 4608 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G550<a id="g550"></a> | P0 | 5684 × 4608 × 4608 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G551<a id="g551"></a> | P0 | 6656 × 4096 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G552<a id="g552"></a> | P0 | 6656 × 4096 × 12288 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G553<a id="g553"></a> | P0 | 6656 × 6144 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G554<a id="g554"></a> | P0 | 6656 × 24576 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G555<a id="g555"></a> | P0 | 22736 × 1152 × 1152 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G556<a id="g556"></a> | P0 | 22736 × 1152 × 1536 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G557<a id="g557"></a> | P0 | 22736 × 1152 × 4304 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G558<a id="g558"></a> | P0 | 22736 × 3456 × 1152 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G559<a id="g559"></a> | P0 | 22736 × 4304 × 1152 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |

<a id="table-4"></a>

## 表 4：linear · FP32 × FP32 → FP32

35 行需求，累计 218 条实例。编号在本清单中全局唯一。

本表的接口级 smoke PASS 依赖当前 Triton 的 S03 LEGACY 补丁；不能推广到未带该补丁的环境。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| G370<a id="g370"></a> | 未指定 | 16 × 129280 × 512 | decode | 10 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G371<a id="g371"></a> | 未指定 | 32 × 129280 × 256 | decode | 5 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G372<a id="g372"></a> | 未指定 | 48 × 129280 × 256 | decode | 15 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G373<a id="g373"></a> | 未指定 | 80 × 1 × 7680 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G374<a id="g374"></a> | 未指定 | 80 × 4 × 28672 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G375<a id="g375"></a> | 未指定 | 80 × 384 × 7168 | decode | 6 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G376<a id="g376"></a> | 未指定 | 96 × 4 × 28672 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G377<a id="g377"></a> | 未指定 | 96 × 384 × 7168 | decode | 8 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G378<a id="g378"></a> | 未指定 | 160 × 1 × 4352 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G379<a id="g379"></a> | 未指定 | 160 × 4 × 16384 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G380<a id="g380"></a> | 未指定 | 160 × 256 × 4096 | decode | 3 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G381<a id="g381"></a> | 未指定 | 192 × 4 × 16384 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G382<a id="g382"></a> | 未指定 | 192 × 256 × 4096 | decode | 4 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G383<a id="g383"></a> | 未指定 | 240 × 1 × 4352 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G384<a id="g384"></a> | 未指定 | 240 × 4 × 16384 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G385<a id="g385"></a> | 未指定 | 240 × 256 × 4096 | decode | 9 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G386<a id="g386"></a> | 未指定 | 288 × 4 × 16384 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G387<a id="g387"></a> | 未指定 | 288 × 256 × 4096 | decode | 12 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G388<a id="g388"></a> | 未指定 | 2560 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G389<a id="g389"></a> | 未指定 | 2560 × 384 × 7168 | prefill | 16 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G390<a id="g390"></a> | 未指定 | 3584 × 4 × 16384 | prefill | 2 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G391<a id="g391"></a> | 未指定 | 3584 × 256 × 4096 | prefill | 16 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G392<a id="g392"></a> | 未指定 | 4608 × 384 × 7168 | prefill | 4 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G393<a id="g393"></a> | 未指定 | 5120 × 4 × 16384 | prefill | 2 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G394<a id="g394"></a> | 未指定 | 5120 × 256 × 4096 | prefill | 16 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G395<a id="g395"></a> | 未指定 | 6656 × 4 × 16384 | prefill | 4 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G396<a id="g396"></a> | 未指定 | 6656 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G397<a id="g397"></a> | 未指定 | 6656 × 256 × 4096 | prefill | 32 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G398<a id="g398"></a> | 未指定 | 6656 × 384 × 7168 | prefill | 20 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G399<a id="g399"></a> | 未指定 | 20480 × 4 × 28672 | prefill | 1 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G400<a id="g400"></a> | 未指定 | 28672 × 4 × 16384 | prefill | 2 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G401<a id="g401"></a> | 未指定 | 36864 × 4 × 28672 | prefill | 1 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G402<a id="g402"></a> | 未指定 | 40960 × 4 × 16384 | prefill | 2 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G403<a id="g403"></a> | 未指定 | 53248 × 4 × 16384 | prefill | 4 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G404<a id="g404"></a> | 未指定 | 53248 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |

<a id="table-5"></a>

## 表 5：linear · BF16 × BF16 → FP32

27 行需求，累计 172 条实例。编号在本清单中全局唯一。

本表使用 `gemm_a16w16(..., dtype=torch.float32)`，保留 BF16 输入、FP32 输出语义。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| G405<a id="g405"></a> | 未指定 | 96 × 512 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G406<a id="g406"></a> | 未指定 | 96 × 1024 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G407<a id="g407"></a> | 未指定 | 96 × 2048 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G408<a id="g408"></a> | 未指定 | 192 × 512 × 4096 | decode | 2 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G409<a id="g409"></a> | 未指定 | 192 × 1024 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G410<a id="g410"></a> | 未指定 | 192 × 2048 × 4096 | decode | 2 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G411<a id="g411"></a> | 未指定 | 288 × 512 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G412<a id="g412"></a> | 未指定 | 288 × 1024 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G413<a id="g413"></a> | 未指定 | 288 × 2048 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G414<a id="g414"></a> | 未指定 | 2560 × 512 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G415<a id="g415"></a> | 未指定 | 2560 × 1024 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G416<a id="g416"></a> | 未指定 | 2560 × 2048 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G417<a id="g417"></a> | 未指定 | 3584 × 512 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G418<a id="g418"></a> | 未指定 | 3584 × 1024 × 4096 | prefill | 4 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G419<a id="g419"></a> | 未指定 | 3584 × 2048 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G420<a id="g420"></a> | 未指定 | 4608 × 512 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G421<a id="g421"></a> | 未指定 | 4608 × 1024 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G422<a id="g422"></a> | 未指定 | 4608 × 2048 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G423<a id="g423"></a> | 未指定 | 5120 × 512 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G424<a id="g424"></a> | 未指定 | 5120 × 1024 × 4096 | prefill | 4 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G425<a id="g425"></a> | 未指定 | 5120 × 2048 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G426<a id="g426"></a> | 未指定 | 6656 × 512 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G427<a id="g427"></a> | 未指定 | 6656 × 512 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G428<a id="g428"></a> | 未指定 | 6656 × 1024 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G429<a id="g429"></a> | 未指定 | 6656 × 1024 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G430<a id="g430"></a> | 未指定 | 6656 × 2048 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G431<a id="g431"></a> | 未指定 | 6656 × 2048 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |

<a id="table-6"></a>

## 表 6：linear · BF16 × INT4 → BF16

4 行需求、4 条来源，G432–G435 仍为 Kimi EAGLE3 draft。新版现已明确 weight_quant=`int4 g32 bf16*`，因此采用 K32 / BF16* scale；A 为 BF16，无 A scale。星号含义、zero-point、对称性与 packing 仍未给出，不由 Kimi 主模型或 AWQ 的格式反推。AWQ 链接只保留历史接口参考，不证明 draft 的实际格式相容。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G432<a id="g432"></a> | 未指定 | 4 × 7168 × 8192 | decode | 1 | BF16*@1×32（Excel）；packing/zero-point 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G433<a id="g433"></a> | 未指定 | 4 × 7168 × 12288 | decode | 1 | BF16*@1×32（Excel）；packing/zero-point 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G434<a id="g434"></a> | 未指定 | 4 × 24576 × 7168 | decode | 1 | BF16*@1×32（Excel）；packing/zero-point 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G435<a id="g435"></a> | 未指定 | 4 × 24576 × 14336 | decode | 1 | BF16*@1×32（Excel）；packing/zero-point 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |

<a id="table-8"></a>

## 表 8：grouped_gemm · FP8E4M3 × FP4E2M1 → BF16（兼容保留）

按用户要求保留原 34 行、G472–G505、MNK/E、阶段及频次。新版 Excel 的对应 304 条来源实际已改为 FP4×FP4；本表是额外保留的 FP8×FP4 支持需求，不表示新版仍要求 A=FP8。

scale 按新版更新：A/B 均为 E8M0/K32，激活动态；DeepSeek 原 A 的 E8M0/FP32 K128、Q35 down 的 FP32/K32 均不再作为本次 scale 要求。新版 act_quant 的 fp4 数据 dtype 只用于表 9；本表有意保留 A=FP8，仅采用对应新版的 scale 粒度/dtype/动态性。

历史 grouped_gemm_a8wfp4 接口 smoke 与原 34/34 编译证据保留，但不验证本次更新的 scale 组合，完整 MNK/E/scale 的正确性和性能仍待逐项验证。来源次数 304 与表 9 共用谱系，不计作新增 Excel 行。

| 编号 | 优先级 | M × N × K | 分组数 E | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G472<a id="g472"></a> | P0 | 2 × 1024 × 2048 | 80 | decode | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G473<a id="g473"></a> | P0 | 2 × 2048 × 512 | 80 | decode | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G474<a id="g474"></a> | 未指定 | 20 × 6144 × 7168 | 24 | decode | 6 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G475<a id="g475"></a> | 未指定 | 20 × 7168 × 3072 | 24 | decode | 6 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G476<a id="g476"></a> | 未指定 | 24 × 6144 × 7168 | 24 | decode | 8 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G477<a id="g477"></a> | 未指定 | 24 × 7168 × 3072 | 24 | decode | 8 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G478<a id="g478"></a> | 未指定 | 30 × 4096 × 2048 | 32 | decode | 3 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G479<a id="g479"></a> | 未指定 | 30 × 4096 × 4096 | 32 | decode | 3 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G480<a id="g480"></a> | 未指定 | 36 × 4096 × 2048 | 32 | decode | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G481<a id="g481"></a> | 未指定 | 36 × 4096 × 4096 | 32 | decode | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G482<a id="g482"></a> | 未指定 | 45 × 4096 × 2048 | 32 | decode | 9 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G483<a id="g483"></a> | 未指定 | 45 × 4096 × 4096 | 32 | decode | 9 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G484<a id="g484"></a> | 未指定 | 54 × 4096 × 2048 | 32 | decode | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G485<a id="g485"></a> | 未指定 | 54 × 4096 × 4096 | 32 | decode | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G486<a id="g486"></a> | P0 | 64 × 1024 × 2048 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G487<a id="g487"></a> | P0 | 64 × 2048 × 512 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G488<a id="g488"></a> | 未指定 | 160 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G489<a id="g489"></a> | 未指定 | 160 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G490<a id="g490"></a> | P0 | 208 × 1024 × 2048 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G491<a id="g491"></a> | P0 | 208 × 2048 × 512 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G492<a id="g492"></a> | 未指定 | 320 × 6144 × 7168 | 48 | prefill | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G493<a id="g493"></a> | 未指定 | 320 × 7168 × 3072 | 48 | prefill | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G494<a id="g494"></a> | 未指定 | 416 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G495<a id="g495"></a> | 未指定 | 416 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G496<a id="g496"></a> | 未指定 | 576 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G497<a id="g497"></a> | 未指定 | 576 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G498<a id="g498"></a> | 未指定 | 672 × 4096 × 2048 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C007](#c007) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G499<a id="g499"></a> | 未指定 | 672 × 4096 × 4096 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C007](#c007) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G500<a id="g500"></a> | 未指定 | 832 × 6144 × 7168 | 48 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G501<a id="g501"></a> | 未指定 | 832 × 7168 × 3072 | 48 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G502<a id="g502"></a> | 未指定 | 960 × 4096 × 2048 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C008](#c008) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G503<a id="g503"></a> | 未指定 | 960 × 4096 × 4096 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C008](#c008) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G504<a id="g504"></a> | 未指定 | 1248 × 4096 × 2048 | 32 | prefill | 32 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C016](#c016) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G505<a id="g505"></a> | 未指定 | 1248 × 4096 × 4096 | 32 | prefill | 32 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C016](#c016) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |

<a id="table-9"></a>

## 表 9：grouped_gemm · FP4E2M1 × FP4E2M1 → BF16（新版新增）

34 行、304 条新版 Excel 来源，编号 G506–G539。各行与表 8 按相同 MNK/E/来源顺序对应，具体成对 G 编号见本次编号映射的 paired_legacy_G。A/B scale 均为 E8M0/K32，A 为动态量化。

当前仓库的 gemm_afp4wfp4 是 dense FP4×FP4 参考，不能据此宣称已有等长/ragged grouped 入口或本需求 PASS。所有新增行均待 grouped 接口适配与验证。FP4 packing/nibble 顺序、scale 实际布局、offsets/空组规则须由调用数据明确；此处只记录逻辑需求。

| 编号 | 优先级 | M × N × K | 分组数 E | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G506<a id="g506"></a> | P0 | 2 × 1024 × 2048 | 80 | decode | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G507<a id="g507"></a> | P0 | 2 × 2048 × 512 | 80 | decode | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G508<a id="g508"></a> | 未指定 | 20 × 6144 × 7168 | 24 | decode | 6 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G509<a id="g509"></a> | 未指定 | 20 × 7168 × 3072 | 24 | decode | 6 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G510<a id="g510"></a> | 未指定 | 24 × 6144 × 7168 | 24 | decode | 8 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G511<a id="g511"></a> | 未指定 | 24 × 7168 × 3072 | 24 | decode | 8 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G512<a id="g512"></a> | 未指定 | 30 × 4096 × 2048 | 32 | decode | 3 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G513<a id="g513"></a> | 未指定 | 30 × 4096 × 4096 | 32 | decode | 3 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G514<a id="g514"></a> | 未指定 | 36 × 4096 × 2048 | 32 | decode | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G515<a id="g515"></a> | 未指定 | 36 × 4096 × 4096 | 32 | decode | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G516<a id="g516"></a> | 未指定 | 45 × 4096 × 2048 | 32 | decode | 9 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G517<a id="g517"></a> | 未指定 | 45 × 4096 × 4096 | 32 | decode | 9 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G518<a id="g518"></a> | 未指定 | 54 × 4096 × 2048 | 32 | decode | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G519<a id="g519"></a> | 未指定 | 54 × 4096 × 4096 | 32 | decode | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G520<a id="g520"></a> | P0 | 64 × 1024 × 2048 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G521<a id="g521"></a> | P0 | 64 × 2048 × 512 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G522<a id="g522"></a> | 未指定 | 160 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G523<a id="g523"></a> | 未指定 | 160 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G524<a id="g524"></a> | P0 | 208 × 1024 × 2048 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G525<a id="g525"></a> | P0 | 208 × 2048 × 512 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G526<a id="g526"></a> | 未指定 | 320 × 6144 × 7168 | 48 | prefill | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G527<a id="g527"></a> | 未指定 | 320 × 7168 × 3072 | 48 | prefill | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G528<a id="g528"></a> | 未指定 | 416 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G529<a id="g529"></a> | 未指定 | 416 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G530<a id="g530"></a> | 未指定 | 576 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G531<a id="g531"></a> | 未指定 | 576 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G532<a id="g532"></a> | 未指定 | 672 × 4096 × 2048 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C007](#c007) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G533<a id="g533"></a> | 未指定 | 672 × 4096 × 4096 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C007](#c007) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G534<a id="g534"></a> | 未指定 | 832 × 6144 × 7168 | 48 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G535<a id="g535"></a> | 未指定 | 832 × 7168 × 3072 | 48 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G536<a id="g536"></a> | 未指定 | 960 × 4096 × 2048 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C008](#c008) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G537<a id="g537"></a> | 未指定 | 960 × 4096 × 4096 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C008](#c008) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G538<a id="g538"></a> | 未指定 | 1248 × 4096 × 2048 | 32 | prefill | 32 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C016](#c016) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G539<a id="g539"></a> | 未指定 | 1248 × 4096 × 4096 | 32 | prefill | 32 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C016](#c016) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |

<a id="requirements-cross-check"></a>

## 新版 Excel 与历史公开补充核对（2026-11-04）

新版 act_quant / weight_quant 显式标签是本次需求依据。下表的公开信息沿用 2026-10-23 审计中固定 revision 的快照，仅说明历史补充与新版的关系；本次没有重新检索公开 main，也不将候选 checkpoint 等同于 BU 实际部署。历史证据见 [2026-10-23 核对记录](debug_tuning/requirements_audit_20261023/REPORT.md)。

| 模型 / 历史来源 | 新版激活量化 / scale | 新版权重量化 / scale | 与历史公开补充的关系 |
| --- | --- | --- | --- |
| <a id="quant-flash"></a>[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) | dense：FP8、FP32/K128/dyn；grouped：FP4、E8M0/K32/dyn | dense：FP8、E8M0/N128/K128；grouped：FP4、E8M0/K32 | B 与旧公开快照相符；dense A E8M0→FP32、grouped A FP8→FP4 是 Excel 修订。 |
| <a id="quant-pro"></a>[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) | 同 Flash | 同 Flash | 新版 grouped FP4×FP4 列表 9；FP8×FP4 作为兼容需求保留表 8。 |
| <a id="quant-glm"></a>[GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8/blob/f396cf805182f4ca10fa675e1a99815b3ca384db/config.json) | FP8、FP32/K128/dyn，含 indexer_wk | FP8、FP32*/N128/K128 | K/N 粒度与历史补充相符；星号仍未释义，不能当作已确认的无星号存储 dtype。 |
| <a id="quant-q35"></a>[Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8/blob/9d1823d2dee688a6b25e77009dc727688c44936e/config.json) | FP8、FP32/K128/dyn，含 o_proj | FP8、BF16/N128/K128 | 新版已明确这些字段，原“网补”改为 Excel 来源；checkpoint 与框架转换后的 scale dtype 仍需分开。 |
| <a id="quant-q397"></a>[Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/blob/ea5b4f81096f3901c91dea97f81324302495781d/config.json) | 同 Q35 FP8 | FP8、BF16/N128/K128 | 同一 G 混合 DS/Qwen 时，B dtype 按 X 来源保留两种组合。 |
| <a id="quant-hy3"></a>[Hy3-FP8 / Hunyuan-V3](https://huggingface.co/tencent/Hy3-FP8/blob/ecc1d8e194e093f33177f2f0ef7ce8f397b2d68b/config.json) | FP8、FP32/token/dyn，含 o_proj | FP8、FP32*/channel | 替换旧候选的 BF16 scalar 权重 / FP32 scalar 静态激活；粒度与动态性变化不能只用 scale cast 解释。 |
| <a id="quant-mx4"></a>[Qwen3.5-35B-A3B-MXFP4](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/blob/2e19c6576db91e5d5a93455415619262218bf8a1/config.json) | grouped：FP4、E8M0/K32/dyn | grouped：FP4、E8M0/K32 | 新版 A FP4 与历史 AMD 候选方向一致；表 8 的 A FP8 是额外兼容需求。 |
| <a id="quant-kimi"></a>[Kimi-K2.6 / W4A16](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json) | BF16，无 A scale | INT4、BF16*/K32，含 EAGLE3 draft | 新版补齐 draft 的 g32/BF16*；对称性、zero-point 与 packing 未由标签确定，不直接认定为 AWQ。 |

同名 quantize_o_proj 与 o_proj 的 shape 衔接在 Hy3/Qwen 中仍有 18 处需说明；Excel 未给 tensor ID，不能假定相邻同名记录必定直连，也不能据此擅改 MNK。按 GEMM 自身 act_quant 标签记录需求，同时保留连线问题。grouped 量化输出与各专家 token 总数之间的复制/padding/路由映射也未给齐，不能只凭总元素关系推断 ragged offsets。

FP32*/BF16* 共涉及 248 条 GEMM 来源，原工作簿没有星号解释。FP4/INT4 packing、nibble 顺序、scale 物理布局/stride、分组 offsets/空组规则，以及部署 checkpoint/框架 revision，均不由量化标签直接确定。逻辑需求已采用新标签，物理 ABI 与实际数据链路仍须单独确认。

G171 的当前定位、stage/layer_count 和原始量化标签见新编号映射；旧表内的行号和候选量化输出位置不应当作新版坐标。Pro a 的 PP 标签与 layer_count 修订已原样保存在 gemm-sources.csv，不按 layer_count 重算频次。

### 原始文件与部署方案

X01–X13 沿用原文件编号，新增 Qwen3-VL 为 X14；SHA256、完整 sheet 描述见 [新版来源清单](debug_tuning/requirements_audit_20261104/source-manifest.json)。

| 来源 | 新版文件 | prefill / P 侧方案（原文摘要） |
| --- | --- | --- |
| X01 | `shape_01_flash_64k_s150.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X02 | `shape_01_flash_64k_s2000.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X03 | `shape_02_flash_128k_s150.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X04 | `shape_02_flash_128k_s2000.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X05 | `shape_03_pro_128k_s2000_a.xlsx` | P 实例: 16 卡 tp8 dp8 pp2 ep8 / chunk 6656 / 不开投机 |
| X06 | `shape_03_pro_128k_s2000_b.xlsx` | P 实例: 8 卡 tp8 dp4 ep8 / chunk 6656 / 不开投机 |
| X07 | `shape_05_glm_128k_s2000.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X08 | `shape_06_glm_64k_s2000.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X09 | `shape_07_q35_fp8_4k_s150.xlsx` | 1 卡 tp1 / chunk 6656 / 不开投机 |
| X10 | `shape_08_q35_mxfp4_128k_s150.xlsx` | 1 卡 tp1 / chunk 6656 / 不开投机 |
| X11 | `shape_09_kimi26_128k_s2000.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X12 | `shape_10_hy3_128k_s2000.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X13 | `shape_11_q397b_128k_s2000.xlsx` | P 实例: 8 卡 tp8 dp8 ep8 / chunk 6656 / 不开投机 |
| X14 | `shape_12_qwen3vl8b_226_s150.xlsx` | 1 卡 tp1 / bs512 / chunk 6656 / 不开投机 |

### 尚未纳入 G 编号表的 grouped GEMM 需求

保留原 42 个缺口（36 个 FP8×FP8、70 条来源；6 个 BF16×INT4、10 条来源），不擅自升级接口状态。scale/量化标签同步更新为新版 Excel，尤其 Hy3 的 token/channel 与 Kimi 的 BF16*。以下全部待范围确认 / grouped 接口映射与验证；量化值、scale 与分组索引由算子测试预先准备，不扩展为路由/激活/fused_experts 包装验收。

逐条原文件/sheet/行号见 [新版缺口映射](debug_tuning/requirements_audit_20261104/grouped-gaps.csv)。

| M × N × K | 优先级 | E | A × B → 输出 | A scale（新版） | B scale（新版） | X 来源 | 记录数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 × 1024 × 2048<a id="gap-001"></a> | P0 | 221 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 3 × 2048 × 512<a id="gap-002"></a> | P0 | 221 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 3 × 3072 × 4096<a id="gap-003"></a> | 未指定 | 11 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 1 |
| 3 × 4096 × 1536<a id="gap-004"></a> | 未指定 | 11 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 1 |
| 4 × 2048 × 4096<a id="gap-005"></a> | 未指定 | 63 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 1 |
| 4 × 4096 × 1024<a id="gap-006"></a> | 未指定 | 63 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 1 |
| 4 × 4096 × 6144<a id="gap-007"></a> | 未指定 | 15 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 1 |
| 4 × 6144 × 2048<a id="gap-008"></a> | 未指定 | 15 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 1 |
| 6 × 4096 × 6144<a id="gap-009"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 1 |
| 6 × 4096 × 7168<a id="gap-010"></a> | 未指定 | 24 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 1 |
| 6 × 6144 × 2048<a id="gap-011"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 1 |
| 6 × 7168 × 2048<a id="gap-012"></a> | 未指定 | 24 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 1 |
| 11 × 3072 × 4096<a id="gap-013"></a> | 未指定 | 12 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 1 |
| 11 × 4096 × 1536<a id="gap-014"></a> | 未指定 | 12 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 1 |
| 12 × 4096 × 6144<a id="gap-015"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 1 |
| 12 × 6144 × 2048<a id="gap-016"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 1 |
| 15 × 2048 × 4096<a id="gap-017"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 2 |
| 15 × 4096 × 1024<a id="gap-018"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 2 |
| 24 × 4096 × 6144<a id="gap-019"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 1 |
| 24 × 6144 × 2048<a id="gap-020"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 1 |
| 80 × 1024 × 2048<a id="gap-021"></a> | P0 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 80 × 2048 × 512<a id="gap-022"></a> | P0 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 171 × 3072 × 4096<a id="gap-023"></a> | 未指定 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 2 |
| 171 × 4096 × 1536<a id="gap-024"></a> | 未指定 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 2 |
| 208 × 1024 × 2048<a id="gap-025"></a> | P0 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 208 × 2048 × 512<a id="gap-026"></a> | P0 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 560 × 2048 × 4096<a id="gap-027"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 4 |
| 560 × 4096 × 1024<a id="gap-028"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 4 |
| 640 × 4096 × 6144<a id="gap-029"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 2 |
| 640 × 6144 × 2048<a id="gap-030"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 2 |
| 854 × 4096 × 7168<a id="gap-031"></a> | 未指定 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 2 |
| 854 × 7168 × 2048<a id="gap-032"></a> | 未指定 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 2 |
| 1040 × 2048 × 4096<a id="gap-033"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 4 |
| 1040 × 4096 × 1024<a id="gap-034"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 4 |
| 1110 × 4096 × 7168<a id="gap-035"></a> | 未指定 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 2 |
| 1110 × 7168 × 2048<a id="gap-036"></a> | 未指定 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 2 |
| 1152 × 4096 × 6144<a id="gap-037"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 2 |
| 1152 × 6144 × 2048<a id="gap-038"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 2 |
| 1664 × 4096 × 6144<a id="gap-039"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07/X08 | 4 |
| 1664 × 6144 × 2048<a id="gap-040"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07/X08 | 4 |
| 2219 × 3072 × 4096<a id="gap-041"></a> | 未指定 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 2 |
| 2219 × 4096 × 1536<a id="gap-042"></a> | 未指定 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 2 |

<a id="model-config-index"></a>

## 模型与配置索引

以下编号对应主表的「模型 / 配置」列。同一完整组合共用一个编号；原始模型名称、精度和方括号内的配置原样保留。C004 的 Pro a/b 与修订后的 PP stage 以新版来源清单为准；“FP8”等模型标签不覆盖行内明确的混合 dtype。

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

优先级：P0。

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

优先级：P0。

Qwen3.5-35B-A3B/MXFP4[128k/1k/s150]

### C019

优先级：P0。

Qwen3.5-35B-A3B/FP8[4k/1k/s150]; Qwen3.5-35B-A3B/MXFP4[128k/1k/s150]

### C020

优先级：P0。

Qwen3-VL-8B/BF16[主干输入226/输出300/s150；文本30+448×448图像；TP1/bs512]。沿用文件第3行的 s150 标签，需求描述实际输出长度为300；完整原文见 X14 来源清单。

<!-- BoltOps function definitions: links remain on the current branch. -->
[fn-bmm]: ../../boltops/gemm/triton/batched_gemm_bf16.py#L323
[fn-a16]: ../../boltops/gemm/triton/gemm_a16w16.py#L33
[fn-persistent]: ../../boltops/gemm/triton/matmul_persistent.py#L205
[fn-fp8-aiter]: ../../boltops/gemm/triton/gemm_a8w8_blockscale.py#L29
[fn-fp8-vllm]: ../../boltops/gemm/triton/w8a8_triton_block_scaled_mm.py#L127
[fn-awq]: ../../boltops/gemm/triton/awq_gemm_triton.py#L351
[fn-fp8fp4]: ../../boltops/gemm/triton/gemm_a8wfp4.py#L70
[fn-fp4dense]: ../../boltops/gemm/triton/gemm_afp4wfp4.py#L253
