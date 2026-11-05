# 少伯 BU GEMM 支持需求

原编号来源：`/xukang/GEMM支持清单.html`（2026-10-12）；本次于 2026-11-04 按新版目录 `shaobob0_shapes_1102/` 的 14 份 XLSX、39 个 sheet 更新。原 G/C 编号保留，不填补历史删除留下的编号空洞。原 HTML SHA256：`187673895b820bfb89d6b2890ffb35fa378a183e8922238a8f96df2ff738d11c`。

新版 Excel 共 2526 条 GEMM/BMM/grouped 来源：489 个已编号需求覆盖 2446 条，另有 42 个未编号 grouped shape 覆盖 80 条。按用户要求另保留表 8 的 34 个 FP8×FP4 兼容需求（304 条对应来源），因此主表共 **523 行支持需求**，表内频次合计 2750；兼容表与新版 FP4×FP4 表复用同一来源谱系，不能把 2750 当作新版 Excel 独立行数。

新增表 9 的 FP4×FP4 为 G506–G539；新增 Qwen3-VL-8B BF16 linear 为 G540–G559。所有 scale、量化粒度和动态/静态标签以新版 Excel 显式字段为准；原 FP8×FP4 需求与编号保留，其 scale 同步采用新版 E8M0/K32。

接口级 smoke 沿用历史证据；**本次未运行 GPU 测试，新增 FP4 grouped、新 BF16 shape，以及变更后的 scale 组合均未因此获得 PASS。** “接口 smoke 通过”不表示本行完整 MNK/E/scale 已验证或已调优。实验记录规范见 [调优记录](debug_tuning/TUNING.md)。逐行来源、原始量化标签与校验见 [本次核对记录](debug_tuning/requirements_audit_20261104/README.md) / [编号映射](debug_tuning/requirements_audit_20261104/case-provenance.csv)。

<a name="p0-快速索引"></a>

## P0 快速索引

当前 P0：Qwen3.5-35B-A3B 的 FP8、MXFP4 两版，以及 Qwen3-VL-8B BF16。优先级按模型来源覆盖其全部需求，含保留的 FP8×FP4 兼容项；其他模型暂记“未指定”。点击数量可跳到对应首条，展开下方模型编号清单可逐项跳转。

| 模型 / 配置 | 优先级 | BF16 dense | FP8 dense | FP4×FP4 grouped | FP8×FP4 兼容 | 未编号 grouped |
| --- | --- | --- | --- | --- | --- | --- |
| [Qwen3.5-35B-A3B FP8](#c013) | P0 | [7 项](#g291) | [15 项](#g076) | — | — | [6 项](#gap-001) |
| [Qwen3.5-35B-A3B MXFP4](#c018) | P0 | [22 项](#g256) | — | [6 项](#g506) | [6 项](#g472) | — |
| [Qwen3-VL-8B BF16](#c020) | P0 | [20 项](#g540) | — | — | — | — |

合计 **74 条已编号 P0 需求**（其中 47 条 BF16 dense、15 条 FP8 dense、6 条新版 FP4×FP4、6 条 FP8×FP4 兼容），另有 **6 个 P0 grouped 缺口**。G342/G347 为两种 Qwen35 配置共享，主表各只列一次；快速索引的逐模型数量不直接相加。

<a name="p0-q35-fp8"></a>
<details>
<summary>Qwen3.5-35B-A3B FP8：P0 需求编号</summary>

- BF16 dense：[G291](#g291)、[G292](#g292)、[G295](#g295)、[G321](#g321)、[G323](#g323)、[G342](#g342)、[G347](#g347)

- FP8 dense：[G076](#g076)、[G077](#g077)、[G078](#g078)、[G079](#g079)、[G080](#g080)、[G134](#g134)、[G136](#g136)、[G137](#g137)、[G148](#g148)、[G149](#g149)、[G194](#g194)、[G197](#g197)、[G198](#g198)、[G216](#g216)、[G218](#g218)

- 未编号 grouped（待范围 / 接口确认）：[3×1024×2048 / E221](#gap-001)、[3×2048×512 / E221](#gap-002)、[80×1024×2048 / E256](#gap-021)、[80×2048×512 / E256](#gap-022)、[208×1024×2048 / E256](#gap-025)、[208×2048×512 / E256](#gap-026)

</details>

<a name="p0-q35-mxfp4"></a>
<details>
<summary>Qwen3.5-35B-A3B MXFP4：P0 需求编号</summary>

- BF16 dense：[G256](#g256)、[G257](#g257)、[G259](#g259)、[G260](#g260)、[G261](#g261)、[G263](#g263)、[G264](#g264)、[G266](#g266)、[G313](#g313)、[G314](#g314)、[G315](#g315)、[G316](#g316)、[G317](#g317)、[G318](#g318)、[G319](#g319)、[G342](#g342)、[G347](#g347)、[G352](#g352)、[G353](#g353)、[G354](#g354)、[G360](#g360)、[G362](#g362)

- FP4×FP4 grouped：[G506](#g506)、[G507](#g507)、[G520](#g520)、[G521](#g521)、[G524](#g524)、[G525](#g525)

- FP8×FP4 grouped（兼容）：[G472](#g472)、[G473](#g473)、[G486](#g486)、[G487](#g487)、[G490](#g490)、[G491](#g491)

</details>

<a name="p0-qwen3-vl-8b"></a>
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

<a name="table-1"></a>

## 表 1：bmm · BF16 × BF16 → BF16

29 行需求，累计 194 条实例。编号在本清单中全局唯一。

本表仅记录 BF16 BMM；函数使用 `B[batch,N,K]` 权重布局，模型调用需匹配 stride。

| 编号 | 优先级 | M × N × K | batch | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| G001<a name="g001"></a> | 未指定 | 6 × 256 × 512 | 64 | decode | 1 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G002<a name="g002"></a> | 未指定 | 6 × 512 × 192 | 64 | decode | 1 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G003<a name="g003"></a> | 未指定 | 12 × 256 × 512 | 64 | decode | 1 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G004<a name="g004"></a> | 未指定 | 12 × 512 × 192 | 64 | decode | 1 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G005<a name="g005"></a> | 未指定 | 16 × 128 × 512 | 64 | decode | 2 | [C003](#c003) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G006<a name="g006"></a> | 未指定 | 16 × 512 × 128 | 64 | decode | 2 | [C003](#c003) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G007<a name="g007"></a> | 未指定 | 24 × 256 × 512 | 64 | decode | 2 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G008<a name="g008"></a> | 未指定 | 24 × 512 × 192 | 64 | decode | 2 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G009<a name="g009"></a> | 未指定 | 48 × 256 × 512 | 64 | decode | 2 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G010<a name="g010"></a> | 未指定 | 48 × 512 × 192 | 64 | decode | 2 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G011<a name="g011"></a> | 未指定 | 80 × 1024 × 4096 | 16 | decode | 6 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G012<a name="g012"></a> | 未指定 | 96 × 1024 × 4096 | 16 | decode | 8 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G013<a name="g013"></a> | 未指定 | 160 × 1024 × 4096 | 8 | decode | 3 | [C005](#c005) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G014<a name="g014"></a> | 未指定 | 192 × 1024 × 4096 | 8 | decode | 4 | [C005](#c005) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G015<a name="g015"></a> | 未指定 | 240 × 1024 × 4096 | 8 | decode | 9 | [C006](#c006) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G016<a name="g016"></a> | 未指定 | 288 × 1024 × 4096 | 8 | decode | 12 | [C006](#c006) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G017<a name="g017"></a> | 未指定 | 2560 × 256 × 512 | 64 | prefill | 4 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G018<a name="g018"></a> | 未指定 | 2560 × 512 × 192 | 64 | prefill | 4 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G019<a name="g019"></a> | 未指定 | 2560 × 1024 × 4096 | 16 | prefill | 12 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G020<a name="g020"></a> | 未指定 | 2560 × 1024 × 4096 | 8 | prefill | 4 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G021<a name="g021"></a> | 未指定 | 3584 × 1024 × 4096 | 8 | prefill | 16 | [C007](#c007) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G022<a name="g022"></a> | 未指定 | 4608 × 256 × 512 | 64 | prefill | 4 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G023<a name="g023"></a> | 未指定 | 4608 × 512 × 192 | 64 | prefill | 4 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G024<a name="g024"></a> | 未指定 | 4608 × 1024 × 4096 | 16 | prefill | 4 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G025<a name="g025"></a> | 未指定 | 5120 × 1024 × 4096 | 8 | prefill | 16 | [C008](#c008) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G026<a name="g026"></a> | 未指定 | 6656 × 256 × 512 | 64 | prefill | 8 | [C009](#c009) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G027<a name="g027"></a> | 未指定 | 6656 × 512 × 192 | 64 | prefill | 8 | [C009](#c009) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G028<a name="g028"></a> | 未指定 | 6656 × 1024 × 4096 | 8 | prefill | 36 | [C010](#c010) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G029<a name="g029"></a> | 未指定 | 6656 × 1024 × 4096 | 16 | prefill | 16 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |

<a name="table-2"></a>

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

<a name="table-2-scale-2"></a>
<a name="table-2-scale-4"></a>

旧 2.2 的 E8M0/K128 已依新版归入 2.1；旧 2.4 的 per-tensor 来源已依新版 token/dyn 归入 2.3。保留旧锚点用于历史引用。

<a name="table-2-scale-1"></a>

### 2.1 A scale FP32，block size 1×128

181 行需求，累计 1127 条新版来源。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G034<a name="g034"></a> | 未指定 | 6 × 128 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G035<a name="g035"></a> | 未指定 | 6 × 2624 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G036<a name="g036"></a> | 未指定 | 6 × 4096 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G037<a name="g037"></a> | 未指定 | 6 × 4096 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G038<a name="g038"></a> | 未指定 | 6 × 6144 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G039<a name="g039"></a> | 未指定 | 6 × 6144 × 16384 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G040<a name="g040"></a> | 未指定 | 6 × 16384 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G041<a name="g041"></a> | 未指定 | 12 × 128 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G042<a name="g042"></a> | 未指定 | 12 × 2624 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G043<a name="g043"></a> | 未指定 | 12 × 4096 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G044<a name="g044"></a> | 未指定 | 12 × 4096 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G045<a name="g045"></a> | 未指定 | 12 × 6144 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G046<a name="g046"></a> | 未指定 | 12 × 6144 × 16384 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G047<a name="g047"></a> | 未指定 | 12 × 16384 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G054<a name="g054"></a> | 未指定 | 24 × 128 × 6144 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G055<a name="g055"></a> | 未指定 | 24 × 2624 × 6144 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G056<a name="g056"></a> | 未指定 | 24 × 4096 × 2048 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G057<a name="g057"></a> | 未指定 | 24 × 4096 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G058<a name="g058"></a> | 未指定 | 24 × 6144 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G059<a name="g059"></a> | 未指定 | 24 × 6144 × 12288 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G060<a name="g060"></a> | 未指定 | 24 × 6144 × 16384 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G061<a name="g061"></a> | 未指定 | 24 × 16384 × 2048 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G062<a name="g062"></a> | 未指定 | 24 × 24576 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G063<a name="g063"></a> | 未指定 | 48 × 128 × 6144 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G064<a name="g064"></a> | 未指定 | 48 × 2048 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G065<a name="g065"></a> | 未指定 | 48 × 2624 × 6144 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G066<a name="g066"></a> | 未指定 | 48 × 4096 × 1024 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G067<a name="g067"></a> | 未指定 | 48 × 4096 × 2048 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G068<a name="g068"></a> | 未指定 | 48 × 4096 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G069<a name="g069"></a> | 未指定 | 48 × 4096 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G070<a name="g070"></a> | 未指定 | 48 × 6144 × 2048 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G071<a name="g071"></a> | 未指定 | 48 × 6144 × 12288 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G072<a name="g072"></a> | 未指定 | 48 × 6144 × 16384 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G073<a name="g073"></a> | 未指定 | 48 × 8704 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G074<a name="g074"></a> | 未指定 | 48 × 16384 × 2048 | decode | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G075<a name="g075"></a> | 未指定 | 48 × 24576 × 6144 | decode | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G076<a name="g076"></a> | P0 | 64 × 1024 × 2048 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G077<a name="g077"></a> | P0 | 64 × 2048 × 512 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G078<a name="g078"></a> | P0 | 64 × 2048 × 4096 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G079<a name="g079"></a> | P0 | 64 × 9216 × 2048 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G080<a name="g080"></a> | P0 | 64 × 12288 × 2048 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G081<a name="g081"></a> | 未指定 | 80 × 512 × 7168 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G082<a name="g082"></a> | 未指定 | 80 × 1536 × 7168 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G083<a name="g083"></a> | 未指定 | 80 × 6144 × 7168 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G084<a name="g084"></a> | 未指定 | 80 × 7168 × 3072 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G085<a name="g085"></a> | 未指定 | 80 × 7168 × 16384 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G086<a name="g086"></a> | 未指定 | 80 × 65536 × 1536 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G087<a name="g087"></a> | 未指定 | 96 × 512 × 7168 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G088<a name="g088"></a> | 未指定 | 96 × 1536 × 7168 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G089<a name="g089"></a> | 未指定 | 96 × 6144 × 7168 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G090<a name="g090"></a> | 未指定 | 96 × 7168 × 3072 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G091<a name="g091"></a> | 未指定 | 96 × 7168 × 16384 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G092<a name="g092"></a> | 未指定 | 96 × 8192 × 1536 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G093<a name="g093"></a> | 未指定 | 96 × 65536 × 1536 | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G094<a name="g094"></a> | 未指定 | 160 × 512 × 4096 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G095<a name="g095"></a> | 未指定 | 160 × 1024 × 4096 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G096<a name="g096"></a> | 未指定 | 160 × 4096 × 2048 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G097<a name="g097"></a> | 未指定 | 160 × 4096 × 4096 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G098<a name="g098"></a> | 未指定 | 160 × 4096 × 8192 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G099<a name="g099"></a> | 未指定 | 160 × 32768 × 1024 | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G100<a name="g100"></a> | 未指定 | 192 × 512 × 4096 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G101<a name="g101"></a> | 未指定 | 192 × 1024 × 4096 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G102<a name="g102"></a> | 未指定 | 192 × 2048 × 4096 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G103<a name="g103"></a> | 未指定 | 192 × 4096 × 1024 | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G104<a name="g104"></a> | 未指定 | 192 × 4096 × 2048 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G105<a name="g105"></a> | 未指定 | 192 × 4096 × 4096 | decode | 6 | FP32@1×128（动态） | X03：E8M0@128×128（4条）<br>X13：BF16@128×128（2条） | [C014](#c014) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G106<a name="g106"></a> | 未指定 | 192 × 4096 × 8192 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G107<a name="g107"></a> | 未指定 | 192 × 8192 × 1024 | decode | 2 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G108<a name="g108"></a> | 未指定 | 192 × 8704 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G109<a name="g109"></a> | 未指定 | 192 × 10240 × 4096 | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G110<a name="g110"></a> | 未指定 | 192 × 32768 × 1024 | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G111<a name="g111"></a> | 未指定 | 240 × 512 × 4096 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G112<a name="g112"></a> | 未指定 | 240 × 1024 × 4096 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G113<a name="g113"></a> | 未指定 | 240 × 4096 × 2048 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G114<a name="g114"></a> | 未指定 | 240 × 4096 × 4096 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G115<a name="g115"></a> | 未指定 | 240 × 4096 × 8192 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G116<a name="g116"></a> | 未指定 | 240 × 32768 × 1024 | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G117<a name="g117"></a> | 未指定 | 288 × 512 × 4096 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G118<a name="g118"></a> | 未指定 | 288 × 1024 × 4096 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G119<a name="g119"></a> | 未指定 | 288 × 4096 × 2048 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G120<a name="g120"></a> | 未指定 | 288 × 4096 × 4096 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G121<a name="g121"></a> | 未指定 | 288 × 4096 × 8192 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G122<a name="g122"></a> | 未指定 | 288 × 8192 × 1024 | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G123<a name="g123"></a> | 未指定 | 288 × 32768 × 1024 | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G130<a name="g130"></a> | 未指定 | 2560 × 128 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G131<a name="g131"></a> | 未指定 | 2560 × 512 × 6144 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G132<a name="g132"></a> | 未指定 | 2560 × 512 × 7168 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G133<a name="g133"></a> | 未指定 | 2560 × 768 × 7168 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G134<a name="g134"></a> | P0 | 2560 × 1024 × 2048 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G135<a name="g135"></a> | 未指定 | 2560 × 1536 × 7168 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G136<a name="g136"></a> | P0 | 2560 × 2048 × 512 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G137<a name="g137"></a> | P0 | 2560 × 2048 × 4096 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G138<a name="g138"></a> | 未指定 | 2560 × 2624 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G139<a name="g139"></a> | 未指定 | 2560 × 3072 × 6144 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G140<a name="g140"></a> | 未指定 | 2560 × 4096 × 2048 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G141<a name="g141"></a> | 未指定 | 2560 × 6144 × 256 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G142<a name="g142"></a> | 未指定 | 2560 × 6144 × 1536 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G143<a name="g143"></a> | 未指定 | 2560 × 6144 × 16384 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G144<a name="g144"></a> | 未指定 | 2560 × 7168 × 384 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G145<a name="g145"></a> | 未指定 | 2560 × 7168 × 8192 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G146<a name="g146"></a> | 未指定 | 2560 × 7168 × 16384 | prefill | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G147<a name="g147"></a> | 未指定 | 2560 × 8192 × 1536 | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G148<a name="g148"></a> | P0 | 2560 × 9216 × 2048 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G149<a name="g149"></a> | P0 | 2560 × 12288 × 2048 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G150<a name="g150"></a> | 未指定 | 2560 × 16384 × 2048 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G151<a name="g151"></a> | 未指定 | 2560 × 32768 × 1536 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G152<a name="g152"></a> | 未指定 | 2560 × 65536 × 1536 | prefill | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G153<a name="g153"></a> | 未指定 | 3584 × 256 × 4096 | prefill | 4 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G154<a name="g154"></a> | 未指定 | 3584 × 512 × 4096 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G155<a name="g155"></a> | 未指定 | 3584 × 1024 × 4096 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G156<a name="g156"></a> | 未指定 | 3584 × 4096 × 128 | prefill | 4 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G157<a name="g157"></a> | 未指定 | 3584 × 4096 × 256 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G158<a name="g158"></a> | 未指定 | 3584 × 4096 × 8192 | prefill | 19 | FP32@1×128（动态） | X03：E8M0@128×128（8条）<br>X04：E8M0@128×128（8条）<br>X13：BF16@128×128（3条） | [C015](#c015) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G159<a name="g159"></a> | 未指定 | 3584 × 8192 × 1024 | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G160<a name="g160"></a> | 未指定 | 3584 × 17408 × 4096 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G161<a name="g161"></a> | 未指定 | 3584 × 20480 × 4096 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G162<a name="g162"></a> | 未指定 | 3584 × 32768 × 1024 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G165<a name="g165"></a> | 未指定 | 4608 × 128 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G166<a name="g166"></a> | 未指定 | 4608 × 512 × 6144 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G167<a name="g167"></a> | 未指定 | 4608 × 512 × 7168 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G168<a name="g168"></a> | 未指定 | 4608 × 768 × 7168 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G169<a name="g169"></a> | 未指定 | 4608 × 1536 × 7168 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G170<a name="g170"></a> | 未指定 | 4608 × 2624 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G171<a name="g171"></a> | 未指定 | 4608 × 3072 × 6144 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G172<a name="g172"></a> | 未指定 | 4608 × 4096 × 2048 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G173<a name="g173"></a> | 未指定 | 4608 × 6144 × 256 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G174<a name="g174"></a> | 未指定 | 4608 × 6144 × 1536 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G175<a name="g175"></a> | 未指定 | 4608 × 6144 × 16384 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G176<a name="g176"></a> | 未指定 | 4608 × 7168 × 384 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G177<a name="g177"></a> | 未指定 | 4608 × 7168 × 16384 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G178<a name="g178"></a> | 未指定 | 4608 × 8192 × 1536 | prefill | 2 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G179<a name="g179"></a> | 未指定 | 4608 × 16384 × 2048 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G180<a name="g180"></a> | 未指定 | 4608 × 65536 × 1536 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G181<a name="g181"></a> | 未指定 | 5120 × 512 × 4096 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G182<a name="g182"></a> | 未指定 | 5120 × 1024 × 4096 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G183<a name="g183"></a> | 未指定 | 5120 × 4096 × 256 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G184<a name="g184"></a> | 未指定 | 5120 × 4096 × 8192 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G185<a name="g185"></a> | 未指定 | 5120 × 8192 × 1024 | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G186<a name="g186"></a> | 未指定 | 5120 × 32768 × 1024 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G187<a name="g187"></a> | 未指定 | 6656 × 128 × 6144 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G188<a name="g188"></a> | 未指定 | 6656 × 256 × 4096 | prefill | 4 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G190<a name="g190"></a> | 未指定 | 6656 × 512 × 4096 | prefill | 64 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G191<a name="g191"></a> | 未指定 | 6656 × 512 × 6144 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G192<a name="g192"></a> | 未指定 | 6656 × 512 × 7168 | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G193<a name="g193"></a> | 未指定 | 6656 × 768 × 7168 | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G194<a name="g194"></a> | P0 | 6656 × 1024 × 2048 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G195<a name="g195"></a> | 未指定 | 6656 × 1024 × 4096 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G196<a name="g196"></a> | 未指定 | 6656 × 1536 × 7168 | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G197<a name="g197"></a> | P0 | 6656 × 2048 × 512 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G198<a name="g198"></a> | P0 | 6656 × 2048 × 4096 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G199<a name="g199"></a> | 未指定 | 6656 × 2624 × 6144 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G200<a name="g200"></a> | 未指定 | 6656 × 3072 × 6144 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G202<a name="g202"></a> | 未指定 | 6656 × 4096 × 128 | prefill | 4 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G204<a name="g204"></a> | 未指定 | 6656 × 4096 × 256 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G206<a name="g206"></a> | 未指定 | 6656 × 4096 × 2048 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G208<a name="g208"></a> | 未指定 | 6656 × 6144 × 256 | prefill | 4 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G209<a name="g209"></a> | 未指定 | 6656 × 6144 × 1536 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G210<a name="g210"></a> | 未指定 | 6656 × 6144 × 16384 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G211<a name="g211"></a> | 未指定 | 6656 × 7168 × 384 | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G212<a name="g212"></a> | 未指定 | 6656 × 7168 × 8192 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G213<a name="g213"></a> | 未指定 | 6656 × 7168 × 16384 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G214<a name="g214"></a> | 未指定 | 6656 × 8192 × 1024 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G215<a name="g215"></a> | 未指定 | 6656 × 8192 × 1536 | prefill | 10 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G216<a name="g216"></a> | P0 | 6656 × 9216 × 2048 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G218<a name="g218"></a> | P0 | 6656 × 12288 × 2048 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G219<a name="g219"></a> | 未指定 | 6656 × 16384 × 2048 | prefill | 8 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G220<a name="g220"></a> | 未指定 | 6656 × 17408 × 4096 | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G221<a name="g221"></a> | 未指定 | 6656 × 20480 × 4096 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G222<a name="g222"></a> | 未指定 | 6656 × 32768 × 1024 | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G223<a name="g223"></a> | 未指定 | 6656 × 32768 × 1536 | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G224<a name="g224"></a> | 未指定 | 6656 × 65536 × 1536 | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G225<a name="g225"></a> | 未指定 | 20480 × 3072 × 6144 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G226<a name="g226"></a> | 未指定 | 20480 × 6144 × 1536 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G227<a name="g227"></a> | 未指定 | 28672 × 2560 × 4096 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G228<a name="g228"></a> | 未指定 | 28672 × 4096 × 1024 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G229<a name="g229"></a> | 未指定 | 36864 × 3072 × 6144 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G230<a name="g230"></a> | 未指定 | 36864 × 6144 × 1536 | prefill | 1 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G231<a name="g231"></a> | 未指定 | 53248 × 2560 × 4096 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G232<a name="g232"></a> | 未指定 | 53248 × 3072 × 6144 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G234<a name="g234"></a> | 未指定 | 53248 × 4096 × 1024 | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G236<a name="g236"></a> | 未指定 | 53248 × 6144 × 1536 | prefill | 2 | FP32@1×128（动态） | FP32*@128×128（Excel） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a name="table-2-scale-3"></a>

### 2.3 A scale FP32，逐 token 的整行 K

25 行需求，累计 40 条新版来源。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G030<a name="g030"></a> | 未指定 | 4 × 3072 × 4096 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G031<a name="g031"></a> | 未指定 | 4 × 4096 × 1536 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G032<a name="g032"></a> | 未指定 | 4 × 4096 × 8192 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G033<a name="g033"></a> | 未指定 | 4 × 10240 × 4096 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G048<a name="g048"></a> | 未指定 | 16 × 3072 × 4096 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G049<a name="g049"></a> | 未指定 | 16 × 4096 × 1536 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G050<a name="g050"></a> | 未指定 | 16 × 4096 × 8192 | decode | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G051<a name="g051"></a> | 未指定 | 16 × 4096 × 13312 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G052<a name="g052"></a> | 未指定 | 16 × 10240 × 4096 | decode | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G053<a name="g053"></a> | 未指定 | 16 × 26624 × 4096 | decode | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G124<a name="g124"></a> | 未指定 | 512 × 384 × 4096 | prefill | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G125<a name="g125"></a> | 未指定 | 512 × 3328 × 4096 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G126<a name="g126"></a> | 未指定 | 512 × 4096 × 192 | prefill | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G127<a name="g127"></a> | 未指定 | 512 × 4096 × 1664 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G128<a name="g128"></a> | 未指定 | 512 × 4096 × 8192 | prefill | 4 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G129<a name="g129"></a> | 未指定 | 512 × 10240 × 4096 | prefill | 4 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G163<a name="g163"></a> | 未指定 | 4096 × 3328 × 4096 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G164<a name="g164"></a> | 未指定 | 4096 × 4096 × 1664 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G189<a name="g189"></a> | 未指定 | 6656 × 384 × 4096 | prefill | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G201<a name="g201"></a> | 未指定 | 6656 × 3328 × 4096 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G203<a name="g203"></a> | 未指定 | 6656 × 4096 × 192 | prefill | 2 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G205<a name="g205"></a> | 未指定 | 6656 × 4096 × 1664 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G217<a name="g217"></a> | 未指定 | 6656 × 10240 × 4096 | prefill | 4 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G233<a name="g233"></a> | 未指定 | 53248 × 3328 × 4096 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G235<a name="g235"></a> | 未指定 | 53248 × 4096 × 1664 | prefill | 1 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a name="table-2-scale-5"></a>

### 2.5 同一编号含多种 A scale

1 行需求，累计 39 条新版来源。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G207<a name="g207"></a> | 未指定 | 6656 × 4096 × 8192 | prefill | 39 | FP32@1×128（动态） × 35条 / FP32@1×K（逐 token 动态） × 4条 | X01：E8M0@128×128（8条）<br>X02：E8M0@128×128（8条）<br>X03：E8M0@128×128（8条）<br>X04：E8M0@128×128（8条）<br>X12：FP32*@channel（4条）<br>X13：BF16@128×128（3条） | [C017](#c017) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a name="table-3"></a>

## 表 3：linear · BF16 × BF16 → BF16

153 行需求，累计 348 条来源（新增 Qwen3-VL-8B 20 行 / 22 条来源）。编号在本清单中全局唯一。

本表对应 BF16 输入、BF16 输出；`gemm_a16w16` 的权重布局为 `[N,K]`。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| G237<a name="g237"></a> | 未指定 | 1 × 154880 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G238<a name="g238"></a> | 未指定 | 2 × 64640 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G239<a name="g239"></a> | 未指定 | 2 × 129280 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G240<a name="g240"></a> | 未指定 | 3 × 120832 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G241<a name="g241"></a> | 未指定 | 4 × 192 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G242<a name="g242"></a> | 未指定 | 4 × 4096 × 8192 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G243<a name="g243"></a> | 未指定 | 4 × 7168 × 21504 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G244<a name="g244"></a> | 未指定 | 4 × 120832 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G245<a name="g245"></a> | 未指定 | 4 × 154880 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G246<a name="g246"></a> | 未指定 | 4 × 163840 × 7168 | decode,prefill | 3 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G247<a name="g247"></a> | 未指定 | 6 × 32 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G248<a name="g248"></a> | 未指定 | 6 × 256 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G249<a name="g249"></a> | 未指定 | 6 × 6144 × 12288 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G250<a name="g250"></a> | 未指定 | 6 × 154880 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G251<a name="g251"></a> | 未指定 | 8 × 16160 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G252<a name="g252"></a> | 未指定 | 8 × 19360 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G253<a name="g253"></a> | 未指定 | 8 × 129280 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G254<a name="g254"></a> | 未指定 | 8 × 248320 × 4096 | prefill | 2 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G255<a name="g255"></a> | 未指定 | 12 × 32 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G256<a name="g256"></a> | P0 | 12 × 64 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G257<a name="g257"></a> | P0 | 12 × 256 × 2048 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G258<a name="g258"></a> | 未指定 | 12 × 256 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G259<a name="g259"></a> | P0 | 12 × 1024 × 2048 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G260<a name="g260"></a> | P0 | 12 × 2048 × 512 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G261<a name="g261"></a> | P0 | 12 × 2048 × 4096 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G262<a name="g262"></a> | 未指定 | 12 × 6144 × 12288 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G263<a name="g263"></a> | P0 | 12 × 9216 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G264<a name="g264"></a> | P0 | 12 × 12288 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G265<a name="g265"></a> | 未指定 | 12 × 154880 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G266<a name="g266"></a> | P0 | 12 × 248320 × 2048 | decode,prefill | 3 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G267<a name="g267"></a> | 未指定 | 16 × 192 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G268<a name="g268"></a> | 未指定 | 16 × 384 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G269<a name="g269"></a> | 未指定 | 16 × 2112 × 7168 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G270<a name="g270"></a> | 未指定 | 16 × 4096 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G271<a name="g271"></a> | 未指定 | 16 × 7168 × 2048 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G272<a name="g272"></a> | 未指定 | 16 × 7168 × 8192 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G273<a name="g273"></a> | 未指定 | 16 × 7168 × 18432 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G274<a name="g274"></a> | 未指定 | 16 × 12288 × 1536 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G275<a name="g275"></a> | 未指定 | 16 × 16160 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G276<a name="g276"></a> | 未指定 | 16 × 36864 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G277<a name="g277"></a> | 未指定 | 16 × 120832 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G278<a name="g278"></a> | 未指定 | 16 × 163840 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G279<a name="g279"></a> | 未指定 | 24 × 32 × 6144 | decode | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G280<a name="g280"></a> | 未指定 | 24 × 256 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G281<a name="g281"></a> | 未指定 | 24 × 15104 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G282<a name="g282"></a> | 未指定 | 24 × 154880 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G283<a name="g283"></a> | 未指定 | 32 × 19360 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G284<a name="g284"></a> | 未指定 | 32 × 20480 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G285<a name="g285"></a> | 未指定 | 48 × 32 × 6144 | decode | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G286<a name="g286"></a> | 未指定 | 48 × 256 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G287<a name="g287"></a> | 未指定 | 48 × 512 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G288<a name="g288"></a> | 未指定 | 48 × 4096 × 8192 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G289<a name="g289"></a> | 未指定 | 48 × 124160 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G290<a name="g290"></a> | 未指定 | 48 × 154880 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G291<a name="g291"></a> | P0 | 64 × 64 × 2048 | decode | 1 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G292<a name="g292"></a> | P0 | 64 × 256 × 2048 | decode | 2 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G293<a name="g293"></a> | 未指定 | 64 × 16160 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G294<a name="g294"></a> | 未指定 | 64 × 31040 × 4096 | prefill | 2 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G295<a name="g295"></a> | P0 | 64 × 248320 × 2048 | decode,prefill | 3 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G296<a name="g296"></a> | 未指定 | 80 × 129280 × 7168 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G297<a name="g297"></a> | 未指定 | 96 × 64 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G298<a name="g298"></a> | 未指定 | 96 × 512 × 7168 | decode | 6 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G299<a name="g299"></a> | 未指定 | 96 × 7168 × 21504 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G300<a name="g300"></a> | 未指定 | 96 × 129280 × 7168 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G301<a name="g301"></a> | 未指定 | 160 × 129280 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G302<a name="g302"></a> | 未指定 | 192 × 64 × 4096 | decode | 3 | [C014](#c014) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G303<a name="g303"></a> | 未指定 | 192 × 512 × 4096 | decode | 5 | [C014](#c014) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G304<a name="g304"></a> | 未指定 | 192 × 4096 × 12288 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G305<a name="g305"></a> | 未指定 | 192 × 124160 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G306<a name="g306"></a> | 未指定 | 192 × 129280 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G307<a name="g307"></a> | 未指定 | 240 × 129280 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G308<a name="g308"></a> | 未指定 | 288 × 64 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G309<a name="g309"></a> | 未指定 | 288 × 512 × 4096 | decode | 9 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G310<a name="g310"></a> | 未指定 | 288 × 4096 × 12288 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G311<a name="g311"></a> | 未指定 | 288 × 129280 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G312<a name="g312"></a> | 未指定 | 512 × 192 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G313<a name="g313"></a> | P0 | 2048 × 64 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G314<a name="g314"></a> | P0 | 2048 × 256 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G315<a name="g315"></a> | P0 | 2048 × 1024 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G316<a name="g316"></a> | P0 | 2048 × 2048 × 512 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G317<a name="g317"></a> | P0 | 2048 × 2048 × 4096 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G318<a name="g318"></a> | P0 | 2048 × 9216 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G319<a name="g319"></a> | P0 | 2048 × 12288 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G320<a name="g320"></a> | 未指定 | 2560 × 32 × 6144 | prefill | 4 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G321<a name="g321"></a> | P0 | 2560 × 64 × 2048 | prefill | 1 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G322<a name="g322"></a> | 未指定 | 2560 × 64 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G323<a name="g323"></a> | P0 | 2560 × 256 × 2048 | prefill | 2 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G324<a name="g324"></a> | 未指定 | 2560 × 256 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G325<a name="g325"></a> | 未指定 | 3584 × 64 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G326<a name="g326"></a> | 未指定 | 3584 × 128 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G327<a name="g327"></a> | 未指定 | 3584 × 512 × 4096 | prefill | 4 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G328<a name="g328"></a> | 未指定 | 4608 × 32 × 6144 | prefill | 4 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G329<a name="g329"></a> | 未指定 | 4608 × 64 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G330<a name="g330"></a> | 未指定 | 4608 × 256 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G331<a name="g331"></a> | 未指定 | 5120 × 64 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G332<a name="g332"></a> | 未指定 | 5120 × 384 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G333<a name="g333"></a> | 未指定 | 5120 × 512 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G334<a name="g334"></a> | 未指定 | 5120 × 2112 × 7168 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G335<a name="g335"></a> | 未指定 | 5120 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G336<a name="g336"></a> | 未指定 | 5120 × 7168 × 256 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G337<a name="g337"></a> | 未指定 | 5120 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G338<a name="g338"></a> | 未指定 | 5120 × 7168 × 8192 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G339<a name="g339"></a> | 未指定 | 5120 × 12288 × 1536 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G340<a name="g340"></a> | 未指定 | 5120 × 16384 × 512 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G341<a name="g341"></a> | 未指定 | 6656 × 32 × 6144 | prefill | 8 | [C009](#c009) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G342<a name="g342"></a> | P0 | 6656 × 64 × 2048 | prefill | 2 | [C019](#c019) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G343<a name="g343"></a> | 未指定 | 6656 × 64 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G344<a name="g344"></a> | 未指定 | 6656 × 64 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G345<a name="g345"></a> | 未指定 | 6656 × 128 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G346<a name="g346"></a> | 未指定 | 6656 × 192 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G347<a name="g347"></a> | P0 | 6656 × 256 × 2048 | prefill | 4 | [C019](#c019) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G348<a name="g348"></a> | 未指定 | 6656 × 256 × 6144 | prefill | 4 | [C009](#c009) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G349<a name="g349"></a> | 未指定 | 6656 × 384 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G350<a name="g350"></a> | 未指定 | 6656 × 512 × 4096 | prefill | 4 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G351<a name="g351"></a> | 未指定 | 6656 × 512 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G352<a name="g352"></a> | P0 | 6656 × 1024 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G353<a name="g353"></a> | P0 | 6656 × 2048 × 512 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G354<a name="g354"></a> | P0 | 6656 × 2048 × 4096 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G355<a name="g355"></a> | 未指定 | 6656 × 2112 × 7168 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G356<a name="g356"></a> | 未指定 | 6656 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G357<a name="g357"></a> | 未指定 | 6656 × 7168 × 256 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G358<a name="g358"></a> | 未指定 | 6656 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G359<a name="g359"></a> | 未指定 | 6656 × 7168 × 8192 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G360<a name="g360"></a> | P0 | 6656 × 9216 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G361<a name="g361"></a> | 未指定 | 6656 × 12288 × 1536 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G362<a name="g362"></a> | P0 | 6656 × 12288 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G363<a name="g363"></a> | 未指定 | 6656 × 16384 × 512 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G364<a name="g364"></a> | 未指定 | 28672 × 16 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G365<a name="g365"></a> | 未指定 | 40960 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G366<a name="g366"></a> | 未指定 | 40960 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G367<a name="g367"></a> | 未指定 | 53248 × 16 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G368<a name="g368"></a> | 未指定 | 53248 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G369<a name="g369"></a> | 未指定 | 53248 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G540<a name="g540"></a> | P0 | 512 × 4096 × 4096 | decode | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G541<a name="g541"></a> | P0 | 512 × 4096 × 12288 | decode | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G542<a name="g542"></a> | P0 | 512 × 6144 × 4096 | decode | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G543<a name="g543"></a> | P0 | 512 × 24576 × 4096 | decode | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G544<a name="g544"></a> | P0 | 512 × 151936 × 4096 | decode/prefill | 3 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G545<a name="g545"></a> | P0 | 2560 × 4096 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G546<a name="g546"></a> | P0 | 2560 × 4096 × 12288 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G547<a name="g547"></a> | P0 | 2560 × 6144 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G548<a name="g548"></a> | P0 | 2560 × 24576 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G549<a name="g549"></a> | P0 | 5684 × 4096 × 4608 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G550<a name="g550"></a> | P0 | 5684 × 4608 × 4608 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G551<a name="g551"></a> | P0 | 6656 × 4096 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G552<a name="g552"></a> | P0 | 6656 × 4096 × 12288 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G553<a name="g553"></a> | P0 | 6656 × 6144 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G554<a name="g554"></a> | P0 | 6656 × 24576 × 4096 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G555<a name="g555"></a> | P0 | 22736 × 1152 × 1152 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G556<a name="g556"></a> | P0 | 22736 × 1152 × 1536 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G557<a name="g557"></a> | P0 | 22736 × 1152 × 4304 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G558<a name="g558"></a> | P0 | 22736 × 3456 × 1152 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |
| G559<a name="g559"></a> | P0 | 22736 × 4304 × 1152 | prefill | 1 | [C020](#c020) | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent] | 待验证（新增需求） |

<a name="table-4"></a>

## 表 4：linear · FP32 × FP32 → FP32

35 行需求，累计 218 条实例。编号在本清单中全局唯一。

本表的接口级 smoke PASS 依赖当前 Triton 的 S03 LEGACY 补丁；不能推广到未带该补丁的环境。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| G370<a name="g370"></a> | 未指定 | 16 × 129280 × 512 | decode | 10 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G371<a name="g371"></a> | 未指定 | 32 × 129280 × 256 | decode | 5 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G372<a name="g372"></a> | 未指定 | 48 × 129280 × 256 | decode | 15 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G373<a name="g373"></a> | 未指定 | 80 × 1 × 7680 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G374<a name="g374"></a> | 未指定 | 80 × 4 × 28672 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G375<a name="g375"></a> | 未指定 | 80 × 384 × 7168 | decode | 6 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G376<a name="g376"></a> | 未指定 | 96 × 4 × 28672 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G377<a name="g377"></a> | 未指定 | 96 × 384 × 7168 | decode | 8 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G378<a name="g378"></a> | 未指定 | 160 × 1 × 4352 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G379<a name="g379"></a> | 未指定 | 160 × 4 × 16384 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G380<a name="g380"></a> | 未指定 | 160 × 256 × 4096 | decode | 3 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G381<a name="g381"></a> | 未指定 | 192 × 4 × 16384 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G382<a name="g382"></a> | 未指定 | 192 × 256 × 4096 | decode | 4 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G383<a name="g383"></a> | 未指定 | 240 × 1 × 4352 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G384<a name="g384"></a> | 未指定 | 240 × 4 × 16384 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G385<a name="g385"></a> | 未指定 | 240 × 256 × 4096 | decode | 9 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G386<a name="g386"></a> | 未指定 | 288 × 4 × 16384 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G387<a name="g387"></a> | 未指定 | 288 × 256 × 4096 | decode | 12 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G388<a name="g388"></a> | 未指定 | 2560 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G389<a name="g389"></a> | 未指定 | 2560 × 384 × 7168 | prefill | 16 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G390<a name="g390"></a> | 未指定 | 3584 × 4 × 16384 | prefill | 2 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G391<a name="g391"></a> | 未指定 | 3584 × 256 × 4096 | prefill | 16 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G392<a name="g392"></a> | 未指定 | 4608 × 384 × 7168 | prefill | 4 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G393<a name="g393"></a> | 未指定 | 5120 × 4 × 16384 | prefill | 2 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G394<a name="g394"></a> | 未指定 | 5120 × 256 × 4096 | prefill | 16 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G395<a name="g395"></a> | 未指定 | 6656 × 4 × 16384 | prefill | 4 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G396<a name="g396"></a> | 未指定 | 6656 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G397<a name="g397"></a> | 未指定 | 6656 × 256 × 4096 | prefill | 32 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G398<a name="g398"></a> | 未指定 | 6656 × 384 × 7168 | prefill | 20 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G399<a name="g399"></a> | 未指定 | 20480 × 4 × 28672 | prefill | 1 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G400<a name="g400"></a> | 未指定 | 28672 × 4 × 16384 | prefill | 2 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G401<a name="g401"></a> | 未指定 | 36864 × 4 × 28672 | prefill | 1 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G402<a name="g402"></a> | 未指定 | 40960 × 4 × 16384 | prefill | 2 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G403<a name="g403"></a> | 未指定 | 53248 × 4 × 16384 | prefill | 4 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G404<a name="g404"></a> | 未指定 | 53248 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |

<a name="table-5"></a>

## 表 5：linear · BF16 × BF16 → FP32

27 行需求，累计 172 条实例。编号在本清单中全局唯一。

本表使用 `gemm_a16w16(..., dtype=torch.float32)`，保留 BF16 输入、FP32 输出语义。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| G405<a name="g405"></a> | 未指定 | 96 × 512 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G406<a name="g406"></a> | 未指定 | 96 × 1024 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G407<a name="g407"></a> | 未指定 | 96 × 2048 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G408<a name="g408"></a> | 未指定 | 192 × 512 × 4096 | decode | 2 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G409<a name="g409"></a> | 未指定 | 192 × 1024 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G410<a name="g410"></a> | 未指定 | 192 × 2048 × 4096 | decode | 2 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G411<a name="g411"></a> | 未指定 | 288 × 512 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G412<a name="g412"></a> | 未指定 | 288 × 1024 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G413<a name="g413"></a> | 未指定 | 288 × 2048 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G414<a name="g414"></a> | 未指定 | 2560 × 512 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G415<a name="g415"></a> | 未指定 | 2560 × 1024 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G416<a name="g416"></a> | 未指定 | 2560 × 2048 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G417<a name="g417"></a> | 未指定 | 3584 × 512 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G418<a name="g418"></a> | 未指定 | 3584 × 1024 × 4096 | prefill | 4 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G419<a name="g419"></a> | 未指定 | 3584 × 2048 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G420<a name="g420"></a> | 未指定 | 4608 × 512 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G421<a name="g421"></a> | 未指定 | 4608 × 1024 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G422<a name="g422"></a> | 未指定 | 4608 × 2048 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G423<a name="g423"></a> | 未指定 | 5120 × 512 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G424<a name="g424"></a> | 未指定 | 5120 × 1024 × 4096 | prefill | 4 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G425<a name="g425"></a> | 未指定 | 5120 × 2048 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G426<a name="g426"></a> | 未指定 | 6656 × 512 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G427<a name="g427"></a> | 未指定 | 6656 × 512 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G428<a name="g428"></a> | 未指定 | 6656 × 1024 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G429<a name="g429"></a> | 未指定 | 6656 × 1024 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G430<a name="g430"></a> | 未指定 | 6656 × 2048 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G431<a name="g431"></a> | 未指定 | 6656 × 2048 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |

<a name="table-6"></a>

## 表 6：linear · BF16 × INT4 → BF16

4 行需求、4 条来源，G432–G435 仍为 Kimi EAGLE3 draft。新版现已明确 weight_quant=`int4 g32 bf16*`，因此采用 K32 / BF16* scale；A 为 BF16，无 A scale。星号含义、zero-point、对称性与 packing 仍未给出，不由 Kimi 主模型或 AWQ 的格式反推。AWQ 链接只保留历史接口参考，不证明 draft 的实际格式相容。

| 编号 | 优先级 | M × N × K | 阶段 | 频次 | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G432<a name="g432"></a> | 未指定 | 4 × 7168 × 8192 | decode | 1 | BF16*@1×32（Excel）；packing/zero-point 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G433<a name="g433"></a> | 未指定 | 4 × 7168 × 12288 | decode | 1 | BF16*@1×32（Excel）；packing/zero-point 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G434<a name="g434"></a> | 未指定 | 4 × 24576 × 7168 | decode | 1 | BF16*@1×32（Excel）；packing/zero-point 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G435<a name="g435"></a> | 未指定 | 4 × 24576 × 14336 | decode | 1 | BF16*@1×32（Excel）；packing/zero-point 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |

<a name="table-8"></a>

## 表 8：grouped_gemm · FP8E4M3 × FP4E2M1 → BF16（兼容保留）

按用户要求保留原 34 行、G472–G505、MNK/E、阶段及频次。新版 Excel 的对应 304 条来源实际已改为 FP4×FP4；本表是额外保留的 FP8×FP4 支持需求，不表示新版仍要求 A=FP8。

scale 按新版更新：A/B 均为 E8M0/K32，激活动态；DeepSeek 原 A 的 E8M0/FP32 K128、Q35 down 的 FP32/K32 均不再作为本次 scale 要求。新版 act_quant 的 fp4 数据 dtype 只用于表 9；本表有意保留 A=FP8，仅采用对应新版的 scale 粒度/dtype/动态性。

历史 grouped_gemm_a8wfp4 接口 smoke 与原 34/34 编译证据保留，但不验证本次更新的 scale 组合，完整 MNK/E/scale 的正确性和性能仍待逐项验证。来源次数 304 与表 9 共用谱系，不计作新增 Excel 行。

| 编号 | 优先级 | M × N × K | 分组数 E | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G472<a name="g472"></a> | P0 | 2 × 1024 × 2048 | 80 | decode | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G473<a name="g473"></a> | P0 | 2 × 2048 × 512 | 80 | decode | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G474<a name="g474"></a> | 未指定 | 20 × 6144 × 7168 | 24 | decode | 6 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G475<a name="g475"></a> | 未指定 | 20 × 7168 × 3072 | 24 | decode | 6 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G476<a name="g476"></a> | 未指定 | 24 × 6144 × 7168 | 24 | decode | 8 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G477<a name="g477"></a> | 未指定 | 24 × 7168 × 3072 | 24 | decode | 8 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G478<a name="g478"></a> | 未指定 | 30 × 4096 × 2048 | 32 | decode | 3 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G479<a name="g479"></a> | 未指定 | 30 × 4096 × 4096 | 32 | decode | 3 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G480<a name="g480"></a> | 未指定 | 36 × 4096 × 2048 | 32 | decode | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G481<a name="g481"></a> | 未指定 | 36 × 4096 × 4096 | 32 | decode | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G482<a name="g482"></a> | 未指定 | 45 × 4096 × 2048 | 32 | decode | 9 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G483<a name="g483"></a> | 未指定 | 45 × 4096 × 4096 | 32 | decode | 9 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G484<a name="g484"></a> | 未指定 | 54 × 4096 × 2048 | 32 | decode | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G485<a name="g485"></a> | 未指定 | 54 × 4096 × 4096 | 32 | decode | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G486<a name="g486"></a> | P0 | 64 × 1024 × 2048 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G487<a name="g487"></a> | P0 | 64 × 2048 × 512 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G488<a name="g488"></a> | 未指定 | 160 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G489<a name="g489"></a> | 未指定 | 160 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G490<a name="g490"></a> | P0 | 208 × 1024 × 2048 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G491<a name="g491"></a> | P0 | 208 × 2048 × 512 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G492<a name="g492"></a> | 未指定 | 320 × 6144 × 7168 | 48 | prefill | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G493<a name="g493"></a> | 未指定 | 320 × 7168 × 3072 | 48 | prefill | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G494<a name="g494"></a> | 未指定 | 416 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G495<a name="g495"></a> | 未指定 | 416 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G496<a name="g496"></a> | 未指定 | 576 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G497<a name="g497"></a> | 未指定 | 576 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G498<a name="g498"></a> | 未指定 | 672 × 4096 × 2048 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C007](#c007) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G499<a name="g499"></a> | 未指定 | 672 × 4096 × 4096 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C007](#c007) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G500<a name="g500"></a> | 未指定 | 832 × 6144 × 7168 | 48 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G501<a name="g501"></a> | 未指定 | 832 × 7168 × 3072 | 48 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G502<a name="g502"></a> | 未指定 | 960 × 4096 × 2048 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C008](#c008) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G503<a name="g503"></a> | 未指定 | 960 × 4096 × 4096 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C008](#c008) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G504<a name="g504"></a> | 未指定 | 1248 × 4096 × 2048 | 32 | prefill | 32 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C016](#c016) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G505<a name="g505"></a> | 未指定 | 1248 × 4096 × 4096 | 32 | prefill | 32 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C016](#c016) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |

<a name="table-9"></a>

## 表 9：grouped_gemm · FP4E2M1 × FP4E2M1 → BF16（新版新增）

34 行、304 条新版 Excel 来源，编号 G506–G539。各行与表 8 按相同 MNK/E/来源顺序对应，具体成对 G 编号见本次编号映射的 paired_legacy_G。A/B scale 均为 E8M0/K32，A 为动态量化。

当前仓库的 gemm_afp4wfp4 是 dense FP4×FP4 参考，不能据此宣称已有等长/ragged grouped 入口或本需求 PASS。所有新增行均待 grouped 接口适配与验证。FP4 packing/nibble 顺序、scale 实际布局、offsets/空组规则须由调用数据明确；此处只记录逻辑需求。

| 编号 | 优先级 | M × N × K | 分组数 E | 阶段 | 频次 | 激活 scale（新版 Excel） | 权重 scale（新版 Excel） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G506<a name="g506"></a> | P0 | 2 × 1024 × 2048 | 80 | decode | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G507<a name="g507"></a> | P0 | 2 × 2048 × 512 | 80 | decode | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G508<a name="g508"></a> | 未指定 | 20 × 6144 × 7168 | 24 | decode | 6 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G509<a name="g509"></a> | 未指定 | 20 × 7168 × 3072 | 24 | decode | 6 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G510<a name="g510"></a> | 未指定 | 24 × 6144 × 7168 | 24 | decode | 8 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G511<a name="g511"></a> | 未指定 | 24 × 7168 × 3072 | 24 | decode | 8 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G512<a name="g512"></a> | 未指定 | 30 × 4096 × 2048 | 32 | decode | 3 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G513<a name="g513"></a> | 未指定 | 30 × 4096 × 4096 | 32 | decode | 3 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G514<a name="g514"></a> | 未指定 | 36 × 4096 × 2048 | 32 | decode | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G515<a name="g515"></a> | 未指定 | 36 × 4096 × 4096 | 32 | decode | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C005](#c005) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G516<a name="g516"></a> | 未指定 | 45 × 4096 × 2048 | 32 | decode | 9 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G517<a name="g517"></a> | 未指定 | 45 × 4096 × 4096 | 32 | decode | 9 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G518<a name="g518"></a> | 未指定 | 54 × 4096 × 2048 | 32 | decode | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G519<a name="g519"></a> | 未指定 | 54 × 4096 × 4096 | 32 | decode | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C006](#c006) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G520<a name="g520"></a> | P0 | 64 × 1024 × 2048 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G521<a name="g521"></a> | P0 | 64 × 2048 × 512 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G522<a name="g522"></a> | 未指定 | 160 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G523<a name="g523"></a> | 未指定 | 160 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G524<a name="g524"></a> | P0 | 208 × 1024 × 2048 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G525<a name="g525"></a> | P0 | 208 × 2048 × 512 | 256 | prefill | 2 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C018](#c018) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G526<a name="g526"></a> | 未指定 | 320 × 6144 × 7168 | 48 | prefill | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G527<a name="g527"></a> | 未指定 | 320 × 7168 × 3072 | 48 | prefill | 12 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G528<a name="g528"></a> | 未指定 | 416 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G529<a name="g529"></a> | 未指定 | 416 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G530<a name="g530"></a> | 未指定 | 576 × 6144 × 7168 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G531<a name="g531"></a> | 未指定 | 576 × 7168 × 3072 | 48 | prefill | 4 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G532<a name="g532"></a> | 未指定 | 672 × 4096 × 2048 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C007](#c007) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G533<a name="g533"></a> | 未指定 | 672 × 4096 × 4096 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C007](#c007) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G534<a name="g534"></a> | 未指定 | 832 × 6144 × 7168 | 48 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G535<a name="g535"></a> | 未指定 | 832 × 7168 × 3072 | 48 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C004](#c004) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G536<a name="g536"></a> | 未指定 | 960 × 4096 × 2048 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C008](#c008) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G537<a name="g537"></a> | 未指定 | 960 × 4096 × 4096 | 32 | prefill | 16 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C008](#c008) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G538<a name="g538"></a> | 未指定 | 1248 × 4096 × 2048 | 32 | prefill | 32 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C016](#c016) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |
| G539<a name="g539"></a> | 未指定 | 1248 × 4096 × 4096 | 32 | prefill | 32 | E8M0@1×32（动态） | E8M0@1×32（Excel） | [C016](#c016) | 待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense] | 待 grouped 适配与验证 |

<a name="requirements-cross-check"></a>

## 新版 Excel 与历史公开补充核对（2026-11-04）

新版 act_quant / weight_quant 显式标签是本次需求依据。下表的公开信息沿用 2026-10-23 审计中固定 revision 的快照，仅说明历史补充与新版的关系；本次没有重新检索公开 main，也不将候选 checkpoint 等同于 BU 实际部署。历史证据见 [2026-10-23 核对记录](debug_tuning/requirements_audit_20261023/REPORT.md)。

| 模型 / 历史来源 | 新版激活量化 / scale | 新版权重量化 / scale | 与历史公开补充的关系 |
| --- | --- | --- | --- |
| <a name="quant-flash"></a>[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) | dense：FP8、FP32/K128/dyn；grouped：FP4、E8M0/K32/dyn | dense：FP8、E8M0/N128/K128；grouped：FP4、E8M0/K32 | B 与旧公开快照相符；dense A E8M0→FP32、grouped A FP8→FP4 是 Excel 修订。 |
| <a name="quant-pro"></a>[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) | 同 Flash | 同 Flash | 新版 grouped FP4×FP4 列表 9；FP8×FP4 作为兼容需求保留表 8。 |
| <a name="quant-glm"></a>[GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8/blob/f396cf805182f4ca10fa675e1a99815b3ca384db/config.json) | FP8、FP32/K128/dyn，含 indexer_wk | FP8、FP32*/N128/K128 | K/N 粒度与历史补充相符；星号仍未释义，不能当作已确认的无星号存储 dtype。 |
| <a name="quant-q35"></a>[Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8/blob/9d1823d2dee688a6b25e77009dc727688c44936e/config.json) | FP8、FP32/K128/dyn，含 o_proj | FP8、BF16/N128/K128 | 新版已明确这些字段，原“网补”改为 Excel 来源；checkpoint 与框架转换后的 scale dtype 仍需分开。 |
| <a name="quant-q397"></a>[Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/blob/ea5b4f81096f3901c91dea97f81324302495781d/config.json) | 同 Q35 FP8 | FP8、BF16/N128/K128 | 同一 G 混合 DS/Qwen 时，B dtype 按 X 来源保留两种组合。 |
| <a name="quant-hy3"></a>[Hy3-FP8 / Hunyuan-V3](https://huggingface.co/tencent/Hy3-FP8/blob/ecc1d8e194e093f33177f2f0ef7ce8f397b2d68b/config.json) | FP8、FP32/token/dyn，含 o_proj | FP8、FP32*/channel | 替换旧候选的 BF16 scalar 权重 / FP32 scalar 静态激活；粒度与动态性变化不能只用 scale cast 解释。 |
| <a name="quant-mx4"></a>[Qwen3.5-35B-A3B-MXFP4](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/blob/2e19c6576db91e5d5a93455415619262218bf8a1/config.json) | grouped：FP4、E8M0/K32/dyn | grouped：FP4、E8M0/K32 | 新版 A FP4 与历史 AMD 候选方向一致；表 8 的 A FP8 是额外兼容需求。 |
| <a name="quant-kimi"></a>[Kimi-K2.6 / W4A16](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json) | BF16，无 A scale | INT4、BF16*/K32，含 EAGLE3 draft | 新版补齐 draft 的 g32/BF16*；对称性、zero-point 与 packing 未由标签确定，不直接认定为 AWQ。 |

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
| 3 × 1024 × 2048<a name="gap-001"></a> | P0 | 221 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 3 × 2048 × 512<a name="gap-002"></a> | P0 | 221 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 3 × 3072 × 4096<a name="gap-003"></a> | 未指定 | 11 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 1 |
| 3 × 4096 × 1536<a name="gap-004"></a> | 未指定 | 11 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 1 |
| 4 × 2048 × 4096<a name="gap-005"></a> | 未指定 | 63 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 1 |
| 4 × 4096 × 1024<a name="gap-006"></a> | 未指定 | 63 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 1 |
| 4 × 4096 × 6144<a name="gap-007"></a> | 未指定 | 15 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 1 |
| 4 × 6144 × 2048<a name="gap-008"></a> | 未指定 | 15 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 1 |
| 6 × 4096 × 6144<a name="gap-009"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 1 |
| 6 × 4096 × 7168<a name="gap-010"></a> | 未指定 | 24 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 1 |
| 6 × 6144 × 2048<a name="gap-011"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 1 |
| 6 × 7168 × 2048<a name="gap-012"></a> | 未指定 | 24 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 1 |
| 11 × 3072 × 4096<a name="gap-013"></a> | 未指定 | 12 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 1 |
| 11 × 4096 × 1536<a name="gap-014"></a> | 未指定 | 12 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 1 |
| 12 × 4096 × 6144<a name="gap-015"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 1 |
| 12 × 6144 × 2048<a name="gap-016"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 1 |
| 15 × 2048 × 4096<a name="gap-017"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 2 |
| 15 × 4096 × 1024<a name="gap-018"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 2 |
| 24 × 4096 × 6144<a name="gap-019"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 1 |
| 24 × 6144 × 2048<a name="gap-020"></a> | 未指定 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 1 |
| 80 × 1024 × 2048<a name="gap-021"></a> | P0 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 80 × 2048 × 512<a name="gap-022"></a> | P0 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 171 × 3072 × 4096<a name="gap-023"></a> | 未指定 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 2 |
| 171 × 4096 × 1536<a name="gap-024"></a> | 未指定 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 2 |
| 208 × 1024 × 2048<a name="gap-025"></a> | P0 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 208 × 2048 × 512<a name="gap-026"></a> | P0 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X09 | 2 |
| 560 × 2048 × 4096<a name="gap-027"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 4 |
| 560 × 4096 × 1024<a name="gap-028"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 4 |
| 640 × 4096 × 6144<a name="gap-029"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 2 |
| 640 × 6144 × 2048<a name="gap-030"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X08 | 2 |
| 854 × 4096 × 7168<a name="gap-031"></a> | 未指定 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 2 |
| 854 × 7168 × 2048<a name="gap-032"></a> | 未指定 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 2 |
| 1040 × 2048 × 4096<a name="gap-033"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 4 |
| 1040 × 4096 × 1024<a name="gap-034"></a> | 未指定 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | BF16@128×128（Excel） | X13 | 4 |
| 1110 × 4096 × 7168<a name="gap-035"></a> | 未指定 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 2 |
| 1110 × 7168 × 2048<a name="gap-036"></a> | 未指定 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | BF16*@1×32（Excel） | X11 | 2 |
| 1152 × 4096 × 6144<a name="gap-037"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 2 |
| 1152 × 6144 × 2048<a name="gap-038"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07 | 2 |
| 1664 × 4096 × 6144<a name="gap-039"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07/X08 | 4 |
| 1664 × 6144 × 2048<a name="gap-040"></a> | 未指定 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128（动态） | FP32*@128×128（Excel） | X07/X08 | 4 |
| 2219 × 3072 × 4096<a name="gap-041"></a> | 未指定 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 2 |
| 2219 × 4096 × 1536<a name="gap-042"></a> | 未指定 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×K（逐 token 动态） | FP32*@channel（Excel） | X12 | 2 |

<a name="model-config-index"></a>

## 模型与配置索引

以下编号对应主表的「模型 / 配置」列。同一完整组合共用一个编号；原始模型名称、精度和方括号内的配置原样保留。C004 的 Pro a/b 与修订后的 PP stage 以新版来源清单为准；“FP8”等模型标签不覆盖行内明确的混合 dtype。

<a name="c001"></a>

### C001

GLM-5.1/FP8[128k/1k/s2000]

<a name="c002"></a>

### C002

GLM-5.1/FP8[64k/1k/s2000]

<a name="c003"></a>

### C003

Kimi-2.6/W4A16[128k/1k/s2000]

<a name="c004"></a>

### C004

DeepSeek-V4-Pro/FP8[128k/1k/s2000]

<a name="c005"></a>

### C005

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150]

<a name="c006"></a>

### C006

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s2000,64k/1k/s150,64k/1k/s2000]

<a name="c007"></a>

### C007

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000]

<a name="c008"></a>

### C008

DeepSeek-V4-Flash/FP4(MXFP4)[64k/1k/s150,64k/1k/s2000]

<a name="c009"></a>

### C009

GLM-5.1/FP8[128k/1k/s2000,64k/1k/s2000]

<a name="c010"></a>

### C010

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000]; DeepSeek-V4-Pro/FP8[128k/1k/s2000]

<a name="c011"></a>

### C011

Hunyuan-V3/FP8[128k/1k/s2000]

<a name="c012"></a>

### C012

Qwen3.5-397B-A17B/FP8[128k/1k/s2000]

<a name="c013"></a>

### C013

优先级：P0。

Qwen3.5-35B-A3B/FP8[4k/1k/s150]

<a name="c014"></a>

### C014

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000]

<a name="c015"></a>

### C015

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000]

<a name="c016"></a>

### C016

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000]

<a name="c017"></a>

### C017

DeepSeek-V4-Flash/FP4(MXFP4)[128k/1k/s150,128k/1k/s2000,64k/1k/s150,64k/1k/s2000]; Hunyuan-V3/FP8[128k/1k/s2000]; Qwen3.5-397B-A17B/FP8[128k/1k/s2000]

<a name="c018"></a>

### C018

优先级：P0。

Qwen3.5-35B-A3B/MXFP4[128k/1k/s150]

<a name="c019"></a>

### C019

优先级：P0。

Qwen3.5-35B-A3B/FP8[4k/1k/s150]; Qwen3.5-35B-A3B/MXFP4[128k/1k/s150]

<a name="c020"></a>

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
