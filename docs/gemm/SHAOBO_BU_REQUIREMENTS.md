# 少伯 BU GEMM 支持需求

原编号来源：`/xukang/GEMM支持清单.html`（生成日期 2026-10-12）。2026-10-23 复核实际原始目录 `/mnt/c/Users/xukang/Downloads/shaobob0_shapes/`（用户称 `shaobo_shapes`），共 13 份 XLSX 和 1 份 ResNet DOCX。现有 7 张编号表保留 469 行需求和 2424 条 Excel 来源记录，表号 1–6、8 及 G 编号不重排；另补列 42 个尚未编号的 grouped shape 作为覆盖缺口，见文末。

原 HTML SHA256：`187673895b820bfb89d6b2890ffb35fa378a183e8922238a8f96df2ff738d11c`。

本清单用于逐项记录支持和优化进度。接口验证状态截至 2026-10-20；需求与公开配置核对日期为 2026-10-23。**“接口 smoke 通过”表示该算子族 / dtype 接口已有小尺寸 PASS，不表示本行完整 M/N/K、batch/分组数、scale 和分组寻址已逐项运行，也不表示性能已调优。** 实验详情与记录规范见 [调优记录](debug_tuning/TUNING.md)。

## 记录口径

- M/N/K 沿用 HTML 中的激活行数、输出通道数和规约维，不据此推断矩阵 stride。
- batch / 分组数 E 的有效值沿用源数据；bmm 表使用 batch 列，grouped_gemm 表将原 experts 列记为分组数 E，linear 表原本全空的该列省略。M/N/K 原值不变，不以 M×E 替换 M，也不据来源 M 假定各组等长；ragged 各组 M 与 offsets 须由调用方提供。
- 输出精度指写回显存的精度，与累加精度分开验证。
- 有量化的表分别记录 Excel 激活 scale 与公开 checkpoint 的权重 scale；缺失值记“未给出”，不能从 FP8/MXFP4 总称或“离线”推断 dtype。公开补充项未与 BU checkpoint/框架 commit/启动参数绑定前，不升级为 BU 已确认配置。
- “频次 / 实例数”是 Excel 算子记录出现次数，不是 `layer_count` 展开后的 kernel launch 次数；例如 G171 频次为 1，其来源 `layer_count=3`，两者不冲突。
- 状态分为接口级「接口 smoke 通过」「待 grouped GEMM smoke」「接口 smoke 受阻」与需求尺寸级「待验证」「编译通过」「完整尺寸正确性通过」「已调优」；不得把接口 smoke 或编译通过升级为本行完整尺寸 PASS。
- BoltOps 函数 / 实现列链接实际源码定义；它表示当前可对接的接口，不替代模型布局、量化格式与分组寻址适配。表 8 只关注单次 grouped GEMM，输入量化值、scale 和分组索引由测试预先准备；不把在线量化、路由生成、激活或结果合并纳入验收。
- ResNet50 INT8 隐式 GEMM 卷积 24 类和末端 FC 不在这 469 行中，仍以原 HTML 及当前支持说明为准。

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
| [表 8](#table-8) | grouped_gemm | FP8E4M3 × FP4E2M1 → BF16 | 34 | 304 |

<a id="table-1"></a>

## 表 1：bmm · BF16 × BF16 → BF16

29 行需求，累计 194 条实例。编号在本清单中全局唯一。

本表仅记录 BF16 BMM；函数使用 `B[batch,N,K]` 权重布局，模型调用需匹配 stride。

| 编号 | M × N × K | batch | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| G001 | 6 × 256 × 512 | 64 | decode | 1 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G002 | 6 × 512 × 192 | 64 | decode | 1 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G003 | 12 × 256 × 512 | 64 | decode | 1 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G004 | 12 × 512 × 192 | 64 | decode | 1 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G005 | 16 × 128 × 512 | 64 | decode | 2 | [C003](#c003) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G006 | 16 × 512 × 128 | 64 | decode | 2 | [C003](#c003) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G007 | 24 × 256 × 512 | 64 | decode | 2 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G008 | 24 × 512 × 192 | 64 | decode | 2 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G009 | 48 × 256 × 512 | 64 | decode | 2 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G010 | 48 × 512 × 192 | 64 | decode | 2 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G011 | 80 × 1024 × 4096 | 16 | decode | 6 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G012 | 96 × 1024 × 4096 | 16 | decode | 8 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G013 | 160 × 1024 × 4096 | 8 | decode | 3 | [C005](#c005) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G014 | 192 × 1024 × 4096 | 8 | decode | 4 | [C005](#c005) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G015 | 240 × 1024 × 4096 | 8 | decode | 9 | [C006](#c006) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G016 | 288 × 1024 × 4096 | 8 | decode | 12 | [C006](#c006) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G017 | 2560 × 256 × 512 | 64 | prefill | 4 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G018 | 2560 × 512 × 192 | 64 | prefill | 4 | [C002](#c002) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G019 | 2560 × 1024 × 4096 | 16 | prefill | 12 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G020 | 2560 × 1024 × 4096 | 8 | prefill | 4 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G021 | 3584 × 1024 × 4096 | 8 | prefill | 16 | [C007](#c007) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G022 | 4608 × 256 × 512 | 64 | prefill | 4 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G023 | 4608 × 512 × 192 | 64 | prefill | 4 | [C001](#c001) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G024 | 4608 × 1024 × 4096 | 16 | prefill | 4 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G025 | 5120 × 1024 × 4096 | 8 | prefill | 16 | [C008](#c008) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G026 | 6656 × 256 × 512 | 64 | prefill | 8 | [C009](#c009) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G027 | 6656 × 512 × 192 | 64 | prefill | 8 | [C009](#c009) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G028 | 6656 × 1024 × 4096 | 8 | prefill | 36 | [C010](#c010) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |
| G029 | 6656 × 1024 × 4096 | 16 | prefill | 16 | [C004](#c004) | [batched_gemm_bf16][fn-bmm] | 接口 smoke 通过 |

<a id="table-2"></a>

## 表 2：linear · FP8E4M3 × FP8E4M3 → BF16

207 行需求，累计 1206 条 Excel 记录；G 编号不变。

A scale 优先由同一 sheet/stage/module、形状一致的前序量化输出 dtype 和 scale shape 推导；没有对应输出的项记为“未给出”；`indexer_wk` 只能找到同形状 `quantize_qkv_a` 输出，仍记为候选、连线待确认，不把形状匹配升级为确定需求。量化输出与 GEMM 未给出 tensor ID，来源关联及原始单元格见 [逐行映射](debug_tuning/requirements_audit_20261023/case-provenance.csv)。

**B scale 列新增的是对应公开 checkpoint 的信息，不代表 BU 使用了该 checkpoint 或未经转换的布局。** Excel 的 linear 输入没有列出 B scale；旧版全写“离线 1×128”缺少依据，现撤销。checkpoint dtype、框架加载后的 dtype、BU 实际 kernel 输入须分别确认。N/K 二维 block 统一按逻辑权重 `[N,K]` 解释，与 Excel 的 `[K,N]` 显示顺序及 kernel BM/BN/BK 区分。

前两个接口源码支持按 N/K 分块的 B scale，但此前性能实验常用 `GROUP_N=1`。补全需求不代表 128×128 scale、checkpoint BF16 scale 或 Hy3 scalar 路径已新增 smoke PASS；状态列保留旧接口证据，不升级完整组合验收。模型差异与冲突见 [本次核对](#requirements-cross-check)。

### 按 A scale dtype 与 block size 分组

B scale 必须再按模型细分，不能由本组 A dtype 推断。整行 `1×K` 的具体 K 值保留在行内。明确记录 `1×128` 的条目即使 K=128 也仍留在 K128 组。

| 分组 | A scale dtype | A scale block size | B scale | 需求行数 | Excel 记录数 |
| --- | --- | --- | --- | ---: | ---: |
| [2.1](#table-2-scale-1) | FP32 | 1×128 | 按模型，见各行 / 核对表 | 90 | 186 |
| [2.2](#table-2-scale-2) | E8M0 | 1×128 | 按模型，见各行 / 核对表 | 79 | 888 |
| [2.3](#table-2-scale-3) | FP32 | 1×K | 按模型，见各行 / 核对表 | 22 | 33 |
| [2.4](#table-2-scale-4) | 候选 / 未给出 | 候选 K128 / 未给出 | 按模型，见各行 / 核对表 | 10 | 29 |
| [2.5](#table-2-scale-5) | 混合 | 逐来源确认 | 按模型，见各行 / 核对表 | 6 | 70 |

<a id="table-2-scale-1"></a>

### 2.1 A scale FP32，block size 1×128

90 行需求，累计 186 条 Excel 记录。

| 编号 | M × N × K | 阶段 | 频次 | 激活 scale（Excel） | 权重 scale（公开 checkpoint 补充） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| G035 | 6 × 2624 × 6144 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G036 | 6 × 4096 × 2048 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G037 | 6 × 4096 × 6144 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G038 | 6 × 6144 × 2048 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G039 | 6 × 6144 × 16384 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G040 | 6 × 16384 × 2048 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G042 | 12 × 2624 × 6144 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G043 | 12 × 4096 × 2048 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G044 | 12 × 4096 × 6144 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G045 | 12 × 6144 × 2048 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G046 | 12 × 6144 × 16384 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G047 | 12 × 16384 × 2048 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G055 | 24 × 2624 × 6144 | decode | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G056 | 24 × 4096 × 2048 | decode | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G057 | 24 × 4096 × 6144 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G058 | 24 × 6144 × 2048 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G059 | 24 × 6144 × 12288 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G060 | 24 × 6144 × 16384 | decode | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G061 | 24 × 16384 × 2048 | decode | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G062 | 24 × 24576 × 6144 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G064 | 48 × 2048 × 4096 | decode | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G065 | 48 × 2624 × 6144 | decode | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G066 | 48 × 4096 × 1024 | decode | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G067 | 48 × 4096 × 2048 | decode | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G068 | 48 × 4096 × 4096 | decode | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G069 | 48 × 4096 × 6144 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G070 | 48 × 6144 × 2048 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G071 | 48 × 6144 × 12288 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G072 | 48 × 6144 × 16384 | decode | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G073 | 48 × 8704 × 4096 | decode | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G074 | 48 × 16384 × 2048 | decode | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G075 | 48 × 24576 × 6144 | decode | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G076 | 64 × 1024 × 2048 | decode | 2 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G077 | 64 × 2048 × 512 | decode | 2 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G079 | 64 × 9216 × 2048 | decode | 1 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G080 | 64 × 12288 × 2048 | decode | 1 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G102 | 192 × 2048 × 4096 | decode | 2 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G103 | 192 × 4096 × 1024 | decode | 2 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G108 | 192 × 8704 × 4096 | decode | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G109 | 192 × 10240 × 4096 | decode | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G131 | 2560 × 512 × 6144 | prefill | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G134 | 2560 × 1024 × 2048 | prefill | 2 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G136 | 2560 × 2048 × 512 | prefill | 2 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G138 | 2560 × 2624 × 6144 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G139 | 2560 × 3072 × 6144 | prefill | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G140 | 2560 × 4096 × 2048 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G141 | 2560 × 6144 × 256 | prefill | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G142 | 2560 × 6144 × 1536 | prefill | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G143 | 2560 × 6144 × 16384 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G148 | 2560 × 9216 × 2048 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G149 | 2560 × 12288 × 2048 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G150 | 2560 × 16384 × 2048 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G153 | 3584 × 256 × 4096 | prefill | 4 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G156 | 3584 × 4096 × 128 | prefill | 4 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G160 | 3584 × 17408 × 4096 | prefill | 2 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G161 | 3584 × 20480 × 4096 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G166 | 4608 × 512 × 6144 | prefill | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G170 | 4608 × 2624 × 6144 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G171 | 4608 × 3072 × 6144 | prefill | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G172 | 4608 × 4096 × 2048 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G173 | 4608 × 6144 × 256 | prefill | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G174 | 4608 × 6144 × 1536 | prefill | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G175 | 4608 × 6144 × 16384 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G179 | 4608 × 16384 × 2048 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G188 | 6656 × 256 × 4096 | prefill | 4 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G191 | 6656 × 512 × 6144 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G194 | 6656 × 1024 × 2048 | prefill | 2 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G197 | 6656 × 2048 × 512 | prefill | 2 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G199 | 6656 × 2624 × 6144 | prefill | 8 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G200 | 6656 × 3072 × 6144 | prefill | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G202 | 6656 × 4096 × 128 | prefill | 4 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G206 | 6656 × 4096 × 2048 | prefill | 8 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G208 | 6656 × 6144 × 256 | prefill | 4 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G209 | 6656 × 6144 × 1536 | prefill | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G210 | 6656 × 6144 × 16384 | prefill | 8 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G216 | 6656 × 9216 × 2048 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G218 | 6656 × 12288 × 2048 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G219 | 6656 × 16384 × 2048 | prefill | 8 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G220 | 6656 × 17408 × 4096 | prefill | 2 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G221 | 6656 × 20480 × 4096 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G225 | 20480 × 3072 × 6144 | prefill | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G226 | 20480 × 6144 × 1536 | prefill | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G227 | 28672 × 2560 × 4096 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G228 | 28672 × 4096 × 1024 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G229 | 36864 × 3072 × 6144 | prefill | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G230 | 36864 × 6144 × 1536 | prefill | 1 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G231 | 53248 × 2560 × 4096 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G232 | 53248 × 3072 × 6144 | prefill | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G234 | 53248 × 4096 × 1024 | prefill | 1 | FP32@1×128 | 公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C012](#c012) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G236 | 53248 × 6144 × 1536 | prefill | 2 | FP32@1×128 | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a id="table-2-scale-2"></a>

### 2.2 A scale E8M0，block size 1×128

79 行需求，累计 888 条 Excel 记录。

| 编号 | M × N × K | 阶段 | 频次 | 激活 scale（Excel） | 权重 scale（公开 checkpoint 补充） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| G081 | 80 × 512 × 7168 | decode | 6 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G082 | 80 × 1536 × 7168 | decode | 6 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G083 | 80 × 6144 × 7168 | decode | 6 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G084 | 80 × 7168 × 3072 | decode | 6 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G085 | 80 × 7168 × 16384 | decode | 6 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G086 | 80 × 65536 × 1536 | decode | 6 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G087 | 96 × 512 × 7168 | decode | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G088 | 96 × 1536 × 7168 | decode | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G089 | 96 × 6144 × 7168 | decode | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G090 | 96 × 7168 × 3072 | decode | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G091 | 96 × 7168 × 16384 | decode | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G092 | 96 × 8192 × 1536 | decode | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G093 | 96 × 65536 × 1536 | decode | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G094 | 160 × 512 × 4096 | decode | 3 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G095 | 160 × 1024 × 4096 | decode | 3 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G096 | 160 × 4096 × 2048 | decode | 3 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G097 | 160 × 4096 × 4096 | decode | 3 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G098 | 160 × 4096 × 8192 | decode | 3 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G099 | 160 × 32768 × 1024 | decode | 3 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G100 | 192 × 512 × 4096 | decode | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G101 | 192 × 1024 × 4096 | decode | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G104 | 192 × 4096 × 2048 | decode | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G106 | 192 × 4096 × 8192 | decode | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G107 | 192 × 8192 × 1024 | decode | 2 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G110 | 192 × 32768 × 1024 | decode | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C005](#c005) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G111 | 240 × 512 × 4096 | decode | 9 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G112 | 240 × 1024 × 4096 | decode | 9 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G113 | 240 × 4096 × 2048 | decode | 9 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G114 | 240 × 4096 × 4096 | decode | 9 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G115 | 240 × 4096 × 8192 | decode | 9 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G116 | 240 × 32768 × 1024 | decode | 9 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G117 | 288 × 512 × 4096 | decode | 12 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G118 | 288 × 1024 × 4096 | decode | 12 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G119 | 288 × 4096 × 2048 | decode | 12 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G120 | 288 × 4096 × 4096 | decode | 12 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G121 | 288 × 4096 × 8192 | decode | 12 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G122 | 288 × 8192 × 1024 | decode | 6 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G123 | 288 × 32768 × 1024 | decode | 12 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C006](#c006) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G132 | 2560 × 512 × 7168 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G133 | 2560 × 768 × 7168 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G135 | 2560 × 1536 × 7168 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G144 | 2560 × 7168 × 384 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G145 | 2560 × 7168 × 8192 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G146 | 2560 × 7168 × 16384 | prefill | 12 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G147 | 2560 × 8192 × 1536 | prefill | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G151 | 2560 × 32768 × 1536 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G152 | 2560 × 65536 × 1536 | prefill | 12 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G154 | 3584 × 512 × 4096 | prefill | 32 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G155 | 3584 × 1024 × 4096 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G157 | 3584 × 4096 × 256 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G159 | 3584 × 8192 × 1024 | prefill | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G162 | 3584 × 32768 × 1024 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C007](#c007) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G167 | 4608 × 512 × 7168 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G168 | 4608 × 768 × 7168 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G169 | 4608 × 1536 × 7168 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G176 | 4608 × 7168 × 384 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G177 | 4608 × 7168 × 16384 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G178 | 4608 × 8192 × 1536 | prefill | 2 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G180 | 4608 × 65536 × 1536 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G181 | 5120 × 512 × 4096 | prefill | 32 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G182 | 5120 × 1024 × 4096 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G183 | 5120 × 4096 × 256 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G184 | 5120 × 4096 × 8192 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G185 | 5120 × 8192 × 1024 | prefill | 8 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G186 | 5120 × 32768 × 1024 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C008](#c008) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G190 | 6656 × 512 × 4096 | prefill | 64 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G192 | 6656 × 512 × 7168 | prefill | 20 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G193 | 6656 × 768 × 7168 | prefill | 20 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G195 | 6656 × 1024 × 4096 | prefill | 32 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G196 | 6656 × 1536 × 7168 | prefill | 20 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G204 | 6656 × 4096 × 256 | prefill | 32 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G211 | 6656 × 7168 × 384 | prefill | 20 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G212 | 6656 × 7168 × 8192 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G213 | 6656 × 7168 × 16384 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G214 | 6656 × 8192 × 1024 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G215 | 6656 × 8192 × 1536 | prefill | 10 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G222 | 6656 × 32768 × 1024 | prefill | 32 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Flash](#quant-flash)） | [C016](#c016) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G223 | 6656 × 32768 × 1536 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G224 | 6656 × 65536 × 1536 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@128×128（[DS-Pro](#quant-pro)） | [C004](#c004) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a id="table-2-scale-3"></a>

### 2.3 A scale FP32，block size 1×K（整行）

22 行需求，累计 33 条 Excel 记录。

| 编号 | M × N × K | 阶段 | 频次 | 激活 scale（Excel） | 权重 scale（公开 checkpoint 补充） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| G030 | 4 × 3072 × 4096 | decode | 1 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G031 | 4 × 4096 × 1536 | decode | 1 | FP32@1×1536 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G033 | 4 × 10240 × 4096 | decode | 1 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G048 | 16 × 3072 × 4096 | decode | 1 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G049 | 16 × 4096 × 1536 | decode | 1 | FP32@1×1536 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G051 | 16 × 4096 × 13312 | decode | 1 | FP32@1×13312 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G052 | 16 × 10240 × 4096 | decode | 2 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G053 | 16 × 26624 × 4096 | decode | 1 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G124 | 512 × 384 × 4096 | prefill | 2 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G125 | 512 × 3328 × 4096 | prefill | 1 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G126 | 512 × 4096 × 192 | prefill | 2 | FP32@1×192 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G127 | 512 × 4096 × 1664 | prefill | 1 | FP32@1×1664 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G129 | 512 × 10240 × 4096 | prefill | 4 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G163 | 4096 × 3328 × 4096 | prefill | 1 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G164 | 4096 × 4096 × 1664 | prefill | 1 | FP32@1×1664 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G189 | 6656 × 384 × 4096 | prefill | 2 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G201 | 6656 × 3328 × 4096 | prefill | 1 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G203 | 6656 × 4096 × 192 | prefill | 2 | FP32@1×192 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G205 | 6656 × 4096 × 1664 | prefill | 1 | FP32@1×1664 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G217 | 6656 × 10240 × 4096 | prefill | 4 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G233 | 53248 × 3328 × 4096 | prefill | 1 | FP32@1×4096 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G235 | 53248 × 4096 × 1664 | prefill | 1 | FP32@1×1664 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a id="table-2-scale-4"></a>

### 2.4 A scale 候选 / 未给出，待确认

10 行需求，累计 29 条 Excel 记录。

| 编号 | M × N × K | 阶段 | 频次 | 激活 scale（Excel） | 权重 scale（公开 checkpoint 补充） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| G032 | 4 × 4096 × 8192 | decode | 1 | 未给出 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G034 | 6 × 128 × 6144 | decode | 1 | FP32@1×128（候选，连线待确认） | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G041 | 12 × 128 × 6144 | decode | 1 | FP32@1×128（候选，连线待确认） | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G050 | 16 × 4096 × 8192 | decode | 2 | 未给出 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G054 | 24 × 128 × 6144 | decode | 2 | FP32@1×128（候选，连线待确认） | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G063 | 48 × 128 × 6144 | decode | 2 | FP32@1×128（候选，连线待确认） | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G128 | 512 × 4096 × 8192 | prefill | 4 | 未给出 | 公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认） | [C011](#c011) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G130 | 2560 × 128 × 6144 | prefill | 4 | FP32@1×128（候选，连线待确认） | 公开 FP32@128×128（[GLM](#quant-glm)） | [C002](#c002) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G165 | 4608 × 128 × 6144 | prefill | 4 | FP32@1×128（候选，连线待确认） | 公开 FP32@128×128（[GLM](#quant-glm)） | [C001](#c001) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G187 | 6656 × 128 × 6144 | prefill | 8 | FP32@1×128（候选，连线待确认） | 公开 FP32@128×128（[GLM](#quant-glm)） | [C009](#c009) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a id="table-2-scale-5"></a>

### 2.5 同一编号含多种 A scale / 来源缺失

6 行需求，累计 70 条 Excel 记录。

| 编号 | M × N × K | 阶段 | 频次 | 激活 scale（Excel） | 权重 scale（公开 checkpoint 补充） | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| G078 | 64 × 2048 × 4096 | decode | 2 | FP32@1×128(1) / 未给出(1) | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G105 | 192 × 4096 × 4096 | decode | 6 | FP32@1×128(2) / FP8E8M0@1×128(4) | 公开 E8M0@128×128（[DS-Flash](#quant-flash)）<br>公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C014](#c014) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G137 | 2560 × 2048 × 4096 | prefill | 2 | FP32@1×128(1) / 未给出(1) | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G158 | 3584 × 4096 × 8192 | prefill | 19 | FP32@1×128(1) / FP8E8M0@1×128(16) / 未给出(2) | 公开 E8M0@128×128（[DS-Flash](#quant-flash)）<br>公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C015](#c015) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G198 | 6656 × 2048 × 4096 | prefill | 2 | FP32@1×128(1) / 未给出(1) | 公开 BF16@128×128（[Q35](#quant-q35)；框架可转 FP32） | [C013](#c013) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |
| G207 | 6656 × 4096 × 8192 | prefill | 39 | FP32@1×128(1) / FP8E8M0@1×128(32) / 未给出(6) | 公开 E8M0@128×128（[DS-Flash](#quant-flash)）<br>公开 BF16@per-tensor（[Hy3](#quant-hy3)；BU 待确认）<br>公开 BF16@128×128（[Q397](#quant-q397)；框架可转 FP32） | [C017](#c017) | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm] | 接口 smoke 通过 |

<a id="table-3"></a>

## 表 3：linear · BF16 × BF16 → BF16

133 行需求，累计 326 条实例。编号在本清单中全局唯一。

本表对应 BF16 输入、BF16 输出；`gemm_a16w16` 的权重布局为 `[N,K]`。

| 编号 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- |
| G237 | 1 × 154880 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G238 | 2 × 64640 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G239 | 2 × 129280 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G240 | 3 × 120832 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G241 | 4 × 192 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G242 | 4 × 4096 × 8192 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G243 | 4 × 7168 × 21504 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G244 | 4 × 120832 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G245 | 4 × 154880 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G246 | 4 × 163840 × 7168 | decode,prefill | 3 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G247 | 6 × 32 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G248 | 6 × 256 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G249 | 6 × 6144 × 12288 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G250 | 6 × 154880 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G251 | 8 × 16160 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G252 | 8 × 19360 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G253 | 8 × 129280 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G254 | 8 × 248320 × 4096 | prefill | 2 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G255 | 12 × 32 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G256 | 12 × 64 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G257 | 12 × 256 × 2048 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G258 | 12 × 256 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G259 | 12 × 1024 × 2048 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G260 | 12 × 2048 × 512 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G261 | 12 × 2048 × 4096 | decode | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G262 | 12 × 6144 × 12288 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G263 | 12 × 9216 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G264 | 12 × 12288 × 2048 | decode | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G265 | 12 × 154880 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G266 | 12 × 248320 × 2048 | decode,prefill | 3 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G267 | 16 × 192 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G268 | 16 × 384 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G269 | 16 × 2112 × 7168 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G270 | 16 × 4096 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G271 | 16 × 7168 × 2048 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G272 | 16 × 7168 × 8192 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G273 | 16 × 7168 × 18432 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G274 | 16 × 12288 × 1536 | decode | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G275 | 16 × 16160 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G276 | 16 × 36864 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G277 | 16 × 120832 × 4096 | decode | 1 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G278 | 16 × 163840 × 7168 | decode | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G279 | 24 × 32 × 6144 | decode | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G280 | 24 × 256 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G281 | 24 × 15104 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G282 | 24 × 154880 × 6144 | decode | 1 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G283 | 32 × 19360 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G284 | 32 × 20480 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G285 | 48 × 32 × 6144 | decode | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G286 | 48 × 256 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G287 | 48 × 512 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G288 | 48 × 4096 × 8192 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G289 | 48 × 124160 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G290 | 48 × 154880 × 6144 | decode | 1 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G291 | 64 × 64 × 2048 | decode | 1 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G292 | 64 × 256 × 2048 | decode | 2 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G293 | 64 × 16160 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G294 | 64 × 31040 × 4096 | prefill | 2 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G295 | 64 × 248320 × 2048 | decode,prefill | 3 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G296 | 80 × 129280 × 7168 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G297 | 96 × 64 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G298 | 96 × 512 × 7168 | decode | 6 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G299 | 96 × 7168 × 21504 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G300 | 96 × 129280 × 7168 | decode | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G301 | 160 × 129280 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G302 | 192 × 64 × 4096 | decode | 3 | [C014](#c014) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G303 | 192 × 512 × 4096 | decode | 5 | [C014](#c014) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G304 | 192 × 4096 × 12288 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G305 | 192 × 124160 × 4096 | decode | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G306 | 192 × 129280 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G307 | 240 × 129280 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G308 | 288 × 64 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G309 | 288 × 512 × 4096 | decode | 9 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G310 | 288 × 4096 × 12288 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G311 | 288 × 129280 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G312 | 512 × 192 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G313 | 2048 × 64 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G314 | 2048 × 256 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G315 | 2048 × 1024 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G316 | 2048 × 2048 × 512 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G317 | 2048 × 2048 × 4096 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G318 | 2048 × 9216 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G319 | 2048 × 12288 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G320 | 2560 × 32 × 6144 | prefill | 4 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G321 | 2560 × 64 × 2048 | prefill | 1 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G322 | 2560 × 64 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G323 | 2560 × 256 × 2048 | prefill | 2 | [C013](#c013) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G324 | 2560 × 256 × 6144 | prefill | 2 | [C002](#c002) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G325 | 3584 × 64 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G326 | 3584 × 128 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G327 | 3584 × 512 × 4096 | prefill | 4 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G328 | 4608 × 32 × 6144 | prefill | 4 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G329 | 4608 × 64 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G330 | 4608 × 256 × 6144 | prefill | 2 | [C001](#c001) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G331 | 5120 × 64 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G332 | 5120 × 384 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G333 | 5120 × 512 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G334 | 5120 × 2112 × 7168 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G335 | 5120 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G336 | 5120 × 7168 × 256 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G337 | 5120 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G338 | 5120 × 7168 × 8192 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G339 | 5120 × 12288 × 1536 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G340 | 5120 × 16384 × 512 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G341 | 6656 × 32 × 6144 | prefill | 8 | [C009](#c009) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G342 | 6656 × 64 × 2048 | prefill | 2 | [C019](#c019) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G343 | 6656 × 64 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G344 | 6656 × 64 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G345 | 6656 × 128 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G346 | 6656 × 192 × 4096 | prefill | 2 | [C011](#c011) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G347 | 6656 × 256 × 2048 | prefill | 4 | [C019](#c019) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G348 | 6656 × 256 × 6144 | prefill | 4 | [C009](#c009) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G349 | 6656 × 384 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G350 | 6656 × 512 × 4096 | prefill | 4 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G351 | 6656 × 512 × 7168 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G352 | 6656 × 1024 × 2048 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G353 | 6656 × 2048 × 512 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G354 | 6656 × 2048 × 4096 | prefill | 2 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G355 | 6656 × 2112 × 7168 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G356 | 6656 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G357 | 6656 × 7168 × 256 | prefill | 2 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G358 | 6656 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G359 | 6656 × 7168 × 8192 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G360 | 6656 × 9216 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G361 | 6656 × 12288 × 1536 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G362 | 6656 × 12288 × 2048 | prefill | 1 | [C018](#c018) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G363 | 6656 × 16384 × 512 | prefill | 4 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G364 | 28672 × 16 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G365 | 40960 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G366 | 40960 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G367 | 53248 × 16 × 4096 | prefill | 1 | [C012](#c012) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G368 | 53248 × 4608 × 7168 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G369 | 53248 × 7168 × 2304 | prefill | 1 | [C003](#c003) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |

<a id="table-4"></a>

## 表 4：linear · FP32 × FP32 → FP32

35 行需求，累计 218 条实例。编号在本清单中全局唯一。

本表的接口级 smoke PASS 依赖当前 Triton 的 S03 LEGACY 补丁；不能推广到未带该补丁的环境。

| 编号 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- |
| G370 | 16 × 129280 × 512 | decode | 10 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G371 | 32 × 129280 × 256 | decode | 5 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G372 | 48 × 129280 × 256 | decode | 15 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G373 | 80 × 1 × 7680 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G374 | 80 × 4 × 28672 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G375 | 80 × 384 × 7168 | decode | 6 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G376 | 96 × 4 × 28672 | decode | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G377 | 96 × 384 × 7168 | decode | 8 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G378 | 160 × 1 × 4352 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G379 | 160 × 4 × 16384 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G380 | 160 × 256 × 4096 | decode | 3 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G381 | 192 × 4 × 16384 | decode | 1 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G382 | 192 × 256 × 4096 | decode | 4 | [C005](#c005) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G383 | 240 × 1 × 4352 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G384 | 240 × 4 × 16384 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G385 | 240 × 256 × 4096 | decode | 9 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G386 | 288 × 4 × 16384 | decode | 3 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G387 | 288 × 256 × 4096 | decode | 12 | [C006](#c006) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G388 | 2560 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G389 | 2560 × 384 × 7168 | prefill | 16 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G390 | 3584 × 4 × 16384 | prefill | 2 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G391 | 3584 × 256 × 4096 | prefill | 16 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G392 | 4608 × 384 × 7168 | prefill | 4 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G393 | 5120 × 4 × 16384 | prefill | 2 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G394 | 5120 × 256 × 4096 | prefill | 16 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G395 | 6656 × 4 × 16384 | prefill | 4 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G396 | 6656 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G397 | 6656 × 256 × 4096 | prefill | 32 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G398 | 6656 × 384 × 7168 | prefill | 20 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G399 | 20480 × 4 × 28672 | prefill | 1 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G400 | 28672 × 4 × 16384 | prefill | 2 | [C007](#c007) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G401 | 36864 × 4 × 28672 | prefill | 1 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G402 | 40960 × 4 × 16384 | prefill | 2 | [C008](#c008) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G403 | 53248 × 4 × 16384 | prefill | 4 | [C016](#c016) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |
| G404 | 53248 × 4 × 28672 | prefill | 2 | [C004](#c004) | [matmul_persistent][fn-persistent] | 接口 smoke 通过（S03） |

<a id="table-5"></a>

## 表 5：linear · BF16 × BF16 → FP32

27 行需求，累计 172 条实例。编号在本清单中全局唯一。

本表使用 `gemm_a16w16(..., dtype=torch.float32)`，保留 BF16 输入、FP32 输出语义。

| 编号 | M × N × K | 阶段 | 频次 | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- |
| G405 | 96 × 512 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G406 | 96 × 1024 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G407 | 96 × 2048 × 7168 | decode | 4 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G408 | 192 × 512 × 4096 | decode | 2 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G409 | 192 × 1024 × 4096 | decode | 1 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G410 | 192 × 2048 × 4096 | decode | 2 | [C005](#c005) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G411 | 288 × 512 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G412 | 288 × 1024 × 4096 | decode | 3 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G413 | 288 × 2048 × 4096 | decode | 6 | [C006](#c006) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G414 | 2560 × 512 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G415 | 2560 × 1024 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G416 | 2560 × 2048 × 7168 | prefill | 8 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G417 | 3584 × 512 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G418 | 3584 × 1024 × 4096 | prefill | 4 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G419 | 3584 × 2048 × 4096 | prefill | 8 | [C007](#c007) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G420 | 4608 × 512 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G421 | 4608 × 1024 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G422 | 4608 × 2048 × 7168 | prefill | 2 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G423 | 5120 × 512 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G424 | 5120 × 1024 × 4096 | prefill | 4 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G425 | 5120 × 2048 × 4096 | prefill | 8 | [C008](#c008) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G426 | 6656 × 512 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G427 | 6656 × 512 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G428 | 6656 × 1024 × 4096 | prefill | 8 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G429 | 6656 × 1024 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G430 | 6656 × 2048 × 4096 | prefill | 16 | [C016](#c016) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |
| G431 | 6656 × 2048 × 7168 | prefill | 10 | [C004](#c004) | [gemm_a16w16][fn-a16] | 接口 smoke 通过 |

<a id="table-6"></a>

## 表 6：linear · BF16 × INT4 → BF16

4 行需求，累计 4 条实例。编号在本清单中全局唯一。

这 4 行全部来自 Kimi Excel 的 **EAGLE3 draft**：G432=`eagle3_o_proj`、G433=`eagle3_down_proj`、G434=`eagle3_gate_up_proj`、G435=`eagle3_qkv_proj`，不是 Kimi 主模型 routed experts。公开 Kimi-K2.6 主模型的 INT4 group32/对称/pack-quantized 配置不能套到未提供 checkpoint 的 EAGLE3；这里继续保留 scale dtype / group size / zero-point / packing 待确认。AWQ 入口仅代表该格式的小尺寸接口证据，不证明 EAGLE3 的量化格式与 AWQ 相容。

| 编号 | M × N × K | 阶段 | 频次 | 权重 scale | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| G432 | 4 × 7168 × 8192 | decode | 1 | EAGLE3：scale dtype / group / packing 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G433 | 4 × 7168 × 12288 | decode | 1 | EAGLE3：scale dtype / group / packing 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G434 | 4 × 24576 × 7168 | decode | 1 | EAGLE3：scale dtype / group / packing 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |
| G435 | 4 × 24576 × 14336 | decode | 1 | EAGLE3：scale dtype / group / packing 待确认 | [C003](#c003) | [awq_gemm_triton][fn-awq] | 接口 smoke 通过 |

<a id="table-8"></a>

## 表 8：grouped_gemm · FP8E4M3 × FP4E2M1 → BF16

34 行需求，累计 304 条实例。编号在本清单中全局唯一。

`grouped_gemm_a8wfp4` 支持等长及 ragged（含空组）；A scale 可为 K32/K128，W scale 为 K32。原完整需求的 34/34 编译通过不等于 34/34 精度通过；状态仍为旧接口 smoke 证据。W 的 E8M0/K32 由公开 DeepSeek-V4 / AMD MXFP4 配置与权重 header 补充；BU 实际文件需绑定。Q35 MXFP4 的公开 AMD 版本使用 FP4 activation，而 Excel 要求 FP8 activation，属于版本/路径冲突，不将其直接合并。

| 编号 | M × N × K | 分组数 E | 阶段 | 频次 | 激活 scale | 权重 scale | 模型 / 配置 | BoltOps 函数 / 实现 | 状态 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- |
| G472 | 2 × 1024 × 2048 | 80 | decode | 2 | FP8E8M0@1×32 | 公开 E8M0@1×32（AMD Q35；BU 激活路径不同） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G473 | 2 × 2048 × 512 | 80 | decode | 2 | FP32@1×32 | 公开 E8M0@1×32（AMD Q35；BU 激活路径不同） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G474 | 20 × 6144 × 7168 | 24 | decode | 6 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G475 | 20 × 7168 × 3072 | 24 | decode | 6 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G476 | 24 × 6144 × 7168 | 24 | decode | 8 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G477 | 24 × 7168 × 3072 | 24 | decode | 8 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G478 | 30 × 4096 × 2048 | 32 | decode | 3 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G479 | 30 × 4096 × 4096 | 32 | decode | 3 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G480 | 36 × 4096 × 2048 | 32 | decode | 4 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G481 | 36 × 4096 × 4096 | 32 | decode | 4 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C005](#c005) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G482 | 45 × 4096 × 2048 | 32 | decode | 9 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G483 | 45 × 4096 × 4096 | 32 | decode | 9 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G484 | 54 × 4096 × 2048 | 32 | decode | 12 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G485 | 54 × 4096 × 4096 | 32 | decode | 12 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C006](#c006) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G486 | 64 × 1024 × 2048 | 256 | prefill | 2 | FP8E8M0@1×32 | 公开 E8M0@1×32（AMD Q35；BU 激活路径不同） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G487 | 64 × 2048 × 512 | 256 | prefill | 2 | FP32@1×32 | 公开 E8M0@1×32（AMD Q35；BU 激活路径不同） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G488 | 160 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G489 | 160 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G490 | 208 × 1024 × 2048 | 256 | prefill | 2 | FP8E8M0@1×32 | 公开 E8M0@1×32（AMD Q35；BU 激活路径不同） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G491 | 208 × 2048 × 512 | 256 | prefill | 2 | FP32@1×32 | 公开 E8M0@1×32（AMD Q35；BU 激活路径不同） | [C018](#c018) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G492 | 320 × 6144 × 7168 | 48 | prefill | 12 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G493 | 320 × 7168 × 3072 | 48 | prefill | 12 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G494 | 416 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G495 | 416 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G496 | 576 × 6144 × 7168 | 48 | prefill | 4 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G497 | 576 × 7168 × 3072 | 48 | prefill | 4 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G498 | 672 × 4096 × 2048 | 32 | prefill | 16 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C007](#c007) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G499 | 672 × 4096 × 4096 | 32 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C007](#c007) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G500 | 832 × 6144 × 7168 | 48 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G501 | 832 × 7168 × 3072 | 48 | prefill | 16 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C004](#c004) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G502 | 960 × 4096 × 2048 | 32 | prefill | 16 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C008](#c008) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G503 | 960 × 4096 × 4096 | 32 | prefill | 16 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C008](#c008) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G504 | 1248 × 4096 × 2048 | 32 | prefill | 32 | FP32@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C016](#c016) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |
| G505 | 1248 × 4096 × 4096 | 32 | prefill | 32 | FP8E8M0@1×128 | 公开 E8M0@1×32（DS-V4 MXFP4） | [C016](#c016) | [grouped_gemm_a8wfp4][fn-fp8fp4] | 接口 smoke 通过 |

<a id="requirements-cross-check"></a>

## 原始需求与公开模型核对（2026-10-23）

13 份 XLSX 共提取 2504 条 linear / BMM / 单次 grouped GEMM 记录。现有 469 行全部能按算子族、M/N/K、batch/E、输入与输出 dtype 匹配，频次合计仍为 2424；没有发现现有编号行的 shape/dtype/频次抄录冲突。其余 80 条记录归并为下文 42 个 grouped shape。对齐 Excel 不等于原始模型运行 trace 正确，也不等于任何精度/性能验收。

公开信息选用模型作者或量化发布方的配置，并读取少量 safetensors 文件头核实 dtype/shape；未下载完整模型或权重分片。固定 revision、采样 tensor、来源 hash 及局限见 [核对记录](debug_tuning/requirements_audit_20261023/REPORT.md) / [模型来源](debug_tuning/requirements_audit_20261023/model-sources.json)。

| 模型 / 来源 | Excel 激活要求 | 公开 checkpoint 的权重 scale | 对比结论 |
| --- | --- | --- | --- |
| <a id="quant-flash"></a>[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) | E8M0/K128；grouped down 为 FP32/K128 | dense：E8M0，N128/K128；grouped FP4：E8M0，N1/K32 | 模型是混合精度，不能把文件名 FP4 套到 dense FP8 权重。 |
| <a id="quant-pro"></a>[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) | 同 Flash | dense：E8M0，N128/K128；grouped FP4：E8M0，N1/K32 | Excel 简称 FP8，但 grouped 输入确实为 FP8×FP4；公开 expert_dtype=fp4，与此一致。 |
| <a id="quant-glm"></a>[GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8/blob/main/config.json) | 明确量化输出为 FP32/K128；indexer_wk 仅候选、连线待确认 | FP32，N128/K128 | G171 旧实验 B E8M0/N1/K128 不等于原始 checkpoint。DeepGEMM 特定路径可重新量化为 UE8M0，需实际后端证据。 |
| <a id="quant-q35"></a>[Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8/blob/main/config.json) | 已列输出为 FP32/K128；部分 o_proj 缺失 | BF16，N128/K128（实读 dense/shared/expert header）；标准 SGLang FP8 层分配 FP32 scale | BF16 存储与 FP32 kernel 输入不是矛盾；需记录加载转换，不能改标 E8M0。 |
| <a id="quant-q397"></a>[Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/blob/main/config.json) | 同 Q35 FP8 | BF16，N128/K128（实读 dense/shared/expert header）；框架可转 FP32 | 同一 G 含 DS/Qwen 时须保留两种 scale，不能按 shape 合并成同一量化配置。 |
| <a id="quant-hy3"></a>[Hy3-FP8 / Excel Hunyuan-V3](https://huggingface.co/tencent/Hy3-FP8/blob/main/config.json) | 逐行 FP32/整行 K；部分 o_proj 没有量化输出 | weight_scale 为 BF16 scalar；input_scale 为 FP32 scalar；activation_scheme=static | 公开静态 per-tensor 与 Excel 逐行量化路径不一致。保留 BU 要求，必须确认使用版本或再量化过程。 |
| <a id="quant-mx4"></a>[Qwen3.5-35B-A3B-MXFP4（AMD 候选）](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4/blob/main/config.json) | grouped gate_up 为 FP8 + E8M0/K32；down 为 FP8 + FP32/K32 | U8 存储 E8M0，N1/K32；公开 activation/weight 均为 FP4 | Excel A 是 FP8，公开 AMD A 是 FP4；同名 MXFP4 不足以确认实际 BU checkpoint。 |
| <a id="quant-kimi"></a>[Kimi-K2.6 / W4A16](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json) | grouped A 为 BF16；无 A scale | routed experts：INT4，对称 K32，BF16 scale；pack-quantized（compressed-tensors） | 可补充主模型 grouped 权重背景；表6是独立 EAGLE3，不能据此补 group32 或认定 AWQ。 |

证据优先级：BU Excel 决定本地调用形状和显式 dtype；公开模型补充 checkpoint 信息；框架代码只说明可能的加载/转换路径。冲突保留在表中等待 BU 实际 checkpoint revision、SGLang commit、量化参数和后端选择，不用公开默认值覆盖 Excel。GLM 表头同时写“5.2/5.1”、Kimi 表头写“2.7/2.6”，第 3 行描述分别明确 5.1、2.6，本清单据第 3 行命名，版本歧义仍保留。

G171 原始定位：`shape_05_glm_128k_s2000.xlsx / prefill_shape!F201:I201`，`mlp_gate_up_proj`，`Prefill_Remainder`，`[4608,6144] × [6144,3072] → [4608,3072]`；A scale 来自 `H200:I200` 的 `[4608,48] / FP32`，推导 K128。GLM gate/up 的公开 FP32 scale 为 `[96,48]` 对应单个 `[12288,6144]` 权重；TP8 合并 gate/up 后的逻辑 B scale 可推导为 `[24,48]`，仍需核对运行时拼接/stride。

前两个 FP8 接口可表达 B 的 N128/K128：AITer wrapper 从 scale shape 推导 `GROUP_N/GROUP_K`；vLLM wrapper 接收 `block_size=[128,128]`。此前 N1 实验不是这组 scale 的验证。Hy3 公开 K 不变 scalar 语义与 `triton_scaled_mm` 相容，但公开静态 A 与 Excel 逐行 A 仍待确认。

本次只校正文档及来源；不新增测试 PASS，不恢复暂停进程，不修改任何生产 kernel 或默认 config。ResNet DOCX 是 BW1000 上 INT8 融合卷积/quant_gemm 记录，继续独立于这 469 行；公开 [TorchVision ResNet50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) 的标准 FC 为 2048→1000，只作结构参考，不能补写 BU 实际量化 scale/zero-point，也不把融合卷积 FPS 作为 dense GEMM 指标。

### 原始文件与部署方案

X 编号用于回到 Excel 原文件；SHA256 见模型来源 JSON。DeepSeek Pro 的 P 侧 a/b 方案在旧 C004 中合并，现补回方案信息；不因此更改已有 M/N/K。全部 sheet（包括 prefill_shape_cp8）均已核对。

| 来源 | 原始文件 | P 侧方案（原文摘要） |
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

### 尚未纳入 G 编号表的 grouped GEMM 需求

以下是 Excel 中的单次 grouped GEMM，不是路由、激活或 fused_experts 包装接口。36 个 FP8×FP8 shape（70 条来源记录）及 6 个 BF16×INT4 shape（10 条来源记录）此前未列入 469 行。现在补列为**待范围确认 / 待接口映射与验证**，不恢复已删除的 MoE 表、不重排 G 编号，也不宣称它们已被现有接口支持。E、M 沿用源输入形状，不能据此证明实际等长或 ragged 分组。逐条原文件/sheet/行号见 [缺口清单](debug_tuning/requirements_audit_20261023/grouped-gaps.csv)。

| M × N × K | E | A × B → 输出 | Excel A scale | 模型来源 | 记录数 |
| --- | ---: | --- | --- | --- | ---: |
| 6 × 4096 × 7168 | 24 | BF16 × INT4 → BF16 | 不适用（BF16） | X11 | 1 |
| 6 × 7168 × 2048 | 24 | BF16 × INT4 → BF16 | 不适用（BF16） | X11 | 1 |
| 854 × 4096 × 7168 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | X11 | 2 |
| 854 × 7168 × 2048 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | X11 | 2 |
| 1110 × 4096 × 7168 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | X11 | 2 |
| 1110 × 7168 × 2048 | 48 | BF16 × INT4 → BF16 | 不适用（BF16） | X11 | 2 |
| 3 × 1024 × 2048 | 221 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X09 | 2 |
| 3 × 2048 × 512 | 221 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X09 | 2 |
| 3 × 3072 × 4096 | 11 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×4096 | X12 | 1 |
| 3 × 4096 × 1536 | 11 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×1536 | X12 | 1 |
| 4 × 2048 × 4096 | 63 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X13 | 1 |
| 4 × 4096 × 1024 | 63 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X13 | 1 |
| 4 × 4096 × 6144 | 15 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X07 | 1 |
| 4 × 6144 × 2048 | 15 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X07 | 1 |
| 6 × 4096 × 6144 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X08 | 1 |
| 6 × 6144 × 2048 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X08 | 1 |
| 11 × 3072 × 4096 | 12 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×4096 | X12 | 1 |
| 11 × 4096 × 1536 | 12 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×1536 | X12 | 1 |
| 12 × 4096 × 6144 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X07 | 1 |
| 12 × 6144 × 2048 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X07 | 1 |
| 15 × 2048 × 4096 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X13 | 2 |
| 15 × 4096 × 1024 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X13 | 2 |
| 24 × 4096 × 6144 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X08 | 1 |
| 24 × 6144 × 2048 | 16 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X08 | 1 |
| 80 × 1024 × 2048 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X09 | 2 |
| 80 × 2048 × 512 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X09 | 2 |
| 171 × 3072 × 4096 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×4096 | X12 | 2 |
| 171 × 4096 × 1536 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×1536 | X12 | 2 |
| 208 × 1024 × 2048 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X09 | 2 |
| 208 × 2048 × 512 | 256 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X09 | 2 |
| 560 × 2048 × 4096 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X13 | 4 |
| 560 × 4096 × 1024 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X13 | 4 |
| 640 × 4096 × 6144 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X08 | 2 |
| 640 × 6144 × 2048 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X08 | 2 |
| 1040 × 2048 × 4096 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X13 | 4 |
| 1040 × 4096 × 1024 | 64 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X13 | 4 |
| 1152 × 4096 × 6144 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X07 | 2 |
| 1152 × 6144 × 2048 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X07 | 2 |
| 1664 × 4096 × 6144 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X07/X08 | 4 |
| 1664 × 6144 × 2048 | 32 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×128 | X07/X08 | 4 |
| 2219 × 3072 × 4096 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×4096 | X12 | 2 |
| 2219 × 4096 × 1536 | 24 | FP8E4M3 × FP8E4M3 → BF16 | FP32@1×1536 | X12 | 2 |

Kimi 主模型 grouped INT4 的公开 group32/对称/packed 格式只作为该组候选补充；GLM/Qwen grouped FP8 可参照各自公开 N128/K128 权重 scale。实际调用的 layout、packing、scale dtype 及空组/offsets 仍须由部署方提供。

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
