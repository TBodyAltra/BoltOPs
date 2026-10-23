# 少伯 BU 原始需求与公开模型核对

日期：2026-10-23。读取实际目录 `/mnt/c/Users/xukang/Downloads/shaobob0_shapes`，13 份 XLSX、1 份 ResNet DOCX；未改原始文件、生产 kernel、默认配置或测量环境。完整下载及分析材料留在仓库外 `/home/kxu/shaobo-requirements-audit-20261023/`。

## 核对范围与结果

- XLSX 的 `prefill_shape`、`prefill_shape_cp8`、`decode_shape` 全部纳入；仅抽取 linear、BMM、单次 grouped GEMM，不恢复路由/激活/fused_experts 接口表。
- 2504 条原始记录中，2424 条匹配现有 469 个 G 编号。按算子族、M/N/K、batch/E、输入/输出 dtype 及频次逐项核对，无现有编号行的抄录冲突。
- 频次统计的是 Excel 行出现次数；没有乘 `layer_count`。例如 G171 出现 1 次，原行 layer_count=3，均保留各自含义。
- 余下 80 条记录为 42 个未编号 grouped shape：36 个 FP8×FP8（70 条记录）、6 个 BF16×INT4（10 条）。补入需求文档的待范围确认清单；没有添加接口支持或 PASS 声明。
- 表2仍按 A scale 五组组织，行数 90/79/22/10/6；B scale 按模型补充，不能将同 shape 的不同模型量化配置合并。7 个 indexer_wk 编号仍只有同形 quantize_qkv_a 输出候选，没有 tensor ID/显式量化连线，继续待确认；3 个纯 o_proj 缺失项不再猜 E8M0/FP32。

原 Excel 未包含 B scale 参数；旧文档统一写“离线 1×128”是没有直接来源的补写。此次全部撤销，明确标注公开 checkpoint 信息，并保留 BU 版本/运行时绑定未完成这一限制。

## 逐模型关键结论

| 模型 | 已核实的公开格式 | 与 BU/旧文档的关系 |
| --- | --- | --- |
| DeepSeek-V4-Flash | dense E8M0/N128/K128；FP4 expert E8M0/N1/K32 | 全模型“FP4”不代表 dense 也为 FP4；旧 dense B/N1 无依据 |
| DeepSeek-V4-Pro | 同 Flash，`expert_dtype=fp4` | BU 模型标签 FP8 与 grouped FP4 是混合精度，非 shape 冲突；补回 P 侧 a/b 部署区别 |
| GLM-5.1-FP8 | dense FP32/N128/K128，gate/up scale `[96,48]`，weight `[12288,6144]` | G171 旧 FP32 A + E8M0/N1 B 的性能结果不等于官方 checkpoint 组合；不能据此验证 N128 B |
| Qwen3.5-35B-A3B-FP8 | dense/shared/expert sample 的 scale 均为 BF16，N128/K128 | checkpoint BF16 与框架可分配 FP32 必须分开；不自动改成 E8M0 |
| Qwen3.5-397B-A17B-FP8 | 同 Q35 FP8 | 同一 G 可能同时来自 DeepSeek 与 Qwen，B dtype 不同，逐来源保留 |
| tencent/Hy3-FP8 | weight_scale 为 BF16 scalar；input_scale 为 FP32 `[1]`；静态激活量化 | Excel 却列逐行 FP32 scale/量化算子；属于公开版本与 BU 路径差异，不能静默覆盖 |
| amd/Qwen3.5-35B-A3B-MXFP4 | weight U8 编码 E8M0/K32；配置/模型卡 activation 也是 FP4 | Excel grouped A 是 FP8，明确冲突。AMD 仓库仅作为候选版本，不能认定就是 BU 所用 |
| moonshotai/Kimi-K2.6 | routed experts INT4、对称 K32、BF16 scale、compressed-tensors pack-quantized | 可供新增 grouped INT4 需求参考。表6四个算子是独立 EAGLE3 draft，group/packing/scale 仍未知，不能套用主模型或假定 AWQ |

所有模型的固定 revision、原配置 hash、实际 header 采样 tensor/dtype/shape/URL 在 [model-sources.json](model-sources.json)。只作采样张量结论；不宣称所有层都同 dtype。Range 读取文件前缀并解析 header，未下载完整模型或权重分片。AMD MXFP4 的 excluded dense attention 另抽查了 in_proj_qkv/in_proj_z/out_proj，均为 BF16，与 Excel 该部分 BF16 权重一致；这不消除 grouped activation A4/A8 的差异。

## 框架与接口边界

[SGLang FP8 层](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py) 的标准 block-FP8 分配浮点 FP32 scale；MXFP8 则可使用 uint8。checkpoint 文件中的 BF16 scale 与 kernel 最终输入 dtype 不应混写。这是公开 main 的实现说明，不是 BU 部署 commit 证明。

[SGLang DeepGEMM 转换](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_utils.py) 中 `requant_block_scale_ue8m0_for_deepgemm` 仅在选中 DeepGEMM、128×128 block、满足相应形状/输出等条件时重新量化权重和 scale。它不意味着所有 GLM/Qwen checkpoint 本来就是 E8M0，更不意味着 N 分组变成 1。

BoltOps 前两个 FP8 wrapper 有 N128/K128 的表达能力：`gemm_a8w8_blockscale` 从 B scale shape 推导 GROUP_N/GROUP_K（还要求 GROUP_K=BLOCK_SIZE_K）；`w8a8_triton_block_scaled_mm` 直接接收 `[block_n,block_k]`。此前优化副本大多固定 GROUP_N=1，不能复用为新确认组合的数值/性能证据。`triton_scaled_mm` 仅支持 K 不变 scalar/per-row/per-column scale，可表达 Hy3 公开 scalar 语义，实际 BU 路径仍未知。本次不跑测试、不修改支持状态。

## 仍需绑定的信息

BU 实际 model path/revision、SGLang commit、checkpoint 转换过程、启动参数及 GEMM 后端；Hy3 静态/逐行路径；Q35 MXFP4 的 A8/W4 与公开 A4/W4 差异；indexer_wk/o_proj 的 A scale 连线；EAGLE3 独立 checkpoint/INT4 打包；grouped 的 token 布局/offsets 和各组实际 M。Excel 的 E/M 只代表原输入形状，不证明等长执行。

GLM 表头“5.2/5.1”、Kimi 表头“2.7/2.6”有命名歧义，第3行分别写5.1、2.6，保持旧模型名并记录歧义。ResNet DOCX 的 BW1000 融合卷积 FPS/INT8 quant_gemm 继续独立记录，不能用于 dense GEMM 性能结论。另查 [TorchVision 官方 ResNet 实现](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)：标准 ResNet50 的 FC 为 2048→1000；这只提供结构背景，不能据此补写 BU INT8 的 scale、zero-point、融合布局或 FC 实际量化调用。

## 可追溯文件

- [case-provenance.csv](case-provenance.csv)：469 行到 Excel 原行/候选量化输出的映射。X 编号对应 model-sources.json 的原文件名及 SHA256。
- [grouped-gaps.csv](grouped-gaps.csv)：42 个未编号 grouped shape 及来源行号，状态为待范围/接口确认。
- [verification.json](verification.json)：行身份/频次/原始数据/采样 header/文档链接检查结果。

原始 XLSX、权重数据、完整文件头、日志和运行产物不入 Git；不保存新的 smoke 证据，不运行验证队列。历史实验数据保留，新增来源边界说明，不改写其测量值。
