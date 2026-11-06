# 海光接口与送测默认值（2026-11-06）

依据仓库 `zhangjingz/xpu-perf` 默认分支 `shaobo-support`，固定提交 f50e796c5b2b056c1a684b57a4bb2198a7b14d9b。此记录补充此前公开 GitHub 仓库的接口分析，不将新增默认值当成 Excel 原文或 GPU 已验收结果。

## 当前需求口径

- Basic INT8 默认输出 BF16，普通 FP8 采用已有 LMSlim BF16 输出约定；Basic BF16 输出 BF16。Basic FP4 的输出 dtype 原表未规定，仍保留待确认。
- 未提供的量化 scale：A 使用 FP32 `[M]`，按 token；B/W 使用 FP32 `[N]`，按输出通道。此为用户于 2026-11-06 指定的实现默认值。普通 FP4 沿用此默认，不擅自解释为 NVFP4。
- Basic BF16 不需要量化 scale。quant_matmul BF16 若沿用其统一 scale 接口，按 FP32 token/输出通道默认值列出，不表示 BF16 数据发生了低精度量化。
- MXFP8 / MXFP4 的 scale 已由标准规定为 E8M0、每 32 元素共享，保留该已知约定；矩阵分组轴、物理布局和 scale 动态/静态生成策略未由现有实现确定，不用 FP32 默认值覆盖。
- `6192` 来自 Excel；海光 DCU workload 为 `6144`。本文保持 Excel 数值并记录差异。
- 统一默认值见 [implementation-policy.json](../implementation-policy.json)，生成后的各组表格及 expanded-cases.csv 均应用此口径，source.json 的原始 Excel 摘录不改写。

## Basic GEMM

| 组合 | 现有代码路径 | 输出 / scale |
| --- | --- | --- |
| BF16 × BF16 | GemmOp → torch.matmul | BF16；无量化 scale |
| INT8 × INT8 | LMSlim → hipblaslt_w8a8_gemm | 默认 BF16；FP32 `[M]` / `[N]` |
| FP8 × FP8 | E4M3FN 别名归一化 → LMSlim → hipblaslt_w8a8_gemm | 默认 BF16；FP32 `[M]` / `[N]` |
| FP4 × FP4 | utils 已有 E2M1×2 映射，但 GEMM parser 跳过 | 输出待确认；scale 按本文 FP32 默认，待实现 |
| MXFP8 / MXFP4 | 此 Basic provider 未实现 | 标准 scale 定义不等于代码已实现 |

Basic 输入为 A[M,K]、B[K,N]。LMSlim 执行函数中创建两份 FP32 全 1 scale，并将 B 转为连续的 [N,K]，以 NT 布局调用 native 接口。scale 创建与布局转换在调用内，直接测得的延迟可能包含这些开销。

**INT8/FP8 的 LMSlim Python 封装没有先把数据反量化成 BF16 再执行普通 mm，而是直接传递低精度 tensor 给 W8A8 GEMM 接口。** 底层实现位于外部 LMSlim/hipBLASLt；仅凭该封装不能证明实际用了哪条 MMAC 指令。尚无 native 源码、编译 ISA 或对应 perf 证据，因此 MMAC 的硬件指令类型标为待验证，不新增性能 PASS。

基础 GemmOp 的 INT8 是另一条路径：直接返回 torch.matmul，不携带量化 scale；默认输出 INT8。即使设置 dst_dtype=BF16，其执行函数也未显式 cast。CPU 小尺寸 M=2/N=3/K=4 验证结果为“元数据 BF16、实际 INT8”。因此本文统一 INT8 输出 BF16 是送测要求，base 的该行为仍需修复/对齐，不能据文档默认值宣称代码已修复。

## quant_matmul

海光 LMSlim 类调用 `hipblaslt_w8a8_gemm(A, W, scale_a, scale_b, M, N, K, "NT", out_dtype, bias)`，A[M,K]、W[N,K]、FP32 scale_a[M]、FP32 scale_b[N]，当前 INT8 输出 BF16。该类没有覆盖 base 的 INT8-only 参数检查，因此另外五组目前仍不支持。

与之不同，quant_matmul 的 base 参考实现明确执行：INT8 → BF16 mm → FP32 乘 scale → BF16 输出。该 BF16 模拟路径不能代表 Basic LMSlim 的实现或原生 INT8 性能。

当前 PMD 环境未安装 lmslim，PyTorch 2.6 也没有 float4_e2m1fn_x2 属性。此前 PMD launch 记录来自公开仓库，不能作为这个 fork 的 native 实现验证；本次未新增 GPU 正确性或性能 PASS。

## 源码与标准

- [Basic LMSlim 接口](https://gitlab.hygon.cn/zhangjingz/xpu-perf/-/blob/f50e796c5b2b056c1a684b57a4bb2198a7b14d9b/projects/micro_perf/vendor_ops/DCU/ops/lmslim/gemm.py)
- [Basic GemmOp](https://gitlab.hygon.cn/zhangjingz/xpu-perf/-/blob/f50e796c5b2b056c1a684b57a4bb2198a7b14d9b/projects/micro_perf/op_defs/basic_ops/tensor_gemm_ops.py)
- [quant_matmul LMSlim 接口](https://gitlab.hygon.cn/zhangjingz/xpu-perf/-/blob/f50e796c5b2b056c1a684b57a4bb2198a7b14d9b/projects/micro_perf/vendor_ops/DCU/ops/lmslim/quant_matmul.py)
- [quant_matmul 参数、scale 与 base 计算](https://gitlab.hygon.cn/zhangjingz/xpu-perf/-/blob/f50e796c5b2b056c1a684b57a4bb2198a7b14d9b/projects/micro_perf/op_defs/llm_ops/quant_matmul.py)
- [dtype 映射](https://gitlab.hygon.cn/zhangjingz/xpu-perf/-/blob/f50e796c5b2b056c1a684b57a4bb2198a7b14d9b/src/xpu_perf/micro_perf/core/utils.py)
- [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

源码哈希与独立检查范围见 [source-findings.json](source-findings.json)。
