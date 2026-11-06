# quant_matmul 六组精度：代码约定与运行结果

当前需求默认值与海光 fork 的实现信息见 [2026-11-06 补充口径](hygon_fork_20261106/README.md)。以下 CPU/PMD 记录按原采集版本保留。

2026-11-05 重新 fetch `bytedance/xpu-perf` 的 main，远端与本地均为 `71a12101509635421d02c3d28c9202951ded7745`。以 Excel 六组精度构造的 [quant-matmul.json](quant-matmul.json) 为输入，用上游 `parse_json_file` 展开 **6 × 4 × 46 = 1104** 个组合，逐项实例化原始 QuantMatmulOp。没有绕过精度检查或给缺失的精度补实现。

## 六组详细信息

全部配置的输出要求为 BF16。下面的“默认”来自算子接口文档，可由厂商覆盖；“标准”来自 OCP MX，不表示本次代码已执行；“实测”来自 CPU 原始参考路径。

| Excel 配置：A / W / compute / 输出 | A、W 元素格式 | A scale | W scale | scale 共享粒度 | 本次原始入口结果 |
| --- | --- | --- | --- | --- | --- |
| BF16 / BF16 / BF16 / BF16 | 工具映射 BF16 | 此精度路径未提供 | 此精度路径未提供 | 未实现 | 184 个组合均抛 ValueError |
| INT8 / INT8 / INT8 / BF16 | 实测 A、W 均 torch.int8 | 实测 FP32[M]、全 1 | 实测 FP32[N]、全 1 | A per-token，W per-output-channel；共享整个 K | 184 个组合通过参数准备；4 个 MNK 完成 CPU 执行 |
| FP8 / FP8 / FP8 / BF16 | 工具默认映射 E4M3FN | 文档默认 FP32；未创建 | 文档默认 FP32；未创建 | 文档默认 per-token / per-channel；无指定 K 分块 | 184 个组合均抛 ValueError |
| FP4 / FP4 / FP4 / BF16 | `float4` 未注册，具体编码和 packing 未定义 | 文档提供 FP32 默认；FP4 具体约定待定 | 文档提供 FP32 默认；FP4 具体约定待定 | 文档提供 per-token / per-channel 默认；最终依格式/厂商定义 | 184 个组合均抛 ValueError；单独查 dtype 映射抛 KeyError |
| MXFP8 / MXFP8 / MXFP8 / BF16 | 工具映射 E4M3FN；OCP 也定义 E5M2 型 MXFP8 | 标准 E8M0；未创建 | 标准 E8M0；未创建 | 标准每 32 元素共享 1 scale；矩阵分组轴、布局未在此实现 | 184 个组合均抛 ValueError |
| MXFP4 / MXFP4 / MXFP4 / BF16 | 标准 E2M1；工具映射 float4_e2m1fn_x2，creator 返回 INT8 存储 | 标准 E8M0；未创建 | 标准 E8M0；未创建 | 标准每 32 元素共享 1 scale；矩阵分组轴、布局未在此实现 | 184 个组合均抛 ValueError |

FP4/FP8 的元素名称本身没有确定 scale；这里补充的是接口文档提供的默认值，不能与已完成的运行结果混写。MXFP4/MXFP8 的 scale 格式及 32 元素分组由标准定义，不能因为代码缺失就说标准格式不确定。所有精度的动态/静态量化策略、厂商最终选择和真实 kernel BLOCK_M/N/K，均不能由这几个配置字符串直接确定。

后续已补做 [PMD launch.py 验证](pmd_launch_20261105/README.md)，原始入口及两份适配补丁、六组结果和 GPU 缺库日志均保留。以下为之前的本机 CPU 运行记录。

## 实际运行范围

环境：SH-3SQXPV3 / WSL2 / Python 3.10.12 / PyTorch 2.14.1+cpu，无可用 CUDA 设备。CLI 尝试的完整依赖见 [launch-environment.txt](launch-environment.txt)。1104 次为参数准备/元数据检查，不是 1104 次 GEMM 执行或性能测试。五组精度在 vendor_parser 即抛 ValueError，尚未创建 tensor、scale 或进入 kernel。

INT8 执行的四组真实 Excel MNK（M=1 是原表值）：

| M | N | K | A | W | 输出 |
| --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 4096 | INT8[1,4096] | INT8[4096,4096] | BF16[1,4096] |
| 1 | 8192 | 8192 | INT8[1,8192] | INT8[8192,8192] | BF16[1,8192] |
| 1 | 1024 | 8192 | INT8[1,8192] | INT8[1024,8192] | BF16[1,1024] |
| 1 | 8192 | 1024 | INT8[1,1024] | INT8[8192,1024] | BF16[1,8192] |

W 的物理 shape 为 [N,K]，计算时转置。原始 creator 生成零值输入/权重与全 1 scale。捕获计算链为：INT8 A/W 转 BF16 → BF16 mm → 转 FP32 → 乘 A scale 和 W scale → 转 BF16。因此 compute_dtype=int8 是此接口要求，但参考实现用 BF16 模拟；不能将该路径的速度当成原生 INT8 性能，也不能从输出有限值推断完整数值正确性。

已按原文命令和新版 CLI 各尝试一次，日志见 [legacy-cli.txt](legacy-cli.txt)、[current-cli.txt](current-cli.txt)，完整命令和退出码见 [cli-attempts.json](cli-attempts.json)。旧 `--hardware_type` 不被当前 CLI 接受；新版 `--backend GPU` 因本机 CUDA 不可用，GPU 后端导入失败，未进入端到端 GPU benchmark。这与上述算子精度参数拒绝是两个不同阶段。

正确的仓库入口是 clone 仓库根地址 `https://github.com/bytedance/xpu-perf.git`，当前项目目录为 `projects/micro_perf`。在具备 GPU 环境及相应厂商实现后，命令形式为：

```bash
cd xpu-perf/projects/micro_perf
python launch.py --backend GPU --workload /path/to/quant-matmul.json
```

## latency、算力、带宽、MFU、MBU

当前基础 summary 计算 latency(us)、calc_flops_power(tflops)、mem_bw(GB/s) 等，但没有直接计算/输出 MFU、MBU 的实现。完整输出字段见固定版本 core/op.py 的 summary；“带宽”按逻辑 tensor 字节数估算，不是 profiler 实测 HBM 流量。

- 运算量：2×M×N×K。
- 算力（TFLOP/s）：2×M×N×K / latency_us / 10⁶。
- 带宽（GB/s）：io_bytes / latency_us / 10³。INT8 参考的 io_bytes 包含 A、W、A/W scale 与输出，即 MK + NK + 4M + 4N + 2MN；不包含模拟计算产生的内部转换中间量，故不能据此代表硬件真实带宽利用率。
- MFU 还需采用对应精度、设备配置及一致统计口径的峰值算力；MBU 还需峰值带宽。当前环境和基础代码不能给出这两项送测结果，也没有生成 GPU latency/TFLOP/s/GB/s 数据。

## 证据与复现

- [six-precision-cases.csv](six-precision-cases.csv)：1104 个组合、原始精度、MNK、参数检查状态、INT8 scale dtype/shape 与原始报错，保留 6192。
- [six-precision-runtime.json](six-precision-runtime.json)：四个执行记录、输入输出/scale dtype、shape、实际 scale 值、aten 调用链、逻辑字节数及环境。
- [capture_six.py](capture_six.py)：用上游解析器生成网格并抓取未修改算子的脚本；运行参数为 `--repo /path/to/xpu-perf --workload quant-matmul.json --output-dir /path/to/output`。
- 原有 [runtime-contract.json](runtime-contract.json) 另含 dtype creator 的独立抓取及 Basic BF16 路径；Basic BF16 通过不代表 quant_matmul BF16 通过。
- 精度 gate、tensor 创建与计算：[quant_matmul.py](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/projects/micro_perf/op_defs/llm_ops/quant_matmul.py)。
- 可覆盖的默认 scale/粒度：[quant_matmul.md](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/projects/micro_perf/op_defs/llm_ops/quant_matmul.md)。
- dtype 映射和 BF16 模拟计算：[core/utils.py](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/src/xpu_perf/micro_perf/core/utils.py)。
- 输出指标计算：[core/op.py](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/src/xpu_perf/micro_perf/core/op.py)。
- 标准格式：[OCP MX v1.0，表 1](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

[运行方式、editable 安装和 provider 路由核查](repository_audit_20261105/README.md)
