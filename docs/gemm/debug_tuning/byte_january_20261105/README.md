# 1 月字节送测 GEMM 需求来源与核对

当前需求默认值与海光 fork 的实现信息见 [2026-11-06 补充口径](hygon_fork_20261106/README.md)。以下 CPU/PMD 记录按原采集版本保留。

来源：`GPU单体测试用例 v2.1.xlsx`，工作表 `测试用例`，物理行 72–97（含端点）。提取日期：2026-11-05。文件 SHA256、逐行 B/C/D/E 原文与链接见 [source.json](source.json)。

- 纳入第 72 行 `GPU_Micro_001` Basic GEMM，以及第 86 行 `GPU_Micro_015` quant_matmul。
- 按用户要求继续排除 MoE：第 87–89、93–97 行。第 93 行虽名为 group_matmul，但本表具有 expert_num / topk 参数，属本次排除范围。
- 第 73–85、90–92 行是归约、归一化、通信、搬运或 attention 等，不拆成独立 GEMM 需求。
- 优先级：12 月 BU Qwen 三个配置保持 P0；本批 1 月字节送测为 P1；原后续支持改为 P2。

## 网格核对

| 来源 | 参数化需求组 | M 数量 | 精度数 | K/N 数 | 展开测试数 |
| --- | --- | --- | --- | --- | --- |
| 第 72 行 Basic 必测 | 12 | 14 | 3 | 4 | 168 |
| 第 72 行 Basic FP4（如有） | 4 | 14 | 1 | 4 | 56 |
| 第 86 行 quant_matmul | 24 | 46 | 6 | 4 | 1104 |
| 合计 | 40 | — | — | — | 1328（必列 1272，条件项 56） |

主文档一行固定算子、精度组合与 K/N，M 取对应完整列表，不按 M 再分配编号；所有展开 MNK 见 [expanded-cases.csv](expanded-cases.csv)。`source_row` 可回查原文，`conditional_FP4` 仅对第 72 行 FP4 为 true。第 86 行普通 FP4 与 MXFP4 均按原文列出，不视为“如有”。`6192` 原样保留。KN 是 K、N，非对称的两组未互换。

## 仓库核对

固定仓库版本：[`bytedance/xpu-perf@71a12101509635421d02c3d28c9202951ded7745`](https://github.com/bytedance/xpu-perf/tree/71a12101509635421d02c3d28c9202951ded7745/projects/micro_perf)。检查文件的 SHA256 见 source.json。

- Excel 的旧 `micro_perf` 路径已迁到 `projects/micro_perf`。当前 launch.py 使用 `--backend` / `--workload`，不能直接照抄 Excel 的 `--hardware_type`。
- Basic 源码默认路径为 `op_defs/basic_ops/tensor_gemm_ops.py`，A[M,K]、B[K,N]。基础实现只接受 float32/tfloat32/float16/bfloat16；Excel 要求的 INT8、FP8、FP4 需供应商实现。默认 Basic workload 的 M、dtype 列表也与 Excel 不同。
- quant_matmul 源码 `op_defs/llm_ops/quant_matmul.py` 给出 M=num_tokens//sp_size、K=hidden_size、N=new_hidden_size。`expert_weight` 为二维 W[N,K]，无 expert_num / topk / 路由，因此保留为 dense 算子。基础实现仅接受 INT8/INT8/INT8→BF16，并调用 fake_quant_gemm，不能作为真实原生 INT8 性能实现或已验收证据。
- 默认 `workloads/llm/vendor_test/quant_matmul.json` 只有 INT8、FP8、MXFP8、MXFP4 四组，缺 Excel 的 BF16、普通 FP4 两组。以 Excel 六组为准。
- INT8 基础参考的 scale 为 FP32 per-token / per-channel，创建为全 1。项目 quant_matmul.md 提供可由 vendor 扩展的默认参数说明；这些参考约定不能替代本次 Excel 未提供的 scale/格式/动态性要求，更不套用其他 BU 模型的 scale。需要送测实现明确这些接口。

同目录 `basic-required.json`、`basic-fp4-optional.json`、`quant-matmul.json` 是按 Excel 生成的候选 workload 参数网格，非当前基础后端全精度可执行性承诺。对接厂商实现后，可从固定版本项目环境使用 `python launch.py --backend <后端名> --workload <文件路径>` 加载；先按该后端说明准备环境与实现。

验收原文：性能满足厂商宣称值，差异不超过 10%；无软硬件异常。Basic 输出各 shape 算力；quant_matmul 输出 latency、算力、带宽、MFU、MBU。宣称值基线及具体指标计算口径需随送测结果提供。

已运行 CPU 参考路径，并补做 [PMD launch.py 验证](pmd_launch_20261105/README.md)：五组参数拒绝，INT8 受 gfx946 BLAS 缺库阻塞，未新增 GPU 正确性或性能 PASS。

六组 quant_matmul 全网格入口检查、四个 INT8 MNK 的执行结果和指标核对见 [SIX_PRECISION_DETAILS.md](SIX_PRECISION_DETAILS.md)。

## CPU 代码抓取（2026-11-05）

固定版本仍为上面的 71a1210。环境：WSL2 主机 SH-3SQXPV3，Python 3.10.12、PyTorch 2.14.1+cpu；依赖锁定见 [capture-environment.txt](capture-environment.txt)。[capture_contract.py](capture_contract.py) 仅提供 CPU device 选择器，直接实例化未修改的 GemmOp / QuantMatmulOp，调用其原始 tensor creator 和 core_run，并用 TorchDispatchMode 记录真实计算链；没有跳过 parser，也没有替代厂商实现。这不是 launch.py 端到端送测。

- 40 组 dtype/K/N 入口中，Basic BF16 接受 4 组，quant_matmul INT8 接受 4 组，其余 32 组抛出原始异常。全部准备参数、tensor 元数据和异常见 [runtime-contract.json](runtime-contract.json)。
- 两条接受的路径分别执行 M=4、N=4096、K=4096。Basic A/B/输出均为 BF16，无 scale。
- quant_matmul 的实际输入 A[4,4096]、W[4096,4096] 为 INT8；A scale[4]、B scale[4096] 为 FP32，均全 1。输入与权重由上游 creator 创建为零，故本次有限值输出不能当成非平凡数值正确性测试。
- 计算链抓到 INT8 转 BF16 → BF16 mm → FP32 乘 A/B scale → BF16 输出。这里只确认 tensor 的精度，不能据此判定 GEMM 内部累加器精度或原生 INT8 硬件支持。
- A scale 按 token 共享该行的整个 K；B scale 按输出通道共享整个 K。按逻辑 A[M,K]、B[K,N] 表达，scale 共享 block 分别为 1×K、K×1。源码和运行均无 K 分块或动态 scale 计算，不能补成 K32/K128。也没有 kernel 的 BLOCK_M/N/K 配置可抓取，实际 matmul 下放 PyTorch。
- 独立调用上游 dtype 映射和默认 creator：float8、mxfloat8 创建为 torch.float8_e4m3fn；mxfloat4 映射 torch.float4_e2m1fn_x2，但 creator 实际返回 INT8 存储；普通 float4 抛 KeyError。仅此映射/存储结果不证明 MX 数值编码、packing 布局或 GEMM 可运行。
- MX 路径在 parser 即被拒绝，未创建 scale。工具虽注册 float8_e8m0 类型，但本次 GEMM 路径没有使用它，不能将其写成已确认的 MX scale 精度。标准 MX 的 scale dtype=E8M0、block=32 可以确定；此代码尚未执行它们。动态性、矩阵分组轴和实际布局仍需厂商实现确定。

复现时 checkout 固定版本，并包含 `src/` 与 `projects/micro_perf/`，使用锁定环境运行：

```bash
python capture_contract.py --repo /path/to/xpu-perf \
  --source source.json --output runtime-contract.json
```

这里只抓取两个代表性 MNK 的计算路径，未将全部 1328 个组合跑成 GPU 正确性或性能结果。报告包含环境、脚本/输入快照/上游源码 SHA256、输入输出 dtype/shape/stride、scale 值和完整 aten 调用链。

## 复现

`source.json` 保留此工作簿行区间的原始内容与规范化参数。主生成器调用 [byte_january_view.py](../requirements_audit_20261104/byte_january_view.py) 从此快照生成表格、网格与校验统计；运行方式见 [少伯审计记录](../requirements_audit_20261104/README.md)。完整性统计见 [verification.json](verification.json)。

[运行方式、editable 安装和 provider 路由核查](repository_audit_20261105/README.md)
