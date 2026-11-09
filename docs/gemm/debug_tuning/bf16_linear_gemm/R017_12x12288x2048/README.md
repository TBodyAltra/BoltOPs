# R017 · 12 × 12288 × 2048 · BF16 linear（decode）

本目录记录需求编号 R017 的调优结果。目前只有**通用版**（`generic/`）：使用所有 shape 共用的 [`_gemm_bf16_generic_kernel`](../operators/gemm_a16w16_generic.py)，没有特化 kernel，只保存配置和复现结果。配置来自动态 kernel 的 TT autotune 归档，未做额外优化。

## 需求信息

| 项 | 内容 |
| --- | --- |
| 需求编号 | R017（`in_proj_qkvz`），见 [少伯 BU 需求](../../../SHAOBO_BU_REQUIREMENTS.md#r017) |
| M × N × K | 12 × 12288 × 2048 |
| 阶段 / 频次 | decode / 1 |
| 模型 / 配置 | 需求表未填写（—） |
| 精度 | BF16 × BF16 → BF16，FP32 累加 |
| 对接接口 | [gemm_a16w16](../../../../../boltops/gemm/triton/gemm_a16w16.py)（权重布局 `[N,K]`）。本次直接测通用 kernel，不是生产接口端到端测试 |
| 需求表当前状态 | 接口 smoke 通过（尚未回填调优结果） |

## 版本

| | 通用版 `generic/` |
| --- | --- |
| kernel | [`_gemm_bf16_generic_kernel`](../operators/gemm_a16w16_generic.py)（所有 shape 共用，不在本目录内保存副本） |
| 算法 | 动态 M/N/K，单 kernel；NUM_KSPLIT=1，不做 split-K |
| 配置 | [`generic/best_config.json`](generic/best_config.json)：BM16 / BN128 / BK64，`GROUP_SIZE_M=1`，4 warps，stages 2，MLS，unroll 2，`mmac_layout_force=1` |
| 配置来源 | [`results_TT_dyn/shapes/R017_12x12288x2048_TT.json`](../autotune/results_TT_dyn/shapes/R017_12x12288x2048_TT.json) 的 `best.config`，内容完全一致；`best.labels` 只作来源描述，不是配置项 |
| 自然 grid | ceil(12/16) × ceil(12288/128) × 1 = 96 个 TG（`completed_wgs=96`，48 个活跃 CU） |
| 布局 | TT 为主；TN 直接套用同一份 TT 配置，**未独立调优，不代表 TN 最优** |
| 正确性 | PASS：TT、TN 各 2 个进程（种子 29 / 30），每轮 147456 个输出逐个与 BF16 舍入后的 FP32 参考比较（`atol=0.02, rtol=0.01`），各轮 `mismatches=0`；全输出有限、前后各 128 个哨兵不变；对未舍入 FP32 参考的平均误差满足 `≤ 1.05 × 舍入下限 + 1e-6` |
| 性能（PMD 单 die，kernel-only） | 各 10 个正式样本的中位数：<br>**TT 33.697303 µs，17.9237 TFLOPS，MMAC 利用率 4.1490%，有效带宽 1.5039 TB/s**（配置带宽达成率 83.56%；74060 cycles，范围 33.647708–33.770558 µs）；<br>**TN 33.810371 µs，17.8637 TFLOPS，MMAC 利用率 4.1351%，有效带宽 1.4988 TB/s**（配置带宽达成率 83.28%；74308.5 cycles，范围 33.713228–33.854278 µs）；<br>**理论外存带宽约 1.8 TB/s**（模型配置上限，实际生效值 1.7996485 TB/s）；**BF16 理论计算峰值参考 432 TFLOPS**（单 die 48 CU 的峰值，依据见[上层 README](../README.md#性能指标公共口径)）；本 shape 算术强度 11.9185 FLOP/B，**带宽屋顶线 21.4492 TFLOPS**。有效带宽按逻辑数据量 / 完整 kernel 耗时计算，不是实测外存流量（公共口径见[上层 README](../README.md#性能指标公共口径)）；两布局均为 58 VGPR、LDS 36864 B、0 spill |
| 复现 | `bash ../repro_generic/run.sh R017_12x12288x2048 [--layout TN]`；结果见 [`generic/runs/`](generic/runs/)，用法见[上层 README](../README.md#复现通用版) |

说明：

- 带宽与屋顶线计算口径：BF16 每元素 2 B，逻辑数据量 `D = 2 × (M×K + K×N + M×N) = 50,675,712 B`；有效带宽 `D / 完整kernel耗时`，算术强度 `2MNK / D = 11.9185 FLOP/B`，经典屋顶线 `min(432 TFLOPS, 算术强度 × 1.7996485 TB/s) = 21.4492 TFLOPS`。不计缓存复用、重复读取、事务放大和 tile padding；屋顶线是该数据访问假设下的参考，不是所有实现的严格性能上界。有效带宽使用现有 TT/TN summary 的未舍入中位耗时计算，未重测基线。
- 同配置连续读补测（2026-11-07）：256 MiB 数据集的完整 kernel 外存读吞吐为 **1.7344 TB/s**（配置上限的 96.37%）；48 MiB、与 B 同大小的数据集为 **1.5035 TB/s**。两者均取本轮两种执行配置中较优项的 3 次计时中位数，采用外存请求量核对；这是 PMD 读带宽微基准，不是实卡带宽或全局峰值证明，也不能将 GEMM 的逻辑有效带宽直接当作同口径的实际外存吞吐。方法与原始记录见[带宽补测说明](../../../../../tmp/1107/r017_bandwidth_probe/README.md)（本地 tmp 路径，清理后可能失效）。

- MMAC 利用率按 BF16 理论峰值 **432 TOPS**（浮点运算口径为 432 TFLOPS）计算：`利用率 = (2 × M × N × K / kernel耗时秒) / (432 × 10^12) × 100%`，乘加计 2 次运算。本 shape 的有效运算量为 603,979,776 FLOPs；以上用各自 summary 的未舍入 `median_us` 计算。该值是有效 GEMM 吞吐占给定峰值的比例，不是硬件计数器测得的 MMAC 指令活跃率；不额外计入 M 从 12 补齐到 16 的 tile padding 运算，峰值对应单 die 48 CU（依据见[上层 README](../README.md#性能指标公共口径)）。
- 汇总：[TT](generic/runs/TT_20261106_baseline/summary.json)、[TN](generic/runs/TN_20261106_baseline/summary.json)。单轮中位数：TT 74004 / 74093 cycles，TN 74295 / 74319 cycles；最终中位数在两轮全部 10 个正式样本上计算，不是两个单轮中位数的平均。
- 逐轮平均绝对误差 / BF16 舍入误差下限：TT 种子 29 为 0.050873640924692154 / 0.05087363347411156，种子 30 为 0.05076529085636139 / 0.05076529085636139；TN 种子 29 为 0.051008064299821854 / 0.051008060574531555，种子 30 为 0.05100562795996666 / 0.05100562050938606。
- 来源 autotune 记录为 73960 cycles / 33.651803 µs（预热 1 次 + 正式 3 次）。本次是重新测量，计时协议和测量上下文不同，没有同进程 A/B，**不把两者差异解释为加速或退化**。
- 正确性检查覆盖每轮最后一次 dispatch 的完整输出，不是逐个检查全部 10 个计时 dispatch；仅覆盖两个随机种子、固定 shape、连续存储的 A 和两种 B 布局，未启用输入侧 NaN padding，不据此声称任意 stride 或特殊数值场景已验证。
- 计时均为 PMD `system.gcSimCycles` 经 `gcFrequency` 换算，只含 kernel，不含 JIT 编译、分配、主机拷贝和生产接口开销；PMD 为模拟器，不代表真机性能。

## 测量条件

| 项 | 内容 |
| --- | --- |
| 执行环境 | PMD，`GPU_CHIP=sb`（gfx946），`PMD_PATH=/opt/rocm/pmd`；日志显示模拟配置为**单 die、48 CU**（`--num-cu=48`、`single die dispatch`），每 WG 256 线程。summary 未单独记录 die/CU 数，依据是各轮 `round_*/worker.log`（见 `tmp/1106/r017_baseline`，不进本目录）；详见[上层 README](../README.md#环境与版本) |
| 布局 | TT：A 行主序，B 为 `[K,N]` 行主序。TN：W 为 `[N,K]`，K 连续 |
| 计时 | 2 个进程（种子 29 / 30），各预热 1 次、正式 5 次，取 10 个样本的中位数；TT、TN 并发执行，作业墙钟约 1247.5 s / 1253.0 s，不是单次 kernel 延迟 |
| 软件 | Triton `release/3.6.x` 提交 `e58481668e29fda141014f1d4b1f0f7bd82afd41`，`tracked_changes=0`；`libtriton.so` SHA256 `b264c76926a43ad3767c23fe0140f422573ea9f55b04f1828b46389f2f8d5cf6`；clang-18 `/opt/dtk/aillvm/bin/clang-18`，`ld.lld` 同目录。两份通用 kernel 的 SHA256 均为 `e108b0bb94c719409fc0a8b387e295f1d39bf1e02d364494a1d92c5e361313fb` |
| BoltOPs 版本 | 测量时的 BoltOPs 目录不是 Git 仓库，分支、提交和改动统计在 summary 中记为 `unknown`，是版本溯源局限，不是测试失败 |
| 与 R015 的差异 | 计时与正确性协议对齐 [R015](../R015_12x2048x4096/README.md)，但 Triton 提交和 `libtriton.so` 与 R015 记录不同，不声称完全相同的软件环境 |

[调优记录](../../TUNING.md) 要求每次性能测量保存环境 manifest。`summary.json` 已包含 Triton 提交、`libtriton.so` 与 kernel 源码的 SHA256；BoltOPs 提交补充后，这些数据才能用于性能排名；die/CU 数已核对（见上表）。

日志中出现过 steer 配置回退、缺少 pydot、`amdgpu_get_auth` 失败、矩阵读取测试和 pending cacheline 等提示；四轮均未导致非零返回码、正确性失败或缺失计时样本。

## 复现

`--dispatches 6` 即 1 次预热 + 5 次计时；输出写入新的 `--out` 目录，不覆盖已有测量：

```bash
REPO=/public/home/duyubo/bolt/github/BoltOPs
cd "$REPO/docs/gemm/debug_tuning/bf16_linear_gemm/repro_generic"
bash run.sh R017_12x12288x2048 --out /abs/new/TT_dir                   # TT
bash run.sh R017_12x12288x2048 --layout TN --out /abs/new/TN_dir       # TN
```

## 实际路径说明

结果先在 `tmp/1106/r017_baseline` 生成，再复制到本目录。两份 `summary.json` 原样保留当时的实际测量路径（`config_file` 与 `results_directory` 仍指向 tmp），没有改写。`round_*/` 的原始输出（`case.json`、`check.json`、日志、汇编等）只保留在 tmp，不进本目录；运行器默认清理逐 dispatch 的模拟统计目录，不能声称所有原始 `stats.txt` 均已归档。tmp 不是长期归档位置，清理后 summary 中的历史路径可能失效。

## 待办

1. TN 仅套用 TT 配置，如需 TN 的最优配置须单独调优。
2. 在同一次运行里 A/B 对比 autotune 来源记录与本次配置。
3. 尚无特化版；如有，随该 shape 归档为 `specialized_<特征>/`。

## 目录

```text
R017_12x12288x2048/
├── README.md
└── generic/
    ├── best_config.json
    └── runs/<layout>_<时间>/summary.json   # repro_generic 的复现结果（round_*/ 被忽略）
```
