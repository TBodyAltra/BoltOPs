# R015 · 12 × 2048 × 4096 · BF16 linear（decode）

本目录记录需求编号 R015（旧 G261）的调优结果。目前只有**通用版**（`generic/`）：使用所有 shape 共用的 [`_gemm_bf16_generic_kernel`](../operators/gemm_a16w16_generic.py)，没有特化 kernel，只保存配置和复现结果。配置由 autotune 最佳配置手工微调得到（见下文“配置来源与对照”）。

## 需求信息

| 项 | 内容 |
| --- | --- |
| 需求编号 | R015（旧 G261；`o_proj` / `out_proj`），见 [少伯 BU 需求](../../../SHAOBO_BU_REQUIREMENTS.md#r015) |
| M × N × K | 12 × 2048 × 4096 |
| 阶段 / 频次 | decode / 2 |
| 模型 / 配置 | 需求表未填写（—） |
| 精度 | BF16 × BF16 → BF16，FP32 累加 |
| 对接接口 | [gemm_a16w16](../../../../../boltops/gemm/triton/gemm_a16w16.py)（权重布局 `[N,K]`）。本次直接测通用 kernel，不是生产接口端到端测试 |
| 需求表当前状态 | 接口 smoke 通过（尚未回填调优结果） |

## 版本

| | 通用版 `generic/` |
| --- | --- |
| kernel | [`_gemm_bf16_generic_kernel`](../operators/gemm_a16w16_generic.py)（所有 shape 共用，不在本目录内保存副本） |
| 算法 | 动态 M/N/K，单 kernel；NUM_KSPLIT=1，不做 split-K |
| 配置 | [`generic/best_config.json`](generic/best_config.json)：BM16 / BN32 / BK128，`GROUP_SIZE_M=1`，2 warps，**stages 3**，MLS，**unroll 1**，`mmac_layout_force=3`，`enable_v_mmac_cluster=1`，**关闭 max-ILP 调度**（`amdgpu_enable_max_ilp_scheduling_strategy=false`） |
| 配置来源 | autotune 最佳配置（[`R015_12x2048x4096_TT.json`](../autotune/results_TT_dyn/shapes/R015_12x2048x4096_TT.json) 对应的 BM16 / BN32 / BK128、2 warps、2 级流水、unroll 4、max-ILP 开）经手工扫描改为 3 级流水、unroll 1、关闭 max-ILP；与归档的 `best.config` **不相同** |
| 自然 grid | ceil(12/16) × 2048/32 × 1 = 64 个 TG |
| 布局 | TT 为主；TN 直接套用同一份 TT 配置，**未独立调优，不代表 TN 最优** |
| 正确性 | PASS：TT、TN 各 2 个进程，全部 24576 个输出元素与 CPU FP32 参考（舍入到 BF16 作为 gold）比较，`|got − gold| ≤ 0.02 + 0.01·|gold|`，0 个超限；平均误差 `|got − ref|` 不超过 BF16 舍入误差下限的 1.05 倍；输出全为有限值；输出前后各 128 个哨兵值未被改写 |
| 性能（PMD 单 die，kernel-only） | 各 10 个正式样本的中位数：<br>**TT 14.472869 µs，13.9106 TFLOPS，MMAC 利用率 3.2201%，有效带宽 1.1694 TB/s**（配置带宽达成率 64.98%；31808.5 cycles，范围 14.463–14.506 µs）；<br>**TN 14.512681 µs，13.8725 TFLOPS，MMAC 利用率 3.2112%，有效带宽 1.1662 TB/s**（配置带宽达成率 64.80%；31896 cycles，范围 14.465–14.595 µs）；<br>**理论外存带宽约 1.8 TB/s**（模型配置上限，实际生效值 1.7996485 TB/s）；**BF16 理论计算峰值参考 432 TFLOPS**（单 die 48 CU 的峰值，依据见[上层 README](../README.md#性能指标公共口径)）；本 shape 算术强度 11.8955 FLOP/B，**带宽屋顶线 21.4076 TFLOPS**（屋顶线耗时 9.404 µs）。有效带宽按逻辑数据量 / 完整 kernel 耗时计算，不是实测外存流量（公共口径见[上层 README](../README.md#性能指标公共口径)）；资源：TT 58 VGPR，TN 38 VGPR，LDS 均为 36864 B，0 spill |
| 复现 | `bash ../repro_generic/run.sh R015_12x2048x4096 [--layout TN]`；结果见 [`generic/runs/`](generic/runs/)，用法见[上层 README](../README.md#复现通用版) |

说明：

- 带宽与屋顶线计算口径：BF16 每元素 2 B，逻辑数据量 `D = 2 × (M×K + K×N + M×N) = 16,924,672 B`；有效带宽 `D / 完整 kernel 耗时`，算术强度 `2MNK / D = 11.8955 FLOP/B`，经典屋顶线 `min(432 TFLOPS, 算术强度 × 1.7996485 TB/s) = 21.4076 TFLOPS`（耗时 9.404 µs）。不计缓存复用、重复读取、事务放大和 tile padding；屋顶线是该数据访问假设下的参考，不是所有实现的严格性能上界。有效带宽使用各 summary 的未舍入 `median_us` 计算。
- MMAC 利用率按 BF16 理论峰值 **432 TOPS**（浮点运算口径为 432 TFLOPS）计算：`利用率 = (2 × M × N × K / kernel耗时秒) / (432 × 10^12) × 100%`，乘加计 2 次运算。本 shape 的有效运算量为 201,326,592 FLOPs；以上用各自 summary 的未舍入 `median_us` 计算。该值是有效 GEMM 吞吐占给定峰值的比例，不是硬件计数器测得的 MMAC 指令活跃率；不额外计入 M 从 12 补齐到 16 的 tile padding 运算，峰值对应单 die 48 CU（依据见[上层 README](../README.md#性能指标公共口径)）。
- 汇总：[TT](generic/runs/TT_20261105_062328/summary.json)、[TN](generic/runs/TN_20261105_062328/summary.json)。`summary.json` 中的 `shape_id` 为 `G261`，路径仍指向当时的 `G261_12x2048x4096` 目录，属测量时原值，未改写。
- 下文“配置来源与对照”中 B 矩阵 20605 周期（约 9.375 µs）的下限与屋顶线基本一致；`gcSimCycles` 另含约 7938 周期的模拟器启动/收尾常数，小 shape 的有效带宽会因此偏低。
- 计时均为 PMD `system.gcSimCycles` 经 `gcFrequency` 换算，只含 kernel，不含 JIT 编译、分配、主机拷贝和生产接口开销；PMD 为模拟器，不代表真机性能。

## 配置来源与对照

- 配置是在 12×2048×4096 TT 上手工扫描得到的：在 autotune 作业 `autotune_20261104_dyn` 的最佳配置（BM16 / BN32 / BK128、2 warps、2 级流水、unroll 4、max-ILP 开）基础上，改为 3 级流水、unroll 1、关闭 max-ILP。
- 该 autotune 最佳配置在 autotune 口径下是 33776 cycles（15.37 µs，预热 1 次 + 正式 3 次）。本目录的 31808.5 cycles 来自不同的计时协议（见上，10 个样本），**没有在同一次运行里 A/B 对照**，因此约 5.8% 的差距只能作参考，不能当作严格的加速比。
- 小 K 的预估模型和各配置扫描见 `xk/BoltOPs/tmp/1105/12x2048x4096_TT_mmac_bubble/SCALING_MODEL.md`。该实验的口径是 `rotate_b`（每次 dispatch 的 B 都从 HBM 冷读），在该口径下 s3u1 + 关 max-ILP 的 K=4096 为 32364 cycles。
- 瓶颈：K=4096 时 B 矩阵 16.78 MB，按 819 B/周期的 HBM 峰值至少需要 20605 周期。稳态循环已达到该下限的约 96%；`gcSimCycles` 里另有约 7938 周期是模拟器启动/收尾的常数。因此继续在配置上调优的空间很小。
- 更多 stage（4、5）、`num_warps=1`、BK64 / BK256、unroll 2 / 4 都在同一扫描中试过，均比本配置慢。

## 测量条件

| 项 | 内容 |
| --- | --- |
| 执行环境 | PMD，`GPU_CHIP=sb`（gfx946），`PMD_PATH=/opt/rocm/pmd`；**单 die、48 CU**。summary 未记录 die/CU 数，依据是本地 `generic/runs/TT_ir_asm_perf/g/` 下的 `stats.txt`：含 `CUs00` – `CUs47` 共 48 个 CU，且只输出 `xcd0`；[调优记录](../../TUNING.md) 规定每 die 48 CU，因此是 1 die。详见[上层 README](../README.md#环境与版本) |
| 布局 | TT：A 行主序，B 为 `[K,N]` 行主序。TN：W 为 `[N,K]`，K 连续 |
| 计时 | 2 个进程，各预热 1 次、正式 5 次，取 10 个样本的中位数 |
| 软件 | Triton `release/3.6.x` 提交 `cdf6827b53b1a59104da41a1bfc7736134b13a2e`，`tracked_changes=0`；`libtriton.so` SHA256 `919b51f14bcf5e177dc710922daa0b7e8d9758e88ede0819680b4d8f7e42caa7`；clang-18 `/opt/dtk/aillvm/bin/clang-18`，`ld.lld` 同目录。两份通用 kernel 的 SHA256 均为 `e108b0bb94c719409fc0a8b387e295f1d39bf1e02d364494a1d92c5e361313fb` |
| BoltOPs 版本 | 提交 `673ede654b7cda290f68ce98665fb6f1dcb1efd8`，分支 `duyubo/bf16-gemm-tuning`，`tracked_changes=7`（不是干净工作区）；**该提交已不在当前仓库**（分支 rebase 后丢失） |

[调优记录](../../TUNING.md) 要求每次性能测量保存环境 manifest。`summary.json` 已包含 Triton 提交、`libtriton.so` 与 kernel 源码的 SHA256；die/CU 数已核对（见上表）。

## 复现

`--dispatches 6` 即 1 次预热 + 5 次计时；不指定 `--out` 时结果写入 `generic/runs/<layout>_<时间>/`：

```bash
cd ../repro_generic
bash run.sh R015_12x2048x4096                 # TT
bash run.sh R015_12x2048x4096 --layout TN     # TN（gemm_a16w16 接口布局）
```

## 待办

1. 在同一次运行里 A/B 对比 autotune 最佳配置与本配置。
2. 本配置只在 TT 上调过，TN 的结果是直接套用同一份配置。
3. 尚无特化版；如有，随该 shape 归档为 `specialized_<特征>/`。

## 目录

```text
R015_12x2048x4096/
├── README.md
└── generic/
    ├── best_config.json
    └── runs/<layout>_<时间>/summary.json   # repro_generic 的复现结果（round_*/ 被忽略）
```
