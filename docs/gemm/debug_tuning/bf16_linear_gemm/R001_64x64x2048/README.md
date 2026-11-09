# R001 · 64 × 64 × 2048 · BF16 linear（decode）

本目录记录需求编号 R001 的调优结果。目前只有**通用版**（`generic/`）：使用所有 shape 共用的 [`_gemm_bf16_generic_kernel`](../operators/gemm_a16w16_generic.py)，没有特化 kernel，只保存配置和复现结果。配置来自动态 kernel 的 TT autotune 归档，未做额外优化。

## 需求信息

| 项 | 内容 |
| --- | --- |
| 需求编号 | R001（`in_proj_ba`），见 [少伯 BU 需求](../../../SHAOBO_BU_REQUIREMENTS.md#r001) |
| M × N × K | 64 × 64 × 2048 |
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
| 配置 | [`generic/best_config.json`](generic/best_config.json)：BM16 / BN32 / BK128，`GROUP_SIZE_M=1`，2 warps，stages 2，MLS，unroll 4，`mmac_layout_force=-1`（自动），`enable_v_mmac_cluster=1` |
| 配置来源 | [`R001_64x64x2048_TT.json`](../autotune/results_TT_dyn/shapes/R001_64x64x2048_TT.json) 的 `best.config`，内容完全一致；`best.labels` 只作来源描述，不是配置项（其中的 `sched_v2=on`、`regpress=on` 等标签在 `config` 里没有对应的键） |
| 自然 grid | 64/16 × 64/32 × 1 = 8 个 TG（`completed_wgs=8`，8 个活跃 CU） |
| 布局 | TT 为主；TN 直接套用同一份 TT 配置，**未独立调优，不代表 TN 最优** |
| 正确性 | PASS：TT、TN 各 2 个进程（种子 29 / 30），每轮 4096 个输出逐个与 BF16 舍入后的 FP32 参考比较（`atol=0.02, rtol=0.01`），各轮 `mismatches=0`；全输出有限、前后各 128 个哨兵不变；对未舍入 FP32 参考的平均误差满足 `≤ 1.05 × 舍入下限 + 1e-6` |
| 性能（PMD 单 die，kernel-only） | 各 10 个正式样本的中位数：<br>**TT 7.486571 µs，2.2410 TFLOPS，MMAC 利用率 0.5187%，有效带宽 0.0711 TB/s**（配置带宽达成率 3.95%；16454 cycles，范围 7.460636–7.676306 µs）；资源 203 VGPR，LDS 49152 B，0 spill；<br>**TN 6.733546 µs，2.4916 TFLOPS，MMAC 利用率 0.5768%，有效带宽 0.0791 TB/s**（配置带宽达成率 4.39%；14799 cycles，范围 6.672121–6.899166 µs）；资源 134 VGPR，LDS 49152 B，0 spill；<br>**理论外存带宽约 1.8 TB/s**（模型配置上限，实际生效值 1.7996485 TB/s）；**BF16 理论计算峰值参考 432 TFLOPS**（单 die 48 CU 的峰值，依据见[上层 README](../README.md#性能指标公共口径)）；本 shape 算术强度 31.5077 FLOP/B，**带宽屋顶线 56.7028 TFLOPS**（屋顶线耗时 0.296 µs）。实测耗时约为带宽下限的 25 倍，数据量仅约 0.5 MB，耗时不是带宽受限。有效带宽按逻辑数据量 / 完整 kernel 耗时计算，不是实测外存流量（公共口径见[上层 README](../README.md#性能指标公共口径)） |
| 复现 | `bash ../repro_generic/run.sh R001_64x64x2048 [--layout TN]`；结果见 [`generic/runs/`](generic/runs/)，用法见[上层 README](../README.md#复现通用版) |

说明：

- 带宽与屋顶线计算口径：BF16 每元素 2 B，逻辑数据量 `D = 2 × (M×K + K×N + M×N) = 532,480 B`；有效带宽 `D / 完整 kernel 耗时`，算术强度 `2MNK / D = 31.5077 FLOP/B`，经典屋顶线 `min(432 TFLOPS, 算术强度 × 1.7996485 TB/s) = 56.7028 TFLOPS`。不计缓存复用、重复读取、事务放大和 tile padding；屋顶线是该数据访问假设下的参考，不是所有实现的严格性能上界。有效带宽使用各 summary 的未舍入 `median_us` 计算。该 shape 的数据量太小，按带宽屋顶线只需约 0.296 µs，实测 6.7–7.5 µs 说明耗时主要来自固定开销、延迟和 K 方向串行（推断，未做剖析），因此带宽达成率很低不表示带宽还有 20 倍以上的提升空间，这个指标对本 shape 的参考价值有限。
- MMAC 利用率按 BF16 理论峰值 **432 TOPS**（浮点运算口径为 432 TFLOPS）计算：`利用率 = (2 × M × N × K / kernel耗时秒) / (432 × 10^12) × 100%`，乘加计 2 次运算。本 shape 的有效运算量为 16,777,216 FLOPs；以上用各自 summary 的未舍入 `median_us` 计算。该值是有效 GEMM 吞吐占给定峰值的比例，不是硬件计数器测得的 MMAC 指令活跃率；不额外计入 tile padding 运算。本 shape 只有 8 个 TG，只占用 48 个 CU 中的 8 个，利用率低主要由此和固定开销造成（推断）。
- 汇总：[TT](generic/runs/TT_20261107_074134/summary.json)、[TN](generic/runs/TN_20261107_074134/summary.json)。单轮中位数：TT 16472 / 16420 cycles，TN 14799 / 14799 cycles；最终中位数在两轮全部 10 个正式样本上计算，不是两个单轮中位数的平均。逐轮平均绝对误差 / BF16 舍入误差下限：TT 种子 29 为 0.05107378959655762 / 0.051073744893074036，种子 30 为 0.051363371312618256 / 0.051363371312618256；TN 种子 29 为 0.05102463811635971 / 0.05102463811635971，种子 30 为 0.05202776566147804 / 0.052027761936187744。
- TT 单次计时周期呈两档（约 16400–16470 与约 16850–16870 cycles），原因本次未剖析；中位数取该分布的中位，不代表其他分布下的结果。
- TN 比 TT 快约 10%（14799 vs 16454 cycles），VGPR 也更少（134 vs 203）。两布局用同一份配置，TN 并非独立调优，**不要据此推断 TN 配置更优**，也没有做同进程 A/B。
- 来源 autotune 记录（TT）：`best.val` 为 16424 cycles / 7.472921 µs（2.2451 TFLOPS），`best.proxy` 为 16631.5 cycles / 7.567333 µs；autotune 搜索了 18 个 tile、共 211 次评估，作业墙钟 3957 s，完成于 2026-11-04 09:27:46。本次 TT 复现 16454 cycles，比 autotune 的 `val` 高约 0.18%；两者计时协议和软件栈不同（autotune 记录没有环境信息，按完成时间归入上层 README “环境时间线”的第一段，属于推断；复现属于 11-05 12:44 之后的第二段），没有同进程 A/B，**不把差异解释为加速或退化**。
- 归档的 `finalists` 另有 BM16 / BN32 / BK64（`mmac_layout_force=3`，val 17496 cycles / 7.960681 µs）和 BM32 / BN32 / BK128（val 18121 cycles / 8.245056 µs）两个候选，均比最佳配置慢；本次未重测。
- `gcSimCycles` 含模拟器启动/收尾的常数周期（R015 的分析中约 7938 周期，本 shape 未单独测量），因此 16454 cycles 中该常数占比可能较大。
- 正确性检查覆盖每轮最后一次 dispatch 的完整输出，不是逐个检查全部 10 个计时 dispatch；仅覆盖两个随机种子、固定 shape、连续存储的 A 和两种 B 布局，不据此声称任意 stride 或特殊数值场景已验证。
- 计时均为 PMD `system.gcSimCycles` 经 `gcFrequency` 换算，只含 kernel，不含 JIT 编译、分配、主机拷贝和生产接口开销；PMD 为模拟器，不代表真机性能。

## 测量条件

| 项 | 内容 |
| --- | --- |
| 执行环境 | PMD，`GPU_CHIP=sb`（gfx946），`PMD_PATH=/opt/rocm/pmd`；模拟配置为**单 die、48 CU**（PMD 参数 `--num-cu=48`，日志 `single die dispatch`），GPU 频率 2.197802 GHz（`--gpu-clock=2.2GHz`，约 455 ps/周期），`--umc-bw=1.8TB/s`。依据是本次各轮 `round_*/worker.log`（本地文件，不提交；`summary.json` 未单独记录这些项）；每轮 `completed_wgs=8`、`active_cus=8`，详见[上层 README](../README.md#环境与版本) |
| 布局 | TT：A 行主序，B 为 `[K,N]` 行主序。TN：W 为 `[N,K]`，K 连续 |
| 计时 | 2 个进程（种子 29 / 30），各预热 1 次、正式 5 次，取 10 个样本的中位数；TT、TN 并发执行 |
| 软件 | Triton `release/3.6.x` 提交 `e58481668e29fda141014f1d4b1f0f7bd82afd41`，`tracked_changes=0`；`libtriton.so` SHA256 `b264c76926a43ad3767c23fe0140f422573ea9f55b04f1828b46389f2f8d5cf6`（构建于 2026-11-06 01:22:45）；clang-18 `/opt/dtk/aillvm/bin/clang-18`，`ld.lld` 同目录。两份通用 kernel 的 SHA256 均为 `e108b0bb94c719409fc0a8b387e295f1d39bf1e02d364494a1d92c5e361313fb`，属于上层 README “环境时间线”中 11-05 12:44 之后的 Triton 环境 |
| BoltOPs 版本 | 提交 `4a03e47944c56734d1007e4f59b20ca75ba6c9f8`，分支 `gemm`，`tracked_changes=5`（工作区已有文档改动，不是干净工作区） |

[调优记录](../../TUNING.md) 要求每次性能测量保存环境 manifest。`summary.json` 已包含 Triton 提交、`libtriton.so` 与 kernel 源码的 SHA256 以及 BoltOPs 提交。

## 复现

`--dispatches 6` 即 1 次预热 + 5 次计时；不指定 `--out` 时结果写入 `generic/runs/<layout>_<时间>/`：

```bash
cd ../repro_generic
bash run.sh R001_64x64x2048                  # TT
bash run.sh R001_64x64x2048 --layout TN      # TN（gemm_a16w16 接口布局）
```

## 本地 IR、汇编与 Perf

`generic/runs/TT_ir_asm_perf/` 与 `generic/runs/TN_ir_asm_perf/` 分别保存当前配置的 TT、TN 本地产物，布局沿用 R004 和 `repro_generic/keep_ir_asm_perf.sh` 的约定：

- `kernel.ttir`、`kernel.ttgir`、`kernel.llir`：三个编译阶段的 IR；`kernel.amdgcn`：对应汇编。
- `g/0/0/`：预热 dispatch；`g/0/1/` 至 `g/0/5/`：五次正式 dispatch。各目录保留二进制 `*_python3.10.perf`、文本统计 `stats_xcd0.perf`、`.kinfo`，以及 `stats.txt` 和 `ttrace_model__*.out` 追踪文件。

这些文件于 2026-11-07 使用 `repro_generic/capture.sh R001_64x64x2048 generic TT`（TN 对应最后一个参数改为 `TN`）独立补采，不是上文 `TT_20261107_074134` / `TN_20261107_074134` 两轮复现的原始文件，也不替换已有 `summary.json` 或性能结论。每种布局各执行一个进程，seed 29，预热 1 次、正式 5 次，启用 PMD `TT` 和 `Perf` 调试标志；最后一次输出全量正确性检查 PASS，每次 dispatch 完成 8 个工作组。两份完整采集均通过 `repro_generic/validate_capture.py` 的配置、源码及产物哈希、dispatch 数量和统计一致性检查。

完整采集保留在以下本地目录，其中 `manifest.json` 记录环境、来源及文件 SHA256，`check.json` 记录正确性结果：

- TT：`/public/home/duyubo/bolt/github/BoltOPs/tmp/capture_R001_64x64x2048_generic_TT.8BM3H8/`
- TN：`/public/home/duyubo/bolt/github/BoltOPs/tmp/capture_R001_64x64x2048_generic_TN.ITbqxG/`

IR、汇编与 Perf 目录遵循上层 `.gitignore` 的现有规则，仅供本地查看，不提交。

## 待办

1. TN 仅套用 TT 配置，如需 TN 的最优配置须单独调优。
2. 在同一次运行里 A/B 对比 autotune 来源记录与本次配置。
3. 尚无特化版；如有，随该 shape 归档为 `specialized_<特征>/`。

## 目录

```text
R001_64x64x2048/
├── README.md
└── generic/
    ├── best_config.json
    └── runs/
        ├── TT_20261107_074134/summary.json   # 既有 TT 两轮复现结果
        ├── TN_20261107_074134/summary.json   # 既有 TN 两轮复现结果
        ├── TT_ir_asm_perf/                 # 独立补采，本地文件，不提交
        │   ├── kernel.ttir / kernel.ttgir / kernel.llir / kernel.amdgcn
        │   └── g/0/{0,1,2,3,4,5}/          # 1 次预热 + 5 次正式 dispatch
        └── TN_ir_asm_perf/                 # 同上，TN 布局
            ├── kernel.ttir / kernel.ttgir / kernel.llir / kernel.amdgcn
            └── g/0/{0,1,2,3,4,5}/
```
