# G256 · 12 × 64 × 2048 · BF16 linear（decode）

本目录记录需求编号 G256 的调优结果。同一个 shape 保留两版实现：**通用版**（`generic/`）与**特化版**（`specialized_fused_splitk/`）。两版各自带有自己的 `best_config.json`，配置只对同目录的 kernel 有意义，不能互相套用。

## 需求信息

| 项 | 内容 |
| --- | --- |
| 需求编号 | G256，见 [少伯 BU 需求 · 表 3](../../../SHAOBO_BU_REQUIREMENTS.md#table-3) |
| M × N × K | 12 × 64 × 2048 |
| 阶段 / 频次 | decode / 1 |
| 模型 / 配置 | C018：Qwen3.5-35B-A3B / MXFP4 [128k/1k/s150] |
| 精度 | BF16 × BF16 → BF16，FP32 累加 |
| 对接接口 | [gemm_a16w16](../../../../../boltops/gemm/triton/gemm_a16w16.py)（权重布局 `[N,K]`） |
| 需求表当前状态 | 接口 smoke 通过（尚未回填调优结果） |

目录名使用 G 编号加 shape。C 编号是模型配置的组合编号（例如 C019 = C013 + C018），模型配置变化时可能改变，因此只写在本表中，不进入目录名。

## 两个版本

| | 通用版 `generic/` | 特化版 `specialized_fused_splitk/` |
| --- | --- | --- |
| kernel | [`_gemm_bf16_generic_kernel`](../operators/gemm_a16w16_generic.py)（所有 shape 共用，不在本目录内保存副本） | [`fused_kernel.py`](specialized_fused_splitk/fused_kernel.py)（随本目录归档） |
| 算法 | 动态 M/N/K，单 kernel；本配置 NUM_KSPLIT=1，不做 split-K | 单次 dispatch 的 fused split-K：各 split 写 FP32 partial，atomic 计数，最后到达的 wave 在 kernel 内归约并写回 BF16 |
| 配置 | [`generic/best_config.json`](generic/best_config.json)：BM16 / BN32 / BK256，2 warps，stages 2，MLS，unroll 2，`mmac_layout_force=2` | [`best_config.json`](specialized_fused_splitk/best_config.json)：BM16 / BN32 / BK128，NUM_KSPLIT 8（每 split 256 个 K），1 warp，stages 2，MLS，unroll 2 |
| 自然 grid | 1 × 2 × 1 = 2 个 TG | 1 × 2 × 8 = 16 个 TG（`completed_wgs=16`） |
| 限制 | 无特殊限制 | 要求 `num_warps == 1`（`readfirstlane` 只在 wave 内广播 is_last） |
| 布局 | TT（另测了 TN） | TT |
| 正确性 | PASS：TT 与 TN 各 2 个进程（种子 29 / 30），每轮 768 个输出逐个与 BF16 舍入后的 FP32 参考比较（`atol=0.02, rtol=0.01`），各轮 `mismatches=0`；全输出有限、前后各 128 个哨兵不变；对未舍入 FP32 参考的平均误差通过 `≤ 1.05 × 舍入下限 + 1e-6` | 历史 PASS（未重测）：BF16 输出对未舍入 FP32 参考检查 `rtol=0.01, atol=0.001`，FP32 partial、保护区、计数器复位均 PASS；输入缩放 0.25，与通用版判据和输入不同 |
| 性能（PMD 单 die，kernel-only） | `best_config.json`（BK256），各 10 个正式样本的中位数：**TT 7.537531 µs，MMAC 利用率 0.0966%**（0.4173 TFLOPS；16566 cycles，范围 7.526156–7.738641 µs，202 VGPR）；**TN 7.180128 µs，MMAC 利用率 0.1014%**（0.4381 TFLOPS；15780.5 cycles，范围 7.092996–7.366906 µs，134 VGPR）；两布局 LDS 均为 49152 B，spill 为 0；<br>**有效带宽：TT 0.0415 TB/s（配置带宽达成率 2.31%），TN 0.0436 TB/s（2.42%）**；本 shape 算术强度 10.0556 FLOP/B，**带宽屋顶线 18.0966 TFLOPS**（屋顶线耗时 0.174 µs，口径见[上层 README](../README.md#性能指标公共口径)），实测仅达 2.3–2.4%，数据量仅约 0.3 MB，耗时不是带宽受限 | 历史 **6.59 µs，MMAC 利用率 0.1105%**（0.4773 TFLOPS；14485 cycles，3 个正式样本的中位数，范围 6.576–6.607 µs；本次未重测）；<br>**有效带宽 0.0475 TB/s**（配置带宽达成率 2.64%）；带宽屋顶线同通用版（18.0966 TFLOPS） |
| 复现 | `bash ../repro_generic/run.sh G256_12x64x2048 [--layout TN]`；结果见 [`generic/runs/`](generic/runs/)，用法见[上层 README](../README.md#复现通用版) | `bash run.sh`，见 [特化版 README](specialized_fused_splitk/README.md) |

说明：

- MMAC 利用率按 BF16 理论峰值 **432 TOPS**（浮点运算口径为 432 TFLOPS）计算：`利用率 = (2 × M × N × K / kernel耗时秒) / (432 × 10^12) × 100%`，乘加计 2 次运算。本 shape 的有效运算量为 3,145,728 FLOPs；以上用各自 summary 的未舍入 `median_us` 计算。该值是有效 GEMM 吞吐占给定峰值的比例，不是硬件计数器测得的 MMAC 指令活跃率；不额外计入 tile padding 或 split-K 归约运算，峰值对应单 die 48 CU（依据见[上层 README](../README.md#性能指标公共口径)）。

- 带宽与屋顶线（公共口径、理论带宽 1.8 TB/s 与 432 TFLOPS 峰值见[上层 README](../README.md#性能指标公共口径)）：本 shape 逻辑数据量 `D = 2 × (M×K + K×N + M×N) = 312,832 B`，有效带宽 `D / 完整 kernel 耗时`，算术强度 `2MNK / D = 10.0556 FLOP/B`，`min(432 TFLOPS, 10.0556 × 1.7996485 TB/s) = 18.0966 TFLOPS`。有效带宽用各 summary 的未舍入 `median_us` 计算（特化版用 6.590676 µs）。该 shape 的数据量太小，按带宽屋顶线只需约 0.174 µs，实测 6.6–7.5 µs 说明耗时主要来自固定开销、延迟和 K 方向串行（推断，未做剖析），因此带宽达成率很低不表示带宽还有 40 倍以上的提升空间，这个指标对本 shape 的参考价值有限。
- 特化版 6.59 µs 为本包 `fused_kernel.py`（`source_sha256=c4d2b5e5…`）的结果，汇总见 [`runs/verification_heuristics_20261104/summary.json`](specialized_fused_splitk/runs/verification_heuristics_20261104/summary.json)。
- 通用版 `best_config.json`（BK256）于 2026-11-05 按当前判据重新复现，配置未改。新汇总：[TT](generic/runs/TT_20261105_061739/summary.json)、[TN](generic/runs/TN_20261105_062009/summary.json)；原先按未舍入参考与 `atol=0.001` 测得的通用版汇总已删除。TT 平均绝对误差约 0.05065726 / 0.05007705，TN 约 0.04871519 / 0.05067509，均通过舍入下限检查；当前汇总不提供最大绝对误差。
- `autotune/results_TT_dyn/shapes/12x64x2048_TT.json` 归档的是另一配置（BK128、unroll 4，真实参数在 `best.config`），本次未重测；不能用当前 BK256 的 PASS 为该归档背书。历史 `OPT_compare_TT` 的 BK64 特化配置也不在本包内，源码不同，尚不能当作本包结果。
- 两版的计时均为 PMD `system.gcSimCycles` 经 `gcFrequency` 换算，不含编译与分配；正确性判据并未统一。通用版本次为 2 个进程各预热 1 次、正式 5 次，取 10 个样本的中位数。

## 测量条件

| 项 | 内容 |
| --- | --- |
| 执行环境 | PMD，`GPU_CHIP=sb`（gfx946），频率 2.197802 GHz；**单 die、48 CU**。summary 未记录 die/CU 数，依据是本地 `generic/runs/TT_ir_asm_perf/g/` 与 `TN_ir_asm_perf/g/` 下的 `stats.txt`：含 `CUs00` – `CUs47` 共 48 个 CU，且只输出 `xcd0`；[调优记录](../../TUNING.md) 规定每 die 48 CU，因此是 1 die。特化版的历史记录没有留下对应日志，按同一 PMD 配置计，未逐条核实。详见[上层 README](../README.md#环境与版本) |
| 布局 | 特化版与通用版主结果为 TT：A 行主序，B 为 `[K,N]` 行主序。通用版另测 TN：W 为 `[N,K]`，K 连续 |
| 计时 | 特化版：1 轮、预热 1 次、正式 3 次，取中位数。通用版复现：2 个进程各预热 1 次、正式 5 次，取 10 个样本的中位数。通用版 autotune：预热 1 次、正式 3 次，取中位数 |
| 软件 | 通用版 TT/TN 均记录 BoltOPs `673ede654b7cda290f68ce98665fb6f1dcb1efd8`、`tracked_changes=2`（测量时状态）；Triton `cdf6827b53b1a59104da41a1bfc7736134b13a2e`、`tracked_changes=0`。两份通用 kernel 的 SHA256 均为 `e108b0bb94c719409fc0a8b387e295f1d39bf1e02d364494a1d92c5e361313fb`，`libtriton.so` 为 `919b51f14bcf5e177dc710922daa0b7e8d9758e88ede0819680b4d8f7e42caa7`；路径和依赖文件 SHA 见上述 summary。特化版历史 summary 没有记录 commit 和 `libtriton.so` SHA，不能据此确认与通用版使用完全相同的软件栈 |

[调优记录](../../TUNING.md) 要求每次性能测量保存环境 manifest（commit、私有库 SHA256、完整有效 config、环境变量等）。通用版的 `summary.json` 已包含 commit、`libtriton.so` 与 kernel 源码的 SHA256；die/CU 数已核对（见上表）。

## 目录

```text
G256_12x64x2048/
├── README.md
├── generic/
│   ├── best_config.json
│   └── runs/<layout>_<时间>/summary.json   # repro_generic 的复现结果（round_*/ 被忽略）
└── specialized_fused_splitk/
    ├── README.md            # 特化版的运行方式与动态 MNK 说明
    ├── fused_kernel.py
    ├── best_config.json
    ├── bench.py / run.sh    # 独立计时脚本，不依赖其他目录
    ├── .gitignore           # 原始 PMD 输出不进 Git
    └── runs/*/summary.json  # 每次验证的汇总（round_*/ 原始输出被忽略）
```
