# G261 · 12 × 2048 × 4096 · BF16 linear（decode）

本目录只有通用版：使用所有 shape 共用的 [`_gemm_bf16_generic_kernel`](../operators/gemm_a16w16_generic.py)，没有特化 kernel，只保存配置和复现结果。

## 需求信息

| 项 | 内容 |
| --- | --- |
| 需求编号 | G261（需求表中对应 R015，`o_proj` / `out_proj`），见 [少伯 BU 需求 · 表 3](../../SHAOBO_BU_REQUIREMENTS.md#table-3) |
| M × N × K | 12 × 2048 × 4096 |
| 阶段 / 频次 | decode / 2 |
| 精度 | BF16 × BF16 → BF16，FP32 累加 |
| 对接接口 | [gemm_a16w16](../../../../../boltops/gemm/triton/gemm_a16w16.py)（权重布局 `[N,K]`） |

## 通用版

配置 [`generic/best_config.json`](generic/best_config.json)：BM16 / BN32 / BK128，2 warps，**3 级流水**，MLS，**unroll 1**，`mmac_layout_force=3`，`enable_v_mmac_cluster=1`，**关闭 max-ILP 调度**（`amdgpu_enable_max_ilp_scheduling_strategy=false`）。自然 grid 为 1 × 64 × 1 = 64 个 TG，NUM_KSPLIT=1。

复现（`repro_generic`，每个布局 2 个独立进程，各预热 1 次、正式 5 次，取 10 个样本的中位数，PMD `system.gcSimCycles`，只含 kernel）：

| 布局 | 结果 | 范围 | 资源 | 正确性 |
| --- | --- | --- | --- | --- |
| TT | **31808.5 cycles，14.473 µs，MMAC 利用率 3.2201%**（13.9106 TFLOPS） | 14.463–14.506 µs | 58 VGPR，LDS 36864 B，0 spill | PASS |
| TN | 31896 cycles，14.513 µs，**MMAC 利用率 3.2112%**（13.8725 TFLOPS） | 14.465–14.595 µs | 38 VGPR，LDS 36864 B，0 spill | PASS |

结果目录：[`generic/runs/TT_20261105_062328`](generic/runs/TT_20261105_062328/summary.json)、[`generic/runs/TN_20261105_062328`](generic/runs/TN_20261105_062328/summary.json)。

MMAC 利用率按 BF16 理论峰值 **432 TOPS**（浮点运算口径为 432 TFLOPS）计算：`利用率 = (2 × M × N × K / kernel耗时秒) / (432 × 10^12) × 100%`，乘加计 2 次运算。本 shape 的有效运算量为 201,326,592 FLOPs；以上用各自 summary 的未舍入 `median_us` 计算。该值是有效 GEMM 吞吐占给定峰值的比例，不是硬件计数器测得的 MMAC 指令活跃率；不额外计入 tile padding 运算，峰值适用的 die/CU 范围仍需与测量环境核对。

精度按 `repro_generic` 的口径：全部 24576 个输出元素与 CPU FP32 参考（舍入到 BF16 作为 gold）比较，`|got − gold| ≤ 0.02 + 0.01·|gold|`，0 个超限；平均误差 `|got − ref|` 不超过 BF16 舍入误差下限的 1.05 倍；输出全为有限值；输出前后各 128 个哨兵值未被改写。

```bash
cd ../repro_generic
bash run.sh G261_12x2048x4096                 # TT
bash run.sh G261_12x2048x4096 --layout TN     # TN（gemm_a16w16 接口布局）
```

## 配置来源与对照

- 配置是在 12×2048×4096 TT 上手工扫描得到的：在 autotune 作业 `autotune_20261104_dyn` 的最佳配置（BM16 / BN32 / BK128、2 warps、2 级流水、unroll 4、max-ILP 开）基础上，改为 3 级流水、unroll 1、关闭 max-ILP。
- 该 autotune 最佳配置在 autotune 口径下是 33776 cycles（15.37 µs，预热 1 次 + 正式 3 次）。本目录的 31808.5 cycles 来自不同的计时协议（见上，10 个样本），**没有在同一次运行里 A/B 对照**，因此约 5.8% 的差距只能作参考，不能当作严格的加速比。
- 小 K 的预估模型和各配置扫描见 `xk/BoltOPs/tmp/1105/12x2048x4096_TT_mmac_bubble/SCALING_MODEL.md`。该实验的口径是 `rotate_b`（每次 dispatch 的 B 都从 HBM 冷读），在该口径下 s3u1 + 关 max-ILP 的 K=4096 为 32364 cycles。
- 瓶颈：K=4096 时 B 矩阵 16.78 MB，按 819 B/周期的 HBM 峰值至少需要 20605 周期。稳态循环已达到该下限的约 96%；`gcSimCycles` 里另有约 7938 周期是模拟器启动/收尾的常数。因此继续在配置上调优的空间很小。
- 更多 stage（4、5）、`num_warps=1`、BK64 / BK256、unroll 2 / 4 都在同一扫描中试过，均比本配置慢。

## 待办

1. 补充 die / CU 数，口径同 [G256](../G256_12x64x2048/README.md)。
2. 在同一次运行里 A/B 对比 autotune 最佳配置与本配置。
3. 本配置只在 TT 上调过，TN 的结果是直接套用同一份配置。
