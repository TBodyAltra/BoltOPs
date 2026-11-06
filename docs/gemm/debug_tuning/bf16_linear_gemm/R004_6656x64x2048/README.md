# R004 · 6656 × 64 × 2048 · BF16 linear（prefill）

本目录记录需求编号 R004 的调优结果。同一个 shape 保留两版实现：**通用版**（`generic/`）与**特化版**（`specialized_overlap/`）。两版各自带有自己的 `best_config.json`，配置只对同目录的内核有意义，不能互相套用。

## 需求信息

| 项 | 内容 |
| --- | --- |
| 需求编号 | [R004](../../../SHAOBO_BU_REQUIREMENTS.md#r004)；R023 也包含相同 shape |
| 算子 | `in_proj_ba` |
| M × N × K | 6656 × 64 × 2048 |
| 阶段 / 频次 | prefill / 1 |
| 精度 | BF16 × BF16 → BF16，FP32 累加 |
| 对接接口 | [gemm_a16w16](../../../../../boltops/gemm/triton/gemm_a16w16.py)（权重布局 `[N,K]`） |
| 需求表当前状态 | 接口 smoke 通过；不代表本特化内核已接入接口 |

## 两个版本

| | 通用版 `generic/` | 特化版 `specialized_overlap/` |
| --- | --- | --- |
| 内核 | [`_gemm_bf16_generic_kernel`](../autotune/scripts/gemm_a16w16_generic.py)（所有 shape 共用，不在本目录保存副本） | [`kernel.py`](specialized_overlap/kernel.py)（随本目录归档） |
| 算法 | 动态 M/N/K，单 kernel；本配置不做 split-K | 以 16 行为单位划分输出所有权：每个程序读取并计算完整 64 行块，相邻读取窗口可重叠，写回只保留各自负责的行。无 split-K、原子或归约 kernel |
| 配置 | [`generic/best_config.json`](generic/best_config.json)：BM128 / BN32 / BK32，4 warps，stages 2，异步拷贝与 pingpong | [`best_config.json`](specialized_overlap/best_config.json)：BM64 / BN64 / BK64，144 个程序，MLS 与 warp specialization，`num_warps=4`（编译元数据为 5 wave），stages 2 |
| 自然 grid | 104 个工作组 | 144 个工作组（历史运行中 48 个计算单元各完成 3 个） |
| 资源 | 94 VGPR，LDS 30720 B，spill 0 | 57 VGPR，LDS 40960 B，spill 0 |
| 限制 | 通用内核，能力由原算子实现决定 | 仅验证连续 BF16 TT 输入 6656×64×2048；不提供通用内核的 bias、激活、任意尾块等功能，不能直接替换通用算子 |
| 布局 | TT | TT |
| 正确性 | PASS：seed 29，最后一次输出 425984 个元素全量比较，`mismatches=0`；输出有限，前后保护区不变；对未舍入 FP32 参考的平均误差通过 `≤ 1.05 × 舍入下限 + 1e-6` | 历史 PASS：seed 29 与 143 均全量比较、0 mismatch，判据同通用版，另检查计数器保持为 0；独立复现包 seed 29，预热及每次正式 dispatch 均全量 PASS |
| 性能（PMD，kernel-only） | 历史基线，1 个进程、预热 1 次、正式 2 次的中位数：**24.8396 µs，MMAC 利用率 16.26%**（54592.5 cycles；70.24 TFLOPS；范围 24.806–24.873 µs） | 2 个进程各预热 1 次、正式 2 次，4 个样本合并中位数：**24.2581 µs，MMAC 利用率 16.65%**（53314.5 cycles；71.93 TFLOPS；范围 24.159–24.291 µs）；较同期基线周期降低约 **2.34%** |
| 复现 | 在本目录执行 `bash ../repro_generic/run.sh R004_6656x64x2048 --layout TT --rounds 1 --dispatches 3` | 在 `specialized_overlap/` 执行 `bash run.sh --rounds 1 --warmup 1 --reps 2 --seed 29`，见[特化版 README](specialized_overlap/README.md) |

说明：

- MMAC 利用率按 BF16 理论峰值 **432 TOPS** 计算：`利用率 = (2 × M × N × K / kernel耗时秒) / (432 × 10^12) × 100%`。本 shape 有效运算量为 1,744,830,464 FLOPs，用各自 `median_us` 计算。该值是有效 GEMM 吞吐占给定峰值的比例，不是硬件计数器测得的 MMAC 指令活跃率；不额外计入 tile padding，峰值适用的 die/CU 范围仍需与测量环境核对。
- 通用版配置来自[原始调优结果](../autotune/results_TT_dyn/shapes/R004_6656x64x2048_TT.json)的 `best.config`。[通用版 summary](generic/runs/TT_v2final_recorded/summary.json) 与[特化版历史 summary](specialized_overlap/runs/verification_v2final_recorded/summary.json)均为既有实验结果的转录，不是整理本目录时重新执行的记录；样本数较少，2.34% 仅代表该批次的测量差异。
- 特化配置中的 `SK_SPLIT=16` 在该实验接口里表示**输出行所有权的对齐粒度**，不是 split-K 数。
- 特化复现包另在临时目录独立实跑一次（seed 29，预热 1 次、正式 2 次），中位数 **53115 cycles / 24.1673 µs**（MMAC 利用率 16.71%），每次 dispatch 均全量校验通过，见[实跑 summary](specialized_overlap/runs/package_smoke_seed29_20261106/summary.json)。该次没有同时重跑基线，不能与历史基线组成新的加速比。
- 历史对照中另有通用版 `final_baseline_seed29`（25.0905 µs）与特化版 `final_overlap_m64_k64_seed29`（24.5049 µs），均记录在特化版历史 summary 中；它们是不同批次，不与上表主结果混用。

## 测量条件

| 项 | 内容 |
| --- | --- |
| 执行环境 | PMD，gfx946，48 个计算单元，频率 2.197802 GHz；结果为模拟值 |
| 布局 | TT：A 为连续 `[M,K]`，B 为连续 `[K,N]`，均为行主序 |
| 计时 | `system.gcSimCycles / (gcFrequency × 1000)`，不含编译、分配、复制与 CPU 校验。通用版历史：1 轮、预热 1 次、正式 2 次。特化版历史：2 个进程各预热 1 次、正式 2 次，取 4 个样本中位数。复现包实跑：1 轮、预热 1 次、正式 2 次 |
| 正确性 | BF16 舍入后的 CPU FP32 参考逐元素比较，`atol=0.02, rtol=0.01`；另检查输出有限性、保护区，以及相对未舍入参考的平均绝对误差。历史 worker 检查最后一次输出，新复现脚本检查每次 dispatch |
| 软件 | 通用版与特化版历史 summary 均记录归档 tune 库 SHA256 `9f52a590…`；内核源码 SHA256：通用 `e108b0bb…`，特化 `be5ab01d…`。未记录 commit 与 `libtriton.so` SHA，不能确认软件栈完全一致 |
| 来源 | 原始路径、资源及样本见各 `summary.json`；不要将新运行的计时与历史记录混为同一次比较 |

## 本地 IR、汇编与 Perf

`generic/runs/TT_ir_asm_perf/v2final_baseline/` 对应通用版历史基线；`specialized_overlap/runs/TT_ir_asm_perf/` 下的 `v2final_oldbest_a/`、`v2final_oldbest_b/` 对应特化版两次历史复测，`package_smoke_seed29/` 对应复现包实跑。这些目录为本地查看产物，遵循现有忽略规则，不提交。

每个运行目录中，`kernel.ttir`、`kernel.ttgir`、`kernel.llir` 是不同阶段的 IR，`kernel.amdgcn` 是汇编；`g/0/{0,1,2}/` 分别是预热与两次正式 dispatch，其中 `stats_xcd0.perf` 是文本统计，另有 `.kinfo`。文件复制自对应原始运行，不是重新采集。

## 目录

```text
R004_6656x64x2048/
├── README.md
├── generic/
│   ├── best_config.json
│   └── runs/
│       ├── TT_v2final_recorded/summary.json
│       └── TT_ir_asm_perf/v2final_baseline/       # 本地文件，不提交
└── specialized_overlap/
    ├── README.md            # 特化版的运行方式说明
    ├── kernel.py
    ├── best_config.json
    ├── bench.py / run.sh    # 独立计时脚本，不依赖其他目录
    ├── .gitignore           # 原始 PMD 输出不进 Git
    └── runs/
        ├── verification_v2final_recorded/summary.json
        ├── package_smoke_seed29_20261106/summary.json
        └── TT_ir_asm_perf/                        # 本地文件，不提交
            ├── v2final_oldbest_a/
            ├── v2final_oldbest_b/
            └── package_smoke_seed29/
```
