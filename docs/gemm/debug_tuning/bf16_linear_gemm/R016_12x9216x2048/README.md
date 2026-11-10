# R016 · 12 × 9216 × 2048 · BF16 linear（decode）

本目录记录需求编号 R016 的调优结果。同一个 shape 保留两版实现：**通用版**（`generic/`）与**特化版**（`specialized_bn32_w1/`）。两版各自带有自己的 `best_config.json`，配置只对同目录的 kernel 有意义，不能互相套用。

特化版是一个只改 kernel 和配置的方案：32 列 × 1 wave 的小块，M、N、K 与 stride 均为运行时参数，**不使用“M、N、stride 静态化”，也不使用 `slc` 缓存策略位**，因此不需要任何编译器改动。这两项改动是在其上的进一步方案，只在下文“补丁方案说明”中解释，本目录不保存其代码和结果。

## 需求信息

| 项 | 内容 |
| --- | --- |
| 需求编号 | R016（`qkv_proj`），见 [少伯 BU 需求](../../../SHAOBO_BU_REQUIREMENTS.md#r016) |
| M × N × K | 12 × 9216 × 2048 |
| 阶段 / 频次 | decode / 1 |
| 模型 / 配置 | 需求表未填写（—） |
| 精度 | BF16 × BF16 → BF16，FP32 累加 |
| 对接接口 | [gemm_a16w16](../../../../../boltops/gemm/triton/gemm_a16w16.py)（权重布局 `[N,K]`）。本次直接测 kernel，不是生产接口端到端测试，特化版也未接入该接口 |
| 需求表当前状态 | 接口 smoke 通过（尚未回填调优结果） |

## 结论

在同一批次、独立 PMD 进程、相同种子下，特化版把完整 kernel 耗时从 **27.237667 µs 降到 25.736850 µs**，降低 **5.51%**（1.058×，逐种子 5.35% / 5.73%）；MMAC 利用率从 **3.8497% 升到 4.0742%**，有效带宽从 **1.3958 TB/s 升到 1.4772 TB/s**（配置带宽的 77.56% → 82.08%）。每个工作组改为 1 个 wave、32 列后，总启动 wave 数不变（288），但共享内存 bank conflict 和 barrier 清零，全局访存流水线的信用停顿大幅下降。

该 shape 的 B 矩阵占 99.3% 的数据量，算术强度 11.9147 FLOP/B，按本仓库口径属于访存受限，带宽屋顶线为 **21.4422 TFLOPS**；特化版达到屋顶线的 82.08%。数据量除以配置带宽为 21.126 µs，再加上每个 dispatch 约 3.61 µs 的 wave 区间外耗时，参考值为 24.738 µs，特化版实测比它高约 4.0%。这是参考，不是严格下界。

## 两个版本

| | 通用版 `generic/` | 特化版 `specialized_bn32_w1/` |
| --- | --- | --- |
| kernel | [`_gemm_bf16_generic_kernel`](../operators/gemm_a16w16_generic.py)（所有 shape 共用，不在本目录内保存副本） | [`r016_kernel.py`](specialized_bn32_w1/r016_kernel.py)（随本目录归档） |
| 算法 | 动态 M/N/K，单 kernel；NUM_KSPLIT=1，不做 split-K | 同样无 split-K，M/N/K 与 stride 同样是运行时参数；B 的 N 偏移每个工作组重定位一次；工作组沿 N 一维排列；32 列 × 1 wave 的小块 |
| 配置 | [`generic/best_config.json`](generic/best_config.json)：BM16 / BN128 / BK128，4 warps，stages 2，MLS，`mmac_layout_force=3`，`utc_warmup=true` | [`best_config.json`](specialized_bn32_w1/best_config.json)：BM16 / BN32 / BK64，1 warp，stages 2，MLS，`mmac_layout_force=3`，`schedule_hint="local-prefetch"`，unroll 1 |
| 配置来源 | [`results_TT_dyn/shapes/R016_12x9216x2048_TT.json`](../autotune/results_TT_dyn/shapes/R016_12x9216x2048_TT.json) 的 `best.config`，内容完全一致 | 本次局部优化搜索结果 |
| 自然 grid | ceil(12/16) × ceil(9216/128) = 72 个 TG（`completed_wgs=72`） | 9216/32 = 288 个 TG（`completed_wgs=288`），每 TG 1 wave，总启动 wave 数与通用版同为 288 |
| 资源 | 58 VGPR，LDS 36864 B，spill 0 | 33 VGPR，LDS 12288 B，spill 0 |
| 限制 | 无特殊限制 | 仅验证完整形状 12×9216×2048、TT、连续 BF16 输入；隐含 M ≤ 16、N % 32 == 0、K % 64 == 0；不依赖编译器改动 |
| 布局 | TT（TN 未测） | TT（TN 未测） |
| 正确性 | PASS：seed 29 / 30，每个 dispatch 的 110592 个输出逐个与 BF16 舍入后的 FP32 参考比较（`atol=0.02, rtol=0.01`），各 dispatch `mismatches=0`；全输出有限、前后各 128 个哨兵不变；对未舍入 FP32 参考的平均误差满足 `≤ 1.05 × 舍入下限 + 1e-6` | PASS：同一判据；seed 29 / 30 的基线与特化版各 5 个 dispatch、独立 B（seed 32，基线与特化版各 3 个）合计 26 个 dispatch，全部逐次全量校验 0 不匹配；另有 seed 31 的 NaN 行尾 padding / 不同 leading dimension 测试（2 次 dispatch，只检查末次输出，仅作正确性证据） |
| 性能（PMD 单 die，kernel-only，2 个进程 × 4 个计时样本 = 8 个样本的合并中位数） | 转录批次 **27.141890 µs，MMAC 利用率 3.8633%**（16.6895 TFLOPS；59652.5 cycles；范围 27.057942–27.258142 µs），有效带宽 1.4008 TB/s（77.83%）；与特化版同轮重测的基线为 **27.237667 µs，3.8497%**，1.3958 TB/s（77.56%） | **25.736850 µs，MMAC 利用率 4.0742%**（17.6006 TFLOPS；56564.5 cycles；范围 25.719332–25.765287 µs）；较同轮基线降低 **5.51%**；有效带宽 **1.4772 TB/s**（配置带宽达成率 82.08%） |
| 带宽屋顶线 | 本 shape 算术强度 11.9147 FLOP/B，**屋顶线 21.4422 TFLOPS**（屋顶线耗时 21.126 µs，口径见[上层 README](../README.md#性能指标公共口径)） | 同通用版；实测达到屋顶线的 **82.08%** |
| 复现 | `bash ../repro_generic/run.sh R016_12x9216x2048`（尚未用该脚本跑过，见待办）；本目录结果见 [`generic/runs/TT_final_recorded/`](generic/runs/TT_final_recorded/summary.json) | 在 `specialized_bn32_w1/` 执行 `bash run.sh --name <新名字> --compare --seed 29`，见[特化版 README](specialized_bn32_w1/README.md) |

## 指标口径说明

三类指标的口径与 [R014](../R014_12x64x2048/README.md)、[R017](../R017_12x12288x2048/README.md) 及[上层 README](../README.md#性能指标公共口径)相同，并由 [`metrics.py`](metrics.py) 用同一组公式统一计算（`python3 -I metrics.py` 可核对本目录全部 `summary.json`）：

- **MMAC 利用率** = `(2 × M × N × K / 完整 kernel 耗时) / 432 TFLOPS`。本 shape 有效运算量 2MNK = 452,984,832 FLOPs；乘加计 2 次运算，不计 M 从 12 补齐到 16 的 tile padding。它是有效 GEMM 吞吐占给定峰值的比例，**不是**硬件计数器测得的 MMAC 指令活跃率。
- **有效带宽** = `D / 完整 kernel 耗时`，`D = 2 × (M×K + K×N + M×N) = 49,152 + 37,748,736 + 221,184 = 38,019,072 B`（A 与 B 各读一次、C 写一次，BF16 每元素 2 B）。它是逻辑数据量除以耗时，不是实测外存流量，不计缓存复用、重复读取、事务放大和 tile padding；配置带宽达成率的分母为 1.7996485 TB/s。
- **带宽屋顶线** = `min(432 TFLOPS, 11.9147 FLOP/B × 1.7996485 TB/s) = 21.4422 TFLOPS`。这是该数据访问假设下的参考，不是所有实现的严格性能上界；该 shape 的 MMAC 利用率在此口径下的参考上限约 4.96%。
- 分母统一为 **summary 中未舍入的完整 kernel 耗时** `median_us`（PMD `system.gcSimCycles` 经 `gcFrequency` 换算），不使用 SIMD 活跃时间或 wave 区间时间；README 中的四舍五入值不能反算。

硬件计数器口径的“MMAC 忙碌率”统计的是 MMAC 指令忙碌时间，包含 M 从 12 补齐到 16 的运算，分母和峰值也不同，**不能与本表或 R014 / R015 / R017 的“MMAC 利用率”比较**。同理，只算 B、只算 wave 区间的带宽数字是另一口径，本目录不使用。

## 瓶颈分析（PMD 计数器，seed 29 最后一次计时 dispatch）

| 计数器 | 基线 | 特化版 |
| --- | --- | --- |
| 完成的矩阵计算指令数 | 73728 | 73728 |
| 共享内存 bank conflict | 6912 | 0 |
| barrier 累计计数 | 33798 | 0 |
| `GlobalMemPipeline.cmdCreditStall` | 71275 | 2062 |
| UMC 读请求数 | 147649 | 147649 |
| 标量指令 / VMEM 指令 | 184896 / 43776 | 274752 / 55584 |
| `waitVmCounter` | 1497781 | 1688927.5 |
| `waitLgkmCounter` | 153182.5 | 83668 |
| wave 区间 / 区间外 cycles | 51937 / 7940 | 48689 / 7938 |

- 矩阵计算指令数相同，没有以少算换速度。bank conflict 和 barrier 清零、全局访存流水线的信用停顿大幅下降，换来更好的工作组粒度和预取调度；标量和 VMEM 指令反而更多，所以不是每个计数器都变好。stall / wait 是跨 CU、SIMD 累加且可重叠的计数，不能直接当作 kernel 壁钟耗时。
- 两版的 UMC 读请求数相同，说明本方案的收益来自调度和同步，而不是减少读取次数。
- 主要数据是 B 的 37,748,736 B（36 MiB），A 为 49,152 B，C 为 221,184 B。仅从数据量看，`D / 1.7996485 TB/s = 21.126 µs`；每个 dispatch 另有约 7938 cycles（**3.612 µs**）落在 wave 区间之外，二者相加为 24.738 µs，特化版实测 25.737 µs，高约 4.0%。这是**配置带宽与逻辑流量的参考，不是严格物理下界**：区间外耗时没有被证明可以消除，缓存、写回和访存重叠都会影响解释。
- UMC 请求数 × 256 B 只能作等价流量估算；PMD `agent_info` 中约 899.8 GB/s 的数字受 `EABusSize` / `ea_bus_size` 字段名不一致影响，本实验未修改 PMD，不采用。

## 搜索过程摘要

搜索覆盖了 N 方向的块大小、K 方向的块大小、1 / 2 / 4 / 8 wave、流水级数、展开、MMAC 布局、PID 排列、部分 N 块与缓存策略位等维度，以 seed 29 的完整形状复测和 PMD 访存 / 同步计数为依据逐步收敛。为降低 PMD 成本，先用固定 N=9216、K 缩为 512 / 1024 的小形状筛选，再用完整形状复测；小形状只用于排除明显较差的候选，最终排序以完整形状为准。

部分候选被明确拒绝：BN64 / 4 wave 与 BN128 / 8 wave 在 K=512 的筛选中正确性失败，BN16 / 1 wave 在 `TritonHCUMlsEncodingInsertion` 阶段编译失败，耗时均不计入优化成果。收敛依据是完整形状复测、访存 / 同步计数的改善以及接近带宽参考，**不是穷举或全局最优证明**。逐候选的耗时表不随本目录归档。

## 补充实验

- **独立 B**（seed 32，每次 dispatch 使用新分配、新数据的 B，全部 B 保持存活，3 次 dispatch 取后 2 次中位数，未显式清缓存）：基线 27.353692 µs（3.8334%，1.3899 TB/s）→ 特化版 26.341317 µs（3.9807%，1.4433 TB/s），降低 **3.70%**。
- **NaN 行尾 padding**（seed 31，不同 leading dimension，2 次 dispatch，仅检查末次输出的完整 110592 个值）：全部有限、哨兵不变、0 不匹配。这是正确性证据，不与连续布局比较性能。

两项以及上面的主结果均来自 [`specialized_bn32_w1/runs/final/summary.json`](specialized_bn32_w1/runs/final/summary.json)，由本目录的脚本在 2026-11-07 实际重跑得到。

## 补丁方案说明（仅说明，不记录代码与结果）

在特化版之上，早期实验还尝试过两项进一步的改动。它们需要改编译器或让 kernel 与具体 shape 绑定，所以没有作为本目录的特化方案保留：

1. **M、N、stride 静态化。** 把 `M`、`N` 和全部 stride 声明为 `tl.constexpr`（并对 `stride_ak == stride_bn == stride_cn == 1`、`N % BLOCK_SIZE_N == 0` 等做 `static_assert`），K 仍保持运行时参数（`do_not_specialize=['K']`）以保留软件流水。代价是每个 shape / 布局单独编译，kernel 不能被其他 shape 复用。
2. **`slc` 缓存策略位。** 给 HCU 后端增加默认关闭的 opt-in 编译选项 `mls_row_major_slc`（参与 `HIPOptions.hash()`，只允许 gfx946）。启用后，在 LLVM 调度之前，把 `llvm.hcu.matrix.load.*` 调用末尾 5 个标志恰为 `t=1,r=0,glc=0,slc=0,bps=0` 的调用的 `slc` 置 1。在本 TT kernel 中这个模式对应 B 的加载（共 8 处），但它不是适用于任意 kernel 或布局的“B 判定”。可以用进程内的后端扩展实现，也可以做成对 `third_party/hcu/backend/compiler.py` 的原生补丁；原生补丁只在内存中验证过语法和 IR 变换等价性，**没有应用、重建或端到端验证**。

两项叠加在特化版之上，早期实验中完整 kernel 耗时约再低 2%（约 25.2 µs 对本目录的 25.7 µs，批次不同，只作数量级参考，不能当作本目录结论）。如需采用，应在隔离的 Triton 副本中应用补丁、重建并按本目录的协议重测。原始实现、补丁和实验记录留在 `tmp/1105/r016_tt_extreme_2305/`（`r016_kernel.py`、`local_backend.py`、`compiler_mls_slc.patch`、`dynamic_ablation/`），tmp 不是长期归档位置。

## 测量条件

| 项 | 内容 |
| --- | --- |
| 执行环境 | PMD，`GPU_CHIP=sb`（gfx946），`PMD_PATH=/opt/rocm/pmd`，频率 2.197802 GHz（core period 455 ps）；运行记录中 `cu_count=48`，即**单 die、48 CU**（[调优记录](../../TUNING.md) 规定每 die 48 CU）。summary 未单独记录 die 数；详见[上层 README](../README.md#环境与版本) |
| 布局 | TT：A 为 `[M,K]` 行主序，B 为 `[K,N]` 行主序 |
| 计时 | 每个条件 2 个独立进程（seed 29 / 30），各 1 次预热 + 4 次计时（5 次 dispatch），同一 B 重复使用，取 8 个样本的中位数；基线与特化版的进程同时启动；PMD `system.gcSimCycles / gcFrequency`，只含 kernel，不含 JIT 编译、分配、主机拷贝和生产接口开销 |
| 软件 | Python 3.10.12，Triton 3.6.0（`triton_3.6/triton/build/lib.linux-x86_64-3.10`），`libtriton.so` SHA256 `b264c76926a43ad3767c23fe0140f422573ea9f55b04f1828b46389f2f8d5cf6`；clang-18 `/opt/dtk/aillvm/bin/clang-18`。2026-11-07 运行结束后核对：该 Triton checkout 的 HEAD 为 `e58481668e29fda141014f1d4b1f0f7bd82afd41`，已跟踪文件无改动，`libtriton.so` 的 SHA256 与上述一致。**运行时没有保存 manifest，这些核对结果是运行后取得的**。它不同于上层 README“环境与版本”中 R014 / R015 使用的 `cdf6827b…`，两者不能混称同一软件栈 |
| 文件 SHA256 | 通用 kernel `e108b0bb94c719409fc0a8b387e295f1d39bf1e02d364494a1d92c5e361313fb`；特化 `r016_kernel.py` `f9719643125968f73968086181fc7683c0ef8bef730861603f2b3ccaf14d3974`；`best_config.json` `b7215bcd8d72ba4960b100d464161eb7fea77f3bb7059e9960bad3e9dad1bd8d`；`tune_lib.py` 前缀 `9f52a590…` |
| BoltOPs 版本 | 工作目录不是 Git 仓库，没有提交号，是版本溯源局限 |
| 与 R017 的差异 | 计时与正确性判据接近 [R017](../R017_12x12288x2048/README.md)（同一 `tune_lib.py`），但本批每进程是 1 次预热 + 4 次计时，而 R017 为 1 + 5；通用版基线来自本批同时测量的对照，不是 `repro_generic` 的复现 |

[调优记录](../../TUNING.md) 要求每次性能测量保存环境 manifest。本批没有保存，因此数据在补齐 manifest 前不应用于性能排名；die / CU 数已核对（见上表）。

## 结果文件的来源

- `specialized_bn32_w1/runs/final/summary.json`：本目录脚本的实际重跑结果（2026-11-07），原始运行目录在汇总后已删除，只保留 `runs/TT_ir_asm_perf/final29_best/` 中的 IR、汇编和 perf 文件（本地，被忽略规则排除，不提交）。
- `generic/runs/TT_final_recorded/summary.json`：从 `tmp/1105/r016_tt_extreme_2305` 早期运行的 `result.json` **转录并统一重算指标**得到，没有重新执行 GEMM；`source_run` 字段指向原运行。tmp 清理后这些来源路径会失效。`generic/runs/TT_ir_asm_perf/` 下的 IR、汇编和 perf 文件复制自同一批运行，同样不提交。
- 通用版的转录批次与特化版的实跑批次不是同一批，因此特化版的降幅以其同轮重测的基线为准。

## 复现

```bash
REPO=/public/home/duyubo/bolt/github/BoltOPs
cd "$REPO/docs/gemm/debug_tuning/bf16_linear_gemm/R016_12x9216x2048/specialized_bn32_w1"
bash run.sh --name my_cmp_seed29 --compare --seed 29 --dispatches 5               # 基线 + 特化版
cd .. && python3 -I metrics.py                                                     # 核对全部 summary 的指标
```

## 待办

1. 补保存环境 manifest。
2. 用 `repro_generic` 按统一协议（2 个进程 × 1 预热 + 5 次计时，`bash ../repro_generic/run.sh R016_12x9216x2048`）重测通用版，并补充 TN；本目录通用版结果是早期批次的转录。
3. 特化版只有 TT；TN 须另行验证。
4. 如要并入通用算子或 `gemm_a16w16`：需要决定 M ≤ 16、N % 32 == 0、K % 64 == 0 的前置条件如何检查；如还要采用“补丁方案说明”中的两项改动，需另行验证。

## 目录

```text
R016_12x9216x2048/
├── README.md
├── metrics.py                          # 统一的 MMAC 利用率 / 有效带宽 / 屋顶线公式，并核对各 summary
├── generic/
│   ├── best_config.json
│   └── runs/
│       ├── TT_final_recorded/summary.json   # 基线 seed 29 / 30 的转录（含 8 样本合并中位数）
│       └── TT_ir_asm_perf/final29_baseline/ # 本地 IR / 汇编 / perf，不提交
└── specialized_bn32_w1/
    ├── README.md            # 特化版的文件说明、运行方式、判据、局限
    ├── r016_kernel.py       # kernel 与入口（M/N/K、stride 为运行时参数）
    ├── best_config.json
    ├── bench.py / reproduce.py / runtime_checks.py / run.sh
    ├── .gitignore           # 原始 PMD 输出不进 Git，只保留 summary.json
    └── runs/
        ├── final/summary.json                 # 基线 vs 特化版、独立 B、padding、计数器
        └── TT_ir_asm_perf/final29_best/       # 本地，不提交
```
