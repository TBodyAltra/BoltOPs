# R018 · 12 × 248320 × 2048 · BF16 linear（decode / prefill）

本目录记录需求编号 R018 的调优结果。同一个 shape 保留两版实现：**通用版**（`generic/`）与**特化版**（`specialized_mixbn_t64/`）。两版各自带有自己的 `best_config.json`，配置只对同目录的 kernel 有意义，不能互相套用（特化版配置比通用版多 `TAIL_BN`、`CU_MULT` 两项）。

特化版只改 kernel 与任务映射：主体用与通用版相同的 BN=256 块，不能被 48 个 CU 整除的尾部改用 BN=64 的窄块，让最后一轮的负载更均衡。它需要自定义 grid，**不是纯 config 调优**，但不依赖任何编译器或后端改动。

## 需求信息


| 项         | 内容                                                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| 需求编号      | R018（`lm_head_decode` / `lm_head_prefill`），见 [少伯 BU 需求](../../../SHAOBO_BU_REQUIREMENTS.md#r018)                   |
| M × N × K | 12 × 248320 × 2048                                                                                                 |
| 阶段 / 来源数  | prefill / decode；来源数 3                                                                                             |
| 模型 / 配置   | 需求表未填写（—）                                                                                                          |
| 精度        | BF16 × BF16 → BF16，FP32 累加                                                                                         |
| 对接接口      | [gemm_a16w16](../../../../../boltops/gemm/triton/gemm_a16w16.py)（权重布局 `[N,K]`）。本次直接测 kernel，不是生产接口端到端测试，特化版也未接入该接口 |
| 需求表当前状态   | 接口 smoke 通过（尚未回填调优结果）                                                                                              |




## 结论

在各自独立的 PMD 进程、相同种子（29）下，特化版把完整 kernel 耗时从 **584.986273 µs 降到 579.665502 µs**，降低 **0.91%**（1.0092×）；MMAC 利用率从 **4.8297% 升到 4.8741%**，有效带宽从 **1.7490 TB/s 升到 1.7650 TB/s**（配置带宽的 97.18% → 98.08%）。

该 shape 的 B 矩阵占逻辑数据量的 99.4%，算术强度 11.9295 FLOP/B，按本仓库口径属于访存受限，带宽屋顶线为 **21.4689 TFLOPS**；通用版已经达到屋顶线的 97.18%，所以可优化空间本来就很小。特化版达到 98.08%。数据量除以配置带宽为 568.515 µs，再加上 wave 区间之外约 3.61 µs，参考值为 572.127 µs；通用版实测比它高约 2.2%，特化版高约 1.3%。这是参考，不是严格下界。

**证据强度有限**：每个版本只有 1 次预热 + 1 次计时，没有多样本收敛。同一进程内预热与计时两次 dispatch 本身相差 0.33%（通用版）/ 0.42%（特化版），与 0.91% 的收益同一量级。预热 dispatch 的对比方向一致（1,289,911 → 1,279,396 周期，降低 0.82%），但这不能替代重复测量。**目前不能把 0.91% 写成已确认的收益**；验证方案、判据（含噪声底和“kernel 代码路径”对照）见[待办](#待办)。

## 两个版本


|                                   | 通用版 `generic/`                                                                                                                                         | 特化版 `specialized_mixbn_t64/`                                                                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| kernel                            | `[_gemm_bf16_generic_kernel](../operators/gemm_a16w16_generic.py)`（所有 shape 共用，不在本目录内保存副本；SHA256 `e108b0bb…`）                                          | `[r018_mixbn_kernel.py](specialized_mixbn_t64/r018_mixbn_kernel.py)`（随本目录归档，SHA256 `0d86027b…`）                                                      |
| 算法                                | 动态 M/N/K，单 kernel；NUM_KSPLIT=1，不做 split-K                                                                                                              | 同样无 split-K、无原子、无归约 kernel；前 `n_main` 个工作组各算一个 BN=256 的主体块，其余工作组各算一个 BN=64 的尾部块；每个输出元素只被写一次                                                          |
| 配置                                | `[generic/best_config.json](generic/best_config.json)`：BM16 / BN256 / BK32，4 warps，stages 2，MLS，unroll 2，`mmac_layout_force=-1`                        | `[best_config.json](specialized_mixbn_t64/best_config.json)`：同上，另加 `TAIL_BN=64`、`CU_MULT=48`                                                         |
| 配置来源                              | `[results_TT_dyn/shapes/R018_12x248320x2048_TT.json](../autotune/results_TT_dyn/shapes/R018_12x248320x2048_TT.json)` 的 `best.config`，内容完全一致            | 通用版配置 + 两个任务映射参数；`TAIL_BN` 在缩小形状（N=26880）上比较过 32 / 64 / 128，见下文                                                                                      |
| 启动方式                              | 默认 grid：970 个 TG（`completed_wgs=970`）                                                                                                                  | 自定义 grid（`grid_fn=true`，由 kernel 文件的 `grid_programs()` 计算）并通过 `extra_constexpr` 传入 `TAIL_BN`、`CU_MULT`：960 个 BN=256 主体块 + 40 个 BN=64 尾部块 = 1000 个 TG |
| 资源                                | 58 VGPR，LDS 34816 B，spill 0                                                                                                                            | 58 VGPR，LDS 34816 B，spill 0                                                                                                                          |
| 限制                                | 无特殊限制                                                                                                                                                  | 仅验证完整形状 12×248320×2048、TT；隐含 M ≤ 16（只有一个 M tile）；没有 bias（签名保留 `bias_ptr` / `ADD_BIAS` 但未使用）；必须配套自定义 grid，否则无法启动                                      |
| 布局                                | TT（TN 未测）                                                                                                                                              | TT（TN 未测）                                                                                                                                            |
| 正确性                               | PASS：seed 29，最后一次 dispatch 的全部 2,979,840 个输出逐个与 BF16 舍入后的 FP32 参考比较（`atol=0.02, rtol=0.01`），`mismatches=0`；全输出有限、前后各 128 个哨兵不变；平均误差 0.051060207 等于舍入下限 | PASS：同一判据，同一随机输入（seed 29），`mismatches=0`，平均误差同样等于舍入下限                                                                                                |
| 性能（PMD 单 die，kernel-only，1 个计时样本） | **584.986273 µs，MMAC 利用率 4.8297%**（20.8645 TFLOPS；1,285,684 cycles），有效带宽 1.7490 TB/s（97.18%）                                                           | **579.665502 µs，MMAC 利用率 4.8741%**（21.0560 TFLOPS；1,273,990 cycles），有效带宽 1.7650 TB/s（98.08%）；较通用版降低 **0.91%**                                        |
| 带宽屋顶线                             | 本 shape 算术强度 11.9295 FLOP/B，**屋顶线 21.4689 TFLOPS**（屋顶线耗时 568.515 µs，口径见[上层 README](../README.md#性能指标公共口径)）                                             | 同通用版；实测达到屋顶线的 **98.08%**                                                                                                                             |
| 复现                                | `bash ../repro_generic/run.sh R018_12x248320x2048`（尚未用该脚本跑过，见待办）；本目录结果见 `[generic/runs/TT_s2_base_real/](generic/runs/TT_s2_base_real/summary.json)`   | 在 `specialized_mixbn_t64/` 执行 `bash run.sh --name <新名字>`，见[特化版 README](specialized_mixbn_t64/README.md)                                              |


自动调优阶段的估算值：[autotune 归档](../autotune/results_TT_dyn/shapes/R018_12x248320x2048_TT.json)按 N=122880 实测线性外推得 1,291,563 周期（587.66 µs）；本次整形实测 584.99 µs 比它低 0.45%。归档只是外推，不是整形测量。

## 指标口径说明

三类指标的口径与 [R014](../R014_12x64x2048/README.md)、[R016](../R016_12x9216x2048/README.md)、[R017](../R017_12x12288x2048/README.md) 及[上层 README](../README.md#性能指标公共口径)相同，并由 `[metrics.py](metrics.py)` 用同一组公式统一计算（`python3 -I metrics.py` 可核对本目录全部 `summary.json`）：

- **MMAC 利用率** = `(2 × M × N × K / 完整 kernel 耗时) / 432 TFLOPS`。本 shape 有效运算量 2MNK = 12,205,424,640 FLOPs；乘加计 2 次运算，不计 M 从 12 补齐到 16 的 tile padding。它是有效 GEMM 吞吐占给定峰值的比例，**不是**硬件计数器测得的 MMAC 指令活跃率。
- **有效带宽** = `D / 完整 kernel 耗时`，`D = 2 × (M×K + K×N + M×N) = 49,152 + 1,017,118,720 + 5,959,680 = 1,023,127,552 B`（A 与 B 各读一次、C 写一次，BF16 每元素 2 B）。它是逻辑数据量除以耗时，不是实测外存流量，不计缓存复用、重复读取、事务放大和 tile padding；配置带宽达成率的分母为 1.7996485 TB/s。
- **带宽屋顶线** = `min(432 TFLOPS, 11.9295 FLOP/B × 1.7996485 TB/s) = 21.4689 TFLOPS`。这是该数据访问假设下的参考，不是所有实现的严格性能上界；该 shape 的 MMAC 利用率在此口径下的参考上限约 4.97%。
- 分母统一为 **summary 中未舍入的完整 kernel 耗时** `median_us`（PMD `system.gcSimCycles` 经 `gcFrequency` 换算），不使用 SIMD 活跃时间或 wave 区间时间；README 中的四舍五入值不能反算。

硬件计数器口径的“MMAC 忙碌率”不能与本表或 R014 / R015 / R016 / R017 的“MMAC 利用率”比较。

## 瓶颈分析（PMD 计数器，最后一次计时 dispatch）


| 计数器                           | 通用版                         | 特化版                        |
| ----------------------------- | --------------------------- | -------------------------- |
| 完成的工作组数                       | 970                         | 1000                       |
| UMC 读请求数                      | 3,976,514                   | 3,978,742                  |
| 读请求折算带宽（请求数 × 256 B / 周期）     | 791.79 B/周期                 | 799.50 B/周期                |
| 各 CU 总周期（最大 / 最小）             | 1,277,689 / 1,235,414       | 1,265,952 / 1,246,177      |
| 各 CU 完成的工作组数分布                | 38 个 CU × 20 + 10 个 CU × 21 | 8 个 CU × 20 + 40 个 CU × 21 |
| `waitVmCounter` 密度            | 0.4653                      | 0.4715                     |
| 共享内存 bank conflict（每条 LDS 指令） | 0.0732                      | 0.0725                     |


- 通用版 970 个 BN=256 的块除以 48 个 CU 不整除（20 轮余 10 块），最后一轮只有 10 个 CU 在读，其余 38 个 CU 空闲。特化版把这最后 2560 列拆成 40 个 BN=64 的窄块，分给 40 个 CU，各 CU 的总周期差从 42,275 降到 19,775 周期（最大值的 3.3% → 1.6%）。
- 两版的 UMC 读请求数相差 0.06%，说明收益来自负载均衡，而不是减少读取次数。读请求 × 256 B 约为 1.018 GB，与 B 矩阵的 1.017 GB 吻合。
- 逻辑数据量除以配置带宽为 568.515 µs；每个 dispatch 另有约 3.61 µs（`firstWaveStartTick`）落在 wave 区间之外。特化版实测 579.67 µs，距这两项之和 572.13 µs 还差约 7.5 µs（1.3%），通用版差约 12.9 µs（2.2%）。**这是参考，不是严格物理下界**：区间外耗时没有被证明可以消除，缓存、写回和访存重叠都会影响解释。
- stall / wait 计数是跨 CU、SIMD 累加且可重叠的量，不能直接当作 kernel 壁钟耗时。



## 搜索过程摘要

通用版配置来自 autotune（171 次评估）。特化版的探索在 `tmp/1106/r018_opt/` 完成，覆盖 BN / BK / 流水级数 / warp 数 / 展开 / 缓存提示 / 混合 N 分块等方向，多数在缩小形状（N=12288 ~ 122880、N=26880）上用 PMD 筛选，再在完整形状上确认。与本特化版直接相关的结果：

- 缩小形状 N=26880（105 个 BN=256 块，同样不能被 48 整除）上，基线 175,803 周期；混合 N 分块 `TAIL_BN=32 / 64 / 128` 分别为 178,723 / 148,677 / 152,860 周期，因此选 `TAIL_BN=64`。小形状只用于排除明显较差的候选，最终结论以完整形状为准。
- 完整形状上，`TAIL_BN=64` 为本目录的特化版。另一个在 `TAIL_BN=64` 之上加了编译器 hook（给 B 的加载置缓存提示位）的实验，完整形状为 1,300,769 周期 / 591.849948 µs，比通用版慢约 1.2%，**没有采用，也不在本目录归档**。
- 这些筛选结果均为单次测量，不是穷举或全局最优证明；逐候选的耗时表不随本目录归档。



## 测量条件


| 项          | 内容                                                                                                                                                                                                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 执行环境       | PMD，`GPU_CHIP=sb`（gfx946），频率 2.197802 GHz（core period 455 ps）；运行记录中 `cu_count=48`，即**单 die、48 CU**（[调优记录](../../TUNING.md) 规定每 die 48 CU）。PMD 构建信息（`worker.log`）：commit 日期 Fri Oct 9 02:56:18 2026，`CoreArch:HEAD_1763(lib_ini_opt)`；详见[上层 README](../README.md#环境与版本)                                    |
| 布局         | TT：A 为 `[M,K]` 行主序，B 为 `[K,N]` 行主序                                                                                                                                                                                                                                                                        |
| 计时         | 每个版本 1 个独立进程（seed 29），1 次预热 + 1 次计时（2 次 dispatch），同一 B 重复使用；PMD `system.gcSimCycles / gcFrequency`，只含 kernel，不含 JIT 编译、分配、主机拷贝和生产接口开销。作业墙钟约 4678 s / 4748 s（通用版 / 特化版），不是单次 kernel 延迟                                                                                                                     |
| 软件         | Python 3.10，Triton 3.6.0（`release/3.6.x`）；两次运行都没有保存 Triton 提交号或 `libtriton.so` 的 SHA256，按时间（2026-11-06 / 11-07）归入[上层 README](../README.md#环境时间线) 的第二段环境（`e58481668e29fda141014f1d4b1f0f7bd82afd41`），**这是推断，不是运行时记录**。`tune_lib.py` SHA256 前缀 `9f52a590…`                                                  |
| 文件 SHA256  | 通用 kernel `e108b0bb94c719409fc0a8b387e295f1d39bf1e02d364494a1d92c5e361313fb`（与 `operators/` 和 `autotune/scripts/` 内两份一致）；特化 `r018_mixbn_kernel.py` `0d86027b1e4260b6c5bbd4b8e3009ce743f59c292723b34d9477dcf4dad532f5`；autotune 归档 JSON `bf743bfb9db07a6ba0e905bdc739da66db27921bed815ae5cfb6f796c2bfa76c` |
| BoltOPs 版本 | 工作目录的提交与改动统计没有随运行保存，是版本溯源局限                                                                                                                                                                                                                                                                               |
| 与 R016 的差异 | 精度判据相同（同一 `tune_lib.py`），但本批每个版本只有 1 次预热 + 1 次计时，且只校验最后一次 dispatch 的完整输出；R016 为 1 + 4 次计时、每次 dispatch 都全量校验                                                                                                                                                                                               |


[调优记录](../../TUNING.md) 要求每次性能测量保存环境 manifest。本批没有保存，因此数据在补齐 manifest 前不应用于性能排名；die / CU 数已核对（见上表）。

## 结果文件的来源

- `generic/runs/TT_s2_base_real/summary.json` 与 `specialized_mixbn_t64/runs/r_mix_t64/summary.json`：从 `tmp/1106/r018_opt/runs/{s2_base_real,r_mix_t64}/{result.json,case.json}` **转录并统一重算指标**得到，没有重新执行完整 GEMM；`source_run` 字段指向原运行，`case_as_run` 保留当时的完整 case（其中的路径指向 tmp，未改写）。tmp 清理后这些来源路径会失效。特化版 summary 中的 `baseline` 节点与通用版 summary 是同一份数据。
- `generic/runs/TT_ir_asm_perf/s2_base_real/` 与 `specialized_mixbn_t64/runs/TT_ir_asm_perf/r_mix_t64/`：从同一批运行复制的 `kernel.{ttir,ttgir,llir,amdgcn,json}` 与 `g/0/{0,1}/` 的 `stats.txt`、`.perf`、`.kinfo`，仅本地保留，被忽略规则排除，不提交。
- 特化包的 `bench.py` / `run.sh` 已在归档位置用 `--compile-only` 验证：两个用例都能编译，网格（970 / 1000）、VGPR（58）、LDS（34816 B）、spill（0）与原实测一致。编译产物的 `asm_sha` 与原运行不同，但同一配置连续编译两次的 `asm_sha` 也彼此不同（路径与调度顺序会改变汇编文本，指令行数基本相同：695 / 694、969 / 969），所以 `asm_sha` 不能用作复现指纹。**完整形状的 PMD 计时没有在归档位置重跑**（单次约 80 分钟）。



## 复现

```bash
REPO=/public/home/duyubo/bolt/github/BoltOPs
cd "$REPO/docs/gemm/debug_tuning/bf16_linear_gemm/R018_12x248320x2048/specialized_mixbn_t64"
bash run.sh --name my_cmp                   # 基线 + 特化版，两个独立进程，各约 80 分钟
bash run.sh --name my_chk --compile-only    # 只编译，约 20 秒
cd .. && python3 -I metrics.py              # 核对全部 summary 的指标
```



## 待办

以下实验**均未执行**：本节只是方案，下面“待验证”各项的结论目前都不能写成已确认。2026-11-07 只做了编译层面的检查（见“已做的准备检查”），没有启动任何完整形状的 PMD 仿真。

### 一、特化版的 0.91% 是收益还是噪声（最优先）

**为什么现有数据判不了。** 通用版与特化版各只有 1 个进程、1 个计时 dispatch（seed 29），差值只有 1 个点；同一进程内预热与计时两次 dispatch 本身相差 0.33%（通用版）/ 0.42%（特化版）。仓库里已有的其他数据给出的单样本波动量级相近：R016 同进程两次计时 dispatch 相差 0.47%（基线）/ 0.05%（特化版）；R004 复现包的 10 个计时样本跨度 1.8%（该 shape 只有 24 µs，相对波动更大，只作量级参考）。所以 0.91% 约为已见波动的 2–3 倍，**提示有收益，但不足以排除噪声**，也还没有排除“收益来自 kernel 代码本身不同，而不是尾部拆分”。

**需要先搞清楚的两件事。**

1. **PMD 在相同输入下是否逐周期可复现。** 若相同 seed、相同配置重跑得到完全相同的周期数，则模拟器本身没有随机性，“噪声”只来自输入/地址/缓存状态的扰动（种子只改数值，不一定改变访存地址），此时多种子的信息量比预想的小，需要改用别的扰动（见下方 B 组的解读）；若不完全相同，B 组直接给出噪声底。
2. **特化 kernel 的代码路径本身是否改变了速度。** 特化版不是“通用 kernel 加一行”，而是同一函数里两份 `_panel`（主体与尾部）：编译后指令数（`TAIL_BN=256` 的对照 959 条、`TAIL_BN=64` 的特化版 970 条）远多于通用版的 690 条，其中 `TAIL_BN=256` 的对照里 `v_mmac_f32_16x16x16_bf16` 为 80 条（两份 `_panel` 各 40），通用版 40 条（主循环本身是否逐指令等价没有核对）。因此需要一个**映射与通用版相同、但走新 kernel 文件**的对照（C 组）。

**实验矩阵。** 一个进程 = 一个独立样本；同一进程内的计时 dispatch 共用同一份 B 和缓存状态，不当作独立样本。每个进程 4 次 dispatch（1 次预热 + 3 次计时），取后 3 次的中位数作为该进程的结果。

| 组 | 内容 | 进程数 | 目的 |
| --- | --- | --- | --- |
| A | 通用版 vs 特化版（`TAIL_BN=64`），种子 29–34 各一对 | 12 | 主对比；6 对配对差 |
| B | 通用版、特化版各用 seed 29 **原样重跑一次**（与 A 中同种子的进程比较） | 2 | A/A：相同输入的两个进程是否逐周期相同；给出噪声底 |
| C | 特化 kernel + `TAIL_BN=256`（网格 970，与通用版映射相同，但走新 kernel 文件），种子 29–31 | 3 | 区分“kernel 代码不同”与“尾部拆分”的作用 |
| D | 特化 kernel + `TAIL_BN=128`（网格 980），种子 29–31 | 3 | 剂量方向：期望周期随 `TAIL_BN` 变小单调下降（256 → 128 → 64） |

合计 20 个进程，每个约 160 分钟（4 次 dispatch，每次约 40 分钟）。PMD 周期按理不受宿主机负载影响（未专门验证），可并行；本机 128 核、约 1.5 TB 内存，但**单个完整形状进程的内存占用没有测过**，建议先启动 A 组的 12 个，观察内存后再启动其余。

```bash
REPO=/public/home/duyubo/bolt/github/BoltOPs
cd "$REPO/docs/gemm/debug_tuning/bf16_linear_gemm/R018_12x248320x2048/specialized_mixbn_t64"

# A 组：6 个种子，每个种子一个基线进程 + 一个特化进程，同时写 manifest.json
bash run.sh --name conf --seeds "29 30 31 32 33 34" --dispatches 4

# B 组：seed 29 原样重跑（与 conf_baseline_s29 / conf_best_s29 逐周期比较）
bash run.sh --name aa --seeds "29" --dispatches 4

# C、D 组：run.sh 不带 TAIL_BN 选项，直接用 bench.py 和 config 覆盖（已在编译层验证：C 网格 970，D 网格 980，均为 58 VGPR、LDS 34816 B、0 spill）
python3 -I - <<'EOF'
import json
cfg = json.load(open('best_config.json'))
cases = [dict(name=f'ctl_t{t}_s{s}', shape=[12, 248320, 2048], seed=s, dispatches=4, timeout=7500 * 4,
              operator='r018_mixbn_kernel.py', grid_fn=True, extra_constexpr=['TAIL_BN', 'CU_MULT'],
              config={**cfg, 'TAIL_BN': t}) for t in (256, 128) for s in (29, 30, 31)]
json.dump(cases, open('/tmp/r018_ctl_cases.json', 'w'), indent=1)
EOF
python3 -B bench.py suite /tmp/r018_ctl_cases.json --jobs 6
```

**判据（先写死，再看数据）。** 对每个种子取配对差 `d_s = 特化中位数 / 基线中位数 − 1`。

- **噪声底 N**：取下面三者的最大值——B 组两个进程的相对差；A 组基线在 6 个种子间的（最大 − 最小）/ 均值；单个进程内 3 个计时 dispatch 的（最大 − 最小）/ 中位数。
- **判定“收益真实”需同时满足**：(1) 6 个 `d_s` 全部 < 0（符号检验双侧 p = 2/64 ≈ 0.031；只有 5 对时 p 为 0.0625，所以取 6 个种子）；(2) `d_s` 均值的配对 t 检验 95% 置信区间（自由度 5，t = 2.571）上界 < 0；(3) |`d_s` 均值| > N；(4) C 组相对基线的差在 N 以内，即收益不是 kernel 代码路径带来的；(5) 机制一致：用已有计数器估计，若耗时由最慢 CU 决定，预期降幅为 1 − 1,265,952 / 1,277,689 ≈ **0.92%**（见上文瓶颈表），新数据中的 `cu_cycles_max` 降幅应在该量级。
- **D 组**不是必要条件，只是旁证：周期按 256 > 128 > 64 单调下降才支持“尾部越窄越均衡”这一解释。
- **结论的写法**：全部满足 → 写“收益已确认，约 x%（区间 …）”，并把本节替换为结果；(1)–(3) 不满足 → 写“与噪声无法区分”，特化版标为未证实；(1)–(3) 满足但 C 组本身相对基线偏离超过 N → 收益混有 kernel 代码路径的影响，应同时报告“特化 vs 通用版”（对使用者的实际收益）与“特化 vs C 组”（尾部拆分本身的作用）。

**B 组结果的解读。** 若 B 组与 A 组同种子进程逐周期相同，说明 PMD 对相同输入是确定的：此时 A 组 6 个种子仍然有用（检验对数据值的敏感性），但噪声底 N 只能来自种子间与 dispatch 间的差异；若种子之间的周期也完全相同，则这类扰动没有信息量，需要改变地址相关因素（例如给 B 的分配加偏移），这需要改动 `bench.py`，到时再另行评估。

### 二、其他待办

1. **补环境记录。** `run.sh` 已改为在启动仿真前写出 `runs/<name>_cases/manifest.json`（BoltOPs 与 Triton 的提交、工作区改动数、`libtriton.so` 的 SHA256、clang 版本、`PATH_SH` 环境，以及两份 kernel、`tune_lib.py`、`bench.py`、`run.sh`、两份配置与 autotune 归档的 SHA256）。尚未随完整形状的计时运行过；上一节的运行产生它之后，把关键字段写进“测量条件”，并去掉“环境为推断”的说明。已有的两次转录运行（seed 29）无法补记，仍保持推断标注。
2. **用 `repro_generic` 重测通用版。** `bash ../repro_generic/run.sh R018_12x248320x2048 --layout TT --rounds 2 --dispatches 4`（种子 29、30，每轮 1 次预热 + 3 次计时）；跑完后用其 `summary.json` 替换当前的转录结果，并与上一节 A 组的基线交叉核对（同一 seed 29 应落在噪声底内）。
3. **TN 布局。** 通用版：`bash ../repro_generic/run.sh R018_12x248320x2048 --layout TN ...`，生产接口 `gemm_a16w16` 的权重布局是 `[N,K]`（即 TN）。特化版：`bench.py` 已支持在用例里写 `"layout": "TN"`（kernel 只用步长参数，两种布局编译出的网格与资源相同：970 / 1000，58 VGPR），但这只是**编译层面**的检查，TN 的正确性与性能都没测过——TN 下 B 的 `K` 方向连续，访存模式与 TT 不同，`TAIL_BN=64` 与 `CU_MULT=48` 不一定仍是合适取值。TT 的收益确认之后再做。
4. **并入通用算子或 `gemm_a16w16` 前需要决定**：自定义 grid 如何接入启动代码、`M ≤ 16` 的前置条件如何检查，以及 N 尾部不是 `TAIL_BN` 整数倍时的行为（当前 2560 列恰好是 64 的整数倍，尾部不需要掩码；其他 N 走带掩码的路径，未测）。

### 已做的准备检查（2026-11-07，仅编译，不含 PMD 计时）

- `run.sh` 的 `--seeds "29 30"`、`--compile-only`：4 个用例（2 个版本 × 2 个种子）全部编译通过，网格 970 / 1000，58 VGPR；manifest 正确写出（BoltOPs `4a03e47`、`tracked_changes=5`，Triton `e58481668e29fda141014f1d4b1f0f7bd82afd41`，`libtriton.so` SHA256 `b264c769…`，Python 3.10.12）。首次检查时发现 `run.sh` 把仓库根目录解析错一级、manifest 步骤会失败，已修正后复查。
- C、D 组配置（`TAIL_BN=256 / 128`）与 TN 布局的两个版本都已编译通过。
- 编译检查产生的 `runs/chk_*` 目录已删除，没有保留任何仿真输出。



## 目录

```text
R018_12x248320x2048/
├── README.md
├── metrics.py                          # 统一的 MMAC 利用率 / 有效带宽 / 屋顶线公式，并核对各 summary
├── generic/
│   ├── best_config.json
│   └── runs/
│       ├── TT_s2_base_real/summary.json     # 通用版 seed 29 的转录（1 预热 + 1 计时）
│       └── TT_ir_asm_perf/s2_base_real/     # 本地 IR / 汇编 / perf，不提交
└── specialized_mixbn_t64/
    ├── README.md            # 特化版的文件说明、运行方式、局限
    ├── r018_mixbn_kernel.py # kernel、grid_programs() 与公开参数（TAIL_BN、CU_MULT）
    ├── best_config.json
    ├── bench.py / run.sh
    ├── .gitignore           # 原始 PMD 输出不进 Git，只保留 summary.json
    └── runs/
        ├── r_mix_t64/summary.json           # 通用版 vs 特化版的转录与配对比较
        └── TT_ir_asm_perf/r_mix_t64/        # 本地，不提交
```

