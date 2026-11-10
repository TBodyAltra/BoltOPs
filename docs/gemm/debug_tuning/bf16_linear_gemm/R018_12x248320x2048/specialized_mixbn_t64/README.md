# 特化版：R018 · 12×248320×2048 · TT 混合 N 分块（主体 BN=256 + 尾部 BN=64，无编译器改动）

> 上级目录的 [README](../README.md) 记录该 shape 的需求信息、通用版对照与结果汇总；性能指标口径见 [`../metrics.py`](../metrics.py)。以下命令均在本目录下执行。

目标：BF16 × BF16 → BF16，FP32 累加；逻辑 M=12 / N=248320 / K=2048；TT（A 为 `[M,K]` 行主序，B 为 `[K,N]` 行主序）；PMD gfx946（`GPU_CHIP=sb`）。无 bias、激活、split-K。

## 要解决的问题

该 shape 是纯外存读带宽受限（B 约 1.017 GB，只读一次）。通用版用 BN=256 的块，共 248320 / 256 = 970 个工作组，48 个 CU 跑 20 轮后还剩 10 个块：最后一轮只有 10 个 CU 在读，其余 38 个空闲，读带宽只用到整机的约 1/5。

本版本的做法：前 `n_main = (N // BLOCK_SIZE_N) // CU_MULT * CU_MULT = 960` 个工作组保持 BN=256（正好 20 轮 × 48 CU），剩下的 `248320 − 960 × 256 = 2560` 列拆成 `TAIL_BN=64` 的窄块，共 40 个工作组，让最后一轮分散到更多 CU 上、每个 CU 读的字节数更少。总 grid = 960 + 40 = 1000。每个输出元素只被写一次，没有 partial、原子或归约 kernel。

## 文件

- `r018_mixbn_kernel.py`：kernel 与 `grid_programs()`。`_gemm_bf16_generic_kernel` 保持与通用版相同的参数签名，另加两个 constexpr：`TAIL_BN`、`CU_MULT`。`_panel()` 计算一个 `[BM, BN]` 输出块的完整 K 归约，加载与存储结构同通用版（`USE_MLS` 时用 `matrix_load`）。`grid_programs(M, N, K, BLOCK_SIZE_N, TAIL_BN, CU_MULT, **_)` 在主机侧给出工作组总数，**kernel 与 grid 公式在同一文件，不会各自漂移**。与 `tmp/1106/r018_opt/kernels/k1_mixbn.py` 内容完全相同（SHA256 `0d86027b…`），只改了文件名。
- `best_config.json`：扁平配置，BM16 / BN256 / BK32，`GROUP_SIZE_M=1`，`num_warps=4`，stages 2，MLS，unroll 2，`mmac_layout_force=-1`，`TAIL_BN=64`，`CU_MULT=48`。除后两项外与通用版 `generic/best_config.json` 相同。
- `bench.py`：隔离的 PMD 计时与全量校验脚本，由 `tmp/1106/r018_opt/bench.py` 归档而来。依赖仓库内 [`../../autotune/scripts/tune_lib.py`](../../autotune/scripts/tune_lib.py)、[`../../autotune/results_TT_dyn/shapes/R018_12x248320x2048_TT.json`](../../autotune/results_TT_dyn/shapes/R018_12x248320x2048_TT.json)（基线配置）和 [`../../operators/gemm_a16w16_generic.py`](../../operators/gemm_a16w16_generic.py)（基线 kernel）；`REPO` 按本包位置解析为当前 checkout，也可用 `BOLTOPS_REPO` 覆盖。与原文件相比只改了路径和默认形状，并删除了本次未用到的 `hook` / `pre_import` 选项（编译器层面的实验），测量代码不变。
- `run.sh`：生成 `runs/<name>_cases/cases.json` 并启动 `bench.py suite`；Triton / clang / PMD 环境由 `tune_lib.py` 内的 `PATH_SH` 加载，与本机绑定。
- `runs/`：见下文“结果文件”。

## 为什么需要自定义 grid

`bench.py` 对带 `grid_fn=true` 的用例，在**本进程**内把 `tune_lib` 的 `num_tg` 替换为 kernel 文件的 `grid_programs()`，并把 `TAIL_BN`、`CU_MULT` 追加到传给 kernel 的 constexpr 名单（`extra_constexpr`）。这只是测试进程内的适配，不修改已安装的 Triton / 编译器，也不修改公共测试库文件。**只把 config 交给原始通用启动器无法复现这一版本**——它会按 BN=256 启动 970 个工作组，尾部 2560 列将不被计算。其他调用方需要等价实现这两点。

## 运行

每个 `--name` 必须是新的，已有 `runs/<name>_*/` 会拒绝覆盖。完整形状的 PMD 运行每次 dispatch 约 40 分钟（原始 2 次 dispatch 的墙钟 4678 s / 4748 s），建议放到后台。

```bash
# 基线（通用 kernel + 通用版配置）与特化版，两个独立进程同时启动（默认 seed 29，2 次 dispatch = 1 预热 + 1 计时）
bash run.sh --name my_cmp

# 多种子、更多计时 dispatch：每个（版本，种子）一个独立进程，4 = 1 预热 + 3 计时
bash run.sh --name conf --seeds "29 30 31 32 33 34" --dispatches 4

# 只跑其中一个
bash run.sh --name my_best --only best
bash run.sh --name my_base --only baseline

# 只编译，不跑 PMD（约 20 秒），用来检查网格、寄存器和 LDS：期望网格 970 / 1000，58 VGPR，LDS 34816 B，spill 0
bash run.sh --name my_chk --compile-only
```

可选参数还有 `--jobs N`（并行进程数，默认等于用例数）。启动仿真前会写出 `runs/<name>_cases/manifest.json`（BoltOPs / Triton 提交、`libtriton.so` 与各文件的 SHA256、编译器版本）。多种子实验的设计与判据见[上层 README 的待办](../README.md#待办)。

输出在 `runs/<name>_baseline_s<seed>/` 与 `runs/<name>_best_s<seed>/`，含 `result.json`（含逐 dispatch 计数器）、`check.json`、`compile.json`、`case.json`、原始 `g/0/<n>/stats.txt`、IR / 汇编和 `worker.log`。`runs/*/*` 被 [.gitignore](.gitignore) 忽略，仅 `summary.json` 进 Git；`run.sh` 产生的是 `result.json`，需要归档时按 `runs/r_mix_t64/summary.json` 的格式整理。

## 正确性及计时判据

- 种子化 CPU BF16 随机输入（未缩放 `randn`，seed 29），CPU FP32 matmul 为参考，转回 BF16 得到逐元素比较目标 `gold`。
- 所有 dispatch 结束后检查最后一次的**全部 2,979,840 个输出**：有限值；`torch.isclose(got, gold, atol=0.02, rtol=0.01)` 的不匹配数为 0；前后各 128 个哨兵元素不变；`mean(|got-ref|) ≤ 1.05 × mean(|gold-ref|) + 1e-6`。这是容差内通过，不要求与 CPU 参考逐位相同。判据与通用版 `tune_lib.py::worker()` 相同。
- 计时为 PMD `system.gcSimCycles / (gcFrequency × 1000)`，只含 kernel，不含编译、分配、复制和 CPU 校验。默认 2 次 dispatch：第 1 次为预热，第 2 次为计时样本。

## 结果文件

| 文件 | 性质 | 内容 |
| --- | --- | --- |
| [`runs/r_mix_t64/summary.json`](runs/r_mix_t64/summary.json) | 转录（来自 `tmp/1106/r018_opt/runs/{s2_base_real,r_mix_t64}`） | 通用版与特化版各自的周期 / 耗时 / 指标 / 正确性 / 资源 / 最后一次计时 dispatch 的计数器、两次 dispatch 的配对比较与屋顶线参考值 |
| `runs/TT_ir_asm_perf/r_mix_t64/` | 本地复制 | `kernel.{ttir,ttgir,llir,amdgcn,json}` 与 `g/0/{0,1}/` 的 `stats.txt`、`stats_xcd0.perf`、`.perf`、`.kinfo`；被忽略规则排除，不提交 |

`summary.json` 中带 `metrics` 的节点都由 [`../metrics.py`](../metrics.py) 的同一组公式生成，可用 `python3 -I ../metrics.py` 重新核对。

核心数据（每个版本 1 次预热 + 1 次计时）：通用版 1,285,684 周期 / 584.986273 µs；特化版 1,273,990 周期 / 579.665502 µs，降低 0.91%。预热 dispatch 的对比同向（1,289,911 → 1,279,396 周期，降低 0.82%），但同一进程内预热与计时本身相差 0.3–0.4%，**收益尚未经过多样本收敛确认**。

## 局限

- PMD 是模拟器，结果只代表 gfx946 / sb 的模拟配置，不是真机性能，也不是全局最优证明。
- 只验证了 12 × 248320 × 2048 的 TT 连续输入；TN 布局、其他 M/N/K、bias、激活均未验证。
- 仅覆盖一个 M tile：BM=16，要求 0 < M ≤ 16，入口没有检查。没有 split-K，没有 bias（`bias_ptr` / `ADD_BIAS` 只为保持签名，未使用）。
- `TAIL_BN=64` 的选择依据是缩小形状（N=26880）上的比较（32 / 64 / 128 对应 178,723 / 148,677 / 152,860 周期，基线 175,803 周期），完整形状上只测了 64。`CU_MULT=48` 对应当前 48 CU 的模拟配置，换 CU 数需重新取值。
- 当前 N 的尾部 2560 列恰为 64 的整数倍，尾部块不需要 N 方向掩码（`_TAIL_EXACT`）；不是整数倍时会走带掩码的路径，该路径未测。
- 该版本改了分块与任务映射，**不能归类为纯 config 调优**。
