# 特化版：R016 · 12×9216×2048 · TT 单 wave 小 N 块（运行时参数，无编译器改动）

> 上级目录的 [README](../README.md) 记录该 shape 的需求信息、通用版对照与结果汇总；性能指标口径见 [`../metrics.py`](../metrics.py)。以下命令均在本目录下执行。

目标：BF16 × BF16 → BF16，FP32 累加；逻辑 M=12 / N=9216 / K=2048；TT（A 为 `[M,K]` 行主序，B 为 `[K,N]` 行主序）；PMD gfx946（`GPU_CHIP=sb`）。无 bias、激活、split-K。

本版本只改 kernel 与配置：M、N、K 和全部 stride 都是普通运行时参数，也不使用 `slc` 缓存策略位，因此不依赖任何编译器改动，也不需要进程内后端选项。“M、N、stride 静态化 + `slc` 缓存策略位”是在此基础上的进一步方案，见上级 README 的“补丁方案说明”，本目录不保存其代码和结果。

## 文件

- `r016_kernel.py`：kernel 与公开入口。M、N、K 和所有 stride 为运行时参数（`do_not_specialize=['M','N','K']`，保留 `tl.assume` 的正值假设）；B 的 N 偏移在每个工作组外层一次性重定位（`b_ptr + on`）；工作组沿 N 一维排列；A 的 12 行有效数据保留 `matrix_load` 的 M 边界检查与零填充，C 保留 M 尾部掩码。入口 `gemm_r016(a, b, out=None)` 只接受完整目标形状、同一 gfx946 设备上的 BF16 TT 输入（行内连续、行间不重叠），输出必须连续，并在启动前拒绝与输入重叠的输出。kernel 本身没有硬编码 M=12，但仅在完整形状上验证过。
- `best_config.json`：扁平配置，BM16 / BN32 / BK64，`GROUP_SIZE_M=1`，`num_warps=1`，stages 2，MLS，`mmac_layout_force=3`，`schedule_hint="local-prefetch"`，`LOOP_UNROLL_FACTOR=1`。
- `bench.py`、`reproduce.py`、`runtime_checks.py`：隔离的 PMD 计时与全量校验脚本。依赖仓库内 [`../../autotune/scripts/tune_lib.py`](../../autotune/scripts/tune_lib.py) 和 [`../../autotune/results_TT_dyn/shapes/R016_12x9216x2048_TT.json`](../../autotune/results_TT_dyn/shapes/R016_12x9216x2048_TT.json)（基线配置），`bench.py` 的 `REPO` 按本包位置解析为当前 checkout。
- `run.sh`：启动 `reproduce.py`；Triton / clang / PMD 环境由 `tune_lib.py` 内的 `PATH_SH` 加载，与本机绑定。
- `runs/`：见下文“结果文件”。

## 运行

每个 `--name` 必须是新的，已有 `runs/<name>/` 会拒绝覆盖。默认 5 次 dispatch，即 1 次预热 + 4 次计时，逐次全量校验。

```bash
# 同时启动独立进程的基线（通用 kernel + 原 best.config）与特化版，约 5–6 分钟
bash run.sh --name my_cmp_seed29 --compare --seed 29 --dispatches 5

# 每次 dispatch 用不同的 B（所有 B 保持存活，不显式清缓存）
bash run.sh --name my_rotate_seed32 --compare --seed 32 --dispatches 3 --rotate-b

# NaN 行尾 padding：只校验末次输出，不作性能对照
bash run.sh --name my_pad_seed31 --seed 31 --dispatches 2 --poison-pad
```

输出在 `runs/<name>_baseline/` 与 `runs/<name>_best/`，含 `result.json`（含逐 dispatch 计数器）、`check.json`、`dispatch_checks.json`、`compile.json`、原始 `g/0/<n>/stats.txt`、IR / 汇编和 `worker.log`。`runs/*/*` 被 [.gitignore](.gitignore) 忽略，仅 `summary.json` 进 Git。

## 正确性及计时判据

- 种子化 CPU BF16 随机输入（未缩放 `randn`），CPU FP32 matmul 为参考，转回 BF16 得到逐元素比较目标 `gold`。
- **每次 dispatch** 检查全部 **110592** 个输出：有限值；`torch.isclose(got, gold, atol=0.02, rtol=0.01)` 的不匹配数为 0；前后各 128 个哨兵元素不变；`mean(|got-ref|) ≤ 1.05 × mean(|gold-ref|) + 1e-6`。这里是容差内通过，不要求与 CPU 参考逐位相同（`bitwise_equal_to_bf16_gold=false`）。判据与通用版 `tune_lib.py::worker()` 及 R017 相同。
- 计时为 PMD `system.gcSimCycles / (gcFrequency × 1000)`，只含 kernel，不含编译、分配、复制和 CPU 校验。第 1 次 dispatch 为预热，其余取中位数。
- 公开入口 `gemm_r016` 会拒绝形状、dtype、设备、非行主序 stride 和与输入重叠的输出。

## 结果文件

| 文件 | 性质 | 内容 |
| --- | --- | --- |
| [`runs/final/summary.json`](runs/final/summary.json) | 本包实跑（2026-11-07） | 基线与特化版各 2 个进程（seed 29 / 30，各 4 个计时样本）、合并中位数、配对加速、独立 B（seed 32）、NaN padding（seed 31）、seed 29 最后一次计时 dispatch 的计数器 |
| `runs/TT_ir_asm_perf/final29_best/` | 本地复制 | `kernel.{ttir,ttgir,llir,amdgcn}` 与 `g/0/{0..4}/` 的 `stats.txt`、`stats_xcd0.perf`、`.perf`、`.kinfo`，来自 seed 29 的特化版运行；被忽略规则排除，不提交 |

所有 `summary.json` 中带 `metrics` 的节点都由 [`../metrics.py`](../metrics.py) 的同一组公式生成，可用 `python3 -I ../metrics.py` 重新核对。其余原始运行目录（缓存、hsaco、各 dispatch 的完整输出）在汇总后已删除。

## 局限

- PMD 是模拟器，结果只代表 gfx946 / sb 的模拟配置，不是真机性能，也不是全局最优证明。
- 只验证了 12 × 9216 × 2048 的 TT 连续输入；TN 布局、其他 M/N/K、bias、激活均未验证。kernel 对 M ≤ 16、N % 32 == 0、K % 64 == 0 有隐含要求，入口未对其他形状放行。
- 运行时参数的 kernel 与静态化相比，早期消融（开启 `slc` 的前提下，结果保留在 tmp）显示相差约 0.17%，在单次 dispatch 波动量级；该数值不是本目录实测。
