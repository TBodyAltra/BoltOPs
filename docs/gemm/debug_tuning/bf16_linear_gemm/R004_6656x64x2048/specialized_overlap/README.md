# R004 特化交付包：重叠 MLS 读取（6656×64×2048，TT BF16）

本目录交付独立内核、固定配置和独立验证/计时入口，以及两份既有结果汇总。执行代码不依赖原实验目录或 staging 的测试脚本。运行需要具备 `matrix_load` 扩展的 Triton 3.6、PyTorch、PMD gfx946 和对应工具链；`--check-only` 仅需 Python 3.10+ 标准库，不加载 GPU 运行时，也不创建运行目录。

## 交付文件

- [kernel.py](kernel.py)：原 `best_kernel.py` 的逐字节副本，SHA-256 为 `be5ab01dda606e8f611c23bf7b1823e9c42a309c4470251073251eabe2d0f915`。
- [best_config.json](best_config.json)：与原 `final_cases.json` 的 seed29/143 特化项及 `v2final_cases.json` 的 `v2final_oldbest_a/b` 配置一致。
- [bench.py](bench.py)：独立启动本目录内核并全量验证；不导入原实验的 `bench.py`、`tune_lib.py` 或仓库算子模块。
- [run.sh](run.sh)：标准入口，在隔离子进程中初始化工具链与 PMD；离线检查直接使用标准库。
- [.gitignore](.gitignore)：忽略未来实跑产生的每轮原始输出、编译缓存及 Python 缓存。
- [历史结果汇总](runs/verification_v2final_recorded/summary.json)：原 `r004_opt` 结果转录，保留来源、配置、样本、资源和全量正确性指标，标记 `record_type=recorded_existing_results`、`results_were_executed_by_this_package=false`。
- [staging smoke 汇总](runs/package_smoke_seed29_20261106/summary.json)：刚完成的 staging 真实实跑汇总的原样副本，**仅交付 summary.json**。其中执行时间、`results_directory`、源码哈希和 `fresh_package_run` 等字段仍描述原 staging 执行，复制到正式目录不代表又执行了一次。

正式交付不包含缓存、测试脚本或每轮日志。另有用于本地查看、不提交的 `runs/TT_ir_asm_perf/`，其中包含从原运行复制的 IR、汇编、Perf 和 `.kinfo`；检查和运行不依赖这些本地产物。

## 支持范围和算法

仅支持 A 为连续 BF16 `[6656,2048]`、B 为连续 BF16 `[2048,64]`、C 为 BF16 `[6656,64]`；TT 在本包中表示上述逻辑行主序布局。虽然内核签名含运行时 M/N/K，本交付版不承诺动态 shape，不做矩阵加载边界检查，也没有 bias、激活或额外归约。

每个程序计算一个 64×64 的完整矩阵加载窗口，144 个程序按 16 行对齐分配输出行所有权，每个程序仅存储自己拥有的 32 或 48 行。读取窗口重叠，输出行不重叠；最后窗口向前移动以保证矩阵加载在界内。`SK_SPLIT=16` 是所有权行对齐数，不是 split-K；没有原子操作或归约 kernel。接口保留的 workspace/counter 均不使用。

## 离线检查和新执行

以下命令均在本目录中执行：

```bash
python3 -B bench.py --check-only
bash run.sh --check-only
```

离线检查核对必要交付文件、Python 语法、内核 SHA-256、完整固定配置、历史汇总内部一致性，以及全部 6656 行的唯一所有权和矩阵加载范围。它只读取本包文件，不读取 staging 的 smoke 原始产物或原 `r004_opt` 数据，不表示重新编译、执行内核或取得新的正确性/性能结果。即使指定 `--out`，`--check-only` 也不会写文件。

需要新执行时：

```bash
bash run.sh --rounds 1 --warmup 1 --reps 2 --seed 29
bash run.sh --rounds 1 --warmup 1 --reps 1 --seed 143 --out runs/my_new_seed143_run
```

默认 `bash run.sh` 使用两个独立进程，每轮预热一次、正式两次。默认输出是新的 `runs/<时间戳>/`，指定的输出目录同样必须尚不存在；不会覆盖已交付的两份汇总。新运行才会生成 `round_*/` 中的 `result.json`、`checks.json`、`compile.json`、`command.json`、`environment.json`、`run.log`、源码快照、生成代码和 `g/*/*/stats.txt`。现有 smoke 汇总旁没有这些文件；原始产物位置见下节。

工具链默认使用 `/public/home/duyubo/triton_all_version/triton_3.6/triton/path.sh`、clang-18 和 `ld.lld`。其他机器可覆盖环境变量，无需编辑共享文件：

```bash
TRITON_PATH_SH=/path/to/triton/path.sh \
OPT_CLANG=/path/to/clang-18 OPT_LLD=/path/to/ld.lld \
OPT_HSA_TOOLS_LIB=/path/to/libperf_gen_helper.so \
PYTHON=/usr/bin/python3 bash run.sh --rounds 1 --warmup 1 --reps 1
```

`path.sh` 需导出 Triton build 的 `PYTHONPATH`、`PMD_PATH` 和 `GPU_CHIP`。bench 在干净环境中读取它，不继承外部 `PYTHONPATH` 的实验目录；每轮隔离 cwd、HOME、TMPDIR、Triton/PyTorch/XDG 缓存及 PMD 输出，显式设置 `HSA_TOOLS_LIB`。GPU 运行时只在隔离 worker 中导入。`--timeout` 默认每轮 1800 秒，超时或中断会终止该轮进程组。

`--verify-source` 是可选的原实验来源审计选项，只有原 `r004_opt` 数据仍在本机时才能使用，会读取外部原始结果和 PMD 文件；它不是默认检查或执行的依赖。

## 正确性及计时判据

- `NUM_PROGRAMS=144` 用作 grid，并作为最后一个普通参数 `NUM_DP` 传入，不作为编译选项。constexpr 参数与 `HIPOptions` 分开传递，未知编译选项直接报错。
- BM/BN/BK=64/64/64，配置 `num_warps=4`；开启 WARP_SPECIALIZE 后原编译元数据为 5 个 wave，不应把配置随意改成 5。
- 使用 seed 指定的 CPU BF16 随机输入，CPU 全矩阵 FP32 matmul 为参考，并转回 BF16 得到逐元素比较目标。每次 dispatch 前重新用 NaN 填充全部 C。
- 每次预热和正式 dispatch 均检查全部 **425,984 个输出元素**：有限值、BF16 舍入参考逐元素 `atol=0.02, rtol=0.01`、相对未舍入 FP32 参考的平均绝对误差不超过 `1.05 × BF16 舍入误差均值 + 1e-6`。
- 每次检查 C 的前后 128 元素保护区、未使用 workspace（含 NaN 内容和前后保护区）及 counter（含前后保护区）未变化。
- 显式 `kernel.warmup(...)._init_handles()` 后按原 R004 的 `kernel[grid]` 路径执行；要求每次恰好一个 dispatch、144 个完成的 workgroup 和 48 个活跃 CU。
- 计时为 PMD `system.gcSimCycles / (gcFrequency × 1000)`，只统计正式 dispatch 的内核时间，排除编译、分配、复制、CPU 校验和宿主机模拟耗时。每次 CPU 校验可能改变缓存或调度条件，新计时应独立记录。

## 结果性质与原始产物位置

### 原实验历史结果转录

[历史汇总](runs/verification_v2final_recorded/summary.json) 的来源根目录为 `/public/home/duyubo/bolt/github/BoltOPs/tmp/1106/r004_opt/`。两次复测真实文件为该目录下的 `runs/v2final_oldbest_a/result.json`、`runs/v2final_oldbest_b/result.json`，对应基线为 `runs/v2final_baseline/result.json`；它们原本不是 `summary.json`。源码快照和日志仍位于各原运行目录内；部分中间表示、汇编和性能产物复制在 `runs/TT_ir_asm_perf/`。

- `v2final_oldbest_a` 的正式样本为 53,269 / 53,097 cycles，中位数 **24.198267 µs**；`v2final_oldbest_b` 为 53,386 / 53,360 cycles，中位数 **24.284717 µs**。四个样本合并中位数 **24.258100 µs**。
- 对应历史基线 `v2final_baseline` 为 54,666 / 54,519 cycles，中位数 **24.839590 µs**；历史特化合并中位数约快 2.34%（约 1.024×）。差异较小，不推断跨环境稳定收益。基线转录另存于 [generic 汇总](../generic/runs/TT_v2final_recorded/summary.json)。
- 原 `final_cases.json` 的 seed29 特化 **24.504937 µs**，对应 `final_baseline_seed29` **25.090522 µs**；seed143 特化 **24.833447 µs**，仅一个正式样本，没有声称配对的 seed143 基线。这些样本没有混入上述四样本合并值。
- 原 autotune `best.val` 为 **24.874397 µs**（54,669 cycles），仅单独保留来源记录，不替代配对复测基线。其源文件为仓库的 `docs/gemm/debug_tuning/bf16_linear_gemm/autotune/results_TT_dyn/shapes/R004_6656x64x2048_TT.json`。
- 6 个原来源运行均记录 PASS、全量 425,984 点和 0 mismatch。原 worker 对最后一次输出全量检查；交付 bench 对每次 dispatch 全量检查。

### staging 已完成的 smoke：仅交付汇总

[smoke 汇总](runs/package_smoke_seed29_20261106/summary.json) 记录于 2026-11-06 06:23–06:31 UTC，在 staging 执行一轮、一次预热、两次正式，seed29。三次 dispatch 均全量检查通过且 0 mismatch；正式样本 53,010 / 53,220 cycles，中位数 **24.167327 µs**。资源为 LDS 40,960 B、57 寄存器、0 spill、元数据 5 wave。本次 smoke 没有重跑基线，不与旧基线组成新的加速比结论。

该次原始产物目录为：

`/public/home/duyubo/bolt/github/BoltOPs/tmp/1106/r004_delivery_stage/specialized_overlap/runs/package_smoke_seed29_20261106/`

staging 中的 `round_0/` 保留原 `result.json`、逐次 `checks.json`、工具链及环境记录、日志、生成代码和 `g/0/0..2/stats.txt`。本目录保留该次 `summary.json`，并将 IR、汇编、Perf 和 `.kinfo` 复制至本地 `runs/TT_ir_asm_perf/package_smoke_seed29/`；summary 中的 `results_directory` 仍指向 staging。本次整理没有重新运行 GEMM。

## 正式目录验证

正式包可用标准库独立执行 `python3 -I -S -B bench.py --check-only`，也可在工具链脚本不可用时执行 `bash run.sh --check-only`。两者均不依赖未交付的测试脚本、缓存、smoke 每轮文件或原始 PMD 数据。README 中的相对文件链接只指向本次交付的文件；绝对来源路径仅用于说明与可选审计。
