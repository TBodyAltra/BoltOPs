# BF16 linear（表 3）调优记录

对应[少伯 BU 需求](../../SHAOBO_BU_REQUIREMENTS.md)中的 linear，BF16 × BF16 → BF16；目录使用新版 R 编号，旧版表 3 的 G 编号只作历史追溯。比较口径与 manifest 要求见 [调优记录](../TUNING.md)。

覆盖核对（2026-11-05）：`autotune/shapes_priority_table3.json` 仍为旧版表 3 的 27 个 shape，并未覆盖当前全部 P0/P1 需求。按算子、输入/输出精度与 M/N/K 匹配，P0 覆盖 17/37 行（R001–R004、R014–R026），缺 R027–R046；P1 覆盖 0/84 行，缺 R047–R061、R131–R142、R199–R215、R281–R305、R312–R316、R335–R344。按 M/N/K 去重后，P0 为 16/36，P1 为 0/83；JSON 另有 11 个 shape 不属于当前这类需求。本次仅核对覆盖情况，未扩充 JSON 或改写其中的旧 G 编号。

## 环境与版本

本目录的结果统一使用下表的软件栈。每次复现的 `summary.json` 会再记录当次实际的 commit 与文件 SHA256，以 `summary.json` 为准。

| 项 | 内容 |
| --- | --- |
| 执行环境 | PMD，`GPU_CHIP=sb`（gfx946），请求频率约 2.198 GHz（455 ps/周期）；单 die、48 CU 沿用 [调优记录](../TUNING.md) 的口径，本目录的 summary 未单独记录 die/CU 数 |
| Triton | 3.6，源码 `release/3.6.x`，commit `cdf6827b53b1a59104da41a1bfc7736134b13a2e`，工作区无改动；`/public/home/duyubo/triton_all_version/triton_3.6/triton`，构建目录 `build/lib.linux-x86_64-3.10`，`libtriton.so` SHA256 `919b51f1…42caa7`（构建时间 2026-10-31） |
| 编译器 | clang-18 `/opt/dtk/aillvm/bin/clang-18`，`ld.lld` `/opt/dtk/aillvm/bin/ld.lld` |
| BoltOPs（通用版复现） | commit `673ede654b7cda290f68ce98665fb6f1dcb1efd8`，分支 `duyubo/bf16-gemm-tuning`；本次 TT/TN summary 均记录 `tracked_changes=2`（测量时已有 `tune_lib.py` 的用户改动及 `repro.py` 修改），不是干净工作区；脚本和旧结果已纳入版本管理 |
| 计时口径 | PMD `system.gcSimCycles` 经 `gcFrequency` 换算为 µs，只含 kernel，不含 JIT 编译、分配和主机拷贝 |

通用版当前精度判据以 `autotune/scripts/tune_lib.py::worker()` 为准：CPU FP32 参考为 `ref`，先舍入成 BF16 得到 `gold`，再用 `torch.isclose(got, gold, atol=0.02, rtol=0.01)` 统计 `mismatches`，要求为 0；同时要求 `mean(|got-ref|) ≤ 1.05 × mean(|gold-ref|) + 1e-6`、输出全为有限值、输出前后各 128 个哨兵值不变。`repro_generic` 设置 `full_check=True`，检查全部输出；autotune 未设置时只对抽样点计算参考与误差，有限值及哨兵仍检查全输出。`repro.py` 本地记录上述容差，`tune_lib.py` 不导出精度常量，也不再输出最大绝对误差。

R014 fused split-K 特化版保留独立的历史判据：BF16 输出对未舍入 FP32 参考检查 `rtol=0.01, atol=0.001`，另检查 FP32 partial（`rtol=atol=1e-4`）、保护区和计数器复位；本次未重测。它与通用版判据及输入不同（通用版为未缩放 `randn`，R014 特化版缩放 0.25），不能声称精度口径统一。R004 新增 `specialized_static_mnk_strides/` 则使用与 `repro_generic` 一致的种子 29 / 30、未缩放输入、全量 BF16 gold / 平均误差 / finite / 128 哨兵判据；两类特化包的判据不能混用。

## 目录约定

```text
bf16_linear_gemm/
├── operators/                    # 可被多个 shape 共用的 kernel
│   └── gemm_a16w16_generic.py    # 通用版：动态 M/N/K，由 config 决定 tile、流水和 split-K
├── autotune/                     # 调优/验证脚本；results_TT_dyn/shapes/ 归档动态 kernel 的 TT 结果（14 个 JSON）
├── repro_generic/                # 通用版复现脚本：任意 R<编号>_<M>x<N>x<K> 目录都可以用
│   └── run.sh / repro.py
└── R<编号>_<M>x<N>x<K>/          # 每个需求行一个目录，例如 R014_12x64x2048
    ├── README.md                 # 需求信息、各版本对照、待办
    ├── generic/
    │   ├── best_config.json      # 通用版配置
    │   └── runs/<layout>_<时间>/summary.json   # repro_generic 的复现结果（round_*/ 原始输出不进 Git）
    └── specialized_<特征>/       # 特化版：kernel、config、计时脚本自成一包
```

- 目录名用当前需求表的 R 编号加 M×N×K（R014 对应旧 G256，R015 对应旧 G261）；旧 G/C 编号只用于历史追溯。历史 `summary.json` 中的编号和绝对路径保留测量时原值，新复现使用 R 编号。
- 每个版本的 `best_config.json` 保持扁平 JSON，只放传给 kernel 的参数，不加结果字段。结果写在 README，或保留该版本的运行 `summary.json`。
- 特化版的 kernel 随该 shape 归档。若多个 shape 复用同一个特化 kernel，把它提升到 `operators/`。
- `autotune/results_TT_dyn/shapes/` 的 14 个 JSON 从仍在运行的动态 kernel TT 调优作业复制归档，并非全部候选或最终完整结果；真实配置在 `best.config`，不能把整个结果对象作为扁平配置。这 14 个配置尚未全部按当前判据重新验证；下文 R014 的复现使用其自身 `generic/best_config.json`，不代表验证了该归档。
- `autotune/results_TT_dyn/shapes/` 中能匹配当前 BF16 linear 需求的文件使用 `R<编号>_<M>x<N>x<K>_TT.json`；同一 shape 对应多个 R 编号时取首次出现的编号，例如 `R004_6656x64x2048_TT.json` 同时适用于 R004、R023。其余 6 个旧版 shape（12×256×2048、12×1024×2048、12×2048×512、64×256×2048、2048×256×2048、2560×256×2048）在新版中没有对应 R 编号，保留原文件名；所有 JSON 内容保持不变。
- 原始 PMD 输出（log、stats、hsaco、perf、IR/汇编）不进 Git，规则同 [TUNING.md](../TUNING.md)；本目录的 [.gitignore](.gitignore) 已忽略各版本（`generic/`、`specialized_*/`）`runs/*/` 下的 `round_*/`，只提交各次运行的 `summary.json`。

## 复现通用版

```bash
cd repro_generic
bash run.sh R014_12x64x2048                  # TT，2 个独立进程，每个进程预热 1 次 + 正式 5 次，约 1 分钟
bash run.sh R014_12x64x2048 --layout TN      # TN：权重 W[N,K]，K 连续，对应 gemm_a16w16 接口布局
bash run.sh R014_12x64x2048 --config /abs/other_config.json   # 扁平配置，仍写入该 shape 的 generic/runs/
bash run.sh R014_12x64x2048 --config ../autotune/results_TT_dyn/shapes/R014_12x64x2048_TT.json  # 自动取 best.config
```

结果写入 `R014_12x64x2048/generic/runs/<layout>_<时间>/summary.json`，包含配置、正确性、cycles / µs、寄存器与 LDS、BoltOPs 与 Triton 的 commit 及相关文件 SHA256。传入结果 JSON 时仍以 `shape_dir` 决定 M/N/K、`--layout` 决定布局，归档元数据不符会打印警告。`tune_lib.py` 中的 `REPO`、`PATH_SH`、`PYTHON`、`CLANG`、`LLD` 及 Triton/PMD 环境路径均与本机绑定，换机器必须调整；`repro.py` 会覆盖 `REPO` 为当前 checkout，亦可用 `BOLTOPS_REPO=/path/to/BoltOPs` 指定。`AUTOTUNE_CLANG` 可覆盖编译器路径。

## 已记录的 shape

| 编号 | M × N × K | 通用版 | 特化版 | 目录 |
| --- | --- | --- | --- | --- |
| R004 / R023 | 6656 × 64 × 2048 | 当前判据 PASS；TT 29.486050 µs，TN 38.909784 µs | 无（通用版） | [R004_6656x64x2048](R004_6656x64x2048/README.md) |
| R014（旧 G256） | 12 × 64 × 2048 | 当前判据 PASS；TT 7.537531 µs，TN 7.180128 µs | fused split-K，历史 6.59 µs（TT，独立判据，未重测） | [R014_12x64x2048](R014_12x64x2048/README.md) |
| R015（旧 G261） | 12 × 2048 × 4096 | 当前判据 PASS；TT 14.472869 µs，TN 14.512681 µs | 无（通用版） | [R015_12x2048x4096](R015_12x2048x4096/README.md) |

## 注意

`autotune/scripts/gemm_a16w16_generic.py` 与 `operators/gemm_a16w16_generic.py` 目前内容完全相同，两份需要同步修改。`tune_lib.py` 等脚本（包括 `repro_generic`）实际加载的是 `scripts/` 内的副本；`repro.py` 会把两份的 SHA256 都写进 `summary.json`，可以据此核对。
