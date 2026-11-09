# BF16 linear（表 3）调优记录

对应[少伯 BU 需求](../../SHAOBO_BU_REQUIREMENTS.md)中的 linear，BF16 × BF16 → BF16；目录使用新版 R 编号，旧版表 3 的 G 编号只作历史追溯。比较口径与 manifest 要求见 [调优记录](../TUNING.md)。

覆盖核对（2026-11-05）：`autotune/shapes_priority_table3.json` 仍为旧版表 3 的 27 个 shape，并未覆盖当前全部 P0/P1 需求。按算子、输入/输出精度与 M/N/K 匹配，P0 覆盖 17/37 行（R001–R004、R014–R026），缺 R027–R046；P1 覆盖 0/84 行，缺 R047–R061、R131–R142、R199–R215、R281–R305、R312–R316、R335–R344。按 M/N/K 去重后，P0 为 16/36，P1 为 0/83；JSON 另有 11 个 shape 不属于当前这类需求。本次仅核对覆盖情况，未扩充 JSON 或改写其中的旧 G 编号。

## 环境与版本

公共部分（全部结果相同）：

| 项 | 内容 |
| --- | --- |
| 执行环境 | PMD，`GPU_CHIP=sb`（gfx946），请求频率约 2.198 GHz（455 ps/周期）；**单 die、48 CU**（核对见下），与 [调优记录](../TUNING.md) 的口径一致 |
| 编译器 | clang-18 `/opt/dtk/aillvm/bin/clang-18`，`ld.lld` `/opt/dtk/aillvm/bin/ld.lld` |
| Triton 路径 | `/public/home/duyubo/triton_all_version/triton_3.6/triton`，分支 `release/3.6.x`，构建目录 `build/lib.linux-x86_64-3.10` |
| 通用 kernel | `gemm_a16w16_generic.py`，SHA256 `e108b0bb…` |
| 计时口径 | PMD `system.gcSimCycles` 经 `gcFrequency` 换算为 µs，只含 kernel，不含 JIT 编译、分配和主机拷贝 |

die / CU 数核对（2026-11-07）：`summary.json` 不记录 die/CU 数，数据取自各运行留下的 PMD 输出。

| 证据 | 内容 | 覆盖的结果 |
| --- | --- | --- |
| PMD 启动参数（`round_*/worker.log`） | `--num-cu=48`、`--num-se=4`、`--gpu-clock=2.2GHz`、`--umc-bw=1.8TB/s`，并有 `single die dispatch` | R001 复现、R017（均为本地文件，不提交） |
| 运行记录的 `cu_count` | `cu_count=48`，`active_cus` 不超过 48 | R004、R016 |
| 逐 CU 统计 | 本地 `stats.txt` 含 `CUs00` – `CUs47` 共 48 个 CU，且只输出 `xcd0`（单 XCD） | R014、R015（`runs/TT_ir_asm_perf/` 下，本地文件） |

[调优记录](../TUNING.md) 规定每 die 48 CU，因此 48 CU 即 1 die。autotune 归档（`autotune/scripts/tune_lib.py` 中 `NUM_CU = 48`）和 R014 特化版、R004 特化版的历史记录没有留下对应日志，按同一 PMD 配置计，未逐条核实。

### 环境时间线

时间为 UTC。每次复现的 `summary.json` 会记录当次实际的提交与文件 SHA256，以 `summary.json` 为准。

| 时间 | Triton 提交 | BoltOPs 提交 | 该时间完成的记录 |
| --- | --- | --- | --- |
| 2026-11-02 – 11-05 | `cdf6827b53b1a59104da41a1bfc7736134b13a2e` | `673ede654b7cda290f68ce98665fb6f1dcb1efd8` | R001 等 autotune 归档（`autotune_20261104_dyn`，R001 完成于 11-04 09:27）；R014 特化版 6.59 µs（11-04）；R014 通用版 TT / TN（11-05 06:17 / 06:20）；R015 通用版 TT / TN（11-05 06:23） |
| 2026-11-05 12:44 起（Triton 更新一次，之后一直如此） | `e58481668e29fda141014f1d4b1f0f7bd82afd41` | 同上 | 此后的结果都在该环境下测得，不再逐条罗列 |

说明：

- Triton 的 `libtriton.so` 在更新提交后重建较晚：第一段对应的库 SHA256 为 `919b51f1…42caa7`（构建于 2026-10-31），第二段对应的库为 `b264c76926a43ad3767c23fe0140f422573ea9f55b04f1828b46389f2f8d5cf6`，构建于 2026-11-06 01:22。11-05 12:44 到 11-06 01:22 之间源码已是新提交而库还是旧的，目前没有发现落在这一段的结果。
- BoltOPs 按最初的版本记录。`673ede65` 已不在当前仓库（分支 rebase 后丢失），R014 / R015 的 summary 记录其 `tracked_changes` 为 2 / 7。后来的运行里，R016、R017 的运行目录不是 Git 仓库而没有提交号，R004、R001 通用版复现记录的是 `4a03e47`（`tracked_changes=5`）；这些只是脚本与文档的版本差异，通用 kernel 的 SHA256 没有变。
- 无环境记录的结果（autotune 归档、R014 特化版 6.59 µs）是按完成时间归入第一段的，属于推断。R016 的 summary 没有记录 Triton 提交号，归入第二段是依据其 `libtriton.so` SHA256 与 R017 一致作出的推断。
- 第一段与第二段之间没有做过同 shape、同配置的对比，不要把 R014 / R015 与后来结果的差异直接解释为 shape 或配置的影响，也不要据此判断 Triton 版本的快慢。第一段的 `libtriton.so` 已被覆盖，当前环境无法原样复现；需要统一时，建议在第二段环境上重跑 R014、R015 的通用版。

## 性能指标公共口径

各 shape README 中的 MMAC 利用率、有效带宽、带宽屋顶线都按下表计算，shape 目录只记录各自的数据量、算术强度和结果。

| 项 | 取值 / 公式 | 备注 |
| --- | --- | --- |
| 理论外存带宽 | 约 1.8 TB/s | 模型配置上限，实际生效值 **1.7996485 TB/s**（约 819 B/周期 × 2.198 GHz）；下文的达成率和屋顶线都用生效值 |
| BF16 理论计算峰值 | 432 TFLOPS（即 432 TOPS） | 单 die 48 CU 的峰值：`4096 FLOP/CU/周期 × 48 CU × 2.197802 GHz = 432.1 TFLOPS`（`tune_lib.py` 的 `PEAK_FLOPS_PER_CU_CYCLE`、`NUM_CU`）。每 CU 每周期 4096 FLOP 取自该文件注释所称的 `best-configs.json`，未独立核对硬件规格 |
| 有效运算量 | `2 × M × N × K` FLOP | 乘加计 2 次运算，不计 tile padding 与 split-K 归约 |
| MMAC 利用率 | `(2MNK / kernel耗时) / 432 TFLOPS` | 有效 GEMM 吞吐占峰值的比例，不是硬件计数器测得的 MMAC 活跃率 |
| 逻辑数据量 `D` | `2 × (M×K + K×N + M×N)` B | BF16 每元素 2 B，A、B 各读一次、C 写一次 |
| 有效带宽 | `D / 完整 kernel 耗时` | 不是实测外存流量；不计缓存复用、重复读取、事务放大和 tile padding |
| 配置带宽达成率 | `有效带宽 / 1.7996485 TB/s` | |
| 算术强度 | `2MNK / D` FLOP/B | |
| 带宽屋顶线 | `min(432 TFLOPS, 算术强度 × 1.7996485 TB/s)` | 该数据访问假设下的参考，不是所有实现的严格性能上界 |

使用时的限制：

- 耗时用各 `summary.json` 的未舍入 `median_us`，不要用 README 中四舍五入后的值反算。
- 达成率和屋顶线只在数据量足够大、受外存带宽限制时有参考价值。数据量很小的 shape（如 R014，`D` 约 0.3 MB）耗时主要来自固定开销，达成率很低不代表带宽有提升空间。
- `gcSimCycles` 含模拟器启动/收尾的常数周期（R015 的分析中约 7938 周期），小 shape 的有效带宽会因此偏低。
- 只有 R017 做过外存读带宽补测：同配置连续读 256 MiB 数据集为 1.7344 TB/s（配置上限的 96.37%），见 [R017 说明](R017_12x12288x2048/README.md)。该值是 PMD 读带宽微基准，其他 shape 尚无对应数据。
- PMD 为模拟器，以上带宽和峰值均为模拟配置值，不代表真机性能。

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
| R001 | 64 × 64 × 2048 | 当前判据 PASS；TT 7.486571 µs，TN 6.733546 µs（2026-11-07 复现；autotune 归档 TT 7.472921 µs） | 无（通用版） | [R001_64x64x2048](R001_64x64x2048/README.md) |
| R004 / R023 | 6656 × 64 × 2048 | 当前判据 PASS；TT 24.865752 µs（2026-11-07 复现，10 个样本中位数；此前转录记录 24.8396 µs） | 无（通用版） | [R004_6656x64x2048](R004_6656x64x2048/README.md) |
| R014（旧 G256） | 12 × 64 × 2048 | 当前判据 PASS；TT 7.537531 µs，TN 7.180128 µs | fused split-K，历史 6.59 µs（TT，独立判据，未重测） | [R014_12x64x2048](R014_12x64x2048/README.md) |
| R015（旧 G261） | 12 × 2048 × 4096 | 当前判据 PASS；TT 14.472869 µs，TN 14.512681 µs | 无（通用版） | [R015_12x2048x4096](R015_12x2048x4096/README.md) |
| R016 | 12 × 9216 × 2048 | 当前判据 PASS；TT 27.141890 µs（MMAC 3.8633%，有效带宽 1.4008 TB/s；实验批次转录，非 `repro_generic` 复现） | 单 wave 小 N 块（BN32 / 1 wave，运行时 M/N/stride，无编译器改动）：TT 25.736850 µs（MMAC 4.0742%，有效带宽 1.4772 TB/s），较同轮重测的基线（27.237667 µs）−5.51%；未接入生产接口。M/N/stride 静态化与 `slc` 缓存策略位的补丁方案仅在 R016 README 中说明 | [R016_12x9216x2048](R016_12x9216x2048/README.md) |
| R018 | 12 × 248320 × 2048 | 当前判据 PASS；TT 584.986273 µs（MMAC 4.8297%，有效带宽 1.7490 TB/s；实验批次转录，1 预热 + 1 计时，非 `repro_generic` 复现） | 混合 N 分块（主体 BN=256 + 尾部 BN=64，自定义 grid，无编译器改动）：TT 579.665502 µs（MMAC 4.8741%，有效带宽 1.7650 TB/s），较通用版 −0.91%，单样本、未经多样本确认；未接入生产接口 | [R018_12x248320x2048](R018_12x248320x2048/README.md) |

## 注意

`autotune/scripts/gemm_a16w16_generic.py` 与 `operators/gemm_a16w16_generic.py` 目前内容完全相同，两份需要同步修改。`tune_lib.py` 等脚本（包括 `repro_generic`）实际加载的是 `scripts/` 内的副本；`repro.py` 会把两份的 SHA256 都写进 `summary.json`，可以据此核对。
