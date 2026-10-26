# FP16 单 TG block 流水配置地图（2026-10-26）

此地图按 **BM×BN×BK 与 TT/TN 输入布局**选流水，供 BU 阶段调优 TG 排布时作为起点。没有按完整 GEMM 的 M/N/K 选 tile，也没有将 PMD 结果视为实卡验收。

机器为 node60，容器 `kxu_pmd_8569_fp16_map_20261020`，PMD gfx946。输入/输出 FP16、FP32 累加，grid=1；BM/BN 扫 32、64、128、256，BK 扫 16、32、64、128。128×128×128 按要求保留为待优化，因此为 63 个 block、126 个布局组合。

## 文件与使用

- [best-configs.json](best-configs.json)：每个 profile 的完整 config、launch_spec、有效 waves/VGPR 分区、LDS、runtime spill、汇编 spill、流水成本、精度和证据路径。
- `sources/`：每个已选方案使用的 kernel 快照、已有源补丁和源 manifest；JSON 的 sources 表将原始路径、内容 SHA 与这些文件关联。
- JSON 的 `profiles[].choices` 分四类：`best` / `best_no_spill` 为 128 KiB LDS，`best_64k` / `best_64k_no_spill` 为 64 KiB LDS。后两类方便受限环境选择。缺失的 choice 表示没有相应已选记录，不能外推。

按 tile 和 layout 查找 profile，再选择 LDS / scratch 约束；使用 config **及其记录的 kernel/compiler**。`launch_spec` 保留 MLS、BPP、async、unroll 等入口差异，不能只复制字典到当前生产 API。BU 上先重新验证精度、资源与流水成本，再选 TG 排布。

`no_spill` 沿用原地图命名，含义是 runtime `n_spills=0` 且汇编 private bytes=0，即**无 scratch**。它不保证 SGPR spill 指令计数为零：SGPR 仍可能 spill 到 VGPR。`evidence.assembler_fields` 同时保留 private bytes、SGPR/VGPR spill 与 next_free_vgpr；`assembler_zero_spills` 才要求三者均为零。WDRA 必须看 requested/effective/actual partition regs 与 partition warps，不能把平均 n_regs 当作所有分区的寄存器配额。

## 测量和验证边界

每个配置以 BK 的 16/32/64 次迭代标定流水斜率，K 只是迭代校准长度。每点 1 次 warmup + 3 次计时取中位数，检查完整输出精度（seed29，scale1，atol/rtol=1e-3）、三个长度的完整汇编 SHA 一致和中点残差≤2%。有效 MMAC 利用率 = `2*BM*BN*BK / slope / 4096`；这是有用运算量相对模型峰值的比例，**不是硬件 busy counter**。JSON 还保留实际 MMOP 与逻辑运算比，避免将额外执行的计算误算为有用性能。

`validation_status` 分为 `independent_repeat_confirmed`（有独立复测且稳定）、`steady_measured`（稳定测量，未声称独立复测）和 `provisional_nonsteady`（不作已验证推荐）。续跑产生的新晋升要求初测和预先声明的独立复测均超过旧基线 3%；失败复测不反复重跑挑最好一次。

本快照主选择有 125/126 个稳定 profile，14 个达到 80%，4 个达到 90%。四类选择总计 503 条记录，证据标签计数为 `{'steady_measured': 444, 'provisional_nonsteady': 15, 'independent_repeat_confirmed': 44}`。目标 80–90% 尚未在所有 block 达到。

## 本轮新增：单侧完整预取、另一侧流式读取

本轮只更新 256×256×128 的 TT/TN 两个 profile，各自 `best` 与 `best_no_spill` 共 4 条选择；其余 499 条选择与 round84 快照完全一致，包含此前晋升的四个矩形 profile。仍按初测最快且独立复测通过的原规则选择，不按复测最好一次挑选。

| block | 布局 | 旧 best | 初测 | 独立复测 | 相对旧 best 的复测提升 |
|---|---|---:|---:|---:|---:|
| 256×256×128 | TT | 65.3793% | 69.1425% | 68.5810% | 4.8971% |
| 256×256×128 | TN | 65.0725% | 72.0942% | 73.2194% | 12.5198% |

新的两个主选择与零 spill 选择相同，runtime scratch、物理 SGPR/VGPR spill 全为 0。旧 `best` 的 runtime `n_spills=73` 已被替换；73 是 spill 计数，不是利用率。原来的零 scratch 选择也分别获得约 5.47%（TT）与 13.79%（TN）的独立复测提升。

完整 global MLS 仍搬运 BK128。A-first producer 首先搬运 A；consumer 等待实际 A-ready，将 A 的两个 K64 片段完整读入寄存器，完成 LGKM 等待后释放 A，再进入 MMAC 交棒与 B-ready 等待。B 按 K64 在计算窗口内流式读取，自己的读取完成后才释放其 LDS；没有要求 B 的全部片段在首次 MMAC 前捕获。每个 consumer 原有两次 K64 dot、global 拷贝数量和 LDS 分配不变。

两者均为 WASP/WDRA 开启、12 waves（4+4+4）、128 KiB LDS、请求寄存器 [8,248,248]、实际分区寄存器 [2,225,225]、单 buffer、stage1、earlyFalse、UF1、K64 split2、A-first、turn2（每 K64 交棒），ILP 开启，cluster0，async/BPP 关闭。TT 使用 MMAC auto / modelV2=False；TN 使用 layout3 / modelV2=True。完整参数及资源见 JSON。

`hcu_wasp_prefetch_one_operand=1` 是私有实验选项，需包含的 data_barrier_map、prefetch_one_operand、partial/turn helper、backend 补丁和 round68 ABI 匹配的私有原生库。它不等价于两侧完整预取；本轮新选择没有开启 full-prefetch 或 first-operand-early。turn2 的 phase 对应 K64 片段 0/1，token 4/5 销毁前保留原有全分区结束同步。

## B-first 同步修复与回归

round86 的 B-first 单侧 A 预取先导有 4 个完整输出精度失败，原始失败输入、golden、输出和日志均保留；这些候选没有晋升。旧 helper 按数学操作数编号假定 A 使用 empty0/ready1、B 使用 empty2/ready3，但 B-first producer 的真实分配恰好相反，因而可能等待错误 ready、读取未就绪的 A 并释放错误 LDS。

round87 改为从真实 producer MatrixLoad、empty wait、ready arrival 和 consumer memdesc 根推导通道，并交叉验证 A/B 的绑定。first-operand-early、one-operand-prefetch 和流式 MMAC 首次等待共用这一映射，不新增用户开关。round84 晋升的四个矩形选择均为 A-first，不受该绑定问题影响；旧页面中对 B-first 的有限通过记录不能外推为所有提前读取路径都已正确。新的修复及失败审计在本快照补充，旧源码与证据 SHA 保持原样。

[原始失败审计](round86-failure-audit.json)、[派生绑定静态审计](round87-derived-bindings-static-audit.json)、[四个原失败配置修复回归](round88-regression-fix-short-proof.json)保留具体条件和 SHA。同步模型覆盖 32 个正例、15016 个状态及 22 个反例，包括与上述失败对应的错误通道/数据代际反例；全部通过。

36 个旧配置的实际指令与资源对照保持一致。56 个新配置中 40 个编译通过，另 16 个完整 B 预取的立方体配置超寄存器配额，未进入原生测量。40 个编译通过配置均经真实 ready/read/LGKM/release、所有权、MMAC 数量和结束同步审计。源级请求的 full-prefetch 在后端可能将另一侧读取交织到计算中，因此实际位置以静态审计 placements 为准，不声称所有操作数都在首次 MMAC 前完整读入。

20 个原生短先导均通过，其中四个历史失败配置以相同 K16、seed29、scale1 回归，mismatch/nonfinite 均为 0。长测共 140 个原始点全部精度 PASS：8 对照 + 40×3 初测 + 4×3 独立复测；40 个初测方案中 32 个满足稳态门槛，其余不豁免。原始统计独立重读得到 44 组实际/逻辑 MMAC 比均为 1，无额外计算。新晋升均满足三长度完整汇编一致、残差≤2%、初测和独立复测均提升≥3%，每个选择的初测/复测共六个独立进程。

修复后的矩形 B-first 方案本轮没有超过 round84 已发布的 A-first 最佳方案，故保留原选择。

32×256×128 / TN 的长窗口仍非稳态，保留候选但标 provisional，不晋升为已验证最优；128³ 仍待优化。64/128 KiB、零 scratch 与允许 spill 的配置并列保存，不以最快有 spill 的方案覆盖零 scratch 基点。

## 主选择地图

数值为有用 MMAC 利用率百分比，列为 BK。每格为 TT / TN；`*` 表示非稳态，`—` 表示按要求待优化。

| BM×BN | BK16 | BK32 | BK64 | BK128 |
|---|---:|---:|---:|---:|
| 32×32 | 6.51 / 6.80 | 11.18 / 10.81 | 9.99 / 13.74 | 13.03 / 16.97 |
| 32×64 | 12.11 / 12.66 | 20.15 / 21.26 | 16.75 / 16.67 | 20.47 / 18.51 |
| 32×128 | 8.81 / 8.99 | 26.57 / 27.94 | 33.65 / 22.27 | 22.70 / 25.49 |
| 32×256 | 14.28 / 15.96 | 22.11 / 28.56 | 33.45 / 33.43 | 11.24 / 17.86* |
| 64×32 | 12.76 / 12.63 | 22.39 / 20.85 | 16.00 / 17.82 | 19.81 / 18.59 |
| 64×64 | 22.82 / 19.14 | 35.57 / 37.52 | 33.56 / 33.40 | 28.32 / 32.16 |
| 64×128 | 16.41 / 16.89 | 47.23 / 47.52 | 47.04 / 48.29 | 36.14 / 42.64 |
| 64×256 | 27.75 / 32.20 | 30.50 / 41.57 | 38.46 / 46.94 | 33.22 / 32.66 |
| 128×32 | 8.01 / 7.88 | 15.16 / 14.49 | 24.08 / 22.82 | 21.79 / 22.94 |
| 128×64 | 15.48 / 15.49 | 47.48 / 47.11 | 48.49 / 48.59 | 35.06 / 36.59 |
| 128×128 | 49.78 / 45.44 | 66.66 / 65.17 | 66.81 / 60.41 | — / — |
| 128×256 | 80.36 / 57.45 | 85.53 / 89.04 | 84.37 / 85.63 | 68.38 / 68.09 |
| 256×32 | 13.66 / 14.16 | 23.53 / 25.75 | 25.86 / 27.80 | 21.10 / 12.67 |
| 256×64 | 25.62 / 27.68 | 34.43 / 40.78 | 42.41 / 46.38 | 31.24 / 34.36 |
| 256×128 | 69.46 / 57.38 | 79.99 / 87.03 | 89.86 / 88.89 | 73.34 / 73.72 |
| 256×256 | 92.76 / 91.88 | 86.02 / 81.52 | 95.60 / 95.86 | 69.14 / 72.09 |

## 版本、复现与原始产物

测量使用 Triton `46f0bed3561364b63ba4046d9c8c0cb70380ea30` 和 BoltOPs `840d816b671983527e5b1acc831c5e67618fe981` 的隔离 worktree，以及逐方案 manifest 中记录的本地补丁/编译器。部分新参数（例如 `MMA_K_SPLIT`、`LOOP_UNROLL_FACTOR`、`LOAD_B_FIRST`、`hcu_fragment_mls_layout`）依赖实验 kernel / backend。原生库与完整编译依赖保留在 node60，Git 中不放大型二进制；sources 中补丁不是脱离其基线的通用安装包。

2026-10-26 重新查询的 Triton `release/3.6.x` 为 `1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3`，BoltOPs `gemmOptimization` 为 `c08010ce35d8d1cf9c242c3bcf8c02fa9eac8000`。这不是新 head 的重测结果；测量期间没有替换已验证的编译器。具体方案以 source manifest 的 hashes 为准。

主机原始根目录：`/home/hygon/xukang/code/pmd-triton-sessions/`。JSON 内容器证据路径的 `/xukang/` 前缀对应主机 `/home/hygon/xukang/`。每条记录给出 source_ids、context、result/ASM SHA；独立复测路径另列。主地图 head 及原始 round 审计保留在 continuation / round 根目录；本文件发布选择结果及精确审计，不复制全部原始统计。已结束的部分轮次统计以各 case 的 raw-stats.tar.gz + stats-archive.json 无损保存，每个成员字节与 SHA 均验证；原始逻辑路径可由对应 stats_io.py read/restore 读取或恢复。

此前归档保持不变：`/home/hygon/xukang/code/pmd-triton-sessions/single-tg-overnight-20261022/single-tg-fp16-map-final-20261023.tar.gz`，SHA256 `1682c6080b058ba4bf4bc5f0a605c6e651fdfd62687219ca702048b64b1abaaa`。它覆盖早期轮次；后续轮次另在各自目录，不能认为全部都已进入此 tar。

## 精确审计索引

[evidence-index.json](evidence-index.json) 记录 24 个逐字节导出的审计/脚本及原始 SHA；[integration-proof.json](integration-proof.json) 记录父地图、晋升选择、六进程复测和 499 条未变选择。`validation.json` 记录本发布快照的差异及完整性检查。
