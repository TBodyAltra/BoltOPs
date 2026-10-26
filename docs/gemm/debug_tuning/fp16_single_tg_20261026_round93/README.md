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

本快照主选择有 125/126 个稳定 profile，14 个达到 80%，4 个达到 90%。四类选择总计 503 条记录，证据标签计数为 `{'steady_measured': 442, 'provisional_nonsteady': 15, 'independent_repeat_confirmed': 46}`。目标 80–90% 尚未在所有 block 达到。

## 本轮新增：细长 block 的已复测方案

本轮只更新 **256×32×128 / TN** 的 `best` 和 `best_no_spill`，共 2 条选择；其余 501 条与 round88 快照逐条完全一致，64 KiB 选择也保持原样。

| block | 布局 | 旧 best | 初测 | 独立复测 | 复测相对提升 |
|---|---|---:|---:|---:|---:|
| 256×32×128 | TN | 12.6725% | 17.5950% | 18.0255% | 42.2415% |

新选择为 A-first、完整 A 寄存器预取、B 按两个 K64 片段流式读取；global MLS 仍完整搬运 A/B 的 BK128，数学计算仍是每个 consumer 两次 K64 dot。真实 producer 通道由 data_barrier_map 推导并核对 consumer 根描述符。A-ready 后完成两片读取、LGKM 等待和 A-empty 释放，随后进入首次 MMAC turn 等待，再等待 B-ready；B 读取完成后才释放其 LDS。

WASP/WDRA 开启，12 waves（4+4+4），72 KiB LDS，请求寄存器 [8,248,248]、实际分区 [2,85,85]，单 buffer、stage1、earlyFalse、UF1、split2、turn2、MMAC layout3、cluster0；ILP=False、modelV2=True，async/BPP 关闭。runtime scratch 和汇编 SGPR/VGPR spill 全为 0。这个选择超过 64 KiB，因此不替换 64 KiB 预算下的方案。18%左右的利用率仍明显低于 80–90% 目标，本轮没有声称已达到目标。

CPU 筛选保持已验证的 reader、compiler、kernel 和 private622e 原生库不变，只扩展编译探测的 block 几何。8 个旧几何配置重复对照的指令及资源完全一致，原有 68 个对照由相同源码/原生库 SHA 关联。32×256×128、256×32×128 的 TT/TN 共 64 个候选全部编译通过，真实指令审计均为 72 KiB、零物理 spill、64 条静态 MMAC（每个 consumer 32 条），并检查 ready/read/LGKM/release、A/B-first 绑定、turn0/2 和全分区结束同步。CPU 编译不代表原生精度或性能。

原生先完成 8 个旧几何对照和 32 个分组短先导，再测全部 64 个候选的 16/32/64 次迭代。共 206 个原始点全部精度 PASS（8 对照 + 64×3 初测 + 2×3 预先声明的独立复测），完整输出为 8192 个元素，seed29/scale1、atol/rtol=1e-3。64 个初测中 12 个满足稳态门槛，其他 52 个不豁免。原始统计重新读取，66 组实际/逻辑 MMAC 比均为 1；没有额外运算。新晋升的初测和复测均超过旧基线 3%，六个独立模型进程、完整汇编一致、残差均≤2%。

第二个预先声明的候选未通过复测稳态门槛，不晋升，也未反复复测挑选最好一次；完整结果见 [confirmation](round93-confirmation.json)。本轮没有修复 32×256×128 / TN 的非稳态问题。

## 同期第 90 轮的负结果

取消附加 MMAC 交棒的立方体方案，以及矩形 OneA/OneB 的 turn0/1/2 方案共 104 个候选完成 320 个原始点，全部精度 PASS、零物理 spill，73 个初测稳态、104 组实际/逻辑 MMAC 比为 1。但没有候选符合既定晋升条件，故未触发独立复测，地图不变。不能将这一轮的正确性通过当作性能提升。详见 [不晋升记录](round90-map-not-promoted.json) 和 [原始统计审计](round90-native-audit.json)。

此前 B-first 的通道绑定错误、失败留存及修复回归见 [round88 说明](../fp16_single_tg_20261026_round88/README.md)。本轮保留该修复，未变更 producer、kernel、native ABI 或旧配置默认行为。私有参数需搭配各条选择记录的源文件与补丁，不能直接复制到生产 API。

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
| 256×32 | 13.66 / 14.16 | 23.53 / 25.75 | 25.86 / 27.80 | 21.10 / 17.60 |
| 256×64 | 25.62 / 27.68 | 34.43 / 40.78 | 42.41 / 46.38 | 31.24 / 34.36 |
| 256×128 | 69.46 / 57.38 | 79.99 / 87.03 | 89.86 / 88.89 | 73.34 / 73.72 |
| 256×256 | 92.76 / 91.88 | 86.02 / 81.52 | 95.60 / 95.86 | 69.14 / 72.09 |

## 版本、复现与原始产物

测量使用 Triton `46f0bed3561364b63ba4046d9c8c0cb70380ea30` 和 BoltOPs `840d816b671983527e5b1acc831c5e67618fe981` 的隔离 worktree，以及逐方案 manifest 中记录的本地补丁/编译器。部分新参数（例如 `MMA_K_SPLIT`、`LOOP_UNROLL_FACTOR`、`LOAD_B_FIRST`、`hcu_fragment_mls_layout`）依赖实验 kernel / backend。原生库与完整编译依赖保留在 node60，Git 中不放大型二进制；sources 中补丁不是脱离其基线的通用安装包。

2026-10-26 此前核对的 Triton `release/3.6.x` 为 `1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3`，BoltOPs `gemmOptimization` 最后验证的远端 head 为 `c08010ce35d8d1cf9c242c3bcf8c02fa9eac8000`。后续 Git 连接超时，round88 文档提交 `1a7fe20358f5b5a6fb8fb3c047c169dc8cfa4d00` 已在本地保存；本快照不据此声称它已推送。这不是新 head 的重测结果；测量期间没有替换已验证的编译器。具体方案以 source manifest 的 hashes 为准。

主机原始根目录：`/home/hygon/xukang/code/pmd-triton-sessions/`。JSON 内容器证据路径的 `/xukang/` 前缀对应主机 `/home/hygon/xukang/`。每条记录给出 source_ids、context、result/ASM SHA；独立复测路径另列。主地图 head 及原始 round 审计保留在 continuation / round 根目录；本文件发布选择结果及精确审计，不复制全部原始统计。已结束的部分轮次统计以各 case 的 raw-stats.tar.gz + stats-archive.json 无损保存，每个成员字节与 SHA 均验证；原始逻辑路径可由对应 stats_io.py read/restore 读取或恢复。

此前归档保持不变：`/home/hygon/xukang/code/pmd-triton-sessions/single-tg-overnight-20261022/single-tg-fp16-map-final-20261023.tar.gz`，SHA256 `1682c6080b058ba4bf4bc5f0a605c6e651fdfd62687219ca702048b64b1abaaa`。它覆盖早期轮次；后续轮次另在各自目录，不能认为全部都已进入此 tar。

## 精确审计索引

[evidence-index.json](evidence-index.json) 记录 32 个逐字节导出的审计/脚本及原始 SHA；[integration-proof.json](integration-proof.json) 记录父地图、六进程复测、2 条晋升与 501 条未变选择。`validation.json` 记录本快照的差异、源文件与审计完整性检查。
