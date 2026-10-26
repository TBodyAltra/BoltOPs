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

[上一轮 round76 快照](../fp16_single_tg_20261026_round76/README.md) 保留原样。本轮逐文件审计索引见 [evidence-index.json](evidence-index.json)，晋升核对见 [integration-proof.json](integration-proof.json)。

## 本轮新增：矩形 BK128 的完整寄存器预取

本轮仅更新 128×256×128 / TT 的 `best` 与 `best_no_spill`；其余 501 条选择与 round76 快照完全一致，包括上一轮 256×256×128 TT/TN 的零 spill 基点。旧 TT 基线 46.2960%，新方案初测 49.8287%、独立复测 49.6854%，复测相对旧基线提升 7.3211%。初测略快的另一个预选候选复测未通过稳态门槛，未晋升，也未反复重跑。

完整搬运 BK128，两个 consumer 各按 K64 直接读取两个片段；全部操作数进入寄存器并完成 LGKM 等待后，提前释放 LDS，再执行两次 dot。选中配置合并 A/B 数据同步：empty0/count8、ready1/count4，producer 在两侧完整传输都发出后通知 ready。使用普通 abarrier4/5 在两个 consumer 间按整 BK 交棒，count4，producer 四个 wave 各 arrive 一次初始化 ping；`hcu_wasp_mmac_turn=1` 的 phase 为 iteration&1。token 销毁前保留编译器已有的全分区结束同步。

LDS 96 KiB，12 waves（4+4+4），WDRA 请求寄存器 [8,248,248]，单 buffer、B-first、MMAC layout3、ILP 与调度 modelV2 均开启，async/BPP 关闭；runtime scratch 与物理 SGPR/VGPR spill 均为 0。`hcu_wasp_prefetch_full_k=True`、`hcu_wasp_merge_ab=True` 及上述 turn 选项均为私有实验选项，默认关闭；必须配套 sources 中的 prefetch/merge/turn/partial helper、backend 补丁和 round68 ABI 匹配的私有原生库。

70 个旧编译对照与 16 个矩形几何对照保持指令/资源一致。96 个新候选中 80 个实际生成预期的完整预取流水，另外 16 个后端回退方案被排除。同步模型覆盖 24 个正例与 3 个反例，仍需原生验证。PMD 最终 254 个点全部精度 PASS（8 对照 + 80×3 初测 + 2×3 预先声明的独立复测）；80 个初测方案中 43 个满足稳态门槛，其余保留原始证据且不因精度通过就晋升。独立重新读取每个 dispatch 的原始 MMOP 计数，实际/逻辑 MMAC 比均为 1，无额外计算。最终仅晋升同时满足完整汇编一致、中点残差≤2%、初测与复测均提升≥3%的方案。

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
| 128×256 | 80.36 / 57.45 | 85.53 / 89.04 | 84.37 / 85.63 | 49.83 / 57.75 |
| 256×32 | 13.66 / 14.16 | 23.53 / 25.75 | 25.86 / 27.80 | 21.10 / 12.67 |
| 256×64 | 25.62 / 27.68 | 34.43 / 40.78 | 42.41 / 46.38 | 31.24 / 34.36 |
| 256×128 | 69.46 / 57.38 | 79.99 / 87.03 | 89.86 / 88.89 | 56.21 / 56.82 |
| 256×256 | 92.76 / 91.88 | 86.02 / 81.52 | 95.60 / 95.86 | 65.38 / 65.07 |

## 版本、复现与原始产物

测量使用 Triton `46f0bed3561364b63ba4046d9c8c0cb70380ea30` 和 BoltOPs `840d816b671983527e5b1acc831c5e67618fe981` 的隔离 worktree，以及逐方案 manifest 中记录的本地补丁/编译器。部分新参数（例如 `MMA_K_SPLIT`、`LOOP_UNROLL_FACTOR`、`LOAD_B_FIRST`、`hcu_fragment_mls_layout`）依赖实验 kernel / backend。原生库与完整编译依赖保留在 node60，Git 中不放大型二进制；sources 中补丁不是脱离其基线的通用安装包。

2026-10-26 重新查询的 Triton `release/3.6.x` 为 `1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3`，BoltOPs `gemmOptimization` 为 `7f620de6eec471d0413dd22cec031a10dd60b55e`。这不是新 head 的重测结果；测量期间没有替换已验证的编译器。具体方案以 source manifest 的 hashes 为准。

主机原始根目录：`/home/hygon/xukang/code/pmd-triton-sessions/`。JSON 内容器证据路径的 `/xukang/` 前缀对应主机 `/home/hygon/xukang/`。每条记录给出 source_ids、context、result/ASM SHA；独立复测路径另列。主地图 head 及原始 round 审计保留在 continuation / round 根目录；本文件发布选择结果及精确审计，不复制全部原始统计。已结束的部分轮次统计以各 case 的 raw-stats.tar.gz + stats-archive.json 无损保存，每个成员字节与 SHA 均验证；原始逻辑路径可由对应 stats_io.py read/restore 读取或恢复。

此前归档保持不变：`/home/hygon/xukang/code/pmd-triton-sessions/single-tg-overnight-20261022/single-tg-fp16-map-final-20261023.tar.gz`，SHA256 `1682c6080b058ba4bf4bc5f0a605c6e651fdfd62687219ca702048b64b1abaaa`。它覆盖早期轮次；后续轮次另在各自目录，不能认为全部都已进入此 tar。
