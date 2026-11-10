# FP16 单 TG block 流水配置地图（2026-10-27，round103）

本快照新增 **32×256×128 的 TT/TN 逐 K64 MMAC 轮转方案**。两者均完成独立复测，原先 TN 主选择的非稳态问题已解决。除暂缓的 128³ 外，63 个 block、126 个布局组合的主选择现均通过稳态检查。

| 32×256×128 | 原主选择 | 本轮初测 | 独立复测 | 复测残差 |
|---|---:|---:|---:|---:|
| TT | 11.2353% | 21.7110% | 21.5810% | 0.6272% |
| TN | 17.8620%（非稳态） | 20.1039% | 20.0727% | 0.1563% |

TT 相对原稳定主选择提升 92.0819%。TN 的旧 17.8620% 来自非稳态记录；本轮新基点为经独立确认的 20.0727%，不把旧数值当成已验证的稳态性能。仅更新两个 profile 的 `best` / `best_no_spill`，共 4 条选择，其余 499 条与 round100 完全一致，包括此前 256×256×128 TN 的 77.7116% 独立复测方案。64 KiB 约束下的备选也保持原样。

## 按 block 使用

[best-configs.json](best-configs.json) 按 BM×BN×BK 和 TT/TN 给出完整 config、launch_spec、有效 waves/VGPR 分区、LDS、scratch、汇编 spill、编译器 metadata 与原始证据。BM/BN 为 32/64/128/256，BK 为 16/32/64/128；128×128×128 暂缓。

选择对应 profile，再按 128/64 KiB LDS 与 scratch 约束选 `best` / `best_no_spill` / `best_64k` / `best_64k_no_spill`。使用该行记录的 kernel/compiler；私有参数不能单独复制到生产 API。BU 阶段先重新验证精度、资源和流水成本，再调 TG 排布。K=BK×16/32/64 仅标定单 TG 流水，不按完整 GEMM 的 MNK 选 tile。

## 本轮流水和配置

A 全 BK128 捕获进寄存器，B 拆成 K0..63、K64..127 两个独立槽位；保留原 72 KiB 父 LDS 分配。数据 empty/ready 分别为 A0/1、B0 2/3、B1 4/5；独立 MMAC 轮转使用 6/7，计数为每个 consumer 组 4 waves。逐 K64 轮转在对应 B-ready 等待之前获取 token，在该片段 MMAC 发射之后交棒，phase 为 0/1，每个 BK 完成两轮。调度屏障固定发射边界；数据槽释放前完整等待 LDS 读取。退出前全部 12 waves 在 join2 会合，再回收八个信号量。

| 参数 | TT / TN |
|---|---|
| WASP / WDRA | 均开启，producer4 + consumer4 + consumer4 |
| LDS / spill | 72 KiB；runtime scratch、private bytes、SGPR/VGPR spill 均为 0 |
| stage / buffer / unroll | stage1，private single buffer，earlyFalse，UF1，split2 |
| producer / 预取 | A-first，OneA 完整捕获；B 为两个 K64 全局拷贝和 LDS 读取 |
| MMAC / 调度 | layout force3，cluster0，ILP=False，modelV2=True |
| 新轮转开关 | hcu_wasp_split_b_slot_turn=2，独立于旧 hcu_wasp_mmac_turn=0 |
| 其他 | split_b_global_probe=True，split_b_slots=True，fragment MLS=True，schedule_hint=none，mls_global_prefetch=0；async_copy/BPP=False |

TT 请求/有效寄存器分区为 `[8, 248, 248]`，实际为 `[2, 49, 49]`，n_regs=168。WDRA 应查看 actual partition regs，平均/整函数 n_regs 不能代替各分区配额。

TN 请求/有效寄存器分区为 `[8, 248, 248]`，实际为 `[2, 41, 41]`，n_regs=168。WDRA 应查看 actual partition regs，平均/整函数 n_regs 不能代替各分区配额。

这两项仍低于 80–90% 的优化目标。主选择 126/126 均通过稳态检查，不表示所有 block 已达到目标，也不表示所有历史记录都有独立复测。64 KiB 备选保留各自验证标签。四类共 503 条选择，标签计数为 `{'steady_measured': 440, 'independent_repeat_confirmed': 50, 'provisional_nonsteady': 13}`。

## 证据与负结果

CPU101 保留 native95 原生库和已有数据槽实现，仅增加 6/7 号 MMAC 轮转。4 个默认关闭轮转的对照与既有指令/资源一致，8 个候选全部编译通过且无物理 spill。逐 producer wave、逐迭代核对 16/32/64 轮的全局源地址与父 LDS 目的地址，与完整 B128 参考相同，且旧别名反例能被检出。实际 LLVM/ISA 检查消费者读取、完整 LGKM 等待、数据释放、token 等待在对应 B-ready 之前、MMAC 发射在交棒之前，以及 all12-wave join2 后的八信号量回收。

有限协议模型按角色保留 4+4 个独立 consumer waves，对两代流水穷举：整 BK 轮转 1285106 个状态，逐 K64 轮转 600707 个状态；另含 phase 回绕用例。提前释放、错误 B1 empty 计数、缺失 token seed、错误 phase、错误 turn 计数均能检出问题。这些是有限协议与实际指令证据；MMAC 模型约束发射顺序，不声称物理完成时刻或无界证明。

native102 完成 28 点（4 对照 + 8 候选×3 长度），native103 完成 10 点（4 对照 + 2 候选×3 独立长度）。共 **38 个模型、152 个 dispatch** 全部完整输出精度 PASS，重读原始统计的实际/逻辑 MMOP 比均为 1，全部无物理 spill。先跑四个瘦长块短先导，再允许其余候选和长循环；共享 PMD lease、最多 8 个自有模型，所有 cohort 自然结束后才交接。

独立复测在观察其结果之前固定为每个初测合格配方按 32/16/64 轮各测一次；每个配方初测/复测六个模型进程均独立。完整汇编在每个系列的三个长度内一致，跨 worktree 的规范化指令一致。初测与复测均要求残差≤2%、无物理 spill、完整精度、MMOP 比=1、相对旧选择收益≥3%；没有追加重复来挑选好结果。

整 BK 轮转的 32×256 TT/TN 初测残差为 3.4402% / 3.9932%，未晋升。四个 256×256×128 轮转候选虽然稳定，但性能低于已有选择，也未晋升。此前无 MMAC 轮转的独立 B 槽位方案仍保留为 256×256×128 TN 的最佳；256×256×128 与瘦长块分别保留适合各自的流水。

## 测量定义和主选择地图

输入/输出 FP16、FP32 累加，grid=1。每点 1 次 warmup + 3 次计时取中位数，完整输出 seed29/scale1、atol/rtol=1e-3。利用率为 `2*BM*BN*BK / cycles_per_bk_slope / 4096`，是有用工作量相对模型峰值的比例，非独立硬件 busy counter。残差为中点对端点拟合的偏差除以长循环耗时。

`no_spill` 沿用 runtime scratch/private bytes 为零的命名；历史记录的 SGPR→VGPR spill 仍须查看汇编字段。`assembler_zero_spills` 才要求 private bytes、SGPR 和 VGPR spill 同时为零。

下表为主选择初测 MMAC 利用率百分比，TT / TN；`—` 为暂缓。达到 80% 的有 14 个，达到 90% 的有 4 个。

| BM×BN | BK16 | BK32 | BK64 | BK128 |
|---|---:|---:|---:|---:|
| 32×32 | 6.51 / 6.80 | 11.18 / 10.81 | 9.99 / 13.74 | 13.03 / 16.97 |
| 32×64 | 12.11 / 12.66 | 20.15 / 21.26 | 16.75 / 16.67 | 20.47 / 18.51 |
| 32×128 | 8.81 / 8.99 | 26.57 / 27.94 | 33.65 / 22.27 | 22.70 / 25.49 |
| 32×256 | 14.28 / 15.96 | 22.11 / 28.56 | 33.45 / 33.43 | 21.71 / 20.10 |
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
| 256×256 | 92.76 / 91.88 | 86.02 / 81.52 | 95.60 / 95.86 | 69.14 / 77.74 |

## 版本、文件与 host 路径

测量使用 Triton `46f0bed3561364b63ba4046d9c8c0cb70380ea30`、BoltOPs `840d816b671983527e5b1acc831c5e67618fe981` 的隔离 worktree 与逐方案补丁。独立 B 子视图依赖 native95 的父布局 MLS 地址修正，原生库 SHA256 为 `3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990`。其他配方保留各自版本。此前核对的 release/3.6.x 参考 `1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3` 未替换正在测量的编译器。

- [best-configs.json](best-configs.json)：完整选择与 source_ids、context、结果/汇编 SHA、独立复测路径。
- [integration-proof.json](integration-proof.json)：4 条更新、499 条未变，保留 round100 大块方案和 64 KiB 选择。
- [evidence-index.json](evidence-index.json)：62 个逐字节导出的审计/脚本，包括历史别名反例与 native95 构建证据。
- [初测原始审计](round102-native-raw-audit.json)、[复测原始审计](round103-native-raw-audit.json)、[复测计划](round103-confirmation-plan.json)、[确认结果](round103-confirmation.json)。
- `sources/`：JSON sources 表所列的 kernel/backend/CPP/runner、补丁、manifest 与构建证据；`validation.json`：全量选择差异、有限 JSON、资源、metric、源文件和审计 SHA 检查。

原始 host 根目录为 `/home/hygon/xukang/code/pmd-triton-sessions/`；容器 `/xukang/` 对应 host `/home/hygon/xukang/`。本轮初测在 `single-tg-pipeline-20261027-round102-slot-turns-native/runs/`，独立复测在 `single-tg-pipeline-20261027-round103-slot-turns-confirm/runs/`。本包保存选择结果、源码和审计，原始 PMD 统计另留于 host；历史已无损归档的 stats 由各 case 的 raw-stats.tar.gz / stats-archive.json 与对应 stats_io.py 读取或恢复。Git 不包含大型 SO。

Git 远端此前连接超时。提交、打包与实际 push 分别记在外部 receipt，不能由本文件推断已推送成功。
