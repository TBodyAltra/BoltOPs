# FP16 单 TG block 流水配置地图（2026-10-27，round113）

本快照新增三个独立复测确认的流水配置：128×256×128 TT 达到 81.0801%，256×256×128 TT 为 72.5234%，32×256×128 TN 为 24.8185%。分别相对旧主选择提升 18.5718%、4.8898%、23.4511%。只更新三个 profile 的 best / best_no_spill，共六条选择；其余 497 条与 round108 逐条完全一致。

[best-configs.json](best-configs.json) 按 BM×BN×BK 与 TT/TN 提供完整 config、launch_spec、实际寄存器分区、LDS、汇编 spill 及原始证据。63 个 block、126 个主选择均通过稳态检查；128×128×128 继续暂缓。503 条历史约束选择保留各自验证标签，不全是稳态或独立复测确认。

## 五个预先提名配置的独立复测

每个 block/layout 仅提名初测稳态且提升至少 3% 的最高性能配置；然后按预先固定的 32/16/64 轮各测一次。下表均为 MMAC 利用率，旧选择为其初测基点，独立复测未通过者不晋升。

| BM×BN×BK / 布局 | wait_mask | 原主选择 | 初测 | 独立复测 | 复测残差 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 32×256×128 TN | 1 | 20.1039% | 24.7502% | 24.8185% | 0.1755% | 晋升 |
| 32×256×128 TT | 2 | 21.7110% | 23.2004% | 22.8637% | 2.3617% | 残差超限，未晋升 |
| 128×256×128 TN | 3 | 70.9105% | 80.7014% | 80.5203% | 3.0668% | 残差超限，未晋升 |
| 128×256×128 TT | 3 | 68.3806% | 80.9113% | 81.0801% | 1.0713% | 晋升 |
| 256×256×128 TT | 2 | 69.1425% | 72.4731% | 72.5234% | 0.1576% | 晋升 |

128×256×128 TN 虽有约 80.52% 的复测斜率利用率，但中点残差 3.0668% 超过 2%，保留此前独立复测 71.1641% 的主选择。32×256×128 TT 的复测残差 2.3617% 也超限，保留此前独立复测 21.5810% 的配置。没有增加重复来挑选好结果。256×256×128 TN 保留此前 77.7116% 独立确认的无 MMAC token 方案；所有 64 KiB 约束备选不变。

## 本轮流水

WASP/WDRA，producer4 + consumer4 + consumer4，stage1、private single buffer、earlyFalse、UF1、split2。A-first / OneA 完整捕获 A；B 为两个独立 K64 数据槽，保留原父 LDS 分配。数据 empty/ready 为 A0/1、B0 2/3、B1 4/5；MMAC 轮转使用 6/7，每个 consumer 角色四个 waves，每 BK 两个 K64 轮次，phase0/1。退出前全部 12 waves 在 join2 会合，再回收八个信号量。

旧 mmac_turn=0，split_b_slot_turn=2。新 hcu_wasp_split_b_slot_turn_wait_mask 的 bit0/1 分别控制第一/第二 K64 片段：置位时，把对应 token wait 从 B-ready 之前移到 B 读取、完整 LGKM 等待与 empty release 之后，紧挨该片段首条 MMAC 之前；未置位保留原位置。MMAC 发射后才交棒。全局拷贝、LDS 地址、数据所有权与数学工作量保持原样，访存可以提前进行。新参数默认0；私有 API 不能直接用于生产代码。

| 晋升配置 | wait_mask | LDS | 请求寄存器分区 | 实际寄存器分区 | MMAC force | ILP / modelV2 |
|---|---:|---:|---|---|---:|---|
| 32×256×128 TN | 1 | 72 KiB | [8, 248, 248] | [2, 65, 65] | 3 | False / True |
| 128×256×128 TT | 3 | 96 KiB | [8, 248, 248] | [2, 161, 161] | -1 | True / True |
| 256×256×128 TT | 2 | 128 KiB | [8, 248, 248] | [2, 225, 225] | -1 | True / False |

三项均为 cluster0、fragment MLS=True、split_b_global_probe=True、split_b_slots=True、mls_global_prefetch=0、schedule_hint=none，async_copy/BPP=False。runtime scratch、private bytes、SGPR/VGPR spill 均为零。n_regs=168 是整函数字段；WDRA 配额应查看实际分区，不能用 n_regs 替代。全部参数以对应 JSON 行为准。

## 证据与排除项

CPU109 六个默认关闭新等待移动的对照，ISA/资源与既有 turn2 配方完全一致；首个 256×256×128 TT mask1 被编译器以 per-SIMD VGPR budget 超限拒绝，未进入 PMD。CPU110 使用同一冻结隔离 worktree 的源代码继续完整网格：六个对照再次一致，18 个候选中 14 个编译通过且无物理 spill；256×256×128 TT/TN 的 mask1/3 共四项因 VGPR 预算超限被排除。这些是编译拒绝，不是实际 spill 测量。

14 项均核对全部四个 producer waves、16/32/64 轮的全部 112 个迭代：实际全局源地址、父 LDS 目的地址及标志与参考相同，B 两半不别名、不重复搬运。实际 LLVM/ISA 检查八个 consumer wave ID 的读取地址/标志、完整 LGKM 等待后释放、条件 token 等待位置、MMAC 发射后交棒、初始化计数、八信号量回收与 all12-wave join2；没有 consumer ebarrier1。

有限协议模型保留每个角色四个独立 consumer waves，两代流水全可达状态穷举。mask1/2/3 分别检查 4760123 / 1806536 / 8433236 个状态，另含每角色一个 wave、四代 phase 回绕用例。提前释放、错误 B1 empty 计数、缺失 seed、错误 phase、错误 turn 计数共 15 个负例均被检出。CPU109 的旧五百万状态上限不足，保留未完成日志；CPU110 用无损整数编码及两千万上限完成全部检查，没有减少 waves 或代数，12 个小型正负例与原编码结果一致。模型约束 MMAC 发射，不声称无界证明或物理完成时刻。

native111 初测为六对照 + 14 候选×三长度，共 48 个模型；native112 独立复测为六对照 + 五提名×三长度，共 21 个模型。合计 **69 个模型、276 个 dispatch**，全部完整输出精度 PASS、原始统计实际/逻辑 MMOP 比=1、仅 CU0、物理 spill 零、源码哈希不变。每个测量系列三个长度完整汇编一致；跨 worktree 规范化指令一致。每个提名的初测/复测六个模型进程独立，五个预定复测系列中三项通过稳态与收益门槛。

初测瘦长 TT mask1/3、TN mask2/3 共四项残差超标，包含接近 29% 的较快结果，均未提名。其他低于 3% 收益的稳态项也不晋升。共享 PMD lease、最多八个自有模型；独立复测最多四个，前一轮全部模型自然结束后才交接。

## 测量与使用

单 TG、grid=1，输入/输出 FP16、FP32 累加。K=BK×16/32/64 仅用于流水标定，不按完整 GEMM MNK 选 tile。每点 1 次 warmup + 3 次计时取中位数，完整输出 seed29/scale1，对 CPU FP32 reference 使用 atol=rtol=1e-3。斜率为 (cycles64-cycles16)/48；利用率为 2*BM*BN*BK / slope / 4096，是有用工作量相对模型峰值的比例，不是独立硬件 busy counter。

按 profile 和 LDS/scratch 约束选 best / best_no_spill / best_64k / best_64k_no_spill，并使用该行 kernel/compiler 版本。BU 上重新验证精度、资源及流水，再调 TG 排布。no_spill 沿用 runtime scratch/private bytes 为零的历史定义；assembler_zero_spills 才同时要求 SGPR/VGPR spill 为零。

主选择 126/126 稳态，15 个初测利用率达到 80%、四个达到 90%，仍需继续优化。全量选择验证标签计数：`{'steady_measured': 440, 'independent_repeat_confirmed': 50, 'provisional_nonsteady': 13}`。

下表为主选择初测 MMAC 利用率百分比，TT / TN；独立复测值保存在 JSON 中，— 为暂缓。

| BM×BN | BK16 | BK32 | BK64 | BK128 |
|---|---:|---:|---:|---:|
| 32×32 | 6.51 / 6.80 | 11.18 / 10.81 | 9.99 / 13.74 | 13.03 / 16.97 |
| 32×64 | 12.11 / 12.66 | 20.15 / 21.26 | 16.75 / 16.67 | 20.47 / 18.51 |
| 32×128 | 8.81 / 8.99 | 26.57 / 27.94 | 33.65 / 22.27 | 22.70 / 25.49 |
| 32×256 | 14.28 / 15.96 | 22.11 / 28.56 | 33.45 / 33.43 | 21.71 / 24.75 |
| 64×32 | 12.76 / 12.63 | 22.39 / 20.85 | 16.00 / 17.82 | 19.81 / 18.59 |
| 64×64 | 22.82 / 19.14 | 35.57 / 37.52 | 33.56 / 33.40 | 28.32 / 32.16 |
| 64×128 | 16.41 / 16.89 | 47.23 / 47.52 | 47.04 / 48.29 | 36.14 / 42.64 |
| 64×256 | 27.75 / 32.20 | 30.50 / 41.57 | 38.46 / 46.94 | 33.22 / 32.66 |
| 128×32 | 8.01 / 7.88 | 15.16 / 14.49 | 24.08 / 22.82 | 21.79 / 22.94 |
| 128×64 | 15.48 / 15.49 | 47.48 / 47.11 | 48.49 / 48.59 | 35.06 / 36.59 |
| 128×128 | 49.78 / 45.44 | 66.66 / 65.17 | 66.81 / 60.41 | — / — |
| 128×256 | 80.36 / 57.45 | 85.53 / 89.04 | 84.37 / 85.63 | 80.91 / 70.91 |
| 256×32 | 13.66 / 14.16 | 23.53 / 25.75 | 25.86 / 27.80 | 21.10 / 17.60 |
| 256×64 | 25.62 / 27.68 | 34.43 / 40.78 | 42.41 / 46.38 | 31.24 / 34.36 |
| 256×128 | 69.46 / 57.38 | 79.99 / 87.03 | 89.86 / 88.89 | 73.34 / 73.72 |
| 256×256 | 92.76 / 91.88 | 86.02 / 81.52 | 95.60 / 95.86 | 72.47 / 77.74 |

## 版本与文件

测量使用 Triton 46f0bed3561364b63ba4046d9c8c0cb70380ea30、BoltOPs 840d816b671983527e5b1acc831c5e67618fe981 的隔离 worktree 及逐方案补丁。新配方使用 native95 父布局 MLS 修正，原生库 SHA256 为 3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990。CPU110/native111/112 只读使用已冻结的 CPU109 隔离源 worktree，输出、控制器和编译缓存分开，源文件逐字节校验。此前 release/3.6.x 参考 1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3 未替换测量编译器，其他配方保留各自版本。

- [best-configs.json](best-configs.json)：完整选择、资源、source_ids、结果/汇编 SHA 和独立复测路径。
- [integration-proof.json](integration-proof.json)：六条更新、497 条不变。
- [evidence-index.json](evidence-index.json)：121 个逐字节审计、脚本及历史负例。
- [初测原始审计](round111-native-raw-audit.json)、[复测原始审计](round112-native-raw-audit.json)、[复测计划](round112-confirmation-plan.json)、[确认结果](round112-confirmation.json)、[CPU 同步审计](round110-ownership-isa-audit.json)。
- sources/ 保存 kernel/backend/CPP/runner、补丁及原生构建证据；validation.json 保存全量选择差异、资源、metric、SHA 与链接检查。

host 原始根目录为 /home/hygon/xukang/code/pmd-triton-sessions/。初测位于 single-tg-pipeline-20261027-round111-late-slot-turn-native/runs/，独立复测位于 single-tg-pipeline-20261027-round112-late-slot-turn-confirm/runs/。容器 /xukang/ 对应 host /home/hygon/xukang/。本包不含大型 SO 与全部原始 PMD 统计；历史归档统计通过 stats_io.py 读取/恢复。round21/25 共 455 个已结束、已审计 PASS case 的 1820 个统计成员已逐字节无损归档，回收 5.1723 GiB，结果与汇编 SHA 保持不变，证明见 evidence-index。

提交、打包、host 镜像及实际 push 分别记录在外部 receipt，不能由本文推断已经推送成功。
