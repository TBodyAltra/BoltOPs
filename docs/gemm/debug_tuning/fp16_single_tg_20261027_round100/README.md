# FP16 单 TG block 流水配置地图（2026-10-27，round100）

本快照将 **256×256×128 / TN** 的独立 B0/B1 流水纳入最佳配置：MMAC 利用率初测 **77.7389%**、独立复测 **77.7116%**，相对原最佳 **72.0942%** 提升 **7.7918%**。仅更新 `best`、`best_no_spill` 两条选择，其余 501 条与 round93 逐条完全一致。

按 BM×BN×BK 与 TT/TN 布局查找 [best-configs.json](best-configs.json)。范围为 BM/BN=32/64/128/256，BK=16/32/64/128，63 个 block、126 个布局组合；128³ 按要求暂缓。K=BK×16/32/64 仅标定单 TG 流水，grid=1，不按完整 GEMM 的 MNK 选 tile。

## 新配置与流水

| 参数 | 配置 |
|---|---|
| block / 布局 | 256×256×128 / TN |
| WASP / WDRA | 开 / 开；12 waves（producer4 + consumer4 + consumer4） |
| 寄存器分区 | requested/effective [8,248,248]；actual [2,225,225]；n_regs=168 |
| LDS / spill | 128 KiB；runtime scratch、汇编 private bytes、SGPR/VGPR spill 均为 0 |
| stage / buffer / unroll | stage1，private single buffer，earlyFalse，UF1 |
| MLS / math | A-first；完整 A 寄存器捕获；B 为两个 K64 子视图；每 consumer 两次 K64 dot |
| MMAC / 调度 | layout force=-1（auto），cluster0，额外 MMAC turn=0，ILP=True，modelV2=True |
| 其他开关 | split2，fragment MLS layout=True，schedule_hint=none，mls_global_prefetch=0，async_copy/BPP=False |
| 新开关 | hcu_wasp_split_b_global_probe=True，hcu_wasp_split_b_slots=True |

A 使用 empty0/ready1，B0 使用 empty2/ready3，B1 使用 empty4/ready5。两个 B 子块分别等待上一代读取释放，再搬运、发布；消费者完成对应 LDS 读取并等待 LGKM 后释放各自槽位。保持原 BK128 父分配、相同全局字节和实际 MLS 指令工作量，数学累加顺序不变。实际指令允许 MMAC 与 LDS 读取交错；审计检查最后一次 LDS 操作已被完整等待覆盖后才释放。结束时全部 12 waves 在 join2 会合，再回收信号量。

本选择用满 128 KiB LDS，不替换 64 KiB 约束下的备选。使用 JSON 中的完整 config、launch_spec、compiler_metadata 及对应源码；这些私有参数依赖实验 backend/kernel，需在 BU 再验证精度、资源与流水成本。

## 编译器修正与验证

新的 MatrixLoad 子视图必须使用 native95 的父布局地址修正，原生库 SHA256 为 `3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990`。其他 501 条选择继续使用各自记录的版本。

原 MLS lowering 忽略目的子视图的父几何和偏移，两个 K64 拷贝会写入同一 LDS 区域。CPU94 检出后未运行该错误用例。修正按父布局计算坐标，并将四个 producer waves 分配到不重复的 N 区域。TT 的两个 K64 起始偏移差为 32768 字节，TN 为 8192 字节，不能统一加固定偏移。native95 只重编译一个 MLS 对象，再与 ABI 匹配的已有对象链接；其余 503 个链接输入、E4/typed-SCF/parser 修正保持一致。完整 CPP、源码补丁、编译依赖和构建计划均由 `sources` 表定位。

CPU95 的 11 个旧配置指令及资源一致；后续重复对照也一致。CPU96/98 对四个 block/layout 用例，在所有 4 个 producer waves、16/32/64 轮的每一次迭代中比较实际 LLVM 全局源地址与 LDS 目的地址，和完整 B128 参考完全等价，无重复搬运，两个子块的 MLS 目的起始地址集合不相交，且旧别名反例能被检出。CPU98 同时检查消费者真实读取地址、释放前等待和退出同步。有限协议模型穷举了 8 个 consumer waves、两代流水的 1644974 个状态；注入提前释放、错误 empty 计数、错误 phase、缺失初始 empty 后均能检出问题。这是有限协议与实际指令证据，原生精度另行验证。

native97 的 8 个完整输出检查与 32 个 dispatch 原始统计全部通过。native99 共 16 点（4 对照 + 4 候选×3 长度）全部通过，native100 共 7 点（4 对照 + 唯一合格候选×3 独立长度）全部通过。后两轮合计 **23 个模型、92 个 dispatch** 重读统计，MMOP/逻辑工作量比均为 1。新方案初测与复测的六个模型进程互相独立；完整汇编在各自 16/32/64 长度内一致，跨 worktree 的规范化指令一致。

| 256×256×128 TN | 初测 | 独立复测 |
|---|---:|---:|
| cycles / BK 斜率 | 5268.916667 | 5270.770833 |
| 有用 MMAC 利用率 | 77.7389% | 77.7116% |
| 中点残差 / 长长度耗时 | 0.3864% | 0.6817% |

独立复测事先固定为 32/16/64 轮各一次；初测与复测均要求残差≤2%、无物理 spill、完整精度 PASS、实际/逻辑 MMOP=1、收益≥3%。失败候选不追加重复来挑选好结果。

## 其余结果与测量边界

256³ TT 的新候选为 71.0258%，相对原最佳仅提升 2.7239%，未达晋升门槛。32×256×128 的 TT/TN 虽然短循环更快，初测残差为 4.2685% / 7.8341%，未晋升，继续保留为后续优化证据。此快照尚未解决 32×256×128 TN 的非稳态问题；后续 MMAC 轮转实验不在本快照选择中。

主选择稳定 125/126 个；达到 80% 的有 14 个，达到 90% 的有 4 个。80–90% 尚未覆盖所有 block。四类共 503 条选择，验证标签计数为 `{'steady_measured': 442, 'provisional_nonsteady': 15, 'independent_repeat_confirmed': 46}`。`no_spill` 沿用无 runtime scratch/private bytes 的命名；历史行仍须检查 `evidence.assembler_fields` 中 SGPR/VGPR spill，`assembler_zero_spills` 才表示三者均为零。

每点 1 次 warmup、3 次计时取中位数，完整输出使用 seed29、scale1，atol/rtol=1e-3。利用率为 `2*BM*BN*BK / slope / 4096`，表示有用工作量相对模型峰值，非独立硬件 busy counter。TT/TN 表示输入搬运布局。完整 GEMM 的 TG 排布、实卡性能和 MNK tile 选择留给 BU 阶段。

## 主选择地图

数值为初测有用 MMAC 利用率百分比，TT / TN；`*` 为非稳态，`—` 为暂缓。

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
| 256×256 | 92.76 / 91.88 | 86.02 / 81.52 | 95.60 / 95.86 | 69.14 / 77.74 |

## 文件、版本与原始路径

- [best-configs.json](best-configs.json) 含全部 126 个 profile 的 503 条选择与原始 source_ids、结果/汇编 SHA、独立复测路径。
- [integration-proof.json](integration-proof.json) 记录父地图、2 条更新及 501 条未变选择。
- [evidence-index.json](evidence-index.json) 索引 38 个逐字节导出的审计/脚本；[初测原始审计](round99-native-raw-audit.json)、[复测原始审计](round100-native-raw-audit.json)、[确认结果](round100-confirmation.json) 和 [预先固定的复测计划](round100-confirmation-plan.json) 分别保留。
- `sources/` 保存 JSON sources 表所列源码、补丁、manifest 与构建证据；`validation.json` 记录有限数值、资源、源文件、精确审计和全量选择差异检查。

测试基线为 Triton `46f0bed3561364b63ba4046d9c8c0cb70380ea30`、BoltOPs `840d816b671983527e5b1acc831c5e67618fe981` 的隔离 worktree 与逐方案补丁。此前 2026-10-26 核对的 release/3.6.x 参考为 `1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3`，未以它替换正在测量的编译器。源码与原生库版本以各行 manifest 为准；Git 不保存大型 SO，构建和测量原件保留于 node60。

host 原始根目录为 `/home/hygon/xukang/code/pmd-triton-sessions/`；容器 `/xukang/` 对应 host `/home/hygon/xukang/`。初测在 `single-tg-pipeline-20261027-round99-split-b-slots-native/runs/`，独立复测在 `single-tg-pipeline-20261027-round100-split-b-slots-confirm/runs/`。本包包含选择结果、源码和审计，不复制全部 PMD 原始统计。历史已压缩的 stats 保持其原有 raw-stats.tar.gz / stats-archive.json，可用对应 stats_io.py 读取或恢复。

Git 远端在此前尝试中连接超时；提交、打包与是否实际 push 分别记在外部 receipt，不能由本文件推断已推送成功。
