# FP16 单 TG block 流水配置地图（2026-10-27，round108）

本快照新增 **128×256×128 TN 的独立 B0/B1 槽流水**。初测利用率 70.9105%，独立复测 71.1641%，相对原稳定选择 68.0912% 提升 4.5129%。初测/复测残差分别为 1.2062% / 1.0986%。只更新该 profile 的 best / best_no_spill，两条选择变化，其余 501 条与 round103 逐条完全一致。

[best-configs.json](best-configs.json) 保存完整 config、launch_spec、实际寄存器分区、LDS、汇编 spill 和原始证据。BM/BN 为 32/64/128/256，BK 为 16/32/64/128；128×128×128 继续暂缓。63 个 block、126 个 TT/TN 主选择均通过稳态检查。四类共 503 条选择，历史受限备选保留各自验证标签，不能全部称为稳态或独立复测通过。

## 本轮配置

WASP/WDRA 开启，producer4 + consumer4 + consumer4；stage1、private single buffer、earlyFalse、UF1、split2。A-first / OneA 完整捕获 A，B 拆为两个独立 K64 槽。数据 empty/ready 为 A0/1、B0 2/3、B1 4/5；本方案 slot_turn=0、旧 mmac_turn=0，不使用 MMAC token6/7。保留父 LDS 96 KiB。请求寄存器分区 [8,248,248]，实际 [2,145,145]，n_regs=168。runtime scratch、private bytes、SGPR/VGPR spill 均为零。

MMAC layout force=-1，cluster0，ILP=False，modelV2=True；fragment MLS=True、split_b_global_probe=True、split_b_slots=True，mls_global_prefetch=0、schedule_hint=none；async_copy/BPP=False。退出前全部 12 waves 在 join2 会合，再回收六个数据信号量。具体参数以 JSON 为准。

## 验证与负结果

CPU104 对两个原 full-B 对照逐条比对指令和资源，六个候选全部无物理 spill。逐 producer wave、逐迭代核对 16/32/64 轮的全局源地址、父 LDS 目的地址和拷贝标志，与完整 B128 参考一致；检查实际消费者读取地址、完整 LGKM 等待后释放、MMAC 发射顺序、join2 和信号量回收。保留逐 wave 有限协议模型及负例，其证明范围不包含无界执行和 MMAC 物理完成时刻。

native105：2 个对照 + 6 候选×3 长度，共 20 个模型；native107 独立复测：2 个对照 + 1 候选×3 长度，共 5 个模型。**25 个模型、100 个 dispatch** 的完整输出精度、原始统计 MMOP 实际/逻辑比=1、仅 CU0、物理 spill 零及源文件哈希检查均通过。每点校验全部 32768 个输出。每个系列三个长度完整汇编一致，初测/复测六个候选进程独立。独立复测预先固定 32/16/64 顺序，没有追加重复挑选结果。初测和复测均要求稳态残差≤2%、相对旧选择提升≥3%。共享 PMD lease、最多 8 个自有模型，完成所有在途任务后交接。

round106 因缺少三个协议快照文件，在初始化模型之前失败，spec/result/manifest/model 数均为零。保留原始日志；round107 在新隔离目录补齐元数据后重试，GPU 代码及提名不变，setup-retry-proof 记录对应检查。该事件不能算一次失败性能测量。

TT 的三种槽轮转初测为 64.8165% / 47.1064% / 54.3967%，低于原 68.3806%；TN 的 token 轮转1/2为 57.0957% / 56.0686%，也未晋升。保留此前独立确认的 256×256×128 TN 77.7116%、32×256×128 TT 21.5810% / TN 20.0727%，以及全部 64 KiB 约束备选。

## 测量与使用

单 TG，grid=1；输入/输出 FP16、FP32 累加。K=BK×16/32/64 仅用于拟合流水成本，不按完整 GEMM MNK 选择 tile。每点 1 次 warmup + 3 次计时取中位数，seed29/scale1，完整输出与 CPU FP32 reference 比较，atol=rtol=1e-3。利用率为 `2*BM*BN*BK / cycles_per_bk_slope / 4096`，斜率为 `(cycles64-cycles16)/48`；这是有用工作量相对模型峰值的比例，不是硬件 busy counter。

按 profile 及 128/64 KiB LDS 与 scratch 约束选择 best / best_no_spill / best_64k / best_64k_no_spill，使用该行记录的 kernel 和编译器。no_spill 沿用 runtime scratch/private bytes 为零的历史定义；assembler_zero_spills 才同时要求 SGPR/VGPR spill 为零。BU 上重新验证精度、资源和流水后再调 TG 排布，私有参数不能单独复制到生产 API。

主选择 126/126 稳态，14 个达到 80%、4 个达到 90%，仍有优化空间；不表示每种 block 都能达到 80–90%。全量选择验证标签计数：`{'steady_measured': 440, 'independent_repeat_confirmed': 50, 'provisional_nonsteady': 13}`。

下表为主选择初测 MMAC 利用率百分比，TT / TN；独立复测值保存在 JSON 对应字段，— 为暂缓。

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
| 128×256 | 80.36 / 57.45 | 85.53 / 89.04 | 84.37 / 85.63 | 68.38 / 70.91 |
| 256×32 | 13.66 / 14.16 | 23.53 / 25.75 | 25.86 / 27.80 | 21.10 / 17.60 |
| 256×64 | 25.62 / 27.68 | 34.43 / 40.78 | 42.41 / 46.38 | 31.24 / 34.36 |
| 256×128 | 69.46 / 57.38 | 79.99 / 87.03 | 89.86 / 88.89 | 73.34 / 73.72 |
| 256×256 | 92.76 / 91.88 | 86.02 / 81.52 | 95.60 / 95.86 | 69.14 / 77.74 |

## 版本与证据

测量使用隔离 worktree：Triton 46f0bed3561364b63ba4046d9c8c0cb70380ea30，BoltOPs 840d816b671983527e5b1acc831c5e67618fe981，逐方案补丁和 manifest 见 sources。独立 B 子视图依赖 native95 父布局 MLS 修正，原生库 SHA256 为 3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990。其他配方保留各自版本。release/3.6.x 参考 1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3 未替换测量编译器。

- [best-configs.json](best-configs.json)：完整配置、资源、source_ids、结果/汇编 SHA、独立复测路径。
- [integration-proof.json](integration-proof.json)：两条更新、501 条不变。
- [evidence-index.json](evidence-index.json)：89 个逐字节审计文件和脚本的原始路径与 SHA。
- [初测原始审计](round105-native-raw-audit.json)、[复测原始审计](round107-native-raw-audit.json)、[复测计划](round107-confirmation-plan.json)、[确认结果](round107-confirmation.json)、[重试证据](round107-setup-retry-proof.json)。
- sources/ 保存 kernel/backend/CPP/runner、补丁及构建记录；validation.json 保存差异、资源、metric、SHA 与链接检查。

host 原始根目录为 /home/hygon/xukang/code/pmd-triton-sessions/。初测在 single-tg-pipeline-20261027-round105-rect-split-slots-native/runs/，独立复测在 single-tg-pipeline-20261027-round107-rect-slots-confirm-retry/runs/。容器 /xukang/ 对应 host /home/hygon/xukang/。本包不含大型 SO 与全部 PMD 原始统计；历史无损归档的统计通过 stats_io.py 读取/恢复，round90 新归档证明也在 evidence-index 中。

提交、打包、host 镜像及实际 push 分别记录在外部 receipt，不能由本文推断已经推送成功。
