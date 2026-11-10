# FP16 单 TG 流水配置地图：round148（2026-10-27）

本次确认 64×256×128 TT 在 64 KiB LDS 档的性能改进，更新 best_64k / best_64k_no_spill 两条选择；其余 501 条与 round139 完全一致。best / best_no_spill 主选择保持不变。128³ 继续暂缓。

| 配置 | 原 64 KiB 最佳 | 初测 | 独立复测 | 复测相对收益 | 复测中点残差 |
|---|---:|---:|---:|---:|---:|
| 64×256×128 TT | 20.9682% | 24.4385% | 23.1252% | 10.2871% | 0.7047% |

初测和复测指标分别保留。每个提名只运行一组独立复测，不追加重复挑选结果。以上是有用 MMAC 工作量相对模型峰值的比例，不是硬件 busy counter，也不替代不限 64 KiB 档的主选择。

## 新配置

WASP/WDRA producer4 + consumer4 + consumer4，stage1、private single buffer、MLS、OneA 完整捕获、UF1、MMA_K_SPLIT2、A-first、A/B 64 KiB LDS 别名、独立 B0/B1 槽；slot_turn=2、wait_mask=1、旧 mmac_turn=0、early_B1=False，async_copy API/BPP 关闭。请求寄存器分区 [8,116,116]，实际需求 [2,113,113]，整体描述符 next_free_vgpr=80；实际 LDS 65536 B，scratch、SGPR/VGPR spill 均为零。

其余编译选项与 launch_spec 在 JSON 中逐项保留。新配置依赖冻结 CPU133 GPU worktree、CPU140 资格检查及 native95 库 SHA256 3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990。worker 的精确配额准入依赖打包的 CPU-admission 快照；这些私有参数需要配套源代码。

## 寄存器资源备选

| BM×BN×BK / 布局 | 原始同流水 seed 配额总和 → 新配额总和 | 独立复测 | 中点残差 |
|---|---:|---:|---:|
| 64×256×128 TT | 504 → 240 | 23.1252% | 0.7047% |
| 128×256×128 TT | 504 → 360 | 48.7665% | 1.7118% |
| 256×256×128 TN | 504 → 468 | 52.0098% | 1.3635% |
| 256×256×128 TT | 504 → 468 | 50.6555% | 0.7127% |

配额总和按每 SIMD 的三个四-wave 分区计算。资源对比基准是各自的原始 WASP 流水 seed（总和 504）；吞吐对比基准是原性能地图的 best_64k。64×256×128 TT 的旧性能基准是 NoWASP 方案，没有 WASP 分区额度字段，两个比较基准分别记录。

资源备选要求初测与独立复测均稳态、吞吐至少达到原 best_64k 的 98%、实际物理 spill 为零，并相对原同流水 seed 减少配额。四项中包含新性能赢家，另三项作为资源备选单独保存，性能地图仍按至少 3% 的初测及独立收益门槛选择。这里没有测量多 TG occupancy。

32×256×128 TT 的资源备选复测残差 2.4812%、吞吐约低 2.98%；128×256×128 TN 残差 7.3287%。这两项未接受，原选择保留。复测准备阶段曾错误读取 NoWASP 基准的分区配额字段，任务在执行模型前退出；新目录明确分开资源 seed 与性能基准后按原提名规则执行，失败日志保留。

## 验证与负结果

CPU140：13 个默认对照 ISA/资源完全一致，26 个配额候选均编译成功、无物理 spill。检查实际 LLVM wdra.init、分区设置指令与 ISA/描述符、四个 producer waves 的 112 个迭代全局/LDS 地址、八个 consumer waves 全 64 lanes 的 A/B0/B1 读取、phase、完整 drain 后释放、MMAC token 与退出同步。原 GPU 源码和 native 库保持冻结。

native141 初测 91 点 / 364 dispatch，native145 独立复测 31 点 / 124 dispatch；合计 122 点 / 488 dispatch 全输出精度 PASS、实际/逻辑 MMOP=1、仅 CU0、无物理 spill。每项三长度完整汇编一致，跨目录规范化 ISA 与 CPU 参考一致，初测和复测的六个模型进程不同，源文件 SHA 不变。

另外完成去掉跨 consumer-role MMAC turn 的探索：数据所有权同步完整保留，使用六个数据信号量 0..5。有限 DATA-only 模型通过 10 个对称归约图对照、6 个正例、54 个错误变体、32 个有限等价保护条件图对照；数学指令在该数据投影中视为 stuttering，不声明 MMAC 发射顺序、物理完成时刻或无界证明。第一版错误地预期所有保护条件移除都必须产生反例，分类修正及原失败记录保留，实际 GPU 数据保护条件未删除。

CPU143 的八个新配置通过实际地址、全 lane 读取、六信号量同步与退出检查；native146 的 32 点 / 128 dispatch 全部精度及原始统计校验通过。两项稳态但慢于现有选择，六项未稳态，没有提名独立复测或晋升。round148 将这八项负结果追加进地图，全部 503 条选择与 round147 相同。

## 测量与使用

126 个 profile，覆盖 BM/BN=32/64/128/256、BK=16/32/64/128、TT/TN，暂缓 128³；共 503 条选择。单 TG、grid=1，输入/输出 FP16、FP32 累加。K=BK×16/32/64 用于流水标定，不按完整 GEMM MNK 选择 tile。TT：A[K,1]、B[N,1]；TN：A[K,1]、B[1,K]。

每点一次 warmup 加三次计时取中位数；seed29/scale1，与 CPU FP32 reference 全输出比较，atol=rtol=1e-3。斜率=(cycles64-cycles16)/48；利用率=2×BM×BN×BK/slope/4096；中点残差相对长测量时间不超过 2%。新性能晋升要求初测和独立复测均至少提升 3%。126 个主选择均稳态，其中 15 项初测达到 80%、4 项达到 90%；历史约束备选仍保留各自验证标签。

no_spill 沿用 runtime scratch/private bytes 为零的定义，可能包含 SGPR-to-VGPR spill；assembler_zero_spills 同时检查 SGPR/VGPR spill。BU 上应按记录的源版本复核正确性、资源与流水，再调整 TG 排布。

测量基于 Triton 46f0bed3561364b63ba4046d9c8c0cb70380ea30、BoltOPs 840d816b671983527e5b1acc831c5e67618fe981 的隔离 worktree 和逐方案补丁。release/3.6.x 参考头 1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3 未作为本轮测量编译器。新增固定 pitch / prefix 输入诊断尚不属于此自然 stride 地图，后续按独立 context 记录。

- [best-configs.json](best-configs.json)：完整配置、资源、独立复测与原始路径。
- [性能整合](round145-integration-round147.json)、[无 turn 负结果整合](round146-integration-round148.json)、[证据索引](evidence-index.json)。
- [初测原始审计](round141-native-raw-audit.json)、[独立复测审计](round145-native-raw-audit.json)、[确认结果](round145-confirmation.json)。
- [资源备选](compact-quota-resources-published-compact-quota-resource-alternatives-round145.json)。
- [配额资格](round140-cpu-screen.json)、[全 lane 读取](round140-consumer-read-audit.json)、[同步 ISA](round140-ownership-isa-audit.json)、[精确准入数据](round141-cpu-admission.json)。
- [无 turn 模型](round142-alias-no-turn-protocol-model.json)、[无 turn 审计](round146-native-raw-audit.json)。

host 原始根目录 /home/hygon/xukang/code/pmd-triton-sessions/，容器 /xukang/ 对应 host /home/hygon/xukang/。初测 single-tg-pipeline-20261027-round141-alias-vgpr-quota-native/runs/，复测 single-tg-pipeline-20261027-round145-alias-vgpr-quota-confirm-retry/runs/。包中不含大型 SO 和全部原始 PMD 统计。已关闭统计无损归档后通过 stats_io.py 读取/恢复，归档证明保留原始成员 SHA/长度、结果与汇编 SHA。本次还完成 round10 历史 PASS 统计的 1363 case / 5452 成员无损归档，释放约 15.5 GiB；它是存储校验，不是新性能资格结论。

提交、tar、host 镜像和实际 push 由外部 receipt 分别记录；本文不表示 push 已成功。
