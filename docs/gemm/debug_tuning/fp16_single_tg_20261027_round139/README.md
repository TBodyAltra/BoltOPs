# FP16 单 TG 流水地图：round139（2026-10-27）

本轮确认四个 profile 的 64 KiB LDS 改进，更新 best_64k / best_64k_no_spill 共八条选择。其余 495 条选择与 round119 完全一致；best / best_no_spill 主选择保持不变。128×128×128 继续暂缓。

| BM×BN×BK / 布局 | 原 64 KiB 最佳 | 初测 | 独立复测 | 复测相对收益 | 复测中点残差 |
|---|---:|---:|---:|---:|---:|
| 128×256×128 TN | 34.8086% | 39.8586% | 39.7026% | 14.0597% | 1.5164% |
| 128×256×128 TT | 35.9906% | 48.8081% | 48.8193% | 35.6446% | 1.6561% |
| 256×256×128 TN | 45.5795% | 52.0123% | 51.9849% | 14.0533% | 1.6241% |
| 256×256×128 TT | 45.1429% | 49.6408% | 49.6826% | 10.0564% | 0.1345% |

以上新配置实际 LDS 均为 65536 B，scratch/private bytes、SGPR spill、VGPR spill 均为零。百分比是有用 MMAC 工作量相对模型峰值的比例，不是硬件 busy counter。此表只比较 64 KiB 档；不限此 LDS 档的主配置性能更高，仍保留原选择。

## 配置与流水

沿用冻结 CPU133 worktree、native95 库（SHA256 3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990）。WASP/WDRA producer4 + consumer4 + consumer4、stage1、private single buffer、OneA 完整捕获、UF1、MMA_K_SPLIT2、A-first、A/B LDS 别名、独立 B0/B1 槽；async_copy API / BPP 均关闭。请求 VGPR 分区为 [8,248,248]；实际分区需求与完整编译参数在 JSON 对应行中，不应混淆请求额度和实际需求。

改进来自 hcu_wasp_split_b_slot_turn_wait_mask：128×256×128 TT/TN 使用 mask3，256×256×128 TT/TN 使用 mask2。slot_turn=2，旧 mmac_turn=0，early_B1=False。bit0/1 将对应 K64 fragment 的 MMAC token 等待移到数据 ready、完整 LDS 读取/LGKM drain、empty release 之后，第一条 MMAC 之前，使数据捕获能够提前进行。未移动的 fragment 仍允许原有 MMAC 与 LDS 指令交错。

别名保护仍完整保留：A 写之前等待所有实际重叠 B 半槽的 empty；A-ready 后等待八个 consumer waves 完成当前代 A 捕获，再允许 B 覆盖。TT B1 最低物理地址 32768 B、TN 8192 B，按实际 MLS 布局判断重叠，不能按逻辑半块判断。数据使用信号量 0..5，MMAC turn 使用 6/7；12 waves 全部 join2 后再回收八个信号量。

## 验证与未采用项

CPU134 的八个默认对照 ISA/资源完全一致。24 个 wait-mask 候选中 20 项编译成功且无物理 spill；256×256×128 TT/TN 的 mask1/3 共四项因分区 VGPR 预算不可行被编译器拒绝，未在 PMD 执行。

20 项均检查四个 producer waves 在 16/32/64 长度的 112 个迭代的实际全局/LDS 地址和标志；consumer 检查覆盖八个 wave ID、全部 64 lanes、首/中/末迭代及 A/B0/B1 各区域。实际 LLVM/ISA 审计检查当代 A phase、所有 DS 读取位于 ready→empty 所有权窗口、完整 drain 后释放、移动位的等待位置、MMAC token/handoff、退出 join 和信号量回收。审计器初版过度要求未移动 fragment 的 empty 必须先于第一条 MMAC，修正为保留合法交错；GPU 源码未改，失败记录与修正说明保留。

耦合别名和 late-wait 的有限协议模型：六个 mask0 对照与旧模型状态/转移数量一致；18 个正例 PASS，51 个负例检出。覆盖 mask1/2/3、A 仅覆盖 B0/同时覆盖 B0+B1、每角色一 wave 四代以及每角色四 waves 的二代/四代。模型检查有限数据所有权、phase/计数和 MMAC 发射顺序，不能声称无界证明或物理 MMAC 完成时刻。

native136：八个默认对照加 20 项三长度，共 68 点 / 272 dispatch；native137：八个对照加预先提名的五项独立三长度，共 23 点 / 92 dispatch。共 91 点 / 364 dispatch，全部全输出精度 PASS、实际/逻辑 MMOP=1、仅 CU0、无物理 spill，源文件 SHA 不变。每项初测和复测六个模型进程不同，三长度完整汇编一致，跨目录规范化 ISA 一致。独立复测顺序为 32/16/64，只运行一组，不追加重复挑选结果。

64×256×128 TT mask1 初测稳态且约有 10% 收益，但复测中点残差为 3.1180%，超过 2% 门槛，未晋升。32×256×128 TT/TN、64×256×128 TN 的部分候选虽然表观更快，初测不稳态，同样未晋升。

另外保留两轮负结果：native130 的三项 early-B1 提前发布方案，和 native135 的七项 BM64 full-B / split-B / alias / early-B1 方案，均未产生满足晋升条件的选择。CPU133 已为 BM64 完成实际地址、全 lane 读取和同步资格检查；这不代表其性能已达到目标。full-B 数据协议模型采用八个等价 data-only waves 的精确全局对称归约，并保留小图节点/边等价检查；此归约不用于角色相关的 MMAC turn 模型。

round123 还确认了一个 256×256×128 TT 的无 WASP/WDRA、真实零物理 spill 的资源备选，复测 42.4517%。它相对当时 legacy SGPR23 选择较慢，未替换最佳项；本轮新别名方案也已高于该资源备选。256×128×128 TT 的资源备选复测未稳态，未接受。原始报告在资源备选 JSON 中，不能把它解释为全局最快零 spill 配置。

round139 地图整合共审计 native130/135/136/137 的 127 点 / 508 dispatch，追加 30 条候选证据，保留所有未晋升记录。历史失败、模型反例、控制器修正和统计归档均保留来源与 SHA。

## 测量与使用

126 个 profile，覆盖 BM/BN=32/64/128/256、BK=16/32/64/128、TT/TN，暂缓 128³。共 503 条选择。单 TG、grid=1；输入/输出 FP16、FP32 累加。K=BK×16/32/64 只用于流水标定，不按完整 GEMM MNK 选择 tile。TT：A[K,1]、B[N,1]；TN：A[K,1]、B[1,K]。

每点一次 warmup 加三次计时取中位数；seed29/scale1，对 CPU FP32 reference 全输出比较，atol=rtol=1e-3。斜率=(cycles64-cycles16)/48；利用率=2×BM×BN×BK/slope/4096。中点残差相对长测量时间不超过 2%；新晋升要求初测和独立复测均至少提升 3%。主选择均稳态，其中 15 项初测达到 80%、4 项达到 90%；不能将主选择结论推广到所有历史备选。

no_spill 沿用 runtime scratch/private bytes 为零的定义，仍可能有 SGPR-to-VGPR spill。assembler_zero_spills 同时检查 SGPR/VGPR spill。使用时读取每条选择的资源和验证标签。BU 上还需复核正确性、资源和流水，再调整 TG 排布；这里没有测量多 TG occupancy。

测量基于 Triton 46f0bed3561364b63ba4046d9c8c0cb70380ea30、BoltOPs 840d816b671983527e5b1acc831c5e67618fe981 的隔离 worktree 和逐方案补丁。release/3.6.x 参考头 1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3 未作为本轮测量编译器。配置依赖记录的私有接口，不能直接作为当前生产 API 的配置字典。

- [best-configs.json](best-configs.json)：完整配置、资源、独立复测与原始路径。
- [integration-proof.json](integration-proof.json)：八条选择更新、495 条保持不变。
- [evidence-index.json](evidence-index.json)：原始证据与脚本的逐字节来源及 SHA。
- [初测审计](round136-native-raw-audit.json)、[独立复测审计](round137-native-raw-audit.json)、[确认结果](round137-confirmation.json)。
- [实际地址](round134-parent-address-audit.json)、[全 lane 读取](round134-consumer-read-audit.json)、[同步 ISA](round134-ownership-isa-audit.json)、[有限协议模型](round134-alias-late-turn-protocol-model.json)。
- [资源备选](resource-alternatives-published-zero-physical-spill-alternatives-round123.json)。
- sources/：选中配置依赖的源码、补丁和构建证据；validation.json：差异、资源和导出检查。

host 原始根目录 /home/hygon/xukang/code/pmd-triton-sessions/，容器 /xukang/ 对应 host /home/hygon/xukang/。初测 single-tg-pipeline-20261027-round136-alias-late-turn-native/runs/，复测 single-tg-pipeline-20261027-round137-alias-late-turn-confirm/runs/。包中不含大型 SO 和全部原始 PMD 统计。已关闭统计的无损归档通过 stats_io.py 读取/恢复，归档证明纳入索引，结果和汇编 SHA 不变。

提交、tar、host 镜像和实际 push 由外部 receipt 分别记录；本文不表示 push 已成功。
