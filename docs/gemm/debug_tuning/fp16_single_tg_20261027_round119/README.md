# FP16 单 TG 流水配置地图：round119（2026-10-27）

本轮确认两项 64 KiB LDS 档的 TT 改进。仅更新对应 profile 的 best_64k / best_64k_no_spill，共四条选择；其余 499 条记录与 round113 完全一致，所有主选择 best / best_no_spill 保持不变。128×128×128 继续暂缓。

| BM×BN×BK / 布局 | 原 64 KiB 最佳 | 初测 | 独立复测 | 复测相对收益 | 复测中点残差 |
|---|---:|---:|---:|---:|---:|
| 32×256×128 TT | 11.2353% | 14.7467% | 14.7762% | 31.5162% | 1.8050% |
| 128×256×128 TT | 34.7029% | 35.9906% | 35.9730% | 3.6597% | 1.4731% |

两项实际 LDS 均为 65536 B，scratch/private bytes、SGPR spill、VGPR spill 均为零。以上为有用 MMAC 工作量相对模型峰值的比例，不是硬件 busy counter。64 KiB 限制增加了流水等待；这些数值不替代更宽松 LDS 档的主配置。

## 流水及适用条件

沿用 WASP/WDRA 的 producer4 + consumer4 + consumer4、stage1、private single buffer、OneA 完整捕获、UF1、MMA_K_SPLIT2，开启 hcu_wasp_alias_ab_slots。完整 config、launch_spec、请求和实际寄存器分区均在 [best-configs.json](best-configs.json) 对应行中。

A 不再单独分配 LDS，而是以原来的类型和布局 reinterpret 整个 B 的连续 64 KiB 父分配。A 写入前等待所有物理重叠 B 槽的 empty；A-ready 发布后，等待所有八个 consumer waves 完成当代 A 捕获，再允许 B 覆盖。当前代 A 捕获等待使用 Aempty0、上一 A phase xor1。原有上一代 Aempty 等待仍保留；移动原来的 Bempty 等待，不重复消费。

TT 的 B1 地址从 32768 B 开始，TN 的 K-minor MLS 分片交错排列、B1 从 8192 B 开始，不能用逻辑半块大小判断别名。已检查的六项为 BM32/128/256、BN256、BK128、TT/TN；BM64 尚未获得该新流水的地址与协议资格，不能直接扩大适用范围。两项胜出 TT 的 A 只与 B0 重叠。

数据 empty/ready 使用 A0/1、B0 2/3、B1 4/5；MMAC slot-turn2 使用 6/7，旧 mmac_turn0、late wait mask0。退出时全部 12 waves 先 join2，再回收八个信号量。若 A 同时覆盖 B1，producer 路径存在额外同步，不能沿用旧流水的无 producer ebarrier6 断言。async_copy/BPP 均关闭。

## 验证与未采用项

CPU114 的初版地址重叠判断遗漏 128×256×128 TN 的交错 B1；实际地址检查找到反例后即排除，未在 PMD 执行。CPU115 在新的隔离 worktree 修正布局判断：六个默认关闭别名功能的对照 ISA/资源完全一致，六个新配置全部 64 KiB、无物理 spill。

六项均检查四个 producer waves 在 16/32/64 长度、共 112 个迭代的全部实际地址和标志；另核对八个 consumer wave ID 的 A/B 读取地址、完整 LGKM 等待后释放、当前 A 捕获等待相位、copy/ready 顺序、MMAC token 与退出同步。负例保留。读取审计器初版未识别 LLVM variadic barrier 调用签名，修正解析器后完成检查；GPU 源码保持不变。

有限协议模型 v3 显式计算信号量计数与 phase，覆盖一角色一 wave 四代，以及每角色四 waves 的二代和四代用例，分别检查 A 只覆盖 B0 与同时覆盖 B0/B1。六个正例 PASS，17 个负例检出；前两版未通过模型验证，不作为准入证据。模型使用明确的 producer collective-copy 抽象，不能声称无界证明或物理 MMAC 完成时刻。

native116 在六个默认对照 PASS 后因控制器沿用旧字段而退出，未执行任何 alias 候选，单独保留 6 点 / 24 dispatch 审计。native117 仅修正控制器，使用冻结的 CPU115 GPU 源码：六个对照加六项候选三长度，共 24 点 / 96 dispatch。native118 为六个对照加两个预先提名配置三长度，共 12 点 / 48 dispatch。两轮合计 36 点 / 144 dispatch，全输出精度 PASS、实际/逻辑 MMOP=1、仅 CU0、无物理 spill，源文件 SHA 保持不变。每个胜出配置初测和独立复测的六个模型进程互不相同，三长度完整汇编一致，跨目录规范化 ISA 相同。未增加重复挑选结果。

初测 32×256×128 TN 的残差为 4.6328%，128×256×128 TN 为 5.9259%，均未稳态；256×256×128 TT/TN 虽稳态但慢于旧 64 KiB 选择。以上四项不晋升。无 spill 不能单独代表性能更好；旧约束选择保留原资源与验证标签。

## 测量与使用

单 TG、grid=1；输入/输出 FP16、FP32 累加。K=BK×16/32/64 只用于流水标定，不按完整 GEMM MNK 选 tile。每点 1 次 warmup + 3 次计时取中位数；seed29/scale1，对 CPU FP32 reference 全输出比较，atol=rtol=1e-3。斜率为 (cycles64-cycles16)/48；利用率为 2×BM×BN×BK / slope / 4096，中点残差相对长测量时间必须不超过 2%。晋升要求初测与独立复测均至少提升 3%。

126 个主选择均稳态，主选择中 15 项初测达到 80%、4 项达到 90%。503 条选择包含历史非稳态约束项，不能将主选择的检查结论推广到全部备选。no_spill 沿用 runtime scratch/private bytes 为零的定义；assembler_zero_spills 才同时要求 SGPR/VGPR spill 为零。BU 上仍需复核正确性、资源和流水，再调整 TG 排布。

测量使用 Triton 46f0bed3561364b63ba4046d9c8c0cb70380ea30、BoltOPs 840d816b671983527e5b1acc831c5e67618fe981 的隔离 worktree 和逐方案补丁。新别名流水使用冻结的 CPU115 源码及 native95 库 SHA256 3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990。release/3.6.x 参考头 1842e37b2a5b0bc21984fc0d4550b30ae2d3b8c3 未作为本轮测量编译器。配置依赖记录的私有前端和后端接口，不能直接用于生产 API。

- [best-configs.json](best-configs.json)：完整配置、资源、独立复测及原始结果路径。
- [integration-proof.json](integration-proof.json)：四条选择更新、499 条保持不变。
- [evidence-index.json](evidence-index.json)：逐字节原始审计与脚本的来源和 SHA。
- [初测审计](round117-native-raw-audit.json)、[独立复测审计](round118-native-raw-audit.json)、[确认结果](round118-confirmation.json)。
- [实际地址](round115-alias-parent-address-audit.json)、[consumer 读取](round115-alias-consumer-read-audit.json)、[同步 ISA](round115-alias-ownership-isa-audit.json)、[协议 v3](round115-protocol-model-v3.json)。
- sources/：选中配方的源码、补丁和构建证据；validation.json：地图差异、资源、指标及导出检查。

host 原始根目录为 /home/hygon/xukang/code/pmd-triton-sessions/；初测 single-tg-pipeline-20261027-round117-alias-ab-slots-native-retry/runs/，复测 single-tg-pipeline-20261027-round118-alias-ab-slots-confirm/runs/。容器 /xukang/ 对应 host /home/hygon/xukang/。包中不含大型 SO 和全部原始 PMD 统计；已结束历史统计的无损归档可通过 stats_io.py 读取/恢复，归档证明纳入索引，结果和汇编 SHA 保持不变。

提交、归档、host 镜像和实际 push 分别由外部 receipt 记录；本文不表示 push 已成功。
