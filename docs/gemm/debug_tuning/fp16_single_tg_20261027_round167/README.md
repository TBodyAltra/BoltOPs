# FP16 单 TG：round148 后续调优证据（截至 round167）

本目录补充已结束的 26 条候选和一次 matrix-store 正确性修正；503 个已选配置逐条核对未变化。继续使用相邻 `fp16_single_tg_20261027_round148/best-configs.json` 获取最佳配置和源文件。这里的 JSON 是候选增量，不能替换该配置文件。

| 记录 | 新增候选 | 结果 |
|---|---:|---|
| round155 → 地图156 | 6 | 2 个稳态但更慢，4 个未稳态；无晋级 |
| round163 → 地图164 | 10 | 修正后的 matrix-store：8 个稳态但更慢，2 个未稳态；无晋级 |
| round166 → 地图167 | 10 | scratch/VGPR spill 为零，但存在 SGPR spill；8 个稳态，2 个未稳态；无晋级 |

256×128×128 TT、64 KiB LDS 的 round166 最高初测为 34.316354%，旧最佳为 33.908688%，增幅 1.202245%，未达到 3% 晋级门槛，没有进行胜出配置的独立复测。该候选 UF=8、8 warps、optimize_epilogue=true、misched_regpressure=false，SGPR spill=22；不能称为无物理 spill。旧最佳同样存在 SGPR spill=4。全部 10 个 round166 候选的 SGPR spill 原值均保留。

利用率按 `2*BM*BN*BK / (cycles_per_BK * 4096)` 计算，是单 TG 周期斜率折算的模型峰值比，不是硬件 busy counter。采用 16/32/64 次 BK 循环标定、完整输出校验和 <=2% 中间点残差门槛。新的最佳配置仍需初测与独立复测均提速 >=3%，并保持完整 ISA 一致。未稳态项和小幅初测提升均不能替换最佳值。没有多 TG 或 BU 实卡性能结论。128³继续暂缓。

matrix-store 修正从实际 MFMA LinearLayout 生成 repeated CTA tile 的寄存器/lane/warp 到输出地址映射，并检查覆盖与无重叠。256×128×128、4-warp TT/TN 在同配置和同输入下分别从 28659/28658 个错误变为完整输出校验通过。未改写参考答案或置换结果。修正仅针对已验证的 NoWASP/NoWDRA 路径；边界及其他拓扑没有扩展验证结论。详情见 `compiler-fix-round164.txt` 和资格 JSON。

编译器 7eb9fbc909873af0faaf37002362bde031b406530473095d104ba86df3926193 来自隔离 worktree161/native162；仅替换 native95 的一个 MLS 对象。5 个旧配置的 CPU ISA/资源与 PMD 对照通过。round163 的 10 个新配置均无物理 spill，35 个测试点/140 次 dispatch 校验通过；更慢或未稳态，因此无性能晋级。源码、完整源码补丁、增量修正、构建与依赖哈希一起保存于 round162 子目录。它们是可审查快照，不是面向任意 Triton ABI 的通用二进制。

原始 perf、输入/输出和失败快照保留在 node60 host 的 `/home/hygon/xukang/code/pmd-triton-sessions/` 对应 round 目录中；本包包含审计与源码，不包含大体积原始统计或 native 二进制。已归档的 stats 通过对应 case 的 `stats-archive.json` 和 `raw-stats.tar.gz` 读取，逐字节及 SHA256 校验保持一致。原失败记录仍为失败。

`candidate-delta.json` 绑定父地图和当前地图 SHA256，包含全部 26 条原始候选字段及晋级资格记录。`evidence-index.json` 列出本次导出的文件、来源与 SHA256；审计脚本及报告均按原字节复制。后续 round169/171 的预取实验正在独立进行，其未完成结果不在此处作为性能证据。
