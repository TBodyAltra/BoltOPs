# BK128：经过独立复测的无物理 spill 备选（round178）

256×128×128 TT 新增一个 64 KiB LDS、无 SGPR/VGPR/scratch spill 的配置。初测利用率 **33.540207%**，独立复测 **33.491757%**，分别为当前 64 KiB 最佳值 **33.908688%** 的 **98.9133% / 98.7704%**。两轮均通过 16/32/64 次 BK 循环稳态检查，残差分别为 **0.937990% / 0.998647%**。完整输出校验通过，两轮完整汇编逐字节相同，使用六个不同的模型进程测量该配置。

这是资源备选，性能最佳值保持原配置。全部 503 个最佳选项与之前发布版本逐条一致。继续使用相邻 `fp16_single_tg_20261027_round148/best-configs.json`；此前 26 条后续候选和 matrix-store 修正证据在 `fp16_single_tg_20261027_round167`。本目录的 `resource-recipe.json` 提供新的备选参数，不应替换全地图配置文件。

| 参数 | 值 |
|---|---|
| BM / BN / BK / 输入布局 | 256 / 128 / 128 / TT |
| WARP_SPECIALIZE / wasp_wdra | false / false |
| num_warps / num_stages | 8 / 1 |
| LOOP_UNROLL_FACTOR / MMA_K_SPLIT | 2 / 4 |
| LOAD_B_FIRST / SIMPLIFY_GROUP1 | true / true |
| optimize_epilogue / misched_regpressure | false / true |
| MLS / fragment MLS / global prefetch | true / true / 0 |
| async_copy / block_pingpong / matrix_store | false / false / false |
| mmac_layout_force / enable_v_mmac_cluster | -1 / 0 |
| ILP scheduling / gfx946 sched model v2 | true / true |
| LDS / next-free VGPR | 65536 bytes / 157 |
| private bytes / VGPR spill / SGPR spill | 0 / 0 / 0 |

改动只在隔离的 Bolt worktree175 中为内部 MLS kernel 增加默认关闭的 `SIMPLIFY_GROUP1` 常量。打开时要求 GROUP_SIZE_M=1，索引由 `(pid // ceil(N/BN), pid % ceil(N/BN))` 计算。对所有合法 GEMM TG 编号，该映射与原通用分组公式完全一致；代数证明和约 248 万个合法编号的检查记录包含在 round175 目录。没有固定运行时 M/N，没有改变 TG 排布、BK128 的内存传输、MMAC 分片、输出 mask 或参考答案。默认关闭时两个旧配置的 CPU ISA、资源和 PMD 对照保持一致。

本轮 CPU 检查 8 个源代码变体，PMD 初测 26 点、独立复测 5 点，共 **124 次原始 dispatch**（包含 warmup）。初测 7 个稳态、1 个未稳态；所有新候选均未达到性能晋级所需的 3% 初测提速，因此没有性能最佳值晋级。只预先选定一个最快的稳态零物理 spill 配置，进行一次独立复测，并要求两轮吞吐均达到 64 KiB 最佳值的 98%。未反复重跑失败项来选择结果。

利用率为 `2*BM*BN*BK / (cycles_per_BK * 4096)`，是单 TG 周期斜率折算的 PMD 模型峰值比，不是硬件 busy counter。没有 BU 实卡或多 TG 占用率结论。实际 PMD LDS 容量仍是 128 KiB，本配置使用 64 KiB；没有扩大资源上限。128³继续暂缓。

配置依赖所附内部 kernel 和 frozen Triton121/native95，不能直接传给任意生产版 API。Triton 测量基线提交为 46f0bed3561364b63ba4046d9c8c0cb70380ea30，Bolt 源代码基线为 840d816b671983527e5b1acc831c5e67618fe981；完整累计源码补丁和 GROUP1 增量补丁均包含在 round175。编译器 native SHA256 为 3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990，沿用此前已验证的 ABI，没有重建 C++ native。

`candidate-delta.json` 包含完整 8 条新增候选及地图父/当前 SHA256。`evidence-index.json` 将源码、审计和确认报告绑定到原始字节 SHA256。完整 stats、perf、输入输出及 kernel 保留在 node60 host 的 `/home/hygon/xukang/code/pmd-triton-sessions/` 对应 round175/176/177 目录。此文档包不包含大体积原始 stats 或 native 二进制；原始统计哈希和模型进程编号保留在审计中。
