# R001 / R003 / R004 BF16 GEMM：PMD 复测与 Gluon 分发

2026-11-17，node60 / `kxu_pmd_8569_fp8_20261017`，gfx946、1 DIE / 48 CU。BF16 输入/输出、FP32 累加，完整 K=2048。A 按 K 连续；TN 权重按 K 连续，TT 权重按 N 连续。以下为统一 Gluon 分发后的当前源码复测。

| 需求 / 布局 | MNK | BM×BN×BK | 批内块数 / LDS 槽位 / wave 配比 | 完整 dispatch µs | shader 窗口 µs | TOPS | 记录 |
|---|---|---|---|---:|---:|---:|---|
| R001 TN | 64×64×2048 | 16×16×128 | 4 / 2 / 1P+1C | 5.324 | 1.711 | 3.15 | [记录](../r001/1die/tn/README.md) |
| R001 TT | 64×64×2048 | 16×32×128 | 2 / 3 / 1P+1C | 5.744 | 2.132 | 2.92 | [记录](../r001/1die/tt/README.md) |
| R003 TT | 2560×64×2048 | 64×64×64 | 2 / 3 / 1P+4C | 10.565 | 6.954 | 63.52 | [记录](../r003/1die/tt/README.md) |
| R004 TT | 6656×64×2048 | 64×64×64 | 2 / 3 / 1P+4C | 25.327 | 21.695 | 68.89 | [记录](../r004/1die/tt/README.md) |

四项均在随机/全 1/随机三组完整校验后取 **9 次 warm 中位数**。每次计时均回读全部输出，容差 abs0.001+rel0.01，mismatch/nonfinite 均为 0；输出 guard、全部 TG 完成数和 SGPR/VGPR 无 spill 检查通过。完整 dispatch 含命令处理与分发，shader 窗口单独列出，不能与单 CU 流水长度直接比较。

所有完整计时均调用 `gemm_a16w16(..., backend="auto")`，并观察、核对实际选择的编译 handle。各项附原 MNK/stride 下的单 TG PERF；R001 另附完整输出 grid dispatch 的默认 CU 采样 PERF。采集复用计时 cache，并在运行前强校验完整 HSACO SHA256 一致；单 TG PERF 不作为完整 MNK 性能。各项 `record.json` 和 `benchmark.json` 保留原始样本、校验、资源、命令、环境及源码/配置/汇编/二进制哈希。旧测量单列为 `previous_measurement`，不混入本轮样本。

环境：PMD CoreArch HEAD_1703，128 KiB LDS/CU，455 ps shader 周期。Triton release/3.6.x base `6b111ee3` + MR565，LLVM-triton 含 MR21，DTK Clang 6.3.3 `3599a58d7`；实际 libtriton/Clang 二进制哈希见记录。测量源码固定为 `0a079d11a0174e289ff65fbb38323c1b5f35e187`；四项实测 kernel、配置和回放脚本均与该 commit 一致。本轮结果仅为 PMD 测量。

## 配置选择与实现

公共入口按架构/CU、布局、精度及 MNK 查询 config，再启动 config 指定的 kernel，不维护逐 kernel 名称分支。所有 dense Gluon 入口统一接收由实际权重 stride 推导的 `B_K_CONTIGUOUS`；固定布局、形状、精度和整除约束由 kernel 的 JIT 断言检查。persistent 的 FP16 参数已显式列为 precision profile，替代按 kernel 名称复用 BF16 参数。

R003/R004 共用 1P+4C、64 列、三槽流水。`SLOTS` 是 LDS 槽位数量，独立于编译选项 `num_stages`。更大 BK、不同 wave 配比、LDS 槽位及 persistent 候选的比较用于前期选型；本表仅报告当前选定实现的完整复测，抽样 grid 不作为完整性能。

普通 Triton 调用侧同时将非 Split-K 时未使用的 `stride_ck` 改为正值，满足 kernel 原有的正 stride 假设。

## 回归范围

当前环境下分发/原生小规模回归 **55 passed、4 deselected**，配置检查 **32 passed**。4 个未执行用例为 persistent 大网格与尾块用例；两个 persistent TT 尾块编译失败已在重构前源码复现，不计为通过，本轮未测 persistent 性能。此前的对照诊断保留在[验证记录](dispatch_refactor_validation.json)。

本轮[验证命令及运行时核对](retest_validation.json)、[分发日志](retest_dispatch.log)、[配置日志](retest_config.log)随记录保存。
