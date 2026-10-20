# GEMM 关键调优记录

截至 2026-10-20；支持范围见 [README](../README.md)，形状与 scale 见 [少伯 BU 支持需求](../SHAOBO_BU_REQUIREMENTS.md)。生产 kernel、默认 config 与 `builtin_gemm_configs.py` 不随实验修改。PMD 只校准与打磨少量流水候选，真实卡再做 autotune。

## Die 模式与性能记录

少伯支持 1/2/4 die，每 die 48 CU，对应总计 48/96/192 CU。当前 PMD 只能测试 1 die 和 4 die；后续优化重点为 1 die 和 2 die，现阶段优先在 1 die（48 CU）上调优。2 die 暂记待实测，不从 4 die 结果推断已验证。Die 数与 XCD 数分别记录，不能直接等同；每波 CTA 数仍需按实际配置测量。

每次性能测量均在仓库外保存环境 manifest，并在结果行中引用其路径或 ID；配置变化必须生成新的 manifest：

| 必填项 | 记录内容 |
| --- | --- |
| 执行环境 | PMD / 实卡、arch、die 数、每 die CU 数、总 CU 数；XCD 拓扑及证据来源 |
| 频率 | 实际使用的 GPU/CU 时钟及单位；PMD clock domain 与配置值；其他影响结果的时钟单独记录 |
| 软件版本 | Triton、BoltOps 完整 commit ID 和工作区状态；PMD/工具链版本，私有编译器库路径及 SHA256 |
| 编译与运行参数 | 完整有效 config、constexpr、launch grid、warps/stages、WASP/WDRA/BPP/async 等开关，以及编译命令和目标架构选项 |
| 环境变量 | 会影响编译、缓存、选路、仿真或计时的变量有效值，包括相关 TRITON/HCU/HIP/PMD 设置；缓存及私有库的实际选择 |
| 测量口径 | shape、dtype、scale、stride、warmup/重复次数、cycles 统计来源、hostSeconds、指令/资源统计及精度结果 |

默认值也须记录解析后的有效值；未知项明确标记 unknown 并附原因，不补猜值。现有历史数据缺 die/频率证据时标记未核实，不用于新口径下的性能排名。仅在 die 数、频率、软件与有效配置对齐后计算直接性能收益；不同 die 的结果分别汇报。

## 比较口径

- 必须对齐 M/N/K、dtype、scale 粒度/存储、stride、tile、warps、stages、split-K、编译器和 PMD 配置。先 warmup 编译、执行预热，计时不含 JIT。
- 主指标为 PMD 建模 cycles，同时记录 hostSeconds、VALU/MMOP/LDS/L2、LGKM、LDS/VGPR/spill；宿主墙钟只用于预算。
- 单 TG 必须 M=BM、N=BN、grid 自然等于 1 CTA，不缩小 grid 伪造；Track B 固定 NUM_KSPLIT=1、GROUP_SIZE_M=1。
- `gemm_a16w16` 先 `_get_config(M,N,K)` 取基线再 merge 覆盖，保留 GROUP_SIZE_M、cache_modifier、NUM_KSPLIT、SPLITK_BLOCK_SIZE 等必需字段。
- 单 CTA 原始 cycles 不代表整算子最优；还需单位输出成本、驻留与实测波次。不能仅凭 48 CU 假定每波固定 48 CTA。
- stats 仅减同一 worker / XCD 的基准，不跨工作目录读取旧统计。数值校验先于性能结论。

## 已测基线与 stages

### FP8×FP8 对齐基线

两内核均使用真实单 CTA：M=N=BM=BN=64，BK=128，warps=4，stages=2，
GROUP_SIZE_M=1，无 split-K，BF16 输出。A scale 为 FP32，W scale 为 E8M0，
W 分组为 N1/K128；W scale dtype 是本次实验选择，不能补写为需求默认。
这些是内部 kernel 的对齐配置测试，不是两个包装层的默认配置直接比较。

| A scale | K | gemm_a8w8_blockscale cycles | w8a8_triton_block_scaled_mm cycles |
| --- | --- | --- | --- |
| K128 | 1024 | 21280.3 | 25441.0 |
| K128 | 2048 | 37057.0 | 45225.0 |
| K128 | 4096 | 64041.0 | 80382.3 |
| 整行 | 1024 | 20843.7 | 22947.7 |
| 整行 | 2048 | 31693.0 | 35509.7 |
| 整行 | 4096 | 52976.3 | 60091.7 |

12 点均通过数值检查、无 spill、LDS 16384 bytes。每点先编译、执行预热，再测 3 次；
主指标 PMD `system.gcSimCycles`，不含 JIT 和执行预热。hostSeconds 只作预算。
两个 K1024 点的 range/mean 抖动略超 2%（2.096%、2.109%），不能据此解释小幅收益。
单 CTA 排名不能直接用作真实 shape 的配置选择规则。

### Track A：FP8×FP4

仓库外参数化副本的 BK32 stages 补测：16 项输出对照全部一致，12 个性能点无 spill。
16×32 tile 的 stages2/3 相对 stages1 有约 3.2%–6.6% 收益；32×64 tile 反而慢约
1.7%–3.7%。stages 应按 tile 保留候选，不能直接套用 BF16 的收益；未改生产 kernel。

### Track B：FP16/BF16 gemm_a16w16

单 TG 成本表共实际编译加载 66 点：54 点完成测量，12 点因 metadata LDS 超出
arch_info 的 64 KiB 容量域而排除执行，不能记作编译失败。
最低单 TG 原始周期为 64×64×128、warps8、stages2：67022.3 cycles，零 spill。
该排名不等于完整 GEMM 最快；仍需拟合单位输出成本、资源驻留和真实 shape 波次。
FP16 两个配对点与 BF16 相差 -0.0432% / +0.2587%。未运行 split-K 或多 CTA 回验。

PMD 已实测建模软件流水：Track B 校准中 stages3 快约 16.3%，LGKM 等待约减半。不能删掉 stages 维度，也不能把 BF16 的收益直接套用 FP8×FP4。

## FP8×FP8 async / BPP 候选

以下来自第一轮外部副本的 27 个精度/性能点；其后 WDRA 调度断言已修复，见 [debug](DEBUG.md)。这批数据不是新 WDRA 编译器上的性能排名。

| 路径 | 同配置配对结果 | 当前判定 |
| --- | --- | --- |
| vLLM风格64×64/BK128/4warps async，K4096 | 普通77145→async66327 cycles，下降14.02% | 保留候选；实际生成async LDS加载，零spill；配对中的量化group_n/group_k均为constexpr |
| aiter风格128×128/BK64/8warps BPP，K1024 | 普通54620.7→BPP50586.7，下降7.39% | 确认BPP实际应用 |
| 同上，K4096 | 普通212888.7→BPP162118.3，下降23.85% | 同BK有收益，但不等于全局最优 |
| aiter128×128/BK128/8warps普通流水，K4096 | 130158.3 cycles | 比上述BK64 BPP更快，必须保留作跨BK参考 |
| vLLM128×128/BK64 BPP | 未出现BPP应用标记；常量分组后仍未选中 | 不能计作BPP支持；开关改变cache swizzle不等于实现该流水 |
| 大tile async | aiter128×128的4/8warps均变慢；vLLM8warps仅约2.3%差距 | 未找到足够稳健的晋升依据 |


输入 FP8 E4M3、输出 BF16，W scale E8M0/N1/K128，默认 A FP32/K128。另测 A E8M0/K128 与整行 scale，通过 atol=.01 / rtol=.01。BK64 改变分段累加顺序：部分长 K 点存在少量非零差异（最大 .00390625），不能宣称跨 BK 逐元素一致。

生产 aiter 路径仍要求 GROUP_K==BK；外部 BK64 副本使用 `(k*BK)//128` 读 K128 scale。尚未晋升公开接口。BPP 匹配器 `BlockPingpong.cpp` 对 post-dot scale 模式排除 BK128；普通非 BufferLds 路径还要求 M/N 按 128 粒度，不能只打开开关就判定 BPP 生效。必须检查 IR / 汇编实际选路。

## WASP / WDRA 与资源约束

| 模式 | `wasp_partition_warps` | `wasp_wdra` |
| --- | --- | --- |
| WDRA48 | `(4,4,4)` | True |
| WDRA44 | `(4,4)` | True |
| WASP44 | `(4,4)` | False |
| WDRA88 | `(4,4,4,4)` | True |

循环还需显式 `warp_specialize`；只传 launch option 不会启用。当前 WASP 缓冲 stages 允许 2/4，不直接套普通流水 stages3。consumer ping-pong 由编译器控制，旧配置字段只作历史参考。

`wasp_partition_regs` 使用 producer-first 顺序。每分区 quota 按 4 对齐且受独立上限约束；等权 WDRA48 总和需被 12 整除，WDRA88 被 16 整除。例如 88+208+208=504 合法，不能机械补到 512。Clang 打印顺序不一定等于参数顺序，应按分区和值核对。当前 FP8 大 tile 自动 quota `(308,96,100)` 虽总和为 504，producer 仍超过 256，不能绕过校验。

布局必须记录 stride：TT 指 A K-major、B N-major；TN 指 A/B 均 K-major。历史 FP16 256×256×64 的 TT `ds_read_m32x16_fp16` 可用 quota 112/196/196；TN `ds_read2_b64` 不能直接照搬，需要 BK32 或 MLS 路径再验证。旧胜出配置仍保存在 [gfx946 JSON](../../../boltops/gemm/triton/configs/gemm_fp16/gfx946/)，版本变化后必须复测，旧 TOPS 不作为当前基线。

新 WDRA48 已执行 128×128/BK32（LDS 50432）和 128×64/BK64（42112）；128×128/BK64（66816）与 BK128（99584）超当前 65536-byte 容量，未执行。不以编译通过代替资源可运行性或精度 PASS。

## 下一步与证据位置

优先测合法 WDRA48 配置的流水成本，分析 scale 中转/布局、LDS 与 producer VGPR。小 tile 保留普通流水、WDRA44/WASP44、async；BPP 只保留实际匹配域内候选。按 A FP32/K128、E8M0/K128、整行 scale 分别验收；补尾块、W 浮点 scale 和跨 CTA 验证后，再形成上卡 autotune 家族。第三个 FP8 接口仅维持兼容性 smoke；暂不扩大 split-K 或完整需求扫描。

全部路径以前缀 `/xukang/code/pmd-triton-sessions/` 为准：

| 内容 | 外部证据 |
| --- | --- |
| FP8 普通路径基线 | `fp8-dense-smoke-20261020/performance-baseline.csv`、`REPORT.md` |
| async / BPP 第一轮 | `fp8-pipeline-probes-20261020/performance-summary.csv`、`final-results.json`、`REPORT.md` |
| Track A 参数化副本 | `perf-phase0-20261018/param_kernel.py` |
| Track A stages | `perf-trackA-stages-20261019/measurements.json/csv`、`equivalence.json`、`REPORT.md` |
| Track B 校准 / 成本 | `perf-trackB-20261018/REPORT-calibration-20261018.md`、`cost-model.csv`、`cost-model-metrics.csv`、`REPORT.md` |
| WDRA48 新编译器验证 | `wdra48-fix-20261020/final-results.json/csv`、`REPORT.md` |

Git 只保留关键结论和证据索引，不接收原始日志、StatLog、输入输出、IR、汇编、HSACO 或快照。后续关键记录至少含环境/commit、对齐配置、精度阈值及误差、建模周期与重复次数、资源限制和外部证据路径。
