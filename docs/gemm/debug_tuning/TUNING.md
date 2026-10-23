# GEMM 关键调优记录

截至 2026-10-23；支持范围见 [README](../README.md)，形状与 scale 见 [少伯 BU 支持需求](../SHAOBO_BU_REQUIREMENTS.md)。生产 kernel、默认 config 与 `builtin_gemm_configs.py` 不随实验修改。PMD 只校准与打磨少量流水候选，真实卡再做 autotune。

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

首轮按 arch_info 的64KiB门槛排除部分配置；后续确认PMD实际LDS为128KiB，HIP报告256KiB。128×128/BK64、BK128及128×256/BK128已有在实际PMD容量内的运行结果；生产容量表未改。不以编译通过代替资源可运行性或精度PASS。

## 当前实测配置（单die）

本轮代码基线：Triton `9fab6e70ed511f9de6a52cd4dd95eb60a3b74916`、BoltOps `840d816b671983527e5b1acc831c5e67618fe981`；私有库及构建溯源见debug记录。单die48CU，请求GPU2.2GHz，实际455ps/周期。

完整表二G171：4608×3072×6144、E1、自然432CTA；BM128/BN256、stages2、WDRA48(4,4,4)、GROUP_SIZE_M1、NUM_KSPLIT1，FP8 E4M3×E4M3→BF16；A FP32/K128，B选uint8/E8M0、N1/K128。相对本轮手动WDRA48 BK64基线；第一接口kernel外部副本，未改生产默认配置。每配置一次独立精度/预热与一次正式计时，JIT和CPU参考不计入cycles。

| 配置 | quota（producer/consumer） | cycles | LDS bytes | 精度 |
| --- | --- | ---: | ---: | --- |
| BK64，自动LIT，无epilogue优化 | 120/192/192 | 3256644 | 116224 | max_abs0.125，超限0 |
| BK64，仅重分配quota | 64/220/220 | 3255478 | 116224 | max_abs0.125，超限0 |
| BK128，LTS，优化epilogue | 104/200/200 | 1748203 | 99840 | max_abs0，mismatch0 |

三项均PASS、无scratch。quota单独仅减少0.0358%，没有实质收益；BK/layout/epilogue/quota组合减少46.32%（1.863倍），不是layout单因素结论。MMOP均10616832，VALU减少72.99%；VMEM指令增加41.79%，TCC请求仅增加1.95%，不能把VMEM指令数当作A/B加载字节数。正式完整尺寸每配置仅一次，不声称完整尺寸抖动区间。

### 单CTA受控对照

单die48CU，请求GPU2.2GHz，实际455ps/周期（约2.197802GHz）。以下是单CTA自然shape128×256×6144、BK128、WDRA48、LTS、stages2、优化epilogue、quota104/200/200的三次均值；不能代替完整需求shape结论。使用外部副本，未改生产kernel或默认config。 第一接口指 `gemm_a8w8_blockscale`，第二接口指 `w8a8_triton_block_scaled_mm`；本轮输入为E4M3×E4M3、输出BF16。B scale选uint8/E8M0作实验存储，需求只规定离线N1/K128，不据此推断B scale dtype。

| 对照 | 基线cycles | 候选cycles | 下降 | 精度 |
| --- | ---: | ---: | ---: | --- |
| 第一接口：dot提前到scale读取前 | 199843 | 189052 | 5.40% | max_abs0，mismatch0 |
| 第二接口：K改constexpr，保留K mask | 208286.67 | 199947 | 4.00% | max_abs0，mismatch0 |
| 第二接口：在静态K基础上dot提前 | 199947 | 187831 | 6.06% | max_abs0，mismatch0 |

三个候选对照均无private scratch，重复波动小于0.5%。dot提前不改变乘法/加法关联、scale helper或量化实现；MMAC/LDS/访存指令数不变，主要收益指向调度顺序。原E8M0 A/B大tile BK128 LTS因分区预算524>504失败；dot提前后actual102/190/190，已通过数值验证。

容量/性能负例：stages4需要32个abarrier ID，超过16容量；LIT布局的大tile BK128仍受VGPR限制；紧凑E8解码虽逐bit等价但未改善资源且变慢，不采用。async/BPP按实际生成路径和相同shape对照选用，没有普遍加速结论。

完整G171的LTS项使用原始外部副本，不包含后续dot提前。单CTA dot提前的5.4%收益不能直接叠加到完整G171的46.32%。第二接口的K constexpr对照以group_n/group_k已为constexpr的副本为基线，未改生产kernel。G036第一接口16×128/BK128 WASP44、dot提前完成7次测量，36683.86 cycles，波动1.524%，逐元素一致且无scratch；第二接口相同流水36507.29 cycles但波动2.397%，暂不标稳定，两接口均值差小于噪声。

G081（80×512×7168，A/B E8M0/K128）完成同副本受控对照：

| tile / 流水 | 自然CTA数 | cycles（三次均值） | 波动 |
| --- | ---: | ---: | ---: |
| 32×128/BK128 普通4warps | 12 | 114570.67 | 0.601% |
| 32×64/BK128 普通4warps | 24 | 102543.33 | 0.403% |
| 32×64/BK128 WASP44 | 24 | 79736.33 | 0.143% |
| 64×64/BK128 普通4warps | 16 | 135223.33 | 0.168% |

全部逐元素一致、无scratch。BN减半减少10.50% cycles；固定tile改WASP44再减少22.24%，合计30.40%。64×64将M padding从96增到128，实际MMOP增加1/3，表现更差。原source与dot提前副本的32×128普通流水仅差0.26%，不能把此例总收益归到source顺序。

第二接口对应副本复用32×64/BK128/stages2：普通流水107146 cycles，WASP44为80298，固定tile减少25.06%；两项均三次测量、逐元素一致、无scratch，波动分别0.511%/0.578%。相对其旧32×128普通流水143063.33，tile与流水组合减少43.87%。同WASP44配置的第一接口为79736.33，第二接口高约0.70%；仅三次样本，不作普遍接口优劣判断。新增两点的测量HEAD为ffafaa8（仅文档前进，kernel与840d基线无差异），逐点记录保留实际commit。

### 完整 G184：E8M0 A scale

M5120/N4096/K8192、E1、自然640CTA；输入E4M3×E4M3、输出BF16，A E8M0/K128，B离线N1/K128、实验存储E8M0。两项均BM128/BN256、stages2、WDRA48(4,4,4)、quota104/200/200、LTS与优化epilogue；同一编译器/代码基线，各一次精度预热、一次正式计时。

| 外部副本配置 | cycles | LDS bytes | 实际分区VGPR | max_abs / mismatch / 超限 |
| --- | ---: | ---: | --- | --- |
| 原顺序，BK64 | 5480083 | 49920 | 69/165/165 | 2 / 1226 / 0 |
| dot提前，BK128 | 3699477 | 99072 | 102/190/190 | 0 / 0 / 0 |

两项均PASS、全部输出有限且无scratch。候选组合降低32.49% cycles（1.481倍），不是单独BK或调度收益。正式计时各一次，不宣称重复性；hostSeconds分别16160.8/12480.75，仅作预算。正式MMOP均20971520，按原生16×16×32算量核对等于2MNK=343597383680 FLOP，未缩grid或跳过仿真。8CTA派生pilot曾减少32.74%，完整尺寸结果单独保存，不能相互替代。

BK64基线有packed-FMA未写先读检查警告，完整精度零超限仍不能证明警告误报；候选无该警告。完整文本与计数在仓库外保留，见debug记录。本轮共91个配置点，80个通过、11个因VGPR/barrier资源未编译通过；这不是接口验收通过率。逐点环境/版本/配置/数值/指标完整性审计通过，不改写原始结果。

## G171：原生汇编经验迁移实验（2026-10-22）

本轮未确认稳定收益，保留原基线。实验代码和精简数据见 [`g171_native_20261022/`](g171_native_20261022/)，不接入生产 dispatch、不修改默认 config。生成的 IR、汇编、日志、StatLog、PERF、输入输出和快照均留在仓库外。

环境为 node60 / `kxu_pmd_8569_fp8_20261017`，gfx946/sb，单 die、48 CU，建模周期 455 ps（约 2.198 GHz）。测量 BoltOps 为 `b23492fbf1e34d1cdcea2da4f180c96005a8d9ce`，Triton 源码 HEAD 为 `9fab6e70ed511f9de6a52cd4dd95eb60a3b74916`；实际私有编译器库 SHA、环境变量、编译/launch 选项在 `manifest.json`、`results.json` 及外部逐点 `environment.json` 中记录。发布分支随后前进不改变测量版本。

G171 派生自然单 TG：M128/N256/K6144，BM128/BN256/BK128，stages2，WDRA48 分区 (4,4,4)、VGPR 配额 (8,248,248)，LTS；A/B 输入 E4M3，输出 BF16；A scale FP32、每行/K128，B scale E8M0、每 N/K128。每点先编译、一次精度预热，再正式计时；JIT 不计入 cycles，hostSeconds 仅作预算。

| 实验 | 样本数 | 平均 modeled cycles | VALU / TG | 实际最高显式 VGPR 数 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原基线 | 2 进程共 6 次 | 96326.17 | 39232 | 190 | 保留 |
| 先合并 A/B scale | 2 进程共 6 次 | 96131.50 | 39232 | 200 | 仅快约 0.20%，波动 3.61%，无稳定收益 |
| 原 scale 预取版 | 3 次 | 106778.67 | 46728 | 210 | 慢于原基线 |
| A scale 寄存器轮换 | 3 次 | 105679.00 | 40968 | 226 | 去搬运仍未胜过基线 |
| A/B 均去搬运 | 3 次 | 106283.67 | 39464 | 227 | VALU 降低未转化为收益 |

修正后本次 seed20 的上述版本均与 CPU 参考及原基线逐元素一致，max_abs=0、mismatch=0、全部有限、spill=0；LDS 均为 101376 bytes，MMOP=24576、LDS 指令数=10008。此处的逐元素一致只针对已测输入，不能推出极端 scale、下溢、NaN/Inf 的普遍等价。原基线已使用 `v_pk_mul_f32`/`v_pk_fma_f32`，packed 运算并非本轮新增。先合并 scale 改变 FP32 运算关联，但没有减少本次布局的乘法/FMA 数。

轮换副本将循环展开两轮，交替 A scale VGPR 84:99 与 210:225；B 版本在消费完当前 raw B 后才发出下一轮读取，地址使用 v226。原有硬件 wait/barrier 全部保留。第一版遗漏 epilogue 对 raw A 68:83 的 live-out，数值失败即停止计时；最终版在奇/偶退出边恢复所需状态，已重新验证。错误版与首次 debug relocation 编译错误保留于外部目录，不作为 PASS。单独两轮展开的诊断也逐元素一致。实验结果说明“增加预取但去掉搬运”仍不足以解决当前流水瓶颈。

复现工具只准备新外部目录，不启动进程或队列；要求使用 manifest 固定的外部 IR 和 PMD 环境：

```bash
python docs/gemm/debug_tuning/g171_native_20261022/prepare.py \
  --evidence-root /xukang/code/pmd-triton-sessions/fp8-G171-native-ideas-20261022 \
  --output-root /xukang/code/pmd-triton-sessions/g171-native-replay
# 需要执行时显式逐 case 运行；本次发布未重跑：
# bash /xukang/code/pmd-triton-sessions/g171-native-replay/control/launch.sh --measure
```

`native-transform.py` 保留正确的两个轮换分支与 unroll-only 诊断；`scale_product.py` 保留 scale 关联改写；`bench-worker.py` 为已测驱动原样副本，依赖 manifest 中的 node60 历史私有环境。准备阶段校验输入 SHA256，输出目录必须是会话根下的新目录且位于 Git 外。离线检查确认提交的转换输出与实际执行汇编/IR 相同；不宣称重新完成 GPU/PMD 验证。

外部证据根为 `/xukang/code/pmd-triton-sessions/fp8-G171-native-ideas-20261022/`。合并 scale 的 warm PERF 为宿主机绝对路径：

```text
/home/hygon/xukang/code/pmd-triton-sessions/fp8-G171-native-ideas-20261022/best-profile/m5out/0/1/134522_python3.10.perf
```

PERF 开启后的独立 warm launch 为 99106 cycles，单列诊断，不混入上表未开启 PERF 的均值。关键快照 SHA256 为 `5a867fee4c8fe414cfda71159614c8266a9f159fc49ba20c84b98892a4f89f31`，仅引用，不把快照或 PERF 提交到 Git。

作为流水参考，gemmPerf `main@54bf03f5ce53a98c54d7fc800d52ae008c205aa2` 已在同环境编译，原生 128×128 tile 的 K1024/K4096 单 TG 工具 CPU dice 检查各 3/3 PASS，平均 modeled cycles 为 24873.67/53854.67。但其 A scale 为 FP32/128M×128K、B 为 FP32/1N×128K，且输出 FP16，不能与 G171 作同规格性能比较。构建使用仓库外头文件 overlay 修复两处 SDK 失效链接，并以 `OMP_TOOL=disabled` 避开 CPU golden 的 libarcher 启动崩溃；未替换计算实现。独立 gemmPerf kernel 跑通不代表已修复安装版 rocBLAS 缺 gfx946 kernel library 的环境限制。详细版本、修复与数据保留于 `gemmperf-run-20261022/summary.json`。

## 下一步与证据位置

优先将已验证候选带到实际卡上复测，并补未覆盖的scale类型、尾块和两die代表shape。小tile保留普通流水、WDRA44/WASP44、async；BPP只保留实际匹配域内候选。按 A FP32/K128、E8M0/K128、整行 scale 分别验收；补尾块、W 浮点 scale 和跨 CTA 验证后，再形成上卡 autotune 家族。第三个 FP8 接口仅维持兼容性 smoke；暂不扩大 split-K 或完整需求扫描。

上卡建议最小集：两个性能接口各测G036、G081、G171、G184；另补129×257×1000尾块、A整行scale一项、B FP32 scale一项。尾块先用可编译的小tile普通/WASP路径，不能强推已触及资源墙的大tile。第二die优先复核四个真实shape及实际频率、容量、CU/映射，再决定是否复用单die选择规则。全部属于后续待验证项，不提前标PASS。

当前生产第一接口与实验副本调用`remap_xcd(pid, GRID_MN)`时未接收返回值，helper本身返回新pid；本轮不宣称启用了8XCD重排，也不据此解释收益。该静态问题及大tile M子块方案在仓库外`STATIC-FOLLOWUP.md`单列，未改生产源码或创建新编译器补丁。

全部路径以前缀 `/xukang/code/pmd-triton-sessions/` 为准：

| 内容 | 外部证据 |
| --- | --- |
| FP8 普通路径基线 | `fp8-dense-smoke-20261020/performance-baseline.csv`、`REPORT.md` |
| async / BPP 第一轮 | `fp8-pipeline-probes-20261020/performance-summary.csv`、`final-results.json`、`REPORT.md` |
| Track A 参数化副本 | `perf-phase0-20261018/param_kernel.py` |
| Track A stages | `perf-trackA-stages-20261019/measurements.json/csv`、`equivalence.json`、`REPORT.md` |
| Track B 校准 / 成本 | `perf-trackB-20261018/REPORT-calibration-20261018.md`、`cost-model.csv`、`cost-model-metrics.csv`、`REPORT.md` |
| WDRA48 新编译器验证 | `wdra48-fix-20261020/final-results.json/csv`、`REPORT.md` |
| 完整G171三项对照 | `overnight-gemm-20261020/full-G171-results.json`、`FULL-G171.md` |
| 单CTA受控归因 | `overnight-gemm-20261020/controlled-pipeline-decomposition.json`、`CONTROLLED-DECOMPOSITION.md` |
| G036七次重复性 | `overnight-gemm-G036-repeatability-20261020/REPORT-repeatability.md` |
| G081第二接口迁移 | `overnight-gemm-vllm-G081-transfer-20261020/REPORT-transfer.md` |
| G081同副本tile/流水对照 | `overnight-gemm-G081-occupancy-20261020/REPORT-controlled.md` |
| 完整G184两项对照 | `overnight-gemm-20261020/full-G184-results.json`、`FULL-G184.md`、`G184-terminal-status.json` |
| 全部点的环境/代码/编译/数值/指标索引 | `overnight-gemm-20261020/record-index.csv`、`records/` |

Git 只保留关键结论和证据索引，不接收原始日志、StatLog、输入输出、IR、汇编、HSACO 或快照。后续关键记录至少含环境/commit、对齐配置、精度阈值及误差、建模周期与重复次数、资源限制和外部证据路径。


### FP8 E8M0 与小 tile WASP 补充（2026-10-23）

[实测代码、配置与结果](fp8_scale_wasp_20261022/README.md)：G048 派生单 TG 的 native E8M0 decode 在两个接口均约降低 22.5% cycles；G081 小规模自然 grid 的 tile + WASP22 组合约降低 10%。256×256×64 仍受 VGPR 配额或 spill 限制，G171 未找到可靠的新收益。原生硬件 scale 指令已在最小及 stages1 实验执行，但较大点更慢，流水路径存在固定编译器缺口，暂不采用。全部是单 die 小规模实验，未更改生产 dispatch/default config，不等于完整需求或双 die 验收。

表 2 已按 scale dtype 与 block size 分组，见 [少伯 BU 需求](../SHAOBO_BU_REQUIREMENTS.md#table-2)。源需求中 B scale dtype 未注明，实测采用的 E8M0 配置不能反推为所有需求的确定 dtype；选取后续调优代表点时须保留这一边界。

2026-10-23 原始需求复核：此前 G171/G036 等 N1 B scale 实验只证明所测组合，不能作为 GLM 公开 FP32/N128/K128 scale 的验收；Qwen checkpoint scale 为 BF16，DeepSeek dense 为 E8M0/N128/K128，Hy3 为 scalar。后续先按 Excel 明确字段优先、缺失项采用公开配置补齐的口径对齐 scale，再继续性能比较；部署版本绑定另作上线复核。完整来源与冲突见 [需求核对](requirements_audit_20261023/REPORT.md)。历史数据不改写，不将新确认配置标成已测。


### G171 对齐 FP32/N128/K128 scale 后的流水筛选（2026-10-23）

[配置、关键代码与数据](g171_fp32_scale_20261023/REPORT.md)：采用 A FP32/M1/K128、B FP32/N128/K128，B scale 物理紧凑存储；自然单 TG 为 M128/N256/K6144、tile128×256×128、stages2。最佳 WDRA48 aligned 普通加载/dot-first 副本两进程共6次均值149829.8 cycles，较无 spill 的生产8warp基线198953.7少24.69%；两个随机种子均与CPU/生产输出逐元素一致，spill=0。实际动态VGPR配额(112,196,196)，LDS101376 bytes。环境固定单die48CU、455ps，完整commit/库指纹/编译参数随记录保留。

MLS未稳定胜过匹配的普通加载；A/B下一轮LDS预取相对重编IR对照约少4.1%，仍慢于上述最佳。普通加载预取分支受VGPR预算限制，没有绕过守卫。生产与最佳的真实PERF已采集，绝对路径及SHA见记录；未计算本次MMAC利用率。未改生产kernel/default config，未跑完整G171、双die、第二接口迁移或尾块。历史E8M0/N1结果继续保留原口径，不能用作本次FP32/N128需求验收。
