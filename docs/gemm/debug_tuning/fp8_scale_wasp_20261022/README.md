# FP8 scale 与小 tile WASP 对照（2026-10-22—23）

本记录为 node60 / PMD gfx946 的仓库外实验，目标是为上卡 autotune 选择少量流水家族。没有修改生产 kernel、默认配置或测量编译器，没有恢复旧精度队列。所有尺寸均为裁剪后的自然 grid；不能视为完整 G 编号需求的性能验收，也不是全卡 TFLOPS 或 80% MMAC 利用率证明。第三个 FP8 接口保持原 smoke 范围，本轮性能工作针对前两个接口。

环境：**1 die、48 CU、455 ps/cycle（2.197802 GHz）**，PMD LDS 上限 131072 B；HIP/arch_info 的容量报告不同，不能替代实测容量。测量 Triton 源 HEAD `9fab6e70ed511f9de6a52cd4dd95eb60a3b74916`，BoltOps HEAD `b23492fbf1e34d1cdcea2da4f180c96005a8d9ce`；实际私有运行库 SHA、环境变量在 [environment.json](environment.json)，逐点有效编译参数、资源、源码/二进制 SHA 在 [results.json](results.json)。MR !506 的后续审计提交不用于这些性能比较。

每点独立进程：JIT warmup 编译，一次实际 launch 做精度/预热，再三次计时；主指标是 `stats.txt` 的 `system.gcSimCycles`，不含 JIT。hostSeconds 仅用于预算。周期样本、VALU/MMOP/LDS/VMEM/SCA、LGKM 等待计数保留在 [results.csv](results.csv) / JSON；等待计数不作可相加的 wall-time 分解。失败点与 spill 排除点也保留，不绕过资源检查。

## 可保留的候选

| 派生用例 / 接口 | M/N/K | 配置 | baseline cycles | candidate cycles | 结论 |
|---|---|---|---:|---:|---|
| G048 / aiter | 16/128/4096 | BM16 BN128 BK128，WASP(4,4)，s2；整行 A scale FP32，B E8M0/K128 | 46976.33 | 36431.67；独立复测 36141 / 36422 | 原生 E8M0 decode 约减少 22.5% |
| G048 / vLLM | 同上 | 同上 | 47159.33 | 36518.33 | 同接口受控对照约减少 22.6% |
| G081 / aiter | 80/128/7168 | baseline BM32 BN64 BK128 WASP(4,4)；candidate BM16 BN64 BK128 WASP(2,2)，s2 | 79665.33 | 71571.67 | tile + warp 组合约减少 10.2% |
| G081 / vLLM | 同上 | 同上 | 79207.33 | 71176 | 同接口组合约减少 10.1% |

上表全部 max_abs=0、mismatch=0、无 spill。G081 的自然 CTA 数由 6 变为 10，同时减少 M padding；这不是“只改变流水”所得的收益。M32/N64/K7168 的 P2C2 基线还做了三次独立进程、共九次计时，均值 72367.67 / 72371.67 / 72323.67；第二接口同配置 71769.33。第一接口该小形状的原生 E8M0 decode 追加点为 70922.33，仅作为进一步候选。

WASP(2,2) 的调用参数是 `num_warps=2, num_stages=2, wasp_partition_warps=(2,2), wasp_wdra=False`，kernel 的 `WARP_SPECIALIZE=True`。实际整个 CTA 的 wave 数由编译器 metadata 确认。避免把 producer/consumer 各 2 waves 写成 WDRA48，或只按输入参数推断 CTA 总 wave 数。

原生解码使用 `v_cvt_pk_f32_e8m0`，保留低 FP32 半部；FP32 scale 直接原样返回。单 kernel 穷举 256 个编码的高/低结果，逐 bit 校验通过，包括 code0 的 FP32 位模式 `0x00400000` 和 code255 的 `0x7fc00000`。证据根为 `/xukang/code/pmd-triton-sessions/overnight-fp8-native-cvt-20261022/`。这是实际支持的 ISA 转换，不是 PMD 不支持指令的软件替代，不改变全局 scale 布局。

追加小 N 对照：G036 BN64/P2C2 为 26863.67 cycles，BN32/P1C1 为 26992.33；G048 对应为 35953 / 36037.67。均逐元素一致、无 spill，但与已验证 native BN128 家族仅有约 1–2% 差距，且自然 CTA 数变化，不据此推出通用选择规则。小 BN 的 LDS 更低，可列入上卡候选。

## 没有采纳的尝试与边界

- **256×256×64** 已实际编译。WDRA48 普通/MLS 源及原生 decode 版本均触发每 SIMD VGPR 配额检查；普通 8/16-warp 原生 decode 版本分别 spill 117/78，未进入性能执行。此 tile 仍没有可采纳的结果。
- 当前优化 G171 单 TG（128/256/6144）中，原生 decode 将 VALU 39232 降到 33104，但 cycles 96126.33 对 95950.33，候选三次样本波动 4.34%，没有可靠收益；不因指令减少而替换现版本。
- G171 producer 原生解码：同轮 control 96328.67，late 98805，prefetch 102835.33；均正确、无 spill，但更慢。LDS bytes 不变，额外 LDS 写和等待仍在，未采纳。
- G171 解码放到 MMAC turn signal 后、或结合 scale register carry，分别 108957 / 101491 / 102540.33，对照 97000；均更慢。A scale 改 M 连续也没有在当前优化 IR 上得到收益；producer vec4 版本仍有 LLVM translation 缺口。
- 整行 A scale 的单纯 load hoist 几乎不改变 G048；将 A scale 移到 K reduction 之后会改变 FP32 运算结合顺序，当前样本虽通过但收益很小，暂不作为通用默认方案。
- 早期四个 `*_nativecvt` 编译失败来自临时 helper 的不一致返回类型，随后用显式 constexpr else 修复为 `*_nativecvt_v2`；不能把这些失败记成硬件不支持。完整原始失败日志仍在外部目录。

## 原生硬件 scale：已执行，不是当前快路径

独立最小 M16/N16/K128、BK32、A/B 均为 E8M0/K128（重复用于四个 K32 硬件分组）的 `tl.dot_scaled`，实际生成并执行 `v_mmac_scale`，CPU max_abs=0/mismatch=0、无 spill，证明该环境可以执行此 native FP8×FP8 指令。不能据此宣称两个生产接口已改为硬件 scale。

较大单 TG 的 stages1 路径也通过精度容差：G184 派生 128/256/8192、8 warps，两次 907069 / 904651.67 cycles；4 warps 为 852142.33；max_abs=0.00048828125、mismatch=2、容差违规 0。G081 派生 32/64/7168、2 warps 为 266758 cycles，max_abs=7.6293945e-6、mismatch=1、容差违规 0。它们明显慢于已有路径，且不逐 bit 相等，**不采用**。

流水化仍有编译缺口：普通 stages2/3 在 `mxscale_copy_ds2buf` 报“pipeliner doesn't know how to predicate this op”；WASP/WDRA 在 HCU specialization pass 失败。未进入设备执行，属于当前固定编译器的缺口，不能标 PMD 不支持。BK128 在旧测量库中还有 scaled 指令选择/SmallVector 越界；已合入 !509 的最新源码已有混合格式选择修正，本轮未切换运行库，不能把旧库的失败归于最新 release。完整日志、IR 和环境分别在 `overnight-fp8-hw-scale*-20261022/` 外部目录。

## 复现与下一步

本目录 Python 为实测的外部副本和 driver，不接入生产 dispatch。`run_scale_round.py` 必须先复制到业务仓库外，并使用独立实验根目录；`env.sh`、`guard.py` 针对原 node60 快照，其他环境需明确替换。计划只取 [configs.json](configs.json) 中已执行的配置，`source` 对应模块名。运行根下保留 `round_name/case_name/request.json`；driver 从该目录启动，先编译检查 LDS/spill，再预热、校验和计时。不要在 Git 目录中启动 driver。原始证据根为 `/xukang/code/pmd-triton-sessions/overnight-fp8-20261022/`，host mount 为 `/home/hygon/xukang/code/pmd-triton-sessions/overnight-fp8-20261022/`。

下一阶段优先把两个已确认的家族带到实际单 die / 双 die 卡上 autotune：小 M 的 BM16/BN128 WASP44 + native E8 decode；G081 的 BM16/BN64 WASP22。保留对应普通流水候选，重新测真实 shape、尾块和分布；每点继续记录 die 数、实际频率、Triton/BoltOps commit、运行库 SHA、编译选项与环境。G171 保留现有 128×256×128 WDRA48；256×256×64 与硬件 scale 流水作为明确有资源/编译缺口的后续项，不提前算性能优化通过。

本次主表共 81 个探索点，其中 66 PASS、5 个 spill 排除、10 个编译失败；包含已明确修正的临时脚本错误，不能用作接口通过率。每个 PASS 的输出、源代码/HSACO SHA、计时样本数、资源和环境已逐点复核，见 [publication-verification.json](publication-verification.json)。

两项 native 候选另各采集了一次精度/预热与一次 warm PERF，均 max_abs=0/mismatch=0。外部绝对路径与 SHA 见 [perf-manifest.json](perf-manifest.json)；索引 0 为 cold，1 为 warm，raw PERF/StatLog 不进 Git。

收尾补测：G171 的 B scale 原生硬件融合、A scale 保留 FP32 后乘，在小 M32/N64/K1024 及 M128/N256/K6144、stages1 上均逐元素一致且无 spill，但周期为 40147.67 / 569451.67；大点远慢于现有约 96k 的外部优化版本，不采纳。两项完整证据位于 `overnight-fp8-hw-scale-gemm-20261022/round-final-bscale/`，没有改生产实现或量化 A。
