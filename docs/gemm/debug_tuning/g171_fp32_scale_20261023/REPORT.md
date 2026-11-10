# G171：FP32 block scale 单 TG 流水筛选

日期：2026-10-23。采用 Excel 优先、缺失字段按公开模型补齐的口径；B FP32/N128/K128 的来源见 [需求核对](../requirements_audit_20261023/REPORT.md)。本轮只验证第一接口内部 kernel 的单 TG；没有运行完整需求尺寸或第二接口迁移。

**同日增量实验的最新候选：普通abarrier MMAC turn，6次均值106722 cycles；详见末节。前面的表保留初次筛选结果。**

**最佳源代码候选：WDRA48 + aligned 普通加载、dot 后读 scale，149862.7 cycles，独立复测 149797.0；两次共 6 launch 均值 149829.8，较无 spill 的生产 8-warp 基线少 24.69% cycles（约 1.328×）。** 不修改生产 kernel、默认 config、编译器或 libtriton。

## 固定条件

- PMD gfx946/sb，单 die、48 CU；实际 config.ini 的 48 个 CU section 与 shader/cache clock=455 ps，约 2.197802 GHz。system clock=1000 ps，UMC=2276 ps。原始配置 SHA 与时钟证据见 [hardware-config.json](hardware-config.json)。没有测试双 die。
- 测量 BoltOps `b23492fbf1e34d1cdcea2da4f180c96005a8d9ce`，Triton `9fab6e70ed511f9de6a52cd4dd95eb60a3b74916`。实际使用私有 Python/parser 库，不能只凭 Git commit 重建环境；库及 backend/工具链指纹见 [software-identity.json](software-identity.json)，完整编译/运行变量见 [environment.json](environment.json)。
- G171 全尺寸 M4608/N3072/K6144；本轮是它的自然单 TG M128/N256/K6144，BM128/BN256/BK128，grid=(1,)，stages2、GROUP_SIZE_M=1、无 split-K。绝非缩小 grid 截断完整 shape。
- A/B=E4M3 FP8，C=BF16；A scale FP32 `[128,48]`、M1/K128；B scale **FP32 物理紧凑 `[2,48]`、N128/K128**，两者 K-group 连续。非二次幂随机 scale；没有 N1 展开输入、E8M0 解码或软件替代计算。
- 每点先 warmup 编译并执行预热/精度检查，再独立计 3 launch。主指标 `system.gcSimCycles`，只取各自 m5out/0/1..3；JIT 与预热不计入。hostSeconds 为宿主仿真预算。
- 两组编译参数均显式 optimize_epilogue=True、mmac_layout_force=3。B=内置合并参数：matrix_instr_nonkdim=16/kpack=2/waves_per_eu=2；H=昨日参数：0/1/1。详见每行 effective_options。normal 基线 warps8；WDRA48 请求 num_warps4、partition=(4,4,4)，实际 metadata num_warps=12。
- WDRA 请求 regs=(8,248,248) 会被 backend 重新分配。最佳点实际 s_set_vgpr_size 为 **(112,196,196)**，压力需求 (100,190,190)；不可把请求值当实际分配。LDS101376 bytes，spill=0。

## 已测数据

| 点 | 路径 | 编译参数 | cycles（3次均值） | range/mean % | LDS bytes | spill |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| P01-production-w8 | 生产普通8warp | B | 198953.7 | 0.353 | 49152 | 0 |
| P02-production-wdra48 | 生产副本+WDRA48 | B | 161802.3 | 0.136 | 101376 | 0 |
| P06-wdra48-oldflags | 生产副本+WDRA48 | H | 197451.3 | 0.380 | 101376 | 0 |
| P03-dotfirst-wdra48 | dot-first | H | 178203.7 | 0.254 | 101376 | 0 |
| P10-aligned-load-oldflags | aligned 普通加载 | H | 179370.7 | 0.284 | 101376 | 0 |
| P04-MLS-dotfirst-wdra48 | aligned MLS | H | 169456.0 | 3.485 | 101376 | 0 |
| P07-MLS-compact-Bscale | MLS+紧凑B scale LDS | H | 166482.7 | 0.624 | 99344 | 0 |
| P11-aligned-load-builtinflags | **aligned 普通加载（最佳）** | B | 149862.7 | 0.202 | 101376 | 0 |
| P08-MLS-builtinflags | aligned MLS | B | 171130.0 | 0.746 | 101376 | 0 |
| P09-compact-Bscale-builtinflags | MLS+紧凑B scale LDS | B | 166243.7 | 0.352 | 99344 | 0 |
| P15-IR-control-oldflags | 重编IR对照 | H | 166004.0 | 0.249 | 99344 | 0 |
| P12r-matrix-prefetch-normalized | IR A/B预取 | H | 159192.7 | 0.378 | 99344 | 0 |
| P16-IR-control-builtinflags | 重编IR对照 | B | 166063.3 | 0.206 | 99344 | 0 |
| P13-matrix-prefetch-builtinflags | IR A/B预取 | B | 159290.3 | 0.154 | 99344 | 0 |
| P20-best-repeat | 最佳独立进程复测 | B | 149797.0 | 0.176 | 101376 | 0 |

匹配比较：同 B 参数 P01→P02 少 18.67%；同 H 参数 P06→P03（dot-first）少 9.75%。纯 MLS 需对照同样移除 M/N 取模、对齐假设相同的普通加载：H 下 P10→P04 少 5.53%，B 下 P11→P08 却慢 14.19%。P04 的抖动3.485%超过2%目标，因此 H 下5.53%只是观察值，尚不能作为稳健收益。不能把对齐和 MLS 的作用混为一个收益，也不能混用 H/B 比较后归因流水。

紧凑 B scale LDS 副本仅加载/暂存每 N128 一个 FP32，再在 consumer 展开，保持全局 `[2,48]` 原布局和数学次序；LDS101376→99344 bytes。P04→P07 / P08→P09 的小幅周期收益作为候选，不宣称单独确定收益。

下一轮 A/B LDS 预取在当前 MMAC 交接后、当前 scale 算术前发出；末轮不越界，数学操作顺序保持原样。P15→P12r 少 4.10%，P16→P13 少 4.08%。源 TTGIR 在两组参数下恰好相同，这不是两个独立 kernel 家族的重复证明。MLS encoding 使用现有私有 parser 的等价 spelling，不修改编译器；重编本身会改变 HSACO，因此只对各自重编的原 IR 对照归因预取。结果仍慢于最佳普通加载，不晋升。生成器输出与实际执行 IR 的 SHA 逐一一致，见 [offline-verification.json](offline-verification.json)。

## 数值与复测

CPU golden 按 K128 分段 FP32 dot，并按 M 行及 N//128 取 A/B scale，最后 BF16 round。所有已执行候选对 CPU golden 均 max_abs=0、mismatch=0；seed20 候选对生产输出也逐元素一致。另用 seed21 分别独立 launch 生产与最佳：两者 CPU/生产对照同样全零误差。只覆盖这两个随机输入及此对齐 shape，不能代替尾块、整尺寸或所有量化范围验收。
最佳两个独立进程各 3 launch 的抖动 0.202% / 0.176%，均值差 0.044%。[results.csv](results.csv) 为紧凑表；[results.json](results.json) 保留配置、指令/资源、结果路径与二进制指纹。没有重跑旧暂停队列。

## 指令与等待诊断

| 指标 | 生产8warp | 最佳普通加载 |
| --- | ---: | ---: |
| VALU | 45912.000 | 40900.000 |
| MMOP | 24576.000 | 24576.000 |
| LDS | 8448.000 | 14808.000 |
| VMEM | 10496.000 | 6080.000 |
| SCA | 4936.000 | 29924.000 |
| totalTCCRequests | 45153.000 | 42228.000 |
| waitVmCounter | 22282.667 | 997.444 |
| waitLgkmCounter | 11781.667 | 13834.222 |
| hostSeconds | 10.327 | 10.123 |

指令数为 PMD CU 计数求和；VM/LGKM 等待也是跨 SIMD 的统计汇总，不能相加解释 wall time。MMOP 数保持相同，WDRA 减少部分 VALU/VMEM，但 SCA/LDS 仍显著，不能据此声称 MMAC 达 80%。尚未从本次 PERF 计算 MMAC 利用率。

## PERF

已采集真实 HYGPERF 二进制并核验 magic 与 SHA；warmup 与计时各一份，文字 stats_xcd0.perf 不混入。详见 [perf-manifest.json](perf-manifest.json)。PERF 点独立计时一次，不混入上表未开 PERF 的三次均值。

- 生产8warp：198846 cycles；node60 宿主机绝对路径：`/home/hygon/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/PF-production/m5out/0/1/151721_python3.10.perf`。
- 最佳WDRA48：149727 cycles；node60 宿主机绝对路径：`/home/hygon/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/PF-best/m5out/0/1/151841_python3.10.perf`。

## 排除项与保留边界

- 生产4warp 编译得到 spill141；H 参数生产8warp spill13。均在实际执行前排除，不能拿 spill 配置充当速度基线。生产基线是无 spill 的可执行8warp。
- 普通加载的 A/B 预取版本触发原生 VGPR 容量守卫：需要 [104,223,234]，4对齐 [104,224,236] 合计564，超过每 SIMD budget504；producer 也有1处 spill。未修改 quota 校验或强制执行，完整诊断在外部 prefetch-compilation.json。
- 4 次早期原型准备错误（多传 WARP_SPECIALIZE、MLS encoding 的不可回读 spelling、两版 SSA 作用域）均保留外部原错误；修正驱动/外部转换器后才进入有效测试，不计作接口失败或环境缺指令。没有失败后软件替代计算。
- 全部原始日志、StatLog、IR、汇编、HSACO、输入输出及 PERF 留在外部目录。Git 仅保留关键实验代码、参数、环境指纹与汇总。本次5个受保护进程的主线程/全部子线程均保持 T，CPU ticks 未增长；5个测量工作区 git porcelain 为空，见外部 final-safety.json。

## 复现与下一步

准备脚本已在仓库外执行并逐文件核验 SHA，未启动 kernel 或队列，见 [preparation-verification.json](preparation-verification.json)。发布的是历史实验代码，不是生产调度策略。以下命令仅准备全新仓库外目录，校验代码与外部 IR SHA，不启动 kernel/队列：

```bash
python docs/gemm/debug_tuning/g171_fp32_scale_20261023/prepare.py \
  --evidence-root /xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023 \
  --output-root /xukang/code/pmd-triton-sessions/g171-fp32-replay
# 如需复测，再显式、串行运行（本轮未启动此 replay）：
python /xukang/code/pmd-triton-sessions/g171-fp32-replay/run_one.py P01-production-w8
python /xukang/code/pmd-triton-sessions/g171-fp32-replay/run_one.py P11-aligned-load-builtinflags
```

下一步最小验证建议：同 scale 的自然4CTA小 shape（256×512×6144）检查生产/候选；第二 FP8 wrapper 对应的 N128/K128 FP32 scale 迁移；129×257×1000 的小 tile普通路径检查尾块。最佳副本只允许 EVEN_MN/EVEN_K，不能用于尾块。真实卡再验证完整 G171、频率与双 die，最后形成 autotune 家族；本轮不做全量尺寸或生产 dispatch 修改。

外部证据根：`/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/`。代码/IR输入SHA与复现文件表：[manifest.json](manifest.json)。

## A/B MLS + FP32 scale buffer load LDS 追加（2026-10-23）

**用户指定路径已实现并执行通过，仍为仓库外 IR 实验副本。** M128/N256/K6144、tile128×256×128、WDRA48/stages2、单die48CU/455ps，以及全部dtype、scale存储和软件版本保持上述条件。生产kernel、默认config、编译器库没有修改。

- A/B 的 IR 各为 `amdg.matrix_load_to_local`；汇编共8条静态 `matrix_load_128x16_b8 … lds`，保留MLS原有同步。
- A/B scale 各为 `amdg.buffer_load_to_local`，汇编各一条 `buffer_load_dword … offen lds`。payload直接进入LDS；地址仍使用VGPR。当前每份copy提交后wait0，再通过原有ready abarrier发布，不声称跨copy充分overlap。
- A scale的producer从MFMA slice映射改为一维blocked（sizePerThread1/64线程/4warps），consumer布局、scale索引与所有浮点运算顺序不变。B原有producer已为blocked。
- 全局A scale仍`[128,48]`、B仍物理紧凑`[2,48]`，K方向连续，没有转置输入。为对齐加载方式，本轮保留原有B在LDS中按256输出通道重复值的暂存，不采用compact-B LDS变体；这不改变全局N128/K128粒度。

| 点 | scale producer / 加载 | cycles（3次） | 抖动 % | 实际VGPR配额 | LDS bytes | spill |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| P31-MLS-scale-IR-control | 原映射/普通加载（同IR重编对照） | 171701.3 | 0.309 | (88, 208, 208) | 101376 | 0 |
| P32r-MLS-scale-layout-control | blocked/普通加载 | 146387.3 | 0.320 | (8, 248, 248) | 101376 | 0 |
| P33-MLS-scale-buffer-LDS | blocked/原生LDS直传 | 149649.3 | 0.206 | (8, 248, 248) | 101376 | 0 |

目标直传路径较原映射对照少 12.84% cycles；**但比同blocked布局的普通scale加载慢 2.23%**。主要收益来自producer所有权调整，不能全部归功于LDS直传。和前述普通A/B加载最佳约149830 cycles的差距不足0.2%，不据此宣称额外性能收益。当前保留用户要求的直传路径用于继续看PERF；没有替换生产dispatch。

| PMD动态统计 | 原映射对照 | blocked普通加载 | blocked原生直传 |
| --- | ---: | ---: | ---: |
| VALU | 37140.000 | 30808.000 | 30516.000 |
| MMOP | 24576.000 | 24576.000 | 24576.000 |
| LDS | 13272.000 | 10392.000 | 10008.000 |
| VMEM | 8384.000 | 2432.000 | 2336.000 |
| SCA | 40880.000 | 40880.000 | 43464.000 |
| totalTCCRequests | 45136.667 | 45655.333 | 45154.000 |
| waitVmCounter | 1653.333 | 1056.444 | 1011.000 |
| waitLgkmCounter | 17026.889 | 12353.000 | 12359.111 |
| hostSeconds | 10.600 | 9.973 | 9.860 |

直传相对blocked普通加载减少LDS/部分VALU，但SCA从40880增至43464，静态ebarrier35→37。每份copy独立commit/wait是当前保守同步方式，需后续配对实验才能判断其具体耗时贡献；本轮不继续扩大调优范围。旧static_instructions字段名`ASYNC_MLS`实际正则统计buffer-load-LDS，不能当作MLS数量；新增`load_path`单独核验两种原生操作。

数值：目标路径seed20和seed21均与CPU golden及对应生产输出逐元素一致，max_abs=0、mismatch=0；无spill，VGPR压力需求(4,169,169)，分配配额(8,248,248)，LDS101376 bytes。数值、三次计时与PERF使用同一HSACO，指纹见 [scale-buffer-LDS-verification.json](scale-buffer-LDS-verification.json)。

PERF独立计时 149716 cycles，预热后文件已核验HYGPERF magic、大小和SHA，见 [scale-buffer-LDS-perf-manifest.json](scale-buffer-LDS-perf-manifest.json)。node60宿主机绝对路径：

```text
/home/hygon/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/PF-MLS-scale-buffer-LDS/m5out/0/1/152939_python3.10.perf
```

只设`use_async_copy=True`的P30未选中scale直传，执行前拦下，不能误报成已启用。已验证的是生成器显式原生IR，不是自动选路修复。布局对照P32第一版遗漏同步buffer_load结果encoding而触发MLIR类型校验，未执行；修正外部类型后P32r通过，原始错误及IR保留在仓库外。没有软件替代实现、指令模拟或环境能力伪装。

复现仍用上面的prepare.py，在全新仓库外目录准备后先跑P01生产基线，再显式逐点选P31、P32r或P33；没有队列调度。原生转换工具 [scale_buffer_lds.py](scale_buffer_lds.py) 和 [转换证明](scale-buffer-LDS-proof.json) 随记录保存；原始IR、日志、统计、HSACO和PERF继续只留仓库外。5个暂停进程保持T且CPU ticks不变，测量Git工作区干净。

## 四组LDS读取集中提前与分批等待（2026-10-23）

**同一K轮集中发射A/B/A scale/B scale，再分两批等待，已在PMD执行通过。** 最佳两次独立进程共6次计时均值 **128298.5 cycles**，比新跑的原顺序对照149772.3少 **14.34%**。单TG M128/N256/K6144、tile128×256×128、WDRA48/stages2、单die48CU/455ps；A/B FP8 E4M3、A FP32/M1/K128 scale、B FP32/N128/K128 scale、BF16输出，版本和有效配置保持前文口径。

### 当前实现

四个ready等待先完成，再发出16条A/B matrix LDS read和10条FP32 scale LDS read。所有empty释放通知延到MMAC之后；scale读完后才通知producer复用buffer。原有ping-pong交接、phase、producer MLS/scale buffer-load-LDS、浮点运算及其顺序均保留。没有新增LDS分配，也没有跨K轮预取第二份完整A/B tile。

```text
ready_wait(A, B, AS, BS)
ds_read_matrix(A) ×8 + ds_read_matrix(B) ×8
ds_read2(AS) ×8 + ds_read2(BS) ×2
s_waitcnt lgkmcnt(10)     # 前16条A/B读取完成，允许最多10条后发scale读取仍在途
原ping-pong同步 / MMAC / 原交接
s_waitcnt lgkmcnt(0)      # scale读取完成
empty_arrive(A, B, AS, BS)
原scale乘法/FMA
```

原生编译器对集中发射版本仍在MMAC前发出lgkmcnt(0)。分批版本是在外部汇编副本上细化该等待，并在MMAC后、任何empty通知或scale使用前补wait0，使用原后端组装为原生HSACO；没有修改生产源码、编译器pass或库。包含最终剥离块在内的3个静态MMAC段均处理。

转换器逐段核验：16条matrix read先于10条scale read；窗口内无额外SMEM/VMEM/写入或分支；scale目标VGPR与全部64条MMAC的输入/输出寄存器不相交；全部非wait原生指令及其顺序、寄存器/LDS声明保持不变。阈值10仅适用于这份实际发射序列，其他tile必须重新推导，不能作为通用常量。转换证明见 [group-read-verification.json](group-read-verification.json)，工具为 [group_lds_reads.py](group_lds_reads.py) 和 [refine_group_wait.py](refine_group_wait.py)。

### 数值与建模周期

| 点 | 调度 | cycles（3次均值） | 抖动 % | VGPR压力P/C0/C1 | spill |
| --- | --- | ---: | ---: | --- | ---: |
| P50-group-read-control | 原顺序，新进程对照 | 149772.3 | 0.236 | (4, 169, 169) | 0 |
| P51-group-read-late-release | 四组提前发射，MMAC前全量等待 | 137286.0 | 0.469 | (4, 189, 189) | 0 |
| P52-group-read-fine-wait | 四组提前发射，分批等待 | 128306.7 | 0.293 | (4, 189, 189) | 0 |
| P53-group-read-fine-repeat | 分批版独立复测 | 128290.3 | 0.525 | (4, 189, 189) | 0 |

集中发射本身比原顺序少8.34% cycles；在完全相同非wait机器指令基础上，分批等待再比全量等待少6.54%。最佳两个进程均值差0.013%。seed20及seed21均与CPU golden和对应生产输出逐元素一致：max_abs=0、mismatch=0。

所有点先编译、执行一次精度/预热，再计3次，JIT不计入cycles；PERF另测不混入均值。VGPR配额保持(8,248,248)，consumer压力169→189，无spill，LDS101376 bytes不变。

| PMD统计 | 原顺序 | 集中发射/全量等待 | 分批等待 |
| --- | ---: | ---: | ---: |
| VALU | 30516.000 | 30148.000 | 30148.000 |
| MMOP | 24576.000 | 24576.000 | 24576.000 |
| LDS | 10008.000 | 10008.000 | 10008.000 |
| VMEM | 2336.000 | 2336.000 | 2336.000 |
| SCA | 43464.000 | 43464.000 | 43464.000 |
| MISC | 1000.000 | 1000.000 | 1000.000 |
| totalTCCRequests | 45154.000 | 45154.000 | 45154.000 |
| waitVmCounter | 1065.333 | 7819.111 | 6875.555 |
| waitLgkmCounter | 12388.667 | 8660.223 | 4253.222 |
| hostSeconds | 9.850 | 9.867 | 9.857 |

集中发射也改变了编译器的地址/寄存器调度，VALU略减；分批版与集中发射版的非wait机器指令完全相同。LGKM等待为跨SIMD的聚合统计，不能直接当作wall time分解或MMAC利用率。

新PERF独立计时 **128684 cycles**，预热后文件已核验HYGPERF magic和SHA，见 [group-read-perf-manifest.json](group-read-perf-manifest.json)。node60宿主机绝对路径：

```text
/home/hygon/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/PF-group-read-fine/m5out/0/1/153844_python3.10.perf
```

本轮6个独立case均完成；原始IR、汇编、HSACO、StatLog、输入输出和PERF仍全部在仓库外。复现：prepare到全新外部目录，先P01建立seed20生产基线，再逐点P50/P51/P52；种子21先P90再P54。P52/P53/P54/PF使用worker_group_read.py/validate中的显式原生等待细化，不能只用TTGIR声称已得到分批等待。生产/default config及编译器库未改，5个暂停进程保持T且CPU ticks不变，5个测量工作区干净。仅验证该对齐单TG及两个随机输入，未跑完整需求尺寸、多CTA、双die或实卡。

## 普通abarrier管理MMAC ping-pong（2026-10-23）

**此前方案已迁移到当前FP32/N128 scale及四组LDS提前读取版本。** 两个独立进程共6次计时均值 **106722.0 cycles**，较新跑的ebarrier分批等待基线128096.3减少 **16.69%**；比仅合并A/B同步的125854.7再少15.20%。源kernel/default config与编译器库均未改，仍为仓库外IR实验。

本轮保持单TG M128/N256/K6144、tile128×256×128、stages2、WDRA48(4,4,4)、单die48CU/455ps；FP8 E4M3矩阵、A FP32/M1/K128及B FP32/N128/K128 scale、BF16输出，全局scale存储/stride、LDS布局/容量和浮点顺序不变。软件commit、实际私有库指纹与编译/运行选项仍按前文environment及逐点effective_options记录。

### 同步协议

- 数据同步：A/B共用0..3两个stage的ready/empty，物理A/B LDS仍分别分配。empty计数8，producer每wave预到达2；ready计数4。producer等待一次empty，一次seq覆盖两组MLS，B传输发出后每wave发布一次ready；consumer两组LDS读取完成后每wave释放一次empty。没有将到达次数翻倍。scale继续使用8..15。
- MMAC turn：新增4/5，均计数4；producer初始每wave向4到达1，授予ping首轮，5初始pending。ping按k&1等待4，发射MMAC簇后向5普通arrive；pong等待5，发射后向4普通arrive。**turn不使用abarrier_seq**；seq只保留在数据传输。普通arrive交接指令簇发射权，不充当硬件MMAC完成计数。
- 移除consumer ebarrier1的两个对称会合点、pong初始偏移以及最后一轮特殊会合条件；保留已有启动/退出join。两组都在scale VALU之前交接turn，因而无需等待另一组VALU结束。两个消费者循环均覆盖48个K-group；turn在现有退出join后失效。
- 使用14个abarrier ID：0..5、8..15；6/7空闲。输入张量的读取形状及次数、LDS分配没有变化；实际L2请求统计见表。

### 编译器原生分批等待

**新turn版本直接使用编译器生成的ASM/HSACO，没有手工修改等待。** 去掉consumer ebarrier1后，后端按MMAC实际寄存器依赖自动发出lgkmcnt(14/13/11/10)，允许后续A/B片段及scale读取在途；MMAC后编译器仍在turn arrive前发出lgkmcnt(0)，确保所有LDS读取完成，再通知复用及执行scale math。该版本不是把旧lgkmcnt10硬套到新汇编。

两个静态MMAC簇各64条（上一版有第三个剥离簇），动态MMOP仍24576，不能把静态指令少一簇解释为少算。初始化ID/count、每个turn wait/arrive的位置和原生等待见 [abarrier-turn-native-proof.json](abarrier-turn-native-proof.json)；转换和资源证明见 [abarrier-turn-verification.json](abarrier-turn-verification.json)，工具为 [abarrier_turn.py](abarrier_turn.py)。

### 精度与周期

| 点 | 方案 | modeled cycles（3次） | 抖动 % | spill |
| --- | --- | ---: | ---: | ---: |
| P60-abarrier-ebarrier-control | 独立A/B同步、ebarrier、已有分批等待 | 128096.3 | 0.330 | 0 |
| P61-abarrier-merged-AB | 仅合并A/B，仍用ebarrier | 125854.7 | 0.160 | 0 |
| P62-abarrier-MMAC-turn | 合并A/B＋普通abarrier turn | 106602.7 | 0.297 | 0 |
| P63-abarrier-MMAC-turn-repeat | turn独立进程复测 | 106841.3 | 0.624 | 0 |

仅合并A/B相对新基线少1.75%。turn两次均值差0.224%；seed20/21均与CPU golden及对应生产输出逐元素一致，max_abs=0、mismatch=0。每个性能点先编译并做精度/预热，再计3次，不含JIT；hostSeconds仅作仿真预算。

| PMD统计 | 新ebarrier基线 | 仅合并AB | 普通turn |
| --- | ---: | ---: | ---: |
| VALU | 30148.000 | 30148.000 | 30144.000 |
| MMOP | 24576.000 | 24576.000 | 24576.000 |
| LDS | 10008.000 | 10008.000 | 10008.000 |
| VMEM | 2336.000 | 2336.000 | 2336.000 |
| SCA | 43464.000 | 39172.000 | 39714.000 |
| MISC | 1000.000 | 984.000 | 996.000 |
| totalTCCRequests | 45154.000 | 45154.000 | 45154.000 |
| waitVmCounter | 6859.333 | 6657.334 | 2184.111 |
| waitLgkmCounter | 4148.889 | 4094.333 | 496.555 |
| hostSeconds | 9.887 | 9.817 | 9.850 |

所有点实际VGPR配额(8,248,248)、压力(4,189,189)、spill0、LDS101376 bytes。turn同时改变了原生等待和最后一轮控制流，收益属于这一完整同步方案，不能全部当成单条barrier的成本。LGKM等聚合统计不是wall time分解；本轮没有单独计算MMAC利用率。

真实PERF独立预热后计时 **106962 cycles**，HYGPERF magic/大小/SHA已核验，见 [abarrier-turn-perf-manifest.json](abarrier-turn-perf-manifest.json)。node60宿主机绝对路径：

```text
/home/hygon/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/PF-abarrier-MMAC-turn/m5out/0/1/154748_python3.10.perf
```

一次SSH客户端连接中断后，只读恢复确认P62原case已PASS、remote returncode0，未重启或重复启动case。全部6项已完成；5个暂停进程及其线程保持T、CPU ticks不变，5个测量工作区干净。原始日志/IR/ASM/HSACO/StatLog/PERF仍全部在仓库外；Git仅留关键代码和汇总。没有新验证队列、完整尺寸、多CTA、双die或实卡测试。

复现：prepare到新的仓库外目录，先P01建立seed20生产基线，再显式逐点P60/P61/P62；种子21先P90再P64。P60/P61仍使用旧细化等待作为对照；P62/P63/P64/PF的fine_group_wait=False，使用原生编译结果。
