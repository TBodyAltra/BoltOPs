# GEMM 支持与优化统一入口

更新：2026-10-20。后续支持修复、精度 smoke、性能选择规则及调优配置统一在
[`perf/gemm-optimization`](http://42.228.13.241:10068/dcutoolkit/deeplearing/BoltOPs/-/tree/perf/gemm-optimization)
推进。当前只汇总已有改动和实测，不修改默认配置，不宣称完成全尺寸性能验收。

## 分支来源与历史 MR

| 来源 | 归并方式 | 内容 |
| --- | --- | --- |
| main `fbbe74e` | 基线 | 包含已合并的 GEMM WASP/WDRA 支持与调优工作（[!115](http://42.228.13.241:10068/dcutoolkit/deeplearing/BoltOPs/-/merge_requests/115)） |
| `feat/fp8-fp4-grouped-gemm` `1376197` | 完整继承，作为统一分支起点 | [!122](http://42.228.13.241:10068/dcutoolkit/deeplearing/BoltOPs/-/merge_requests/122) 的 E8M0 解码、FP8×FP4 dense/grouped 接口及回归；远端已将原 `789aa19` 系列重放到最新 main |
| `fix/fp8-dense-scale-smoke` `0cc243e` | cherry-pick | 前两个 FP8 接口的整行 A scale 支持；原 E8M0 分支 `70bfbe9` 的代码已经包含在上行，未重复应用 |
| `shb_bu_wasp` `4fd6274` | cherry-pick | [!121](http://42.228.13.241:10068/dcutoolkit/deeplearing/BoltOPs/-/merge_requests/121) 的需求表模型列收窄与配置链接；该 MR 此前已关闭 |

旧支持 MR 只保留历史讨论，不再作为开发或集成入口。本次不删除旧分支，不自动合并到 main，
也不创建替代 MR；后续代码统一推送本分支。旧实验工作区保留，以免影响暂停进程和历史重现。

## 验收范围与当前 FP8 状态

需求文件为 [REQUIREMENTS.md](REQUIREMENTS.md)。dense 性能评测统一 E=1；
表 7–9 的 grouped/MoE 不做专家维度调优。各对外接口保留小尺寸正确性 smoke。

| FP8×FP8 接口 | 最终 smoke | 支持范围与后续工作 |
| --- | --- | --- |
| `gemm_a8w8_blockscale` | 15/15 PASS | A 按 K128 分组或整行一个 scale，W 按每 N / K128 分组；继续性能调优 |
| `w8a8_triton_block_scaled_mm` | 17/17 PASS | 同上；另覆盖 A 前导维展开和显式 block_size；继续性能调优 |
| `triton_scaled_mm` | 14/14 PASS | 保留 scalar/per-row/per-column 的 K 不变 scale 语义；只做兼容性 smoke，不能替代 W 按 K 分组的需求 |

共 46/46 PASS，逐元素数值一致，max_abs=0、mismatch=0。覆盖 FP32/E8M0 scale
独立组合、FP16/BF16/FP32 输出、K 尾块、历史 block_n=128、随机正态输入及默认配置。
这是小尺寸结果，不等于完整需求矩阵通过。原始 FP8 输入在 CPU 生成后传输，
没有替换 GPU 量化 helper，没有软件实现替代 PMD 不支持的指令。

## 已完成的性能实验

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

### FP8×FP8 流水首轮探针（2026-10-20）

当前改为先修通少量流水候选，再上卡autotune。已完成27个单CTA精度/性能点；
小tile async有可复现候选，BK64 BPP在aiter副本中实际生效，但尚非跨BK最快。
首轮scaled-loop的分区依赖断言已在Triton同名分支e33b79cf1修复，
统一GEMM环境采用已验证的私有编译库。34项runtime检查通过，同BK普通流水配对4/4完全一致；
较大tile仍有LDS/VGPR限制，详见[修复记录](WDRA48_FIX_20261020.md)。
首轮基线数据保留在[流水候选与阻塞](FP8_PIPELINE_20261020.md)。生产kernel和默认配置未改。

## 环境、证据与存储边界

实测环境：node60，容器 `kxu_pmd_8569_fp8_20261017`，PMD gfx946 / 48 CU，
arch_info LDS 65536 bytes，Triton release/3.6.x
`46f0bed3561364b63ba4046d9c8c0cb70380ea30`。
libtriton SHA256：`169f9640add95d07a540ad1b20275bbec7fc66ffe5d85f0fbcd83a1efc5f033e`。
现有 S03 LEGACY 编译器补丁保留在仓库外的 Triton 工作树；不混入 BoltOps 源码提交。

全部路径位于容器 `/xukang/code/pmd-triton-sessions/` 下：

| 证据 | 仓库外路径 |
| --- | --- |
| FP8 三接口 smoke、12 点基线、环境与安全快照 | `fp8-dense-smoke-20261020/REPORT.md`、`final-results.json/csv`、`performance-baseline.csv`、`safety-final.json` |
| FP8 基线 IR / 汇编 | `fp8-dense-smoke-20261020/perf-aiter/ir/`、`perf-vllm/ir/` |
| Track A stages 数据及报告 | `perf-trackA-stages-20261019/REPORT.md`、`measurements.json/csv`、`equivalence.json` |
| Track A 原参数化副本 | `perf-phase0-20261018/param_kernel.py` |
| Track B 校准与成本表 | `perf-trackB-20261018/REPORT.md`、`REPORT-calibration-20261018.md`、`cost-model.csv`、`cost-model-metrics.csv` |

Git 仅保存源码、已有测试和本精简记录。原始日志、StatLog、stats、输入/输出张量、
临时脚本、IR、汇编、HSACO 和大型快照均留在仓库外，不因分支汇总而导入。
PMD 不支持的指令必须如实标记环境能力缺失，不能软件绕过后计 PASS。

## 后续顺序

先修通关键流水，再上真实卡做autotune，不在PMD展开207个shape的全量配置搜索。
大tile优先WDRA48；较小tile保留普通流水、WDRA44/WASP44及async候选，BPP仅在匹配器
允许的tile/BK域内尝试。post-dot scale的跨分区stage依赖断言已修复；接下来分析合法WDRA48配置的流水成本，
再优化scale中转/布局及较大tile的容量和VGPR需求，不删资源校验或绕过scale。

目前生产aiter内核仍要求GROUP_K==BK。仓库外副本的BK64使用正确的K128 scale索引，
已完成有限测试，尚未晋升公开接口或默认配置。A FP32/K128、E8M0/K128与整行scale
分别验收。第三个FP8接口维持兼容性smoke，Track A/B结果保留作参考。
流水补齐尾块和跨CTA验收后再形成上卡autotune候选，任何默认配置替换另需数值和性能证据。

五个暂停进程 61399/63997/31138/54355/31159 必须继续保持 T；不启动旧验证队列，
不恢复完整需求尺寸测试。2026-10-17 文档中的“队列运行中”属于历史状态，现已暂停。
