# BK128：SGPR 生命周期与分配策略筛选（round183）

本轮没有性能最佳值晋级。503 个已选配置和上一轮经过独立复测的无物理 spill 备选均保持不变。最佳配置继续使用相邻 `fp16_single_tg_20261027_round148/best-configs.json`，无 spill 备选见 `fp16_single_tg_20261027_round178/resource-recipe.json`。

在隔离的 Triton worktree180 中，只给 Python 后端增加默认值为 0 的实验参数 `hcu_sgpr_liveness_strategy`，将 LLVM 的参数加载、指令下沉、SGPR 分配设置显式传给现有 clang18。沿用 native95，没有重建 C++ native，也没有扩大 SGPR、VGPR 或 LDS 上限。kernel 源代码继续使用已冻结的 worktree175；generic 与 GROUP1 两种索引保持相同的合法 TG 排布，没有固定运行时 M/N。

| 策略 | LLVM 设置 | 本轮结果 |
|---|---|---|
| 0 | 原默认配置 | 两个旧配置 CPU ISA/资源及 PMD 完整输出对照通过 |
| 1 | amdgpu-ir-lower-kernel-arguments=false | generic 的 SGPR spill 从4降到2；GROUP1仍为4 |
| 2 | spill-aware sinking，扩大 cycle/load 查询范围 | 两种索引均与已有 ISA/资源重复，没有再跑 PMD |
| 3 | sgpr-regalloc=basic | 两种索引均为18个 SGPR spill，精度通过 |
| 4 | enable-subreg-liveness=true | 两种索引均与已有 ISA/资源重复，没有再跑 PMD |
| 5 | sgpr-regalloc=fast | SGPR spill 为140/141；两种索引各有1414个输出不匹配，均拒绝 |

所有实验均为 **256×128×128 TT、NoWASP/NoWDRA、8 warps、stage1、UF4、MMA_K_SPLIT4、EPTrue、Bfirst、fragment MLS、LDS64 KiB**。参数加载与分配策略仅在该配置范围得到本轮结论，不能推广为其他 tile 或 WASP 拓扑的资格。

| 策略 | 索引 | SGPR spill | 利用率% | 中间点残差% | 结论 |
|---|---|---:|---:|---:|---|
| 1 | generic | 2 | 33.850773 | 5.868612 | 未稳态，不晋级 |
| 1 | GROUP1 | 4 | 33.954600 | 1.071488 | 稳态；相对64 KiB最佳值仅提升约0.135%，不晋级 |
| 3 | generic | 18 | 33.786087 | 5.835029 | 未稳态，不晋级 |
| 3 | GROUP1 | 18 | 33.902607 | 0.946093 | 稳态但更慢，不晋级 |

四个完成标定的配置均为 scratch/VGPR spill=0，但保留真实 SGPR spill，因此没有新增零物理 spill 资源备选。没有独立复测候选；未重跑未稳态配置来选择更好的结果。原有性能晋级门槛仍为初测与独立复测均提速至少3%，并满足16/32/64循环稳态与完整汇编一致。

CPU阶段为2个精确旧对照、10个新参数组合；4个组合因实际ISA/资源重复而去除。PMD短测共有8个模型点（2对照、6候选），其中6点通过、2个fast策略精度失败。失败没有非有限值，原输入、输出、gold与precision-failure.pt快照保留；没有放宽容差或改写参考答案。失败策略没有进入长循环。

后续阶段直接复用6个原始通过短测点，只新增4×64和4×32循环测试。接续前核对原worker、scanner、GPU源码、native、环境及测量上下文逐项相同；这是一轮初测的接续，不是独立复测。两份审计分别覆盖8点/32次dispatch与14点/56次dispatch，其中6点/24次dispatch重叠，**去重后是16个模型点、64次原始dispatch**；不能把两份审计直接相加成88次。去重后14点通过，2点为原始精度失败。

所有原始dispatch均检查MMOP工作量比1、仅CU0活跃，完整输入seed29、scale1，FP16输入/FP32累加/FP16输出，完整输出atol=rtol=1e-3。kernel编译通过公共compiler.lock串行，并以单调时钟区间验证；模型最多8个，接续阶段最多4个，沿用同一个PMD lease。原失败门禁仍然失败，所有模型自然退出。128³继续暂缓。

利用率是单TG周期斜率折算的模型峰值比 `2*BM*BN*BK/(cycles_per_BK*4096)`，不是硬件busy counter。没有BU实卡、多TG或占用率结论。native SHA256为3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990，源代码基线Triton46f0bed3561364b63ba4046d9c8c0cb70380ea30/Bolt840d816b671983527e5b1acc831c5e67618fe981；需匹配所附累计源码补丁。默认0保留旧编译参数；None沿用既有选项语义，视为未设置并解析为0。

`candidate-delta.json`保存4条新增候选及地图SHA；`precision-rejections.json`保存2条原始失败记录、快照哈希与路径。源码、增量/累计补丁、类型检查、审计及接续证明按原字节导出并绑定于`evidence-index.json`。CPU选项复核最初误将None视为非法值，修正为既有默认选项语义；原复核脚本及失败记录一并保留，此修正未改变测量源码或重跑CPU编译。

大体积stats、失败张量和native二进制留在node60 host的 `/home/hygon/xukang/code/pmd-triton-sessions/` 对应round180/181/182目录；本包包含报告、源码及哈希。原最佳配置和round178无spill备选仍有效，fast策略没有任何性能资格。
