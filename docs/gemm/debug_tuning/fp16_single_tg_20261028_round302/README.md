# 单 TG：逻辑BK128的global K分片流水

完整地图8977候选行，best-configurations.json含503条当前选择。本批7个配方独立确认，更新best/best_no_spill的14条，489条不变；所有新配方使用72/80KiB LDS，不进入64KiB排名。

|BM×BN×BK|布局|初测利用率|唯一独立复测利用率|复测相对原最佳提升|
|---|---|---:|---:|---:|
|32×256×128|TT|28.258995%|28.736801%|32.360473%|
|32×256×128|TN|30.836407%|30.842599%|24.615346%|
|256×32×128|TT|28.636681%|28.766738%|36.309580%|
|256×32×128|TN|27.764472%|27.792102%|57.954494%|
|64×256×128|TT|45.715986%|43.950463%|32.298475%|
|64×256×128|TN|48.008908%|47.102567%|44.232446%|
|256×64×128|TT|46.234162%|46.375061%|48.441333%|

所有新配方：WASP开启、WDRA关闭，producer/consumer [2,4]，num_warps4、总6波，num_stages2，新增kernel constexpr WASP_GLOBAL_K_FRAGMENTS=2；逻辑BLOCK_SIZE_K=128，实际global load/dot K=64，MMA_K_SPLIT=1，STREAM_K_FRAGMENTS=1，UF1，纯MLS，无async/BPP及matrix_store。

复现时在Bolt基线840d816b671983527e5b1acc831c5e67618fe981的隔离worktree应用compiler297/bolt_ops-complete-source.patch。另附对已冻结Bolt197的增量patch和完整gemm_fp16.py。WASP_GLOBAL_K_FRAGMENTS是kernel参数；默认1时两组既有控制的指令、物理资源和除cache hash外metadata不变。Triton/native仍固定209，源码/native provenance见前序材料。

K仍为运行时输入，沿用K%逻辑BK128=0输入域；M/N仍为运行时输入。逻辑16/32/64轮分别执行物理32/64/128轮，逻辑odd17对应物理34轮。实际TTIR dot形状及运行时transport contract在physical-fragment-contract-proof.json验证。无MNK目标特化或TG排布变更。

初测8候选、34模型点，独立复测7配方各一次16/32/64系列及2控制，共23点。总57模型/228 dispatch，全部GOLD seed29/scale1、FP16输入/FP32累加/FP16输出，atol=rtol=1e-3。有用MMOP工作比1、CU0，稳态残差<=2%，同shader4TG非原点M/N及逻辑odd17边界。新候选scratch/VGPR/SGPR spill0；旧控制之一的SGPR spill4如实保留。

21对实际HSACO与实测ASM绑定，全部allocated ELF段（含AMDGPU metadata）及非调试段逐字节一致，原始ASM仅调试路径差异。metadata除cache hash外一致，配置、形状、步长、dtype、输入和精度契约相同。每个候选只有一次独立复测，没有重跑挑选结果。

perf收录42个初测/复测斜率点和168份raw dispatch stats的无损gzip，解压SHA256见evidence-index.json。百分比为有用FLOPs/4096模型峰值比，不是BU实卡忙碌率。128³继续暂缓；physicalK32及更大tile仍在后续搜索中。
