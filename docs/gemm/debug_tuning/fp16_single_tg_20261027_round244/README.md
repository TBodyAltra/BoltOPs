# 单 TG：64×64×128 TN 的独立确认提升

将 consumer 固定为 4 waves、producer 从 4 减为 2 waves，WASP 开启、WDRA 关闭，取得两组可复测提升。原最佳理论峰值比 32.162488%。

| LDS 预算 | stages | 初测 | 唯一独立复测 | 初测提升 | 复测提升 |
|---|---|---|---|---|---|
|128KiB|4|34.255129%|34.231273%|6.506467%|6.432292%|
|64KiB|2|33.905414%|33.610503%|5.419127%|4.502188%|

共同参数：BM=BN=64，BK=128，wasp_partition_warps=[2,4]（最终共6 waves），编译输入 num_warps=4，wasp_wdra=false，MMA_K_SPLIT=1（完整BK128），LOOP_UNROLL_FACTOR=1，mmac_layout_force=3，cluster=2，empty_arrive_after_mmac=true，max-ILP开启、sched-model-v2关闭，async copy/BPP关闭，GROUP_SIZE_M=1、SIMPLIFY_GROUP1=false。精确参数见 JSON。两组 scratch/VGPR/SGPR spill 均为0。

两组在16/32/64循环中保持同一程序、稳态残差不超过2%、动态MMOP/逻辑工作量=1、单TG仅CU0。初测另有同一shader的4TG非原点、M/N边缘及K/BK=17检查。每点warmup1次、测量3次，seed29/scale1，FP16输入、FP32累加、FP16输出，完整GOLD atol=rtol=1e-3。复测进程和measurement context独立，每个配方仅一次复测。

跨worktree原始ASM的调试路径不同。六对实际HSACO逐一与实测ASM绑定：所有allocated ELF段（含AMDGPU metadata、text、rodata）及全部非调试段逐字节一致。metadata除cache hash外一致，运行形状、步长、dtype、精度标准及配置相同。源码冻结Bolt197/Triton209，native209 SHA256为 e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a；完整源码补丁及增量构建provenance见前一批round213的round226补充。

地图四条选择更新：TN的best、best_no_spill使用128KiB配方，best_64k、best_64k_no_spill使用64KiB配方；其余499条保持原值。地图累计8856候选行，best-configurations.json收录当前全部503条选择。

本批还保留round233的24个64方形候选、round235的16个32方形候选，均稳态但未晋级；连同round241的9个候选及round243复测，共210模型点、840 dispatch，其中49个仅正确性边界点。本批所有新候选严格MMOP=1。历史32方形TT最佳的MMOP=2、TN=1，单独清单保留。既有256×128×128 TT 64KiB旧选择的SGPR spill4及round192零spill备选沿用原记录，本次未改写。

best-64x64x128-TN.json汇总两个配方、初测/复测、每点周期及证据位置。perf/包含12个相关模型点的配置、result、ASM、executor，以及48份原始dispatch stats的gzip无损副本；解压后逐字节等于原始PMD输出。索引提供压缩文件和原始文本SHA256及host路径。百分比是有用FLOPs/模型理论峰值比，不是BU实卡busy百分比或多TG占用率。128³继续暂缓。
