# 单 TG：32×128×128 TT 的独立确认提升

WASP [4,4]、WDRA关闭、2 stages，初测28.608013%，唯一独立复测27.682534%；原最佳22.702163%，分别提升26.014481%和21.937867%。LDS从原120KiB降至80KiB。四波producer与四波consumer，共8 waves；编译输入num_warps=4。两条128KiB预算内的best/best_no_spill选择更新，其余501条选择不变。80KiB方案不参与64KiB预算比较。

共同配置：BM=32、BN=128、BK=128，MMA_K_SPLIT=1、LOOP_UNROLL_FACTOR=1，mmac_layout_force=3、cluster=2、empty_arrive_after_mmac=false，max-ILP开启、sched-model-v2关闭，async copy/BPP关闭，GROUP_SIZE_M=1、SIMPLIFY_GROUP1=false，USE_MATRIX_STORE=false。精确配置见JSON。amdhsa_next_free_vgpr=97，scratch/VGPR/SGPR spill均0。

初测26个模型点覆盖6个候选、16/32/64循环以及同一shader的4TG非原点M/N边界和K/BK=17检查；复测5点覆盖两个既有控制和胜出配方唯一一次16/32/64独立系列，共31点、124 dispatch。所有新候选稳态残差<=2%、MMOP/逻辑工作量=1、完整GOLD seed29/scale1、FP16输入/FP32累加/FP16输出、atol=rtol=1e-3。没有重跑非稳态或复测失败项。

三对实际HSACO与实测ASM逐一绑定，所有allocated段、AMDGPU metadata以及全部非调试ELF段逐字节相同；原始ASM仅调试路径不同。metadata除cache hash外一致，配置、形状、步长和数据精度契约完全匹配。固定Bolt197/Triton209/native209，无新GPU源码或native修改。完整累计源码及构建provenance见round213/round226补充，前一批64方形TN的两组确认提升见round244。

best-configurations.json收录当前503条选择，地图累计8862个候选行。best-32x128x128-TT.json含初测、复测及每点周期；perf/提供6个相关模型点、24份原始dispatch stats的gzip无损副本，解压文本SHA256在索引中。百分比为16/32/64周期斜率得到的有用FLOPs/4096模型峰值比，不是实卡busy或多TG占用率结论。历史例外沿用前一批说明。128³继续暂缓。
