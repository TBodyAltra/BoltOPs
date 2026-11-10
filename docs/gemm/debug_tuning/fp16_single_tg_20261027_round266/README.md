# 单 TG：32×128×128 TN 的独立确认提升

初测27.234043%，唯一独立复测26.760747%；原最佳25.486902%，分别提升6.855053%和4.998040%。LDS为80KiB，WASP开启、WDRA关闭。原TN已是WASP；实际原/新配方见comparison.json，不把输入参数补齐误称为算法改动。

两条128KiB预算内的best/best_no_spill选择更新，其余501条保持原值。80KiB配方不参与64KiB比较。当前地图8880候选行，best-configurations.json收录全部503条当前选择。

初测14个模型点覆盖3个显式layout3配方、16/32/64循环和同一shader的4TG非原点M/N边界及K/BK=17检查。复测5点含两个既有控制及胜出配方唯一一次独立16/32/64系列，共19点、76 dispatch。新候选稳态残差<=2%、MMOP/逻辑工作量=1，scratch/VGPR/SGPR spill0；完整GOLD seed29/scale1、FP16输入/FP32累加/FP16输出、atol=rtol=1e-3。

三对实际HSACO与实测ASM绑定，所有allocated ELF段（含AMDGPU metadata）和全部非调试段逐字节一致。原始ASM仅调试路径不同。metadata除cache hash外一致；配置、形状、步长、dtype、输入及精度契约相同。固定Bolt197/Triton209/native209，无新GPU源码或native修改。每个配方只有一次独立复测，没有重跑挑选结果。

perf/包含6个相关模型点以及24份原始dispatch stats的gzip无损副本，解压文本SHA256在索引中。百分比是有用FLOPs/4096模型峰值比，不是BU实卡busy或多TG占用率。其他三组确认提升及完整源码provenance见前序round244/255和round213/226补充。128³继续暂缓。
