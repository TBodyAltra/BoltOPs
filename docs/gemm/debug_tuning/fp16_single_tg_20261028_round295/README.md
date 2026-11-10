# 单 TG：128×32×128 TT/TN 独立确认提升

当前地图8969候选行，best-configurations.json含全部503条最佳选择；本批更新best/best_no_spill中的TT、TN四条，另外499条不变。新配方LDS80KiB，不进入64KiB预算排名。natural-stride-map.json.gz为完整地图。

TT初测31.127774%、唯一独立复测28.818011%，原最佳21.793417%；复测提升32.232645%。TN初测26.228388%、唯一独立复测26.352134%，原最佳22.937784%；复测提升14.885267%。TT两次系列有波动，均如实保留，没有追加复测挑选更好值。

两者WASP开启、WDRA关闭，producer/consumer为[2,4]，输入num_warps4、实际总6波；num_stages2、mmac_layout_force3、MMA_K_SPLIT1、UF1、cluster2、maxILP=true、gfx-v2=false，纯MLS、无async/BPP、无matrix_store；FP16输入/FP32累加/FP16输出。精确物理寄存器、默认后端选项及配置见JSON与perf元数据。scratch/VGPR/SGPR spill均为0。

初测32个有效配方、130模型点覆盖单TG16/32/64轮和同shader4TG非原点M/N、K/BK=17边界；独立复测每个配方仅一次16/32/64系列，加2个既有控制，共8点。合计138模型点、552 dispatch，稳态残差<=2%，MMOP/逻辑工作量1、单CU0、完整GOLD seed29/scale1、atol=rtol=1e-3。

六对实际HSACO与实测ASM绑定，所有allocated ELF段（含AMDGPU元数据）及全部非调试段逐字节一致。原始ASM调试路径不同，metadata除cache hash外一致；配置、形状、步长、dtype、输入及精度契约相同。固定Bolt197/Triton209/native209，没有本批GPU源或native修改。

perf包含12个初测/复测斜率点、48份原始dispatch stats的gzip无损副本；解压SHA256在索引中。百分比是有用FLOPs/4096模型峰值比，不是BU实卡忙碌率。known-overcomputed-configurations.json明确另8个TT8consumer配置的2x/4x工作量，不作为有效候选或提升。128³继续暂缓。
