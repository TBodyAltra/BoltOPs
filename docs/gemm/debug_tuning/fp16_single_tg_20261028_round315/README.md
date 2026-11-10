# BK128：大矩形与256×256单TG流水

完整地图9043候选行、503条当前选择。本批新增66候选，更新12条选择，491条不变。

|逻辑BM×BN×BK|布局|LDS KiB|global/dot K|初测利用率|唯一复测利用率|复测相对原最佳提升|结果|
|---|---|---:|---:|---:|---:|---:|---|
|128×256×128|TN|48|32|49.926611%|49.751757%|24.820714%|确认|
|256×128×128|TT|48|32|45.980710%|45.945036%|35.496354%|确认|
|256×128×128|TN|48|32|51.036259%|50.935507%|44.812614%|不提升（保留复测失败）|
|256×256×128|TT|128|64|96.812125%|97.092768%|33.970715%|确认|
|256×256×128|TT|64|32|76.353214%|76.374273%|53.853714%|确认|
|256×256×128|TN|128|64|96.816416%|97.177710%|25.005190%|确认|
|256×256×128|TN|64|32|79.516289%|79.380165%|52.618107%|确认|

百分比为单TG有用FLOPs/4096模型峰值的斜率比，不是BU实卡忙碌率。64KiB与128KiB LDS独立排名。best-configurations.json保留每项完整配置、初测及复测source IDs；available_partition_regs是最终可用配额，backend_required_partition_regs是后端需求，requested_partition_regs是请求，三者分开记录。

逻辑BK128不变；内部WASP_GLOBAL_K_FRAGMENTS=2/4将global load和dot K分为64/32。K为运行时输入、输入域仍K%128=0；逻辑16/32/64轮对应物理32/64/128或64/128/256轮；odd17为物理34/68轮。实际TTIR及transport contract验证见各轮physical-fragment-contract-proof.json。无MNK目标特化或TG排布改变，128³暂缓。

Bolt297隔离worktree的完整kernel patch附于compiler297/bolt_ops-complete-source.patch，基线840d816b671983527e5b1acc831c5e67618fe981；默认参数1的旧控制兼容性证明已附。Triton/native209固定，native SHA e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a，完整编译器源码provenance见前序发布。

293个模型点、1172份dispatch原始stats全部无损gzip附带，含未获提升和稳态失败的负结果。所有新候选fullGOLD seed29/scale1、FP16输入/FP32累加/FP16输出、atol=rtol=1e-3；scratch/VGPR/SGPR spill0、有效MMOP比1。既有控制中的SGPR spill4保留。初测包含每个候选4TG非原点M/N及odd17边界；边界结果只用于精度，排除性能斜率。

每个入围配方恰好一次独立16/32/64复测，禁止复测失败后重跑或候补。初测/复测均需>=3%提升及<=2%稳态残差，全部allocated及非debug ELF段/资源/runtime contract一致；原始ASM仅调试路径重定位差异。

测量编号已包含global fragments及显式WDRA寄存器请求。audit311对310的78个计划引用验证为74个不同契约/74个ID；4个引用是同一pilot契约重用。原schema-only检查未改写，独立实际契约核验另存。此前304碰撞失败记录和已发布修复保持原状。
