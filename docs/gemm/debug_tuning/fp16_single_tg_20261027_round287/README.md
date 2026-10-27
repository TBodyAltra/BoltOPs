# 单 TG 地图：确定性 BlockPingpong 与负结果

完整地图为8937候选行，best-configurations.json保存全部503条当前最佳选择，本批未替换最佳配置。完整地图为natural-stride-map.json.gz。相较本地提交2366d5d的8880行快照，追加fragment-stream24、automatic-layout TN3、vector async/BPP30，共57行。前序已独立确认提升见round244/255/266材料。

BlockPingpong最终化的指针DenseMap会改变A/B静态LDS分配顺序，造成真实TTGIR与指令差异。failed-native277保留GOLD/工作量通过但ISA失败的原始记录，未纳入地图；diagnostic279记录首次差异pass。新worktree改用MapVector按首次使用顺序遍历。私有native仅替换一个HCU BlockPingpong目标文件，其余503个链接输入不变；编译/依赖/链接哈希、完整源补丁在compiler280。这是增量构建，不声称完整重建。

新库通过TT8次、TN4次独立编译，分别覆盖Torch先/后导入；两个旧MLS控制保留原指令与资源。36个候选重编译后均无物理spill，去重30个进入PMD。控制之一既有的SGPR spill4如实保留，不将控制误称为新候选零spill。

122个模型点、488 dispatch通过完整GOLD、CPU/PMD指令与资源匹配、MMOP工作量比1、单TG CU0、同shader16/32/64轮及4TG非原点M/N和K/BK=17边界。29个稳态、1个不稳态；没有同预算下超过当前最佳3%的候选，没有独立性能复测或推广。cohort-best-diagnostics.json仅为本批内部最好值，不是全局最佳，也不是BU实卡忙碌率。

cases包含全部122点的result/spec/executor/ASM。perf包含本批TT/TN内部最佳的6个斜率点、2个边界点和2个控制，共40份raw stats无损gzip；其余448份仍保留于host，路径/哈希见native-raw-audit.json。gzip解压SHA256见evidence-index.json，完整tar成员逐字节核验。128³仍暂缓，矩形非对称WASP和旧spill的新库筛选继续进行。
