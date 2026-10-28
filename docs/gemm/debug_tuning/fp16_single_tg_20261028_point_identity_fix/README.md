# Benchmark point identity:global K fragments

48候选混合WASP_GLOBAL_K_FRAGMENTS=2/4时，旧SPEC_KEYS遗漏wasp_global_k_fragments。16个短程K=32项复用了K=64记录，一致性门槛停止round304，未导入地图。它未完成长程斜率或任何性能推广。原始失败记录保留。

新scan.py将该参数纳入point key，并将省略值归一为默认1。对真实记录的完整194契约，旧函数仅生成130个ID，新函数生成194个不同ID；省略/显式1相等。verify_point_identity.py可离线运行，仅依赖Python，不分配或执行GPU。

已推广的round298/300只使用固定fragment2（控制为默认1）。34个初测点和23个独立复测点的原ID逐条核验、无碰撞，配置与实测片数匹配；此前GOLD、工作量、资源和21对ELF证明保持有效。旧快照与旧caseID不修改。

后续混合2/4搜索使用这里的scan.py或补丁；完整新measurement context为round307。当前最佳JSON和kernel/perf证据见同分支fp16_single_tg_20261028_round302。此修复仅影响benchmark缓存索引，kernel和GPU源不变。
