# BK128 调优与 MLS 确定性降低（round213）

503 个已选配置保持不变，没有新的性能最佳或资源备选晋级。最佳配置仍见相邻 `fp16_single_tg_20261027_round148/best-configs.json`；已独立确认的零物理 spill 备选仍见 round192。本次新增45条已闭环候选记录，未将 CPU 结果或失败实验加入性能地图。

| PMD轮次 | 候选 | 稳态 | 模型点 | dispatch | 仅正确性边界点 |
|---|---:|---:|---:|---:|---:|
| 195 | 5 | 3 | 23 | 92 | 6 |
| 200 | 12 | 6 | 51 | 204 | 13 |
| 202 | 4 | 1 | 19 | 76 | 5 |
| 210 | 24 | 8 | 99 | 396 | 25 |

合计192个独立模型点、768次dispatch，其中143个单TG性能点、49个仅正确性边界点。18个稳态候选、27个非稳态候选；没有候选达到同资源预算最佳值至少3%的初测晋级门槛，因此没有安排独立复测，也没有重跑非稳态项挑选结果。

round195进一步测试256×128×128 TT的UF4、MLS基址重定位及参数降低；scratch/VGPR spill为0，但SGPR仍有4/6个，最快稳态模型峰值比34.209830%。round200/202测试64×256×128 TT/TN的K64/K32分片、单分片预取、MLS双缓冲及合法async/BPP距离2/1。round210增加layout3、4/8warps、TT/TN各自原有scheduler/cluster种子，24项全部零物理spill。

round210最快稳态TT为32.155965%、TN为30.553863%，两者LDS均80KiB；不能将它们与64KiB配置比较。所有候选分别按实际128KiB/64KiB预算筛选，没有性能最佳晋级。完整配置、16/32/64循环中位数、残差、SGPR/VGPR/scratch与来源ID见各轮JSON。

本次发现并修复MLS降低中按指针哈希表迭代顺序发射descriptor groups的问题。round207的TT pilot虽完整精度及工作量通过，CPU/native实际汇编不同，原门禁拒绝记录保留。round208以4次纯CPU AST编译定位第一处差异在ConvertTritonHCUToLLVM。隔离Triton worktree209仅将该descriptor-group返回及局部容器从DenseMap改为MapVector，保留几何插入顺序；坐标查询DenseMap不变。只替换native95的MLS目标文件，其余503个链接输入逐字节不变。补丁与依赖/构建证明均附带。原 `triton-source.patch` 仅含 tracked diff，缺少 13 个新增源码文件；其原始字节保留用于追溯。现补充 `round209/triton-parent209-complete-source.patch`，已在独立 worktree 从相同基线应用，并逐字节验证 19 个改动源码文件。生成的 `_cy_utils` 二进制依赖另记录路径及 SHA256，不纳入源码补丁。该证明验证源码导出完整性；实测 native 仍是替换一个 MLS 对象、复用其余 503 个链接输入的增量构建，未声称完成全量重编译。

新native SHA256：`e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a`。4个独立编译进程（交替Torch预导入）汇编一致，两个既有TT/TN控制的汇编与物理资源精确不变。round210的99个模型点进一步全部通过CPU/native汇编、资源、完整精度及工作量门禁。Triton基线46f0bed3561364b63ba4046d9c8c0cb70380ea30，Bolt基线840d816b671983527e5b1acc831c5e67618fe981，须匹配累计补丁，不能将其称为未修改的上游代码。

每点warmup1次、测量3次，输入seed29/scale1，FP16输入、FP32累加、FP16输出，完整输出atol=rtol=1e-3。以16/32/64循环的周期斜率换算模型峰值比，残差上限2%；逐dispatch核对动态MMOP与逻辑工作量比1，单TG仅CU0。round200/202/210每个候选另做4TG非原点M/N边缘、奇数K/BK=17检查，汇编与单TG完全一致，边界点排除性能排名。这不是BU实卡busy百分比或多TG占用率结论。

BK128已选配置物理资源清查绑定120个选择记录、168份唯一ASM：仅256×128×128 TT 64KiB的原最佳配方有SGPR4（在两个旧选择组出现）；其余118条均为零物理spill，主best组30条BK128全部为零物理spill。旧n_spills字段不能单独排除SGPR spill；round178/192的已确认零物理spill备选继续有效。

CPU负向记录：160KiB LDS超容量8项、原距离1/1非法4项、合法3/1实际shader别名4项。round211针对256×128×128 TT再筛UF2分片32项，其中28项为零物理spill且LDS16/32KiB，另4项SGPR10已排除；这些仅是CPU结果，尚无新的PMD性能资格。round212保留两个控制PASS及UF2候选被worker旧UF1限制拒绝的原记录，候选尚未编译/执行，不列入上述192个模型点。后续在独立目录扩展调用支持并保留全部精度/汇编/工作量门禁。

公共PMD lease保持同一inode179052475/dev66310，自有模型最多8个，native编译用公共compiler.lock串行；128³继续暂缓，无kill/reset。原始结果、ASM、统计与native保留在node60 host `/home/hygon/xukang/code/pmd-triton-sessions/`。闭环原始统计可无损压缩，按stats_io.py原逻辑路径读取，成员SHA保持不变。`evidence-index.json`列出本包每个来源与SHA。
