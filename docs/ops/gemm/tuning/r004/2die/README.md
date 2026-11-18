# R004 双 DIE TN / TT 调优记录

完整 MNK=6656×64×2048，BF16输入/输出、FP32累加。PMD `sbx2`，2 DIE/96 CU，GC/shader周期455ps；不是实卡测量。沿用R002的MR565 runtime与DTKClang6.3.3。实测repo为 `2ef6411e7c7f729d879e1a739c1d57082362d38d`，默认分发配置未修改。

| 布局 | 测量顺序 | 配对基线 µs | 最优 µs | 耗时降低 | TOPS | A+B读/C写 TB/s |
|---|---|---:|---:|---:|---:|---:|
| TN | 连续warm | 17.630795 | 14.208740 | 19.41% | 122.80 | 1.997 |
| TT | 连续warm | 18.007535 | 14.625520 | 18.78% | 119.30 | 1.940 |
| TN | 交替配置 | 17.416490 | 14.905800 | 14.42% | 117.06 | 1.904 |
| TT | 交替配置 | 17.046575 | 15.063685 | 11.63% | 115.83 | 1.884 |

连续warm最优均为BM128/BN64/BK64、CHUNKS=1、SLOTS=3、1P+4C、寄存器预取，VGPR130、LDS72KiB，无spill。52TG分别由两个DIE各执行26TG，实际活跃52CU。交替配置时BM64更优：TN用CHUNKS=2、双槽、无预取；TT用CHUNKS=1、三槽、预取。两种测量顺序分别记录，缓存状态可能参与差异，尚未做计数器归因。

每布局扫描16候选，分批扫描14个通过；基线和前两名通过两组random及ones校验，随后交替正反顺序各测9次。连续模式另做完整random校验，每配置连续9次，取中位数且不剔除切换后的首个样本。每次检查425984个输出与两侧guard，最终样本mismatch/nonfinite均0，无SGPR/VGPR spill。

TN对照为无spill ring参考配置，TT对照为原TT ring参数在双DIE上的显式复测。TN公开auto fallback的单次筛选warm为75.999560µs，26TG/26CU、VGPR spill=5，该单次值未混入配对比较。

文件按最终方案组织：[TN记录](tn/record.json)、[TT记录](tt/record.json)各保留配置、资源、9次样本、校验、两种测量顺序及工具链/源码哈希；`kernel.amdgcn`为连续warm最优汇编，`stats.txt.gz`为其中位数dispatch的原始统计。实验kernel共用`kernel.py`，复测共用`benchmark.py`。未采集新的single-TG PERF。

复现时先source `/xukang/code/pmd-triton-sessions/mr565-rocdl-20261116/env.sh`，将此目录复制到新的任务目录（输出子目录必须不存在），使用所记录的repo提交和编译器：`python3 benchmark.py TN /path/to/repo`，TT将TN替换为TT。脚本按record.json读取环境与三个入围配置，验证完整随机输入后每配置连续测9次。其计算、校验和计时沿用实测脚本，只合并了输入路径；此次打包调整未新增PMD复测。

扫描过程、交替复测脚本、完整测量结果、HSACO及全部312份stats保留在本地 `/home/kxu/r004-sbx2-tn-tt-20261118-results/evidence.tar.gz`，SHA-256为 `9101cedeaed6ff7439d50c30d78096abee992cae42b56ee4821c02a6983a5756`。远端原始任务为node60容器 `kxu_pmd_8569_fp8_20261017` 内 `/xukang/code/pmd-triton-sessions/r004-sbx2-tn-tt-20261118`。每份record记录其原始源码哈希与来源，不将一次候选通过视为跨所有输入/缓存状态的稳定性证明。
