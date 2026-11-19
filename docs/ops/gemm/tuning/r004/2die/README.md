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

文件按最终方案组织：[TN记录](tn/record.json)、[TT记录](tt/record.json)各保留配置、资源、9次样本、校验、两种测量顺序及工具链/源码哈希；`kernel.amdgcn`为连续warm最优汇编。实验kernel共用`kernel.py`，复测共用`benchmark.py`。[TN PERF](tn/single_tg.perf) / [TT PERF](tt/single_tg.perf)分别对应连续warm最优方案，内嵌HSACO与全量实测逐字节一致。

复现时先source `/xukang/code/pmd-triton-sessions/mr565-rocdl-20261116/env.sh`，将此目录复制到新的任务目录（输出子目录必须不存在），使用所记录的repo提交和编译器：`python3 benchmark.py TN /path/to/repo`，TT将TN替换为TT。脚本按record.json读取环境与三个入围配置，验证完整随机输入后每配置连续测9次。其计算、校验和计时沿用实测脚本，只合并了输入路径；此次打包调整未新增PMD复测。

扫描过程、交替复测脚本、完整测量结果、HSACO及全部312份stats保留在本地 `/home/kxu/r004-sbx2-tn-tt-20261118-results/evidence.tar.gz`，SHA-256为 `9101cedeaed6ff7439d50c30d78096abee992cae42b56ee4821c02a6983a5756`。远端原始任务为node60容器 `kxu_pmd_8569_fp8_20261017` 内 `/xukang/code/pmd-triton-sessions/r004-sbx2-tn-tt-20261118`。每份record记录其原始源码哈希与来源，不将一次候选通过视为跨所有输入/缓存状态的稳定性证明。

PERF采集沿用R010的三次单TG方式，完整MNK与输入输出strides不变，仅启动1TG，覆盖首个128×64 tile并计算完整K；每次8192个输出精确匹配，另检查417792个未覆盖输出保持NaN及两侧guard。排除第0次，在第1/2次中选择cycles较小的一次；PMD会轮换调度DIE，所选trace的DIE分配记录在record.json中。单TG耗时独立于表内全量耗时，不用于替换全量结果。采集命令：`python3 benchmark.py TN /path/to/repo --single-tg-perf`（TT将TN替换为TT），使用新任务目录；更换源码绝对路径时允许debug信息不同，但必须匹配所记录的`.text`/`.rodata`。

本次三次采集的完整本地证据为 `/home/kxu/r004-sbx2-tn-tt-20261118-results/perf-evidence.tar.gz`，SHA-256：`6b18efad99f98aa789450604e5f670c193cff89f5a85231851c2bc89a855d190`。采集日志与中间文件均留在本地，MR只保留两份选定PERF及现有record.json中的校验摘要。

## 性能回归门槛

TN/TT 两条记录已显式注册到 BF16 回归和 Qwen3.5 FP8 模型回归套件。回放保持原始三个配置的顺序与相同 A/B/C 缓冲区，每配置连续测 9 次，取包含首个样本的中位数。门槛固定绑定已发布的 `m128n64k64_ch1s3p1c4f1`，不随复测中的最快候选切换。此注册不改变公开接口的默认分发配置。

| 布局 | 基线（µs） | 默认上限：基线 × 1.01（µs） |
|---|---:|---:|
| TN | 14.208740 | 14.3508274 |
| TT | 14.625520 | 14.7717752 |

超过上限判定退化，恰好增加 1% 通过。统计验证覆盖两个 DIE / 96 CU、完整输出和 TG 数，拒绝缺失采样、改变配置、spill、数值错误及重复或乱序的统计文件。新输出目录中运行：

```bash
python scripts/performance/gemm_a16w16_pmd_regression.py --dtype bf16 --case R004/2die/tn --case R004/2die/tt --output /tmp/r004-bf16-regression
```

两条基线数值、kernel 和原始 benchmark 保持不变；每次回归在输出目录内创建独立回放副本，支持 `--evaluate` 评估已保存的结果。
