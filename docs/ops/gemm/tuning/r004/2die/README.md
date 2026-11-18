# R004 双 die TN / TT autotune

2026-11-18，完整 MNK=6656×64×2048，BF16输入/输出、FP32累加。PMD sbx2 / 2 die / 96 CU，455ps；不是实卡测量。沿用R002的MR565 runtime与DTKClang6.3.3，哈希已核对。隔离worktree起始提交2ef6411e，未替换公开接口默认配置。

每布局扫描16个候选，每个执行完整输出与guard校验、一次random验证和一次warm。基线与前两名进一步执行两组random加ones验证，每组重新NaN初始化输出，之后用相同A/B/C地址交替正反顺序各9次warm。观察到测量顺序会显著影响BM128方案，再独立执行同缓冲区、每配置连续9次dispatch，取中位数。连续模式另有完整random验证，全部最终配置每次检查425984个输出，mismatch/nonfinite均0，无SGPR/VGPR spill。不同进程同配置的HSACO哈希一致。

两种顺序分别报告，不能用连续warm的优势推断交替执行不同配置时的表现。配置切换后的缓存状态可能参与差异，这是推断，尚未做cache计数器归因。连续模式首个样本可能包含切换后的冷态，仍计入9次中位数。

| 布局 | 测量顺序 | 配对基线 µs | 最优 µs | 耗时降低 | TOPS | 有效带宽 TB/s |
|---|---|---:|---:|---:|---:|---:|
| TN | 连续warm | 17.630795 | 14.208740 | 19.41% | 122.80 | 1.997 |
| TN | 交替配置 | 17.416490 | 14.905800 | 14.42% | 117.06 | 1.904 |
| TT | 连续warm | 18.007535 | 14.625520 | 18.78% | 119.30 | 1.940 |
| TT | 交替配置 | 17.046575 | 15.063685 | 11.63% | 115.83 | 1.884 |

TN 连续warm最优：BM128 BN64 BK64，CHUNKS=1，SLOTS=3，1P+4C，PREFETCH=True。SGPR 51、VGPR 130、LDS 72 KiB、5 waves/TG，无spill。共52TG，实际活跃52CU（die0: 26TG/26CU、die1: 26TG/26CU）。

TN 交替模式最优：BM64 BN64 BK64，CHUNKS=2，SLOTS=2，1P+4C，PREFETCH=False。初筛14/16通过。配置、每次样本、失败候选和额外顺序诊断详见[TN完整记录](tn/record.json)。单次初筛通过不等于跨所有输入和缓存状态稳定。

TT 连续warm最优：BM128 BN64 BK64，CHUNKS=1，SLOTS=3，1P+4C，PREFETCH=True。SGPR 54、VGPR 130、LDS 72 KiB、5 waves/TG，无spill。共52TG，实际活跃52CU（die0: 26TG/26CU、die1: 26TG/26CU）。

TT 交替模式最优：BM64 BN64 BK64，CHUNKS=1，SLOTS=3，1P+4C，PREFETCH=True。初筛14/16通过。配置、每次样本、失败候选和额外顺序诊断详见[TT完整记录](tt/record.json)。单次初筛通过不等于跨所有输入和缓存状态稳定。

TN配对基线是无spill的ring参考配置；当前公开auto fallback另有独立初筛warm 75.999560µs、26TG/26CU、VGPR256且spill5，不作为配对9次中位数。TT基线是生产TT原配置在双die上的复测，不混入单die25.32712µs。

带宽包含A+B读、C写：2×(M×K+K×N+M×N)/耗时。MMAC利用率按全96CU、455ps下864.210989TOPS峰值计算；即使方案只使用52CU，分母仍为全设备峰值。具体利用率及每次样本在record.json中。

原始统计、全部候选汇编/二进制、模拟器日志、输入布局、编译器哈希和完整脚本保存在本地 `/home/kxu/r004-sbx2-tn-tt-20261118-results/evidence.tar.gz`。远端为node60容器`kxu_pmd_8569_fp8_20261017`的`/xukang/code/pmd-triton-sessions/r004-sbx2-tn-tt-20261118`。

复现时在远端source MR565 env.sh，使用所记录的repo提交和编译器，将脚本和JSON放入包含repo子目录的全新任务根（输出目录不得已存在），运行`python3 standalone_worker.py TN`或`TT`测连续模式，`python3 confirm_worker.py TN`或`TT`测交替模式。候选是独立实验kernel，直接调用；公开API自动dispatch尚未接入。

MR 随附 `tn/continuous_measurements.json` 与 `tt/continuous_measurements.json`，保留连续模式三个入围配置的全部校验和9次样本；交替模式的全部样本已在各自 `record.json` 中。`tn/stats/` 与 `tt/stats/` 保存每种模式基线及最优项的中位数 dispatch 原始 stats，以及对应 PMD config.ini（gzip压缩）。`evidence_manifest.json` 给出压缩文件与解压内容的 SHA-256，并标明原始来源。HSACO 不纳入 Git，二进制哈希留在记录中；完整二进制及全部312份 stats 位于上述原始证据包。
