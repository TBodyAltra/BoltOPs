# 双 DIE TN 测量

`64×32×128` 和 `64×64×128` tile 入口共享 FP32 scale 预载及 A/B 双缓冲流水。消费者数量由输出 tile 推导；接口和 kernel 不包含需求编号、固定 MNK 或 DIE/CU 分支。输入与 scale recipe 的选择由现有配置系统负责。

## 测量

SBX2 PMD CoreArch1703 / gfx946，2 DIE × 48 CU，GC=455 ps，TN shape 64×2048×4096。E4M3FN 输入、FP32 scale buffer，B scale 为 BF16 值无损提升到 FP32。全部发射覆盖 131,072 个输出；不截断 grid。计时为完整系统 gcSimCycles，排除冷发射和 JIT，不累加两个 DIE 的耗时。

| 方案 | 暖发射平均 μs |
|---|---:|
| 原自动接口（Triton） | 42.078400 |
| 原 Gluon，64×64 tile / 32 TG | 16.606817 |
| 新 Gluon，64×32 tile / 64 TG | 14.564853 |

同工具链、seed37；基线各 2 次暖发射，新版 3 次。新版 73.721 TOPS，比原 Gluon 耗时下降 12.30%；原自动接口的后端选择差异另使整体加速达到 2.889×。两个 DIE 各执行 32 TG，没有归并或额外临时输出。

4 次完整发射均 mismatch=0、nonfinite=0，canary 完好。这里 mismatch=0 指通过 `0.01 + 0.02*abs(reference)`，不是 FP32 golden 零误差；max_abs 见 JSON。BF16 输出与原 Gluon 的同 seed 结果完全一致。另一个 seed 的全量校验保存在 evidence.tar.gz 内的 measurement_seed91.json；共享流水前后两种 tile 的 .text/.rodata 均相同，见 record.json 的 evidence.executable_equivalence。6 项针对性回归通过，覆盖 K=128/256/384、多个 M tile、B scale 边界、输出 canary 及配置匹配。

## 证据与重放

工具链沿用已验证的 Triton f8cbbee18498 + Clang22 c3cca66、ds_read2 与 MMAC C no-sink guard。版本、二进制哈希、完整 PMD 配置、源码与配置提交见 tn/record.json。

- `tn/benchmark.py`、`tn/record.json`：实际运行的公开接口重放脚本。加载 record.json 的 environment_variables，将 PYTHONPATH 中的源码目录改为待测 checkout，再在空输出目录运行 `python /path/to/benchmark.py /path/to/record.json`。
- `tn/evidence.tar.gz` 内的 measurement.json、stats_repeat*.txt.gz：全部发射结果及三次暖发射原始统计。
- `tn/kernel.amdgcn`、`tn/cu0.perf` 及 evidence.tar.gz 内的 perf_measurement.json、perf_stats.txt.gz：匹配的汇编、完整 64 TG 发射中的 CU0 轨迹及统计。PERF 选择最后一次暖发射（14.515865 μs），内嵌 HSACO 已核对一致。

提交仅包含已采用的流水、配置、测试和对应测量产物。候选方案、运行日志和诊断缓存保留在外部实验目录。

The 32/64-column tiles use one configurable `bm64_bk128` kernel; the existing
64-column selector remains an alias. Final measurement JSON and native statistics
are in `tn/evidence.tar.gz`, indexed and hashed by `tn/record.json`; extract the
archive to inspect the original files. Replay takes `tn/record.json` as its argument.
Logical read/write bandwidth including FP32 scales is 0.613 TB/s; this is not a hardware HBM counter.
