# Qwen3.5-35B-A3B FP8 双 DIE 测量

测量使用 node60 上的 PMD sbx2、gfx946、2×48 CU，GC 周期 455 ps。每个布局的 `record.json` 保留实际配置、工具链指纹、输出校验、TG 分布、采样及对应汇编的 SHA-256。

- BF16 用例 R001/R003 TT 采用公开入口的默认配置，取独立复测的 warm 样本。R002 TN 记录公开入口的单次全网格冷发射。
- FP8 decode 取四个 warm 样本的中位数。FP8 prefill 记录单次冷发射。FP8 文件记录实际指定的候选配置。persistent TT 候选使用 [预取边界修复和布局约束](gemm_a8w8_blockscale_persistent.patch)。[小 M 实验补丁](gemm_a8w8_blockscale.patch)用于复现完整的 FP8 测量源码快照。
- 记录中的有效带宽按 A/B 各读取一次、C 写出一次计算。FP8 scale 流量未计入。
- 性能来自完整网格统计。单 TG PERF 仍待采集。

在匹配该条记录的 node60 PMD 环境中执行

```sh
python3 docs/ops/gemm/tuning/qwen35_fp8_2die/benchmark.py \
    docs/ops/gemm/tuning/r005/2die/tt/record.json --output /tmp/qwen35-r005-tt
```

回放会将记录固定的提交和测量补丁放到输出目录中，并运行完整网格、输入布局和输出校验。用 `--dry-run` 检查源提交和参数，也可用 `--repo` 指向包含源提交的 BoltOPs 仓库。
