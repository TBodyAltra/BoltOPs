# R001 / 1 DIE / TN

BF16×BF16→BF16，M×N×K = 64×64×2048；FP32 累加，无 scale。通过公共 `gemm_a16w16(A, W, out=C)` 的 auto 分派执行原生 Gluon kernel，A/W 均为连续的 `[64,2048]`。

BM×BN×BK=16×16×128，展开4，1 wave/TG、16 TG，64 KiB LDS；无 Split-K。仅记录此组合的最佳 Gluon 实现。

完整网格复测通过两组随机输入和全1、全部4096输出、边界哨兵及16 TG检查；9次计时去掉首次，8个暖态样本中位数：接口墙钟 **1,023,208.4 μs**、Kernel **2.194 μs**、完整dispatch **5.806 μs**。接口墙钟包含PMD仿真开销；Kernel为first-wave至last-wave，dispatch为gcSimCycles。逐次数据与精度见 [result.json](result.json)，接口测试见 [dispatch_validation.json](dispatch_validation.json)。

环境为node60上的 `kxu_pmd_8569_fp8_20261017`，gfx946单DIE/48 CU；GC周期455 ps。工具链提交、编译器哈希、flags和环境变量见 [environment.json](environment.json)。[PERF](kernel.perf)来自同一kernel/config的独立采集，单次dispatch为5.725265 μs；其ISA与本次汇编逐行去除尾部空白后完全相同。

在匹配的PMD容器内、仓库根目录执行：

```bash
python scripts/perf/bench_a16w16_pmd.py \
  --case docs/ops/gemm/tuning/r001/1die/tn/benchmark.json \
  --environment docs/ops/gemm/tuning/r001/1die/tn/environment.json \
  --output /tmp/r001-1die-tn
```

脚本预编译后从公共接口测量，复用预分配输出，校验实际选择的kernel/config；每次运行使用输出目录内独立缓存。环境文件中的绝对工具链路径需要在复测容器中可用。
