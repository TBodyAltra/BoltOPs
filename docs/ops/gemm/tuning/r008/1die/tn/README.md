# R008：MMAC 轮次同步修复

M/N/K=`2560/2048/4096`，block=`128×256×128`，完整 grid 为 48 TG / 160 tiles。环境为 gfx946、单 die、48 CU、GC455 ps（2.197802 GHz）；Triton/Clang commit、编译选项和环境变量见 [record.json](record.json)。

原汇编在领取轮次前发射第一条 MMAC，并在第二条后便交出轮次。R008 的分段 scale 流水现在使用独立的原生 ebarrier helper，并在交接两侧增加编译期调度屏障，6 处静态指令段均保持：

```text
ebarrier_sync → 本轮完整 64 条 MMAC → 对侧 ebarrier_arrive
```

没有新增 `lgkmcnt(0)`；LDS 112.25 KiB、VGPR 128、spill 0。其他 persistent 入口保留原 helper，记录及表格未改。

| 测量范围 | 版本 | cycles | 耗时 | TOPS |
|---|---|---:|---:|---:|
| 完整 grid | 修复前历史基线 | 235768 | 107.274 μs | 400.37 |
| 完整 grid | 当前版直接 replay | 215184 | 97.909 μs | 438.67 |
| 单 TG、4 tiles | 当前版独立 profiling | 210629 | 95.836 μs | — |

完整 grid 相比历史基线耗时减少 8.73%。当前点为先编译、再执行一次完整冷启动，计时不含 JIT、输入准备、回读及 CPU golden；历史对照未重新运行。

全部 5242880 个输出通过非 1 scale 数值检查，max_abs=0、mismatch=0，48 个 CU 均完成一个 TG，边界 canary 保持不变。13 项回归通过，覆盖指令顺序、K2048/4096/6144、scale bank 复用和 TG 尾轮。

单 TG 保留完整 shape 编译特化与 `NUM_SMS=48`，profiling grid=(1,)，处理 tile 0/48/96/144；131072 个输出逐元素一致。其指令段与完整 grid 一致，PERF 内嵌 HSACO 已核验。

当前 [源码](https://github.com/TBodyAltra/BoltOPs/blob/1119a5fa81722505a846c11760709770c2129ae6/boltops/gemm/gluon/gemm_a8w8_blockscale_persistent.py#L768)、[config](https://github.com/TBodyAltra/BoltOPs/blob/1119a5fa81722505a846c11760709770c2129ae6/boltops/gemm/configs/gfx946/48cu/gluon/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=2048-K=4096.json)、[汇编](kernel.amdgcn)、[PERF](single_tg.perf)、[复现脚本](benchmark.py) 和 [记录](record.json) 已对齐。历史基线通过固定 commit 链接保留；原始日志、IR 和中间产物在仓库外。

```bash
# 完整 grid：性能和全部输出精度
python docs/ops/gemm/tuning/r008/1die/tn/benchmark.py \
  --record docs/ops/gemm/tuning/r008/1die/tn/record.json \
  --output /tmp/r008-fixed-full-grid

# 独立单 TG 流水采集
python docs/ops/gemm/tuning/r008/1die/tn/benchmark.py \
  --record docs/ops/gemm/tuning/r008/1die/tn/record.json \
  --output /tmp/r008-fixed-single --single-tg-perf
```
