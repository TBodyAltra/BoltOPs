# R008：MMAC 轮次同步修复

R008 为 `M=2560, N=2048, K=4096`，block 为 `128×256×128`。PMD 使用 gfx946、单 die、48 CU，GC 周期 455 ps（2.197802 GHz）；编译器 commit、选项和环境见 [record.json](record.json)。

原汇编将第一条 MMAC 排到 `ebarrier_sync` 之前，第二条之后便发出 `ebarrier_arrive`，其余 MMAC 落在交出轮次之后。标量同步与展开后的 MMAC 缺少调度顺序约束。现在使用原生 ebarrier 接口，并在交接 helper 两侧增加编译期调度屏障，使两组 consumer 的 6 处静态指令段均保持：

```text
ebarrier_sync → 本轮全部 64 条 MMAC → 对侧 ebarrier_arrive
```

没有新增 `lgkmcnt(0)`；LDS 112.25 KiB、VGPR 128、spill 0，资源占用不变。14 项回归通过，包含 R008/R009 的汇编顺序检查和 K=2048/4096/6144、bank 复用、TG 尾轮的数值测试。

| 测量范围 | 版本 | cycles | 耗时 | TOPS |
|---|---|---:|---:|---:|
| 完整 grid：48 TG / 160 tiles | 修复前历史基线 | 235768 | 107.274 μs | 400.37 |
| 完整 grid：48 TG / 160 tiles | 当前修复版 | 待测 | 待测 | 待测 |
| 单 TG：4 tiles | 修复前，同环境重跑 | 240160 | 109.273 μs | — |
| 单 TG：4 tiles | 当前修复版，仓库 replay | 210534 | 95.793 μs | — |

单 TG 两个点均为独立进程中先编译、再执行一次冷启动。当前点由 [benchmark.py](benchmark.py) 直接启动仓库入口采集，单次实测耗时降低 12.34%。该结果不能当作完整 GEMM 的耗时或收益。

单 TG 保留完整 shape 的编译特化和 `NUM_SMS=48`，仅 profiling grid 为 `(1,)`，处理 tile 0/48/96/144。131072 个输出在非 1 scale 下 max_abs=0、mismatch=0；未执行区域及前后 canary 保持不变。A/B scale buffer 为 FP32，B 数值沿用 BF16 精度；需求表精度不变。

当前 [源码](https://github.com/TBodyAltra/BoltOPs/blob/d9f85cc8553b1f7417578cc47c2d688df7fcd648/boltops/gemm/gluon/gemm_a8w8_blockscale_persistent.py#L750)、[config](https://github.com/TBodyAltra/BoltOPs/blob/d9f85cc8553b1f7417578cc47c2d688df7fcd648/boltops/gemm/configs/gfx946/48cu/gluon/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=2048-K=4096.json)、[汇编](kernel.amdgcn)、[PERF](single_tg.perf) 和 [记录](record.json) 已对齐。PERF 内嵌 HSACO 已核验，当前可执行指令段与入库源码编译结果一致。原始日志、IR 和其余实验版本保存在仓库外；历史全 grid 证据通过记录中的固定 commit 链接保留。

复现当前单 TG：

```bash
python docs/ops/gemm/tuning/r008/1die/tn/benchmark.py \
  --record docs/ops/gemm/tuning/r008/1die/tn/record.json \
  --output /tmp/r008-fixed-single-tg --single-tg-perf
```

R009 共用此同步 helper，因此它的旧完整 grid 成绩也标为历史基线；当前版仅完成编译顺序检查，完整 grid 性能待复测。本记录不对 Wavefronts 页中某一段长条的内部阻塞原因作结论。
