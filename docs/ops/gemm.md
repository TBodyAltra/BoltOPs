# Gemm 系列算子说明

`boltops.gemm` 提供统一 GEMM 入口；block-scaled GEMM 按映射选择 Triton 或 Gluon 实现。
实现分别位于 `boltops/gemm/triton/` 和 `boltops/gemm/gluon/`。

当前验收状态见 [GEMM 总体支持能力](../gemm/README.md)，需求见 [少伯 BU 支持需求](../gemm/SHAOBO_BU_REQUIREMENTS.md)，关键调优与资源约束见 [调优记录](../gemm/debug_tuning/TUNING.md)。

## 公开接口

公开入口及旧接口迁移表见 [GEMM dispatch](gemm-dispatch.md)。

```python
from boltops.gemm import (
    gemm_dense, gemm_scaled, gemm_batched, gemm_grouped,
    gemm_batched_a16wfp4_online_quant,
)
```

下文沿用原 kernel 名称的章节记录后端实现与历史测试；这些名字不再从
`boltops.gemm` 公开导出。需要后端专用参数时直接从 `boltops.gemm.triton` 的具体模块导入。

测试统一目录：

```bash
pytest -q test/gemm/
pytest -q test/gemm/ --bench -s   # 含 bench 的用例
```

溯源总表见 `boltops/gemm/triton/SOURCE.md`。

---

## `matmul_persistent`

vLLM Triton persistent GEMM：`C = A @ B [+ bias]`（NN）。

| 参数 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `a` | `[M, K]` | FP16/BF16/FP32 | 左矩阵 |
| `b` | `[K, N]` | 同 `a` | 右矩阵 |
| `bias` | `[N]` | 同 `a` | 可选 1D |

Launch 按 `utility.arch_info.get_lds_capacity_bytes()`：LDS ≤ 64KiB 用小 tile，
否则用 vLLM 默认（FP16 `128/256/64`、`num_stages=3`）。

实现：`triton/matmul_persistent.py`。溯源：vLLM `batch_invariant.py`。

---

## `gemm_a16w16`

aiter Triton A16W16：`Y = X @ W^T [+ bias]`（`W` 为 `[N, K]`）。

```python
y = gemm_a16w16(x, weight, bias=None, dtype=None, config=None, activation=None)
```

可选 epilogue activation（`gelu` / `silu` / `relu` 等）。暂无 HCU JSON；
默认见 `builtin_gemm_configs.py`。

实现：`triton/gemm_a16w16.py` + `_op.py`。溯源：ROCm/aiter `gemm_a16w16`。

---

## `awq_gemm_triton`

vLLM Triton AWQ（int4 packed weights）。

```python
out = awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters, config=None)
```

- `qweight`：`[K, N//8]` int32；`split_k_iters`：2 的幂，≤ 32
- HCU JSON：`triton/configs/awq_gemm_triton/`（`K100_AI` / `BW200`）

实现：`triton/awq_gemm_triton.py`。溯源：vLLM `awq_triton.py`。

---

## `gemm_afp4wfp4`

aiter Triton MXFP4：`Y = X @ W^T`。需 FP4 硬件（`utility.arch_info.is_fp4_avail()`）。

```python
y = gemm_afp4wfp4(x, w, x_scales, w_scales, dtype=torch.bfloat16)
```

- `x` / `w`：`[M, K//2]` / `[N, K//2]`；scales：`[*, K//32]` E8M0

实现：`triton/gemm_afp4wfp4.py` + `_op.py`。溯源：aiter `gemm_afp4wfp4`。

---

## `batched_gemm_a16wfp4`

aiter Triton batched A16WFP4：`Y[i] = X[i] @ W[i]^T`。需 FP4 硬件。

```python
y = batched_gemm_a16wfp4(x, w, w_scales, dtype=torch.bfloat16)
```

- `x`：`[B, M, K]` BF16/FP16；`w`：`[B, N, K//2]` FP4

实现：`triton/batched_gemm_a16wfp4.py` + `_op.py`。溯源：aiter batched path。

---

## `batched_gemm_a16w16`

aiter Triton batched BF16/FP16 GEMM：`Y[i] = X[i] @ W[i]^T`。

```python
y = batched_gemm_a16w16(
    x, w, bias=None, dtype=torch.bfloat16, y=None, config=None
)
```

- `x`：`[B, M, K]`，K 连续；`w` 的逻辑 shape 始终为 `[B, N, K]`。
  `w.stride(2) == 1`（K 连续）时选择 TN，`w.stride(1) == 1`（N 连续）
  时选择 TT；可选 `bias`：`[B, 1, N]`

### BW200 FFN shape 性能

#### 非 async 默认配置更新（2026-09-11）

本次只更新 `B=56,M=234` 的两组 N/K、TN/TT 四份 batch 专用 JSON，
不添加 `use_async_copy`、global/LDS prefetch distance，也不修改 kernel 或接口。
四份配置均关闭 `optimize_epilogue`；只有第二组 TT 保留普通 BlockPingpong。
这不是启用尚未合入的 planned async 流水。

同轮对比：BW200/gfx936（80 CU）、FP16、Torch 2.10.0 / HIP 6.3.26113，
`AMDGCN_USE_BUFFER_OPS=1`、`TRITON_BUFFER_CACHE_SWIZZLE=1`。
使用预分配输出，权重转布局不计时；HIP graph replay，复测 `rep=100 ms`
三轮取中位数。表中 TFLOPS 是逻辑 FLOP/事件时间估算，不是硬件计数。

| `[B,M,N,K]` | 布局 | 更新前默认 ms | 更新后 ms | 估算 TFLOPS | 加速比 | BM/BN/BK | PP |
|---|---|---:|---:|---:|---:|---|---|
| `[56,234,4928,1232]` | TN | 1.3067 | 1.0213 | 155.79 | 1.28x | 256/256/32 | off |
| 同上 | TT | 1.1634 | 0.8311 | 191.46 | 1.40x | 256/256/32 | off |
| `[56,234,1232,2464]` | TN | 0.6100 | 0.5034 | 158.04 | 1.21x | 128/256/64 | off |
| 同上 | TT | 0.4920 | 0.4013 | 198.23 | 1.23x | 256/256/32 | on |

四项均为 async-off 的实测配置，精度阈值 `atol=rtol=0.02`，最终赢家另用
seed 7/123 验证。第二组 TN 的 PP-on 候选实际回退，与 PP-off 的微小时间差
不作为收益，因此采用显式 PP-off。第二组 TT 的普通 PP 已确认实际生效。

版本口径：数据来自 BoltOps `8bbee9b1`、Triton `d138b529f` 的同轮调优；
后者是未合入的开发版本，但上述配置不使用新增 async/G/R 参数。
这不是已发布 Triton wheel 的性能保证，发布环境需复测。搜索固定
`optimize_epilogue=False`，表中是完整配置更新的收益，不能全部归因于 PP
或某个单独开关。原始数据保存在测试机
`/xukang/hygon_tmp/tune_batched_special_20260911/{bench,baselines}.json`。

历史 0814 rocBLAS 的 wide/narrow 为 0.724457/0.513769 ms（见下方历史记录）；
库版本及计时方式不同，不与本表计算严格同口径加速比。

#### 历史对比（默认配置更新前）

固定环境为 BW200/gfx936、FP16、`B=56, M=234`。Triton 数字使用 CUDA/HIP
Event benchmark（warmup 20 ms、rep 100 ms）；rocBLAS 数字使用固定 0814
rocBLAS，warmup 10 次、测量 50 次。TFLOPS 为按 GEMM FLOP 数计算的逻辑值，
不是 profiler 硬件计数。

| shape `[B,M,N,K]` | 实现 | 延迟 (ms) | 逻辑 TFLOPS | 相对原始 Triton |
|---|---|---:|---:|---:|
| `[56,234,4928,1232]` | 原始 Triton TN，固定 256x256x64 | 1.7047 | 93.34 | 1.00x |
| | 当前 Triton TN | 1.3087 | 121.59 | 1.30x |
| | 当前 Triton TT | 1.1710 | 135.89 | 1.46x |
| | rocBLAS 0814 TT | 0.7231 | 220.06 | 2.36x |
| `[56,234,1232,2464]` | 原始 Triton TN，固定 256x256x64 | 0.7715 | 103.12 | 1.00x |
| | 当前 Triton TN | 0.6068 | 131.11 | 1.27x |
| | 当前 Triton TT | 0.5014 | 158.66 | 1.54x |
| | rocBLAS 0814 TT | 0.5141 | 154.74 | 1.50x |
- `dtype`：`bfloat16` / `float16`；`splitK` 必须为 `None`
- HCU JSON（可选）：`triton/configs/batched_gemm_bf16/`  
  - batch 专用：`BATCHED_BF16-B={B}-N={N}-K={K}_{bf16|fp16}_{device}.json`
  - shape 通用：`BATCHED_BF16-N={N}-K={K}_{bf16|fp16}_{device}.json`
  - 设备默认：`BATCHED_BF16-DEFAULT_{bf16|fp16}_{device}.json`
  - TT 布局使用独立的 `BATCHED_BF16-TT-...` 文件，避免与 NT tile 混用
  - 按上述顺序回退，表内选择最接近的 `M`
- 无 JSON 时静默使用内置 large/small tile（`M+N >= 4096` → large）

### BW200 FP16 性能记录

#### BlockPingpong / optimized epilogue 四组合 autotune（2026-08-26）

更新至 Triton `release/3.6.x` 提交 `80d21d979`（llvm-triton
`ddb5c4b4b`）后，对 `BlockPingpong` 开/关和 optimized epilogue 开/关四种组合
重新 autotune。候选初筛使用 warmup 10 ms、repeat 30 ms；四套 winner 再统一使用
warmup 20 ms、repeat 100 ms 复测，表中 latency 为复测中位数。

| 环境项 | 值 |
| --- | --- |
| GPU | BW200, UBB BW1000（gfx936），64 GiB |
| CU 数量 | 80 CU |
| SCLK | DPM 300–1500 MHz；调优后采样活动档为 1350 MHz，未锁频 |
| PyTorch / HIP | PyTorch 2.10.0 / HIP 6.3.26113 |
| Triton | 3.6.0 (`80d21d979`)；llvm-triton `ddb5c4b4b` |
| 输入 / 输出 dtype | FP16 / FP16 |

| Shape `[B,M,N,K]` | 最优配置 `[BM,BN,BK,GROUP,warps,stages,waves,PP,epi]` | latency | TFLOPS |
| --- | --- | ---: | ---: |
| `[56,234,4928,1232]` | `[256,128,32,4,8,2,2,true,false]` | 1.480240 ms | 107.49 |
| `[56,234,1232,2464]` | `[256,256,16,4,8,3,2,true,true]` | 0.726720 ms | 109.48 |

此外，`B=16` 的三组 AITER size（`N/K=1024/4096`、`8192/1024`、
`1280/8192`）已对全部 15 个 M 点重新调优。47 条专用配置的四组合胜出分布为：
PP关/epi关 21 条、PP关/epi开 9 条、PP开/epi关 15 条、PP开/epi开 2 条。
TFLOPS 按 `2*B*M*N*K / latency` 计算。

#### TT 布局 autotune（2026-08-26）

TT 布局中 A 为 `[B,M,K]`（K 连续），B 为 `[B,K,N]`（N 连续）。对两个
`B=56,M=234` shape 搜索 tile、stage、waves、BlockPingpong、optimized
epilogue 和 max-ILP 开/关；初筛 warmup 10 ms、repeat 30 ms，winner 使用
warmup 20 ms、repeat 100 ms 重复 5 轮取中位数。0814 rocBLAS 使用原生 NN
布局，权重转换在计时区外完成。

| Shape `[B,M,N,K]` | NT winner | TT winner | TT vs NT | 0814 rocBLAS NN |
| --- | ---: | ---: | ---: | ---: |
| `[56,234,4928,1232]` | 1.481517 ms | 1.432398 ms | 1.034x | 0.727520 ms |
| `[56,234,1232,2464]` | 0.726879 ms | 0.608160 ms | 1.195x | 0.521759 ms |

两组 TT winner 均为 `BM=256, BN=256, BK=16, warps=8, stages=2`，并开启
BlockPingpong、optimized epilogue 和 max-ILP；前者 `waves_per_eu=2`，后者为
1。TT 相比 NT 分别提升 3.4% 和 19.5%，但 0814 rocBLAS NN 仍分别快约 1.97x
和 1.17x。GPU 未锁频，数据用于同轮相对比较。

#### wide grid-order / BlockPingpong 对齐（2026-08-26）

进一步对齐 rocBLAS wide dispatch：原 Triton launch `(B, tiles)` 在 hipprof 中
显示为 `(1,20,56)`；新增 config 项 `BATCH_LAST_GRID=true` 后使用
`(tiles,1,B)`，实际 trace 为 `(56,1,20)`，与 rocBLAS 完全一致。tile 数、总
workgroup 数和每 block 8 waves 均保持不变。

固定旧配置做消融时，交换 grid 后 1.4267 → 1.2213 ms（1.168x）；在新 grid
下 BlockPingpong on/off 为 1.2213 / 1.2779 ms（PP 贡献 1.046x）。grid 改写后
重新扫描 48 个固定候选，最终配置变为
`BM=256,BN=256,BK=32,warps=8,stages=2,waves=2,PP=true,epi=true,max_ilp=false`。
三轮 warmup=10、iters=50 的最终中位数如下：

| Shape `[B,M,N,K]` | 旧 TT | 新 TT | 新 TT 逻辑 TFLOPS | 0814 rocBLAS | rocBLAS / 新 TT |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[56,234,4928,1232]` | 1.4324 ms | 1.1658 ms | 136.49 | 0.7248 ms | 1.61x |

最终编译 metadata 为 `enable_consumer_pingpong=true`、LDS 32 KiB、VGPR 180；
相较旧 TT 的 PMC，`SQ_WAIT_INST_LDS` 从 26.74M 降至 17.32M（降低约 35%）。
逻辑 TFLOPS 由 FLOP 数/事件时延计算，不是 profiler 硬件利用率。

#### wide rocBLAS 64 KiB LDS 流水消融（2026-08-27）

xprof 中 rocBLAS 的 `thread group limit LDS=1` 表示其 64 KiB LDS 资源限制为每
CU 最多驻留一个 workgroup，并非 100% wave occupancy；同份报告的 theoretical
occupancy 实际为 20%。ISA 显示其 `BK=16` 主循环在 LDS 的 0/16/32/48 KiB 四个
区域之间轮换：下一块通过 `buffer_load_dwordx4 ... lds` 预取时，当前块由
`ds_read_m32x16_b16` / `ds_read_b64` 喂给连续的 `v_mmac_f32_16x16x16_f16`，
并用 `vmcnt(4)`、`lgkmcnt(8)` 和 barrier 控制跨代依赖。

为模仿这一资源形态，固定相同 grid/tile/8 waves/BlockPingpong，扫描 BK、stage、
max-ILP，以及 HCU 的 MMAC cluster、memory cluster、MMAC/DS latency 和
empty-arrive 调度项。所有结果均通过 `max_diff=0`：

| Triton 方案 | LDS | 最优 latency | 相对 32 KiB champion |
| --- | ---: | ---: | ---: |
| `BK=32, stages=2`（保留方案） | 32 KiB | 1.1676 ms | 1.000x |
| `BK=64, stages=2` | 64 KiB | 1.2729 ms | 0.917x |
| `BK=64, stages=2, sched_latency=mmac5-ds10` | 64 KiB | 1.2547 ms | 0.931x |
| `BK=32, stages=3` | 64 KiB | 1.2968 ms | 0.900x |
| `BK=16, stages=5`（四级、64 KiB） | 64 KiB | 1.3682 ms | 0.853x |

因此没有将 64 KiB 配置提升为默认。当前 Triton BlockPingpong 虽能生成多代 LDS
缓冲，但扩大 staging 会增加 LDS/wait 开销，无法自动复刻 Tensile 对 load、DS read
和 MMAC 的逐指令交错；`enable_v_mmac_cluster=1/2` 也分别退化到约 2.35/2.36 ms。
继续逼近 rocBLAS 需要改 HCU BlockPingpong lowering/software-pipeline pass，而不是
继续提高普通 Triton `num_stages`。

#### 与 PyTorch `torch.bmm` 对比（2026-08-26）

使用上述 47 条 FP16 专用配置与 PyTorch 基线
`torch.bmm(x, w.transpose(1, 2), out=y)` 对比。双方均预分配并复用输出张量，
使用 `triton.testing.do_bench`，warmup 10 ms、repeat 30 ms，记录中位数。
测试设备为 BW200 UBB BW1000（gfx936，64 GiB），PyTorch 2.10.0、HIP
6.3.26113；GPU 未锁频。加速比定义为 `torch latency / BoltOps latency`，大于
1 表示 BoltOps 更快。

| Shape 组 | BoltOps 胜出 | 几何平均加速比 | 结果 |
| --- | ---: | ---: | --- |
| `B=16, N=1024, K=4096`（15 个 M） | 5/15 | 0.78x | PyTorch 整体快约 1.28x |
| `B=16, N=1280, K=8192`（15 个 M） | 6/15 | 0.87x | PyTorch 整体快约 1.15x |
| `B=16, N=8192, K=1024`（15 个 M） | 14/15 | 3.99x | BoltOps 优势显著 |
| `B=56, M=234`（2 个 N/K） | 0/2 | 0.78x | PyTorch 快约 1.28x |
| 全部 47 个 shape | 25/47 | 1.36x | 总体值受 `N=8192, K=1024` 组拉高 |

| Shape `[B,M,N,K]` | BoltOps latency | PyTorch latency | 加速比 |
| --- | ---: | ---: | ---: |
| `[16,1,1024,4096]` | 0.158960 ms | 0.352320 ms | 2.22x |
| `[16,256,1024,4096]` | 0.210400 ms | 0.245359 ms | 1.17x |
| `[16,16384,1024,4096]` | 16.688301 ms | 5.859593 ms | 0.35x |
| `[16,192,1280,8192]` | 0.481120 ms | 0.770959 ms | 1.60x |
| `[16,16384,1280,8192]` | 33.791817 ms | 15.482065 ms | 0.46x |
| `[16,512,8192,1024]` | 0.884160 ms | 9.008794 ms | 10.19x |
| `[16,8192,8192,1024]` | 18.431824 ms | 143.936722 ms | 7.81x |
| `[16,16384,8192,1024]` | 36.505257 ms | 15.799188 ms | 0.43x |
| `[56,234,1232,2464]` | 0.726080 ms | 0.571359 ms | 0.79x |
| `[56,234,4928,1232]` | 1.482879 ms | 1.152639 ms | 0.78x |

`N=8192, K=1024` 在 `M <= 8192` 时 PyTorch 基线仅约 15 TFLOPS，BoltOps
在该区间最高达到 10.19x；但到 `M=16384` 时 PyTorch 切换到更高效路径，BoltOps
降至 0.43x。后续优化应优先关注 `N=1024/1280` 的大 M，以及两个 `B=56`
专用 shape。由于测试未锁频，数据用于同轮相对性能比较，不作为稳定性能门槛。

#### xprof 历史记录（2026-08-24）

2026-08-24 使用 GPU 2、`xprof --enable-sqtt` 采集。每个 shape 单独运行，
kernel 过滤器为 `batched_gemm`，计数器覆盖 `speed_of_light`、memory、occupancy、
instruction、compute、scheduler、wave、cache、UMC 和 UTCL2 等全部 11 个 section。

| Shape `[B,M,N,K]` | 专用 launch 配置 `[BM,BN,BK,GROUP,warps,stages,waves]` | 原始 dispatch | PMC replay 中位数 | replay TFLOPS |
| --- | --- | ---: | ---: | ---: |
| `[56,234,4928,1232]` | `[256,128,32,4,4,2,2]` | 2.001 ms | 1.735 ms | 91.73 |
| `[56,234,1232,2464]` | `[256,256,64,4,8,1,1]` | 0.914 ms | 0.703 ms | 113.14 |

TFLOPS 按 `2*B*M*N*K / latency` 计算。以上数据来自 xprof PMC replay，适合分析
单 kernel 的硬件利用率；由于计数器 replay/SQTT 会扰动执行时间，不作为无 profiler
端到端 benchmark 的性能门槛。

实现：`triton/batched_gemm_a16w16.py`。溯源：aiter `batched_gemm_bf16`。

---

## `w8a8_block_int8_matmul`

SGLang Triton INT8 block-scale GEMM。

```python
C = w8a8_block_int8_matmul(A, B, As, Bs, [128, 128], output_dtype=torch.float16)
```

HCU JSON：`triton/configs/w8a8_block_int8_matmul/`（`BW200` / `gfx938`，来自 hygon aiter
`block_w8a8`）。

实现：`triton/w8a8_block_int8_matmul.py`。溯源：SGLang `int8_kernel.py`。


## FP8 × FP4：gemm_a8wfp4 / grouped_gemm_a8wfp4

```python
from boltops.gemm.triton.gemm_a8wfp4 import gemm_a8wfp4, grouped_gemm_a8wfp4

# x[M,K], w[N,ceil(K/2)]
y = gemm_a8wfp4(x, w, x_scales, w_scales, activation_group_size=128)
# x[E,M,K], w[E,N,ceil(K/2)]
y = grouped_gemm_a8wfp4(x, w, x_scales, w_scales, activation_group_size=32)
# 按专家连续排列的 x[T,K]；offsets[E+1] 是 GPU 上的专家行边界
y = grouped_gemm_a8wfp4(x, w, x_scales, w_scales,
                        expert_offsets=offsets, activation_group_size=128)
```

激活为 E4M3 FN。权重为 uint8 打包的 E2M1，低 nibble 表示偶数 K，
高 nibble 表示奇数 K；支持 K 尾部、N/M 尾部及非连续正 stride。
激活 scale 的最后一维为 ceil(K/G)，G 可为 32 或 128；权重 scale 为 ceil(K/32)。
等长分组的 scale 分别为 [E,M,ceil(K/G)]、[E,N,ceil(K/32)]；
ragged 激活 scale 为 [T,ceil(K/G)]，权重 scale 布局不变。
uint8 scale 按原始 E8M0 指数解码；FP32/FP16/BF16 scale 按浮点数值使用。
输出 dtype 支持 BF16（默认）、FP16、FP32，可传入预分配 y。

ragged 的 offsets 必须是连续 int32/int64 GPU tensor，非递减、首项为 0、
末项为 T；允许空专家。接口要求输入已按专家分组，offsets 内容由调用方保证。
这里提供 grouped GEMM 单阶段计算，完整路由、激活和 MoE 两阶段融合由调用方组织。

实现会在 kernel 内把 E2M1 精确扩展为 E4M3，每个 K32 分组执行普通 tl.dot，
再乘对应的 A/B scale 并以 FP32 累加。无需主机反量化，也不调用 tl.dot_scaled。
这是精度优先的接口支持，尚无性能调优结论。
config 支持 BLOCK_SIZE_M、BLOCK_SIZE_N（2 的幂且 ≥16）、num_warps。
默认 16×32、4 warps，K 固定按 scale 边界 32 迭代。

实现：triton/gemm_a8wfp4.py。参考来源和实测记录：
[FP8×FP4 支持与验证](../gemm/README.md)。

## `gemm_scaled`

```python
y = gemm_scaled(a, b, a_scale, b_scale,
                      scale_recipe=((1, 128), (128, 128)),
                      scale_encodings=("numeric", "numeric"), out_dtype=torch.bfloat16, backend="auto")
```

A[M,K]、B[N,K] 连续存储。逐行 scale 使用 A recipe (1,K)，逐输出通道 scale 使用 B recipe (1,K)；
整张量 scale 分别使用 (M,K)、(N,K)，scale shape 均为 (1,1)。B 必须传 [N,K]，不再接受旧 NN 包装的 [K,N]。
接口由 gemm/scaled.py 统一校验，调度映射命中时调用已优化 Gluon，其余走通用 Triton。
scale_recipe 和 scale_encodings 显式传递，不再从 scale shape 猜测分组或把 numeric uint8 隐式当 E8M0。
原三个上游包装只保留内部实现文件，公开入口统一为此函数；不保留兼容别名。
详细能力与布局约定见 [统一接口](../gemm/debug_tuning/fp8_gluon_dense/README.md)。

## `gemm_dense` / `gemm_grouped`

```python
c = gemm_dense(a, b, bias=bias, out_dtype=torch.bfloat16)  # A[M,K], B[N,K]
y = gemm_grouped(a_groups, b_groups)  # A[E,M,K], B[E,N,K], same-size groups
y = gemm_grouped(a_rows, packed_b, group_offsets=offsets,
                 a_scale=as_, b_scale=bs_, b_format="fp4_e2m1",
                 scale_recipe=((1,128),(1,32)))
```

Dense 默认输出为输入 dtype；FP16/BF16 输入可选择 FP16/BF16/FP32 输出，FP32 目前输出 FP32。
Grouped 无 scale 的等长 FP16/BF16 目前输出 FP16/BF16；FP8×FP4 输出 FP16/BF16/FP32。
Grouped scale recipe/encoding 独立指定，numeric uint8 不隐式解释为 E8M0；没有现成原生实现的组合明确报错。
旧 gemm_block_scaled 名称由 gemm_scaled 取代，不保留兼容别名。其他专项算子的已有接口保持原有边界。
调度和当前支持范围见 [统一接口](../gemm/README.md)。

三个统一入口均支持 `backend="auto" | "triton" | "gluon" | "tilelang"`；显式指定不会跨后端回退。
TileLang 当前只注册框架扩展点，所有未实现组合明确报错。具体支持矩阵见 [GEMM 总览](../gemm/README.md)。
