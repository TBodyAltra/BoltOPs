# R001 BF16 TN 最佳 PMD 记录（2026-11-10）

完整 GEMM **64×64×2048**，BF16×BF16→BF16、FP32 累加、无 scale。[需求 R001](../../SHAOBO_BU_REQUIREMENTS.md#r001)。本记录保存当前最快的 **64 KiB 显式双缓冲汇编实验版**，ASM参考保持独立；纯Gluon版本已按下节接入公共 `gemm_a16w16` 分派。

## 配置与性能

| 方案 | BM×BN×BK | 展开 / stages / waves | LDS | VGPR / SGPR | 完整 dispatch 中位数 | shader 区间中位数 |
| --- | --- | --- | --- | --- | --- | --- |
| **当前最佳：显式双缓冲** | **16×16×128** | **4 / 2 / 1** | **64 KiB** | **133 / 91** | **5.75962 μs / 12658.5 cycles** | **2.14760 μs** |
| 同轮原编译器基线 | 16×16×128 | 4 / 2 / 1 | 32 KiB | 133 / 59 | 5.91796 μs / 13006.5 cycles | 2.30549 μs |

相同三个种子、各取8个暖态样本，共24次，中位数降低 **2.68%**。最佳方案均值5.75960 μs，范围5.65110–5.87268 μs。完整 dispatch 来自 `system.gcSimCycles`，以455 ps/cycle换算，包含模型调度开销；shader区间为firstWave到lastWave，不能直接与完整dispatch的周期混用。

无 Split-K；4×4输出网格共16TG，每TG一个wave，完整K=2048。A `[64,2048]` stride `(2048,1)`；物理W `[64,2048]` 连续，B=W.T stride `(1,2048)`；C stride `(64,1)`。TN沿用本任务约定：A沿K连续称T，B沿K连续称N。

## Gluon 公共接口与 config

[公共入口](../../../../../boltops/gemm/gemm_a16w16.py)按现有GEMM组织形式选择backend；[Gluon模块](../../../../../boltops/gemm/gluon/gemm_a16w16.py)只提供device kernels，专用[config](../../../../../boltops/gemm/configs/gfx946/48cu/gluon/gemm_a16w16/GEMM-A16W16-N=64-K=2048-TN.json)用精度profile `DTYPE=bfloat16-OUT=bfloat16`、`M_EQ_64`及`kernel`字段选择实现。device kernel接收统一的M/N/K及tile参数，物理权重接口仍为W[N,K]，在MLS读取中按逻辑W.T解释；不要求调用方绕过公共入口或转置/复制存储。

```python
from boltops.gemm import gemm_a16w16

# A、W均为contiguous BF16 [64,2048]。
C = gemm_a16w16(A, W, backend="gluon")
# backend="auto" 在精度、布局、MNK、设备/CU和记录config匹配时选Gluon。
```

Auto无显式config时只选精确匹配的记录，其余走Triton；既有显式config沿用Triton默认路径。显式Gluon不跨backend重试，未提供精度/形状配置或不支持的布局、bias、activation明确报错。显式Gluon config由公共入口复制后读取`kernel`，不修改调用方字典；网格和kernel选择均来自config。

| 实现 | 完整dispatch中位数（24次） | shader中位数 | LDS / VGPR / SGPR | 对应汇编 |
| --- | --- | --- | --- | --- |
| 原手工ASM参考（当前最快） | **5.75962 μs** | 2.14760 μs | 64KiB / 133 / 91 | [kernel.amdgcn](kernel.amdgcn) |
| **公共接口 Gluon** | **5.81126 μs** | 2.19924 μs | 64KiB / 133 / 81 | [gluon.amdgcn](gluon.amdgcn) |

公共接口复测使用seed23/59的auto分派及seed43的显式Gluon；每次通过观察实际JIT launch确认所选kernel/config与预编译句柄一致，返回用户提供的out。均值5.80756 μs，范围5.69796–5.93138 μs；预分配out，计时为GPU dispatch，不含Python配置查找和分配。零容差外mismatch、零nonfinite，哨兵和16TG检查通过、spill=0；[原始逐次数据](gluon_measurements.json)。

[24项接口测试](dispatch_validation.json)通过，包括auto精度/形状/双操作数布局分派、显式kernel/config选择、输入config不变、unsupported无重试、输出对象复用及公共API的auto/explicit正确性。三个种子、24暖态样本和额外PERF均从公共入口执行，没有ASM/二进制改写。

[gluon.perf](gluon.perf)和[对应stats](gluon.stats.txt.gz)来自额外seed59公共auto调用：完整dispatch **5.725265 μs / 12583 cycles**，shader **2.113020 μs / 4644 cycles**。与最终复测的ISA、机器码和descriptor一致，16CU各一个TG；单次采集不替代24次统计。沿用下方环境，将命令改为`python3 "$record_dir/gluon_replay.py" "$record_dir/gluon_job.json"`可重放公共入口。

## ASM参考实现与汇编

[源 kernel](kernel.py) → [汇编改写](asm_patch.py)的 `pingpong_unrolled` → [实际执行汇编](kernel.amdgcn)。`kernel.py` 是原始编译前端，单独运行不会得到本记录中的双缓冲ISA；[重放器](replay.py)会应用改写、重新汇编，再加载选定HSACO。保留原函数符号名 `_gemm_bf16_fused_splitk_kernel`，但本配置没有Split-K、partial、counter或atomic。

两块32KiB LDS交替使用。当前批DS read发出后，下一批MLS写另一块LDS；将 `lgkmcnt(0)` 移到下一批MLS发出后、当前MMAC前。K=2048对应prologue加三次refill，显式展开，固定两个地址组，最后drain bank1；FP32累加顺序不变。启动的动态LDS、metadata、packed_metadata及缓存JSON同步设置为65536字节，spill=0。ASM改写仅适用于已断言的R001编译结果，不能作为通用配置应用。

汇编归档仅清理行尾空格；原始和归档SHA256、资源、启动选项、工具链哈希见 [manifest](manifest.json)。PERF与seed59最终复测的ISA、ELF `.text`及kernel descriptor一致；完整HSACO哈希可能因调试/构建信息不同而不同，均单独记录。

## 环境

| 项目 | 实际环境 |
| --- | --- |
| 主机 / 容器 | node60（10.62.52.149）/ `kxu_pmd_8569_fp8_20261017` |
| PMD | CoreArch:HEAD_1703，gfx946单die，48CU，64-bank LDS容量128KiB |
| shader时钟 | 455 ps/cycle（配置名义2.2GHz，模型量化后约2.197802GHz） |
| Triton | `release/3.6.x@d8374cbaca28e5c4e7789fcf130b6701f71be0eb` |
| GPU代码生成Clang | `dtk-llvm-6.3.3@3599a58d7aebe66dc1cec4f830f82cb96e505666`，含LDS read pairing优化 |
| 主要编译选项 | `mmac_layout_force=2`、`enable_v_mmac_cluster=2`、`hcu_use_gfx946_sched_model_v2=True`；WASP/WDRA关闭 |
| 编译环境 | 原同提交libtriton产物；GPU Clang及ld.lld位于容器 `/xukang/code/pmd-triton-sessions/llvm-dtk-6.3.3/build/bin/` |
| 采集环境 | `/xukang/code/pmd-triton-sessions/r001-double-buffer-20261110`，独立runtime/cache及每case目录 |

## ASM参考的正确性与 PERF

种子23/43/59，各验证两组随机输入和全1，再计时9次、排除首次。CPU FP32 matmul转BF16为golden，atol=.001、rtol=.01；每次检查全部4096输出、nonfinite、前后哨兵、16个完成TG及实际执行的二进制。全部通过，零容差外mismatch；不声明bitwise一致或实卡验收。各次精度、周期、完整参数、父/改写ASM哈希及HSACO哈希见 [measurements](measurements.json)。原ASM部分复用这些已完成的PMD记录；本PR另对下方纯Gluon实现完成新的三种子PMD复测。

[best.perf](best.perf)为额外的seed59单次暖态采集：完整dispatch **5.643365 μs / 12403 cycles**；shader **2.031120 μs / 4464 cycles**。它不替代24次中位数。采集使用 `StatLog,TT,Perf` 及 `/opt/rocm/lib/xprofiler/libperf_gen_helper.so`，未设置CU过滤；16个CU各执行一个TG。对应 [原始stats（gzip）](warm.stats.txt.gz) 和CU分布/文件SHA256见manifest。

## 重放ASM参考

在上述PMD容器、同版本工具链中执行；每个进程使用自己的工作目录：

```bash
record_dir=/path/to/BoltOPs/docs/ops/gemm/tuning/r001_best_20261110
run_dir=/tmp/r001-best-replay-seed59
mkdir -p "$run_dir"
cd "$run_dir"
source /xukang/code/pmd-triton-sessions/r001-double-buffer-20261110/env.sh
export TRITON_HOME="$run_dir/triton-home" TRITON_CACHE_DIR="$run_dir/cache"
export GPU_DFLAGS="['StatLog']" GPU_ARGS="['--abs-max-tick=1000000000']"
python3 "$record_dir/replay.py" "$record_dir/job.json"
```

生成的`kernel.amdgcn`/`kernel.hsaco`是改写后实际执行版本，`original.*`保留改写前产物；结果与逐次stats路径在`result.json`。切换种子时复制job.json修改seed，并使用新的运行目录。普通复测不启用PERF，PERF需要另设上述采集flags及helper。
