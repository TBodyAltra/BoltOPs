# 特化版：fused split-K 算子包（R014 · 12×64×2048，旧 G256）

> 上级目录的 [README](../README.md) 记录该 shape 的需求编号、通用版对照与结果汇总。以下命令均在本目录下执行。本包原位于 `/public/home/duyubo/bolt/xk/BoltOPs/tmp/20261104/OPT/`，文中的 `../` 相对路径与旧目录引用指向该原位置，不在本仓库内。

目标：BF16、逻辑 M=12 / N=64 / K=2048、TT（A/B 均为行主序），PMD gfx946。

## 文件

- `fused_kernel.py`：fused split-K 内核，通过 `@triton.heuristics` 自动推导 `_M_ALIGNED/_N_ALIGNED/_EVEN_K/_SINGLE_WAVE`，用 MLS `matrix_load` 的 `boundary_check` 保护 M/N/K 越界，调用方不传 `EVEN_MNK`，支持动态 MNK。
- `best_config.json`：搜索得到的最优配置，逐字节拷贝。BM16 / BN32 / BK128、split8、每个 split 处理256个 K 元素、num_warps=1、num_stages=2、unroll=2。
- `bench.py`：只启动本目录内核的独立测试/计时脚本，读取本目录内核和配置；不依赖旧目录的 `tune_lib.py` 或其他测试脚本。
- `run.sh`：加载本机 Triton 环境并启动计时。

内核及配置来源：`/public/home/duyubo/bolt/xk/BoltOPs/tmp/20261103/12*64*2048/sonnet/splitk_fused/`。包内不包含其他内核版本。

## 一键运行

```bash
bash run.sh
```

默认运行两个独立进程，每个进程预热1次、正式计时5次；最终报告10个正式样本的中位数。每次运行自动创建独立的 `runs/<时间戳>/`，隔离 PMD 输出、HOME 和编译缓存，不覆盖已有结果。

快速验证：

```bash
bash run.sh --rounds 1 --warmup 1 --reps 3
```

指定新的输出目录和重复次数：

```bash
bash run.sh \
  --rounds 2 --warmup 1 --reps 5 \
  --out runs/my_new_run
```

只校验文件完整性、Python 语法和配置，无需启动模拟器：

```bash
python3 -B bench.py --check-only
```

## 计时与正确性

- 使用 PMD 每次 dispatch 的 `system.gcSimCycles`，通过 `gcFrequency` 换算微秒：`cycles / (freq_GHz * 1000)`。
- 只计矩阵乘法和内核内 split-K 归约，不包含编译、分配、中间缓冲区/计数器初始化、Python 调用或宿主机模拟时间。
- 中间缓冲区和计数器跨重复调用复用；计数器首次为零，此后由内核自行复位。
- 计时结束后检查最终输出、各 split 的 FP32 部分结果、输出/中间缓冲区/计数器的前后保护区及计数器是否复位。
- 本包保留独立的历史判据：最终 BF16 输出对**未舍入的 CPU FP32 参考**检查 `rtol=0.01, atol=0.001`，FP32 部分结果检查 `rtol=0.0001, atol=0.0001`。它与上层通用版当前的 BF16 舍入参考判据（`atol=0.02, rtol=0.01`，另检查平均误差）不同，本次仅重测通用版，未重测本包；下文 6.59 µs 是历史记录。

结果文件：运行目录下的 `summary.json` 为汇总；`round_*/result.json` 保存每个正式样本、配置和源码哈希，`round_*/run.log` 为模拟器日志，`round_*/g/0/*/stats.txt` 为原始统计。生成代码保存在每轮的 `kernel.amdgcn`、`kernel.ttir`、`kernel.ttgir` 和 `kernel.llir`。

## 动态 MNK 与边界保护

- K 切分：split `pid_k` 负责 `[pid_k*SPLITK_BLOCK_SIZE, k_end)`，最后一个 split 吸收 K 的尾部；K 很小时部分 split 为空（写零 partial，仍参与计数器协议）。
- MLS 以“K 切片”作为逻辑张量（`shape=[M,k_span]` / `[k_span,N]`），`boundary_check` 对 M/N/K 的越界部分补零，不会读到相邻 split 的 K 数据。
- 对齐的维度不做检查（heuristics 在编译期剔除）；`num_warps != 1` 会在编译期 `static_assert` 失败。
- 已知问题：B 仅检查 K 维（`boundary_check=(0,)`）在 gfx946 PMD 上结果为 NaN，因此 K 需要检查时同时检查 N 维。
- A 不需要物理补齐到 16 行，`bench.py` 直接使用逻辑 12x2048 的 A。
- 动态测试：`../OPT_dyn_test/test_dyn.py`（8 组 M/N/K，含 K 非 BK 整数倍、K>NUM_KSPLIT*SPLITK_BLOCK_SIZE、空 split、NaN 保护区、二次调用计数器复位），全部通过。
- 12x64x2048 实测（`runs/verification_heuristics_20261104`，3 个样本中位数）：约 6.59 us。

