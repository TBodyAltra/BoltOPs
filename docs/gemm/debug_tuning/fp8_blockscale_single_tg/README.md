# FP8×FP8 FP32 block-scale：单 TG 最佳流水记录

2026-10-28 选择保留 **416739**：P0 搬 A0+共享 B，P1 搬 A1+全部 scale，
A0/A1 分别同步，B 双缓冲由两组 consumer 共同释放。此目录保存实验源 kernel、
有效配置、环境与精简结果；**不替换生产接口或默认 config**。

## 范围与结果

| 项目 | 配置 / 结果 |
| --- | --- |
| M/N/K | 128/256/6144，自然单 CTA，非完整需求尺寸 |
| BM/BN/BK，stages | 128/256/128，2 |
| dtype | A/B FP8 E4M3FN，输出 BF16 |
| A scale | FP32，M1/K128，物理 `[128,48]`，stride `[48,1]` |
| B scale | FP32，N128/K128，物理 `[2,48]`，stride `[48,1]` |
| 环境 | PMD gfx946/sb，1 die × 48 CU，455 ps/cycle ≈ 2.197802 GHz |
| VGPR 配额 / spill | 8/8/240/240，spill=0 |
| LDS | 123264 B = 120.375 KiB |
| 六次计时均值 | **79054.333 modeled cycles** |
| 同环境 414931 对照 | 81311.333 cycles，下降 2.776% |
| 数值 | seed20/21 均与 CPU golden、保存的生产输出逐元素一致，max_abs=0，mismatch=0 |

两个独立计时批次各 warmup 后测 3 次；计时排除 JIT/精度/warmup。独立
seed21 PERF 为 79184 cycles，不混入均值。后续控制组复测 79069.333；独立分工
候选六次 79410.000（+0.431%），无稳定收益，因此保持原版。
`results.csv` 包含这轮关键候选统计；hostSeconds 只反映仿真预算，LGKM 统计不是
逐条指令等待时间，VMEM 次数也不是 L2 字节数。未验证实卡、多 CTA、其他 die 或
完整需求 shape 的性能，不能把此单 TG 结果登记为完整 GEMM 验收。

此记录对应修正后的 G171 **FP32 A/B scale**，不能与旧 E8M0、N1 B-scale
实验直接比较。scale 类型和布局见 [少伯需求](../../SHAOBO_BU_REQUIREMENTS.md)。

## 代码与复现边界

- `trial_mls_class_aligned_dotfirst.py` 是实际测试的外部源 kernel 副本，保留原
  文件内容及 SHA；其中 decode helper 支持的其他 dtype 路径不在本条结果范围。
- `config.json` 是本条有效配置。基线从生产 `_get_config(M,N,K)[0]` 获取后 merge，
  不改 `builtin_gemm_configs.py`。它包含实验副本专用参数，不可原样传给任意接口。
- 最终性能还依赖手工变换的 TTGIR/AMDGCN，**仅调用这个 Python kernel 不会得到
  416739 流水**。[Triton MR !518](http://42.228.13.241:10068/dcutoolkit/deeplearing/triton/-/merge_requests/518) 的
  `third_party/hcu/perf/gemm/fp8_blockscale_schedule/` 保存完整十一段变换链和带输入/
  输出 SHA 校验的 `replay.py`，只接受已验证的冻结输入，不是通用编译器 pass。
- `manifest.json` 记录实际测量的 Triton/BoltOps commit、私有 compiler SHA、
  1 die/48 CU/频率、有效选项、环境变量及执行 ASM/HSACO/PERF SHA。发布时的
  新分支 commit 不是原测量 commit，不冒充在最新编译器上重新跑过。

配置注入示例（其余参数必须保留）：

```python
import json
from pathlib import Path
from boltops.gemm.triton.gemm_a8w8_blockscale_op import _get_config

config = dict(_get_config(128, 256, 6144)[0])
config.update(json.loads(Path("config.json").read_text()))
```

原始执行资料保留在 node60 容器的仓库外目录：

```text
/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/
  worker_split_A_producers.py
  preload-scale-joint-prefetch.ttgir
  P224-split-A-producers/{request.json,environment.json,launch.sh,result.json}
  P225-split-A-producers-repeat/result.json
  PF-split-A-producers/result.json
```

仅发布复现源代码和关键统计，IR/ASM/HSACO、输出张量、日志、StatLog、PERF 不进 Git。
不要直接重放历史 launch 启动旧队列或影响暂停进程；新的执行需要独立仓库外目录。
生产 kernel、默认配置和正在使用的编译器库均未修改。

PERF 的 node60 宿主机绝对路径：

```text
/home/hygon/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/PF-split-A-producers/m5out/0/1/416739_python3.10.perf
```

## 同步及资源约束

A0/A1 各双槽共 32 KiB，B 双槽 64 KiB，48 组 As 24 KiB，Bs 384 B。
scale 按轮加载但物理切片不复用。A0 empty0/2 仅 C0 到达，A1 empty6/10
仅 C1 到达，均 count4；B empty8/12 count8，必须两组 consumer 均读完才释放。
A1 ready7/11 同时追踪 scale，双方观察后向 count8 的 barrier14 确认，避免
ready phase 跨代覆盖。MMAC turn4/5 保留。完整协议见 Triton 配套目录。

本轮其他方案虽然精度通过，但没有取代原版：单独排空 B 会损失读重叠；B 三槽
需要压缩 A1/scale 缓冲，引入额外生命周期等待；独立负载分工减少约 33% SALU，
周期仍持平。不能通过提前释放尚未读完的共享 B 来消除慢 consumer 的影响。
