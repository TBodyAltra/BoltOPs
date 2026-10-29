# FP8×FP8 FP32 block-scale：单 TG 最佳流水

2026-10-29 当前最佳：consumer 的 MMAC 前准备阶段使用 `s_setprio 3`，MMAC
交接后恢复 0；移除重复的晚设置。P0 搬 A0+共享 B，P1 搬 A1+全部 scale。
MMAC 采用 LTS，B 的 LDS 矩阵读取使用 interleave2；输出使用
`matrix_store_64x16_b8`，保存 BF16 原始位。MMAC 错峰使用 ebarrier8/9。
生产 kernel 与默认 config 未改，这是固定输入的外部汇编调度研究。

| 项目 | 实测 |
|---|---|
| M/N/K；BM/BN/BK；grid | 128/256/6144；128/256/128；(1,1,1) |
| stages / split-K / GROUP_SIZE_M | 2 / 1 / 1 |
| dtype | A/B E4M3FN；FP32 累加；BF16 输出 |
| A scale | FP32，M1/K128，[128,48]，stride[48,1] |
| B scale | FP32，N128/K128，[2,48]，stride[48,1] |
| PMD | gfx946/sb，1 die × 48 CU，455 ps/cycle ≈ 2.197802 GHz |
| LDS / spill / VGPR 配额 | 120.375 KiB / 0 / 8/8/240/240 |
| 当前最佳 cycles | **77,549**；样本 77350/77612/77578/77656 |
| 匹配原版对照 | 78,281.75；降低 **0.936%** |
| 精度 | seed21 与 CPU golden、保存的生产输出逐元素一致，max_abs=0，mismatch=0 |

四版共用输入和设备地址、各独立 function；各 warmup 一次，随后四轮交错计时，
每版在每个顺序位置出现一次。计时排除 JIT。hostSeconds 仅用于估仿真预算。
此为 dense、E=1、自然单 CTA 实验；按要求归档在 moe_grouped_gemm 下，
**不是 grouped/MoE 或完整需求 shape 的验证结果**。

producer 固定角色移出循环可将动态分支 2200→976，热循环无条件 s_branch 全部
消除，但单独精简后 78824.25 cycles，慢 0.693%；组合版 78064.75，仅低 0.277%。
因此当前最佳保留原 producer 角色循环。详细对照见 `results.csv`。

## 代码与产物

- [最佳汇编、PERF 与环境清单](http://42.228.13.241:10068/xukang/triton_gemm_asm/-/tree/main/moe_grouped_gemm/fp8e4m3xfp8e4m3_to_bf16/20261029-xukang-g171-dense-single-tg-priority)
- [执行汇编](http://42.228.13.241:10068/xukang/triton_gemm_asm/-/blob/fc150c69a8442b09fd71e19b470077bb276f25fe/moe_grouped_gemm/fp8e4m3xfp8e4m3_to_bf16/20261029-xukang-g171-dense-single-tg-priority/kernel.amdgcn)
- [下载 PERF](http://42.228.13.241:10068/xukang/triton_gemm_asm/-/raw/fc150c69a8442b09fd71e19b470077bb276f25fe/moe_grouped_gemm/fp8e4m3xfp8e4m3_to_bf16/20261029-xukang-g171-dense-single-tg-priority/432520_python3.10.perf)
- [Triton MR !518](http://42.228.13.241:10068/dcutoolkit/deeplearing/triton/-/merge_requests/518)
  的 `third_party/hcu/perf/gemm/fp8_blockscale_schedule/replay.py` 保存完整获胜变换链。

Python 源 kernel 仍为 `trial_mls_class_aligned_dotfirst.py`，独立调用它并不会生成
上述手调流水。`config.json` 从 `_get_config(M,N,K)[0]` merge 得到，不改 builtin。
manifest 保存测量 commit、私有 compiler SHA、编译选项、环境变量和产物 hash；
本次发布 commit 不是测量 commit，不声称在最新 release compiler 上重新测过。

Triton 测量 commit：`9fab6e70ed511f9de6a52cd4dd95eb60a3b74916`。
BoltOps 测量 commit：`b23492fbf1e34d1cdcea2da4f180c96005a8d9ce`。

PERF 取自独立 profile launch，77749 cycles，不计入性能均值。其嵌入 ELF 与
profile HSACO 完全相同；与计时版汇编和所有非 debug ELF 节相同，只有工作目录
引起的 DWARF line 元数据不同。二进制 hash 分别记录，未混用。

node60 宿主机 PERF：

```text
/home/hygon/xukang/code/pmd-triton-sessions/fp8-G171-fp32-blockscale-20261023/PF-priority-winner/m5out/0/1/432520_python3.10.perf
```

只在指定产物仓库保存最佳汇编/PERF；原始日志、StatLog、张量、IR、其他候选
产物留在仓库外。所有暂停进程保持 T，未启动队列，生产源码和默认配置未改。

产物归档 commit：`fc150c69a8442b09fd71e19b470077bb276f25fe`；汇编与 PERF 下载链接已固定到该 commit。
