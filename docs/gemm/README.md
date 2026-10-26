# GEMM 当前支持能力

接口状态更新：2026-10-20；需求来源核对：2026-10-23。统一开发分支为 `perf/gemm-optimization`。本页汇总当前接口能力与限制；完整形状、scale、模型映射和逐行状态见 [少伯 BU 支持需求](SHAOBO_BU_REQUIREMENTS.md)。接口细节与调用示例见 [GEMM API](../ops/gemm.md)。

## 支持与验证边界

实测环境为 node60 / PMD gfx946，单die48CU、实际PMD LDS128KiB（HIP报256KiB，arch_info报64KiB）；请求GPU2.2GHz，实际455ps/周期。容量差异和编译器版本见[关键debug记录](debug_tuning/DEBUG.md)。下表 PASS 均为小尺寸正确性证据，**不表示生产接口的完整需求尺寸或实际卡性能已验收**。外部副本的G171完整shape及G036/G081等调优结果单列于[调优记录](debug_tuning/TUNING.md)，不据此宣称全部需求已验收。本目录只记录 dense、BMM 与单次 grouped GEMM；dense 性能范围固定 E=1，不做 E 维度调优。grouped GEMM 的输入、scale 和分组寻址元数据由调用方预先准备。

| 算子族 / dtype | 当前接口 | 已有证据与限制 |
| --- | --- | --- |
| BMM，BF16×BF16→BF16 | `batched_gemm_bf16` | S24 PASS；权重布局 `[batch,N,K]`，需匹配 stride |
| dense，FP16/BF16×同精度→FP16/BF16/FP32 | `gemm_a16w16` | BF16→BF16 S05、BF16→FP32 S06 PASS；FP16 另有流水及成本配对测试；尚无全尺寸验收 |
| dense，FP32×FP32→FP32 | `matmul_persistent` | S03 修复后 PASS，64³ max_abs≈1.05e-7；已修正编译器 FP32 布局选择，真实 LTS 路径通过 |
| dense，FP8×FP8→BF16/FP16/FP32 | 下表三个接口 | 46/46 smoke PASS，max_abs=0、mismatch=0；需求表 2 只映射前两个接口 |
| dense，BF16×INT4→BF16 | `awq_gemm_triton` | S18 PASS；仅验证对应 AWQ 打包、scale/zero-point，不能泛化所有 INT4 格式 |
| dense / grouped，FP8×FP4→BF16/FP16/FP32 | `gemm_a8wfp4` / `grouped_gemm_a8wfp4` | 15/15 回归及 9/9 接口 smoke PASS（含复用）；等长、ragged、空组、尾块和 stride 已有覆盖；34/34 完整需求仅编译通过，精度队列暂停 |
| native FP4×FP4；batched FP16/BF16×FP4 | `gemm_afp4wfp4` / `batched_gemm_a16wfp4` | S34–S37 原生路径小尺寸 PASS；需要已修复的 ROCDL translation 构建环境 |

表 8 作为 grouped FP8×FP4 GEMM 需求：保留原 M/N/K、分组数 E、dtype、A/W scale、阶段及频次；E 对应来源中的 experts，不改变数值。不从来源 M 推断所有组等长，变长各组 M 与偏移须另行提供。上层路由、在线量化、激活和结果合并均不在本目录验收范围。

少伯 BU 清单当前含 7 张表、469 行 / 2424 实例：表 1 BMM、表 2–6 dense、表 8 grouped FP8×FP4。保留来源表号及 G 编号，不重排。469 行“接口 smoke 通过”是复用接口证据，不是 469 个完整需求尺寸 PASS。原始 Excel 另有 42 个 grouped shape 尚未纳入编号表，现作为待范围确认的覆盖缺口补列，未增加支持或 PASS 声明。INT8 隐式 GEMM 卷积与末端 FC 不在该清单中，未在此宣称支持。

## FP8 接口与 scale

| 接口 | scale 语义与用途 |
| --- | --- |
| `gemm_a8w8_blockscale` | A 按 K 分组或整行 scale；W 的 N/K 分组由 scale shape 推导，包含 N128/K128 的表达能力。15/15 为旧 smoke，新增需求组合待验证，性能调优对象 |
| `w8a8_triton_block_scaled_mm` | 另接收 `block_size=[block_n,block_k]`（例如 `[128,128]`）并展开 A 前导维。17/17 为旧 smoke，新增需求组合待验证，性能调优对象 |
| `triton_scaled_mm` | scalar / per-row / per-column 的 K 不变 scale。14/14 为旧 smoke；可表达 Hy3 公开 scalar scale 语义，不能表达 K 分组 scale；以 Excel 布局与网补字段作为测试要求 |

前两个接口按 K 分组执行 `dot` 后乘 A/W scale 再累加，不要求 `tl.dot_scaled`。uint8 scale 解码为 E8M0，浮点 scale 保持数值；A/W 可独立选择。Excel 缺失的 scale 信息按公开模型/框架配置补齐并标“网补”，不统一假设 E8M0。2026-10-23 核对发现，GLM/Qwen/DeepSeek 的公开 block-FP8 权重为 N128/K128，Hy3 为 per-tensor，不能统一解释成 N1/K128。以 Excel 明确字段优先；Q35 MXFP4 的 A 保留 FP8，Hy3 的 A 保留 Excel 逐行布局，缺失字段采用网补。静态 scalar 可由框架复制为 [M,1]，不能据 shape 判定动态/静态冲突。checkpoint dtype 与框架转换路径分开记录，详见 [模型配置与冲突](SHAOBO_BU_REQUIREMENTS.md#requirements-cross-check)。此前 G171 的 FP32 A + E8M0/N1 B 性能结果只适用于该实验组合。

FP8×FP4 接口使用预量化 A 与低/高半字节打包的 E2M1 W（偶数 K 取低半字节）；A scale 按 K32 或 K128，W 按 N1/K32。当前实现以 `tl.dot_scaled(a, None, "e4m3", b, None, "e2m1")` 执行每 K32 的原生 F32 混合 MMAC，再乘 A/W scale 并以 FP32 累加，按输出 dtype 写回。gfx946 需要配套的 Triton K-packed FP4 布局/寄存器格式修复；2026-10-21 PMD 验证为 19/19 接口回归及 10/10 新旧实现对照通过，尚不代表完整需求尺寸或实卡验收。实现、环境依赖及证据见 [原生 FP8×FP4 K32 修复](debug_tuning/FP8FP4_NATIVE_K32.md)。不包含在线量化或路由生成。

## 优化状态与后续范围

- FP16 单 TG block 流水的当前最佳配置、资源与复测证据见 [2026-10-26 round84 配置地图](debug_tuning/fp16_single_tg_20261026_round84/README.md) 与配套 JSON；按 BM/BN/BK 查找，供 BU 调优起步使用。历史快照：[round82](debug_tuning/fp16_single_tg_20261026_round82/README.md)、[round76](debug_tuning/fp16_single_tg_20261026_round76/README.md)、[round71](debug_tuning/fp16_single_tg_20261026_round71/README.md)、[round62](debug_tuning/fp16_single_tg_20261026/README.md)、[2026-10-23](debug_tuning/fp16_single_tg_20261023/README.md)。

- WDRA48 的 post-dot scale 分区依赖断言已修复，Triton 同名分支提交 `e33b79cf1`；34 项运行检查 PASS，4 项同 BK 普通流水对照逐元素一致。大 tile 仍受 LDS / VGPR 限制，尚无性能晋升结论。
- FP8×FP8 已建立对齐基线及 async / BPP 候选。下一步先调通少量关键流水，再上真实卡 autotune；不在 PMD 对 207 个需求形状展开全量搜索。
- Track A 的 FP8×FP4 stages 收益依赖 tile；Track B 的 FP16/BF16 已有 54 个单 TG 成本点，不能以单 TG 原始 cycles 直接决定完整 GEMM 配置。
- 生产 kernel 与默认配置尚未采用上述实验流水。旧 MR !122 已关闭，后续在统一优化分支推进。

关键根因、修复边界和环境能力缺口见 [debug 记录](debug_tuning/DEBUG.md)；有效性能数据、资源规则和候选计划见 [调优记录](debug_tuning/TUNING.md)。旧逐日记录不再维护，可从 Git 历史追溯。脚本、原始日志、StatLog、IR/汇编及大型产物均留在仓库外；不恢复暂停任务。
