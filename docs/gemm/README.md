# GEMM 当前支持能力

更新：2026-10-20。统一开发分支为 `perf/gemm-optimization`。本页汇总当前接口能力与限制；完整形状、scale、模型映射和逐行状态见 [少伯 BU 支持需求](SHAOBO_BU_REQUIREMENTS.md)。接口细节与调用示例见 [GEMM API](../ops/gemm.md)。

## 支持与验证边界

实测环境为 node60 / PMD gfx946，48 CU、LDS 65536 bytes，Triton release/3.6.x。下表 PASS 均为小尺寸正确性证据，**不表示完整需求尺寸或真实性能已验收**。本目录只记录 dense、BMM 与单次 grouped GEMM；dense 性能范围固定 E=1，不做 E 维度调优。grouped GEMM 的输入、scale 和分组寻址元数据由调用方预先准备。

| 算子族 / dtype | 当前接口 | 已有证据与限制 |
| --- | --- | --- |
| BMM，BF16×BF16→BF16 | `batched_gemm_bf16` | S24 PASS；权重布局 `[batch,N,K]`，需匹配 stride |
| dense，FP16/BF16×同精度→FP16/BF16/FP32 | `gemm_a16w16` | BF16→BF16 S05、BF16→FP32 S06 PASS；FP16 另有流水及成本配对测试；尚无全尺寸验收 |
| dense，FP32×FP32→FP32 | `matmul_persistent` | S03 修复后 PASS，64³ max_abs≈1.05e-7；依赖编译器 S03 LEGACY 补丁 |
| dense，FP8×FP8→BF16/FP16/FP32 | 下表三个接口 | 46/46 smoke PASS，max_abs=0、mismatch=0；需求表 2 只映射前两个接口 |
| dense，BF16×INT4→BF16 | `awq_gemm_triton` | S18 PASS；仅验证对应 AWQ 打包、scale/zero-point，不能泛化所有 INT4 格式 |
| dense / grouped，FP8×FP4→BF16/FP16/FP32 | `gemm_a8wfp4` / `grouped_gemm_a8wfp4` | 15/15 回归及 9/9 接口 smoke PASS（含复用）；等长、ragged、空组、尾块和 stride 已有覆盖；34/34 完整需求仅编译通过，精度队列暂停 |
| grouped，FP8×FP8→BF16 | [FP8 grouped kernel](../../boltops/fused_moe/triton/moe_op.py#L1292)（内部实现） | 需求表 7 待独立 grouped smoke；独立公开包装接口及部分 scale 组合仍需适配 |
| grouped，BF16×INT4→BF16 | [W4A16 grouped kernel](../../boltops/fused_moe/triton/moe_op.py#L307)（内部实现） | 需求表 9 待独立 grouped smoke；需确认 packed 格式、scale/zero-point 与分组寻址 |
| native FP4×FP4；batched FP16/BF16×FP4 | `gemm_afp4wfp4` / `batched_gemm_a16wfp4` | S34–S37 原生路径小尺寸 PASS；需要已修复的 ROCDL translation 构建环境 |

表 7–9 统一作为 grouped GEMM 需求：保留原 M/N/K、分组数 E、dtype、A/W scale、阶段及频次；E 对应来源中的 experts，不改变数值。不从来源 M 推断所有组等长，变长各组 M 与偏移须另行提供。上层路由、在线量化、激活和结果合并均不在本目录验收范围。FP8 / W4A16 的实现链接指向现有内部 kernel，并不表示已经新增同名 grouped 公共接口。

少伯 BU 清单含 9 张表、511 行 / 2504 实例：表 1 BMM、表 2–6 dense、表 7–9 grouped。469 行“接口 smoke 通过”是复用接口证据，表 7/9 共 42 行仍待直接 smoke。INT8 隐式 GEMM 卷积与末端 FC 不在该 511 行中，未在此宣称支持。

## FP8 接口与 scale

| 接口 | scale 语义与用途 |
| --- | --- |
| `gemm_a8w8_blockscale` | A 每行按 K128 分组或整行一个 scale；W 每输出通道 N1、按 K128 分组。15/15 smoke PASS，性能调优对象 |
| `w8a8_triton_block_scaled_mm` | 同上，另接收 `block_size` 并展开 A 的前导维。17/17 PASS，性能调优对象 |
| `triton_scaled_mm` | scalar / per-row / per-column 的 K 不变 scale。14/14 PASS，仅保留兼容性；不满足需求的 W 按 K128 分组 |

前两个接口按 K 分组执行 `dot` 后乘 A/W scale 再累加，不要求 `tl.dot_scaled`。uint8 scale 解码为 E8M0，浮点 scale 保持数值；A/W 可独立选择。需求中未明确的 W scale dtype 不补写为 E8M0。

FP8×FP4 接口使用预量化 A 与低/高半字节打包的 E2M1 W（偶数 K 取低半字节）；A scale 按 K32 或 K128，W 按 N1/K32。kernel 位映射 E2M1→E4M3、每 K32 普通 dot 后缩放，FP32 累加。它不等同于原生混合 FP8×FP4 指令验收，也不包含在线量化或路由生成。

## 优化状态与后续范围

- WDRA48 的 post-dot scale 分区依赖断言已修复，Triton 同名分支提交 `e33b79cf1`；34 项运行检查 PASS，4 项同 BK 普通流水对照逐元素一致。大 tile 仍受 LDS / VGPR 限制，尚无性能晋升结论。
- FP8×FP8 已建立对齐基线及 async / BPP 候选。下一步先调通少量关键流水，再上真实卡 autotune；不在 PMD 对 207 个需求形状展开全量搜索。
- Track A 的 FP8×FP4 stages 收益依赖 tile；Track B 的 FP16/BF16 已有 54 个单 TG 成本点，不能以单 TG 原始 cycles 直接决定完整 GEMM 配置。
- 生产 kernel 与默认配置尚未采用上述实验流水。旧 MR !122 已关闭，后续在统一优化分支推进。

关键根因、修复边界和环境能力缺口见 [debug 记录](debug_tuning/DEBUG.md)；有效性能数据、资源规则和候选计划见 [调优记录](debug_tuning/TUNING.md)。旧逐日记录不再维护，可从 Git 历史追溯。脚本、原始日志、StatLog、IR/汇编及大型产物均留在仓库外；不恢复暂停任务。
