# GEMM 当前支持能力

接口结构更新：2026-11-06；需求来源核对：2026-10-23。统一开发分支为 `perf/gemm-optimization`。本页汇总当前接口能力与限制；完整形状、scale、模型映射和逐行状态见 [少伯 BU 支持需求](SHAOBO_BU_REQUIREMENTS.md)。接口细节与调用示例见 [GEMM API](../ops/gemm.md)。

## 支持与验证边界

实测环境为 node60 / PMD gfx946，单die48CU、实际PMD LDS128KiB（HIP报256KiB，arch_info报64KiB）；请求GPU2.2GHz，实际455ps/周期。容量差异和编译器版本见[关键debug记录](debug_tuning/DEBUG.md)。下表 PASS 均为小尺寸正确性证据，**不表示生产接口的完整需求尺寸或实际卡性能已验收**。外部副本的G171完整shape及G036/G081等调优结果单列于[调优记录](debug_tuning/TUNING.md)，不据此宣称全部需求已验收。本目录只记录 dense、BMM 与单次 grouped GEMM；dense 性能范围固定 E=1，不做 E 维度调优。grouped GEMM 的输入、scale 和分组寻址元数据由调用方预先准备。

| 算子族 / dtype | 当前接口 | 已有证据与限制 |
| --- | --- | --- |
| BMM，BF16×BF16→BF16 | `gemm_batched` | S24 PASS；权重布局 `[batch,N,K]`，需匹配 stride |
| dense，FP16/BF16×同精度→FP16/BF16/FP32 | `gemm_dense` | BF16→BF16 S05、BF16→FP32 S06 PASS；FP16 另有流水及成本配对测试；尚无全尺寸验收 |
| dense，FP32×FP32→FP32 | `gemm_dense` | S03 修复后 PASS，64³ max_abs≈1.05e-7；已修正编译器 FP32 布局选择，真实 LTS 路径通过 |
| dense，FP8×FP8→BF16/FP16/FP32 | `gemm_scaled` | 统一 scale recipe/encoding；映射命中走已优化 Gluon，其余走通用 Triton |
| dense，BF16×INT4→BF16 | `gemm_awq` | S18 PASS；仅验证对应 AWQ 打包、scale/zero-point，不能泛化所有 INT4 格式 |
| dense / grouped，FP8×FP4→BF16/FP16/FP32 | `gemm_scaled` / `gemm_grouped` | 15/15 回归及 9/9 接口 smoke PASS（含复用）；等长、ragged、空组、尾块和 stride 已有覆盖；34/34 完整需求仅编译通过，精度队列暂停 |
| native FP4×FP4；batched FP16/BF16×FP4 | `gemm_mxfp4` / `gemm_batched_a16wfp4` | S34–S37 原生路径小尺寸 PASS；需要已修复的 ROCDL translation 构建环境 |

表 8 作为 grouped FP8×FP4 GEMM 需求：保留原 M/N/K、分组数 E、dtype、A/W scale、阶段及频次；E 对应来源中的 experts，不改变数值。不从来源 M 推断所有组等长，变长各组 M 与偏移须另行提供。上层路由、在线量化、激活和结果合并均不在本目录验收范围。

少伯 BU 当前业务需求按业务/模型保留 R 编号，来源和精度/scale 字段见 [需求清单](SHAOBO_BU_REQUIREMENTS.md)。接口 smoke 状态不代表这些完整尺寸均已通过；既有 grouped FP8×FP4 支持范围仍单独记录。

公开入口 `gemm_scaled` 支持不同 M/N/K、多CTA和尾块、独立 A/B scale 分组及指定输出 dtype，不限定 E4M3FN/FP32 scale/固定 recipe/BF16 输出；实际能力由目标指令和 PMD 决定。统一接口的 A/B scale 编码通过 `scale_encodings` 独立指定，E8M0 使用显式 uint8 指数码，不与普通数值 uint8 混用。完整说明见 [接口说明](debug_tuning/fp8_gluon_dense/README.md)。

<a name="gluon-layout"></a>

## 普通 GEMM 与独立量化接口

| 接口 | 场景 | 当前实现 |
| --- | --- | --- |
| `gemm_scaled` | 单个带 scale 的 GEMM，显式指定 recipe、编码及打包格式 | 命中表的 FP8×FP8 使用优化 Gluon；其他 native 组合走通用 Triton；FP8×FP4 复用原生混合格式实现 |
| `gemm_dense` | 无 scale 的 A[M,K] @ B[N,K].T，可带 bias[N] | gfx946 FP16 匹配已有配置时复用 gemm_fp16；其他 FP16/BF16 走 a16w16，FP32 走 persistent |
| `gemm_batched` | A[B,M,K] @ B[B,N,K].T，无 batch 广播 | 复用已有 native16 batched kernel |
| `gemm_grouped` | 等长 A[E,M,K] 或带 offsets 的 A[T,K] | 等长 FP16/BF16 复用 batched kernel；FP8×FP4 复用已有等长/ragged kernel |

全部公开入口均接受 `backend="auto"`（默认）或显式的 `"triton"`、`"gluon"`、`"tilelang"`。
另保留 `gemm_awq`（AWQ 零点/打包契约）、`gemm_mxfp4`（预量化 A/B）、
`gemm_batched_a16wfp4`（在线 MXFP4 A 量化）三个不重复的入口。旧公开包装已移除，
旧 kernel/实现函数保留在后端模块。迁移表见 [统一接口](../ops/gemm-dispatch.md)。

三个后端独立注册各接口的实现；公共参数校验与设备查询不导入具体 DSL。

| 后端 | scaled | dense | grouped |
| --- | --- | --- | --- |
| Triton | 通用 native block scale、已有 FP8×FP4 | 已有 FP16/BF16/FP32 | 已有等长 native16、等长/offsets FP8×FP4 |
| Gluon | 已优化 E4M3FN + FP32 A1×128/B128×128 → BF16 | 未注册 | 未注册 |
| TileLang | 未注册 | 未注册 | 未注册 |

自动调度先跨后端查统一性能映射，再选择已安装后端的通用实现。
显式指定后端时只考虑该后端；没有对应组合报 `NotImplementedError`，已注册后端缺运行时依赖报 `ImportError`。
编译或执行错误直接向上传递，不捕获后换后端。TileLang 本轮仅接入框架，没有新增 kernel 或正确性通过声明。

公共框架按 MNK、输入/输出 dtype、scale、布局、arch 和 num_cus 分派；tile 是选出的配置。
每个接口只读取 `configs/` 下同名的 JSON 文件，校验 interface 标识；包括普通四接口及三个独立量化接口，共七份文件。
schema 3 的 match/config 分离，config 显式包含 backend 和 implementation，不存统计、频率、commit 或证据路径。
当前只登记 48 CU 上可接入的配置，不推定 96/192 CU 已调优；未命中的请求使用对应已有通用实现。
其他既有专项算子仍保留原接口；三个统一入口不依赖这些公开包装，直接复用后端实现。

```text
gemm/
  scaled.py                   # gemm_scaled
  dense.py                    # gemm_dense
  grouped.py                  # gemm_grouped
  descriptors.py / registry.py / catalog.py
  performance_map.py
  configs/gemm_scaled.json     # gemm_scaled 的跨后端配置
  configs/gemm_dense.json      # gemm_dense 的跨后端配置
  configs/gemm_grouped.json    # gemm_grouped 的跨后端配置
  configs/gemm_batched.json
  configs/gemm_awq.json / gemm_mxfp4.json / gemm_batched_a16wfp4.json
  triton/
    dispatch.py               # 独立注册 scaled / dense / grouped
    gemm_block_scaled.py / block_scaled_dispatch.py
    dense_dispatch.py         # 适配已有 gemm_fp16、a16w16、persistent
    grouped_dispatch.py       # 适配已有 native16、FP8×FP4 grouped
    configs/gemm_fp16/gfx946/  # 复用原配置，不改其内容
  gluon/
    dispatch.py               # 仅注册已有 scaled 特化
    dense/dispatch.py
    dense/kernels/gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16.py
  tilelang/dispatch.py         # 三接口扩展点，当前能力为空
```

新增后端实现时，在对应 `dispatch.py` 的 `INTERFACES` 中注册 `Implementation`，
声明 backend、配置生成与支持条件；支持条件匹配后才加载选中的 kernel。
优化后的配置可以加入统一映射，tile 仍是配置而非问题匹配条件。

FP16 已有配置需要区分输入布局：旧 kernel 接受 B[K,N]，统一入口接受 B[N,K]，adapter 只做转置 view。
映射中的 k_contiguous/n_contiguous 指统一入口 B 的实际内存连续方向，不是数学转置开关。
已接入复核通过的现有 256×256×64 TT 配置。旧 128×128×128 MLS 配置在当前编译器指令选择中失败，
TN 256×256×64 复核进程发生段错误，原因尚未定位；这两类暂未登记，仍由已有通用实现处理。
文档中的单 TG 实验地图包含专用 kernel/frontend 依赖，保留为调优材料，不直接当生产配置加载。

当前 grouped 的普通 offsets/ragged FP16/BF16、group mask/indexed 等组合没有注册实现，会明确报错。
不通过逐专家调用 dense 或软件解包凑出支持，不新增 MoE 路由或 E 维调优。

## FP8 接口与 scale

| 统一入口 | scale 语义 |
| --- | --- |
| `gemm_scaled` | 独立指定 A/B block recipe；支持 K 分组、逐行、逐输出通道及整张量 scale；numeric/E8M0 显式区分 |

统一实现按共同 K 区间执行 dot 后乘 scale 并累加，不要求 tl.dot_scaled。
Excel 缺失的 scale 信息按公开模型/框架配置补齐并标“网补”，不统一假设 E8M0。2026-10-23 核对发现，GLM/Qwen/DeepSeek 的公开 block-FP8 权重为 N128/K128，Hy3 为 per-tensor，不能统一解释成 N1/K128。以 Excel 明确字段优先；Q35 MXFP4 的 A 保留 FP8，Hy3 的 A 保留 Excel 逐行布局，缺失字段采用网补。静态 scalar 可由框架复制为 [M,1]，不能据 shape 判定动态/静态冲突。checkpoint dtype 与框架转换路径分开记录，详见 [模型配置与冲突](SHAOBO_BU_REQUIREMENTS.md#requirements-cross-check)。此前 G171 的 FP32 A + E8M0/N1 B 性能结果只适用于该实验组合。

FP8×FP4 接口使用预量化 A 与低/高半字节打包的 E2M1 W（偶数 K 取低半字节）；A scale 按 K32 或 K128，W 按 N1/K32。当前实现以 `tl.dot_scaled(a, None, "e4m3", b, None, "e2m1")` 执行每 K32 的原生 F32 混合 MMAC，再乘 A/W scale 并以 FP32 累加，按输出 dtype 写回。gfx946 需要配套的 Triton K-packed FP4 布局/寄存器格式修复；2026-10-21 PMD 验证为 19/19 接口回归及 10/10 新旧实现对照通过，尚不代表完整需求尺寸或实卡验收。实现、环境依赖及证据见 [原生 FP8×FP4 K32 修复](debug_tuning/FP8FP4_NATIVE_K32.md)。不包含在线量化或路由生成。

## 优化状态与后续范围

- FP16 单 TG block 流水的当前最佳配置、资源与复测证据见 [2026-10-27 round148 配置地图](debug_tuning/fp16_single_tg_20261027_round148/README.md) 与配套 JSON；按 BM/BN/BK 查找，供 BU 调优起步使用。历史快照：[round139](debug_tuning/fp16_single_tg_20261027_round139/README.md)、[round119](debug_tuning/fp16_single_tg_20261027_round119/README.md)、[round113](debug_tuning/fp16_single_tg_20261027_round113/README.md)、[round108](debug_tuning/fp16_single_tg_20261027_round108/README.md)、[round103](debug_tuning/fp16_single_tg_20261027_round103/README.md)、[round100](debug_tuning/fp16_single_tg_20261027_round100/README.md)、[round93](debug_tuning/fp16_single_tg_20261026_round93/README.md)、[round88](debug_tuning/fp16_single_tg_20261026_round88/README.md)、[round84](debug_tuning/fp16_single_tg_20261026_round84/README.md)、[round82](debug_tuning/fp16_single_tg_20261026_round82/README.md)、[round76](debug_tuning/fp16_single_tg_20261026_round76/README.md)、[round71](debug_tuning/fp16_single_tg_20261026_round71/README.md)、[round62](debug_tuning/fp16_single_tg_20261026/README.md)、[2026-10-23](debug_tuning/fp16_single_tg_20261023/README.md)。

- WDRA48 的 post-dot scale 分区依赖断言已修复，Triton 同名分支提交 `e33b79cf1`；34 项运行检查 PASS，4 项同 BK 普通流水对照逐元素一致。大 tile 仍受 LDS / VGPR 限制，尚无性能晋升结论。
- FP8×FP8 已建立对齐基线及 async / BPP 候选。下一步先调通少量关键流水，再上真实卡 autotune；不在 PMD 对 207 个需求形状展开全量搜索。
- Track A 的 FP8×FP4 stages 收益依赖 tile；Track B 的 FP16/BF16 已有 54 个单 TG 成本点，不能以单 TG 原始 cycles 直接决定完整 GEMM 配置。
- 生产 kernel 与默认配置尚未采用上述实验流水。旧 MR !122 已关闭，后续在统一优化分支推进。

关键根因、修复边界和环境能力缺口见 [debug 记录](debug_tuning/DEBUG.md)；有效性能数据、资源规则和候选计划见 [调优记录](debug_tuning/TUNING.md)。旧逐日记录不再维护，可从 Git 历史追溯。脚本、原始日志、StatLog、IR/汇编及大型产物均留在仓库外；不恢复暂停任务。
