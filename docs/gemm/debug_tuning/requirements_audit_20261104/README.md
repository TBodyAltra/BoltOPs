# 新版少伯 Excel 需求更新核对

日期：2026-11-04。原文档基线 da99aa344f06a70e40bef1280928cab402ef15df。

## 更新口径

- 用户指定 P0：Qwen3.5-35B-A3B FP8、Qwen3.5-35B-A3B MXFP4、Qwen3-VL-8B BF16，按 X09/X10/X14 来源标记全部关联需求；其他模型未指定。主表 74 条 P0，未编号 grouped 另有 6 条；共享 G342/G347 不重复计数。顶部快速索引包含各类首条入口与可展开的完整编号跳转，表格新增优先级列。
- 新版 `shaobob0_shapes_1102` 的 14 份 XLSX、39 个 sheet、2526 条 GEMM/BMM/grouped 来源全部读取；忽略 Excel 的 `~$` 临时锁文件，不修改原工作簿。
- 原 469 个 G 编号不重排，原 MNK/E、阶段、频次、模型引用、接口引用与历史状态保留。G436–G471 的历史空洞不复用。
- 原 FP8×FP4 的 G472–G505 作为兼容需求保留；A/B scale 按新版统一 E8M0/K32，激活动态。新版对应来源的 A dtype 是 FP4，兼容项的 FP8 不冒充新版数据。
- 新增 FP4×FP4 grouped G506–G539；新增 Qwen3-VL BF16 linear G540–G559，C020/X14。dense FP4 参考接口不等于已有 grouped 适配，新增需求全部待验证。
- 207 个 FP8 dense 的 A scale 分为 FP32/K128 181 行、FP32/token 25 行、混合 1 行；B scale 采用新标签，混合模型按 X 来源配对。DeepSeek dense A 不再使用旧 E8M0；Hy3 不再采用旧 scalar/static 补充值。
- Kimi EAGLE3 与 grouped 的 g32/BF16* 依新标签记录；FP32*/BF16* 星号保留，packing/zero-point 等不擅自补写。旧公开采样快照仅作历史说明。
- 489 个新版已编号需求覆盖 2446 条来源，42 个未编号 grouped 缺口覆盖 80 条，合计恰为 2526；额外保留的 34 个 FP8×FP4 兼容项复用 304 条谱系，不作为新版新增来源计数。

## 尚需明确

18 处同名 quantize_o_proj 与 o_proj 的 shape 候选连线仍不一致，见 named-link-issues.csv；无 tensor ID，不能直接认定实际程序连线或擅改 MNK。grouped 仍缺路由/复制/padding/offsets 与 scale 的对应说明。
248 条来源带 FP32*/BF16*，Excel 未解释星号。scale 物理布局、FP4/INT4 packing/nibble 顺序、INT4 zero-point/对称性，以及部署 checkpoint/框架 revision 均未由本表完全确定。
本次没有运行 GPU 测试或新增任何 smoke/完整尺寸/性能 PASS；历史状态仅按原接口证据保留。

## 可追溯文件与复现

- source-manifest.json：14 份新版 XLSX 的 SHA256、X 编号、sheet 描述与部署原文。
- gemm-sources.csv：2526 条实际来源，含新版行号、stage、layer_count、原始 shape/dtype 与量化标签。
- case-provenance.csv：523 条支持需求，含兼容标志、来源 dtype、成对 G 编号、逐来源 act_quant/weight_quant；可区分支持 dtype 与新版源 dtype。
- grouped-gaps.csv：42 个未编号 grouped shape，采用新版量化字段。
- named-link-issues.csv：18 个 dense 命名连线候选问题。
- verification.json：计数、旧编号不变性、当前来源恰好一次覆盖、dtype/量化标签、scale 分组与锚点检查。
- refresh.py：使用 openpyxl 从本地两版 XLSX 重新生成。基线原文从上述 Git commit 读取，旧版 XLSX 的 SHA256 与 2026-10-23 历史审计逐个校验。

```bash
python docs/gemm/debug_tuning/requirements_audit_20261104/refresh.py \
  --excel-dir /mnt/c/Users/xukang/Downloads/shaobob0_shapes_1102 \
  --old-excel-dir /mnt/c/Users/xukang/Downloads/shaobob0_shapes
```

未把原始 XLSX、模型权重或 GPU 产物加入 Git。2026-10-23 的历史审计文件保留不变。
