# 新版少伯 Excel 需求更新核对

日期：2026-11-04。原文档基线 da99aa344f06a70e40bef1280928cab402ef15df。

## 更新口径

- 用户指定业务一级、模型二级：12 月 BU 性能测试为 P0，包含 Qwen3.5-35B-A3B FP8、Qwen3.5-35B-A3B MXFP4、Qwen3-VL-8B BF16；其余模型归后续支持 P2。全局 P0 为 74 个 G 编号 + 6 个 grouped 缺口，P2 为 449 个 G 编号 + 36 个缺口。主文档按 9 个模型各自分表，每个模型中的每组需求使用独立的 R 编号（其中少伯需求 359 组；合入字节 40 组后为 R001–R399）。旧 G/缺口与来源对应关系仅保存在审计数据中，正文不保留历史编号附录。当前展示范围排除 MoE grouped GEMM、fused_moe_gate、shared_gate_up_proj / shared_down_proj；attention、普通 dense MLP 和 EAGLE3 保留。混合来源只删除 MoE 来源，原始全量数据仍保存在审计 CSV 中。
- 2026-11-05 增加 1 月字节送测 P1（第 72、86 行 dense GEMM），原后续支持调整为 P2；详见 ../byte_january_20261105/README.md。
- 2026-11-06 迁移 GitHub gemm 分支 PR；新增 INT8/BF16 输出和缺省 FP32 token/输出通道 scale 口径，MX 标准约定保留。GitHub 实际渲染检查见 rendered-anchors-github-20261106.json；带星号的 scale 用 code 字面显示以免被识别为账号提及。
- 2026-11-05 修正 GitLab 锚点兼容性：自定义锚点使用 `<a name="...">`，而不是会被该实例清洗掉的 HTML `id` 属性。需求组 R 编号与业务、模型索引均有明确的 named target；实际 GitLab 渲染核验见 rendered-anchors-20261105.json。
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
- model-requirements.csv：业务/模型表的逐项投影，含 P0/P1/P2、独立 R 编号、各模型自己的来源、scale、频次和算子名；legacy_row_id / legacy_G 仅供内部来源核对。每条原始来源恰好归属一个模型；兼容来源复用单独标记。
- business_model_view.py：业务与模型视图生成器，与全局数据提取/核对分离。
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
