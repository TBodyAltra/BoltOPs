# GEMM 关键 debug 结论

截至 2026-10-20；当前支持入口见 [README](../README.md)。这里只保留可复用的根因、修复与验收边界。所有下列相对产物路径均以仓库外 `/xukang/code/pmd-triton-sessions/` 为前缀。

## FP8 scale 解码与分组

uint8 E8M0 不能直接转成整数数值：code 127 应为 1.0；code 0 为 2^-127，255 为 NaN。共享 helper 仅对 FP8 操作数的 uint8 scale 解码，浮点 scale 与非 FP8 路径保持原语义。普通、预排布和 grouped regular/combined/persistent 路径已有回归。

解码及相关回归 24/24 PASS，包含 256 个编码逐位检查。G034（6×128×6144，A E8M0/K128、W FP32/K128）从 767/768 mismatch 修复为 0，max_abs=0。grouped FP8 的 BK 只能受 scale K 粒度限制，不能被 N 粒度 1 裁成 BK=1；不新增 grouped split-K 支持。证据：产物仓库 `triton_gemm_asm/requirements/20261017-fp8-e8m0-fix/`。


## S03 FP32 精度

`matmul_persistent` FP32 64³ 曾有 max_abs=0.5782、2906/4096 元素超限。原先强制 LEGACY 只是临时回退，现已撤掉 FP32 部分；保留原 FP64 保护。

实际根因是 FP32 `16×16×4` intrinsic 只有 LTS 标志、没有 LIT 标志，但默认选择 INTERLEAVE 编码，导致寄存器映射与指令不一致。仅撤掉 LEGACY 回退仍复现原错误；按所选 intrinsic 的 LIT/LTS 能力自动选择 TRANSPOSE 后，汇编生成真实 `v_mmac_16x16x4_f32 ... lts`，S03 相对 FP64 golden max_abs≈1.05e-7、4096 元素无超限。FP32 支持 LTS，不能把此前失败解释为硬件不支持。

Triton 当前提交 `d8790a0de3289b5d65de505b8e62394a8fccdd59`。共享能力查询供选择与 lowering 两处使用：自动模式只选受支持布局；显式 LEGACY/LTS 按请求执行；显式 LIT/LIT+LTS 对该指令报错，不静默替换。lowering 也拒绝预编码的非法布局。原 FP64 LEGACY 保护保留。

6 项选择检查、1 项直接 lowering 拒绝检查均通过；已有类型选择回归与 FP16/BF16/FP8 强制 LIT 编译对照通过。PMD FP32 64³ 和 48×80×97 均无超限，后者 max_abs≈2.09e-7。产物位于 `fp32-layout-capabilities-20261020/`，私有库不覆盖原全局库。

## native FP4 构建修复

S34–S37 的 LLVM IR 转换遇到 `rocdl.ds.scale.copy.ds2buf`：链接的 ROCDL translation archive 陈旧，已有 lowering / intrinsic / 指令实现未进入库。重编私有 translation 对象并重链接后，4 项小尺寸 PASS，原生 MMAC / scale-copy 汇编已确认；没有用软件替代缺失指令。

证据：`fp8-live-records-20261018/blocked-diagnosis-20261018/REPORT.md` 与 `final-results.json/csv` 中的 S34–S37。

## PMD 指令能力边界

在线 FP8 转换 `value.to(float8_e4m3fn)` 曾生成写入 `hwreg(HW_REG_MODE2,0,3),7` 的 `s_setreg_imm32_b32`，PMD 报 `Encountered unimplemented GCN3 instruction: s_setreg_imm32_b32`。这属于环境缺指令；禁止软件替代后计 PASS。grouped GEMM 使用调用方提供的预量化输入和 scale，不执行在线量化，不把该转换的失败作为 grouped GEMM 的验收结果。

## WDRA48 post-dot scale 调度修复

2026-10-20，原`RewritePartitionDependencies.cpp:219 / dist > 0`断言已修复。
这是Triton编译器修复，BoltOps算子源码和默认配置未改。

源码位于Triton的同名[perf/gemm-optimization分支](http://42.228.13.241:10068/dcutoolkit/deeplearing/triton/-/tree/perf/gemm-optimization)，
修复提交[e33b79cf1](http://42.228.13.241:10068/dcutoolkit/deeplearing/triton/-/commit/e33b79cf13535e983c1f0daacc48e092495410a8)。
前一个提交7378080a3保留既有S03 LEGACY基线。本任务未创建或合并MR。

### 原因与改动

分区器先沿scale load传播users，把dot后的mul/add分配给root(stage0)，
后续MMA传播无法覆盖已有归属，形成consumer(stage1)的dot结果被stage0同迭代读取的反向依赖。
scale解码也被留在root，造成不正确的复制/同步。

PartitionScheduling.cpp现在先归属MMA users及其tensor依赖，使缩放、解码和累加留在consumer；
遇到已归属producer的loads则停止，再处理剩余load users。
没有删除断言、跳过scale、替代算术实现或放宽资源校验。
新增MLIR回归在旧版FileCheck失败、修复版通过。

### 验证结果

| 检查 | 结果 |
| --- | --- |
| FP8 WDRA48精度与同BK对照 | 24项PASS；包括两个内核、A/W FP32/E8M0组合、A整行scale、M/N/K尾块、小4CTA网格、K4096长循环 |
| 同BK普通流水配对 | 4/4逐元素一致，max_abs=0、mismatch=0 |
| WDRA44 / WASP44 | 两内核各1项，共4项PASS |
| 普通FP8路径 | 2项PASS |
| 原FP16 / BF16 WDRA48 | 各1项PASS |
| S03 FP32 64³ | PASS，max_abs约1.05e-7 |
| native FP4 16×16×64 | PASS，max_abs=0；先前ROCDL translation修复保留 |

共34项runtime检查PASS。CPU golden门限atol=.01/rtol=.01；BK32长K两项相对CPU
BF16参考max_abs=.015625，0超限元素，且与同BK普通流水完全一致。不能把CPU差异写成全部为0。
IR/LLIR/汇编确认真实warp_specialize、WDRA初始化、MMAC和abarrier，没有静默降回普通路径。

9条相关HCU RUN检查通过，其中包含现有静态WASP/WDRA fixture检查。
另外一项NVIDIA partition-scheduling测试在修复前后均因既有yield分区期望不符而失败，
未作为本补丁引入的问题，也未修改该测试。

### 资源边界仍保留

原13个失败场景均越过原依赖断言；重放结果为4个编译通过、7个因LDS容量排除执行、
2个256×128/BK128停在后续VGPR quota校验。其自动quota=(308,96,100)，producer308超过256上限。

- 128×128/BK32：LDS50432 bytes，可执行。
- 128×64/BK64：LDS42112 bytes，可执行。
- 128×128/BK64：LDS66816 bytes，超64KiB，未执行。
- 128×128/BK128：LDS99584 bytes，超64KiB，未执行。

首次精度验证中另有6项容量排除，不计PASS。
当前结论是依赖bug已修复且存在可执行的WDRA48配置，不是所有大tile已支持或性能已调优。
后续分析scale中转、布局、LDS及producer VGPR需求；不对207形状全量扫描。

## 编译器与复现入口

| 项目 | 当前值 |
| --- | --- |
| Triton 基线 | release/3.6.x `46f0bed3561364b63ba4046d9c8c0cb70380ea30` |
| 原 S03 临时提交（历史） | `7378080a3`；其中 FP32 LEGACY 回退已撤销 |
| WDRA 修复提交 | `e33b79cf13535e983c1f0daacc48e092495410a8`，`perf/gemm-optimization` |
| 当前编译器提交 | `d8790a0de3289b5d65de505b8e62394a8fccdd59`，保留 WDRA 修复，按指令能力选择与校验布局 |
| 已验证私有库 | `fp32-layout-capabilities-20261020/libtriton.layout-capabilities.so` |
| 私有库 SHA256 | `51827dd939eacec4339c83022ce5e0d520792ed91f9e2119b46b7dc75cacc841` |
| 原库 SHA256（未覆盖） | `169f9640add95d07a540ad1b20275bbec7fc66ffe5d85f0fbcd83a1efc5f033e` |
| 统一环境 | `gemm-optimization-20261020/env.sh`，选择独立 Python overlay 与缓存 |
| WDRA 完整证据 | `wdra48-fix-20261020/REPORT.md`、`final-results.json/csv`、`source-version.json`、`active-import-verification.json` |

原 `libtriton.so.prebak2` 与外部 `libtriton.baseline.so` 保留；其他工作区不覆盖。后续复现必须核对实际导入路径和库哈希。

## 仍有参考价值的其他回归

- 普通 FP16 加载的 K 尾部曾无 mask，读入相邻 NaN。已对 A/B 添加 remaining-K mask、越界填零；TT/TN 的 K64/96/128/192 共 8 项 PMD 回归 PASS。MLS 尾块的既有 xfail 不因此解除。
- wheel 曾漏打包 23 份 gfx946 调优 JSON；package-data 已补齐，安装产物的 CPU 配置检查通过，不代表实机执行通过。
- 历史 PMD 的 LLVM `043b33c45cfa` 在 256×256×4096 WDRA48/MLS 上导致 153781→157267 cycles（+2.27%），两版正确性均 PASS；4 个 nop 可恢复周期，具体调度机制未定。证据 `/xukang/code/pmd_perf_before_043b33c4/`、`pmd_perf_after_043b33c4/`。
- 同一历史 reproducer 移除 matrix-store 地址 bit exchange 后 153779→157167 cycles，waitLgkm 933→1458；实际写回映射相同，不能直接归因为地址布局错误。`s_nop 6` 未恢复，具体后端修复仍开放。证据 `/xukang/code/perf_bit_exchange_ab/`、`/xukang/code/m5out_b_rebuilt_32x16_256/`。这些历史版本数据不能冒充当前编译器基线。
