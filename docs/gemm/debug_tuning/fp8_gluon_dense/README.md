# Gluon block-scaled dense GEMM

公开入口：`from boltops.gemm import gemm_block_scaled`。依赖 Triton !520 的 HCU Gluon API。
输入为连续的 A[M,K]、B[N,K]，计算 A @ B.T；四个输入须位于同一 GPU。

```python
c = gemm_block_scaled(
    a, b, a_scale, b_scale,
    scale_recipe=((1, 128), (128, 128)), out_dtype=torch.bfloat16,
)
```

`scale_recipe=((A_M,A_K),(B_N,B_K))` 接受两组正整数，A/B 的 K 分组可以不同。
scale shape 分别为 `[ceil(M/A_M),ceil(K/A_K)]` 和 `[ceil(N/B_N),ceil(K/B_K)]`，
K-group 连续。scale 中保存数值：uint8 按普通整数解释，不隐式解码成 E8M0。
通用路径按 `gcd(A_K,B_K)` 划分共同区间，各区间执行原生 MMAC 后乘 scale 并累加；
累加和 scale 运算采用 FP32，FP64 输入或 scale 对应的运算提升为 FP64。
区间划分不同可能改变浮点舍入顺序。

接口不限定 E4M3FN、FP32 scale、固定 recipe 或 BF16 输出；输出通过 `out_dtype`
指定，默认 BF16。实际输入/累加组合和指令几何由目标 MMAC 指令表判断，非法组合明确报错，
不做输入上转型或软件模拟替代。不意味着 PMD 已支持所有硬件指令。
M/N/K 可为非负值；空 M/N 返回空矩阵，K=0 返回零。
旧长名称入口保留兼容，转发到此接口，仍保留原默认参数；新调用应使用通用名称。

## 调度

- E4M3FN×E4M3FN、FP32 scale、默认 recipe、BF16 输出继续使用既有优化路径。
  M 为128倍数、N为256倍数、K为256倍数且256≤K≤6144、A/B基址16字节对齐时，
  使用 128×256 MLS 流水；其余尺寸使用原 masked MMAC 路径。
- 其他 dtype/recipe/output 使用 32×64 masked 原生 MMAC 路径，处理尾块和独立 scale 分组。
  此通用路径尚未调优。现有算子 dispatch、默认 config 和已优化流水保持不变。

## Triton 布局设计

自动布局继续使用原有 `AMDMFMALayout` 和 `transposed` 参数，不增加可选 `mmac_layout`。
纯 LTS 使用 HCU 的 `MmacLtsLayout`（几何字段沿用 AMD 布局，`transposed=True`）。
IR 返回 Python 时，自动映射模式恢复普通布局，纯 LTS 恢复独立类型；严格区分两者，
不再出现 `None` 与整数模式混用。通用 `expand_dims` 保持原实现，无绕行分支。
优化流水及通用原生 MMAC 路径均已迁移到新 LTS 类型；不新增 IR 属性或模式编号。

`MmacInterleave2Layout` 包装 B 的 `DotOperandLayout(1,F,k_width)` 或结果 F；A 保留普通布局。
要求 rank-2 LTS、每 warp 覆盖完整的32列交错组、B/C 的基础布局匹配。
32列是 MLS interleave2 的数据排列定义，不是固定 MMAC 几何。
布局校验不限定 dtype、16×16×32 或 k_width=8。

`get_mmac_instruction` 复用已有 `MfmaIntrinsic::selectFor` 查询实际类型/几何及 modifier 能力；
MMAC lowering 据所选指令检查 interleave2 的 K 整除性。
MLS load 据 operand fragment 检查 K 和存储位宽；支持已验证的8位 MLS-v3及16位 MLS-v2。
使用标准 local_load/dot/store，不改生成汇编，也没有新增 nInterleave 字段。

## 验证

小尺寸覆盖 E4M3FN、E5M2及二者混合、FP16、BF16、FP32 输入；
FP16/BF16/FP32及普通uint8 scale，独立 M/N/K scale 块、不等 K 分组、长 scale 块、尾块、空维度。
非法混合 FP16×BF16 和目标不支持的 FNUZ 输入由编译能力校验拒绝。
精度测试采用 CPU oracle，实际 kernel 在 PMD 执行；测试不会把 CPU 计算作为实现回退。

FP8 输出的独立探针停于 `s_setreg_imm32_b32 / unimplemented GCN3 instruction`：
PMD 未实现 HW_REG_MODE2 写入，记为 `BLOCKED_PMD_UNSUPPORTED`，没有软件绕过。
FP8 scale 读入使用原始字节 bitcast 和原生转换，E4M3FN/E5M2 scale 的精度测试已通过。

## 性能口径与产物

PMD CoreArch HEAD_1703，gfx946/sb，单 die 48 CU，shader/cache 周期455 ps（约2.198 GHz）。
LLVM `96fd797af621f19f2ce33dae35027dce185d1bb8`，TritonRelBuildWithAsserts。
`GPU_CHIP=sb`、`GPU_ARGS=['--abs-max-tick=1000000000']`、`GPU_DFLAGS=['StatLog']`，
OMP/MKL/OPENBLAS 线程均为1，unset HSA_TOOLS_LIB；隔离 runtime，不替换生产编译库。

2026-11-04 移除 `byte_store` 特殊路径，优化kernel使用 `matrix_store_32x32_b16`。
前端参数、IR属性、B8替代指令及stride/offset倍增逻辑均删除；按结果元素位宽选择写出指令。

默认 recipe 的 M128/N256/K6144 自然单 CTA，同输入、同进程交错预热后各测3次：

| 版本 | modeled cycles 三次 | 平均 cycles | VALU / MMOP / LDS / VMEM |
|---|---|---:|---|
| 保存的B8二进制 | 79005 / 78588 / 79093 | 78895.33 | 28499 / 24576 / 9632 / 1360 |
| B16当前编译结果 | 78417 / 78658 / 78429 | 78501.33 | 28499 / 24576 / 9632 / 1328 |

平均差异 -0.50%，样本范围重叠，不宣称稳定提速；本次未观察到回退。
8次输出均 max_abs=0、mismatch=0，spill=0。PMD hostSeconds均值9.61/9.63仅供预算。
验证通过：46项Gluon原语/布局测试、4项既有TL编译回归、2项dense单/多CTA数值回归。
Triton提交 `7b54354c2e17617c45706efbed70fa70b5935819`；BoltOps提交见本记录所属commit。
私有libtriton SHA256 `9150c2fd792b3feaa0101a43f6f2c551ebd6705355f50cecc7a4ef7f5f27c4f2`。
四分区warps=(4,4,4,4)、寄存器分配=(8,8,240,240)。
日志仍有PMD src1修饰位与packed FMA寄存器读取诊断，完整文本保留，数值测试通过。
本轮证据：`/xukang/code/pmd-triton-sessions/gluon-mr-20261029/remove-byte-store-20261104/`。

仅验收单 die PMD；实卡和2/4 die尚未验收，通用路径不做性能承诺。

原始脚本、IR、汇编、完整错误和统计在仓库外：
`/xukang/code/pmd-triton-sessions/gluon-mr-20261029/generalize-20261102/`。
配对 StatLog 位于同级 `check-generalize-paired/`；通用接口结果在 `generalize-dense-20261102/`。
前置接口 MR（已合入）：[BoltOps !151](http://42.228.13.241:10068/dcutoolkit/deeplearing/BoltOPs/-/merge_requests/151)。
编译器 MR：[Triton !520](http://42.228.13.241:10068/dcutoolkit/deeplearing/triton/-/merge_requests/520)。

## MLS 接口范围收缩（2026-11-04）

Gluon matrix store 只接受显式 interleave2 结果布局，version=3、tile=(32,32)、order=(1,0)，每warp覆盖完整M tile且N方向只有一组64列。普通MMAC布局不再自动转换。MLS interleave2 B读取同样要求显式interleave2 operand布局。原有Triton带marker路径不变。删除普通/native自动重排和扩展M/N分组地址映射，复用既有地址计算。

当前FP8流水无需修改kernel。50项Gluon测试（含不支持组合的拒绝检查）、4项TL编译回归、2项dense单/多CTA测试通过。M128/N256/K6144同进程交错各测3次：收缩前B16为78679/78505/78626（均值78603.33），收缩后78444/78641/78389（均值78491.33），差异-0.14%，在抖动内。2169条提取的机器指令序列一致；VALU/MMOP/LDS/VMEM均为28499/24576/9632/1328；8次输出max_abs=0/mismatch=0/spill=0。

证据：`/xukang/code/pmd-triton-sessions/gluon-mr-20261029/narrow-mls-20261104/`。本轮Triton提交 `b1d868f105d95c7a1111205c4951d035610136d4`。环境与上节相同；暂停进程检查通过。

## LTS 布局表示重设计（2026-11-04）

删除 AMDMFMALayout 可选 mmac_layout 及 expand_dims 的 parent 绕行；普通自动布局往返保持原表示。纯 LTS 由独立 MmacLtsLayout 表达并由 IR 返回同类型，不增加IR属性、不放宽相等比较。几何参数继承既有AMD布局，transposed须为True。普通类型与LTS类型的eq/hash/mangle均区分；不提供已撤回可选参数的兼容别名。

53项Gluon原语/布局/跨目标编译检查、4项TL编译、42项BoltOps接口测试通过。包括gfx938/gfx950普通布局往返、gfx938 LTS往返及gfx950拒绝LTS。恢复原expand_dims后广播数值测试通过；reduction后的布局往返做编译检查。

M128/N256/K6144同输入同进程、每版预热一次后交错各测3次：保存B16二进制78674/78437/78633（均值78581.33），新类型78425/78621/78373（78473），差异-0.14%，在抖动范围。2169条提取机器指令序列一致；8次max_abs=0/mismatch=0/spill=0；VALU/MMOP/LDS/VMEM仍28499/24576/9632/1328。

额外发现：自动LIT布局的行归约数值异常，在保存的旧隔离库及旧显式LIT接口上同样复现，未计PASS，未在本轮修复。原始失败及control.log保存在本轮证据目录。CI所报pa_mqa只有失败名称，尚未独立确认原因。本轮没有宣称原CI两个测试均已实卡通过。

证据：`/xukang/code/pmd-triton-sessions/gluon-mr-20261029/lts-layout-20261104/`；Triton `adda207a271ddaa8d4009166198322d8fb791a03`。环境同上；暂停进程检查通过。
