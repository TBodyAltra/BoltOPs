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
