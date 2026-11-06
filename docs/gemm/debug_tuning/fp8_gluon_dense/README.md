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
统一使用 `gemm_block_scaled`，不保留旧长名称入口。

## 调度

- E4M3FN×E4M3FN、FP32 scale、默认 recipe、BF16 输出继续使用既有优化路径。
  M 为128倍数、N为256倍数、K为256倍数且256≤K≤6144、A/B基址16字节对齐时，
  使用 128×256 MLS 流水；其余尺寸使用原 masked MMAC 路径。
- 其他 dtype/recipe/output 使用 32×64 masked 原生 MMAC 路径，处理尾块和独立 scale 分组。
  此通用路径尚未调优。现有算子 dispatch、默认 config 和已优化流水保持不变。

## 实现组织

- `gluon/gemm_block_scaled.py` 负责公开接口、shape/scale 校验和空矩阵处理。
- `gluon/_block_scaled_dispatch.py` 根据 MNK、dtype、recipe 和对齐条件选择流水及合法 tile 配置；
  目前保留既有选择条件，不加载外部调优文件，也不新增架构支持声明。
- `gluon/pipelines/block_scaled_mls_wdra48_128x256x128.py` 是固定 128×256×128 的优化实现，
  tile、LDS、partition 和寄存器预算作为一套流水维护。其他 tile 流水应注册为独立实现，
  不能仅修改此实现的 launch 参数来替换 tile。
- `gluon/pipelines/block_scaled_masked.py` 包含默认 FP8 recipe 和通用 recipe 的 masked 实现。

## Native ebarrier API migration (2026-11-05)

The optimized kernel now uses `ebarrier_arrive` / `ebarrier_sync` from
`triton.experimental.gluon.language.amd.hcu` for initialization and consumer
MMAC handshakes. Barrier IDs, wave counts and the priority reset are preserved.
This requires Triton !520 including interface commit `cb890751` (tested head
`01203446`); older Triton packages do not expose these APIs.

Single-CTA comparison: M/N/K=128/256/6144, tile=128/256/128, stages=2,
partition waves=(4,4,4,4), gfx946, one die / 48 CU, clock period 455 ps
(~2.198 GHz), LLVM `96fd797af621f19f2ce33dae35027dce185d1bb8`.
The isolated compiler validation base is `97e683db` with the current API changes.
Each version has one untimed launch and three interleaved measured launches;
JIT compilation is excluded from modeled cycles.

| Metric | Inline assembly baseline | Native API |
|---|---:|---:|
| Modeled cycles | 78,619.33 | 78,287.33 |
| LDS bytes | 123,264 | 123,264 |
| Spill | 0 | 0 |
| VALU / MMOP / LDS / VMEM | 28,499 / 24,576 / 9,632 / 1,328 | same |

The -0.42% difference is within the observed 0.60% sample range: no measurable
regression, without claiming a stable speedup. K=256 and K=6144 outputs match
both the original kernel and CPU reference elementwise (max_abs=0, mismatch=0).
Full options, environment, IR, assembly and PMD statistics remain outside Git:
`/xukang/code/pmd-triton-sessions/gluon-mr-20261029/ebarrier-boltops-20261105/`.

## Triton 布局设计

`AMDMFMALayout(..., mmac_layout=MmacLayout.LTS)` 使用必填显式模式。
`MmacLayout` 包含 `LEGACY`、`LIT`、`LTS`、`LIT_LTS`，从 `gl.amd` 导出。
删除 `transposed` 参数及自动映射，不保留旧参数兼容层；普通 AMD MFMA 目标不在支持范围内。
不再单独定义 `HcuMmacLayout`。IR 往返保留同一类型和实际模式，无需在 tensor 构造时规范化。
公共 tensor 构造钩子、`_with_ir_layout` 和辅助 IR 查询绑定已删除，通用 expand_dims 保持原实现。
这只调整前端表达，不增加 IR 布局属性、不改变显式选择后的硬件模式。

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

## 显式 MMAC 布局及迁移（2026-11-04）

HCU kernel 必须显式选择模式。迁移旧的自动映射：gfx938/gfx946 上
`transposed=True` 对应 `MmacLayout.LIT`，`False` 对应 `MmacLayout.LIT_LTS`；
已有 `mmac_layout=3` 的 GEMM 调用保留 LTS 并删除 `transposed`。
LEGACY 目标需显式指定 LEGACY，不会把 LIT/LTS 静默降级。

扫描远端 BoltOps main `e4c2c5848f1123118d3096a73b51e004289c8b80`：3 处 MQA 调用受影响；
gemmOptimization `da99aa344f06a70e40bef1280928cab402ef15df`：相同 3 处 MQA + 2 处 GEMM 调用受影响。
当前 MR 已迁移 MQA 和优化/通用 GEMM；两个目标分支的独立迁移补丁保存在仓库外，尚未直接修改目标分支。

Triton 测试扫描到 50 处旧参数调用（8 个文件），已迁移。MFMA 前端快照改为 HCU 目标；
普通 AMD 目标不再作为支持兼容性承诺。WMMA/NVIDIA 的 transposed 参数与本次 MMAC 参数无关，保留不动。

此前独立发现的 LIT reduction 数值问题仍未修复，旧库也能复现；本轮 reduction 检查仅覆盖类型推导，不能据此宣称数值通过。
产物与完整扫描清单：`/xukang/code/pmd-triton-sessions/gluon-mr-20261029/explicit-mmac-20261104/`；
本地分支迁移补丁：`/home/kxu/gluon-mr-20261029/explicit-mmac-20261104/boltops-{main,gemmOptimization}-migration.patch`。

本轮验证：71 项 Gluon 布局/流水检查、6 项 HCU 前端检查、41 项 Gluon scaled 编译、
4 项既有 TL 编译、42 项 BoltOps GEMM 接口测试通过。MQA 变更完成静态迁移检查，未宣称完整 MQA 数值回归通过。
单 CTA M128/N256/K6144，旧 B16 二进制平均 78607 cycles，新版 78501 cycles（-0.13%，抖动范围）。
8 次输出 max_abs=0、mismatch=0、spill=0；与上版提取的指令序列一致。
环境仍为 gfx946/sb、单 die 48 CU、455ps shader/cache 时钟；LLVM 96fd797，编译参数/环境同前。
本轮 Triton 提交 97e683db29abed7e6b82cb01e4521b7e63379dba；完整编译日志、环境脚本和配对结果在上述仓库外目录。
