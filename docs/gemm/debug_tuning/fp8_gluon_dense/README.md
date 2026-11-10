# FP8×FP8 FP32 block-scale Gluon dense GEMM

公开入口：`from boltops.gemm import gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16`。
名称中 `a1x128` 表示 A scale 的 M1/K128，`b128x128` 表示 B scale 的 N128/K128；
`fp32scale` 表示两个 scale 都是 FP32，`to_bf16` 表示输出 BF16。
依赖 Triton !520 的 HCU Gluon API，面向 gfx946；源码正常编译，不修改生成的汇编。

```python
c = gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16(
    a, b, a_scale, b_scale, scale_recipe=((1, 128), (128, 128))
)
```

| 参数 | 约定 |
|---|---|
| A | E4M3FN，连续 `[M,K]` |
| B | E4M3FN，连续 `[N,K]`，计算 A @ B.T |
| A scale | FP32，`[M,ceil(K/128)]`；每行/K128 |
| B scale | FP32，`[ceil(N/128),ceil(K/128)]`；N128/K128 |
| scale 存储 | K-group 连续；尾组只缩短实际参与的 M/N/K 范围 |
| 输出 | BF16，`[M,N]`；每K128做FP32 dot，再乘scale并FP32累加 |
| MNK | 任意非负值；M/N为0返回空输出，K为0输出全零 |
| 设备 | 四个输入须在同一gfx946设备；错误shape/dtype/非连续输入明确报错 |

接口在launch前检查recipe、每个输入的精确dtype和scale shape，其他recipe/精度直接抛出
`ValueError`，不做隐式转换。recipe是调用方对scale含义的声明，接口不能从数值反推生成方式。

## 调度与资源

- M为128倍数、N为256倍数、K为256倍数且256≤K≤6144，A/B基址16字节对齐：
  使用已调优的 MLS 流水，grid=`(M/128,N/256)`，每CTA处理一个128×256输出tile。
  A/B和scale均按CTA坐标偏移，C采用全矩阵N作为行stride。
- 其他尺寸：64×128×128的 masked Gluon MMAC 路径，grid=`(ceil(M/64),ceil(N/128))`。
  A/B逐元素mask并将越界FP8位模式补零，输出按M/N mask；scale按真实组编号读取。
  仍执行原生FP8 MMAC，没有量化替代或CPU计算回退。
- 当前MLS路径保留immutable scale LDS切片，因此K增大时LDS也增大；K6144为120.375 KiB。
  长K路径逐组处理，资源不随总K线性增长。长K/尾块路径尚未做性能调优。
- API不改变既有算子的dispatch/default config；不支持其他scale粒度、FP4、grouped或batch。

## 布局与标准操作

MLS B 使用 `alt=1`。寄存器排列由 `MmacInterleave2Layout` 描述：
它包装已有的 MMAC 指令布局 F，或包装 B 的 `DotOperandLayout(1,F,8)`。
A 保留普通 DotOperandLayout；B/C 的 N 交错映射由类型表示。

```python
from triton.experimental.gluon.language.amd.hcu import MmacInterleave2Layout

# F 是 LTS AMDMFMALayout，指令 16x16x32、FP32 累加、每 warp 配对 N tiles。
b_layout = MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8))
c_layout = MmacInterleave2Layout(F)
b = async_copy.load_shared_relaxed(smem_b, b_layout)
acc = gl.full((64, 256), 0, gl.float32, c_layout)
dot = gl.amd.cdna4.mfma(a, b, acc)
```

上述调用产生现有 `ttg.local_load` 和 `tt.dot`，不再提供独立的 interleave2 load/dot
函数或 IR 操作。布局类型保留基础 MMAC 几何，因此 matrix_store 从 value 类型推导，
不需要额外的 `mmac_layout` 提示参数。普通 global load/store、scale 运算也使用同一数据映射。
现有 AMDMfma 属性没有增加 nInterleave 字段；普通布局及既有 TL 路径保留原有语义。

## 验证与性能口径

测试：`test/gemm/gluon/test_gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16.py`。
覆盖最小尺寸、M/N/K尾块、M/N分别及二维多CTA、不同K、K8192、空维度。
参考值由CPU生成，核心MMAC在PMD执行，未修改生成汇编。

本次 31/31 接口测试、30/30 Triton 原语/布局测试、4/4 既有 TL 编译回归通过。
原语测试分别检查 MLS 读入、普通 global load、标准 mfma、普通/matrix store，
避免读写错位互相抵消；还检查 B/C 布局错配拒绝、A 不允许包装、布局身份与往返转换。

环境：PMD CoreArch HEAD_1703，gfx946/sb，单 die 48 CU，shader/cache 周期 455 ps。
LLVM 96fd797af621f19f2ce33dae35027dce185d1bb8，TritonRelBuildWithAsserts，
`GPU_CHIP=sb`、`GPU_ARGS=['--abs-max-tick=1000000000']`、`GPU_DFLAGS=['StatLog']`，
OMP/MKL/OPENBLAS 线程均为 1。使用隔离编译库，生产库不替换。

M128/N256/K6144 自然单 CTA，同一进程交错运行历史保存二进制及新布局实现，
各预热一次，再各计时三次：

| 版本 | cycles 三次 | 平均 cycles | VALU | MMOP | LDS 类指令 |
|---|---|---:|---:|---:|---:|
| 历史保存二进制 | 78988 / 78697 / 79101 | 78928.67 | 28499 | 24576 | 9632 |
| 布局类型 + 标准 load/dot | 78697 / 79017 / 78693 | 78802.33 | 28499 | 24576 | 9632 |

差异 -0.16%，在本次 0.41%～0.51% 极差/均值抖动内，不宣称提速。
提取的 2196 条机器指令序列一致，ds_bpermute 为 0。8 次输出与 CPU 逐元素一致
（max_abs=0、mismatch=0），spill=0，LDS=123264 bytes（120.375 KiB）。
四分区 warps=(4,4,4,4)，寄存器分配配置=(8,8,240,240)，tile/config 未调整。
宿主 hostSeconds 约 9.6 秒，仅供预算；性能以 PMD modeled cycles 为准。

Triton 实现提交：`fe795e799cdd9e704075c0e34e462c912c61cb03`（!520）。
公开接口及配套调用见本记录所在 BoltOps 提交（!151）；目标基线 gemmOptimization 652c399。
私有 libtriton SHA256：`71a6f09132839e4d5b265c24f358de0ff397304599f6cb22e391cdc7a842a5c7`。

原始 IR、汇编、统计和编译日志均在仓库外：
`/xukang/code/pmd-triton-sessions/gluon-mr-20261029/layout-redesign-20261102/`。
配对 StatLog 在同级 `check-layout-redesign-paired/`，接口结果在 `layout-redesign-dense-20261102/`。
PMD 既有 src1/read-before-writing 警告保留在日志中；本次精度检查通过。
结果仅覆盖单 die PMD，实卡及 2/4 die 性能尚未验收。
