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

## 验证与性能口径

测试：`test/gemm/gluon/test_gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16.py`。
覆盖最小尺寸、M/N/K尾块、M/N分别及二维多CTA、不同K、K8192、空维度。
参考值由CPU生成，核心MMAC全部在PMD执行；二进制精确输入/scale作逐元素严格比较。

环境为PMD CoreArch HEAD_1703、gfx946/sb、单die48CU、455ps/cycle。
当前回归：31/31 接口测试、18/18 Triton 原语测试、4/4 既有 TL 编译回归通过。
使用目标分支 cdf6827 对应的 LLVM 96fd797af621f19f2ce33dae35027dce185d1bb8
完成独立编译和上述验证；未替换生产编译器。

B 恢复 MLS `alt=1`。B/C 的逻辑映射由已有 `LinearEncoding` 表达，
`native_mmac_load` / `native_mmac` 复用 MLS/MMAC lowering，避免读入及写出边界的寄存器重排。
这些原语限定 E4M3FN 输入、FP32 累加及 LTS 配对 N tile，并验证映射一致性。
`matrix_store` 的 `mmac_layout` 只提供指令几何，数据映射仍由 tensor encoding 决定。
没有新增 `nInterleave` IR 字段或参数，普通 MMAC 和原有 TL matrix-store 路径保留。

K6144 单 TG（M128/N256）同进程配对测试，每版先预热，再交错运行三次：

| 版本 | cycles 三次 | 平均 cycles | VALU | MMOP | LDS 类指令 |
|---|---|---:|---:|---:|---:|
| 历史保存二进制 | 78674 / 79044 / 78577 | 78765.00 | 28499 | 24576 | 9632 |
| 回退版 alt=0 | 82235 / 81934 / 82202 | 82123.67 | 29587 | 24576 | 10656 |
| 修复版 alt=1 + LinearEncoding | 78747 / 79021 / 78744 | 78837.33 | 28499 | 24576 | 9632 |

修复版较回退版减少 4.00% 周期，与历史版差 0.09%，在本次抖动范围内。
12 次 launch 与 CPU 参考逐元素一致（max_abs=0、mismatch=0），spill=0，
LDS=120.375 KiB。修复版没有 `ds_bpermute_b32`；提取的机器指令序列与历史版一致。
回退版增加的 LDS 类计数主要是跨 lane shuffle，不能解释成额外 LDS 内存读取。
修复版 PMD hostSeconds 平均 9.867 秒，仅供预算，不作为性能指标。
此结果仅覆盖单 die PMD；未做实卡或 2/4 die 性能验收。

编译选项及启动环境：`TritonRelBuildWithAsserts`、LLVM96fd797a，
`GPU_CHIP=sb`、`GPU_ARGS=['--abs-max-tick=1000000000']`、
`GPU_DFLAGS=['StatLog']`，OMP/MKL/OPENBLAS 线程均为 1。
Triton 修复基线 4a3fa0823b681856ba0ee1c24da7890ddfab33f6，
BoltOps 修复基线 47e6dc89b8e48b25760871ec73e3f9983c207e58；最终修复见本记录所在提交及 Triton !520。
私有 libtriton SHA256：`403a97b33d3601a6285d0517b53c81208994a1e42367e4dc52f5f368a2152b64`。

原始统计、汇编、结果和脚本全部保存在仓库外：
`/xukang/code/pmd-triton-sessions/gluon-mr-20261029/native-linear-fix/`，
配对 StatLog 在同级 `check-native-linear-paired/`，
接口测试在 `native-linear-dense-20261030/`。
旧汇编重写方案 Triton !518 已撤回，本接口不修改生成的汇编。
