# 统一 block-scaled dense GEMM

唯一公开入口：`from boltops.gemm import gemm_block_scaled`。

```python
c = gemm_block_scaled(
    a, b, a_scale, b_scale,
    scale_recipe=((1, 128), (128, 128)),
    scale_encodings=("numeric", "numeric"), out_dtype=torch.bfloat16,
)
```

输入为连续的 A[M,K]、B[N,K]，计算分块乘 scale 后沿 K 累加的 A @ B.T。
四个输入位于同一 GPU。M/N 为空返回空矩阵，K=0 返回零矩阵。
`scale_recipe=((A_M,A_K),(B_N,B_K))` 独立指定两个操作数的分组；scale shape 为
`[ceil(M/A_M),ceil(K/A_K)]` 和 `[ceil(N/B_N),ceil(K/B_K)]`，K-group 连续。
`scale_encodings` 每项为 `numeric` 或 `e8m0`；后者要求 uint8 存储并显式解码指数码，
numeric uint8 保持数值解释。scale 运算采用 FP32，FP64 输入或 scale 时提升为 FP64。
输入类型和原生 dot 指令是否合法由编译器判断；不通过输入上转型或软件模拟替代。

## 调度

`boltops/gemm/configs/block_scaled_dispatch.json` 是调度映射表，分为 `match` 与 `config` 两部分：
`match` 记录架构、`num_cus`、输入/输出 dtype、scale dtype/encoding/recipe 和 MNK 条件；
`config` 记录选出的实现标识和 tile。tile 不属于请求条件。
不保存实测 cycles、频率、commit 或产物路径；`num_cus` 只作为匹配条件。增加已经调好的配置时增加对应条目。

- 命中表且满足固定 kernel 契约：调度 Gluon 已优化实现。
- 未命中：调度 Triton 通用 block-scaled GEMM。
- 不捕获编译/执行异常再切换实现，错误如实报告。

当前 Gluon 条目匹配 gfx946、48 CU、E4M3FN×E4M3FN、FP32 A1×128/B128×128 scale、BF16 输出、
BM/BN/BK=128/256/128；M 为128倍数、N为256倍数、K为256倍数且256≤K≤6144，
A/B 基址16字节对齐。`num_cus` 读取当前输入设备的 `multi_processor_count`；
CU 数不同或未知时不匹配该条目，不能仅按架构推定 CU 数。其他尺寸、精度或 recipe 统一走 Triton，包括尾块和不同 A/B K 分组。
通用路径按 gcd(A_K,B_K) 划分共同区间，每区间原生 dot 后乘 scale 并累加，支持非整除尺寸。
这不代表完整业务尺寸均已验收或通用配置已经调优。

## 组织与命名

公共入口、描述、注册与映射放在 `boltops/gemm/`；Triton 和 Gluon 各自只放实现。
旧的 `gemm_a8w8_blockscale`、`w8a8_triton_block_scaled_mm`、`triton_scaled_mm`
不再由 `boltops.gemm` 导出，不保留公开兼容别名。上游实现文件仅保留内部参考与独立回归。

| 层次 | 名称 |
| --- | --- |
| 公开计算接口 | `gemm_block_scaled` |
| 通用 Triton kernel | `_gemm_block_scaled_kernel` |
| 专用 Gluon kernel | `_gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16_bm128_bn256_bk128_kernel` |

kernel 名体现计算场景及固定 tile，不使用 masked、MLS、WDRA 等实现手段作名称后缀。
启动函数使用相同场景名加 `launch_` 前缀；流水内部辅助函数使用预取、等待、释放等动作名。
Gluon 当前只保留已优化 TG 实现，不提供另一套通用/尾块 kernel。

## Gluon 实现约束

依赖 Triton !541、!542、!543。`gl.warp_specialize` 声明 `[8,8,240,240]`，
launch 不重复声明寄存器配额；后端保留硬件规范化和 spill 反馈机制。
scale ACK 与持久零 VGPR 的两处内联汇编及 TODO 保留。
不运行 autotune 或 split-K，不调优 E；其他 GEMM 算子族沿用各自已有接口。
