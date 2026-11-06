#### 实现依据与 MMAC 路径

海光 `shaobo-support@f50e796` 中，Basic BF16 调用 `torch.matmul`；Basic INT8/FP8 由 LMSlim 直接将低精度 A/B 传入 `hipblaslt_w8a8_gemm`，封装层没有先转 BF16。实际 MMAC 指令类型尚需底层源码或 ISA 证据，未标记为已验证。Basic FP4、MXFP8/MXFP4 当前无对应 GEMM 路径。

`quant_matmul` 的 LMSlim provider 仍继承 INT8-only 检查；其 base 参考路径才是 INT8 转 BF16 后做 mm，再以 FP32 乘 scale。两条路径需要区分。Basic 的 INT8/FP8 封装会在调用内创建全 1 scale，并执行 `B.t().contiguous()`，计时可能包含这些开销。

INT8 输出统一按 BF16 需求处理；base Basic INT8 仍可能返回 INT8，该实现差异需修复后验收。未规定的量化 scale 按本文 FP32/token/输出通道默认值；MX 的 E8M0/32 元素来自标准，分组轴、物理布局及动态性仍需实现约定。

Excel 的 `6192` 保留；海光 DCU 配置使用 `6144`，差异不静默合并。此前公开仓库的 CPU/PMD 记录仅代表当时版本，未新增 GPU 正确性或性能 PASS。

[海光接口、默认值及 MMAC 证据边界](debug_tuning/byte_january_20261105/hygon_fork_20261106/README.md) · [历史 CPU / PMD 原始记录](debug_tuning/byte_january_20261105/README.md)
