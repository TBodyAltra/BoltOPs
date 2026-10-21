# 原生 FP8×FP4：K32 scale 与 BF16 写回

2026-10-21，node60 / gfx946 PMD，单 die、48 CU。该修复在隔离 worktree 验证；BoltOps 与 Triton 补丁需配套使用。

## 计算语义

A 为 FP8 E4M3，W 为沿 K 打包的 E2M1；偶数 K 位于低半字节。每次读取 32 个 A 元素和 16 个 W 字节，调用无硬件 scale 操作数的 `tl.dot_scaled`，实际指令为 `v_mmac_f32_16x16x32_fp8_fp4`。之后按原顺序计算 `acc += partial * A_scale * W_scale`。A scale 可为 K32/K128，W scale 固定 K32；uint8 为 E8M0，浮点 scale 直接保留数值。跨组累加使用 FP32，需求输出 BF16。

没有把 W 数值转换为 E4M3，没有用 K128 的完整 dot 代替四个各自带 scale 的 K32 dot；函数签名和默认 tile/warps/stages 未变。

## Triton 依赖

- `third_party/hcu/lib/TritonHCUTransforms/AccelerateHCUMatmul.cpp`：非 MLS、K-packed、无硬件 scale、kBase8 的 FP4 dot operand 使用存储宽度 kWidth4；四个字节代表八个逻辑 FP4 元素。
- `third_party/hcu/lib/TritonHCUToLLVM/DotOpToLLVM/MMAC.cpp`：比较 A/B 的逻辑 K 范围，并把一个 i32 中的八个 E2M1 半字节整理为硬件要求的两个 i32。每个字节的 E2M1 位于 [5:2]，即 code<<2，与已验证的 MLS B4Mix 输出一致。此处只整理整数编码和布局，不做 FP4→FP8 浮点格式转换。
- LLVM intrinsic 的现有两个 vector<2xi32> 输入 ABI 不变。未扩展 F16 dot_scaled 前端，当前 MMAC 结果仍为 F32。

本次私有 libtriton SHA256：`1d515b0de46748c68ab5b2beb1f190d20c414fba57c8445f3213dc45ab75dcef`。

## 验证

| 项目 | 结果 | 范围 |
| --- | --- | --- |
| FP8×FP4 接口 pytest | 19/19 PASS | A/B 编码或数值 scale、dense/grouped、ragged/空组、尾块、stride、预分配输出、FP16/FP32 兼容输出 |
| 独立新旧输出对照 | 10/10 PASS | 含非单位 K32 scale、A K32/K128、FP32/E8M0、K1057 尾部、正 stride、多 CTA、K3072；10 项均与 CPU 和旧实现逐元素一致 |
| 全部 E2M1 编码 | PASS，零误差 | 16 种编码，两个半字节位置均覆盖，通过实际 GEMM 检查 |
| 同一 K128 内四个不同 W scale | PASS，零误差 | A K32/K128 × FP32/E8M0，共四项定向回归 |
| 原有 MLS 原生无 scale 路径 | PASS，零误差 | 32×32×128，单位 scale，BF16 输出；独立于新的普通加载路径 |
| 汇编检查 | PASS | 10 项独立对照仅使用原生 F32 FP8×FP4 MMAC，没有 scaled MMAC 或 FP8×FP8 MMAC 替代 |

上述为 PMD 小规模正确性验证，未运行完整需求矩阵、实卡测试或性能评测。完整 pytest 保留了各测试原有的精度阈值；“逐元素一致”仅指 10 项新旧对照以及显式零误差测试。

## 复现与证据

容器：`kxu_pmd_8569_fp8_20261017`。隔离目录：`/xukang/code/pmd-triton-sessions/fp8fp4-k32-native-20261021`。

```bash
source /xukang/code/pmd-triton-sessions/fp8fp4-k32-native-20261021/env.sh
# PYTHONPATH 同时选择私有 libtriton 与本修复的 BoltOps worktree。
cd /xukang/code/pmd-triton-sessions/fp8fp4-k32-native-20261021
TRITON_CACHE_DIR="$PWD/repro-cache" python -m pytest -q -p no:cacheprovider \
  --basetemp="$PWD/repro-pytest-temp" \
  bolt_ops/test/gemm/test_gemm_a8wfp4.py
```

外部目录保存 `environment.json`、`pytest-results.xml`、`pytest.log`、`validation-results.json`、每 case 的输入/scale/CPU 参考/新旧输出与 IR/汇编。StatLog、PMD 数据、缓存和大文件全部在 Git worktree 之外。保护进程与旧工作区检查见 `final-safety.json`。
