# GEMM 开发记录

当前公开 GEMM 接口、参数类型、tensor layout、scale 语义和后端支持范围见 [GEMM API docstrings](../gemm.md)。实现与配置随上游 `main` 同步。

本目录保留 `gemm` 分支的需求清单和 debug/tuning 记录：

- [少伯 BU 支持需求](SHAOBO_BU_REQUIREMENTS.md)
- [关键 debug 记录](debug_tuning/DEBUG.md)
- [调优记录](debug_tuning/TUNING.md)
- [BF16 linear GEMM 调优记录](debug_tuning/bf16_linear_gemm/README.md)

历史记录中的接口名、源码路径、配置、编译器版本及性能结果对应当时的实验环境，不作为当前公开接口或当前版本性能的验收声明。
