# GEMM 支持需求

需求来源：`shaobob0_shapes_1102/` 新版 Excel，以及 `GPU单体测试用例 v2.1.xlsx` 的第 72–97 行（排除 MoE）。

<a name="business-model-index"></a>
<a name="p0-快速索引"></a>

业务与模型快速索引：

| 业务 | 优先级 | 模型 / 测试项 | 需求组数 | 编号范围 |
| --- | --- | --- | --- | --- |
| 12 月 BU 性能测试 | P0 | [Qwen3.5-35B-A3B FP8](#model-q35-fp8) | 13 | [R001–R013](#r001) |
| 12 月 BU 性能测试 | P0 | [Qwen3.5-35B-A3B MXFP4](#model-q35-mxfp4) | 13 | [R014–R026](#r014) |
| 12 月 BU 性能测试 | P0 | [Qwen3-VL-8B BF16](#model-qwen3-vl-8b) | 20 | [R027–R046](#r027) |
| 1 月字节送测需求 | P1 | [GEMM Basic（GPU_Micro_001）](#model-byte-basic) | 16 | [R047–R062](#r047) |
| 1 月字节送测需求 | P1 | [quant_matmul（GPU_Micro_015）](#model-byte-quant) | 24 | [R063–R086](#r063) |
| 后续支持 | P2 | [DeepSeek-V4-Flash](#model-ds-flash) | 84 | [R087–R170](#r087) |
| 后续支持 | P2 | [DeepSeek-V4-Pro](#model-ds-pro) | 68 | [R171–R238](#r171) |
| 后续支持 | P2 | [GLM-5.1 FP8](#model-glm51-fp8) | 82 | [R239–R320](#r239) |
| 后续支持 | P2 | [Kimi-2.6 W4A16](#model-kimi26-w4a16) | 31 | [R321–R351](#r321) |
| 后续支持 | P2 | [Hunyuan-V3 FP8](#model-hy3-fp8) | 23 | [R352–R374](#r352) |
| 后续支持 | P2 | [Qwen3.5-397B-A17B FP8](#model-q397-fp8) | 25 | [R375–R399](#r375) |

<a name="december-bu-p0"></a>

## 12 月 BU 性能测试（P0）

<a name="model-q35-fp8"></a>

### Qwen3.5-35B-A3B FP8

4k/1k、s150；TP1，单卡实例，chunk 6656。

需求来源：`shape_07_q35_fp8_4k_s150.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R001](#r001)<a name="r001"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 64 × 64 × 2048 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R002](#r002)<a name="r002"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode`<br>`lm_head_prefill` | 64 × 248320 × 2048 | — | prefill/decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R003](#r003)<a name="r003"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 2560 × 64 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R004](#r004)<a name="r004"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 6656 × 64 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R005](#r005)<a name="r005"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 64 × 2048 × 4096 | — | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过<br>[48 TG PMD 最优记录](#p0-fp8-48tg-best-20261110) |
| [R006](#r006)<a name="r006"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 64 × 9216 × 2048 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R007](#r007)<a name="r007"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 64 × 12288 × 2048 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R008](#r008)<a name="r008"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 2560 × 2048 × 4096 | — | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R009](#r009)<a name="r009"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 2560 × 9216 × 2048 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R010](#r010)<a name="r010"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 2560 × 12288 × 2048 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R011](#r011)<a name="r011"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 6656 × 2048 × 4096 | — | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R012](#r012)<a name="r012"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 6656 × 9216 × 2048 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R013](#r013)<a name="r013"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 6656 × 12288 × 2048 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |

<a name="p0-fp8-48tg-best-20261110"></a>

#### TG 数为 48 倍数的调优实测（2026-11-10）

本轮 FP8 完整网格实测中，满足该条件的最佳记录为 **R005：BM24、BN128、BK128，grid=3×16=48 TG**。单 TG 调优和仅由形状推算的 TG 数不作为完整 GEMM 性能记录。

测试环境为 node60 / `kxu_pmd_8569_fp8_20261017`，gfx946 单 die、48 CU 的 **PMD CoreArch1703 模拟器**，不是 BU 实卡验收结果。周期读取完整 dispatch 的 `system.gcSimCycles`；每个性能 case 发射三次，第一次数值校验通过但不计入热缓存均值。不同 MNK 或单 TG 的周期不能直接用来判断完整算子加速。

| 编号 / 方案 | M×N×K | 逻辑 BM×BN×BK | TG / 活跃 CU | WASP / WDRA | 热缓存 cycles（两次） | 热缓存均值 cycles | 对应产物 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| **R005，当前最佳 48 TG** | 64×2048×4096 | **24×128×128** | **48 / 48** | 开 / 关；每分区 2 waves | 42174、42396 | **42285.0** | [Gluon 源码](tuning/r005_48tg_20261110/kernel.py)、[完整汇编](tuning/r005_48tg_20261110/kernel.amdgcn) |
| R005，48 TG 大物理 tile 对照 | 64×2048×4096 | 24×128×128；物理计算 64×256 | 48 / 48 | 开 / 开；每分区 4 waves | 51891、53692 | 52791.5 | [校验与性能原始记录](tuning/r005_48tg_20261110/measurements.json) |
| R005，同编译选项的 8 TG 基线 | 64×2048×4096 | 64×256×128 | 8 / 8 | 开 / 开；每分区 4 waves | 54330、54435 | 54382.5 | [校验与性能原始记录](tuning/r005_48tg_20261110/measurements.json) |

最佳 48 TG 版本相对同选项 8 TG 基线的周期减少 **22.25%**；相对 48 TG 大物理 tile 对照减少 **19.90%**。这是 block、wave 数、WDRA 与写回路径共同变化的整体结果，不能把收益全部归因于 TG 数量。仅在历史记录间比较，最初 8 TG 调度为 52190 cycles，本轮减少 18.98%；该历史版本未作为上表的同轮基线。

实现与验证：

- 使用 Gluon `warp_specialize`：2 Producer + 2 Consumer，`wasp_partition_warps=(2,2,2,2)`，共 8 waves/TG；`wasp_wdra=False`。每个 Consumer 计算 16×128，物理计算 tile 为 32×128。三个 M tile 的逻辑行起点为 0、24、48，分别写回 24、24、16 行；A 加载边界补零，C 显式 mask，处理 padding 与尾块。
- 启动时分配的动态 LDS 为 **73984 B（72.25 KiB）**，kernel VGPR=**101**，spill=**0**；汇编的固定 LDS 为 0，动态 LDS 用量以 launch metadata 为准。A 的物理 MLS LDS tile 仍为 64×128；保留 A/B 双缓冲，以及循环前两批 scale 预取。
- A scale 为 FP32；B scale 保留需求中的 BF16 数值，在 **CPU 精确提升到 FP32 后上传**，kernel 接收 FP32。转换和上传不计入上述周期；这不是直接 BF16 B-scale kernel 的计时。BN128 内只有一个 B scale，LDS 中复制为两个相同值以保留原 packed 读取路径。
- seed21 完整网格三次、seed42 完整网格一次均通过：全部 131072 个 BF16 输出与 CPU 参考完全相同，`max_abs=0`、`mismatch=0`、`nonfinite=0`，前后保护行通过。每次完整网格的 `completedWGs` 都确认 **48 个 CU 各完成一个 TG**；这不表示每个 CU 的执行单元达到满利用率。
- Triton 使用 `release/3.6.x@d8374cbaca28e5c4e7789fcf130b6701f71be0eb`；GPU IR 编译使用已构建的 `llvm-project/dtk-llvm-6.3.3@3599a58d7aebe66dc1cec4f830f82cb96e505666` Clang，包含 LDS read pairing 修改。编译器源码未改。为绕过新代码触发的 `HCU Hoist deferred global loads` 断言，最终方案和同轮对照均加 `-mllvm=-enable-hcu-hoist-deferred-loads=false`。大物理 tile 对照仍有 PMD `MLS:wrong: nm_filter out range` 告警，其数值 PASS 不代表日志无告警。

最佳汇编和 PERF 对应同一个已经验证的 HSACO，SHA256 为 `da290a5cd2d00b973afdf660c2ae0b765b378f97afe1674b1c583a57ffe5aa44`。汇编仅去除行尾空白，未改指令，并保留生成时的原始符号名；原始及归档文件哈希均在 manifest 中，实际逻辑 block 以本节及 manifest 为准。[PERF 文件](tuning/r005_48tg_20261110/best.perf)来自额外的完整 48 TG 捕获，三次周期为 45929、42456、42316；选择热缓存最低的 **42316 cycles**，精度与边界检查通过。它是 trace 样本，不替代普通性能测试的均值 **42285.0** 或最低样本 **42174**。

[版本、资源与文件 SHA256](tuning/r005_48tg_20261110/manifest.json) · [各次发射、精度和 CU 分布](tuning/r005_48tg_20261110/measurements.json)。这些文件是本次调优证据；需求行的接口 smoke 状态不等同于该实验配置已接入通用 dispatch。

<a name="model-q35-mxfp4"></a>

### Qwen3.5-35B-A3B MXFP4

128k/1k、s150；TP1，单卡实例，chunk 6656。

需求来源：`shape_08_q35_mxfp4_128k_s150.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R014](#r014)<a name="r014"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 12 × 64 × 2048 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R015](#r015)<a name="r015"></a> | linear<br>BF16×BF16→BF16 | `o_proj`<br>`out_proj` | 12 × 2048 × 4096 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R016](#r016)<a name="r016"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 12 × 9216 × 2048 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R017](#r017)<a name="r017"></a> | linear<br>BF16×BF16→BF16 | `in_proj_qkvz` | 12 × 12288 × 2048 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R018](#r018)<a name="r018"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode`<br>`lm_head_prefill` | 12 × 248320 × 2048 | — | prefill/decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R019](#r019)<a name="r019"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 2048 × 64 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R020](#r020)<a name="r020"></a> | linear<br>BF16×BF16→BF16 | `o_proj`<br>`out_proj` | 2048 × 2048 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R021](#r021)<a name="r021"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 2048 × 9216 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R022](#r022)<a name="r022"></a> | linear<br>BF16×BF16→BF16 | `in_proj_qkvz` | 2048 × 12288 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R023](#r023)<a name="r023"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 6656 × 64 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R024](#r024)<a name="r024"></a> | linear<br>BF16×BF16→BF16 | `o_proj`<br>`out_proj` | 6656 × 2048 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R025](#r025)<a name="r025"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 6656 × 9216 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R026](#r026)<a name="r026"></a> | linear<br>BF16×BF16→BF16 | `in_proj_qkvz` | 6656 × 12288 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |

<a name="model-qwen3-vl-8b"></a>

### Qwen3-VL-8B BF16

主干输入 226 token、输出 300；文本 30 token + 448×448 图像；TP1/bs512。文件保留 s150 标签。

需求来源：`shape_12_qwen3vl8b_226_s150.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R027](#r027)<a name="r027"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 512 × 4096 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R028](#r028)<a name="r028"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 512 × 4096 × 12288 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R029](#r029)<a name="r029"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 512 × 6144 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R030](#r030)<a name="r030"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 512 × 24576 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R031](#r031)<a name="r031"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode`<br>`lm_head_prefill` | 512 × 151936 × 4096 | — | prefill/decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R032](#r032)<a name="r032"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 2560 × 4096 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R033](#r033)<a name="r033"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 2560 × 4096 × 12288 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R034](#r034)<a name="r034"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 2560 × 6144 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R035](#r035)<a name="r035"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 2560 × 24576 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R036](#r036)<a name="r036"></a> | linear<br>BF16×BF16→BF16 | `merger_fc2` | 5684 × 4096 × 4608 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R037](#r037)<a name="r037"></a> | linear<br>BF16×BF16→BF16 | `merger_fc1` | 5684 × 4608 × 4608 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R038](#r038)<a name="r038"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 6656 × 4096 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R039](#r039)<a name="r039"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 6656 × 4096 × 12288 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R040](#r040)<a name="r040"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 6656 × 6144 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R041](#r041)<a name="r041"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 6656 × 24576 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R042](#r042)<a name="r042"></a> | linear<br>BF16×BF16→BF16 | `vit_o_proj` | 22736 × 1152 × 1152 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R043](#r043)<a name="r043"></a> | linear<br>BF16×BF16→BF16 | `vit_patch_embed` | 22736 × 1152 × 1536 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R044](#r044)<a name="r044"></a> | linear<br>BF16×BF16→BF16 | `vit_mlp_fc2` | 22736 × 1152 × 4304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R045](#r045)<a name="r045"></a> | linear<br>BF16×BF16→BF16 | `vit_qkv_proj` | 22736 × 3456 × 1152 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |
| [R046](#r046)<a name="r046"></a> | linear<br>BF16×BF16→BF16 | `vit_mlp_fc1` | 22736 × 4304 × 1152 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]<br>待验证（新增需求） |

<a name="january-byte-p1"></a>

## 1 月字节送测需求（P1）

需求来源：`GPU单体测试用例 v2.1.xlsx` → `测试用例` 第 72–97 行，纳入第 72、86 行；MoE 及非 GEMM 用例不纳入。以下为通用算子测试，不限定模型。

补充约定：INT8 默认输出 BF16；未规定的量化 scale 使用 FP32，A 按 token、B 按输出通道。MXFP8/MXFP4 保留标准 E8M0 / 32 元素分组；Basic BF16 不使用量化 scale。

验收要求：性能与厂商宣称值的差异不超过 10%，测试过程无软硬件异常；宣称值及对应的精度、频率和测量口径需随送测结果提供。

<a name="model-byte-basic"></a>

### GEMM Basic（GPU_Micro_001）

需求来源：`GPU单体测试用例 v2.1.xlsx`，`测试用例!B72:E72`。

`M₇₂` = `4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768`。每行覆盖此列表全部 M；矩阵维度统一写为 **M × N × K**。

Excel 的 KN 按 **K、N** 解释：A[M,K] × B[K,N] → C[M,N]；FP4 为“如有”条件项。输出每组 shape 的算力。本文 BF16、INT8、FP8、FP4 默认输出 BF16。精度与 scale 采用下表口径，不表示相应 GPU 实现已验收。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R047](#r047)<a name="r047"></a> | gemm<br>bfloat16 × bfloat16 → bfloat16 | `gemm` | M₇₂ × 4096 × 4096 | — | — | 1 | — | — | [`gemm_a16w16`](../../boltops/gemm/triton/gemm_a16w16.py)<br>14 组 MNK；待验收 |
| [R048](#r048)<a name="r048"></a> | gemm<br>bfloat16 × bfloat16 → bfloat16 | `gemm` | M₇₂ × 8192 × 8192 | — | — | 1 | — | — | [`gemm_a16w16`](../../boltops/gemm/triton/gemm_a16w16.py)<br>14 组 MNK；待验收 |
| [R049](#r049)<a name="r049"></a> | gemm<br>bfloat16 × bfloat16 → bfloat16 | `gemm` | M₇₂ × 1024 × 8192 | — | — | 1 | — | — | [`gemm_a16w16`](../../boltops/gemm/triton/gemm_a16w16.py)<br>14 组 MNK；待验收 |
| [R050](#r050)<a name="r050"></a> | gemm<br>bfloat16 × bfloat16 → bfloat16 | `gemm` | M₇₂ × 8192 × 1024 | — | — | 1 | — | — | [`gemm_a16w16`](../../boltops/gemm/triton/gemm_a16w16.py)<br>14 组 MNK；待验收 |
| [R051](#r051)<a name="r051"></a> | gemm<br>int8 × int8 → bfloat16 | `gemm` | M₇₂ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>14 组 MNK；待验收 |
| [R052](#r052)<a name="r052"></a> | gemm<br>int8 × int8 → bfloat16 | `gemm` | M₇₂ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>14 组 MNK；待验收 |
| [R053](#r053)<a name="r053"></a> | gemm<br>int8 × int8 → bfloat16 | `gemm` | M₇₂ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>14 组 MNK；待验收 |
| [R054](#r054)<a name="r054"></a> | gemm<br>int8 × int8 → bfloat16 | `gemm` | M₇₂ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>14 组 MNK；待验收 |
| [R055](#r055)<a name="r055"></a> | gemm<br>float8 × float8 → bfloat16 | `gemm` | M₇₂ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>14 组 MNK；待验收 |
| [R056](#r056)<a name="r056"></a> | gemm<br>float8 × float8 → bfloat16 | `gemm` | M₇₂ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>14 组 MNK；待验收 |
| [R057](#r057)<a name="r057"></a> | gemm<br>float8 × float8 → bfloat16 | `gemm` | M₇₂ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>14 组 MNK；待验收 |
| [R058](#r058)<a name="r058"></a> | gemm<br>float8 × float8 → bfloat16 | `gemm` | M₇₂ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>14 组 MNK；待验收 |
| [R059](#r059)<a name="r059"></a> | gemm<br>float4 × float4 → bfloat16 | `gemm` | M₇₂ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | 待实现<br>14 组 MNK；FP4 如有；普通 FP4 / FP32 scale 接口待实现 |
| [R060](#r060)<a name="r060"></a> | gemm<br>float4 × float4 → bfloat16 | `gemm` | M₇₂ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | 待实现<br>14 组 MNK；FP4 如有；普通 FP4 / FP32 scale 接口待实现 |
| [R061](#r061)<a name="r061"></a> | gemm<br>float4 × float4 → bfloat16 | `gemm` | M₇₂ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | 待实现<br>14 组 MNK；FP4 如有；普通 FP4 / FP32 scale 接口待实现 |
| [R062](#r062)<a name="r062"></a> | gemm<br>float4 × float4 → bfloat16 | `gemm` | M₇₂ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | 待实现<br>14 组 MNK；FP4 如有；普通 FP4 / FP32 scale 接口待实现 |

<a name="model-byte-quant"></a>

### quant_matmul（GPU_Micro_015）

需求来源：`GPU单体测试用例 v2.1.xlsx`，`测试用例!B86:E86`。

`M₈₆` = `1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 4096, 6192, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072`。每行覆盖此列表全部 M；矩阵维度统一写为 **M × N × K**。

`sp_size=1`，M=`num_tokens`、K=`hidden_size`、N=`new_hidden_size`。表内精度顺序为 **输入 / 权重 / 计算 / 输出**。

FP4、MXFP4 均需覆盖；输出 latency、算力、带宽、MFU、MBU。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R063](#r063)<a name="r063"></a> | quant_matmul<br>bfloat16 / bfloat16 / bfloat16 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R064](#r064)<a name="r064"></a> | quant_matmul<br>bfloat16 / bfloat16 / bfloat16 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R065](#r065)<a name="r065"></a> | quant_matmul<br>bfloat16 / bfloat16 / bfloat16 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R066](#r066)<a name="r066"></a> | quant_matmul<br>bfloat16 / bfloat16 / bfloat16 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R067](#r067)<a name="r067"></a> | quant_matmul<br>int8 / int8 / int8 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R068](#r068)<a name="r068"></a> | quant_matmul<br>int8 / int8 / int8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R069](#r069)<a name="r069"></a> | quant_matmul<br>int8 / int8 / int8 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R070](#r070)<a name="r070"></a> | quant_matmul<br>int8 / int8 / int8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R071](#r071)<a name="r071"></a> | quant_matmul<br>float8 / float8 / float8 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R072](#r072)<a name="r072"></a> | quant_matmul<br>float8 / float8 / float8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R073](#r073)<a name="r073"></a> | quant_matmul<br>float8 / float8 / float8 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R074](#r074)<a name="r074"></a> | quant_matmul<br>float8 / float8 / float8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [`triton_scaled_mm`](../../boltops/gemm/triton/triton_scaled_mm.py)<br>46 组 MNK；待验收 |
| [R075](#r075)<a name="r075"></a> | quant_matmul<br>float4 / float4 / float4 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | 待实现<br>46 组 MNK；普通 FP4 / FP32 scale 接口待实现 |
| [R076](#r076)<a name="r076"></a> | quant_matmul<br>float4 / float4 / float4 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | 待实现<br>46 组 MNK；普通 FP4 / FP32 scale 接口待实现 |
| [R077](#r077)<a name="r077"></a> | quant_matmul<br>float4 / float4 / float4 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | 待实现<br>46 组 MNK；普通 FP4 / FP32 scale 接口待实现 |
| [R078](#r078)<a name="r078"></a> | quant_matmul<br>float4 / float4 / float4 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | 待实现<br>46 组 MNK；普通 FP4 / FP32 scale 接口待实现 |
| [R079](#r079)<a name="r079"></a> | quant_matmul<br>mxfloat8 / mxfloat8 / mxfloat8 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | 待实现<br>46 组 MNK；MXFP8 / E8M0 scale GEMM 接口待实现 |
| [R080](#r080)<a name="r080"></a> | quant_matmul<br>mxfloat8 / mxfloat8 / mxfloat8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | 待实现<br>46 组 MNK；MXFP8 / E8M0 scale GEMM 接口待实现 |
| [R081](#r081)<a name="r081"></a> | quant_matmul<br>mxfloat8 / mxfloat8 / mxfloat8 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | 待实现<br>46 组 MNK；MXFP8 / E8M0 scale GEMM 接口待实现 |
| [R082](#r082)<a name="r082"></a> | quant_matmul<br>mxfloat8 / mxfloat8 / mxfloat8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | 待实现<br>46 组 MNK；MXFP8 / E8M0 scale GEMM 接口待实现 |
| [R083](#r083)<a name="r083"></a> | quant_matmul<br>mxfloat4 / mxfloat4 / mxfloat4 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | [`gemm_afp4wfp4`](../../boltops/gemm/triton/gemm_afp4wfp4.py)<br>46 组 MNK；E8M0 / K32 接口；待验收 |
| [R084](#r084)<a name="r084"></a> | quant_matmul<br>mxfloat4 / mxfloat4 / mxfloat4 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | [`gemm_afp4wfp4`](../../boltops/gemm/triton/gemm_afp4wfp4.py)<br>46 组 MNK；E8M0 / K32 接口；待验收 |
| [R085](#r085)<a name="r085"></a> | quant_matmul<br>mxfloat4 / mxfloat4 / mxfloat4 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | [`gemm_afp4wfp4`](../../boltops/gemm/triton/gemm_afp4wfp4.py)<br>46 组 MNK；E8M0 / K32 接口；待验收 |
| [R086](#r086)<a name="r086"></a> | quant_matmul<br>mxfloat4 / mxfloat4 / mxfloat4 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | [`gemm_afp4wfp4`](../../boltops/gemm/triton/gemm_afp4wfp4.py)<br>46 组 MNK；E8M0 / K32 接口；待验收 |

<a name="follow-up-p2"></a>

## 后续支持（P2）

<a name="model-ds-flash"></a>

### DeepSeek-V4-Flash

64k/128k × s150/s2000；各文件部署方案分别保留。

需求来源：`shape_01_flash_64k_s150.xlsx`、`shape_01_flash_64k_s2000.xlsx`、`shape_02_flash_128k_s150.xlsx`、`shape_02_flash_128k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R087](#r087)<a name="r087"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 8 × 129280 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R088](#r088)<a name="r088"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 64 × 16160 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R089](#r089)<a name="r089"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 160 × 129280 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R090](#r090)<a name="r090"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 192 × 64 × 4096 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R091](#r091)<a name="r091"></a> | linear<br>BF16×BF16→BF16 | `dspark_ctx_wkv_0`<br>`dspark_ctx_wkv_1`<br>`dspark_ctx_wkv_2` | 192 × 512 × 4096 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R092](#r092)<a name="r092"></a> | linear<br>BF16×BF16→BF16 | `dspark_main_proj` | 192 × 4096 × 12288 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R093](#r093)<a name="r093"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 192 × 129280 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R094](#r094)<a name="r094"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 240 × 129280 × 4096 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R095](#r095)<a name="r095"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 288 × 64 × 4096 | — | decode | 6 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R096](#r096)<a name="r096"></a> | linear<br>BF16×BF16→BF16 | `dspark_ctx_wkv_0`<br>`dspark_ctx_wkv_1`<br>`dspark_ctx_wkv_2` | 288 × 512 × 4096 | — | decode | 9 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R097](#r097)<a name="r097"></a> | linear<br>BF16×BF16→BF16 | `dspark_main_proj` | 288 × 4096 × 12288 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R098](#r098)<a name="r098"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 288 × 129280 × 4096 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R099](#r099)<a name="r099"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 3584 × 64 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R100](#r100)<a name="r100"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 5120 × 64 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R101](#r101)<a name="r101"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 6656 × 64 × 4096 | — | prefill | 16 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R102](#r102)<a name="r102"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 160 × 512 × 4096 | — | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R103](#r103)<a name="r103"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 160 × 1024 × 4096 | — | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R104](#r104)<a name="r104"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 160 × 4096 × 8192 | — | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R105](#r105)<a name="r105"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 160 × 32768 × 1024 | — | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R106](#r106)<a name="r106"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 192 × 512 × 4096 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R107](#r107)<a name="r107"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 192 × 1024 × 4096 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R108](#r108)<a name="r108"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 192 × 4096 × 8192 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R109](#r109)<a name="r109"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 192 × 8192 × 1024 | — | decode | 2 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R110](#r110)<a name="r110"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 192 × 32768 × 1024 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R111](#r111)<a name="r111"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 240 × 512 × 4096 | — | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R112](#r112)<a name="r112"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 240 × 1024 × 4096 | — | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R113](#r113)<a name="r113"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 240 × 4096 × 8192 | — | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R114](#r114)<a name="r114"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 240 × 32768 × 1024 | — | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R115](#r115)<a name="r115"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 288 × 512 × 4096 | — | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R116](#r116)<a name="r116"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 288 × 1024 × 4096 | — | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R117](#r117)<a name="r117"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 288 × 4096 × 8192 | — | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R118](#r118)<a name="r118"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 288 × 8192 × 1024 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R119](#r119)<a name="r119"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 288 × 32768 × 1024 | — | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R120](#r120)<a name="r120"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 3584 × 512 × 4096 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R121](#r121)<a name="r121"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 3584 × 1024 × 4096 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R122](#r122)<a name="r122"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 3584 × 4096 × 8192 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R123](#r123)<a name="r123"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 3584 × 8192 × 1024 | — | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R124](#r124)<a name="r124"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 3584 × 32768 × 1024 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R125](#r125)<a name="r125"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 5120 × 512 × 4096 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R126](#r126)<a name="r126"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 5120 × 1024 × 4096 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R127](#r127)<a name="r127"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 5120 × 4096 × 8192 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R128](#r128)<a name="r128"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 5120 × 8192 × 1024 | — | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R129](#r129)<a name="r129"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 5120 × 32768 × 1024 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R130](#r130)<a name="r130"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 6656 × 512 × 4096 | — | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R131](#r131)<a name="r131"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 6656 × 1024 × 4096 | — | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R132](#r132)<a name="r132"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 6656 × 4096 × 8192 | — | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R133](#r133)<a name="r133"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 6656 × 8192 × 1024 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R134](#r134)<a name="r134"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 6656 × 32768 × 1024 | — | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R135](#r135)<a name="r135"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 160 × 1024 × 4096 | 8 | decode | 3 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R136](#r136)<a name="r136"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 192 × 1024 × 4096 | 8 | decode | 4 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R137](#r137)<a name="r137"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 240 × 1024 × 4096 | 8 | decode | 9 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R138](#r138)<a name="r138"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 288 × 1024 × 4096 | 8 | decode | 12 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R139](#r139)<a name="r139"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 3584 × 1024 × 4096 | 8 | prefill | 16 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R140](#r140)<a name="r140"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 5120 × 1024 × 4096 | 8 | prefill | 16 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R141](#r141)<a name="r141"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 6656 × 1024 × 4096 | 8 | prefill | 32 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R142](#r142)<a name="r142"></a> | linear<br>FP32×FP32→FP32 | `dspark_markov_w2_0`<br>`dspark_markov_w2_1`<br>`dspark_markov_w2_2`<br>`dspark_markov_w2_3`<br>`dspark_markov_w2_4` | 32 × 129280 × 256 | — | decode | 5 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R143](#r143)<a name="r143"></a> | linear<br>FP32×FP32→FP32 | `dspark_markov_w2_0`<br>`dspark_markov_w2_1`<br>`dspark_markov_w2_2`<br>`dspark_markov_w2_3`<br>`dspark_markov_w2_4` | 48 × 129280 × 256 | — | decode | 15 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R144](#r144)<a name="r144"></a> | linear<br>FP32×FP32→FP32 | `dspark_confidence` | 160 × 1 × 4352 | — | decode | 1 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R145](#r145)<a name="r145"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 160 × 4 × 16384 | — | decode | 1 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R146](#r146)<a name="r146"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 192 × 4 × 16384 | — | decode | 1 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R147](#r147)<a name="r147"></a> | linear<br>FP32×FP32→FP32 | `dspark_confidence` | 240 × 1 × 4352 | — | decode | 3 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R148](#r148)<a name="r148"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 240 × 4 × 16384 | — | decode | 3 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R149](#r149)<a name="r149"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 288 × 4 × 16384 | — | decode | 3 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R150](#r150)<a name="r150"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 3584 × 4 × 16384 | — | prefill | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R151](#r151)<a name="r151"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 5120 × 4 × 16384 | — | prefill | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R152](#r152)<a name="r152"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 6656 × 4 × 16384 | — | prefill | 4 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R153](#r153)<a name="r153"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 28672 × 4 × 16384 | — | prefill | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R154](#r154)<a name="r154"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 40960 × 4 × 16384 | — | prefill | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R155](#r155)<a name="r155"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 53248 × 4 × 16384 | — | prefill | 4 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R156](#r156)<a name="r156"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 192 × 512 × 4096 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R157](#r157)<a name="r157"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 192 × 1024 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R158](#r158)<a name="r158"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 192 × 2048 × 4096 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R159](#r159)<a name="r159"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 288 × 512 × 4096 | — | decode | 6 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R160](#r160)<a name="r160"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 288 × 1024 × 4096 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R161](#r161)<a name="r161"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 288 × 2048 × 4096 | — | decode | 6 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R162](#r162)<a name="r162"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 3584 × 512 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R163](#r163)<a name="r163"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 3584 × 1024 × 4096 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R164](#r164)<a name="r164"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 3584 × 2048 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R165](#r165)<a name="r165"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 5120 × 512 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R166](#r166)<a name="r166"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 5120 × 1024 × 4096 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R167](#r167)<a name="r167"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 5120 × 2048 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R168](#r168)<a name="r168"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 6656 × 512 × 4096 | — | prefill | 16 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R169](#r169)<a name="r169"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 6656 × 1024 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R170](#r170)<a name="r170"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 6656 × 2048 × 4096 | — | prefill | 16 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |

<a name="model-ds-pro"></a>

### DeepSeek-V4-Pro

128k/1k、s2000；P 侧 a/b 两种方案，分别为 16 卡 PP2 与 8 卡方案。

需求来源：`shape_03_pro_128k_s2000_a.xlsx`、`shape_03_pro_128k_s2000_b.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R171](#r171)<a name="r171"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 2 × 64640 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R172](#r172)<a name="r172"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 2 × 129280 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R173](#r173)<a name="r173"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 8 × 16160 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R174](#r174)<a name="r174"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 16 × 16160 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R175](#r175)<a name="r175"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 80 × 129280 × 7168 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R176](#r176)<a name="r176"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 96 × 64 × 7168 | — | decode | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R177](#r177)<a name="r177"></a> | linear<br>BF16×BF16→BF16 | `dspark_ctx_wkv_0`<br>`dspark_ctx_wkv_1`<br>`dspark_ctx_wkv_2` | 96 × 512 × 7168 | — | decode | 6 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R178](#r178)<a name="r178"></a> | linear<br>BF16×BF16→BF16 | `dspark_main_proj` | 96 × 7168 × 21504 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R179](#r179)<a name="r179"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 96 × 129280 × 7168 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R180](#r180)<a name="r180"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 2560 × 64 × 7168 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R181](#r181)<a name="r181"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 4608 × 64 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R182](#r182)<a name="r182"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 6656 × 64 × 7168 | — | prefill | 10 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R183](#r183)<a name="r183"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 80 × 512 × 7168 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R184](#r184)<a name="r184"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 80 × 1536 × 7168 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R185](#r185)<a name="r185"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 80 × 7168 × 16384 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R186](#r186)<a name="r186"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 80 × 65536 × 1536 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R187](#r187)<a name="r187"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 96 × 512 × 7168 | — | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R188](#r188)<a name="r188"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 96 × 1536 × 7168 | — | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R189](#r189)<a name="r189"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 96 × 7168 × 16384 | — | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R190](#r190)<a name="r190"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 96 × 8192 × 1536 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R191](#r191)<a name="r191"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 96 × 65536 × 1536 | — | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R192](#r192)<a name="r192"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 2560 × 512 × 7168 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R193](#r193)<a name="r193"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 2560 × 1536 × 7168 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R194](#r194)<a name="r194"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 2560 × 7168 × 8192 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R195](#r195)<a name="r195"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 2560 × 7168 × 16384 | — | prefill | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R196](#r196)<a name="r196"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 2560 × 8192 × 1536 | — | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R197](#r197)<a name="r197"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 2560 × 32768 × 1536 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R198](#r198)<a name="r198"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 2560 × 65536 × 1536 | — | prefill | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R199](#r199)<a name="r199"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 4608 × 512 × 7168 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R200](#r200)<a name="r200"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 4608 × 1536 × 7168 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R201](#r201)<a name="r201"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 4608 × 7168 × 16384 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R202](#r202)<a name="r202"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 4608 × 8192 × 1536 | — | prefill | 2 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R203](#r203)<a name="r203"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 4608 × 65536 × 1536 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R204](#r204)<a name="r204"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 6656 × 512 × 7168 | — | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R205](#r205)<a name="r205"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 6656 × 1536 × 7168 | — | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R206](#r206)<a name="r206"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 6656 × 7168 × 8192 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R207](#r207)<a name="r207"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 6656 × 7168 × 16384 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R208](#r208)<a name="r208"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 6656 × 8192 × 1536 | — | prefill | 10 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R209](#r209)<a name="r209"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 6656 × 32768 × 1536 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R210](#r210)<a name="r210"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 6656 × 65536 × 1536 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R211](#r211)<a name="r211"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 80 × 1024 × 4096 | 16 | decode | 6 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R212](#r212)<a name="r212"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 96 × 1024 × 4096 | 16 | decode | 8 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R213](#r213)<a name="r213"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 2560 × 1024 × 4096 | 16 | prefill | 12 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R214](#r214)<a name="r214"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 2560 × 1024 × 4096 | 8 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R215](#r215)<a name="r215"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 4608 × 1024 × 4096 | 16 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R216](#r216)<a name="r216"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 6656 × 1024 × 4096 | 8 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R217](#r217)<a name="r217"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 6656 × 1024 × 4096 | 16 | prefill | 16 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R218](#r218)<a name="r218"></a> | linear<br>FP32×FP32→FP32 | `dspark_markov_w2_0`<br>`dspark_markov_w2_1`<br>`dspark_markov_w2_2`<br>`dspark_markov_w2_3`<br>`dspark_markov_w2_4` | 16 × 129280 × 512 | — | decode | 10 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R219](#r219)<a name="r219"></a> | linear<br>FP32×FP32→FP32 | `dspark_confidence` | 80 × 1 × 7680 | — | decode | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R220](#r220)<a name="r220"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 80 × 4 × 28672 | — | decode | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R221](#r221)<a name="r221"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 96 × 4 × 28672 | — | decode | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R222](#r222)<a name="r222"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 2560 × 4 × 28672 | — | prefill | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R223](#r223)<a name="r223"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 6656 × 4 × 28672 | — | prefill | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R224](#r224)<a name="r224"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 20480 × 4 × 28672 | — | prefill | 1 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R225](#r225)<a name="r225"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 36864 × 4 × 28672 | — | prefill | 1 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R226](#r226)<a name="r226"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 53248 × 4 × 28672 | — | prefill | 2 | — | — | [matmul_persistent][fn-persistent]<br>接口 smoke 通过（S03） |
| [R227](#r227)<a name="r227"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 96 × 512 × 7168 | — | decode | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R228](#r228)<a name="r228"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 96 × 1024 × 7168 | — | decode | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R229](#r229)<a name="r229"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 96 × 2048 × 7168 | — | decode | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R230](#r230)<a name="r230"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 2560 × 512 × 7168 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R231](#r231)<a name="r231"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 2560 × 1024 × 7168 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R232](#r232)<a name="r232"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 2560 × 2048 × 7168 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R233](#r233)<a name="r233"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 4608 × 512 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R234](#r234)<a name="r234"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 4608 × 1024 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R235](#r235)<a name="r235"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 4608 × 2048 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R236](#r236)<a name="r236"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 6656 × 512 × 7168 | — | prefill | 10 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R237](#r237)<a name="r237"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 6656 × 1024 × 7168 | — | prefill | 10 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R238](#r238)<a name="r238"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 6656 × 2048 × 7168 | — | prefill | 10 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |

<a name="model-glm51-fp8"></a>

### GLM-5.1 FP8

128k/64k、s2000；原表头 5.2/5.1 的命名歧义保留，以第 3 行描述 5.1 为准。

需求来源：`shape_05_glm_128k_s2000.xlsx`、`shape_06_glm_64k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R239](#r239)<a name="r239"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 1 × 154880 × 6144 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R240](#r240)<a name="r240"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 4 × 154880 × 6144 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R241](#r241)<a name="r241"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 6 × 32 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R242](#r242)<a name="r242"></a> | linear<br>BF16×BF16→BF16 | `nextn_eh_proj` | 6 × 6144 × 12288 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R243](#r243)<a name="r243"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 6 × 154880 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R244](#r244)<a name="r244"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 8 × 19360 × 6144 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R245](#r245)<a name="r245"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 12 × 32 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R246](#r246)<a name="r246"></a> | linear<br>BF16×BF16→BF16 | `nextn_eh_proj` | 12 × 6144 × 12288 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R247](#r247)<a name="r247"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 12 × 154880 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R248](#r248)<a name="r248"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 24 × 32 × 6144 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R249](#r249)<a name="r249"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 24 × 154880 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R250](#r250)<a name="r250"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 32 × 19360 × 6144 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R251](#r251)<a name="r251"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 48 × 32 × 6144 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R252](#r252)<a name="r252"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 48 × 154880 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R253](#r253)<a name="r253"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 2560 × 32 × 6144 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R254](#r254)<a name="r254"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 4608 × 32 × 6144 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R255](#r255)<a name="r255"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 6656 × 32 × 6144 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R256](#r256)<a name="r256"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 6 × 128 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R257](#r257)<a name="r257"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 6 × 2624 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R258](#r258)<a name="r258"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 6 × 4096 × 2048 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R259](#r259)<a name="r259"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 6 × 6144 × 16384 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R260](#r260)<a name="r260"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 6 × 16384 × 2048 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R261](#r261)<a name="r261"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 12 × 128 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R262](#r262)<a name="r262"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 12 × 2624 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R263](#r263)<a name="r263"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 12 × 4096 × 2048 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R264](#r264)<a name="r264"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 12 × 6144 × 16384 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R265](#r265)<a name="r265"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 12 × 16384 × 2048 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R266](#r266)<a name="r266"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 24 × 128 × 6144 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R267](#r267)<a name="r267"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 24 × 2624 × 6144 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R268](#r268)<a name="r268"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 24 × 4096 × 2048 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R269](#r269)<a name="r269"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 24 × 6144 × 12288 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R270](#r270)<a name="r270"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 24 × 6144 × 16384 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R271](#r271)<a name="r271"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 24 × 16384 × 2048 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R272](#r272)<a name="r272"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 24 × 24576 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R273](#r273)<a name="r273"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 48 × 128 × 6144 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R274](#r274)<a name="r274"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 48 × 2624 × 6144 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R275](#r275)<a name="r275"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 48 × 4096 × 2048 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R276](#r276)<a name="r276"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 48 × 6144 × 12288 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R277](#r277)<a name="r277"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 48 × 6144 × 16384 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R278](#r278)<a name="r278"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 48 × 16384 × 2048 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R279](#r279)<a name="r279"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 48 × 24576 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R280](#r280)<a name="r280"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 2560 × 128 × 6144 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R281](#r281)<a name="r281"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 2560 × 2624 × 6144 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R282](#r282)<a name="r282"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 2560 × 3072 × 6144 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R283](#r283)<a name="r283"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 2560 × 4096 × 2048 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R284](#r284)<a name="r284"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 2560 × 6144 × 1536 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R285](#r285)<a name="r285"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 2560 × 6144 × 16384 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R286](#r286)<a name="r286"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 2560 × 16384 × 2048 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R287](#r287)<a name="r287"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 4608 × 128 × 6144 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R288](#r288)<a name="r288"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 4608 × 2624 × 6144 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R289](#r289)<a name="r289"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 4608 × 3072 × 6144 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R290](#r290)<a name="r290"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 4608 × 4096 × 2048 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R291](#r291)<a name="r291"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 4608 × 6144 × 1536 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R292](#r292)<a name="r292"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 4608 × 6144 × 16384 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R293](#r293)<a name="r293"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 4608 × 16384 × 2048 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R294](#r294)<a name="r294"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 6656 × 128 × 6144 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R295](#r295)<a name="r295"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 6656 × 2624 × 6144 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R296](#r296)<a name="r296"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 6656 × 3072 × 6144 | — | prefill | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R297](#r297)<a name="r297"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 6656 × 4096 × 2048 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R298](#r298)<a name="r298"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 6656 × 6144 × 1536 | — | prefill | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R299](#r299)<a name="r299"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 6656 × 6144 × 16384 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R300](#r300)<a name="r300"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 6656 × 16384 × 2048 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R301](#r301)<a name="r301"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 20480 × 3072 × 6144 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R302](#r302)<a name="r302"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 20480 × 6144 × 1536 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R303](#r303)<a name="r303"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 36864 × 3072 × 6144 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R304](#r304)<a name="r304"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 36864 × 6144 × 1536 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R305](#r305)<a name="r305"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 53248 × 3072 × 6144 | — | prefill | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R306](#r306)<a name="r306"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 53248 × 6144 × 1536 | — | prefill | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R307](#r307)<a name="r307"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 6 × 256 × 512 | 64 | decode | 1 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R308](#r308)<a name="r308"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 6 × 512 × 192 | 64 | decode | 1 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R309](#r309)<a name="r309"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 12 × 256 × 512 | 64 | decode | 1 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R310](#r310)<a name="r310"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 12 × 512 × 192 | 64 | decode | 1 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R311](#r311)<a name="r311"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 24 × 256 × 512 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R312](#r312)<a name="r312"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 24 × 512 × 192 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R313](#r313)<a name="r313"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 48 × 256 × 512 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R314](#r314)<a name="r314"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 48 × 512 × 192 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R315](#r315)<a name="r315"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 2560 × 256 × 512 | 64 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R316](#r316)<a name="r316"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 2560 × 512 × 192 | 64 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R317](#r317)<a name="r317"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 4608 × 256 × 512 | 64 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R318](#r318)<a name="r318"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 4608 × 512 × 192 | 64 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R319](#r319)<a name="r319"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 6656 × 256 × 512 | 64 | prefill | 8 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R320](#r320)<a name="r320"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 6656 × 512 × 192 | 64 | prefill | 8 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |

<a name="model-kimi26-w4a16"></a>

### Kimi-2.6 W4A16

128k/1k、s2000；包含 EAGLE3 draft。

需求来源：`shape_09_kimi26_128k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R321](#r321)<a name="r321"></a> | linear<br>BF16×BF16→BF16 | `eagle3_fc` | 4 × 7168 × 21504 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R322](#r322)<a name="r322"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode`<br>`lm_head_prefill` | 4 × 163840 × 7168 | — | prefill/decode | 3 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R323](#r323)<a name="r323"></a> | linear<br>BF16×BF16→BF16 | `fused_qkv_a_proj` | 16 × 2112 × 7168 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R324](#r324)<a name="r324"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 16 × 7168 × 8192 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R325](#r325)<a name="r325"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 16 × 7168 × 18432 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R326](#r326)<a name="r326"></a> | linear<br>BF16×BF16→BF16 | `q_b_proj` | 16 × 12288 × 1536 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R327](#r327)<a name="r327"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 16 × 36864 × 7168 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R328](#r328)<a name="r328"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 16 × 163840 × 7168 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R329](#r329)<a name="r329"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 32 × 20480 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R330](#r330)<a name="r330"></a> | linear<br>BF16×BF16→BF16 | `fused_qkv_a_proj` | 5120 × 2112 × 7168 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R331](#r331)<a name="r331"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 5120 × 4608 × 7168 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R332](#r332)<a name="r332"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 5120 × 7168 × 2304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R333](#r333)<a name="r333"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 5120 × 7168 × 8192 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R334](#r334)<a name="r334"></a> | linear<br>BF16×BF16→BF16 | `q_b_proj` | 5120 × 12288 × 1536 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R335](#r335)<a name="r335"></a> | linear<br>BF16×BF16→BF16 | `kv_b_proj` | 5120 × 16384 × 512 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R336](#r336)<a name="r336"></a> | linear<br>BF16×BF16→BF16 | `fused_qkv_a_proj` | 6656 × 2112 × 7168 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R337](#r337)<a name="r337"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 6656 × 4608 × 7168 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R338](#r338)<a name="r338"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 6656 × 7168 × 2304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R339](#r339)<a name="r339"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 6656 × 7168 × 8192 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R340](#r340)<a name="r340"></a> | linear<br>BF16×BF16→BF16 | `q_b_proj` | 6656 × 12288 × 1536 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R341](#r341)<a name="r341"></a> | linear<br>BF16×BF16→BF16 | `kv_b_proj` | 6656 × 16384 × 512 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R342](#r342)<a name="r342"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 40960 × 4608 × 7168 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R343](#r343)<a name="r343"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 40960 × 7168 × 2304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R344](#r344)<a name="r344"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 53248 × 4608 × 7168 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R345](#r345)<a name="r345"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 53248 × 7168 × 2304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R346](#r346)<a name="r346"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 16 × 128 × 512 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R347](#r347)<a name="r347"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 16 × 512 × 128 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm]<br>接口 smoke 通过 |
| [R348](#r348)<a name="r348"></a> | linear<br>BF16×INT4→BF16 | `eagle3_o_proj` | 4 × 7168 × 8192 | — | decode | 1 | — | <code>BF16*@1×32（Excel）</code> | [awq_gemm_triton][fn-awq]<br>接口 smoke 通过 |
| [R349](#r349)<a name="r349"></a> | linear<br>BF16×INT4→BF16 | `eagle3_down_proj` | 4 × 7168 × 12288 | — | decode | 1 | — | <code>BF16*@1×32（Excel）</code> | [awq_gemm_triton][fn-awq]<br>接口 smoke 通过 |
| [R350](#r350)<a name="r350"></a> | linear<br>BF16×INT4→BF16 | `eagle3_gate_up_proj` | 4 × 24576 × 7168 | — | decode | 1 | — | <code>BF16*@1×32（Excel）</code> | [awq_gemm_triton][fn-awq]<br>接口 smoke 通过 |
| [R351](#r351)<a name="r351"></a> | linear<br>BF16×INT4→BF16 | `eagle3_qkv_proj` | 4 × 24576 × 14336 | — | decode | 1 | — | <code>BF16*@1×32（Excel）</code> | [awq_gemm_triton][fn-awq]<br>接口 smoke 通过 |

<a name="model-hy3-fp8"></a>

### Hunyuan-V3 FP8

128k/1k、s2000；激活 FP32/token/dyn，权重 scale 为 FP32*/channel，以新版标签为准。

需求来源：`shape_10_hy3_128k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R352](#r352)<a name="r352"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 3 × 120832 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R353](#r353)<a name="r353"></a> | linear<br>BF16×BF16→BF16 | `nextn_eh_proj` | 4 × 4096 × 8192 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R354](#r354)<a name="r354"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 4 × 120832 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R355](#r355)<a name="r355"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 16 × 120832 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R356](#r356)<a name="r356"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 24 × 15104 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R357](#r357)<a name="r357"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 4 × 4096 × 8192 | — | decode | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R358](#r358)<a name="r358"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 4 × 10240 × 4096 | — | decode | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R359](#r359)<a name="r359"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 16 × 4096 × 8192 | — | decode | 2 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R360](#r360)<a name="r360"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 16 × 4096 × 13312 | — | decode | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R361](#r361)<a name="r361"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 16 × 10240 × 4096 | — | decode | 2 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R362](#r362)<a name="r362"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 16 × 26624 × 4096 | — | decode | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R363](#r363)<a name="r363"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 512 × 3328 × 4096 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R364](#r364)<a name="r364"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 512 × 4096 × 1664 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R365](#r365)<a name="r365"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 512 × 4096 × 8192 | — | prefill | 4 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R366](#r366)<a name="r366"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 512 × 10240 × 4096 | — | prefill | 4 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R367](#r367)<a name="r367"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 4096 × 3328 × 4096 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R368](#r368)<a name="r368"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 4096 × 4096 × 1664 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R369](#r369)<a name="r369"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 6656 × 3328 × 4096 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R370](#r370)<a name="r370"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 6656 × 4096 × 1664 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R371](#r371)<a name="r371"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 6656 × 4096 × 8192 | — | prefill | 4 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R372](#r372)<a name="r372"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 6656 × 10240 × 4096 | — | prefill | 4 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R373](#r373)<a name="r373"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 53248 × 3328 × 4096 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R374](#r374)<a name="r374"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 53248 × 4096 × 1664 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |

<a name="model-q397-fp8"></a>

### Qwen3.5-397B-A17B FP8

128k/1k、s2000；本模型属于后续支持 P2，不在本轮三个 Qwen P0 配置内。

需求来源：`shape_11_q397b_128k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R375](#r375)<a name="r375"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 8 × 248320 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R376](#r376)<a name="r376"></a> | linear<br>BF16×BF16→BF16 | `mtp_fc` | 48 × 4096 × 8192 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R377](#r377)<a name="r377"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 48 × 124160 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R378](#r378)<a name="r378"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 64 × 31040 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R379](#r379)<a name="r379"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 192 × 64 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R380](#r380)<a name="r380"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 192 × 124160 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R381](#r381)<a name="r381"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 3584 × 128 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R382](#r382)<a name="r382"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 6656 × 128 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R383](#r383)<a name="r383"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 28672 × 16 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R384](#r384)<a name="r384"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 53248 × 16 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16]<br>接口 smoke 通过 |
| [R385](#r385)<a name="r385"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 48 × 4096 × 4096 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R386](#r386)<a name="r386"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 48 × 8704 × 4096 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R387](#r387)<a name="r387"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 192 × 4096 × 4096 | — | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R388](#r388)<a name="r388"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 192 × 8704 × 4096 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R389](#r389)<a name="r389"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 192 × 10240 × 4096 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R390](#r390)<a name="r390"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 3584 × 4096 × 8192 | — | prefill | 3 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R391](#r391)<a name="r391"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 3584 × 17408 × 4096 | — | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R392](#r392)<a name="r392"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 3584 × 20480 × 4096 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R393](#r393)<a name="r393"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 6656 × 4096 × 8192 | — | prefill | 3 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R394](#r394)<a name="r394"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 6656 × 17408 × 4096 | — | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R395](#r395)<a name="r395"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 6656 × 20480 × 4096 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R396](#r396)<a name="r396"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 28672 × 2560 × 4096 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R397](#r397)<a name="r397"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `out_proj` | 28672 × 4096 × 1024 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R398](#r398)<a name="r398"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 53248 × 2560 × 4096 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |
| [R399](#r399)<a name="r399"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `out_proj` | 53248 × 4096 × 1024 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-fp8-aiter]<br>[w8a8_triton_block_scaled_mm][fn-fp8-vllm]<br>接口 smoke 通过 |



[fn-bmm]: ../../boltops/gemm/triton/batched_gemm_bf16.py#L323
[fn-a16]: ../../boltops/gemm/triton/gemm_a16w16.py#L33
[fn-persistent]: ../../boltops/gemm/triton/matmul_persistent.py#L205
[fn-fp8-aiter]: ../../boltops/gemm/triton/gemm_a8w8_blockscale.py#L29
[fn-fp8-vllm]: ../../boltops/gemm/triton/w8a8_triton_block_scaled_mm.py#L127
[fn-awq]: ../../boltops/gemm/triton/awq_gemm_triton.py#L351
