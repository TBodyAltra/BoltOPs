# PMD 上通过 launch.py 验证六组 quant_matmul

日期：2026-11-05。源版本 `71a12101509635421d02c3d28c9202951ded7745`；将固定提交的 `git archive` 传至 PMD，避免远端网络和分支漂移。来源见 [source.json](source.json)。远端目录：`/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105`。

机台环境：容器 `c45e31c9b3d9`，PMD gfx946 / Device 66b7，单 die、48 CU、模型频率 2.2 GHz、64 GiB 显存；PMD CoreArch HEAD_1703（2026-07-30 构建），PyTorch `2.6.0+d8c4c7c` / HIP `6.3.42133-88df948`。加载既有 `env-native-v2.sh` 后，`torch.cuda.is_available()` 为 true、device_count=1；此处 torch.cuda 是 ROCm PyTorch 的设备 API 名称，不是 NVIDIA 设备。

## 运行方式和补丁范围

这次直接运行 `projects/micro_perf/launch.py`，走其 GPU backend、provider 注册、ComputeEngine 子进程和原始 QuantMatmulOp，不再使用本机 CPU device 选择器代替 benchmark 入口。六组分别选用原 Excel 中的代表 MNK：**M=1、N=8192、K=1024，sp_size=1**。没有扫描全部 1104 个 GPU 性能组合。

| 阶段 | 代码副本 | 运行结果 |
| --- | --- | --- |
| 原版 baseline | upstream | 设备识别成功；收集 GPU 信息时调用 nvidia-smi，因命令不存在而退出，尚未进入算子 |
| 五组非 INT8 | compat | 对 nvidia-smi 缺失做兼容后，五组均进入原始 vendor_parser，并抛不支持该精度组合的 ValueError |
| INT8 单次诊断 | smoke | GPU 输入创建、INT8→BF16 转换已执行；BF16 mm 因缺少 gfx946 BLAS 库而未完成 |

兼容补丁 [nvidia-info-compat.patch](nvidia-info-compat.patch) 仅让 GPU 信息采集在缺失 nvidia-smi 时保留空 driver_version，继续使用 PyTorch 取得的真实设备信息。原版 backend 在 AMD/PMD 上不能原样通过这一步。

单次诊断补丁 [pmd-one-call-smoke.patch](pmd-one-call-smoke.patch) 限制为一份输入、零 warmup、一次调用，并记录 dtype/shape 与 aten 调用。上游默认逻辑会为缓存规避复制约 1 GiB 数据，不适合作为首次 PMD 冒烟。因此这次不是上游默认迭代配置的性能测量。

**launch.py、quant_matmul.py、core/utils.py 三份文件，在 upstream / compat / smoke 副本中均逐字相同**，哈希见 [provenance.json](provenance.json)。未改精度 gate、量化方式或计算实现，未给 BF16/FP8/FP4/MX 增加实现。

实际命令形式如下，完整环境与锁命令保存在各阶段 run.sh / executor.json：

```bash
python3 -u ../compat/projects/micro_perf/launch.py \
  --backend GPU --workload ../workloads/five-unsupported.json \
  --numa=-1 --device 0 --report_dir reports

python3 -u ../smoke/projects/micro_perf/launch.py \
  --backend GPU --workload ../workloads/int8.json \
  --numa=-1 --device 0 --report_dir reports
```

## 六组实际结果

| A / W / compute / 输出 | launch.py 内部结果 | 可确认的信息 |
| --- | --- | --- |
| BF16 / BF16 / BF16 / BF16 | 原始 vendor_parser 抛 ValueError | BF16 的拒绝在算子参数阶段；不是 GPU 数值计算失败 |
| INT8 / INT8 / INT8 / BF16 | tensor 创建与转换已执行；GEMM 因环境库缺失失败 | A=INT8[1,1024]、W=INT8[8192,1024]；A scale=FP32[1]、W scale=FP32[8192]；mm 输入已转为 BF16 |
| FP8 / FP8 / FP8 / BF16 | 原始 vendor_parser 抛 ValueError | 未创建该精度的 GPU 输入或 scale |
| FP4 / FP4 / FP4 / BF16 | 原始 vendor_parser 抛 ValueError | 未创建该精度的 GPU 输入或 scale |
| MXFP8 / MXFP8 / MXFP8 / BF16 | 原始 vendor_parser 抛 ValueError | 未执行 MX 标准 scale 或分组 |
| MXFP4 / MXFP4 / MXFP4 / BF16 | 原始 vendor_parser 抛 ValueError | 未执行 MX 标准 scale 或分组 |

PMD 的 PyTorch 2.6 没有 `torch.float4_e2m1fn_x2` 属性，故其工具运行时也不会注册条件分支中的 mxfloat4 dtype。此前本机 PyTorch 2.14 的映射/creator 抓取不能直接当作 PMD 2.6 已具备的 dtype 支持。标准 MXFP4 的 E2M1 / E8M0 / 32 元素约定不因此改变；自有实现仍可另行选择原始字节存储接口。

INT8 原始 creator 仍是零值数据、全 1 scale。保存的 [int8-contract.json](smoke-int8/int8-contract.json) 是算子元数据，实际 GPU tensor 与转换 dtype 见 [smoke-int8/launch.txt](smoke-int8/launch.txt) 的 PMD_OP_BEGIN / END。输出 BF16[1,8192] 仅为声明，因 mm 未完成，没有完成输出或数值校验。

## INT8 的环境失败与收尾

日志中的原始缺库错误：

```text
rocblaslt error: Cannot read /opt/rocm/lib/hipblaslt/library/TensileLibrary_lazy_gfx946.dat
rocBLAS error: Cannot read /opt/rocm/lib/rocblas/library/TensileLibrary.dat:
  No such file or directory for GPU arch : gfx946
```

hipBLASLt 失败后回退 rocBLAS，后者同样缺少 gfx946 库。检查 `/opt/rocm`、`/opt/hyhal`、`/xukang/code` 未找到匹配的 TensileLibrary gfx946 数据文件。未通过改架构名、替换不匹配的库或修改算子来伪造通过。

GPU 工作子进程因缺库退出，launch.py 父进程仍等待结果。在确认已无存活的 spawn GPU worker 后，仅向本次等待中的 launch.py 调度进程发送 SIGINT，记录见 [controller-cleanup.json](smoke-int8/controller-cleanup.json)。测试锁已释放，未重置设备。

五组拒绝与 INT8 失败后收尾的 shell 退出码均为 0，但没有有效性能结果，**退出 0 不表示测试通过**。本次没有新增 GPU GEMM 完成、正确性 PASS、性能 PASS，未生成有效的 latency、算力、带宽、MFU、MBU。

日志和补丁保留原始字节，包括上游输出和源码中的行尾空白。

## 原始记录

- [baseline/launch.txt](baseline/launch.txt)：原版入口及 nvidia-smi 错误。
- [compat-five/launch.txt](compat-five/launch.txt)：五组在 launch.py 子进程中的完整参数拒绝记录。
- [smoke-int8/launch.txt](smoke-int8/launch.txt)：GPU dispatch、真实 dtype 轨迹和缺库错误。
- [verification.json](verification.json)：阶段状态、环境、退出码、无 GEMM 完成、配置 SHA256。
- `workloads/`：各精度代表用例与五组联合用例。
- 每个阶段的 `run.sh`、`executor.json`、`exit-code`；compat-five/reports 中保留原始 info.json。

完整 config.ini 等保留在远端各阶段 m5out，以及本机 `/home/kxu/byte-gemm-requirements-20261105/pmd-launch/pmd-artifacts.tar.gz`。仓库保留其 SHA256，避免重复提交三份 3 万余行的模拟器配置。

[补充：标准安装复核及 quant_matmul 实际 provider 注册表](../repository_audit_20261105/README.md)
