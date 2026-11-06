# xpu-perf 运行方式与 provider 路由核查

日期：2026-11-05。分析对象为公开 `main`，提交 `71a12101509635421d02c3d28c9202951ded7745`；远端当前只有 main 分支。结论：launch.py 与 workload 入口使用正确；实际接入的是 base 参考实现，尚无海光 quant_matmul provider，PMD 的现有 BLAS 配套也不完整。此前的拒绝与缺库结果不能当成硬件不支持六组精度的结论。

## 仓库各部分的职责

| 位置 | 职责 | 本次相关事实 |
| --- | --- | --- |
| projects/micro_perf/launch.py | 本地 benchmark 入口 | README 仍提供该启动方式，支持 --backend、--task_dir/--task、--workload、--vendor_ops |
| projects/micro_perf/server.py + client.py | 远程请求入口 | 同样调用 XpuPerfServer 及 normal_bench；切换 server 不会新增 quant_matmul 实现 |
| src/xpu_perf/micro_perf/backends | 设备管理、计时和分发 | 当前公开树只有 GPU backend；其中含 nvidia-smi 信息采集，PMD 需要适配 |
| projects/micro_perf/op_defs | 算子接口、基础参数及 base 参考实现 | quant_matmul 的 base 仅接受 INT8/INT8/INT8→BF16，并用 BF16 mm 模拟 |
| projects/micro_perf/vendor_ops | 厂商或算子库的具体 provider | GPU 目录没有注册 quant_matmul 的 vendor 实现 |
| projects/micro_perf/workloads | 参数与测试矩阵 | 配置列出某精度，不表示对应 kernel 已在公开树中实现 |

当前实际路径：

```text
launch.py
  → XpuPerfServer
  → GPU backend
  → ProviderRegistry[quant_matmul]
  → base QuantMatmulOp
  → fake_quant_gemm
  → torch.matmul(BF16)
  → hipBLASLt / rocBLAS
```

## 已排除的运行方式问题

1. **入口未选错。** 项目 README 明确保留 launch.py 本地入口。server/client 是另一种请求方式，两者使用相同的 engine/backend/provider。
2. **workload 指定有效。** 用上游解析器实测，`--task_dir workloads/llm/vendor_test --task quant_matmul` 与 `--workload workloads/llm/vendor_test/quant_matmul.json` 展开结果逐项相同。默认示例只有四套精度，共 736 个组合；按 Excel 增加 BF16、普通 FP4 后为六套，共 1104 个。`6192` 和非对称 K/N 方向均保留。
3. **没有漏取 git submodule。** 当前提交的 gitlink 数量为 0，vendor_ops 是普通目录，不是需要另行 init 的 git submodule；pyproject.toml 也没有通过插件 entry-point 额外下载算子实现的配置。
4. **排除了 PYTHONPATH 与 editable 安装差异。** 原先直接从指定源码导入，文件路径及哈希均正确。本次又在 PMD 新建独立 venv，按 README 使用 editable 安装源项目，成功安装 xpu-perf 0.1.0 后，provider 注册结果完全相同。安装未覆盖原有 PMD 共享 venv。

安装复核使用 `--no-deps --no-build-isolation`，离线准备 setuptools 78.1.0 / wheel 0.45.1，沿用机台已有 PyTorch 2.6、numpy 1.26.4，并从独立 deps 目录加载此前准备的 jsonlines / prettytable 等依赖。目的是验证源码包的标准 editable 安装及 provider 发现，与先前 PYTHONPATH 加载做对照，没有升级 PMD PyTorch 或替换系统库。命令、安装输出和两份注册表均保留。

## PMD 实际注册表

两种安装/导入方式均得到：

```text
torch provider : gemm
vllm provider  : rms_norm
quant_matmul  : base → QuantMatmulOp
```

- [provider-before.json](provider-before.json)：源码路径直接导入后的注册结果。
- [provider-after-editable.json](provider-after-editable.json)：editable 安装后结果，含 xpu-perf 0.1.0 的安装元数据。
- [editable-install.txt](editable-install.txt)：安装及检查输出。
- [check-editable.sh](check-editable.sh)、[inspect_registry.py](inspect_registry.py)：实际执行的检查脚本。
- [workload-and-source-check.json](workload-and-source-check.json)：参数展开等价性、计数、submodule 和插件入口检查。

公开 GPU vendor 文件注册的算子只有 flash_attention、rms_norm、gemm；没有 quant_matmul 的 vendor 装饰器。PMD 上没有出现 quant_matmul vendor 导入失败后被跳过的情况：公开目录中根本没有对应实现。可选 flash-attn/flashinfer 是否加载不会改变这一点。本次也没有通过 --vendor_ops 指定仓库外的海光私有实现。

ProviderRegistry 先加载 op_defs，再加载默认 GPU vendor 路径与 --vendor_ops 附加路径；**只有一个算子没有任何 vendor 注册时，才回退到其 base**。它不会根据 dtype 自动把 quant_matmul 转发到另一个 gemm 算子，也不会根据 compute_dtype 自动生成低精度 kernel。

## 对已有报错的归因

| 现象 | 归因 | 需要怎样处理 |
| --- | --- | --- |
| 旧 --hardware_type 不识别 | Excel 命令与当前 CLI 版本不同 | 使用 --backend GPU；此前已调整 |
| 直接 --task quant_matmul 无用例 | 默认 task_dir 是 workloads/basic | 指向 llm workload，或直接 --workload；此前已正确指定 |
| 缺少 nvidia-smi | GPU backend 信息采集包含 NVIDIA 环境假设 | 提供 PMD/ROCm 信息采集适配；已有独立兼容补丁 |
| BF16/FP8/FP4/MX 的 ValueError | 当前被选中的 base 精度 gate | 接入真正支持这些配置的 vendor provider |
| INT8 在 BF16 mm 缺 gfx946 Tensile 库 | base 参考路径依赖当前环境的 BLAS 配套 | 补齐匹配库，或让自有 provider 直接调用我们的 kernel |
| exit 0 却没有有效结果 | 框架对算子异常返回空 targets，父进程失败收尾也可能退出 0 | 以有效结果、原始异常及数值验证判定；不能只看 shell 状态 |

INT8 的 BF16 mm 是 base 代码的行为，与 MNK、CLI 选择或 GPU 的原生 INT8 支持无关。当前两份 PMD 适配补丁没有改 launch.py、quant_matmul.py 或 core/utils.py，其哈希证明见 [PMD 运行记录](../pmd_launch_20261105/README.md)。本次注册表检查没有重新执行 GEMM。

## 我们自己的实现如何接入

保留原始需求 workload；新增海光 provider，例如 `hygon/ops/hcu/` 下用 `PROVIDER_NAME="hcu"` 标记包，并通过 `@ProviderRegistry.register_vendor_impl("quant_matmul", "hcu")` 注册具体类。该类需要实现自己的精度检查、tensor/scale 元数据和实际 kernel 调用。框架可组合其与 base 的继承关系，但继承的 vendor_parser 仍限制 INT8，因此必须明确覆盖支持六组配置的检查。

命令形式为：

```bash
python launch.py --backend GPU \
  --vendor_ops /path/to/hygon/ops \
  --workload /path/to/quant-matmul.json
```

同时完善 GPU backend 的 PMD 信息采集和适合模拟器的测试迭代设置。自有 provider 直接调用我们的 kernel 后，可以消除 base 的 fake_quant_gemm 对 BLAS 的依赖；单纯删除 base gate 仍然只会做模拟计算，不能作为六组原生精度的性能实现。

接口层仍需明确普通 FP4 的编码/packing/scale；MX 的 E8M0/32 元素标准已确定，但物理矩阵分组轴与 scale 布局需定义。框架的 perf 路径不自动做完整数值正确性验收，量化参考输入又是零值数据/全 1 scale；自有实现需要独立的非平凡输入及 scale 验证，再产出性能结果。

## 源码依据

以下均固定为上述提交：

- [项目 README](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/projects/micro_perf/README.md)：安装、本地/远程入口。
- [launch.py](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/projects/micro_perf/launch.py)：参数和 workload 入口。
- [core/op.py](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/src/xpu_perf/micro_perf/core/op.py)：provider 注册、继承组合和 base 回退。
- [core/perf_engine.py](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/src/xpu_perf/micro_perf/core/perf_engine.py)：附加 vendor_ops、engine 及分发。
- [quant_matmul.py](https://github.com/bytedance/xpu-perf/blob/71a12101509635421d02c3d28c9202951ded7745/projects/micro_perf/op_defs/llm_ops/quant_matmul.py)：参数 gate、tensor 和 scale。
- [GPU vendor 目录](https://github.com/bytedance/xpu-perf/tree/71a12101509635421d02c3d28c9202951ded7745/projects/micro_perf/vendor_ops/GPU/ops)：公开 provider 实现。
