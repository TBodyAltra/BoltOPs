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

<table>
  <thead>
    <tr>
      <th rowspan="4">编号</th>
      <th rowspan="4">算子 / 精度</th>
      <th rowspan="4">层 / 算子名</th>
      <th rowspan="4">M × N × K</th>
      <th rowspan="4">batch / E</th>
      <th rowspan="4">阶段</th>
      <th rowspan="4">来源数</th>
      <th rowspan="4">A scale</th>
      <th rowspan="4">B scale</th>
      <th rowspan="4">公开接口</th>
      <th colspan="16">性能</th>
    </tr>
    <tr>
      <th colspan="8">1 DIE</th>
      <th colspan="8">2 DIE</th>
    </tr>
    <tr>
      <th colspan="4">TT</th>
      <th colspan="4">TN</th>
      <th colspan="4">TT</th>
      <th colspan="4">TN</th>
    </tr>
    <tr>
      <th>测试环境</th>
      <th>Kernel（μs / TOPS）</th>
      <th>产物</th>
      <th>单 TG PERF</th>
      <th>测试环境</th>
      <th>Kernel（μs / TOPS）</th>
      <th>产物</th>
      <th>单 TG PERF</th>
      <th>测试环境</th>
      <th>Kernel（μs / TOPS）</th>
      <th>产物</th>
      <th>单 TG PERF</th>
      <th>测试环境</th>
      <th>Kernel（μs / TOPS）</th>
      <th>产物</th>
      <th>单 TG PERF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="#r001">R001</a><a name="r001"></a></td>
      <td>linear<br>BF16×BF16→BF16</td>
      <td><code>in_proj_ba</code></td>
      <td>64 × 64 × 2048</td>
      <td>—</td>
      <td>decode</td>
      <td>1</td>
      <td>—</td>
      <td>—</td>
      <td><a href="../../../boltops/gemm/gemm_a16w16.py">gemm_a16w16</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td><a href="tuning/r001/1die/tn/record.json">flags / env</a></td>
      <td><a href="tuning/r001/1die/tn/record.json">5.814 / 2.89</a></td>
      <td><a href="https://github.com/TBodyAltra/BoltOPs/blob/408cf70489a34b217ce9a12dae6177ffac9b3d46/boltops/gemm/gluon/gemm_a16w16.py#L34">源码</a> / <a href="https://github.com/TBodyAltra/BoltOPs/blob/408cf70489a34b217ce9a12dae6177ffac9b3d46/boltops/gemm/configs/gfx946/48cu/gluon/gemm_a16w16/GEMM-A16W16-N=64-K=2048-TN.json">config</a> / <a href="tuning/r001/1die/tn/kernel.amdgcn">汇编</a><br><a href="tuning/r001/1die/tn/benchmark.py">测试代码</a> / <a href="tuning/r001/1die/tn/record.json">记录</a></td>
      <td><a href="tuning/r001/1die/tn/single_tg.perf">PERF</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r002">R002</a><a name="r002"></a></td>
      <td>linear<br>BF16×BF16→BF16</td>
      <td><code>lm_head_decode</code><br><code>lm_head_prefill</code></td>
      <td>64 × 248320 × 2048</td>
      <td>—</td>
      <td>prefill/decode</td>
      <td>3</td>
      <td>—</td>
      <td>—</td>
      <td><a href="../../../boltops/gemm/gemm_a16w16.py">gemm_a16w16</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r003">R003</a><a name="r003"></a></td>
      <td>linear<br>BF16×BF16→BF16</td>
      <td><code>in_proj_ba</code></td>
      <td>2560 × 64 × 2048</td>
      <td>—</td>
      <td>prefill</td>
      <td>1</td>
      <td>—</td>
      <td>—</td>
      <td><a href="../../../boltops/gemm/gemm_a16w16.py">gemm_a16w16</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r004">R004</a><a name="r004"></a></td>
      <td>linear<br>BF16×BF16→BF16</td>
      <td><code>in_proj_ba</code></td>
      <td>6656 × 64 × 2048</td>
      <td>—</td>
      <td>prefill</td>
      <td>1</td>
      <td>—</td>
      <td>—</td>
      <td><a href="../../../boltops/gemm/gemm_a16w16.py">gemm_a16w16</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r005">R005</a><a name="r005"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>o_proj</code><br><code>out_proj</code></td>
      <td>64 × 2048 × 4096</td>
      <td>—</td>
      <td>decode</td>
      <td>2</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r006">R006</a><a name="r006"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>qkv_proj</code></td>
      <td>64 × 9216 × 2048</td>
      <td>—</td>
      <td>decode</td>
      <td>1</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r007">R007</a><a name="r007"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>in_proj_qkvz</code></td>
      <td>64 × 12288 × 2048</td>
      <td>—</td>
      <td>decode</td>
      <td>1</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r008">R008</a><a name="r008"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>o_proj</code><br><code>out_proj</code></td>
      <td>2560 × 2048 × 4096</td>
      <td>—</td>
      <td>prefill</td>
      <td>2</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r009">R009</a><a name="r009"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>qkv_proj</code></td>
      <td>2560 × 9216 × 2048</td>
      <td>—</td>
      <td>prefill</td>
      <td>1</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td><a href="tuning/r009/1die/tn/record.json">flags / env</a></td>
      <td><a href="tuning/r009/1die/tn/record.json">197.728 / 488.74</a></td>
      <td><a href="https://github.com/TBodyAltra/BoltOPs/blob/be845a86e988ba2fac6efe07b73b497dbcd949e2/boltops/gemm/gluon/gemm_a8w8_blockscale_persistent.py#L396">源码</a> / <a href="https://github.com/TBodyAltra/BoltOPs/blob/be845a86e988ba2fac6efe07b73b497dbcd949e2/boltops/gemm/configs/gfx946/48cu/gluon/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=9216-K=2048.json">config</a> / <a href="tuning/r009/1die/tn/kernel.amdgcn">汇编</a><br><a href="tuning/r009/1die/tn/benchmark.py">测试代码</a> / <a href="tuning/r009/1die/tn/record.json">记录</a></td>
      <td><a href="tuning/r009/1die/tn/single_tg.perf">PERF</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r010">R010</a><a name="r010"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>in_proj_qkvz</code></td>
      <td>2560 × 12288 × 2048</td>
      <td>—</td>
      <td>prefill</td>
      <td>1</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td><a href="tuning/r010/1die/tn/record.json">flags / env</a></td>
      <td><a href="tuning/r010/1die/tn/record.json">292.620 / 440.33</a></td>
      <td><a href="https://github.com/TBodyAltra/BoltOPs/blob/62a368fb3824d0895779000f03163e7771f13575/boltops/gemm/gluon/gemm_a8w8_blockscale.py#L328">源码</a> / <a href="https://github.com/TBodyAltra/BoltOPs/blob/35bda3a379adb99cac6ad53279969c41e64e0a62/boltops/gemm/configs/gfx946/48cu/gluon/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=12288-K=2048.json">config</a> / <a href="tuning/r010/1die/tn/kernel.amdgcn">汇编</a><br><a href="tuning/r010/1die/tn/benchmark.py">测试代码</a> / <a href="tuning/r010/1die/tn/record.json">记录</a></td>
      <td><a href="tuning/r010/1die/tn/single_tg.perf">PERF</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r011">R011</a><a name="r011"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>o_proj</code><br><code>out_proj</code></td>
      <td>6656 × 2048 × 4096</td>
      <td>—</td>
      <td>prefill</td>
      <td>2</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td><a href="tuning/r011/1die/tn/record.json">flags / env</a></td>
      <td><a href="tuning/r011/1die/tn/record.json">250.435 / 445.90</a></td>
      <td><a href="https://github.com/TBodyAltra/BoltOPs/blob/64dbaea9fad7a4f510cb02d6eba78bfed8865b5c/boltops/gemm/gluon/gemm_a8w8_blockscale.py#L328">源码</a> / <a href="https://github.com/TBodyAltra/BoltOPs/blob/bb60e65d027f85d2f6aa2e6e85ef4576e68ee31b/boltops/gemm/configs/gfx946/48cu/gluon/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=2048-K=4096.json">config</a> / <a href="tuning/r011/1die/tn/kernel.amdgcn">汇编</a><br><a href="tuning/r011/1die/tn/benchmark.py">测试代码</a> / <a href="tuning/r011/1die/tn/record.json">记录</a></td>
      <td><a href="tuning/r011/1die/tn/single_tg.perf">PERF</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r012">R012</a><a name="r012"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>qkv_proj</code></td>
      <td>6656 × 9216 × 2048</td>
      <td>—</td>
      <td>prefill</td>
      <td>1</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td><a href="tuning/r012/1die/tn/record.json">flags / env</a></td>
      <td><a href="tuning/r012/1die/tn/record.json">599.711 / 418.96</a></td>
      <td><a href="https://github.com/TBodyAltra/BoltOPs/blob/f615d17e64032102cab49a82c7fd717e63f46e0b/boltops/gemm/gluon/gemm_a8w8_blockscale.py#L328">源码</a> / <a href="https://github.com/TBodyAltra/BoltOPs/blob/35bda3a379adb99cac6ad53279969c41e64e0a62/boltops/gemm/configs/gfx946/48cu/gluon/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=9216-K=2048.json">config</a> / <a href="tuning/r012/1die/tn/kernel.amdgcn">汇编</a><br><a href="tuning/r012/1die/tn/benchmark.py">测试代码</a> / <a href="tuning/r012/1die/tn/record.json">记录</a></td>
      <td><a href="tuning/r012/1die/tn/single_tg.perf">PERF</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><a href="#r013">R013</a><a name="r013"></a></td>
      <td>linear<br>FP8E4M3×FP8E4M3→BF16</td>
      <td><code>in_proj_qkvz</code></td>
      <td>6656 × 12288 × 2048</td>
      <td>—</td>
      <td>prefill</td>
      <td>1</td>
      <td>FP32@1×128（动态）</td>
      <td>BF16@128×128（Excel）</td>
      <td><a href="../../../boltops/gemm/gemm_a8w8_blockscale.py">gemm_a8w8_blockscale</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td><a href="tuning/r013/1die/tn/record.json">flags / env</a></td>
      <td><a href="tuning/r013/1die/tn/record.json">777.029 / 431.14</a></td>
      <td><a href="https://github.com/TBodyAltra/BoltOPs/blob/6c4f4b537a7e99915a2b110c271cdd2b65e31ba8/boltops/gemm/gluon/gemm_a8w8_blockscale.py#L328">源码</a> / <a href="https://github.com/TBodyAltra/BoltOPs/blob/35bda3a379adb99cac6ad53279969c41e64e0a62/boltops/gemm/configs/gfx946/48cu/gluon/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=12288-K=2048.json">config</a> / <a href="tuning/r013/1die/tn/kernel.amdgcn">汇编</a><br><a href="tuning/r013/1die/tn/benchmark.py">测试代码</a> / <a href="tuning/r013/1die/tn/record.json">记录</a></td>
      <td><a href="tuning/r013/1die/tn/single_tg.perf">PERF</a></td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
  </tbody>
</table>


<a name="model-q35-mxfp4"></a>

### Qwen3.5-35B-A3B MXFP4

128k/1k、s150；TP1，单卡实例，chunk 6656。

需求来源：`shape_08_q35_mxfp4_128k_s150.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R014](#r014)<a name="r014"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 12 × 64 × 2048 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R015](#r015)<a name="r015"></a> | linear<br>BF16×BF16→BF16 | `o_proj`<br>`out_proj` | 12 × 2048 × 4096 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R016](#r016)<a name="r016"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 12 × 9216 × 2048 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R017](#r017)<a name="r017"></a> | linear<br>BF16×BF16→BF16 | `in_proj_qkvz` | 12 × 12288 × 2048 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R018](#r018)<a name="r018"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode`<br>`lm_head_prefill` | 12 × 248320 × 2048 | — | prefill/decode | 3 | — | — | [gemm_a16w16][fn-a16] |
| [R019](#r019)<a name="r019"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 2048 × 64 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R020](#r020)<a name="r020"></a> | linear<br>BF16×BF16→BF16 | `o_proj`<br>`out_proj` | 2048 × 2048 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R021](#r021)<a name="r021"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 2048 × 9216 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R022](#r022)<a name="r022"></a> | linear<br>BF16×BF16→BF16 | `in_proj_qkvz` | 2048 × 12288 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R023](#r023)<a name="r023"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 6656 × 64 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R024](#r024)<a name="r024"></a> | linear<br>BF16×BF16→BF16 | `o_proj`<br>`out_proj` | 6656 × 2048 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R025](#r025)<a name="r025"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 6656 × 9216 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R026](#r026)<a name="r026"></a> | linear<br>BF16×BF16→BF16 | `in_proj_qkvz` | 6656 × 12288 × 2048 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |

<a name="model-qwen3-vl-8b"></a>

### Qwen3-VL-8B BF16

主干输入 226 token、输出 300；文本 30 token + 448×448 图像；TP1/bs512。文件保留 s150 标签。

需求来源：`shape_12_qwen3vl8b_226_s150.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R027](#r027)<a name="r027"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 512 × 4096 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R028](#r028)<a name="r028"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 512 × 4096 × 12288 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R029](#r029)<a name="r029"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 512 × 6144 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R030](#r030)<a name="r030"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 512 × 24576 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R031](#r031)<a name="r031"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode`<br>`lm_head_prefill` | 512 × 151936 × 4096 | — | prefill/decode | 3 | — | — | [gemm_a16w16][fn-a16] |
| [R032](#r032)<a name="r032"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 2560 × 4096 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R033](#r033)<a name="r033"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 2560 × 4096 × 12288 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R034](#r034)<a name="r034"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 2560 × 6144 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R035](#r035)<a name="r035"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 2560 × 24576 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R036](#r036)<a name="r036"></a> | linear<br>BF16×BF16→BF16 | `merger_fc2` | 5684 × 4096 × 4608 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R037](#r037)<a name="r037"></a> | linear<br>BF16×BF16→BF16 | `merger_fc1` | 5684 × 4608 × 4608 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R038](#r038)<a name="r038"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 6656 × 4096 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R039](#r039)<a name="r039"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 6656 × 4096 × 12288 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R040](#r040)<a name="r040"></a> | linear<br>BF16×BF16→BF16 | `qkv_proj` | 6656 × 6144 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R041](#r041)<a name="r041"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 6656 × 24576 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R042](#r042)<a name="r042"></a> | linear<br>BF16×BF16→BF16 | `vit_o_proj` | 22736 × 1152 × 1152 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R043](#r043)<a name="r043"></a> | linear<br>BF16×BF16→BF16 | `vit_patch_embed` | 22736 × 1152 × 1536 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R044](#r044)<a name="r044"></a> | linear<br>BF16×BF16→BF16 | `vit_mlp_fc2` | 22736 × 1152 × 4304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R045](#r045)<a name="r045"></a> | linear<br>BF16×BF16→BF16 | `vit_qkv_proj` | 22736 × 3456 × 1152 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R046](#r046)<a name="r046"></a> | linear<br>BF16×BF16→BF16 | `vit_mlp_fc1` | 22736 × 4304 × 1152 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |

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

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R047](#r047)<a name="r047"></a> | gemm<br>bfloat16 × bfloat16 → bfloat16 | `gemm` | M₇₂ × 4096 × 4096 | — | — | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R048](#r048)<a name="r048"></a> | gemm<br>bfloat16 × bfloat16 → bfloat16 | `gemm` | M₇₂ × 8192 × 8192 | — | — | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R049](#r049)<a name="r049"></a> | gemm<br>bfloat16 × bfloat16 → bfloat16 | `gemm` | M₇₂ × 1024 × 8192 | — | — | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R050](#r050)<a name="r050"></a> | gemm<br>bfloat16 × bfloat16 → bfloat16 | `gemm` | M₇₂ × 8192 × 1024 | — | — | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R051](#r051)<a name="r051"></a> | gemm<br>int8 × int8 → bfloat16 | `gemm` | M₇₂ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R052](#r052)<a name="r052"></a> | gemm<br>int8 × int8 → bfloat16 | `gemm` | M₇₂ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R053](#r053)<a name="r053"></a> | gemm<br>int8 × int8 → bfloat16 | `gemm` | M₇₂ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R054](#r054)<a name="r054"></a> | gemm<br>int8 × int8 → bfloat16 | `gemm` | M₇₂ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R055](#r055)<a name="r055"></a> | gemm<br>float8 × float8 → bfloat16 | `gemm` | M₇₂ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R056](#r056)<a name="r056"></a> | gemm<br>float8 × float8 → bfloat16 | `gemm` | M₇₂ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R057](#r057)<a name="r057"></a> | gemm<br>float8 × float8 → bfloat16 | `gemm` | M₇₂ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R058](#r058)<a name="r058"></a> | gemm<br>float8 × float8 → bfloat16 | `gemm` | M₇₂ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R059](#r059)<a name="r059"></a> | gemm<br>float4 × float4 → bfloat16 | `gemm` | M₇₂ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R060](#r060)<a name="r060"></a> | gemm<br>float4 × float4 → bfloat16 | `gemm` | M₇₂ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R061](#r061)<a name="r061"></a> | gemm<br>float4 × float4 → bfloat16 | `gemm` | M₇₂ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R062](#r062)<a name="r062"></a> | gemm<br>float4 × float4 → bfloat16 | `gemm` | M₇₂ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |

<a name="model-byte-quant"></a>

### quant_matmul（GPU_Micro_015）

需求来源：`GPU单体测试用例 v2.1.xlsx`，`测试用例!B86:E86`。

`M₈₆` = `1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 4096, 6192, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072`。每行覆盖此列表全部 M；矩阵维度统一写为 **M × N × K**。

`sp_size=1`，M=`num_tokens`、K=`hidden_size`、N=`new_hidden_size`。表内精度顺序为 **输入 / 权重 / 计算 / 输出**。

FP4、MXFP4 均需覆盖；输出 latency、算力、带宽、MFU、MBU。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R063](#r063)<a name="r063"></a> | quant_matmul<br>bfloat16 / bfloat16 / bfloat16 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R064](#r064)<a name="r064"></a> | quant_matmul<br>bfloat16 / bfloat16 / bfloat16 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R065](#r065)<a name="r065"></a> | quant_matmul<br>bfloat16 / bfloat16 / bfloat16 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R066](#r066)<a name="r066"></a> | quant_matmul<br>bfloat16 / bfloat16 / bfloat16 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R067](#r067)<a name="r067"></a> | quant_matmul<br>int8 / int8 / int8 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R068](#r068)<a name="r068"></a> | quant_matmul<br>int8 / int8 / int8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R069](#r069)<a name="r069"></a> | quant_matmul<br>int8 / int8 / int8 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R070](#r070)<a name="r070"></a> | quant_matmul<br>int8 / int8 / int8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R071](#r071)<a name="r071"></a> | quant_matmul<br>float8 / float8 / float8 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R072](#r072)<a name="r072"></a> | quant_matmul<br>float8 / float8 / float8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R073](#r073)<a name="r073"></a> | quant_matmul<br>float8 / float8 / float8 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R074](#r074)<a name="r074"></a> | quant_matmul<br>float8 / float8 / float8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | [gemm_a8w8][fn-a8] |
| [R075](#r075)<a name="r075"></a> | quant_matmul<br>float4 / float4 / float4 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R076](#r076)<a name="r076"></a> | quant_matmul<br>float4 / float4 / float4 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R077](#r077)<a name="r077"></a> | quant_matmul<br>float4 / float4 / float4 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R078](#r078)<a name="r078"></a> | quant_matmul<br>float4 / float4 / float4 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | FP32 / token（默认） | FP32 / 输出通道（默认） | — |
| [R079](#r079)<a name="r079"></a> | quant_matmul<br>mxfloat8 / mxfloat8 / mxfloat8 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | — |
| [R080](#r080)<a name="r080"></a> | quant_matmul<br>mxfloat8 / mxfloat8 / mxfloat8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | — |
| [R081](#r081)<a name="r081"></a> | quant_matmul<br>mxfloat8 / mxfloat8 / mxfloat8 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | — |
| [R082](#r082)<a name="r082"></a> | quant_matmul<br>mxfloat8 / mxfloat8 / mxfloat8 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | — |
| [R083](#r083)<a name="r083"></a> | quant_matmul<br>mxfloat4 / mxfloat4 / mxfloat4 / bfloat16 | `quant_matmul` | M₈₆ × 4096 × 4096 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | [gemm_w4a4][fn-w4a4] |
| [R084](#r084)<a name="r084"></a> | quant_matmul<br>mxfloat4 / mxfloat4 / mxfloat4 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 8192 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | [gemm_w4a4][fn-w4a4] |
| [R085](#r085)<a name="r085"></a> | quant_matmul<br>mxfloat4 / mxfloat4 / mxfloat4 / bfloat16 | `quant_matmul` | M₈₆ × 1024 × 8192 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | [gemm_w4a4][fn-w4a4] |
| [R086](#r086)<a name="r086"></a> | quant_matmul<br>mxfloat4 / mxfloat4 / mxfloat4 / bfloat16 | `quant_matmul` | M₈₆ × 8192 × 1024 | — | — | 1 | E8M0 / 32 元素（标准） | E8M0 / 32 元素（标准） | [gemm_w4a4][fn-w4a4] |

<a name="follow-up-p2"></a>

## 后续支持（P2）

<a name="model-ds-flash"></a>

### DeepSeek-V4-Flash

64k/128k × s150/s2000；各文件部署方案分别保留。

需求来源：`shape_01_flash_64k_s150.xlsx`、`shape_01_flash_64k_s2000.xlsx`、`shape_02_flash_128k_s150.xlsx`、`shape_02_flash_128k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R087](#r087)<a name="r087"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 8 × 129280 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R088](#r088)<a name="r088"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 64 × 16160 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R089](#r089)<a name="r089"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 160 × 129280 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R090](#r090)<a name="r090"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 192 × 64 × 4096 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R091](#r091)<a name="r091"></a> | linear<br>BF16×BF16→BF16 | `dspark_ctx_wkv_0`<br>`dspark_ctx_wkv_1`<br>`dspark_ctx_wkv_2` | 192 × 512 × 4096 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16] |
| [R092](#r092)<a name="r092"></a> | linear<br>BF16×BF16→BF16 | `dspark_main_proj` | 192 × 4096 × 12288 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R093](#r093)<a name="r093"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 192 × 129280 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R094](#r094)<a name="r094"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 240 × 129280 × 4096 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16] |
| [R095](#r095)<a name="r095"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 288 × 64 × 4096 | — | decode | 6 | — | — | [gemm_a16w16][fn-a16] |
| [R096](#r096)<a name="r096"></a> | linear<br>BF16×BF16→BF16 | `dspark_ctx_wkv_0`<br>`dspark_ctx_wkv_1`<br>`dspark_ctx_wkv_2` | 288 × 512 × 4096 | — | decode | 9 | — | — | [gemm_a16w16][fn-a16] |
| [R097](#r097)<a name="r097"></a> | linear<br>BF16×BF16→BF16 | `dspark_main_proj` | 288 × 4096 × 12288 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16] |
| [R098](#r098)<a name="r098"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 288 × 129280 × 4096 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16] |
| [R099](#r099)<a name="r099"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 3584 × 64 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R100](#r100)<a name="r100"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 5120 × 64 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R101](#r101)<a name="r101"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 6656 × 64 × 4096 | — | prefill | 16 | — | — | [gemm_a16w16][fn-a16] |
| [R102](#r102)<a name="r102"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 160 × 512 × 4096 | — | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R103](#r103)<a name="r103"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 160 × 1024 × 4096 | — | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R104](#r104)<a name="r104"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 160 × 4096 × 8192 | — | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R105](#r105)<a name="r105"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 160 × 32768 × 1024 | — | decode | 3 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R106](#r106)<a name="r106"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 192 × 512 × 4096 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R107](#r107)<a name="r107"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 192 × 1024 × 4096 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R108](#r108)<a name="r108"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 192 × 4096 × 8192 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R109](#r109)<a name="r109"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 192 × 8192 × 1024 | — | decode | 2 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R110](#r110)<a name="r110"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 192 × 32768 × 1024 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R111](#r111)<a name="r111"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 240 × 512 × 4096 | — | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R112](#r112)<a name="r112"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 240 × 1024 × 4096 | — | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R113](#r113)<a name="r113"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 240 × 4096 × 8192 | — | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R114](#r114)<a name="r114"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 240 × 32768 × 1024 | — | decode | 9 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R115](#r115)<a name="r115"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 288 × 512 × 4096 | — | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R116](#r116)<a name="r116"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 288 × 1024 × 4096 | — | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R117](#r117)<a name="r117"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 288 × 4096 × 8192 | — | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R118](#r118)<a name="r118"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 288 × 8192 × 1024 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R119](#r119)<a name="r119"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 288 × 32768 × 1024 | — | decode | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R120](#r120)<a name="r120"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 3584 × 512 × 4096 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R121](#r121)<a name="r121"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 3584 × 1024 × 4096 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R122](#r122)<a name="r122"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 3584 × 4096 × 8192 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R123](#r123)<a name="r123"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 3584 × 8192 × 1024 | — | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R124](#r124)<a name="r124"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 3584 × 32768 × 1024 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R125](#r125)<a name="r125"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 5120 × 512 × 4096 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R126](#r126)<a name="r126"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 5120 × 1024 × 4096 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R127](#r127)<a name="r127"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 5120 × 4096 × 8192 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R128](#r128)<a name="r128"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 5120 × 8192 × 1024 | — | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R129](#r129)<a name="r129"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 5120 × 32768 × 1024 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R130](#r130)<a name="r130"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 6656 × 512 × 4096 | — | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R131](#r131)<a name="r131"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 6656 × 1024 × 4096 | — | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R132](#r132)<a name="r132"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 6656 × 4096 × 8192 | — | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R133](#r133)<a name="r133"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 6656 × 8192 × 1024 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R134](#r134)<a name="r134"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 6656 × 32768 × 1024 | — | prefill | 32 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R135](#r135)<a name="r135"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 160 × 1024 × 4096 | 8 | decode | 3 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R136](#r136)<a name="r136"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 192 × 1024 × 4096 | 8 | decode | 4 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R137](#r137)<a name="r137"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 240 × 1024 × 4096 | 8 | decode | 9 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R138](#r138)<a name="r138"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 288 × 1024 × 4096 | 8 | decode | 12 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R139](#r139)<a name="r139"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 3584 × 1024 × 4096 | 8 | prefill | 16 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R140](#r140)<a name="r140"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 5120 × 1024 × 4096 | 8 | prefill | 16 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R141](#r141)<a name="r141"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 6656 × 1024 × 4096 | 8 | prefill | 32 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R142](#r142)<a name="r142"></a> | linear<br>FP32×FP32→FP32 | `dspark_markov_w2_0`<br>`dspark_markov_w2_1`<br>`dspark_markov_w2_2`<br>`dspark_markov_w2_3`<br>`dspark_markov_w2_4` | 32 × 129280 × 256 | — | decode | 5 | — | — | [gemm_a32w32][fn-a32] |
| [R143](#r143)<a name="r143"></a> | linear<br>FP32×FP32→FP32 | `dspark_markov_w2_0`<br>`dspark_markov_w2_1`<br>`dspark_markov_w2_2`<br>`dspark_markov_w2_3`<br>`dspark_markov_w2_4` | 48 × 129280 × 256 | — | decode | 15 | — | — | [gemm_a32w32][fn-a32] |
| [R144](#r144)<a name="r144"></a> | linear<br>FP32×FP32→FP32 | `dspark_confidence` | 160 × 1 × 4352 | — | decode | 1 | — | — | [gemm_a32w32][fn-a32] |
| [R145](#r145)<a name="r145"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 160 × 4 × 16384 | — | decode | 1 | — | — | [gemm_a32w32][fn-a32] |
| [R146](#r146)<a name="r146"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 192 × 4 × 16384 | — | decode | 1 | — | — | [gemm_a32w32][fn-a32] |
| [R147](#r147)<a name="r147"></a> | linear<br>FP32×FP32→FP32 | `dspark_confidence` | 240 × 1 × 4352 | — | decode | 3 | — | — | [gemm_a32w32][fn-a32] |
| [R148](#r148)<a name="r148"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 240 × 4 × 16384 | — | decode | 3 | — | — | [gemm_a32w32][fn-a32] |
| [R149](#r149)<a name="r149"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 288 × 4 × 16384 | — | decode | 3 | — | — | [gemm_a32w32][fn-a32] |
| [R150](#r150)<a name="r150"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 3584 × 4 × 16384 | — | prefill | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R151](#r151)<a name="r151"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 5120 × 4 × 16384 | — | prefill | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R152](#r152)<a name="r152"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 6656 × 4 × 16384 | — | prefill | 4 | — | — | [gemm_a32w32][fn-a32] |
| [R153](#r153)<a name="r153"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 28672 × 4 × 16384 | — | prefill | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R154](#r154)<a name="r154"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 40960 × 4 × 16384 | — | prefill | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R155](#r155)<a name="r155"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 53248 × 4 × 16384 | — | prefill | 4 | — | — | [gemm_a32w32][fn-a32] |
| [R156](#r156)<a name="r156"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 192 × 512 × 4096 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R157](#r157)<a name="r157"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 192 × 1024 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R158](#r158)<a name="r158"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 192 × 2048 × 4096 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R159](#r159)<a name="r159"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 288 × 512 × 4096 | — | decode | 6 | — | — | [gemm_a16w16][fn-a16] |
| [R160](#r160)<a name="r160"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 288 × 1024 × 4096 | — | decode | 3 | — | — | [gemm_a16w16][fn-a16] |
| [R161](#r161)<a name="r161"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 288 × 2048 × 4096 | — | decode | 6 | — | — | [gemm_a16w16][fn-a16] |
| [R162](#r162)<a name="r162"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 3584 × 512 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R163](#r163)<a name="r163"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 3584 × 1024 × 4096 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R164](#r164)<a name="r164"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 3584 × 2048 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R165](#r165)<a name="r165"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 5120 × 512 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R166](#r166)<a name="r166"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 5120 × 1024 × 4096 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R167](#r167)<a name="r167"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 5120 × 2048 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R168](#r168)<a name="r168"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 6656 × 512 × 4096 | — | prefill | 16 | — | — | [gemm_a16w16][fn-a16] |
| [R169](#r169)<a name="r169"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 6656 × 1024 × 4096 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R170](#r170)<a name="r170"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 6656 × 2048 × 4096 | — | prefill | 16 | — | — | [gemm_a16w16][fn-a16] |

<a name="model-ds-pro"></a>

### DeepSeek-V4-Pro

128k/1k、s2000；P 侧 a/b 两种方案，分别为 16 卡 PP2 与 8 卡方案。

需求来源：`shape_03_pro_128k_s2000_a.xlsx`、`shape_03_pro_128k_s2000_b.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R171](#r171)<a name="r171"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 2 × 64640 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R172](#r172)<a name="r172"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 2 × 129280 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R173](#r173)<a name="r173"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 8 × 16160 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R174](#r174)<a name="r174"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 16 × 16160 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R175](#r175)<a name="r175"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 80 × 129280 × 7168 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R176](#r176)<a name="r176"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 96 × 64 × 7168 | — | decode | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R177](#r177)<a name="r177"></a> | linear<br>BF16×BF16→BF16 | `dspark_ctx_wkv_0`<br>`dspark_ctx_wkv_1`<br>`dspark_ctx_wkv_2` | 96 × 512 × 7168 | — | decode | 6 | — | — | [gemm_a16w16][fn-a16] |
| [R178](#r178)<a name="r178"></a> | linear<br>BF16×BF16→BF16 | `dspark_main_proj` | 96 × 7168 × 21504 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R179](#r179)<a name="r179"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 96 × 129280 × 7168 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R180](#r180)<a name="r180"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 2560 × 64 × 7168 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R181](#r181)<a name="r181"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 4608 × 64 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R182](#r182)<a name="r182"></a> | linear<br>BF16×BF16→BF16 | `indexer_weights_proj` | 6656 × 64 × 7168 | — | prefill | 10 | — | — | [gemm_a16w16][fn-a16] |
| [R183](#r183)<a name="r183"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 80 × 512 × 7168 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R184](#r184)<a name="r184"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 80 × 1536 × 7168 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R185](#r185)<a name="r185"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 80 × 7168 × 16384 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R186](#r186)<a name="r186"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 80 × 65536 × 1536 | — | decode | 6 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R187](#r187)<a name="r187"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 96 × 512 × 7168 | — | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R188](#r188)<a name="r188"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 96 × 1536 × 7168 | — | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R189](#r189)<a name="r189"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 96 × 7168 × 16384 | — | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R190](#r190)<a name="r190"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 96 × 8192 × 1536 | — | decode | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R191](#r191)<a name="r191"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 96 × 65536 × 1536 | — | decode | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R192](#r192)<a name="r192"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 2560 × 512 × 7168 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R193](#r193)<a name="r193"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 2560 × 1536 × 7168 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R194](#r194)<a name="r194"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 2560 × 7168 × 8192 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R195](#r195)<a name="r195"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 2560 × 7168 × 16384 | — | prefill | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R196](#r196)<a name="r196"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 2560 × 8192 × 1536 | — | prefill | 8 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R197](#r197)<a name="r197"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 2560 × 32768 × 1536 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R198](#r198)<a name="r198"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 2560 × 65536 × 1536 | — | prefill | 12 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R199](#r199)<a name="r199"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 4608 × 512 × 7168 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R200](#r200)<a name="r200"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 4608 × 1536 × 7168 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R201](#r201)<a name="r201"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 4608 × 7168 × 16384 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R202](#r202)<a name="r202"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 4608 × 8192 × 1536 | — | prefill | 2 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R203](#r203)<a name="r203"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 4608 × 65536 × 1536 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R204](#r204)<a name="r204"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wkv` | 6656 × 512 × 7168 | — | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R205](#r205)<a name="r205"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_a` | 6656 × 1536 × 7168 | — | prefill | 20 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R206](#r206)<a name="r206"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 6656 × 7168 × 8192 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R207](#r207)<a name="r207"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wo_b` | 6656 × 7168 × 16384 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R208](#r208)<a name="r208"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 6656 × 8192 × 1536 | — | prefill | 10 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R209](#r209)<a name="r209"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 6656 × 32768 × 1536 | — | prefill | 4 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R210](#r210)<a name="r210"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `wq_b` | 6656 × 65536 × 1536 | — | prefill | 16 | FP32@1×128（动态） | E8M0@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R211](#r211)<a name="r211"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 80 × 1024 × 4096 | 16 | decode | 6 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R212](#r212)<a name="r212"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 96 × 1024 × 4096 | 16 | decode | 8 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R213](#r213)<a name="r213"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 2560 × 1024 × 4096 | 16 | prefill | 12 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R214](#r214)<a name="r214"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 2560 × 1024 × 4096 | 8 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R215](#r215)<a name="r215"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 4608 × 1024 × 4096 | 16 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R216](#r216)<a name="r216"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 6656 × 1024 × 4096 | 8 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R217](#r217)<a name="r217"></a> | bmm<br>BF16×BF16→BF16 | `wo_a` | 6656 × 1024 × 4096 | 16 | prefill | 16 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R218](#r218)<a name="r218"></a> | linear<br>FP32×FP32→FP32 | `dspark_markov_w2_0`<br>`dspark_markov_w2_1`<br>`dspark_markov_w2_2`<br>`dspark_markov_w2_3`<br>`dspark_markov_w2_4` | 16 × 129280 × 512 | — | decode | 10 | — | — | [gemm_a32w32][fn-a32] |
| [R219](#r219)<a name="r219"></a> | linear<br>FP32×FP32→FP32 | `dspark_confidence` | 80 × 1 × 7680 | — | decode | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R220](#r220)<a name="r220"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 80 × 4 × 28672 | — | decode | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R221](#r221)<a name="r221"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 96 × 4 × 28672 | — | decode | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R222](#r222)<a name="r222"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 2560 × 4 × 28672 | — | prefill | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R223](#r223)<a name="r223"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 6656 × 4 × 28672 | — | prefill | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R224](#r224)<a name="r224"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 20480 × 4 × 28672 | — | prefill | 1 | — | — | [gemm_a32w32][fn-a32] |
| [R225](#r225)<a name="r225"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 36864 × 4 × 28672 | — | prefill | 1 | — | — | [gemm_a32w32][fn-a32] |
| [R226](#r226)<a name="r226"></a> | linear<br>FP32×FP32→FP32 | `hc_head_linear` | 53248 × 4 × 28672 | — | prefill | 2 | — | — | [gemm_a32w32][fn-a32] |
| [R227](#r227)<a name="r227"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 96 × 512 × 7168 | — | decode | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R228](#r228)<a name="r228"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 96 × 1024 × 7168 | — | decode | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R229](#r229)<a name="r229"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 96 × 2048 × 7168 | — | decode | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R230](#r230)<a name="r230"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 2560 × 512 × 7168 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R231](#r231)<a name="r231"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 2560 × 1024 × 7168 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R232](#r232)<a name="r232"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 2560 × 2048 × 7168 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R233](#r233)<a name="r233"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 4608 × 512 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R234](#r234)<a name="r234"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 4608 × 1024 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R235](#r235)<a name="r235"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 4608 × 2048 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R236](#r236)<a name="r236"></a> | linear<br>BF16×BF16→FP32 | `indexer_compressor_wkv_gate` | 6656 × 512 × 7168 | — | prefill | 10 | — | — | [gemm_a16w16][fn-a16] |
| [R237](#r237)<a name="r237"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 6656 × 1024 × 7168 | — | prefill | 10 | — | — | [gemm_a16w16][fn-a16] |
| [R238](#r238)<a name="r238"></a> | linear<br>BF16×BF16→FP32 | `compressor_wkv_gate` | 6656 × 2048 × 7168 | — | prefill | 10 | — | — | [gemm_a16w16][fn-a16] |

<a name="model-glm51-fp8"></a>

### GLM-5.1 FP8

128k/64k、s2000；原表头 5.2/5.1 的命名歧义保留，以第 3 行描述 5.1 为准。

需求来源：`shape_05_glm_128k_s2000.xlsx`、`shape_06_glm_64k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R239](#r239)<a name="r239"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 1 × 154880 × 6144 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R240](#r240)<a name="r240"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 4 × 154880 × 6144 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R241](#r241)<a name="r241"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 6 × 32 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R242](#r242)<a name="r242"></a> | linear<br>BF16×BF16→BF16 | `nextn_eh_proj` | 6 × 6144 × 12288 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R243](#r243)<a name="r243"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 6 × 154880 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R244](#r244)<a name="r244"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 8 × 19360 × 6144 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R245](#r245)<a name="r245"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 12 × 32 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R246](#r246)<a name="r246"></a> | linear<br>BF16×BF16→BF16 | `nextn_eh_proj` | 12 × 6144 × 12288 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R247](#r247)<a name="r247"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 12 × 154880 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R248](#r248)<a name="r248"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 24 × 32 × 6144 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R249](#r249)<a name="r249"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 24 × 154880 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R250](#r250)<a name="r250"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 32 × 19360 × 6144 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R251](#r251)<a name="r251"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 48 × 32 × 6144 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R252](#r252)<a name="r252"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 48 × 154880 × 6144 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R253](#r253)<a name="r253"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 2560 × 32 × 6144 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R254](#r254)<a name="r254"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 4608 × 32 × 6144 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R255](#r255)<a name="r255"></a> | linear<br>BF16×BF16→BF16 | `indexers_proj` | 6656 × 32 × 6144 | — | prefill | 8 | — | — | [gemm_a16w16][fn-a16] |
| [R256](#r256)<a name="r256"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 6 × 128 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R257](#r257)<a name="r257"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 6 × 2624 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R258](#r258)<a name="r258"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 6 × 4096 × 2048 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R259](#r259)<a name="r259"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 6 × 6144 × 16384 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R260](#r260)<a name="r260"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 6 × 16384 × 2048 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R261](#r261)<a name="r261"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 12 × 128 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R262](#r262)<a name="r262"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 12 × 2624 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R263](#r263)<a name="r263"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 12 × 4096 × 2048 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R264](#r264)<a name="r264"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 12 × 6144 × 16384 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R265](#r265)<a name="r265"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 12 × 16384 × 2048 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R266](#r266)<a name="r266"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 24 × 128 × 6144 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R267](#r267)<a name="r267"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 24 × 2624 × 6144 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R268](#r268)<a name="r268"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 24 × 4096 × 2048 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R269](#r269)<a name="r269"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 24 × 6144 × 12288 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R270](#r270)<a name="r270"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 24 × 6144 × 16384 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R271](#r271)<a name="r271"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 24 × 16384 × 2048 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R272](#r272)<a name="r272"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 24 × 24576 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R273](#r273)<a name="r273"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 48 × 128 × 6144 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R274](#r274)<a name="r274"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 48 × 2624 × 6144 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R275](#r275)<a name="r275"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 48 × 4096 × 2048 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R276](#r276)<a name="r276"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 48 × 6144 × 12288 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R277](#r277)<a name="r277"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 48 × 6144 × 16384 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R278](#r278)<a name="r278"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 48 × 16384 × 2048 | — | decode | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R279](#r279)<a name="r279"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 48 × 24576 × 6144 | — | decode | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R280](#r280)<a name="r280"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 2560 × 128 × 6144 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R281](#r281)<a name="r281"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 2560 × 2624 × 6144 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R282](#r282)<a name="r282"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 2560 × 3072 × 6144 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R283](#r283)<a name="r283"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 2560 × 4096 × 2048 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R284](#r284)<a name="r284"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 2560 × 6144 × 1536 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R285](#r285)<a name="r285"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 2560 × 6144 × 16384 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R286](#r286)<a name="r286"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 2560 × 16384 × 2048 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R287](#r287)<a name="r287"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 4608 × 128 × 6144 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R288](#r288)<a name="r288"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 4608 × 2624 × 6144 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R289](#r289)<a name="r289"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 4608 × 3072 × 6144 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R290](#r290)<a name="r290"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 4608 × 4096 × 2048 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R291](#r291)<a name="r291"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 4608 × 6144 × 1536 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R292](#r292)<a name="r292"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 4608 × 6144 × 16384 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R293](#r293)<a name="r293"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 4608 × 16384 × 2048 | — | prefill | 4 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R294](#r294)<a name="r294"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wk` | 6656 × 128 × 6144 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R295](#r295)<a name="r295"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `fused_qkv_a_proj` | 6656 × 2624 × 6144 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R296](#r296)<a name="r296"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 6656 × 3072 × 6144 | — | prefill | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R297](#r297)<a name="r297"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `indexer_wq_b` | 6656 × 4096 × 2048 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R298](#r298)<a name="r298"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 6656 × 6144 × 1536 | — | prefill | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R299](#r299)<a name="r299"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 6656 × 6144 × 16384 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R300](#r300)<a name="r300"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `q_b_proj` | 6656 × 16384 × 2048 | — | prefill | 8 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R301](#r301)<a name="r301"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 20480 × 3072 × 6144 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R302](#r302)<a name="r302"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 20480 × 6144 × 1536 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R303](#r303)<a name="r303"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 36864 × 3072 × 6144 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R304](#r304)<a name="r304"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 36864 × 6144 × 1536 | — | prefill | 1 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R305](#r305)<a name="r305"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 53248 × 3072 × 6144 | — | prefill | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R306](#r306)<a name="r306"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 53248 × 6144 × 1536 | — | prefill | 2 | FP32@1×128（动态） | <code>FP32*@128×128（Excel）</code> | [gemm_a8w8_blockscale][fn-blockscale] |
| [R307](#r307)<a name="r307"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 6 × 256 × 512 | 64 | decode | 1 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R308](#r308)<a name="r308"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 6 × 512 × 192 | 64 | decode | 1 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R309](#r309)<a name="r309"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 12 × 256 × 512 | 64 | decode | 1 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R310](#r310)<a name="r310"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 12 × 512 × 192 | 64 | decode | 1 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R311](#r311)<a name="r311"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 24 × 256 × 512 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R312](#r312)<a name="r312"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 24 × 512 × 192 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R313](#r313)<a name="r313"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 48 × 256 × 512 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R314](#r314)<a name="r314"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 48 × 512 × 192 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R315](#r315)<a name="r315"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 2560 × 256 × 512 | 64 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R316](#r316)<a name="r316"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 2560 × 512 × 192 | 64 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R317](#r317)<a name="r317"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 4608 × 256 × 512 | 64 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R318](#r318)<a name="r318"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 4608 × 512 × 192 | 64 | prefill | 4 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R319](#r319)<a name="r319"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 6656 × 256 × 512 | 64 | prefill | 8 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R320](#r320)<a name="r320"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 6656 × 512 × 192 | 64 | prefill | 8 | — | — | [batched_gemm_bf16][fn-bmm] |

<a name="model-kimi26-w4a16"></a>

### Kimi-2.6 W4A16

128k/1k、s2000；包含 EAGLE3 draft。

需求来源：`shape_09_kimi26_128k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R321](#r321)<a name="r321"></a> | linear<br>BF16×BF16→BF16 | `eagle3_fc` | 4 × 7168 × 21504 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R322](#r322)<a name="r322"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode`<br>`lm_head_prefill` | 4 × 163840 × 7168 | — | prefill/decode | 3 | — | — | [gemm_a16w16][fn-a16] |
| [R323](#r323)<a name="r323"></a> | linear<br>BF16×BF16→BF16 | `fused_qkv_a_proj` | 16 × 2112 × 7168 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R324](#r324)<a name="r324"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 16 × 7168 × 8192 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R325](#r325)<a name="r325"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 16 × 7168 × 18432 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R326](#r326)<a name="r326"></a> | linear<br>BF16×BF16→BF16 | `q_b_proj` | 16 × 12288 × 1536 | — | decode | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R327](#r327)<a name="r327"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 16 × 36864 × 7168 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R328](#r328)<a name="r328"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 16 × 163840 × 7168 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R329](#r329)<a name="r329"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 32 × 20480 × 7168 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R330](#r330)<a name="r330"></a> | linear<br>BF16×BF16→BF16 | `fused_qkv_a_proj` | 5120 × 2112 × 7168 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R331](#r331)<a name="r331"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 5120 × 4608 × 7168 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R332](#r332)<a name="r332"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 5120 × 7168 × 2304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R333](#r333)<a name="r333"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 5120 × 7168 × 8192 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R334](#r334)<a name="r334"></a> | linear<br>BF16×BF16→BF16 | `q_b_proj` | 5120 × 12288 × 1536 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R335](#r335)<a name="r335"></a> | linear<br>BF16×BF16→BF16 | `kv_b_proj` | 5120 × 16384 × 512 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R336](#r336)<a name="r336"></a> | linear<br>BF16×BF16→BF16 | `fused_qkv_a_proj` | 6656 × 2112 × 7168 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R337](#r337)<a name="r337"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 6656 × 4608 × 7168 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R338](#r338)<a name="r338"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 6656 × 7168 × 2304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R339](#r339)<a name="r339"></a> | linear<br>BF16×BF16→BF16 | `o_proj` | 6656 × 7168 × 8192 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R340](#r340)<a name="r340"></a> | linear<br>BF16×BF16→BF16 | `q_b_proj` | 6656 × 12288 × 1536 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R341](#r341)<a name="r341"></a> | linear<br>BF16×BF16→BF16 | `kv_b_proj` | 6656 × 16384 × 512 | — | prefill | 4 | — | — | [gemm_a16w16][fn-a16] |
| [R342](#r342)<a name="r342"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 40960 × 4608 × 7168 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R343](#r343)<a name="r343"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 40960 × 7168 × 2304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R344](#r344)<a name="r344"></a> | linear<br>BF16×BF16→BF16 | `mlp_gate_up_proj` | 53248 × 4608 × 7168 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R345](#r345)<a name="r345"></a> | linear<br>BF16×BF16→BF16 | `mlp_down_proj` | 53248 × 7168 × 2304 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R346](#r346)<a name="r346"></a> | bmm<br>BF16×BF16→BF16 | `vb_bmm` | 16 × 128 × 512 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R347](#r347)<a name="r347"></a> | bmm<br>BF16×BF16→BF16 | `q_nope_bmm` | 16 × 512 × 128 | 64 | decode | 2 | — | — | [batched_gemm_bf16][fn-bmm] |
| [R348](#r348)<a name="r348"></a> | linear<br>BF16×INT4→BF16 | `eagle3_o_proj` | 4 × 7168 × 8192 | — | decode | 1 | — | <code>BF16*@1×32（Excel）</code> | [gemm_a16w4][fn-a16w4] |
| [R349](#r349)<a name="r349"></a> | linear<br>BF16×INT4→BF16 | `eagle3_down_proj` | 4 × 7168 × 12288 | — | decode | 1 | — | <code>BF16*@1×32（Excel）</code> | [gemm_a16w4][fn-a16w4] |
| [R350](#r350)<a name="r350"></a> | linear<br>BF16×INT4→BF16 | `eagle3_gate_up_proj` | 4 × 24576 × 7168 | — | decode | 1 | — | <code>BF16*@1×32（Excel）</code> | [gemm_a16w4][fn-a16w4] |
| [R351](#r351)<a name="r351"></a> | linear<br>BF16×INT4→BF16 | `eagle3_qkv_proj` | 4 × 24576 × 14336 | — | decode | 1 | — | <code>BF16*@1×32（Excel）</code> | [gemm_a16w4][fn-a16w4] |

<a name="model-hy3-fp8"></a>

### Hunyuan-V3 FP8

128k/1k、s2000；激活 FP32/token/dyn，权重 scale 为 FP32*/channel，以新版标签为准。

需求来源：`shape_10_hy3_128k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R352](#r352)<a name="r352"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 3 × 120832 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R353](#r353)<a name="r353"></a> | linear<br>BF16×BF16→BF16 | `nextn_eh_proj` | 4 × 4096 × 8192 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R354](#r354)<a name="r354"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 4 × 120832 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R355](#r355)<a name="r355"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 16 × 120832 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R356](#r356)<a name="r356"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 24 × 15104 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R357](#r357)<a name="r357"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 4 × 4096 × 8192 | — | decode | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R358](#r358)<a name="r358"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 4 × 10240 × 4096 | — | decode | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R359](#r359)<a name="r359"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 16 × 4096 × 8192 | — | decode | 2 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R360](#r360)<a name="r360"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 16 × 4096 × 13312 | — | decode | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R361](#r361)<a name="r361"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 16 × 10240 × 4096 | — | decode | 2 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R362](#r362)<a name="r362"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 16 × 26624 × 4096 | — | decode | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R363](#r363)<a name="r363"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 512 × 3328 × 4096 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R364](#r364)<a name="r364"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 512 × 4096 × 1664 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R365](#r365)<a name="r365"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 512 × 4096 × 8192 | — | prefill | 4 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R366](#r366)<a name="r366"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 512 × 10240 × 4096 | — | prefill | 4 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R367](#r367)<a name="r367"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 4096 × 3328 × 4096 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R368](#r368)<a name="r368"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 4096 × 4096 × 1664 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R369](#r369)<a name="r369"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 6656 × 3328 × 4096 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R370](#r370)<a name="r370"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 6656 × 4096 × 1664 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R371](#r371)<a name="r371"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 6656 × 4096 × 8192 | — | prefill | 4 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R372](#r372)<a name="r372"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 6656 × 10240 × 4096 | — | prefill | 4 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R373](#r373)<a name="r373"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_gate_up_proj` | 53248 × 3328 × 4096 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |
| [R374](#r374)<a name="r374"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `mlp_down_proj` | 53248 × 4096 × 1664 | — | prefill | 1 | FP32@1×K（逐 token 动态） | <code>FP32*@channel（Excel）</code> | [gemm_a8w8][fn-a8] |

<a name="model-q397-fp8"></a>

### Qwen3.5-397B-A17B FP8

128k/1k、s2000；本模型属于后续支持 P2，不在本轮三个 Qwen P0 配置内。

需求来源：`shape_11_q397b_128k_s2000.xlsx`。

| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 公开接口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [R375](#r375)<a name="r375"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 8 × 248320 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R376](#r376)<a name="r376"></a> | linear<br>BF16×BF16→BF16 | `mtp_fc` | 48 × 4096 × 8192 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R377](#r377)<a name="r377"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 48 × 124160 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R378](#r378)<a name="r378"></a> | linear<br>BF16×BF16→BF16 | `lm_head_prefill` | 64 × 31040 × 4096 | — | prefill | 2 | — | — | [gemm_a16w16][fn-a16] |
| [R379](#r379)<a name="r379"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 192 × 64 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R380](#r380)<a name="r380"></a> | linear<br>BF16×BF16→BF16 | `lm_head_decode` | 192 × 124160 × 4096 | — | decode | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R381](#r381)<a name="r381"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 3584 × 128 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R382](#r382)<a name="r382"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 6656 × 128 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R383](#r383)<a name="r383"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 28672 × 16 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R384](#r384)<a name="r384"></a> | linear<br>BF16×BF16→BF16 | `in_proj_ba` | 53248 × 16 × 4096 | — | prefill | 1 | — | — | [gemm_a16w16][fn-a16] |
| [R385](#r385)<a name="r385"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj` | 48 × 4096 × 4096 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R386](#r386)<a name="r386"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 48 × 8704 × 4096 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R387](#r387)<a name="r387"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 192 × 4096 × 4096 | — | decode | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R388](#r388)<a name="r388"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 192 × 8704 × 4096 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R389](#r389)<a name="r389"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 192 × 10240 × 4096 | — | decode | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R390](#r390)<a name="r390"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 3584 × 4096 × 8192 | — | prefill | 3 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R391](#r391)<a name="r391"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 3584 × 17408 × 4096 | — | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R392](#r392)<a name="r392"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 3584 × 20480 × 4096 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R393](#r393)<a name="r393"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `o_proj`<br>`out_proj` | 6656 × 4096 × 8192 | — | prefill | 3 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R394](#r394)<a name="r394"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `qkv_proj` | 6656 × 17408 × 4096 | — | prefill | 2 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R395](#r395)<a name="r395"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 6656 × 20480 × 4096 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R396](#r396)<a name="r396"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 28672 × 2560 × 4096 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R397](#r397)<a name="r397"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `out_proj` | 28672 × 4096 × 1024 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R398](#r398)<a name="r398"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `in_proj_qkvz` | 53248 × 2560 × 4096 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |
| [R399](#r399)<a name="r399"></a> | linear<br>FP8E4M3×FP8E4M3→BF16 | `out_proj` | 53248 × 4096 × 1024 | — | prefill | 1 | FP32@1×128（动态） | BF16@128×128（Excel） | [gemm_a8w8_blockscale][fn-blockscale] |

[fn-bmm]: ../../../boltops/gemm/batched_gemm_bf16.py
[fn-a16]: ../../../boltops/gemm/gemm_a16w16.py
[fn-a32]: ../../../boltops/gemm/gemm_a32w32.py
[fn-a8]: ../../../boltops/gemm/gemm_a8w8.py
[fn-blockscale]: ../../../boltops/gemm/gemm_a8w8_blockscale.py
[fn-a16w4]: ../../../boltops/gemm/gemm_a16w4.py
[fn-w4a4]: ../../../boltops/gemm/gemm_w4a4.py
