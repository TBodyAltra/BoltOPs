# R004 BF16 TT — current-code PMD replay

2026-11-17，MNK=6656×64×2048。BM64/BN64/BK64，每批2块、3槽位，LDS 96KiB，1P+4C。FP32累加、BF16输出，完整K，无Split-K或额外归约。

公开接口 `gemm_a16w16(a, w, out=out, backend="auto")` 已观察并核对实际加载的编译handle。A stride=[2048, 1]，w[N,K] stride=[1, 64]；物理权重布局推导统一的B_K_CONTIGUOUS。config负责实现选择，kernel的JIT断言负责形状/布局/精度约束。

完整dispatch **25.327 µs / 68.89 TOPS**，shader窗口 **21.695 µs**。3组完整输入校验后，取9次warm中位数，范围25.216–25.451 µs；首次随机dispatch 30.216 µs。每次均回读全部输出，mismatch/nonfinite为0，输出guard及全部104个TG完成检查通过。容差abs0.001+rel0.01，SGPR/VGPR spill均为0。

`record.json`包含原始采样、配置、源码commit/SHA、环境和命令；`benchmark.json`为本轮回放原始结果，`kernel.amdgcn`为实测汇编，`single_tg.perf`为原MNK/stride下首个输出tile的独立采集。PERF复用完整计时的编译cache，并强校验完整HSACO hash相同；已检查HYGPERF头。单TG时间不作为全grid性能。

环境：node60 / kxu_pmd_8569_fp8_20261017，PMD HEAD_1703、gfx946、1die48CU、128KiB LDS/CU、455ps shader周期。Triton release/3.6.x base6b111ee3 + MR565，LLVM-triton含MR21，DTK Clang3599a58d7。实际编译器二进制哈希已再次核对；本记录仅为PMD，不声明实机验收。dispatch含命令处理/分发，shader窗口单列。

复现（先加载记录的PMD环境，输出目录须不存在）：

```bash
python docs/ops/gemm/tuning/bf16_tt_mls/benchmark.py --record docs/ops/gemm/tuning/r004/1die/tt/record.json --output /tmp/r004_tt-full
python docs/ops/gemm/tuning/bf16_tt_mls/benchmark.py --record docs/ops/gemm/tuning/r004/1die/tt/record.json --output /tmp/r004_tt-single --single-tg-perf --cache-dir <full输出使用的cache目录> --expect-binary-sha256 263663de167f592bcf21740badc546c72e5118ebe93563d3a90550baa0b133d9
```
