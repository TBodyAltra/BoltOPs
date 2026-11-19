# Qwen3-VL-8B BF16 dual-die replay

These records measure the recorded BF16 GEMM kernel directly on PMD sbx2 (2 dies, 96 CUs). Each measurement validates the complete output, input bytes and output canaries. The reported latency is the median of two warm dispatches at a 455 ps GC period. Each record preserves the physical strides, TG distribution, toolchain fingerprints, archived assembly and per-dispatch statistics fingerprints. The optional PERF column is blank for this batch.

The shared source archive contains the source snapshots used to obtain the measurements. The runner overlays only the recorded source and its dependencies in a disposable copy of BoltOPs. Use the recorded Triton and Clang build with the sbx2 PMD runtime. For example, from the repository root after sourcing that environment:

```bash
python3 docs/ops/gemm/tuning/qwen3_vl_bf16_sbx2/benchmark.py \
  --record docs/ops/gemm/tuning/r027/2die/tt/record.json \
  --output /tmp/qvl-r027-tt-replay
```

Use `--dry-run` to inspect the launch and `--compile-only` to verify the compiled executable sections and register/LDS usage. The driver entry for these measurements is direct JIT. Records do not enroll an automatic public-dispatch regression baseline.

The main table retains the existing TT records and their original sampling protocols. Cross-layout and cross-toolchain comparisons are descriptive.

R042 TN and R045 TN retain their measured HSACO. Recompiling the same source reproduced resource usage and produced a different instruction sequence. For these two records, replay uses the checked measured binary with the freshly built launch metadata.
