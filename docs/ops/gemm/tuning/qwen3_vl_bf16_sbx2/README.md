# Qwen3-VL-8B BF16 dual-die replay

These records measure the recorded BF16 GEMM kernel directly on PMD sbx2 (2 dies, 96 CUs). Each measurement validates the complete output, input bytes and output canaries. The reported latency is the median of two warm dispatches at a 455 ps GC period. Each record preserves the physical strides, TG distribution, toolchain fingerprints, archived assembly and per-dispatch statistics fingerprints. The optional PERF column is blank for this batch.

The shared source archive contains the source snapshots used to obtain the measurements. The runner overlays only the recorded source and its dependencies in a disposable copy of BoltOPs. Use the recorded Torch, Triton and Clang build with the sbx2 PMD runtime. Replay checks both versions and the libtriton and Clang SHA-256 values before compilation. For example, from the repository root after sourcing that environment:

```bash
python3 docs/ops/gemm/tuning/qwen3_vl_bf16_sbx2/benchmark.py \
  --record docs/ops/gemm/tuning/r027/2die/tt/record.json \
  --output /tmp/qvl-r027-tt-replay
```

Use `--dry-run` to inspect the launch. Use `--compile-only` to save the freshly compiled HSACO, check register/LDS usage, and report whether its executable sections match the measurement. The driver entry for these measurements is direct JIT. Records do not enroll an automatic public-dispatch regression baseline.

The main table retains the existing TT records and their original sampling protocols. Cross-layout and cross-toolchain comparisons are descriptive.

R042 TN and R045 TN retain their measured HSACO. Recompiling the same source reproduced resource usage and produced a different instruction sequence. For these two records, `--compile-only` reports the fresh instruction mismatch without loading the archived binary. Full replay loads the archived HSACO after verifying that its target, kernel argument layout, segment sizes, and workgroup requirements match the fresh launcher. The ABI check uses MessagePack metadata from both ELF files.
