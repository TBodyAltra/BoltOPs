# Deterministic BlockPingpong allocation ordering

The round277 64x64x128 TT vector pilot passed full-output FP16 GOLD, exact MMOP work and physical resources, but failed the CPU-to-PMD executable ISA gate. That result is excluded from the performance map.

Four independent CPU-only pass dumps in round279 locate the first executable IR difference at `TritonHCUFinalizeBlockPingpong`. `matchStaticRingBuffers` grouped A/B allocations in a pointer-keyed `DenseMap` and iterated it to build the replacement plans. Different process address layouts could reverse allocation order and subsequently change shared-memory offsets and machine instructions. These are actual executable differences, not debug-path relocation.

This isolated Triton worktree replaces that collection with `llvm::MapVector`, preserving first-use order. The source patch is `deterministic-bpp.patch`; `triton-complete-source.patch` combines the complete qualified parent209 source export and the new fix, and passed reverse-apply verification against this worktree. Parent209 source and native remain unchanged.

The private native build replaces only the HCU BlockPingpong object. All 504 parent link inputs are verified, and 503 are reused unchanged. This is an incremental build, not a full rebuild. Exact compile commands, dependency hashes, parent input hashes and output hashes are recorded in `build-plan.json`, `native/compile-dependency-hashes.json` and `build-result.json`.

`determinism-qualification.json` verifies eight TT and four TN independent compilations, alternating whether Torch is imported first. Each layout produces identical normalized executable ISA and physical resources across its independent compilations. Both existing MLS controls also retain the parent209 executable ISA, shared-memory size and physical register/spill fields. These checks are CPU-only and do not establish performance.

The remaining qualification recompiles 36 admitted vector parameter candidates. A guarded continuation checks the completed CPU source hashes and both MLS controls, removes instruction/resource aliases against the current map, and only then launches round281 PMD qualification. PMD requires full GOLD, CPU executable ISA/resource equality, off-origin four-TG odd-17 correctness checks, and single-TG 16/32/64-loop slopes. Every promotion additionally requires one independent fresh performance series.

Earlier storage-preflight and missing language-extra import failures are retained separately; neither executed a PMD model or supplied a qualifying compile. The failed round277 run is retained unchanged.
