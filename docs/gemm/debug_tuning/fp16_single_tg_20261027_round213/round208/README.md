# MLS code generation consistency diagnostic

Native207 stopped at the TT pilot because CPU and runtime normalized ASM differed, despite identical configurations, semantically identical TTIR/TTGIR, zero physical spills, full GOLD PASS and exact logical MMAC work. Its original failed qualification is retained. No native207 retry was performed.

Four CPU-only full AST compilations reproduced the discrepancy. First divergent IR is immediately after ConvertTritonHCUToLLVM, before ConvertHCUDistributedToLLVM. Two independent native-path compiles differed, while two CPU-path compiles agreed. MlsOpToLLVM.cpp emitted descriptor groups by pointer-keyed DenseMap iteration.

Isolated worktree209 changes only the descriptor-group map return/local type to insertion-ordered llvm::MapVector. Coordinate lookup remains a DenseMap. The native95 MLS source is otherwise exact; the other503 link inputs remain byte identical. New native SHA: e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a.

Four fresh fixed-build compiles, alternating Torch preimport, produced identical normalized ASM. Both TT/TN old controls remain exactly unchanged in normalized ASM and physical resources. All24 layout3 candidates compiled with private/VGPR/SGPR spills zero, LDS16/32/80KiB. This is CPU qualification; PMD210 separately checks full GOLD, exact ASM/resource matching and useful work before admitting the rest and long-loop calibration.

Preserved diagnostic failures: stored TTGIR parsing failed on the existing null CTA layout printer; initial209 setup attempted to recreate an already copied extension symlink. Neither failure dispatched PMD models. The report v1 included a final compiler-result JSON tail in its last IR dump comparison; the corrected report excludes that tail and retains v1.

No best-map promotions result from this diagnostic. Existing map205 and all503 selected choices remain unchanged. 128cubed remains deferred.
