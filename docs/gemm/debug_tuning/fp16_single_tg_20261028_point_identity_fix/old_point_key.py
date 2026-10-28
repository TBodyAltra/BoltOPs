import json,hashlib
SPEC_KEYS = ("tile", "shape", "scope", "layout", "mode", "stages", "regs", "warps",
             "empty_arrive", "cluster", "mmac_layout", "mls", "matrix_store", "repeats", "use_async_copy", "use_block_pingpong", "async_global_prefetch", "async_local_prefetch", "compiler_options", "partition_warps", "pipeline_unroll_factor", "dot_k_split", "load_b_first", "mixed_operand", "mixed_vector_full", "wasp_single_buffer", "optimize_epilogue", "simplify_group1", "rebase_mls_tile", "correctness_only", "stream_k_fragments", "stream_fragment_prefetch")
def point_key(row, context_id):
    data = {k: row.get(k) for k in SPEC_KEYS}
    data["measurement_context_id"] = context_id
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
