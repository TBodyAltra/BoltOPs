"""One isolated configuration: compile, resource gate, warmup, measure, validate."""
import hashlib
import configparser
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
_slot_turn=(spec.get("compiler_options") or {}).get("hcu_wasp_split_b_slot_turn",0)
assert type(_slot_turn) is int and _slot_turn in (0,1,2)
if _slot_turn:
    o=spec["compiler_options"]
    assert o.get("hcu_wasp_split_b_slots") is True and o.get("hcu_wasp_mmac_turn")==0 and o.get("hcu_wasp_prefetch_one_operand")==1
_slots=(spec.get("compiler_options") or {}).get("hcu_wasp_split_b_slots",False)
assert type(_slots) is bool
if _slots:
    assert (spec.get("compiler_options") or {}).get("hcu_wasp_split_b_global_probe") is True

_split=(spec.get("compiler_options") or {}).get("hcu_wasp_split_b_global_probe",False)
assert type(_split) is bool
if _split:
    o=spec["compiler_options"]
    assert o.get("hcu_wasp_prefetch_one_operand")==1 and o.get("hcu_wasp_partial_k_local_read") is True and o.get("hcu_wasp_mmac_turn")==0
    assert spec["tile"] in ([256,256,128],[32,256,128],[128,256,128]) and not spec.get("load_b_first") and spec.get("wasp_single_buffer") and spec.get("dot_k_split")==2 and spec["stages"]==1 and spec["empty_arrive"] is False

_one=(spec.get("compiler_options") or {}).get("hcu_wasp_prefetch_one_operand",0)
assert type(_one) is int and _one in (0,1,2)
if _one:
    opts=spec.get("compiler_options") or {}
    assert opts.get("hcu_wasp_partial_k_local_read") is True and not opts.get("hcu_wasp_prefetch_full_k") and not opts.get("hcu_wasp_first_operand_early") and not opts.get("hcu_wasp_merge_ab") and opts.get("hcu_wasp_mmac_turn") in (0,1,2) and spec["tile"] in ([256,256,128],[128,256,128],[256,128,128],[32,256,128],[256,32,128])
_early=(spec.get("compiler_options") or {}).get("hcu_wasp_first_operand_early",False)
assert type(_early) is bool
if _early:
    assert (spec.get("compiler_options") or {}).get("hcu_wasp_prefetch_full_k") is True and not (spec.get("compiler_options") or {}).get("hcu_wasp_merge_ab",False)
_pref=(spec.get("compiler_options") or {}).get("hcu_wasp_prefetch_full_k",False)
assert type(_pref) is bool
if _pref:
    assert (spec.get("compiler_options") or {}).get("hcu_wasp_partial_k_local_read") is True and spec.get("wasp_single_buffer") and spec.get("empty_arrive") is False and spec["tile"] in ([128,256,128],[256,128,128])
_turn=(spec.get("compiler_options") or {}).get("hcu_wasp_mmac_turn",0)
assert type(_turn) is int and _turn in (0,1,2)
if _turn:
    assert (spec.get("compiler_options") or {}).get("hcu_wasp_partial_k_local_read") is True and spec.get("wasp_single_buffer") and spec.get("empty_arrive") is False
if spec.get("wasp_single_buffer"):
    assert spec.get("empty_arrive") is False, "single-buffer late release has a proven cyclic consumer wait"
result = dict(spec, status="STARTED", phase="import")
started = time.monotonic()
result["started_remote_epoch"] = time.time()
control_path = Path(__file__).resolve().parent / "control.json"
if control_path.exists():
    control = json.loads(control_path.read_text())
    result["clock_offset_reference_minus_remote_seconds"] = control["clock_offset_reference_minus_remote_seconds"]
    result["started_reference_epoch"] = time.time() + control["clock_offset_reference_minus_remote_seconds"]


def save():
    result["wall_seconds"] = time.monotonic() - started
    result["updated_remote_epoch"] = time.time()
    Path("result.tmp").write_text(json.dumps(result, indent=2, default=str))
    Path("result.tmp").replace("result.json")


try:
    import torch
    import triton
    import triton._C.libtriton as native
    native_path=Path(native.__file__).resolve()
    assert native_path==(Path(__file__).resolve().parent/"native/libtriton.so").resolve()
    native_sha=hashlib.sha256(native_path.read_bytes()).hexdigest()
    assert native_sha=="3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990"
    result["loaded_native"]={"path":str(native_path),"sha256":native_sha}
    from boltops.utility import arch_info

    mod = importlib.import_module("boltops.gemm.triton.gemm_fp16")
    bm, bn, bk = spec["tile"]
    m, n, k = spec["shape"]
    assert (m, n) == (bm, bn) and k % bk == 0
    result["kernel_contract"] = "Runtime M/N interface, full K blocks; benchmark grid1 and M=BM/N=BN; K divisible by BK; no K-tail loads"
    mode_builder = mod.mode_launch_kwargs
    if spec["mode"] == "wdra44":
        def mode_builder(mode, **kw):
            return mod.default_launch_config(wasp=True, wdra=True, load_warps=4, mma_warps=4, **kw)
    if spec["mode"] in ("wasp48", "wasp88"):
        def mode_builder(mode, **kw):
            return mod.default_launch_config(wasp=True, wdra=False,
                load_warps=4 if mode == "wasp48" else 8, mma_warps=8, **kw)
    if spec.get("wasp_single_buffer"):
        assert spec["mode"] == "wdra48" and spec["stages"] == 1
        def mode_builder(mode, **kw):
            return mod.default_launch_config(wasp=True, wdra=True, load_warps=4, mma_warps=8, hcu_wasp_single_buffer=True, **kw)
    cfg = mode_builder(spec["mode"], num_stages=spec["stages"],
                                 empty_arrive_after_mmac=spec["empty_arrive"],
                                 enable_v_mmac_cluster=spec["cluster"],
                                 mmac_layout_force=spec["mmac_layout"])
    cfg.update(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
               GROUP_SIZE_M=1, optimize_epilogue=True, HAS_BIAS=False)
    if spec.get("regs"):
        cfg["wasp_partition_regs"] = tuple(spec["regs"])
    if spec["mode"] == "nowasp":
        cfg["num_warps"] = spec.get("warps", 4)
    if spec["mls"]:
        cfg["USE_MATRIX_STORE"] = spec["matrix_store"]
    cfg.update(use_async_copy=bool(spec.get("use_async_copy", False)),
               use_block_pingpong=bool(spec.get("use_block_pingpong", False)))
    if spec.get("async_global_prefetch") is not None:
        cfg["block_pingpong_async_global_prefetch"] = spec["async_global_prefetch"]
    if spec.get("async_local_prefetch") is not None:
        cfg["block_pingpong_async_local_prefetch"] = spec["async_local_prefetch"]
    partitions = spec.get("partition_warps")
    if partitions is not None:
        assert spec["mode"] == "wasp44" and not cfg["wasp_wdra"]
        assert len(partitions) in (2, 3) and all(x in (1, 2, 4, 8) for x in partitions)
        assert all(x >= partitions[0] for x in partitions[1:]) and sum(partitions) <= 16
        assert len(partitions) == 2 or partitions[1] == partitions[2]
        cfg["wasp_partition_warps"] = tuple(partitions)
        cfg["num_warps"] = max(partitions[1:])
        cfg.pop("wasp_partition_regs", None)
        result["custom_partition_warps"] = list(partitions)
    compiler_options = spec.get("compiler_options") or {}
    assert set(compiler_options) <= {"amdgpu_enable_max_ilp_scheduling_strategy", "hcu_use_gfx946_sched_model_v2", "schedule_hint", "mls_global_prefetch", "hcu_fragment_mls_layout", "hcu_wasp_partial_k_local_read", "hcu_wasp_mmac_turn", "hcu_wasp_merge_ab", "hcu_wasp_prefetch_full_k", "hcu_wasp_first_operand_early", "hcu_wasp_prefetch_one_operand", "hcu_wasp_split_b_global_probe", "hcu_wasp_split_b_slots", "hcu_wasp_split_b_slot_turn"}
    if "schedule_hint" in compiler_options:
        assert compiler_options["schedule_hint"] in ("none", "local-prefetch")
    if compiler_options.get("hcu_wasp_partial_k_local_read"):
        assert spec.get("wasp_single_buffer") and spec["empty_arrive"] is False
        assert spec["tile"] in ([256,256,128],[128,256,128],[256,128,128],[32,256,128],[256,32,128]) and spec.get("dot_k_split") == 2
        assert spec.get("pipeline_unroll_factor",1) == 1 and spec["mode"] == "wdra48"
        assert spec["regs"] == [8,248,248] and not spec.get("mixed_operand")
    cfg.update(compiler_options)
    unroll = spec.get("pipeline_unroll_factor", 1)
    assert unroll in (1, 2, 4) and (k // bk) % unroll == 0
    cfg["LOOP_UNROLL_FACTOR"] = unroll
    dot_split = spec.get("dot_k_split", 1)
    assert dot_split in (1, 2, 4) and bk % dot_split == 0
    cfg["MMA_K_SPLIT"] = dot_split
    mixed_operand = int(spec.get("mixed_operand") or 0)
    assert mixed_operand in (0, 1, 2)
    if mixed_operand:
        assert spec["mls"] and dot_split == 4
        cfg["MIXED_OPERAND"] = mixed_operand
    result["transport_contract"] = dict(logical_bk=bk, mixed_operand=mixed_operand,
        matrix_operand="a" if mixed_operand == 1 else "b" if mixed_operand == 2 else "both",
        other_operand_fragments=4 if mixed_operand else 1,
        other_operand_k=bk // 4 if mixed_operand else bk,
        other_source_api="tl.load" if mixed_operand else "tl.matrix_load")

    full_vector = bool(spec.get("mixed_vector_full"))
    if full_vector:
        assert mixed_operand in (1, 2) and dot_split == 4
        cfg["MIXED_VECTOR_FULL"] = True
    result["transport_contract"].update(other_operand_full_vector=full_vector,
        other_operand_load_k=bk if full_vector else bk // 4 if mixed_operand else bk)

    if spec.get("load_b_first"):
        assert spec["mls"]
        cfg["LOAD_B_FIRST"] = True
    result["dot_k_split"] = dot_split
    result["pipeline_unroll_factor"] = unroll
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    if spec["scope"] == "single_cta":
        assert (m, n) == (bm, bn) and grid == (1,)
    result.update(config=cfg, grid=grid, dtype="fp16/fp16/fp32-acc/fp16-out",
                  warmup_launches=1, measured_launches=spec["repeats"],
                  input_seed=29, input_scale=1.0, atol=1e-3, rtol=1e-3)
    torch.manual_seed(29)
    ac = torch.randn((m, k), dtype=torch.float16)
    bc = (torch.randn((n, k), dtype=torch.float16).T if spec["layout"] == "TN"
          else torch.randn((k, n), dtype=torch.float16))
    a, b = ac.cuda(), bc.cuda()
    out = torch.empty((m, n), dtype=torch.float16, device="cuda")
    args = (a, b, out, b, m, n, k, *a.stride(), *b.stride(), *out.stride())
    pointers = {name: tensor.data_ptr() for name, tensor in [("a", a), ("b", b), ("out", out)]}
    result.update(device_virtual_pointers=pointers, device_virtual_pointer_mod_bytes={str(align): {name: address % align for name, address in pointers.items()} for align in [64, 256, 4096, 65536, 1048576, 2097152]})
    result.update(strides={"a": list(a.stride()), "b": list(b.stride()),
                           "out": list(out.stride())},
                  target=str(triton.runtime.driver.active.get_current_target()),
                  device_properties=triton.runtime.driver.active.utils.get_device_properties(0))
    fn = mod._gemm_fp16_mls_kernel if spec["mls"] else mod._gemm_fp16_load_kernel
    result["phase"] = "compile"
    save()
    t = time.monotonic()
    import resource
    compile_limits = resource.getrlimit(resource.RLIMIT_AS)
    compile_vms = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmSize:"))) * 1024
    compile_cap = compile_vms + 8 * 1024 ** 3
    if compile_limits[0] != resource.RLIM_INFINITY:
        compile_cap = min(compile_cap, compile_limits[0])
    result["compile_memory_guard"] = dict(additional_virtual_bytes=8 * 1024 ** 3, baseline_virtual_bytes=compile_vms,
        soft_limit_during_compile=compile_cap, restored_before_kernel_loading=True)
    save()
    resource.setrlimit(resource.RLIMIT_AS, (compile_cap, compile_limits[1]))
    try:
        kernel = fn.warmup(*args, grid=grid, **cfg)
    finally:
        resource.setrlimit(resource.RLIMIT_AS, compile_limits)

    metadata = kernel.metadata._asdict()
    if partitions is not None:
        assert metadata["num_warps"] == sum(partitions)
        assert list(metadata["wasp_partition_warps"]) == partitions

    # Use this PMD's verified capacity; keep the conservative deployment-domain limit explicit.
    arch_cap = arch_info.get_lds_capacity_bytes()
    pmd_config = configparser.ConfigParser()
    pmd_config.read("m5out/config.ini")
    cap = pmd_config.getint("gpu_configuration", "lds_size", fallback=131072)
    if cap != 131072:
        raise RuntimeError(f"unverified PMD LDS capacity changed to {cap}")
    result.update(compile_seconds=time.monotonic() - t, metadata=metadata,
                  lds_capacity_bytes=cap, lds_capacity_source="PMD config.ini lds_size=131072, validated by probes/capacity-128k",
                  arch_info_lds_capacity_bytes=arch_cap,
                  within_arch_info_capacity=metadata["shared"] <= arch_cap,
                  effective_partition_regs=metadata.get("wasp_partition_regs"),
                  actual_partition_regs=metadata.get("wasp_partition_regs_actual"),
                  effective_num_warps=metadata.get("num_warps"))
    for name, asm in kernel.asm.items():
        if name in ("amdgcn", "ttgir", "llir", "ttir"):
            Path("kernel." + name).write_text(asm)
    result["asm_sha256"] = hashlib.sha256(kernel.asm["amdgcn"].encode()).hexdigest()
    result["pipeline_loops"] = k // bk
    result["benchmark_kind"] = "single_TG_streaming_BK_pipeline"
    assert (m, n) == (bm, bn) and k % bk == 0 and grid == (1,)
    import re
    isa = kernel.asm["amdgcn"]
    result["assembly_features"] = dict(
        mmac_opcodes=sorted(set(re.findall(r"\b(v_mmac_\w+)", isa))),
        buffer_to_lds=bool(re.search(r"buffer_load[^\n]*(?:\blds\b|_lds)", isa)),
        set_priority_instructions=len(re.findall(r"\bs_setprio\b", isa)))
    if metadata["shared"] > cap:
        result.update(status="RESOURCE_EXCLUDED", reason="LDS exceeds kernel capacity")
    else:
        result["phase"] = "load"
        kernel._init_handles()
        result.update(n_regs=kernel.n_regs, n_spills=kernel.n_spills)
        result["phase"] = "execute"
        save()
        launches = []
        for i in range(spec["repeats"] + 1):
            t = time.monotonic()
            fn[grid](*args, **cfg)
            torch.cuda.synchronize()
            launches.append({"index": i, "warmup": i == 0,
                             "host_launch_seconds_excluding_jit": time.monotonic() - t})
        result["launches"] = launches
        result["phase"] = "validate"
        actual = out.cpu()
        ref = (ac.float() @ bc.float()).to(torch.float16)
        diff = (actual.float() - ref.float()).abs()
        bad = (~torch.isfinite(actual)) | (diff > 1e-3 + 1e-3 * ref.float().abs())
        result["accuracy"] = dict(elements=actual.numel(), mismatches=int(bad.sum()),
                                   nonfinite=int((~torch.isfinite(actual)).sum()),
                                   max_abs=float(diff.max()),
                                   max_rel=float((diff / ref.float().abs().clamp_min(1e-8)).max()))
        if bad.any():
            torch.save(dict(a=ac, b=bc, expected=ref, actual=actual, config=cfg), "precision-failure.pt")
        result["status"] = "PASS" if not bad.any() else "ACCURACY_FAIL"
except Exception as exc:
    result.update(status="RESOURCE_EXCLUDED" if type(exc).__name__ == "OutOfResources" else "ERROR",
                  error_type=type(exc).__name__, error=traceback.format_exc())
finally:
    save()
    print(json.dumps(result, default=str), flush=True)
