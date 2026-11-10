#!/usr/bin/env python3
"""Resumable PMD sweep; each point owns an independent simulator and stats directory."""
import argparse
import collections
import concurrent.futures
import configparser
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROLLER_SOURCE = Path(__file__).read_bytes()
CONTROLLER_SHA256 = hashlib.sha256(CONTROLLER_SOURCE).hexdigest()
SPEC_KEYS = ("tile", "shape", "scope", "layout", "mode", "stages", "regs", "warps",
             "empty_arrive", "cluster", "mmac_layout", "mls", "matrix_store", "repeats", "use_async_copy", "use_block_pingpong", "async_global_prefetch", "async_local_prefetch", "compiler_options", "partition_warps", "pipeline_unroll_factor", "dot_k_split", "load_b_first", "mixed_operand", "mixed_vector_full", "wasp_single_buffer", "optimize_epilogue")
CONTEXT_CACHE = {}
EXISTING_POINTS = {}
SCHEDULING_LOCK = threading.Lock()
RUNTIME_STOP = threading.Event()
GPU_LEASE_FD = None


class SchedulingStopped(Exception):
    pass


def write_json(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def measurement_context(manifest):
    # Controller scheduling/reporting changes do not change the measured kernel.
    # Worker, compiler, PMD, source state and all effective runtime env remain exact.
    core = {k: v for k, v in manifest.items() if k != "hashes"}
    core["hashes"] = {k: v for k, v in manifest["hashes"].items() if not k.endswith("/scan.py")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def context_for(row):
    if row.get("measurement_context_id"):
        return row["measurement_context_id"]
    manifest_id = row["manifest_id"]
    if manifest_id not in CONTEXT_CACHE:
        path = ROOT / ("manifest-" + manifest_id + ".json")
        CONTEXT_CACHE[manifest_id] = measurement_context(json.loads(path.read_text())) if path.exists() else manifest_id
    return CONTEXT_CACHE[manifest_id]


def point_key(row, context_id):
    data = {k: row.get(k) for k in SPEC_KEYS}
    data["measurement_context_id"] = context_id
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def stop_reason(phase):
    if (ROOT/"STOP_AFTER_CURRENT").exists():
        return "STOP_AFTER_CURRENT requested"
    path=ROOT/"control.json"
    if path.exists():
        control=json.loads(path.read_text())
        stage=os.environ.get("FP16_MAP_STAGE",phase)
        cutoff=min(control.get("deadline_remote_epoch", float("inf")),control.get("cutoffs_remote_epoch",{}).get(stage,float("inf")))
        if time.time()>=cutoff:
            return f"time boundary reached for {stage}"
    return None


def capture_manifest():
    def cmd(*args):return subprocess.check_output(args,text=True).strip()
    repos={n:dict(commit=cmd("git","-C",str(ROOT/n),"rev-parse","HEAD"),status=cmd("git","-C",str(ROOT/n),"status","--porcelain"),tracked_diff_sha256=hashlib.sha256(subprocess.check_output(["git","-C",str(ROOT/n),"diff","HEAD","--binary"])).hexdigest()) for n in ["triton","bolt_ops"]}
    q=json.loads((ROOT/"cpu-admission.json").read_text());files=[Path(p) for p in q['source_hashes']]
    files.extend(ROOT/n for n in ["scan.py","worker.py","stats_io.py","tune.py","env.sh","control.json","native/libtriton.so","deps/pmd/libgem5.so","cpu-controls.json","cpu-admission.json","prepare-proof.json","triton-source.patch","bolt_ops-source.patch","reference-map.json","job.py","prepare.py","scratch0-performance-admission.json","triton/third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp"])
    files.append(ROOT/'strict-zero-rejection-audit.json')
    files.append(Path(os.environ['TRITON_HIP_CLANG_PATH']))
    return dict(repos=repos,host=cmd('hostname'),execution='PMD',arch='gfx946',cu_per_die=48,statistics='gcSimCycles median3 excluding warmup; scratch0/VGPRspill0 SGPR-spilling performance exploration, no physical-zero claim, single TG16/32/64, no automatic promotion',hashes={str(p):digest(p) for p in files},env={k:v for k,v in sorted(os.environ.items()) if k.startswith(("TRITON","HCU","HIP","GPU","PMD","AMD","OMP","OPENBLAS")) or k in ("PYTHONPATH","LD_LIBRARY_PATH","PATH","LLVM_SYSPATH")})


def spec(tile=(64, 64, 32), k=1024, layout="TT", mode="nowasp", stages=2,
         regs=None, warps=4, empty=False, cluster=0, mmac=-1, mls=False,
         matrix_store=False, shape=None):
    return dict(tile=list(tile), shape=list(shape or (tile[0], tile[1], k)),
                scope="full_gemm" if shape else "single_cta", layout=layout,
                mode=mode, stages=stages, regs=regs, warps=warps,
                empty_arrive=empty, cluster=cluster, mmac_layout=mmac,
                mls=mls, matrix_store=bool(matrix_store) if mls else False, repeats=3)


def spec_from_row(row, **overrides):
    values=dict(tile=row["tile"],k=row["shape"][2],layout=row["layout"],mode=row["mode"],
                stages=row["stages"],regs=row["regs"],warps=row["warps"],empty=row["empty_arrive"],
                cluster=row["cluster"],mmac=row["mmac_layout"],mls=row["mls"],matrix_store=row["matrix_store"])
    return spec(**dict(values,**overrides))


def resource_cta_limits(row):
    """VGPR/LDS/wave upper bounds only; excludes SGPR, barriers and other limits."""
    meta=row.get("metadata",{})
    props=row.get("device_properties",{})
    warps=row.get("effective_num_warps") or meta.get("num_warps",row.get("warps",4))
    waves=props.get("max_threads_per_sm",2560)//props.get("warpSize",64)
    shared=((meta.get("shared",0)+511)//512)*512
    quotas=row.get("effective_partition_regs") or meta.get("wasp_partition_regs")
    if row.get("mode","").startswith("wdra") and quotas:
        vgpr_bound=512//max(sum(quotas),1)
    else:
        # Four SIMD pools per CU. One/two-warp workgroups use only a subset;
        # ceil(warps/4) per-SIMD accounting would undercount their possible CTA count.
        regs=((row.get("n_regs",256)+3)//4)*4
        vgpr_bound=(4*512)//max(regs*warps,1)
    return dict(vgpr=vgpr_bound,lds=row.get("lds_capacity_bytes",131072)//max(shared,512),waves=waves//warps)


def resource_cta_bound(row):
    return min(resource_cta_limits(row).values())


def demand_quota_candidates(row):
    """Search pool budgets that can admit different CTA counts, using measured demand."""
    demand=row.get("actual_partition_regs") or row.get("metadata",{}).get("wasp_partition_regs_actual")
    if not demand or len(demand) not in (2,3,4):
        return []
    count=len(demand)
    props=row.get("device_properties",{})
    max_waves=props.get("max_threads_per_sm",2560)//props.get("warpSize",64)
    shared=((row.get("metadata",{}).get("shared",0)+511)//512)*512
    max_ctas=min(max_waves//(4*count),row.get("lds_capacity_bytes",131072)//max(shared,512))
    targets=sorted({(512//(4*count*n))*(4*count) for n in range(1,max_ctas+1)})
    # PMD 46597b6 aborts on WDRA44's full [256,256] descriptor; the
    # adjacent total-504 budgets passed TT/TN controls. Keep all other modes.
    if row.get("mode")=="wdra44":
        targets=[min(t,504) for t in targets]
    producers=2 if count==4 else 1
    recipients=list(range(producers,count))+list(range(producers))
    out=[]
    # Prefer measured headroom, but let tight 4/0-margin budgets cross an
    # occupancy threshold when +8/+16 would exclude a feasible configuration.
    for margin in (8,16,4,0):
        minimum=[((int(n)+margin+3)//4)*4 for n in demand]
        if max(minimum)>256 or sum(minimum)>512:
            continue
        padding=(-sum(minimum))%(4*count)
        for i in range(count):
            step=min(padding,256-minimum[i]);minimum[i]+=step;padding-=step
        if padding or sum(minimum)>512:
            continue
        for target in targets+[sum(minimum)]:
            if target<sum(minimum):
                continue
            q=list(minimum)
            while sum(q)<target:
                progressed=False
                for i in recipients:
                    if q[i]<256 and sum(q)<target:
                        q[i]+=4;progressed=True
                if not progressed:
                    break
            if sum(q)==target and q not in out:
                out.append(q)
    return out


def variants(phase):
    if phase in ("baseline", "smoke"):
        return [dict(mode="nowasp"), dict(mode="wasp44"),
                dict(mode="wdra48", regs=[16, 244, 244]),
                dict(mode="wdra48", regs=[112, 196, 196]),
                dict(mode="wdra88", regs=[64, 64, 192, 192])]
    vs = [dict(mode="nowasp", stages=s, warps=w) for s, w in
          itertools.product((1, 2, 3, 4), (1, 2, 4, 8, 16))]
    quotas48 = [None,[60,96,96],[48,60,60],[36,108,108],[64,160,160]] + [[p, (504-p)//2, (504-p)//2] for p in (16, 40, 64, 88, 112, 136, 160)]
    quotas88 = [None,[48,48,80,80],[32,32,160,160],[48,48,48,48]] + [[p, p, 256-p, 256-p] for p in (16, 32, 64, 96, 128)]
    quotas44 = [[16,240],[32,224],[64,192],[96,160],[128,128],
                [16,248],[32,256],[64,256],[128,256]]
    for mode, stages, quotas in (("wasp44", (2, 4), [None]),
                                  ("wdra44", (2, 4), quotas44),
                                  ("wdra48", (2, 4), quotas48),
                                  ("wdra88", (2,), quotas88)):
        for stage, regs, empty, cluster, mmac, mls in itertools.product(
                stages, quotas, (False, True), (0, 1, 2), (-1, 3), (False, True)):
            vs.append(dict(mode=mode, stages=stage, regs=regs, empty=empty,
                           cluster=cluster, mmac=mmac, mls=mls))
    return vs


def stats_for(directory, result):
    config_path = directory / "m5out/config.ini"
    if config_path.exists():
        config = configparser.ConfigParser(strict=False)
        config.optionxform = str
        config.read(config_path)
        gpu = dict(config["gpu_configuration"]) if "gpu_configuration" in config else {}
        result["pmd_topology"] = dict(
            num_chip=gpu.get("num_chip", "unknown"), num_cu=gpu.get("num_cu", "unknown"),
            die_count="unknown",
            die_evidence="num_chip/num_cu are PMD model fields; no physical die inference",
            config=str(config_path), config_sha256=digest(config_path),
            gpu_clock_ticks=gpu.get("gpu_clock"), sys_clock_ticks=gpu.get("sys_clock"),
            umc_clock_ticks=gpu.get("umc_clock"),
            xcd_topology="unknown; stats output directory index is not a physical die/XCD count")
    dispatches = []
    for path in directory.glob("m5out/*/*/stats.txt"):
        vals = {}
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    vals[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        if vals.get("system.gcSimCycles", 0) <= 0:
            continue
        row = dict(packet=int(path.parent.name), xcd=path.parent.parent.name,
                   cycles=vals["system.gcSimCycles"], frequency_ghz=vals.get("gcFrequency"),
                   host_seconds=vals.get("system.hostSeconds"), stats=str(path))
        instructions={kind:0 for kind in ("VALU","MMOP","LDS","VMEM","SCA")}
        wait_lgkm=vgpr_blocks=lds_blocks=0
        slots={};reuse=[]
        # Classify counters once. The literal prefilters are necessary parts of
        # the original regular expressions; accepted counter names are identical.
        for key,value in vals.items():
            if "numInstrTypeExecuted::" in key:
                match=re.fullmatch(r"system.shaders.CUs\d+.numInstrTypeExecuted::(VALU|MMOP|LDS|VMEM|SCA)",key)
                if match:
                    instructions[match[1]]+=value
            elif value and key.endswith("wvCnt"):
                match=re.fullmatch(r"system.shaders.CUs(\d+).wavefronts\d+.wvCnt",key)
                if match:
                    slots[match[1]]=slots.get(match[1],0)+1;reuse.append(value)
            if key.endswith(".waitLgkmCounter"):
                wait_lgkm+=value
            if key.endswith("numTimesWgBlockedDueVgprAlloc"):
                vgpr_blocks+=value
            if key.endswith("wgBlockedDueLdsAllocation"):
                lds_blocks+=value
        row["active_mmac_cus"] = sorted({int(re.search(r"CUs(\d+)", key).group(1)) for key, value in vals.items()
            if value and re.fullmatch(r"system.shaders.CUs\d+.numInstrTypeExecuted::MMOP", key)})
        row["mmop_runtime_counter_sum"] = sum(value for key, value in vals.items()
            if re.fullmatch(r"system.shaders.CUs\d+.SIMD\d+.mmopRunTimeCounter", key))
        row.update(instructions=instructions,wait_lgkm=wait_lgkm,
            l2_requests=vals.get("system.shaders.cache_system.totalTCCRequests"),
            vgpr_allocation_blocks=vgpr_blocks,lds_allocation_blocks=lds_blocks,
            max_distinct_wave_slots_per_cu=max(slots.values(),default=0),
            max_wave_slot_reuse=max(reuse,default=0))
        dispatches.append(row)
    dispatches.sort(key=lambda d: (d["packet"], d["xcd"]))
    result["dispatches"] = dispatches
    result["stats_parser_sha256"] = CONTROLLER_SHA256
    # This session supports one PMD stats stream. Unexpected topology is not ranked.
    streams = {d["xcd"] for d in dispatches}
    if result["status"] == "PASS":
        if len(streams) != 1 or len(dispatches) != result["repeats"] + 1:
            result.update(status="STATS_INVALID", reason="unexpected dispatch count/topology")
        else:
            measured = dispatches[1:]
            cycles = [d["cycles"] for d in measured]
            freqs = {d["frequency_ghz"] for d in measured}
            if len(freqs) != 1 or None in freqs:
                result.update(status="STATS_INVALID", reason="missing/inconsistent clock")
                return
            freq = next(iter(freqs))
            median = statistics.median(cycles)
            m, n, k = result["shape"]
            result.update(median_cycles=median, gc_frequency_ghz=freq,
                          min_cycles=min(cycles), max_cycles=max(cycles),
                          range_over_mean=(max(cycles)-min(cycles))/statistics.mean(cycles),
                          tops=2*m*n*k*freq/(median*1000),
                          cycles_per_output=median/(m*n))
            assert result["scope"] == "single_cta" and result["grid"] == [1]
            result["pipeline_loops"] = k // result["tile"][2]
            result["cycles_per_bk_including_fill_drain"] = median / result["pipeline_loops"]
            # 16x16x16 FP16 MMAC = 8192 FLOPs; this model configures 8 cycles,
            # with 4 SIMD pipelines per CU. Validate those model parameters.
            if gpu.get("mmop_f16_i8_f8") == "8" and gpu.get("simds_per_cu") == "4":
                result["peak_fp16_flops_per_cu_cycle"] = 4096
                result["mmac_ideal_fraction_including_fill_drain"] = 2*m*n*k/(4096*median)
                result["peak_basis"] = "8192 FLOPs / configured 8 MMOP cycles * configured 4 SIMDs; one CU, not whole device"
            if any(len(d["active_mmac_cus"]) != 1 for d in dispatches):
                result.update(status="STATS_INVALID", reason="single TG must execute MMAC on exactly one CU")
    log = directory / "worker.log"
    if log.exists():
        result["pmd_version_evidence"] = [line for line in log.read_text(errors="replace").splitlines()
                                         if re.search(r"pmd (compiled|commit audate|build version)", line)]


def aggregate(runroot):
    rows = [json.loads(p.read_text()) for p in sorted(runroot.glob("*/result.json"))]
    for row in rows:
        row["measurement_context_id"] = context_for(row)
    best = {}
    for row in rows:
        if row.get("status") != "PASS" or "median_cycles" not in row:
            continue
        key = json.dumps([row["measurement_context_id"], row["scope"], row["shape"], row["layout"], row["tile"]])
        if key not in best or row["median_cycles"] < best[key]["median_cycles"]:
            best[key] = row
    write_json(runroot / "all-results.json", rows)
    write_json(runroot / "best-map.json", list(best.values()))
    flat = []
    for row in best.values():
        flat.append({k: json.dumps(row[k]) if isinstance(row.get(k), (list, dict)) else row.get(k)
                     for k in ("id", "manifest_id", "measurement_context_id", "scope", "shape", "layout", "tile", "mode", "stages",
                               "warps", "regs", "empty_arrive", "cluster", "mmac_layout", "mls",
                               "effective_num_warps", "effective_partition_regs", "actual_partition_regs",
                               "lds_capacity_bytes", "within_arch_info_capacity",
                               "median_cycles", "gc_frequency_ghz", "tops", "cycles_per_output",
                               "range_over_mean", "n_regs", "n_spills", "metadata", "accuracy")})
    if flat:
        with (runroot / "best-map.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    write_json(runroot / "progress.json", dict(total=len(rows), ranked_cells=len(best),
               statuses=dict(collections.Counter(r.get("status") for r in rows))))
    return rows, best


def recoverable_full_rejection(result):
    return not result.get("accuracy", {}).get("nonfinite", 0) and (
        result.get("status") in ("ACCURACY_FAIL", "RESOURCE_EXCLUDED") or
        (result.get("status")=="ERROR" and result.get("phase")=="compile"))


def run_point(point, runroot, manifest_id, context_id):
    key = point_key(point, context_id)
    if key in EXISTING_POINTS:
        result=EXISTING_POINTS[key]
    else:
        guards=ROOT/"case-locks";guards.mkdir(exist_ok=True)
        with (guards/(key+".lock")).open("a") as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            result=_run_point_locked(point,runroot,manifest_id,context_id,guard.fileno())
    if point.get("execution_phase")=="validate" and point.get("validation_role")=="winner" and \
            point.get("scope")=="full_gemm" and point.get("coverage_fallback") and recoverable_full_rejection(result):
        fallback=dict(point["coverage_fallback"],execution_phase="validate",validation_role="coverage_fallback",
                      execution_parallelism_limit=point.get("execution_parallelism_limit",1),fallback_for=key)
        assert fallback["mode"]=="nowasp" and not fallback["mls"]
        assert all(fallback[k]==point[k] for k in ("tile","shape","layout","scope"))
        fallback_key=point_key(fallback,context_id)
        assert fallback_key!=key
        events=ROOT/"coverage-fallbacks";events.mkdir(exist_ok=True)
        event=dict(measurement_context_id=context_id,controller_sha256=CONTROLLER_SHA256,rejected_id=key,rejected_status=result.get("status"),
                   fallback_id=fallback_key,started_remote_epoch=time.time(),status="STARTED")
        write_json(events/(key+".json"),event)
        # Run synchronously in this worker after the rejected model exits and
        # its case lock is released. The existing pool size and GPU lease stay
        # unchanged; the fallback uses the same stop/disk admission checks.
        try:
            recovered=run_point(fallback,runroot,manifest_id,context_id)
        except SchedulingStopped as exc:
            event.update(status="NOT_STARTED",reason=str(exc),finished_remote_epoch=time.time())
            write_json(events/(key+".json"),event)
            raise
        except Exception as exc:
            event.update(status="EXECUTOR_ERROR",reason=repr(exc),finished_remote_epoch=time.time())
            write_json(events/(key+".json"),event)
            raise
        event.update(status=recovered.get("status"),finished_remote_epoch=time.time())
        write_json(events/(key+".json"),event)
    return result


def _run_point_locked(point, runroot, manifest_id, context_id, guard_fd):
    key=point_key(point,context_id)
    point = dict(point, manifest_id=manifest_id, measurement_context_id=context_id, id=key)
    directory = runroot / point["id"]
    directory.mkdir(exist_ok=True)
    output = directory / "result.json"
    if output.exists() and json.loads(output.read_text()).get("status") not in ("STARTED",):
        result=json.loads(output.read_text())
        if result.get("status")=="PASS" and "median_cycles" not in result:
            stats_for(directory,result);write_json(output,result)
        EXISTING_POINTS[key]=result
        return result
    if output.exists() or (directory / "m5out").exists():
        archive = runroot / "interrupted"
        archive.mkdir(exist_ok=True)
        directory.rename(archive / (point["id"] + "-" + str(time.time_ns())))
        directory.mkdir()
    write_json(directory / "spec.json", point)
    with (directory / "worker.log").open("w") as log:
        with SCHEDULING_LOCK:
            reason=stop_reason(point.get("execution_phase","point"))
            if point.get("execution_phase")=="validate" and shutil.disk_usage(ROOT).free<20*1024**3:
                RUNTIME_STOP.set();(ROOT/"STOP_AFTER_CURRENT").touch()
                raise SchedulingStopped("less than 20 GiB free; stopped before model launch")
            if RUNTIME_STOP.is_set() or reason:
                raise SchedulingStopped(reason or "runtime fault stopped scheduling")
            descriptors=(guard_fd,)+( (GPU_LEASE_FD,) if GPU_LEASE_FD is not None else () )
            proc = subprocess.Popen([sys.executable, str(ROOT / "worker.py"), str(directory / "spec.json")],
                                    cwd=directory,stdout=log,stderr=subprocess.STDOUT,pass_fds=descriptors)
            write_json(directory/"executor.json",dict(child_pid=proc.pid,parent_pid=os.getpid(),
                       parallelism_limit=point.get("execution_parallelism_limit",1),started_remote_epoch=time.time()))
        proc.wait()
    result = json.loads(output.read_text()) if output.exists() else dict(point, phase="unknown")
    if proc.returncode:
        result.update(status="PROCESS_FAILED", returncode=proc.returncode)
    stats_for(directory, result)
    write_json(output, result)
    EXISTING_POINTS[key] = result
    print(json.dumps({k: result.get(k) for k in
          ("id", "tile", "shape", "layout", "mode", "status", "median_cycles", "tops", "phase")}), flush=True)
    logtext = (directory / "worker.log").read_text(errors="replace")
    # Stop at process completion on hardware/runtime trouble; never kill PMD.
    if (result.get("phase") in ("load", "execute") and result["status"] in ("ERROR", "PROCESS_FAILED")) or \
            re.search(r"SMBus|I2C.*error|completion.*timeout|register.*timeout", logtext, re.I) or \
            result.get("accuracy", {}).get("nonfinite", 0):
        with SCHEDULING_LOCK:
            RUNTIME_STOP.set();(ROOT/"STOP_AFTER_CURRENT").touch()
        raise RuntimeError("runtime fault; stopped scheduling, inspect " + str(directory))
    return result


def model_unsupported(point):
    return point.get("mode")=="wdra44" and point.get("regs")==[256,256]


def execute_points(points,runroot,manifest_id,context_id,phase,workers):
    """Run bounded independent-model cohorts while preserving shared PMD lock fairness."""
    global GPU_LEASE_FD
    pending=[];seen=set();unsupported=[]
    for point in points:
        key=point_key(point,context_id)
        if model_unsupported(point):
            unsupported.append(dict(point,id=key,reason="PMD WDRA44 [256,256] VRF overallocation; see wdra-quota-boundary.json"))
            continue
        if key in seen or key in EXISTING_POINTS:
            continue
        seen.add(key)
        pending.append(dict(point,execution_phase=phase,execution_parallelism_limit=workers))
    write_json(ROOT/("model-exclusions-"+phase+".json"),dict(measurement_context_id=context_id,points=unsupported))
    print(f"new_points={len(pending)} parallel_workers={workers} model_excluded={len(unsupported)}",flush=True)
    setup_errors=0
    # The final validation stage has one controller after both tuning pools drain.
    # Feed its fixed-size process pool continuously so short configurations can
    # fill long-kernel tails. Tuning retains bounded cohorts for lock fairness.
    cohort_size=max(1,len(pending)) if phase=="validate" else max(1,workers*2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers,thread_name_prefix="pmd") as pool:
        for begin in range(0,len(pending),cohort_size):
            reason=stop_reason(phase)
            if reason or RUNTIME_STOP.is_set():
                print(reason or "runtime fault stopped scheduling",flush=True)
                break
            if shutil.disk_usage(ROOT).free<20*1024**3:
                raise RuntimeError("less than 20 GiB free; stopping before the next cohort")
            failure=None;boundary=False
            with (ROOT/"session.lock").open("a") as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                reason=stop_reason(phase)
                if reason:
                    print(reason+" after acquiring PMD lock",flush=True)
                    break
                assert 1 <= workers <= 8
                audit = json.loads(subprocess.check_output([sys.executable, str(ROOT.parent / "single-tg-continuation-20261023/runtime_active.py")], text=True))
                if audit["native_workers"] != 0:
                    raise RuntimeError("Own PMD models remain before cohort admission: " + json.dumps(audit))
                GPU_LEASE_FD=lease.fileno()
                batch=pending[begin:begin+cohort_size]
                futures={pool.submit(run_point,p,runroot,manifest_id,context_id):p for p in batch}
                try:
                    completed_count=0
                    for future in concurrent.futures.as_completed(futures):
                        completed_count+=1
                        try:
                            result=future.result()
                            bad_setup=result.get("status") in ("ERROR","PROCESS_FAILED") and result.get("phase") in ("import","unknown")
                            setup_errors=setup_errors+1 if bad_setup else 0
                            if setup_errors>=8:
                                raise RuntimeError("8 consecutive setup failures; inspect logs")
                        except SchedulingStopped:
                            boundary=True
                        except Exception as error:
                            failure=failure or error
                            with SCHEDULING_LOCK:
                                RUNTIME_STOP.set();(ROOT/"STOP_AFTER_CURRENT").touch()
                        write_json(ROOT/"dispatch-state.json",dict(phase=phase,workers=workers,
                            queued=len(pending)-begin-len(batch)+sum(not f.done() and not f.running() for f in futures),
                            active=sum(f.running() for f in futures),
                            completed_in_cohort=sum(f.done() for f in futures),cohort_size=len(batch),
                            updated_remote_epoch=time.time()))
                        if phase=="validate" and completed_count%max(1,workers)==0:
                            aggregate(runroot)
                finally:
                    # Every submitted future is drained naturally before releasing the shared lease.
                    concurrent.futures.wait(futures)
                    GPU_LEASE_FD=None
            aggregate(runroot)
            if failure:
                raise failure
            if boundary or stop_reason(phase):
                break
            time.sleep(.05)
    aggregate(runroot)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--phase", choices=("smoke", "baseline", "packed-vgpr", "widen", "tune", "refine", "full", "validate"), default="baseline")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--workers",type=int,choices=range(1,65),default=None)
    args = ap.parse_args()
    runroot = ROOT / "runs"
    runroot.mkdir(exist_ok=True)
    if args.summarize:
        aggregate(runroot)
        return
    if args.phase == "smoke":
        points = [spec(), spec(tile=(128, 64, 32), layout="TN", mode="wdra48", regs=[16,244,244]),
                  spec(mode="wasp44"), spec(mode="wdra88", regs=[64,64,192,192])]
    elif args.phase == "packed-vgpr":
        prior,_=aggregate(runroot)
        active_data=json.loads((ROOT/"active-manifest.json").read_text())
        active_id=active_data.get("measurement_context_id") or context_for(active_data)
        sources={}
        for row in prior:
            if row["measurement_context_id"]!=active_id or row["scope"]!="single_cta" or \
                    row.get("status")!="PASS" or "median_cycles" not in row or not row["mode"].startswith("wdra"):
                continue
            key=json.dumps([row["tile"],row["shape"],row["layout"],row["mode"]])
            if key not in sources or row["median_cycles"]<sources[key]["median_cycles"]:
                sources[key]=row
        points=[]
        for row in sources.values():
            candidates=demand_quota_candidates(row)
            if not candidates:
                continue
            # Test the largest feasible upper bound, preserving headroom preference
            # when several quotas reach the same bound. Actual PMD performance decides.
            quotas=max(candidates,key=lambda q:resource_cta_bound(dict(row,effective_partition_regs=q)))
            if resource_cta_bound(dict(row,effective_partition_regs=quotas))>resource_cta_bound(row):
                points.append(spec_from_row(row,regs=quotas))
    elif args.phase == "widen":
        tiles = list(itertools.product((32,64,128,256), (32,64,128,256), (16,32,64,128)))
        # Wider ordinary kernels address the measured spill-heavy large tiles.
        # 1/2/16-warp support was precision-checked by ordinary_warps_probe.py.
        ordinary=[(16,t) for t in tiles if t[0]*t[1]>=16384]+[(8,t) for t in tiles]
        points=[spec(tile=t,k=k,layout=layout,warps=w) for (w,t),k,layout in
                itertools.product(ordinary,(1024,4096),("TT","TN"))]
        points += [spec(tile=t,k=k,layout=layout,mode="wdra44",regs=[16,240]) for t,k,layout in
                   itertools.product(tiles,(1024,4096),("TT","TN"))]
        small=[(w,t) for w,area in ((1,4096),(2,8192)) for t in tiles if t[0]*t[1]<=area]
        points += [spec(tile=t,k=k,layout=layout,warps=w) for (w,t),k,layout in
                   itertools.product(small,(1024,4096),("TT","TN"))]
    elif args.phase in ("tune", "refine", "validate"):
        prior, best = aggregate(runroot)
        active_file = ROOT / "active-manifest.json"
        active_data = json.loads(active_file.read_text()) if active_file.exists() else {}
        active_id = active_data.get("measurement_context_id")
        if not active_id and active_data:
            active_id = context_for(active_data)
        points = []
        tune_groups=[]
        packed_alternatives={}
        packed_candidates=collections.defaultdict(list)
        higher_cta_alternatives={}
        ordinary_alternatives={}
        wasp_alternatives={}
        classic_alternatives={}
        ordinary_classic_candidates=collections.defaultdict(list)
        lower_warp_alternatives={}
        if args.phase=="validate":
            for candidate in prior:
                if candidate["measurement_context_id"]!=active_id or candidate["scope"]!="single_cta" or \
                        candidate.get("status")!="PASS" or "median_cycles" not in candidate:
                    continue
                key=(tuple(candidate["tile"]),candidate["shape"][2],candidate["layout"])
                if candidate["mode"]=="wasp44":
                    if key not in wasp_alternatives or candidate["median_cycles"]<wasp_alternatives[key]["median_cycles"]:
                        wasp_alternatives[key]=candidate
                if candidate["mode"]=="nowasp":
                    if key not in ordinary_alternatives or candidate["median_cycles"]<ordinary_alternatives[key]["median_cycles"]:
                        ordinary_alternatives[key]=candidate
                    if not candidate["mls"] and not candidate.get("n_spills",1):
                        ordinary_classic_candidates[key].append(candidate)
                    if not candidate["mls"] and (key not in classic_alternatives or
                            candidate["median_cycles"]<classic_alternatives[key]["median_cycles"]):
                        classic_alternatives[key]=candidate
                if candidate["mode"].startswith("wdra") and resource_cta_bound(candidate)>=2:
                    packed_candidates[key].append(candidate)
                    if key not in packed_alternatives or candidate["median_cycles"]<packed_alternatives[key]["median_cycles"]:
                        packed_alternatives[key]=candidate
            for key,candidates in packed_candidates.items():
                fastest=packed_alternatives[key]
                near=[c for c in candidates if not c.get("n_spills",1) and
                      resource_cta_bound(c)>resource_cta_bound(fastest) and
                      c["median_cycles"]<=1.05*fastest["median_cycles"]]
                if near:
                    higher_cta_alternatives[key]=min(near,key=lambda c:(-resource_cta_bound(c),c["median_cycles"]))
            for key,candidates in ordinary_classic_candidates.items():
                fastest=classic_alternatives[key]
                near=[c for c in candidates if c["warps"]<fastest["warps"] and
                      c["median_cycles"]<=1.05*fastest["median_cycles"] and
                      resource_cta_bound(c)>=resource_cta_bound(fastest)]
                if near:
                    lower_warp_alternatives[key]=min(near,key=lambda c:(c["warps"],c["median_cycles"]))
        prior_by_key={point_key(r,active_id):r for r in prior if r["measurement_context_id"]==active_id} if args.phase=="validate" else {}
        for row in best.values():
            if active_id and row["measurement_context_id"] != active_id:
                continue
            if row["scope"] != "single_cta":
                continue
            base = dict(tile=row["tile"], k=row["shape"][2], layout=row["layout"])
            if args.phase == "tune":
                cell_points=[]
                current = dict(base, mode=row["mode"], stages=row["stages"], regs=row["regs"],
                               warps=row["warps"], empty=row["empty_arrive"], cluster=row["cluster"],
                               mmac=row["mmac_layout"], mls=row["mls"], matrix_store=row["matrix_store"])
                options = dict(stages=(1,2,3,4) if row["mode"] == "nowasp" else
                                      (2,) if row["mode"] == "wdra88" else (2,4),
                               mls=(False,True), empty=(False,True), cluster=(0,1,2), mmac=(-1,0,1,2,3,4))
                if row["mls"]:
                    options["matrix_store"]=(False,True)
                if row["mode"] == "nowasp":
                    area=row["tile"][0]*row["tile"][1]
                    options["warps"] = (1,2,4,8) if area<=4096 else (2,4,8,16) if area<=8192 else (4,8,16)
                if row["mode"] == "wdra48":
                    options["regs"] = [None] + [[p,(504-p)//2,(504-p)//2] for p in (16,40,64,88,112,136,160)]
                if row["mode"] == "wdra88":
                    options["regs"] = [None] + [[p,p,256-p,256-p] for p in (16,32,64,96,128)]
                if row["mode"] == "wdra44":
                    options["regs"] = [[16,240],[32,224],[64,192],[96,160],[128,128],
                                       [16,248],[32,256],[64,256],[128,256]]
                if "regs" in options:
                    quotas=demand_quota_candidates(row)+options.pop("regs")
                    unique=[]
                    for q in quotas:
                        if q not in unique:
                            unique.append(q)
                    options={"regs":unique,**options}
                for key, values in options.items():
                    for value in values:
                        if value != current[key]:
                            cell_points.append(spec(**dict(current, **{key:value})))
                # Check this joint layout/path change early, including families where a
                # single-coordinate MLS switch would keep an incompatible layout.
                cell_points.insert(0,spec(**dict(current, mls=True, matrix_store=True, mmac=3)))
                tune_groups.append(cell_points)
            elif args.phase == "refine":
                points.extend(spec(**base, **v) for v in variants("full") if v["mode"] == row["mode"])
            else:
                for dim in (512,1024):
                    points.append(spec(**base, mode=row["mode"], stages=row["stages"], regs=row["regs"],
                        warps=row["warps"], empty=row["empty_arrive"], cluster=row["cluster"],
                        mmac=row["mmac_layout"], mls=row["mls"], matrix_store=row["matrix_store"],
                        shape=(dim,dim,row["shape"][2])))
                    points[-1]["validation_role"]="winner"
                    classic=classic_alternatives.get((tuple(row["tile"]),row["shape"][2],row["layout"]))
                    if classic:
                        winner_point=points[-1]
                        fallback=spec_from_row(classic,shape=(dim,dim,row["shape"][2]))
                        if point_key(fallback,active_id)!=point_key(winner_point,active_id):
                            winner_point["coverage_fallback"]=fallback
                            known=prior_by_key.get(point_key(winner_point,active_id),{})
                            if recoverable_full_rejection(known):
                                # Cached rejected winners are skipped by the executor,
                                # so their replacement must be an explicit early point.
                                points.append(dict(fallback,validation_role="coverage_fallback"))
                    # Validate an actual operator-level gain against the same tile's ordinary baseline.
                    points.append(spec(**base, mode="nowasp", stages=2, warps=4,
                                       shape=(dim,dim,row["shape"][2])))
                    points[-1]["validation_role"]="ordinary_baseline"
                    alternative=packed_alternatives.get((tuple(row["tile"]),row["shape"][2],row["layout"]))
                    grid=math.ceil(dim/row["tile"][0])*math.ceil(dim/row["tile"][1])
                    wasp=wasp_alternatives.get((tuple(row["tile"]),row["shape"][2],row["layout"]))
                    if wasp and grid>row.get("device_properties",{}).get("multiprocessor_count",48) and \
                            spec_from_row(wasp)!=spec_from_row(row):
                        points.append(spec_from_row(wasp,shape=(dim,dim,row["shape"][2])))
                        points[-1]["validation_role"]="wasp_challenger"
                    # TT/TN probes show nearly identical single-CTA latency but
                    # 20-23% operator gains. Compare the packed WDRA challenger
                    # even when its coarse resource bound matches the latency winner.
                    if alternative and grid>row.get("device_properties",{}).get("multiprocessor_count",48) and \
                            spec_from_row(alternative)!=spec_from_row(row):
                        points.append(spec_from_row(alternative,shape=(dim,dim,row["shape"][2])))
                        points[-1]["validation_role"]="packed_challenger"
                    higher=higher_cta_alternatives.get((tuple(row["tile"]),row["shape"][2],row["layout"]))
                    if higher and alternative and grid>48*resource_cta_bound(alternative) and \
                            all(spec_from_row(higher)!=spec_from_row(other) for other in (row,alternative)):
                        points.append(spec_from_row(higher,shape=(dim,dim,row["shape"][2])))
                        points[-1].update(validation_role="higher_cta_challenger",single_cta_source_id=higher["id"],
                            resource_reference_id=alternative["id"],resource_reference_bound=resource_cta_bound(alternative),
                            resource_candidate_bound=resource_cta_bound(higher))
                    ordinary=ordinary_alternatives.get((tuple(row["tile"]),row["shape"][2],row["layout"]))
                    if ordinary and grid>row.get("device_properties",{}).get("multiprocessor_count",48) and \
                            spec_from_row(ordinary)!=spec_from_row(row):
                        points.append(spec_from_row(ordinary,shape=(dim,dim,row["shape"][2])))
                        points[-1]["validation_role"]="ordinary_challenger"
                    # MLS can win single-CTA latency yet lose operator throughput
                    # to a smaller ordinary warp count. Keep the fastest classic
                    # ordinary configuration as a separate multi-CTA challenger.
                    classic=classic_alternatives.get((tuple(row["tile"]),row["shape"][2],row["layout"]))
                    if classic and grid>row.get("device_properties",{}).get("multiprocessor_count",48) and \
                            spec_from_row(classic)!=spec_from_row(row) and \
                            (not ordinary or spec_from_row(classic)!=spec_from_row(ordinary)):
                        points.append(spec_from_row(classic,shape=(dim,dim,row["shape"][2])))
                        points[-1]["validation_role"]="classic_challenger"
                    lower=lower_warp_alternatives.get((tuple(row["tile"]),row["shape"][2],row["layout"]))
                    if lower and grid>row.get("device_properties",{}).get("multiprocessor_count",48) and \
                            all(not other or spec_from_row(lower)!=spec_from_row(other) for other in (row,ordinary,classic)):
                        points.append(spec_from_row(lower,shape=(dim,dim,row["shape"][2])))
                        points[-1]["validation_role"]="lower_warp_challenger"
        if args.phase == "tune":
            points=[point for group in itertools.zip_longest(*tune_groups) for point in group if point is not None]
        elif args.phase=="validate":
            priority={"winner":0,"coverage_fallback":0,"wasp_challenger":1,"packed_challenger":1,"higher_cta_challenger":1,"ordinary_challenger":1,"classic_challenger":1,"lower_warp_challenger":1,"ordinary_baseline":2}
            points.sort(key=lambda p:(priority[p["validation_role"]],p["shape"][0],p["shape"][2],p["tile"],p["layout"]))
    else:
        # Small representative 64x64x32 first; progressively cover the power-of-two grid.
        tiles = list(itertools.product((32,64,128,256), (32,64,128,256), (16,32,64,128)))
        tiles.sort(key=lambda t: (t != (64,64,32), max(t), t))
        # First cover every tile/layout/K with a baseline, then compare mode families.
        # This preserves a usable map even if the overnight deadline truncates later work.
        points = [spec(tile=t, k=k, layout=layout, **v) for v, t, k, layout in
                  itertools.product(variants(args.phase), tiles, (1024,4096), ("TT","TN"))]
    if args.limit:
        points = points[:args.limit]
    write_json(ROOT / ("plan-" + args.phase + ".json"), points)
    print(f"phase={args.phase} points={len(points)}", flush=True)
    if args.plan_only:
        return
    session_lock = (ROOT / "scan.lock").open("w")
    fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = capture_manifest()
    manifest_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    context_id = measurement_context(manifest)
    write_json(ROOT / ("manifest-" + manifest_id + ".json"), manifest)
    provenance = ROOT / "provenance" / manifest_id
    provenance.mkdir(parents=True, exist_ok=True)
    for filename in ("scan.py", "worker.py", "env.sh", "run.sh", "control.json"):
        source = ROOT / filename
        if source.exists() and not (provenance / filename).exists():
            if filename=="scan.py":
                (provenance/filename).write_bytes(CONTROLLER_SOURCE)
            else:
                shutil.copy2(source, provenance / filename)
    for repo in ("triton", "bolt_ops"):
        patch = subprocess.check_output(["git", "-C", str(ROOT / repo), "diff", "HEAD", "--binary"])
        if patch:
            (provenance / (repo + ".patch")).write_bytes(patch)
    write_json(ROOT / "active-manifest.json", dict(manifest_id=manifest_id, measurement_context_id=context_id))
    prior, _ = aggregate(runroot)
    for row in prior:
        if row["measurement_context_id"] != context_id:
            continue
        if (row.get("status")=="PASS" and "median_cycles" in row) or row.get("status") in ("RESOURCE_EXCLUDED", "STATS_INVALID", "ACCURACY_FAIL") or \
                (row.get("status") in ("ERROR", "PROCESS_FAILED") and row.get("phase") == "compile"):
            EXISTING_POINTS[point_key(row, context_id)] = row
    print(f"measurement_context={context_id} reusable_points={len(EXISTING_POINTS)}", flush=True)
    control=json.loads((ROOT/"control.json").read_text()) if (ROOT/"control.json").exists() else {}
    stage=os.environ.get("FP16_MAP_STAGE",args.phase)
    workers=args.workers or int(control.get("parallel_workers_by_stage",{}).get(stage,control.get("parallel_workers",1)))
    if not 1<=workers<=64:
        raise ValueError("parallel_workers must be in [1,64]")
    certificate=json.loads((ROOT/"parallel-validation.json").read_text()) if (ROOT/"parallel-validation.json").exists() else {}
    saturation=json.loads((ROOT/"parallel-saturation-validation.json").read_text()) if (ROOT/"parallel-saturation-validation.json").exists() else {}
    rollout=json.loads((ROOT/"parallel-rollout.json").read_text()) if (ROOT/"parallel-rollout.json").exists() else {}
    required_saturated_workers=workers if workers>8 else 2
    if workers>1 and not (rollout.get("passed") and rollout.get("measurement_context_id")==context_id and
            len(rollout.get("comparisons",[]))>=workers and certificate.get("passed") and
            certificate.get("measurement_context_id")==context_id and len(certificate.get("workers",[]))>=workers and
            saturation.get("isolation_under_load") and saturation.get("measurement_context_id")==context_id and
            saturation.get("worker_count",0)>=required_saturated_workers):
        print("parallel model validation missing for this context; using one worker",flush=True)
        workers=1
    execute_points(points,runroot,manifest_id,context_id,args.phase,workers)


if __name__ == "__main__":
    main()
