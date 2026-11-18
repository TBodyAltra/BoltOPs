# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Replay a recorded 1-die TN blockscale GEMM through the public interface.

Full-grid timing excludes input preparation, uploads, JIT and output readback.
--single-tg-perf selects one TG from the full-shape compiled kernel for profiling;
that partial grid is never used as the full-shape performance measurement.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


def read_stats(path):
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) > 1:
            try:
                values[fields[0]] = float(fields[1])
            except ValueError:
                pass
    counts = {int(c): int(v) for c, v in re.findall(
        r"^system\.shaders\.CUs(\d+)\.completedWGs\s+(\d+)", path.read_text(), re.M)}
    clock = values["system.shaders.clk_domain.clock"]
    return dict(kernel_cycles=values["system.gcSimCycles"],
                kernel_us=values["system.gcSimCycles"] * clock / values["simFreq"] * 1e6,
                completed_TGs=sum(counts.values()), active_CUs=sum(v > 0 for v in counts.values()),
                completedWGs=counts, stats_path=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--single-tg-perf", action="store_true")
    capture.add_argument("--cu0-perf", action="store_true")
    args = parser.parse_args()
    record_path = args.record.resolve()
    record = json.loads(record_path.read_text())
    if record["environment"]["dies"] != 1 or record["layout"] != "TN":
        raise ValueError("this replay supports recorded 1-die TN only")
    marker = "BOLTOPS_PERF_REPLAYED_RECORD"
    if os.environ.get(marker) != str(record_path):
        environment = dict(os.environ)
        for key, value in record["environment"]["environment_variables"].items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        environment["GPU_DFLAGS"] = "['StatLog']"
        environment.pop("HSA_TOOLS_LIB", None)
        if args.single_tg_perf or args.cu0_perf:
            environment.update(record["single_tg_perf"]["environment_overrides"])
        environment[marker] = str(record_path)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    repository = next(parent for parent in Path(__file__).resolve().parents
                      if (parent / "boltops/gemm/config.py").is_file())
    sys.path.insert(0, str(repository))
    import torch
    import triton
    from boltops.gemm import gemm_a8w8_blockscale
    from boltops.gemm.config import get_gemm_config
    from boltops.utility.arch_info import get_device_info

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 48:
        raise ValueError("recorded configuration requires gfx946 / 48 CU")
    torch.set_num_threads(1)
    m, n, k = record["shape"]
    profile = "DTYPE=float8_e4m3fn-AS=float32-BS=float32-OUT=bfloat16-RECIPE_A=1x128-RECIPE_B=128x128"
    config, tuned = get_gemm_config("GEMM-A8W8_BLOCKSCALE", m, n, k,
        profile=profile, backend="gluon", arch=device.arch, cu_num=device.cu_count)
    if not tuned:
        raise ValueError("the shape has no recorded Gluon configuration")
    module = importlib.import_module("boltops.gemm.gluon.gemm_a8w8_blockscale")
    kernel = getattr(module, config.pop("kernel"))
    config = {key: tuple(value) if isinstance(value, list) else value for key, value in config.items()}
    bm, bn = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]
    if m % bm or n % bn or k % 128:
        raise ValueError("the recorded pipeline requires complete tiles")
    total = (m // bm) * (n // bn)
    if "NUM_SMS" in config:
        grid = (1 if args.single_tg_perf else config["NUM_SMS"],)
        ids = [tile for pid in range(grid[0]) for tile in range(pid, total, config["NUM_SMS"])]
    else:
        grid = (1, 1) if args.single_tg_perf else (m // bm, n // bn)
        ids = [row + col * (m // bm) for row in range(grid[0]) for col in range(grid[1])]
    selected = torch.zeros((m, n), dtype=torch.bool)
    for tile in ids:
        row, col = tile % (m // bm), tile // (m // bm)
        selected[row * bm:(row + 1) * bm, col * bn:(col + 1) * bn] = True
    generator = torch.Generator().manual_seed(record["benchmark"].get("seed", 21))
    a = (torch.randn((m, k), generator=generator) * .5).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), generator=generator) * .5).to(torch.float8_e4m3fn)
    sa = torch.rand((m, k // 128), generator=generator) * .2 + .05
    sb = (torch.rand((n // 128, k // 128), generator=generator) * .2 + .05).bfloat16().float()
    golden = torch.zeros((m, n))
    for group in range(k // 128):
        part = slice(group * 128, (group + 1) * 128)
        golden += ((a[:, part].float() @ b[:, part].float().T)
                   * sa[:, group, None] * sb[:, group].repeat_interleave(128)[None, :])
    reference = golden.bfloat16().float()[selected]
    operands = [value.cuda() for value in (a, b, sa, sb)]
    # The profiling helper captures the first code object. Avoid introducing
    # a PyTorch fill kernel ahead of the GEMM in profiling mode.
    if args.single_tg_perf or args.cu0_perf:
        storage = torch.full((m * n + 512,), -4096., dtype=torch.bfloat16).cuda()
    else:
        storage = torch.full((m * n + 512,), -4096., dtype=torch.bfloat16, device="cuda")
    result = storage[256:-256].view(m, n)
    common = dict(M=m, N=n, K=k, GROUP_N=128, GROUP_K=128, **config)
    handle = kernel.warmup(*operands, result, grid=grid, **common)
    handle._init_handles()
    if handle.n_spills or handle.metadata.shared > 131072:
        raise RuntimeError("spill or LDS capacity check failed")
    for suffix in ("ttgir", "llir", "amdgcn"):
        Path("kernel." + suffix).write_text(handle.asm[suffix])
    Path("kernel.hsaco").write_bytes(handle.asm["hsaco"])
    digest = hashlib.sha256(handle.asm["hsaco"]).hexdigest()
    original_run = kernel.run
    calls = []

    def checked_run(*values, **options):
        compiled = original_run(*values, **options)
        if hashlib.sha256(compiled.asm["hsaco"]).hexdigest() != digest:
            raise RuntimeError("public interface dispatched a different kernel")
        calls.append(compiled.metadata.name)
        return compiled

    kernel.run = checked_run
    report = dict(status="RUNNING", shape=[m, n, k], single_tg=args.single_tg_perf,
                  measured_grid=list(grid), tile_ids=ids, launches=[], metadata=handle.metadata._asdict(),
                  kernel_sha256=digest, source_sha256=hashlib.sha256(Path(kernel.fn.__code__.co_filename).read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  operand_strides={name: list(value.stride()) for name, value in
                                   zip(("x", "w", "x_scale", "w_scale"), operands)})
    try:
        for repeat in range(record["benchmark"].get("repeats", 1)):
            torch.cuda.synchronize()
            before = set(Path("m5out/0").glob("*/stats.txt"))
            start = time.perf_counter()
            if args.single_tg_perf:
                kernel[grid](*operands, result, **common)
            else:
                gemm_a8w8_blockscale(*operands, out=result, backend="auto")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if len(calls) != repeat + 1:
                raise RuntimeError("public dispatch was not observed")
            files = set(Path("m5out/0").glob("*/stats.txt")) - before
            if len(files) != 1:
                raise RuntimeError("expected exactly one GEMM dispatch")
            stats = read_stats(files.pop())
            host = storage.cpu().float()
            output_matrix = host[256:-256].view(m, n)
            actual = output_matrix[selected]
            error = (actual - reference).abs()
            accuracy = dict(max_abs=float(error.max()), mismatch=int((actual != reference).sum()),
                            outside_tolerance=int((error > .001 + .01 * reference.abs()).sum()),
                            nonfinite=int((~actual.isfinite()).sum()), elements=actual.numel(),
                            canary_ok=bool((host[:256] == -4096).all() and (host[-256:] == -4096).all()
                                           and (output_matrix[~selected] == -4096).all()))
            report["launches"].append(dict(repeat=repeat, simulator_wall_seconds=elapsed,
                **stats, accuracy=accuracy,
                perf_files=[str(p) for p in Path(stats["stats_path"]).parent.glob("*.perf")]))
            if (stats["completed_TGs"] != math.prod(grid) or stats["active_CUs"] != (1 if args.single_tg_perf else 48)
                    or accuracy["outside_tolerance"] or accuracy["nonfinite"] or not accuracy["canary_ok"]):
                raise RuntimeError("grid, precision or boundary validation failed")
        report["status"] = "PASS"
    except BaseException:
        report["status"] = "ERROR"
        raise
    finally:
        Path("result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        kernel.run = original_run
    print(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
