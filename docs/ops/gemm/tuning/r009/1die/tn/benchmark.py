# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""1-DIE TN blockscale benchmark and single-TG PERF capture for the recorded PMD environment."""

import argparse
import hashlib
import importlib
import json
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
    counts = {
        int(key): int(value)
        for key, value in re.findall(
            r"^system\.shaders\.CUs(\d+)\.completedWGs\s+(\d+)",
            path.read_text(), re.M,
        )
    }
    frequency = values["simFreq"]
    period = values["system.shaders.clk_domain.clock"]
    return dict(
        kernel_cycles=values["system.gcSimCycles"],
        kernel_us=values["system.gcSimCycles"] * period / frequency * 1e6,
        active_CUs=sum(count > 0 for count in counts.values()),
        completed_TGs=sum(counts.values()),
        completedWGs=counts,
        stats_path=str(path),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-tg-perf", action="store_true", help="capture three single-TG launches instead of full-grid timing")
    args = parser.parse_args()
    record_path = args.record.resolve()
    record = json.loads(record_path.read_text())
    case = dict(shape=record["shape"], dies=record["environment"]["dies"],
                layout=record["layout"], seed=record["benchmark"].get("seed", 21),
                repeats=record["benchmark"].get("repeats", 1))
    if args.single_tg_perf:
        case["repeats"] = 3
    if case["dies"] != 1 or case["layout"] != "TN":
        raise ValueError("this benchmark records 1-DIE TN only")
    m, n, k = case["shape"]
    if m % 128 or n % 256 or k % 128:
        raise ValueError("the recorded pipeline requires complete 128x256x128 tiles")
    marker = "BOLTOPS_PERF_REPLAYED_RECORD"
    if os.environ.get(marker) != str(record_path):
        environment = dict(os.environ)
        for key, value in record["environment"]["environment_variables"].items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        if args.single_tg_perf:
            environment.update(record["single_tg_perf"]["environment_overrides"])
        environment[marker] = str(record_path)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    repository = next((parent for parent in Path(__file__).resolve().parents
                       if (parent / "boltops/gemm/config.py").is_file()), None)
    if repository is not None:
        sys.path.insert(0, str(repository))

    import torch
    import triton
    from boltops.gemm.config import get_gemm_config
    from boltops.gemm.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    from boltops.utility.arch_info import get_device_info

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 48:
        raise ValueError("the recorded configuration requires gfx946 / 48 CU")
    torch.set_num_threads(1)
    measured_m, measured_n = (128, 256) if args.single_tg_perf else (m, n)
    generator = torch.Generator().manual_seed(case.get("seed", 21))
    a = (torch.randn((measured_m, k), generator=generator) * .5).to(torch.float8_e4m3fn)
    b = (torch.randn((measured_n, k), generator=generator) * .5).to(torch.float8_e4m3fn)
    sa = torch.rand((measured_m, k // 128), generator=generator) * .2 + .05
    sb = (torch.rand((measured_n // 128, k // 128), generator=generator) * .2 + .05).bfloat16().float()
    golden = torch.zeros((measured_m, measured_n), dtype=torch.float32)
    for group in range(k // 128):
        part = slice(group * 128, (group + 1) * 128)
        golden += ((a[:, part].float() @ b[:, part].float().T)
                   * sa[:, group, None]
                   * sb[:, group].repeat_interleave(128)[None, :])
    golden = golden.bfloat16().float()
    operands = [value.cuda() for value in (a, b, sa, sb)]
    sentinel = -4096.
    storage = torch.full((measured_m + 2, n), sentinel, device="cuda", dtype=torch.bfloat16)
    result = storage[1:-1]
    torch.cuda.synchronize()

    profile = ("DTYPE=float8_e4m3fn-AS=float32-BS=float32-OUT=bfloat16-"
               "RECIPE_A=1x128-RECIPE_B=128x128")
    config, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", m, n, k, backend="gluon",
        arch=device.arch, cu_num=device.cu_count, profile=profile,
    )
    if not tuned:
        raise ValueError("no measured configuration for this shape")
    module = importlib.import_module("boltops.gemm.gluon.gemm_a8w8_blockscale")
    kernel = getattr(module, config.pop("kernel"))
    config = {key: tuple(value) if isinstance(value, list) else value
              for key, value in config.items()}
    grid = (1, 1) if args.single_tg_perf else (m // 128, n // 256)
    handle = kernel.warmup(
        *operands, result, M=m, N=n, K=k, GROUP_N=128, GROUP_K=128,
        grid=grid, **config,
    )
    handle._init_handles()
    if handle.n_spills or handle.metadata.shared > 131072:
        raise RuntimeError("spill or LDS capacity check failed")
    digest = hashlib.sha256(handle.asm["hsaco"]).hexdigest()
    for suffix in ("ttgir", "llir", "amdgcn"):
        Path("kernel." + suffix).write_text(handle.asm[suffix])
    Path("kernel.hsaco").write_bytes(handle.asm["hsaco"])
    launches = []
    original_run = kernel.run
    public_calls = []

    def checked_run(*values, **options):
        compiled = original_run(*values, **options)
        if hashlib.sha256(compiled.asm["hsaco"]).hexdigest() != digest:
            raise RuntimeError("public API dispatched a different kernel")
        public_calls.append(compiled.metadata.name)
        return compiled

    kernel.run = checked_run
    report = dict(
        record=str(record_path), status="RUNNING", shape=[m, n, k], measured_grid=list(grid),
        operand_strides={name: list(tensor.stride())
                         for name, tensor in zip(("x", "w", "x_scale", "w_scale"), operands)},
        kernel_sha256=digest, source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        triton=triton.__version__, metadata=handle.metadata._asdict(), launches=launches,
        timing=dict(simulation_wall="host wall through synchronize, includes PMD simulation",
                    kernel="complete PMD task: system.gcSimCycles times clock period",
                    excluded="CPU reference, scale promotion, uploads, JIT and output readback"),
    )
    try:
        for repeat in range(case.get("repeats", 1)):
            before = set(Path("m5out/0").glob("*/stats.txt"))
            start = time.perf_counter()
            if args.single_tg_perf:
                kernel[grid](*operands, result, M=m, N=n, K=k, GROUP_N=128, GROUP_K=128, **config)
            else:
                gemm_a8w8_blockscale(*operands, out=result, backend="auto")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if len(public_calls) != repeat + 1:
                raise RuntimeError("public API did not select the saved Gluon kernel")
            files = set(Path("m5out/0").glob("*/stats.txt")) - before
            if len(files) != 1:
                raise RuntimeError("expected exactly one GEMM dispatch")
            perf = read_stats(files.pop())
            if not args.single_tg_perf:
                perf["kernel_tops"] = 2 * m * n * k / (perf["kernel_us"] * 1e6)
            host = storage.cpu().float()
            actual = host[1:-1, :measured_n]
            error = (actual - golden).abs()
            accuracy = dict(
                max_abs=float(error.max()),
                mismatch=int((actual != golden).sum()),
                outside_tolerance=int((error > .001 + .01 * golden.abs()).sum()),
                nonfinite=int((~actual.isfinite()).sum()), elements=measured_m * measured_n,
                canary_ok=bool((host[0] == sentinel).all() and (host[-1] == sentinel).all()
                               and (host[1:-1, measured_n:] == sentinel).all()),
            )
            launches.append(dict(simulator_wall_seconds=elapsed, **perf, accuracy=accuracy))
            if (perf["completed_TGs"] != grid[0] * grid[1] or perf["active_CUs"] != (1 if args.single_tg_perf else 48)
                    or accuracy["outside_tolerance"] or accuracy["nonfinite"]
                    or not accuracy["canary_ok"]):
                raise RuntimeError("full grid, precision or boundary check failed")
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
