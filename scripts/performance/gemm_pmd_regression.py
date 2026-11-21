# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Run one representative published PMD benchmark per BF16/FP8 GEMM kernel.

Use --list to inspect the default coverage or --case to select published
records explicitly. Each case keeps its complete MNK, grid, validation and
timing protocol. Existing records provide the latency baseline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
from types import ModuleType

if __package__:
    from . import gemm_a16w16_pmd_regression as linear
    from . import qwen35_fp8_pmd_regression as fp8
else:  # Direct execution with python scripts/performance/...
    import gemm_a16w16_pmd_regression as linear
    import qwen35_fp8_pmd_regression as fp8


@dataclass(frozen=True)
class Benchmark:
    runner: ModuleType
    case: linear.Case | fp8.Case

    @property
    def kernel(self):
        record = self.case.record
        source = record["source"]
        symbol = source.get("current_symbol", source["symbol"])
        config = record.get("config", {})
        if {"path", "profile", "M_key"} <= config.keys():
            # A current config can use a renamed selector for the same replay case.
            path = linear.REPO / config["path"]
            if path.is_file():
                profile = linear.load(path).get("profiles", {}).get(config["profile"], {})
                symbol = profile.get(config["M_key"], {}).get("kernel", symbol)
        return f"{source.get('current_path', source['path'])}:{symbol}"

    @property
    def dtype(self):
        return self.case.dtype if self.runner is linear else "fp8"

    @property
    def aliases(self):
        keys = {self.case.key, self.case.record["requirement"]}
        if self.runner is linear:
            keys.add(self.case.legacy_key)
        return {key.lower() for key in keys}

    @property
    def size_order(self):
        record = self.case.record
        # Prefer the smallest full workload. Break equal MNK work by the
        # published latency and then by a stable identifier.
        return (math.prod(record["shape"]), record["performance"]["kernel_us"], self.case.key)

    def describe(self):
        record = self.case.record
        registration = record.get("regression", {})
        standalone = registration.get("interface") == "variant_blocks" and not registration.get("public_auto")
        return dict(case=self.case.key, dtype=self.dtype, kernel=self.kernel,
                    route="standalone_variant" if standalone else "recorded_replay",
                    shape=record["shape"], baseline_kernel_us=record["performance"]["kernel_us"],
                    record=str(self.case.record_path.relative_to(linear.REPO)))


def select_benchmarks(selectors=()):
    candidates = [
        Benchmark(linear, case) for case in linear.discover_cases() if case.dtype == "bf16"
    ] + [Benchmark(fp8, case) for case in fp8.discover_cases(include_linear=False)]
    keys = {key.lower() for key in selectors}
    if keys:
        known = set().union(*(benchmark.aliases for benchmark in candidates))
        if keys - known:
            raise ValueError("unknown records: " + ", ".join(sorted(keys - known)))
        return [benchmark for benchmark in candidates if benchmark.aliases & keys]
    kernels = {}
    for benchmark in sorted(candidates, key=lambda row: row.size_order):
        kernels.setdefault(benchmark.kernel, benchmark)
    return sorted(kernels.values(), key=lambda row: (row.dtype, row.case.key))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="published requirement or full case key; repeatable")
    parser.add_argument("--output", type=Path, help="new report directory, or measurement directory with --evaluate")
    parser.add_argument("--jobs", type=int, default=1, help="independent core-only PMD processes (default: 1)")
    parser.add_argument("--max-regression-percent", type=float, default=1.0)
    parser.add_argument("--list", action="store_true", help="show selected cases and kernels without starting PMD")
    parser.add_argument("--evaluate", action="store_true", help="validate saved measurements without starting PMD")
    args = parser.parse_args(argv)
    threshold = args.max_regression_percent
    if args.jobs < 1 or not math.isfinite(threshold) or threshold < 0:
        parser.error("jobs must be positive and the regression threshold finite and nonnegative")
    try:
        benchmarks = select_benchmarks(args.case)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.list:
        print(json.dumps([row.describe() for row in benchmarks], indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if args.evaluate:
        if not output.is_dir():
            parser.error("--evaluate requires an existing output directory")
    else:
        if output.exists():
            parser.error("--output must be a new directory")
        output.mkdir(parents=True)

    state = dict(status="RUNNING", max_regression_percent=threshold,
                 cases={row.case.key: {**row.describe(), "status": "QUEUED"} for row in benchmarks})
    lock = threading.Lock()

    def work(benchmark):
        case = benchmark.case
        directory = output / case.directory
        runner = benchmark.runner
        try:
            result = (runner.assess if args.evaluate else runner.run_case)(case, directory, threshold)
        except Exception as error:
            result = dict(status="ERROR", error=str(error))
        result = {**benchmark.describe(), **result}
        directory.mkdir(parents=True, exist_ok=True)
        linear.save(directory / "regression.json", result)
        with lock:
            state["cases"][case.key] = result
            linear.save(output / "report.json", state)
            print(json.dumps(result), flush=True)
        return result

    linear.save(output / "report.json", state)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(work, benchmarks))
    state["status"] = "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"
    linear.save(output / "report.json", state)
    return int(state["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
