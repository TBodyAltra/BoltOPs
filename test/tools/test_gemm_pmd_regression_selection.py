from collections import defaultdict

from scripts.performance import gemm_pmd_regression as regression


def test_default_selects_smallest_published_case_for_each_kernel():
    candidates = defaultdict(list)
    for runner in (regression.linear, regression.fp8):
        for case in runner.discover_cases():
            benchmark = regression.Benchmark(runner, case)
            candidates[benchmark.kernel].append(benchmark)

    selected = {benchmark.kernel: benchmark for benchmark in regression.select_benchmarks()}

    assert selected.keys() == candidates.keys()
    assert all(selected[kernel] == min(rows, key=lambda row: row.size_order)
               for kernel, rows in candidates.items())
    assert {row.dtype for row in selected.values()} == {"bf16", "fp8"}
    assert any("/triton/" in kernel for kernel in selected)
    assert any("/gluon/" in kernel for kernel in selected)


def test_explicit_record_keeps_published_shape_and_timing():
    [benchmark] = regression.select_benchmarks(["R002/1die/tt"])
    assert benchmark.case.key == "R002/bf16/1die/tt"
    assert benchmark.case.record["shape"] == [64, 248320, 2048]
    assert benchmark.case.record["performance"]["launched_TGs"] == 970
    assert regression.linear.timing_protocol(benchmark.case.record) == "public_warm_median"


def test_fp8_discovery_skips_bf16_and_accepts_recorded_completion_count():
    cases = {case.key: case.record for case in regression.fp8.discover_cases()}
    assert "R001/1die/tn" not in cases
    assert cases["R008/2die/tn"]["performance"]["launched_TGs"] == 96


def test_early_public_replay_accepts_perf_metadata_without_path(tmp_path):
    [benchmark] = regression.select_benchmarks(["R001/1die/tn"])
    replay = regression.linear.replay_record(benchmark.case, tmp_path)

    assert replay["benchmark"]["timing_mode"] == "public_warm_median"
    assert replay["performance"]["kernel_us"] == benchmark.case.record["performance"]["kernel_us"]
    assert replay["environment"]["environment_variables"]["GPU_OUT"] == "m5out"
