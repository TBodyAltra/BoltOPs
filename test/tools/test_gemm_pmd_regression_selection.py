from collections import defaultdict

from scripts.performance import gemm_pmd_regression as regression


def test_default_selects_smallest_published_case_for_each_kernel_die_and_layout():
    candidates = defaultdict(list)
    for runner, cases in (
        (regression.linear, regression.linear.discover_cases()),
        (regression.fp8, regression.fp8.discover_cases(include_linear=False)),
    ):
        for case in cases:
            benchmark = regression.Benchmark(runner, case)
            candidates[benchmark.coverage].append(benchmark)

    selected = {benchmark.coverage: benchmark for benchmark in regression.select_benchmarks()}

    assert selected.keys() == candidates.keys()
    assert all(selected[kernel] == min(rows, key=lambda row: row.size_order)
               for kernel, rows in candidates.items())
    assert {row.dtype for row in selected.values()} == {"bf16", "fp8"}
    assert any("/triton/" in kernel for kernel, *_ in selected)
    assert any("/gluon/" in kernel for kernel, *_ in selected)


def test_explicit_record_keeps_published_shape_and_timing():
    [benchmark] = regression.select_benchmarks(["R002/1die/tt"])
    assert benchmark.case.key == "R002/bf16/1die/tt"
    assert benchmark.case.record["shape"] == [64, 248320, 2048]
    assert benchmark.case.record["performance"]["launched_TGs"] == 970
    assert regression.linear.timing_protocol(benchmark.case.record) == "validated_warm_median"


def test_fp8_discovery_skips_bf16_and_accepts_recorded_completion_count():
    cases = {case.key: case.record for case in regression.fp8.discover_cases(include_linear=False)}
    assert "R001/1die/tn" not in cases
    assert cases["R008/2die/tn"]["performance"]["launched_TGs"] == 96


def test_early_public_replay_accepts_perf_metadata_without_path(tmp_path):
    [benchmark] = regression.select_benchmarks(["R001/1die/tn"])
    replay = regression.linear.replay_record(benchmark.case, tmp_path)

    assert replay["benchmark"]["timing_mode"] == "validated_warm_median"
    assert replay["performance"]["kernel_us"] == benchmark.case.record["performance"]["kernel_us"]
    assert replay["environment"]["environment_variables"]["GPU_OUT"] == "m5out"


def test_default_covers_both_streamk_layouts_and_fp8_widths():
    cases = {row.case.key: row for row in regression.select_benchmarks()}
    assert cases["R020/bf16/2die/tn"].kernel == cases["R020/bf16/2die/tt"].kernel
    assert cases["R005/1die/tn"].kernel == cases["R005/2die/tn"].kernel
    assert "R029/bf16/2die/tt" in cases
    assert cases["R004/bf16/2die/tn"].describe()["route"] == "standalone_variant"
    assert cases["R004/bf16/2die/tt"].case.record["regression"]["public_auto"]
