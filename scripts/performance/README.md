# Qwen3.5-35B-A3B FP8 model performance regression

This suite discovers performance records linked from the model's section in
`docs/ops/gemm/SHAOBO_BU_REQUIREMENTS.md`. It includes BF16 operators in the FP8
model (R001) and the updated R005. Requirement, DIE count and TT/TN layout form
separate case keys. A new record linked in this model section is discovered
automatically.

The default limit is **1% kernel latency increase**: exactly 1% passes; more
than 1% is `REGRESSION`. Baselines are read from the existing records and are
never updated automatically. Each case keeps its sampling protocol:

| Records | Complete-grid timing |
| --- | --- |
| R001 | Median of the eight warm samples, excluding initial timed sample |
| R005 | Mean of three warm launches, excluding the cold launch |
| R009–R013 | One complete dispatch, as recorded |

R005 uses its recorded newer Triton wave-sharing runtime; the suite verifies
its compiler library hash. Its record also stores the runtime path and job
protocol needed to replay the published benchmark.

## Run

These are standalone scripts for the Shaobo PMD environment, separate from
the NMZ CI suite under `test/`. Source the recorded gfx946 / 1-DIE / 48-CU
PMD environment first. The runner
starts each benchmark in a separate process, model directory and compilation
cache. It replays the original complete-MNK benchmark via the current code and
configuration. Output directories must be new. Logs and failed measurements
are retained.

```bash
python scripts/performance/qwen35_fp8_pmd_regression.py --list
python scripts/performance/qwen35_fp8_pmd_regression.py --output /tmp/qwen35-fp8-regression
python scripts/performance/qwen35_fp8_pmd_regression.py --case R009/1die/tn --output /tmp/r009-regression
```

Default execution is serial. `--jobs N` permits independent core-only PMD
instances when the machine has sufficient CPU/RAM. Model IPC must be disabled;
these baselines use simulated kernel time rather than host simulation time.
Large complete grids can take hours to simulate.

The runner writes `report.json` plus per-case logs, invocation metadata, model
configuration, statistics and `regression.json`. It fails on a performance
regression, child failure, missing data, a partial grid, incomplete output
validation, nonfinite output, a failed boundary check or incompatible model
geometry/clock/configuration. It independently recomputes timing from the
saved `system.gcSimCycles` and clock, and checks completed TG counts. There is
no single-TG substitution for a complete performance measurement. R009's
recorded `--cu0-perf` flag is preserved for a matching capture environment.

Re-evaluate existing raw measurements with a different limit without running
kernels again:

```bash
python scripts/performance/qwen35_fp8_pmd_regression.py --evaluate --output /tmp/qwen35-fp8-regression --max-regression-percent 1
```

## Check the report validator

The optional CPU-only self-checks require pytest and start no PMD kernels:

```bash
python -m pytest --confcutdir=scripts/performance scripts/performance/check_qwen35_fp8_pmd_report.py -q
```

The scripts do not add tests, options or markers to the NMZ CI suite.
