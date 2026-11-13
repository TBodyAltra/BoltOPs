# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Opt-in full-size PMD regression tests, separate from real-device timings."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('qwen35_pmd_regression', ROOT / 'scripts/performance/qwen35_fp8_pmd_regression.py')
regression = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = regression
SPEC.loader.exec_module(regression)
CASES = regression.discover_cases(ROOT)


@pytest.mark.bench
@pytest.mark.pmd_regression
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.key)
def test_recorded_qwen35_fp8_model_performance(case, request, tmp_path):
    results = request.config.getoption('--pmd-results')
    threshold = request.config.getoption('--pmd-max-regression-percent')
    if results:
        report = regression.assess(case, Path(results) / case.directory, threshold)
    else:
        if os.environ.get('BOLTOPS_RUN_PMD_REGRESSION') != '1':
            pytest.skip('set BOLTOPS_RUN_PMD_REGRESSION=1 for complete core-only PMD replays')
        if not os.environ.get('PMD_PATH'):
            pytest.fail('source the recorded PMD environment before enabling this suite')
        output = request.config.getoption('--pmd-output')
        directory = (Path(output).resolve() / case.directory) if output else tmp_path / case.directory
        report = regression.run_case(case, directory, threshold)
    assert report['status'] == 'PASS', report
