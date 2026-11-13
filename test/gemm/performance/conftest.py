# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

def pytest_addoption(parser):
    group = parser.getgroup('recorded-pmd-performance')
    group.addoption('--pmd-results', help='evaluate saved full-grid results without launching kernels')
    group.addoption('--pmd-output', help='directory for new PMD measurements')
    group.addoption('--pmd-max-regression-percent', type=float, default=1.0,
                    help='maximum allowed kernel latency increase (default: 1%%)')
