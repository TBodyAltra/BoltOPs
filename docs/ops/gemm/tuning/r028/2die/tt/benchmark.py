"""Replay the R028 sbx2 GEMM using the R024 tuning harness."""

from pathlib import Path
from runpy import run_path


if __name__ == '__main__':
    runner = Path(__file__).resolve().parents[3] / 'r024/2die/tt/benchmark.py'
    run_path(str(runner))['main']('R028')
