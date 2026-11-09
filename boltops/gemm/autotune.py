# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Explicit, callback-driven GEMM tuning; normal auto dispatch never benchmarks."""
from __future__ import annotations
from copy import deepcopy
from contextlib import contextmanager
import hashlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile

from .dispatch import ExecutionPlan, candidate_plans, supports


@dataclass(frozen=True)
class Trial:
    plan: ExecutionPlan
    score: float | None
    error: str | None = None


@dataclass(frozen=True)
class AutotuneResult:
    winner: ExecutionPlan
    trials: tuple[Trial, ...]


@contextmanager
def _publish_lock(path):
    # A stable machine-local lock prevents lost updates between tuning workers.
    # Keep it outside the configuration directory (and therefore outside Git).
    identity = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    user = str(os.getuid()) if hasattr(os, 'getuid') else hashlib.sha256(str(Path.home()).encode()).hexdigest()[:16]
    lock_dir = Path(tempfile.gettempdir()) / ('boltops-gemm-config-locks-' + user)
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    with (lock_dir / identity).open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if lock.tell() == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def save_winner(problem, plan, path):
    """Serialize local writers and atomically publish one winner."""
    with _publish_lock(path):
        _save_winner(problem, plan, path)


def _save_winner(problem, plan, path):
    """Atomically publish configuration only, keeping unrelated records intact."""
    from .performance_map import (document, _validate_document,
                                  exact_match, clear_cache)
    if not supports(plan.implementation, problem, plan.config):
        raise ValueError("cannot save an unsupported GEMM plan")
    target = Path(path)
    if target.name != f"gemm_{problem.interface}.json":
        raise ValueError("winner output must use the interface's configuration filename")
    data = (json.loads(target.read_text()) if target.exists() else
            deepcopy(document(problem.interface)))
    _validate_document(data, problem.interface)
    match = exact_match(problem)
    # Put the exact winner before broader rules; preserve other rules' order.
    others = [e for e in data['entries'] if e['match'] != match]
    c = plan.config
    config = {'backend': plan.backend, 'implementation': plan.implementation,
              'params': c.params}
    data['entries'] = [{'match': match, 'config': config}] + others
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, prefix='.' + target.name,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    clear_cache()


def autotune(problem, *, benchmark, validate, plans=None, names=None,
             backend='auto', output_path=None):
    """Select a validated plan by an explicitly supplied lower-is-better metric.

    validate(plan) must compile/warm up, execute and check numerical correctness;
    it may return None/True or raise/return False to reject a candidate.
    benchmark(plan) measures the warmed kernel (device time or PMD cycles).
    No host-clock timing, implicit scans, input conversion or fallback kernels
    are supplied here. Failed tuning trials are returned as failures, never as
    correctness passes. Normal select_plan/launch do not use this failure loop.

    output_path is opt-in and must name gemm_<interface>.json. Point
    BOLTOPS_GEMM_CONFIG_DIR at its directory for later auto dispatch. Runtime
    scores and errors stay in the result, not the configuration file.
    """
    from .dispatch import validate_backend
    validate_backend(backend)
    if not callable(validate) or not callable(benchmark):
        raise TypeError('autotune requires explicit validation and benchmark callbacks')
    if plans is not None and names is not None:
        raise ValueError('provide plans or implementation names, not both')
    plans = (candidate_plans(problem, names, backend=backend, tuning=True)
             if plans is None else plans)
    trials = []
    seen = set()
    for plan in plans:
        if backend != 'auto' and plan.backend != backend:
            continue
        key = (plan.backend, plan.implementation, plan.config)
        if key in seen:
            continue
        seen.add(key)
        try:
            if not supports(plan.implementation, problem, plan.config):
                raise ValueError('unsupported candidate configuration')
            validation = validate(plan)
            if validation is not None and validation is not True:
                raise ValueError('numerical validation rejected candidate')
            metric = benchmark(plan)
            if isinstance(metric, bool):
                raise ValueError("benchmark must return a numeric score, not a validation boolean")
            score = float(metric)
            if not math.isfinite(score) or score <= 0:
                raise ValueError('benchmark must return a finite positive score')
            trials.append(Trial(plan, score))
        except Exception as error:
            trials.append(Trial(plan, None, f'{type(error).__name__}: {error}'))
    valid = [trial for trial in trials if trial.score is not None]
    if not valid:
        errors = [f'{t.plan.implementation}: {t.error}' for t in trials]
        raise RuntimeError(f'no validated GEMM tuning candidate: {errors}')
    best = min(valid, key=lambda t: t.score)
    if output_path is not None:
        save_winner(problem, best.plan, output_path)
    return AutotuneResult(best.plan, tuple(trials))
