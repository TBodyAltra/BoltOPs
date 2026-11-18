# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Standalone Shaobo PMD regression for recorded Qwen3.5 FP8-model GEMMs.

Source the recorded PMD environment, then run from the repository:
  python scripts/performance/qwen35_fp8_pmd_regression.py --output /tmp/qwen35-perf
  python scripts/performance/qwen35_fp8_pmd_regression.py --case R005 --output /tmp/r005-perf
  python scripts/performance/qwen35_fp8_pmd_regression.py --evaluate --output /tmp/qwen35-perf

Kernel latency increases above 1% fail; exactly 1% passes. Published baselines
are never updated. DIE/layout records remain separate. --evaluate checks saved
full-grid measurements without launching kernels. --jobs permits independent
core-only PMD instances. This script is independent of NMZ CI.
The model suite includes its BF16 GEMMs as well as FP8 blockscale GEMMs.
Experimental candidate records participate only when explicitly registered for regression.
R005 reuses its archived benchmark's fixed-kernel timing and public-output check.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import configparser
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import threading

if __package__:
    from . import gemm_a16w16_pmd_regression as linear
else:
    import gemm_a16w16_pmd_regression as linear

REPO = Path(__file__).resolve().parents[2]
DOCUMENT = Path('docs/ops/gemm/SHAOBO_BU_REQUIREMENTS.md')
MODEL = 'Qwen3.5-35B-A3B FP8'


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def normalize_record(record, record_path=None):
    """Adapt the legacy R005 job/mean schema without changing its baseline."""
    record = linear.normalize_record(record, record_path)
    if record.get('benchmark', {}).get('interface') == 'public_job_json':
        performance = record['performance']
        performance['kernel_us'] = performance['mean_kernel_us']
        performance['launched_TGs'] = math.prod(record['pipeline']['grid'])
        if performance['completed_TGs'] != performance['launched_TGs']:
            raise ValueError('public job did not complete its published grid')
        benchmark = record['benchmark']
        benchmark['warm_repeats'] = benchmark['job']['repeats']
        benchmark['repeats'] = benchmark['warm_repeats'] + 1
    elif 'mean_kernel_us' in record['performance']:
        performance = record['performance']
        if 'PMD' not in performance.get('metric', ''):
            raise ValueError('legacy record is not a PMD cycle baseline')
        performance['kernel_us'] = performance['mean_kernel_us']
        performance['launched_TGs'] = math.prod(record['grid'])
        environment = record['environment']
        environment['type'] = 'PMD'
        environment['clock_period_ps'] = {'gc': environment['clock_ps']}
        environment['environment_variables'] = environment['variables']
        # R005's archived benchmark takes a job file, uses seed 37, then
        # measures one cold launch followed by the recorded warm repeats.
        warm_repeats = len(performance['warm_cycles'])
        if warm_repeats == 0:
            raise ValueError('legacy baseline has no warm samples')
        record['benchmark'] = dict(script=record['artifacts']['published']['benchmark'],
                                   interface='job_json', seed=37, warm_repeats=warm_repeats,
                                   repeats=warm_repeats + 1, job_config={'BN': record['tile'][1]})
        environment['environment_variables'].setdefault(
            'PYTHONPATH', str(Path(environment['variables']['TRITON_HOME']).parent / 'runtime'))
    if 'launched_TGs' not in record['performance'] and 'grid' in record.get('pipeline', {}):
        tgs = math.prod(record['pipeline']['grid'])
        if record['performance'].get('completed_TGs') != tgs:
            raise ValueError('completed TG count differs from the published pipeline grid')
        record['performance']['launched_TGs'] = tgs
    return record


@dataclass(frozen=True)
class Case:
    record_path: Path
    record: dict

    @property
    def key(self):
        return f"{self.record['requirement']}/{self.record['environment']['dies']}die/{self.record['layout'].lower()}"

    @property
    def directory(self):
        return self.key.lower()


def discover_cases(repo=REPO):
    repo = Path(repo)
    document = repo / DOCUMENT
    text = document.read_text()
    heading = '### ' + MODEL + '\n'
    if heading not in text:
        raise ValueError('model section is missing')
    section = text.split(heading, 1)[1].split('\n### ', 1)[0]
    links = set(re.findall(r'(?:href="|\]\()([^"\)]+/record\.json)', section))
    cases = []
    for link in sorted(links):
        path = (document.parent / link).resolve()
        path.relative_to(repo.resolve())
        record = load(path)
        if record.get('regression', {}).get('role') == 'compiler_comparison':
            continue
        if (record.get('kernel_symbol') == 'candidate_kernel'
                and record.get('implementation', '').startswith('experimental ')
                and not record.get('regression', {}).get('operator') == linear.OPERATOR):
            continue
        record = normalize_record(record, path)
        positive(record['performance']['kernel_us'], 'baseline kernel_us')
        positive(record['performance']['launched_TGs'], 'baseline launched_TGs')
        if record['environment']['type'] != 'PMD':
            raise ValueError(f'{path}: this suite accepts PMD records only')
        if record['layout'] not in ('TT', 'TN') or record['environment']['dies'] not in (1, 2):
            raise ValueError(f'{path}: invalid DIE/layout record')
        if len(record['shape']) != 3 or not all(isinstance(x, int) and x > 0 for x in record['shape']):
            raise ValueError(f'{path}: invalid MNK')
        cases.append(Case(path, record))
    if not cases or len({case.key for case in cases}) != len(cases):
        raise ValueError('missing or duplicate recorded model cases')
    return sorted(cases, key=lambda case: case.key)


verify_model = linear.verify_model
verify_sample = linear.verify_sample


def evaluate(case, case_output, max_regression_percent=1.0):
    """Validate raw evidence before classifying performance; never update baselines."""
    if linear.is_linear_record(case.record):
        report = linear.evaluate(linear.Case(case.record_path, case.record), case_output,
                                 max_regression_percent)
        report['case'] = case.key
        return report

    if not math.isfinite(max_regression_percent) or max_regression_percent < 0:
        raise ValueError('regression threshold must be finite and nonnegative')
    measurement = Path(case_output) / 'measurement'
    result = load(measurement / 'result.json')
    if result.get('status') != 'PASS' or result.get('single_tg') is True:
        raise ValueError('benchmark did not pass a complete-grid measurement')
    if 'shape' in result and result['shape'] != case.record['shape']:
        raise ValueError('benchmark shape differs from recorded complete MNK')
    model = verify_model(case.record, measurement)
    repeats = case.record['benchmark']['repeats']
    if case.record['benchmark'].get('interface') == 'public_job_json':
        if (result.get('job') != case.record['benchmark']['job']
                or result.get('public_config') != case.record['config']['launch_options']):
            raise ValueError('public job shape, seed, repeats or kernel differs from baseline')
        samples = result['measurements']
        if (len(samples) != repeats or not samples[0]['cold']
                or any(s['cold'] for s in samples[1:])):
            raise ValueError('public job warm sampling differs from baseline')
        times = []
        paths = []
        for sample in samples:
            if len(sample['stages']) != 1:
                raise ValueError('public job must measure one complete dispatch')
            stats = sample['stages'][0]
            accuracy = dict(sample['accuracy'], canary_ok=sample['accuracy']['guards_ok'])
            normalized = dict(accuracy=accuracy, kernel_us=sample['us'],
                              stats_path=stats['path'], completed_TGs=stats['completed_TGs'])
            paths.append((measurement / stats['path']).resolve())
            times.append(verify_sample(case.record, normalized, measurement, model['clock_period_ps']))
        if len(set(paths)) != repeats:
            raise ValueError('public job reuses native stats across samples')
        measured = statistics.mean(times[1:])
        aggregation = 'mean excluding initial cold launch'
    elif 'measurements' in result:
        if result['job']['shape'] != case.record['shape'] or math.prod(result['grid']) != case.record['performance']['launched_TGs']:
            raise ValueError('legacy benchmark used a partial shape/grid')
        samples = result['measurements']
        if len(samples) != repeats or not samples[0]['warmup'] or any(s['warmup'] for s in samples[1:]):
            raise ValueError('legacy warm-mean sampling differs from baseline')
        times = []
        for sample in samples:
            stats = sample['stats']
            accuracy = dict(sample['accuracy'], elements=math.prod(case.record['shape'][:2]), canary_ok=sample['accuracy']['guards_ok'])
            normalized = dict(accuracy=accuracy, kernel_us=stats['cycles']*stats['clock_ps']/1e6,
                              stats_path=stats['path'], completed_TGs=stats['completed_TGs'])
            times.append(verify_sample(case.record, normalized, measurement, model['clock_period_ps']))
        if result.get('public_auto_exact_match') is not True:
            raise ValueError('public auto dispatch was not validated')
        measured = statistics.mean(times[1:])
        aggregation = 'mean excluding initial cold launch'
    elif 'samples' in result:
        if 'median' not in case.record['performance'].get('aggregation', ''):
            raise ValueError('unrecognized median aggregation protocol')
        samples = result['samples']
        if len(samples) != repeats or repeats < 2 or not result.get('validation'):
            raise ValueError('missing recorded warm-median samples or validation')
        for sample in result['validation']:
            verify_sample(case.record, sample, measurement, model['clock_period_ps'])
        times = [verify_sample(case.record, s, measurement, model['clock_period_ps']) for s in samples]
        measured = statistics.median(times[1:])
        aggregation = 'median excluding first of recorded repeats'
    else:
        samples = result['launches']
        if len(samples) != repeats or repeats != 1:
            raise ValueError('single complete-dispatch baseline requires exactly one sample')
        measured = verify_sample(case.record, samples[0], measurement, model['clock_period_ps'])
        times = [measured]
        aggregation = 'one complete dispatch'
    reported = result.get('performance', {}).get('kernel_us')
    if reported is not None and not math.isclose(reported, measured, rel_tol=1e-10):
        raise ValueError('benchmark aggregation differs from recorded protocol')
    baseline = positive(case.record['performance']['kernel_us'], 'baseline kernel_us')
    limit = baseline * (1 + max_regression_percent/100)
    regressed = measured > limit and not math.isclose(measured, limit, rel_tol=1e-12)
    m, n, k = case.record['shape']
    return dict(case=case.key, status='REGRESSION' if regressed else 'PASS',
                shape=case.record['shape'], baseline_kernel_us=baseline, kernel_us=measured,
                kernel_tops=2*m*n*k/(measured*1e6), regression_percent=(measured/baseline-1)*100,
                max_regression_percent=max_regression_percent, aggregation=aggregation,
                samples_kernel_us=times, launched_TGs=case.record['performance']['launched_TGs'],
                environment=model, baseline_record_sha256=digest(case.record_path))


def assess(case, output, threshold):
    try:
        return evaluate(case, output, threshold)
    except (OSError, ValueError, KeyError, TypeError, IndexError, configparser.Error) as error:
        return dict(case=case.key, status='INVALID', error=str(error))


def run_case(case, case_output, threshold=1.0):
    if linear.is_linear_record(case.record):
        report = linear.run_case(linear.Case(case.record_path, case.record), case_output, threshold)
        report['case'] = case.key
        save(Path(case_output) / 'regression.json', report)
        return report
    output = Path(case_output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = json.loads(json.dumps(case.record))
    linear.refresh_sources(record)
    record['environment']['environment_variables'].update(
        TRITON_CACHE_DIR=str(output / 'cache'), TRITON_HOME=str(output / 'triton-home'),
        BOLTOPS_GEMM_CONFIG_DIR=None)
    replay = output / 'replay-record.json'
    save(replay, record)
    script = case.record_path.parent / record['benchmark']['script']
    if record['benchmark'].get('sha256', digest(script)) != digest(script):
        raise ValueError('archived benchmark changed; retain its recorded measurement protocol')
    environment = dict(os.environ)
    if record['benchmark'].get('interface') in ('job_json', 'public_job_json'):
        for key, value in record['environment']['environment_variables'].items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        # The legacy script does not locate the repository itself.
        environment['PYTHONPATH'] = str(REPO) + os.pathsep + environment.get('PYTHONPATH', '')
        expected_library = record['environment'].get('triton_library_sha256')
        if expected_library:
            libraries = [Path(root)/'triton/_C/libtriton.so' for root in environment['PYTHONPATH'].split(os.pathsep)]
            if not any(path.is_file() and digest(path)==expected_library for path in libraries):
                raise ValueError('recorded Triton library does not match; R005 needs its wave-sharing fixes')
        measurement = output / 'measurement'
        measurement.mkdir()
        if record['benchmark']['interface'] == 'public_job_json':
            job = record['benchmark']['job']
        else:
            job = dict(shape=record['shape'], config=record['benchmark']['job_config'],
                       repeats=record['benchmark']['warm_repeats'], seed=record['benchmark']['seed'])
        save(output / 'job.json', job)
        command = [sys.executable, str(script), str(output / 'job.json')]
        working_directory = measurement
    else:
        command = [sys.executable, str(script), '--record', str(replay), '--output', str(output / 'measurement')]
        if '--cu0-perf' in shlex.split(record['benchmark'].get('command', '')):
            command.append('--cu0-perf')
        working_directory = output
    save(output / 'invocation.json', dict(command=command, benchmark_sha256=digest(script),
                                         baseline_record=str(case.record_path), cache_isolated=True))
    with (output / 'benchmark.log').open('w') as log:
        child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=working_directory, env=environment)
    if child.returncode:
        report = dict(case=case.key, status='ERROR', exit_code=child.returncode,
                      log=str(output / 'benchmark.log'))
    else:
        report = assess(case, output, threshold)
    save(output / 'regression.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', default=[], help='requirement or DIE/layout key; repeatable')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=int, default=1, help='independent core-only PMD processes (default: 1)')
    parser.add_argument('--max-regression-percent', type=float, default=1.0)
    parser.add_argument('--evaluate', action='store_true', help='validate existing measurements; launch no kernels')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1 or not math.isfinite(args.max_regression_percent) or args.max_regression_percent < 0:
        parser.error('jobs must be positive; regression threshold must be finite and nonnegative')
    try:
        cases = discover_cases()
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    selectors = {s.lower() for s in args.case}
    known = {c.key.lower() for c in cases} | {c.record['requirement'].lower() for c in cases}
    if selectors - known:
        parser.error('unknown recorded cases: ' + ', '.join(sorted(selectors-known)))
    cases = [c for c in cases if not selectors or c.key.lower() in selectors or c.record['requirement'].lower() in selectors]
    if args.list:
        print(json.dumps([dict(case=c.key, shape=c.record['shape'], baseline_kernel_us=c.record['performance']['kernel_us']) for c in cases], indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required')
    output = args.output.resolve()
    if args.evaluate:
        if not output.is_dir():
            parser.error('--evaluate requires an existing output directory')
    else:
        if output.exists():
            parser.error('output directory must be new; existing measurements are never overwritten')
        output.mkdir(parents=True, exist_ok=False)
    state = dict(model=MODEL, max_regression_percent=args.max_regression_percent, status='RUNNING',
                 cases={c.key: dict(case=c.key, status='QUEUED') for c in cases})
    lock = threading.Lock()

    def update(key, report):
        with lock:
            state['cases'][key] = report
            save(output / 'report.json', state)

    def work(case):
        update(case.key, dict(case=case.key, status='RUNNING'))
        folder = output / case.directory
        try:
            report = assess(case, folder, args.max_regression_percent) if args.evaluate else run_case(case, folder, args.max_regression_percent)
        except Exception as error:
            report = dict(case=case.key, status='ERROR', error=str(error))
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / 'regression.json', report)
        update(case.key, report)
        print(json.dumps(report), flush=True)
        return report

    save(output / 'report.json', state)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        reports = list(pool.map(work, cases))
    state['status'] = 'PASS' if all(row['status']=='PASS' for row in reports) else 'FAIL'
    save(output / 'report.json', state)
    return 0 if state['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
