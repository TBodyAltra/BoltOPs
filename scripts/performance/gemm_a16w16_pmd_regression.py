# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""PMD regression for recorded FP16/BF16 linear GEMMs (gemm_a16w16).

Prepare the recorded PMD/Torch/Triton environment, then run from the repository:
  python scripts/performance/gemm_a16w16_pmd_regression.py --list
  python scripts/performance/gemm_a16w16_pmd_regression.py --dtype bf16 --case R021 --output /tmp/a16-perf
  python scripts/performance/gemm_a16w16_pmd_regression.py --evaluate --output /tmp/a16-perf --case R021

Discover published gemm_a16w16 records and explicitly registered replay cases in the requirements
document, or supply --record (repeatable) instead. FP16 and BF16 require separate
baselines; selecting a precision with no records is an error. Preserve each
record's cold/first-timed/warm-median protocol. Latency increases above 1% fail;
exactly 1% passes. Published baselines and artifacts are never changed.
Public Gluon records also retain every warm sample after full validation.
--list and --evaluate use only the standard library and launch no kernels.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import configparser
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading

REPO = Path(__file__).resolve().parents[2]
DOCUMENT = Path('docs/ops/gemm/SHAOBO_BU_REQUIREMENTS.md')
OPERATOR = 'boltops/gemm/gemm_a16w16.py'
DTYPES = {'fp16': 'fp16', 'float16': 'fp16', 'half': 'fp16',
          'bf16': 'bf16', 'bfloat16': 'bf16'}


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def positive(value, name, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 or (integer and not isinstance(value, int))):
        raise ValueError(f'{name} must be a finite positive ' + ('integer' if integer else 'number'))
    return value


def precision(record):
    types = record['dtype']
    a = DTYPES.get(str(types['A']).lower())
    b = DTYPES.get(str(types.get('W', types.get('B'))).lower())
    output = DTYPES.get(str(types['output']).lower())
    if a is None or a != b or a != output or types.get('scale') is not None:
        raise ValueError('requires unquantized FP16 x FP16 -> FP16 or BF16 x BF16 -> BF16')
    return a


def is_linear_record(record):
    return (record.get('public_api', {}).get('path') == OPERATOR
            or record.get('regression', {}).get('operator') == OPERATOR)


def normalize_variant_record(record, record_path):
    """Enroll a recorded multi-variant benchmark without changing its protocol."""
    if record['status'] != 'PMD_PASS':
        raise ValueError('variant-block baseline is not PMD_PASS')
    registration = record['regression']
    if registration['operator'] != OPERATOR or registration['variant'] != record['best']:
        raise ValueError('regression must select the published linear GEMM variant')
    root = Path(record_path).resolve().parent
    source = (root / registration['source']).resolve()
    script = (root / registration['script']).resolve()
    record['source'] = dict(path=str(source.relative_to(REPO)), symbol=record['kernel_symbol'],
                            sha256=record['provenance']['kernel_sha256'])
    record['benchmark'].update(script=registration['script'],
        sha256=record['provenance']['benchmark_sha256'], interface='variant_blocks',
        timing_mode='variant_block_median', validation_count=registration['validation_count'])
    if digest(script) != record['benchmark']['sha256']:
        raise ValueError('archived variant runner changed; retain its measurement protocol')
    dtype = record['precision']
    record['dtype'] = dict(A=dtype['A'], B=dtype['B'], output=dtype['C'],
                           accumulator=dtype['accumulator'])
    record['status'] = 'PASS'
    cfg = record['config']; m, n, _ = record['shape']
    if m % cfg['BLOCK_SIZE_M'] or n % cfg['BLOCK_SIZE_N']:
        raise ValueError('variant benchmark requires complete output tiles')
    tgs = m // cfg['BLOCK_SIZE_M'] * (n // cfg['BLOCK_SIZE_N'])
    if record['dispatch']['completed_TGs'] != tgs:
        raise ValueError('variant TG count differs from complete grid')
    record['performance']['launched_TGs'] = tgs
    return record


def normalize_paired_record(record, record_path):
    """Enroll a fixed route from the archived counterbalanced measurement."""
    registration = record['regression']
    path = Path(record_path).parent / registration['measurement']
    if digest(path) != record['artifacts'][path.name]['sha256']:
        raise ValueError('paired baseline measurement changed')
    measurement = load(path)
    chosen = measurement['variants'][registration['variant']]
    record['status'] = 'PASS'
    types = record['precision']
    record['dtype'] = dict(A=types['A'], B=types['W'], output=types['C'], accumulator=types['accumulator'])
    record['performance'] = dict(chosen['performance'], launched_TGs=chosen['grid'])
    record['source'] = dict(path='boltops/gemm/gluon/gemm_a16w16.py',
                            symbol=chosen['config']['kernel'])
    record['benchmark'] = dict(script=record['benchmark'], interface='paired_routes',
        sha256=record['artifacts']['benchmark.py']['sha256'], timing_mode='validated_warm_median',
        repeats=measurement['rounds'], validation_count=len(chosen['validation']))
    registration['config'] = chosen['config']
    return record


def normalize_record(record, record_path):
    """Adapt public Gluon measurements without changing their timing baseline."""
    record = deepcopy(record)
    if record.get('regression', {}).get('interface') == 'paired_routes':
        return normalize_paired_record(record, record_path)
    if record.get('regression', {}).get('interface') == 'variant_blocks':
        return normalize_variant_record(record, record_path)
    if record.get('public_api', {}).get('name') != 'gemm_a16w16':
        return record
    if record.get('status') != 'PMD_PASS':
        raise ValueError('public Gluon baseline is not PMD_PASS')
    dtype = record['precision']
    record['dtype'] = dict(A=dtype['A'], B=dtype['B'], output=dtype['C'],
                           accumulator=dtype['accumulator'])
    precision(record)
    record['status'] = 'PASS'
    api = record['public_api']
    api['path'] = OPERATOR
    sources = {item['path']: item for item in record['sources']}
    api['sha256'] = sources[OPERATOR]['sha256']
    source_path = 'boltops/gemm/gluon/gemm_a16w16.py'
    record['source'] = dict(sources[source_path], symbol=record['config']['selected']['kernel'])
    dependencies = {name: item for name, item in sources.items()
                    if name not in (OPERATOR, source_path) and (REPO / name).is_file()}
    # Old measurements may name modules that have since been split or renamed.
    # Track the current entry module's local imports in the private replay copy.
    for node in ast.parse((REPO / source_path).read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            parent = Path(source_path).parent
            for _ in range(node.level - 1):
                parent = parent.parent
            path = parent / (node.module.replace('.', '/') + '.py')
            if (REPO / path).is_file():
                dependencies[str(path)] = dict(path=str(path), sha256=digest(REPO / path))
    record['source_dependencies'] = list(dependencies.values())
    benchmark = record['benchmark']
    if 'median' not in benchmark.get('aggregation', ''):
        raise ValueError('unrecognized public Gluon timing protocol')
    script = Path(record_path).parent / 'benchmark.py'
    if not script.is_file():
        raise ValueError(f'missing public Gluon benchmark: {script}')
    benchmark.update(script=script.name, sha256=benchmark['source_sha256'],
                     timing_mode='validated_warm_median')
    tgs = positive(record['resources']['completed_TGs'], 'complete TG count', integer=True)
    selected = record['config']['selected']
    m, n, _ = record['shape']
    tiles = ((m + selected['BLOCK_SIZE_M'] - 1) // selected['BLOCK_SIZE_M']
             * ((n + selected['BLOCK_SIZE_N'] - 1) // selected['BLOCK_SIZE_N']))
    if tgs != min(tiles, selected.get('NUM_SMS', tiles)):
        raise ValueError('recorded TG count differs from full launch geometry')
    for sample in record.get('samples', []) + record['validation']:
        if sample['completed_TGs'] != tgs:
            raise ValueError('recorded sample TG count differs from complete grid')
    record['performance']['launched_TGs'] = tgs
    return record


def timing_protocol(record):
    """Choose the archived protocol, never infer a new one from current samples."""
    benchmark, performance = record['benchmark'], record['performance']
    repeats = positive(benchmark['repeats'], 'benchmark repeats', integer=True)
    mode = benchmark.get('timing_mode')
    if mode in ('validated_warm_median', 'variant_block_median'):
        positive(benchmark['validation_count'], 'validation count', integer=True)
        return mode
    aggregation = performance.get('aggregation', '')
    if (mode == 'existing_first_timed' or (mode is None and aggregation.startswith('first timed repeat'))) and repeats >= 2:
        return 'first_timed'
    if (mode == 'legacy_median' or (mode is None and aggregation.startswith('median of repeats'))) and repeats >= 2:
        return 'warm_median'
    if mode is None and repeats == 1 and aggregation.startswith('one cold full-grid dispatch'):
        return 'cold'
    raise ValueError('unrecognized recorded timing protocol')


@dataclass(frozen=True)
class Case:
    record_path: Path
    record: dict

    @property
    def dtype(self):
        return precision(self.record)

    @property
    def legacy_key(self):
        return f"{self.record['requirement']}/{self.record['environment']['dies']}die/{self.record['layout'].lower()}"

    @property
    def key(self):
        return f"{self.record['requirement']}/{self.dtype}/{self.record['environment']['dies']}die/{self.record['layout'].lower()}"

    @property
    def directory(self):
        return self.key.lower()


def discover_cases(repo=REPO, records=None):
    """Only linked, published linear records are discovered; scratch runs are excluded."""
    repo = Path(repo).resolve()
    if records:
        paths = sorted({Path(path).resolve() for path in records})
    else:
        document = repo / DOCUMENT
        links = set(re.findall(r'(?:href="|\]\()([^"\)]+/record\.json)', document.read_text()))
        paths = sorted({(document.parent / link).resolve() for link in links})
        for path in paths:
            path.relative_to(repo)
    cases = []
    for path in paths:
        record = normalize_record(load(path), path)
        if not is_linear_record(record):
            if records:
                raise ValueError(f'{path}: not a public linear gemm_a16w16 record')
            continue
        precision(record)
        if record.get('status') != 'PASS':
            raise ValueError(f'{path}: baseline is not PASS')
        if not re.fullmatch(r'R\d+', record['requirement']):
            raise ValueError(f'{path}: invalid requirement identifier')
        environment = record['environment']
        if environment['type'] != 'PMD' or environment['arch'] != 'gfx946':
            raise ValueError(f'{path}: requires a gfx946 PMD baseline')
        if type(environment['dies']) is not int or environment['dies'] not in (1, 2) or record['layout'] not in ('TT', 'TN'):
            raise ValueError(f'{path}: invalid DIE/layout')
        positive(environment['cu_count'], 'CU count', integer=True)
        if len(record['shape']) != 3:
            raise ValueError(f'{path}: invalid MNK')
        for dimension in record['shape']:
            positive(dimension, 'MNK dimension', integer=True)
        positive(record['performance']['kernel_us'], 'baseline kernel_us')
        tgs = positive(record['performance']['launched_TGs'], 'baseline launched_TGs', integer=True)
        if record.get('full_grid_TGs', tgs) != tgs:
            raise ValueError(f'{path}: inconsistent full-grid TG count')
        timing_protocol(record)
        script = record['benchmark'].get('script', record['benchmark'].get('path'))
        if not script or not (path.parent / script).is_file():
            raise ValueError(f'{path}: missing benchmark script')
        cases.append(Case(path, record))
    if not cases or len({case.key for case in cases}) != len(cases):
        raise ValueError('missing or duplicate recorded FP16/BF16 linear GEMM cases')
    return sorted(cases, key=lambda case: case.key)


def objects(node):
    if isinstance(node, dict):
        if 'path' in node:
            yield node
        for value in node.values():
            yield from objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from objects(value)


def configuration(measurement):
    """Both g/config.ini and m5out/config.json are emitted by archived runners."""
    for folder in ('g', 'm5out'):
        path = measurement / folder / 'config.ini'
        if path.is_file():
            parser = configparser.RawConfigParser()
            parser.optionxform = str
            parser.read_string(path.read_text())
            return {section: dict(parser[section]) for section in parser.sections()}
        path = measurement / folder / 'config.json'
        if path.is_file():
            return {obj['path']: obj for obj in objects(load(path))}
    raise ValueError('missing PMD model configuration (g/ or m5out/)')


def comparable(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return str(value).lower()


def verify_model(record, measurement):
    model = configuration(measurement)
    expected = record['environment']
    shaders = [p for p in model if re.fullmatch(r'system(?:\d+)?\.shaders(?:\d+)?', p)]
    cus = [p for p in model if re.fullmatch(r'system(?:\d+)?\.shaders(?:\d+)?\.CUs\d+', p)]
    if len(shaders) != expected['dies'] or len(cus) != expected['cu_count']:
        raise ValueError('PMD DIE/CU count differs from baseline')
    sections = expected.get('runtime_configuration', {}).get('sections', {})
    clock = expected.get('clock_period_ps', {}).get('gc')
    if clock is None:
        clock = float(sections['system.shaders.clk_domain']['clock'])
    positive(clock, 'GC clock period')
    for shader in shaders:
        actual = float(comparable(model[shader + '.clk_domain']['clock']))
        if actual != clock:
            raise ValueError('PMD GC clock differs from baseline')
    gpu = model['gpu_configuration']
    for key in ('ipc_conn', 'skip_cal', 'skip_kernel_exec'):
        if comparable(gpu.get(key)) not in ('false', '0'):
            raise ValueError('requires independent core-only PMD with kernel execution enabled')
    if int(gpu['device_id']) != 0x66b7:
        raise ValueError('PMD device differs from gfx946 baseline')
    configurations = dict(sections.get('gpu_configuration', {}))
    configurations.update(expected.get('gpu_configuration', {}))
    for key, value in configurations.items():
        if comparable(gpu.get(key)) != comparable(value):
            raise ValueError(f'PMD configuration differs: {key}')
    for section, fields in sections.items():
        if section.endswith('.clk_domain') and comparable(model[section]['clock']) != comparable(fields['clock']):
            raise ValueError(f'PMD clock differs: {section}')
    return dict(clock_period_ps=clock, cu_count=len(cus), dies=len(shaders))


def verify_sample(record, sample, measurement, clock):
    accuracy = sample['accuracy']
    if sample.get('status', 'PASS') != 'PASS' or accuracy.get('status', 'PASS') != 'PASS':
        raise ValueError('sample did not pass validation')
    if accuracy['elements'] != math.prod(record['shape'][:2]):
        raise ValueError('precision check did not cover the complete MN output')
    bad = accuracy.get('outside_tolerance', accuracy.get('mismatches', accuracy.get('mismatch')))
    finite = accuracy.get('nonfinite') == 0 or accuracy.get('finite') is True
    if ('nonfinite' in accuracy and accuracy['nonfinite'] != 0) or ('finite' in accuracy and accuracy['finite'] is not True):
        finite = False
    if bad != 0 or not finite or accuracy.get('canary_ok', True) is not True or accuracy.get('guards_ok', True) is not True or accuracy.get('inputs_byte_identical', True) is not True:
        raise ValueError('precision, nonfinite, boundary or input-preservation check failed')
    reference = record.get('validation', {})
    if isinstance(reference, list):
        reference = reference[0].get('accuracy', {}) if reference else {}
    for key in ('atol', 'rtol'):
        if key in reference and accuracy.get(key) != reference[key]:
            raise ValueError('precision tolerances differ from baseline protocol')
    for key in ('canary_ok', 'inputs_byte_identical'):
        if reference.get(key) is True and accuracy.get(key) is not True:
            raise ValueError(f'missing recorded validation: {key}')
    for key in ('max_abs', 'mean_abs_vs_fp32', 'rounding_floor'):
        if key in accuracy and (not math.isfinite(accuracy[key]) or accuracy[key] < 0):
            raise ValueError(f'invalid numerical error summary: {key}')
    if (record['benchmark'].get('timing_mode') in ('legacy_median', 'existing_first_timed')
            or record.get('source', {}).get('symbol') == '_gemm_fp16_mls_kernel'):
        if accuracy['mean_abs_vs_fp32'] > 1.05 * accuracy['rounding_floor'] + 1e-6:
            raise ValueError('mean numerical error exceeds recorded rounding tolerance')
        if accuracy['atol'] != .02 or accuracy['rtol'] != .01:
            raise ValueError('precision tolerances differ from baseline protocol')
    path = (measurement / sample['stats_path']).resolve()
    path.relative_to(measurement.resolve())
    if 'stats_sha256' in sample and digest(path) != sample['stats_sha256']:
        raise ValueError('stats SHA-256 mismatch')
    text = path.read_text()
    if 'End Simulation Statistics' not in text:
        raise ValueError('incomplete PMD statistics')
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        key = fields[0]
        if key in values and (key in ('system.gcSimCycles', 'simFreq', 'gcFrequency') or key.endswith('.completedWGs')):
            raise ValueError(f'duplicate PMD statistic: {key}')
        values[key] = value
    cycles = positive(values['system.gcSimCycles'], 'gcSimCycles')
    if values['simFreq'] != 1e12:
        raise ValueError('PMD tick resolution differs from picosecond clocks')
    counts = [v for k, v in values.items() if re.fullmatch(r'system(?:\d+)?\.shaders(?:\d+)?\.CUs\d+\.completedWGs', k)]
    expected = record['performance']['launched_TGs']
    if (len(counts) != record['environment']['cu_count'] or any(v < 0 or not math.isfinite(v) or not v.is_integer() for v in counts)
            or sum(counts) != expected or sample['completed_TGs'] != expected):
        raise ValueError('completed TG count differs from complete baseline grid')
    if values.get('system.shaders.clk_domain.clock', clock) != clock:
        raise ValueError('stats clock differs from model clock')
    # Preserve the archived conversion: rounded gcFrequency is slightly different
    # from 1000 / clock_period_ps, and must not be silently replaced.
    if 'gcFrequency' in record['performance'].get('time_domain', ''):
        frequency = positive(values['gcFrequency'], 'gcFrequency')
        baseline_frequency = record['performance'].get('gc_frequency_ghz', record['performance'].get('frequency_ghz'))
        if (not math.isclose(frequency, 1000 / clock, rel_tol=1e-6)
                or (baseline_frequency is not None and not math.isclose(frequency, baseline_frequency, rel_tol=1e-10))):
            raise ValueError('PMD GC frequency differs from baseline/model')
        us = cycles / (frequency * 1000)
    else:
        us = cycles * clock / 1e6
    if not math.isclose(positive(sample['kernel_us'], 'kernel_us'), us, rel_tol=1e-10):
        raise ValueError('reported time is not the complete PMD kernel time')
    for key in ('kernel_cycles', 'dispatch_cycles'):
        if key in sample and sample[key] != cycles:
            raise ValueError('reported cycle count differs from raw statistics')
    return us


def variant_result(record, result, measurement):
    """Select the fixed published variant, not whichever variant wins this run."""
    if (result.get('status') != 'COMPLETE' or result.get('stage') != 'complete'
            or result.get('shape') != record['shape'] or result.get('layout') != record['layout']):
        raise ValueError('variant runner did not complete the recorded shape/layout')
    target = record['regression']['variant']
    configs = [('baseline', record['baseline']['config']), (target, record['config']),
               (record['interleaved_comparison']['best'], record['interleaved_comparison']['config'])]
    if list(result['variants']) != [name for name, _ in configs]:
        raise ValueError('variant block order differs from the published protocol')
    paths = []
    model = verify_model(record, measurement)
    for name, cfg in configs:
        variant = result['variants'][name]
        if (variant.get('status') != 'PASS' or variant['config'] != cfg
                or variant['resources']['SGPR_spills'] or variant['resources']['VGPR_spills']):
            raise ValueError('variant configuration, status or spill check failed')
        confirmation = variant['confirmation']
        if (len(confirmation['samples']) != record['benchmark']['repeats']
                or len(confirmation['validation']) != record['benchmark']['validation_count']):
            raise ValueError('variant block has different timing/validation counts')
        m, n, _ = record['shape']
        grid = m // cfg['BLOCK_SIZE_M'] * (n // cfg['BLOCK_SIZE_N'])
        if variant['grid'] != grid:
            raise ValueError('variant did not run the complete grid')
        expected = dict(record, performance=dict(record['performance'], launched_TGs=grid))
        for sample in confirmation['validation'] + confirmation['samples']:
            verify_sample(expected, sample, measurement, model['clock_period_ps'])
        paths.extend(confirmation['validation'])
    for name, _ in configs:
        paths.extend(result['variants'][name]['confirmation']['samples'])
    indexes = [int(Path(sample['stats_path']).parent.name) for sample in paths]
    if indexes != sorted(set(indexes)):
        raise ValueError('variant dispatch blocks are reordered or reuse statistics')
    chosen = result['variants'][target]
    if chosen['grid'] != record['performance']['launched_TGs']:
        raise ValueError('selected variant did not run the complete grid')
    return dict(status='PASS', requirement=record['requirement'], shape=result['shape'],
                layout=result['layout'], samples=chosen['confirmation']['samples'],
                validation=chosen['confirmation']['validation'], performance=chosen['performance'])


def evaluate(case, case_output, max_regression_percent=1.0):
    if not math.isfinite(max_regression_percent) or max_regression_percent < 0:
        raise ValueError('regression threshold must be finite and nonnegative')
    measurement = Path(case_output) / 'measurement'
    result = load(measurement / 'result.json')
    record = case.record
    if record['benchmark'].get('interface') == 'paired_routes':
        if (result.get('status') != 'PASS' or result.get('shape') != record['shape']
                or result.get('rounds') != record['benchmark']['repeats']):
            raise ValueError('paired runner did not complete its recorded protocol')
        target = record['regression']['variant']
        if list(result['variants']) != ['default_triton', target]:
            raise ValueError('paired route order differs from the archived protocol')
        chosen = result['variants'][target]
        if chosen['config'] != record['regression']['config']:
            raise ValueError('paired runner changed the fixed published configuration')
        model = verify_model(record, measurement)
        for variant in result['variants'].values():
            expected = dict(record, performance=dict(record['performance'], launched_TGs=variant['grid']))
            if (len(variant['samples']) != record['benchmark']['repeats']
                    or len(variant['validation']) != record['benchmark']['validation_count']):
                raise ValueError('paired runner changed sampling or validation counts')
            for sample in variant['validation'] + variant['samples']:
                verify_sample(expected, sample, measurement, model['clock_period_ps'])
        result = dict(status='PASS', requirement=record['requirement'], shape=result['shape'],
            samples=chosen['samples'], validation=chosen['validation'], performance=chosen['performance'],
            public_api_verified=True, benchmark_sha256=record['benchmark']['sha256'])
    if timing_protocol(record) == 'variant_block_median':
        result = variant_result(record, result, measurement)
    if result.get('status') != 'PASS' or result.get('single_tg') is True:
        raise ValueError('benchmark did not pass a complete-grid measurement')
    if result['requirement'] != record['requirement'] or result.get('shape', record['shape']) != record['shape']:
        raise ValueError('benchmark requirement/shape differs from baseline')
    for key in ('dtype', 'layout'):
        if key in result and result[key] != record[key]:
            raise ValueError(f'benchmark {key} differs from baseline')
    model = verify_model(record, measurement)
    protocol = timing_protocol(record)
    repeats = record['benchmark']['repeats']
    flag_reset_us = 0.0
    if protocol == 'cold':
        paths = sorted(measurement.glob('g/*/*/stats.txt')) + sorted(measurement.glob('m5out/*/*/stats.txt'))
        setup = result.get('setup_dispatches', [])
        if setup:
            if (record['source']['symbol'] != '_gemm_a16w16_bf16_tt_bm256_bn128_bk64_persistent_kernel'
                    or len(setup) != 1 or setup[0].get('kind') != 'streamk_flag_reset'):
                raise ValueError('unexpected setup dispatch before cold GEMM')
            reset = setup[0]
            path = (measurement / reset['stats_path']).resolve()
            path.relative_to(measurement.resolve())
            if path not in {p.resolve() for p in paths} or digest(path) != reset['stats_sha256']:
                raise ValueError('flag-reset stats are missing or changed')
            text = path.read_text()
            counts = re.findall(r'^system(?:\d+)?\.shaders(?:\d+)?\.CUs\d+\.completedWGs\s+(\d+)', text, re.M)
            mmops = re.findall(r'^system(?:\d+)?\.shaders(?:\d+)?\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)', text, re.M)
            cycles = re.findall(r'^system.gcSimCycles\s+([\deE.+-]+)', text, re.M)
            if ('End Simulation Statistics' not in text or len(counts) != model['cu_count']
                    or sum(map(int, counts)) != 1 or len(mmops) != model['cu_count']
                    or any(map(int, mmops)) or len(cycles) != 1
                    or reset['kernel_cycles'] != float(cycles[0])
                    or not math.isclose(reset['kernel_us'], float(cycles[0]) * model['clock_period_ps'] / 1e6, rel_tol=1e-10)):
                raise ValueError('invalid native flag reset before cold Stream-K GEMM')
            flag_reset_us = reset['kernel_us']
            paths = [p for p in paths if p.resolve() != path]
        if len(paths) != 1:
            raise ValueError('cold baseline requires exactly one complete dispatch')
        if 'samples' in result or 'launches' in result:
            samples = result.get('samples', result.get('launches'))
            if len(samples) != 1:
                raise ValueError('cold baseline requires exactly one complete dispatch')
        else:
            if not isinstance(result['validation'], dict):
                raise ValueError('missing full-output validation for cold dispatch')
            sample = dict(result['performance'], accuracy=result['validation'],
                          stats_path=str(paths[0].relative_to(measurement)), stats_sha256=result['stats_sha256'])
            samples = [sample]
        if (measurement / samples[0]['stats_path']).resolve() != paths[0].resolve():
            raise ValueError('cold sample does not reference the complete dispatch stats')
        if result['performance'].get('execution_warmups', 0) != 0:
            raise ValueError('cold baseline must not have execution warmups')
        validation = []
        selected = [0]
    elif protocol in ('validated_warm_median', 'variant_block_median'):
        samples, validation = result['samples'], result['validation']
        if len(samples) != repeats or len(validation) != record['benchmark']['validation_count']:
            raise ValueError('missing recorded warm samples or complete public validation')
        if any(sample.get('phase', 'timed') != 'timed' for sample in samples):
            raise ValueError('validated warm samples must not include a cold/warmup dispatch')
        if protocol == 'validated_warm_median':
            if result.get('public_api_verified') is not True:
                raise ValueError('missing complete public API validation')
            if result.get('benchmark_sha256') != record['benchmark']['sha256']:
                raise ValueError('measurement benchmark differs from recorded protocol')
        selected = list(range(repeats))
    else:
        samples, validation = result['samples'], result['validation']
        if len(samples) != repeats or not isinstance(validation, list) or len(validation) != 3:
            raise ValueError('missing recorded repeats or three independent validation cases')
        for index, sample in enumerate(samples):
            if sample.get('index', index) != index or sample.get('phase', 'warmup' if index == 0 else 'timed') != ('warmup' if index == 0 else 'timed'):
                raise ValueError('warmup/timed sample order differs from baseline')
        if record['benchmark'].get('timing_mode') or record.get('source', {}).get('symbol') == '_gemm_fp16_mls_kernel':
            seed = record['benchmark']['seed']
            if result['timing_input'] != dict(seed=seed, input_kind='normal_random'):
                raise ValueError('timing inputs differ from baseline')
            for index, sample in enumerate(validation):
                if (sample['seed'] != seed + 1000 * index or sample['phase'] != 'correctness'
                        or sample['input_kind'] != ('ones' if index == 1 else 'uniform_random')):
                    raise ValueError('independent validation inputs differ from baseline')
        selected = [1] if protocol == 'first_timed' else list(range(1, repeats))
    paths = [(measurement / sample['stats_path']).resolve() for sample in validation + samples]
    if len(paths) != len(set(paths)):
        raise ValueError('multiple samples reuse the same stats file')
    for sample in validation:
        verify_sample(record, sample, measurement, model['clock_period_ps'])
    times = [verify_sample(record, sample, measurement, model['clock_period_ps']) for sample in samples]
    measured = statistics.median(times[index] for index in selected) + flag_reset_us
    performance = result['performance']
    if not math.isclose(positive(performance['kernel_us'], 'reported kernel_us'), measured, rel_tol=1e-10):
        raise ValueError('benchmark aggregation differs from recorded protocol')
    if performance.get('selected_indexes', selected) != selected:
        raise ValueError('benchmark selected different timing samples')
    baseline = record['performance']['kernel_us']
    limit = baseline * (1 + max_regression_percent / 100)
    regressed = measured > limit and not math.isclose(measured, limit, rel_tol=1e-12)
    return dict(case=case.key, dtype=case.dtype, status='REGRESSION' if regressed else 'PASS',
                shape=record['shape'], baseline_kernel_us=baseline, kernel_us=measured,
                kernel_tops=2 * math.prod(record['shape']) / (measured * 1e6),
                regression_percent=(measured / baseline - 1) * 100,
                max_regression_percent=max_regression_percent, aggregation=protocol,
                selected_indexes=selected, samples_kernel_us=times,
                launched_TGs=record['performance']['launched_TGs'], environment=model,
                baseline_record_sha256=digest(case.record_path))


def assess(case, output, threshold):
    try:
        return evaluate(case, output, threshold)
    except (OSError, ValueError, KeyError, TypeError, IndexError, configparser.Error) as error:
        return dict(case=case.key, dtype=case.dtype, status='INVALID', error=str(error))


def refresh_sources(record):
    """Refresh only a private replay copy; published baselines stay untouched."""
    for name in ('source', 'config', 'public_api'):
        item = record.get(name)
        if not item or 'path' not in item:
            continue
        path = REPO / item['path']
        item['sha256'] = digest(path)
        if 'kernel_sha256' in item:
            source = path.read_text()
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == item['symbol'])
            start = min([node.lineno] + [n.lineno for n in node.decorator_list])
            item['kernel_sha256'] = hashlib.sha256(''.join(source.splitlines(keepends=True)[start - 1:node.end_lineno]).encode()).hexdigest()
    for item in record.get('source_dependencies', []):
        item['sha256'] = digest(REPO / item['path'])


def replay_record(case, output):
    """Relocate artifacts/runtime state and identify current source in a private copy.

    Archived runners assert source hashes. Refresh only the replay copy so a code
    change is measured, rather than rejected before compilation. Baseline timing,
    precision, sampling and resolved launch options remain unchanged.
    """
    record = deepcopy(case.record)
    for name in ('assembly', 'single_tg_perf'):
        if 'path' in record.get(name, {}):
            record[name]['path'] = str((case.record_path.parent / record[name]['path']).resolve())
    refresh_sources(record)
    runtime = output / 'runtime'
    for name in ('home', 'temp', 'cache', 'triton-home'):
        (runtime / name).mkdir(parents=True)
    record['environment']['environment_variables'].update(
        HOME=str(runtime / 'home'), TMPDIR=str(runtime / 'temp'),
        TRITON_CACHE_DIR=str(runtime / 'cache'), TRITON_HOME=str(runtime / 'triton-home'),
        GPU_OUT='m5out' if (timing_protocol(record) in ('validated_warm_median', 'variant_block_median')
            or case.record['source']['symbol'] == '_gemm_a16w16_bf16_tn_bm16_bn16_bk128_kernel') else 'g',
        RPY_HISTORY_LOG=str(runtime / 'history.log'), RPY_HISTORY_SIZE='1024',
        BOLTOPS_GEMM_CONFIG_DIR=None, PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE=None)
    return record


def run_case(case, case_output, threshold=1.0):
    output = Path(case_output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = replay_record(case, output)
    script = (case.record_path.parent / record['benchmark'].get('script', record['benchmark'].get('path'))).resolve()
    benchmark_hash = digest(script)
    if record['benchmark'].get('sha256', benchmark_hash) != benchmark_hash:
        raise ValueError('archived benchmark changed; retain its recorded measurement protocol')
    replay = output / 'replay-record.json'
    save(replay, record)
    if record['benchmark'].get('interface') == 'variant_blocks':
        runner = output / 'runner'
        runner.mkdir()
        shutil.copyfile(script, runner / script.name)
        source = REPO / record['source']['path']
        shutil.copyfile(source, runner / source.name)
        inputs = runner / record['layout'].lower()
        inputs.mkdir()
        save(inputs / 'record.json', record)
        target = runner / (record['layout'] + '-standalone')
        (output / 'measurement').symlink_to(target, target_is_directory=True)
        command = [sys.executable, str(runner / script.name), record['layout'], str(REPO)]
    elif record['benchmark'].get('interface') == 'paired_routes':
        command = [sys.executable, str(script), '--repo', str(REPO), '--record', str(replay),
                   '--output', str(output / 'measurement'), '--rounds', str(record['benchmark']['repeats'])]
    else:
        command = [sys.executable, str(script), '--record', str(replay), '--output', str(output / 'measurement')]
    environment = dict(os.environ)
    # Each archived runner re-execs with the recorded environment itself. Remove
    # config overrides and Python optimization also from its initial invocation.
    environment.pop('BOLTOPS_GEMM_CONFIG_DIR', None)
    environment.pop('PYTHONOPTIMIZE', None)
    marker = record.get('regression', {}).get('environment_marker')
    if marker:
        environment.pop(marker, None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    save(output / 'invocation.json', dict(command=command, benchmark_sha256=benchmark_hash,
        baseline_record=str(case.record_path), baseline_record_sha256=digest(case.record_path),
        current_sha256={name: record[name]['sha256'] for name in ('source', 'config', 'public_api')
                        if 'sha256' in record.get(name, {})},
        cache_isolated=True))
    with (output / 'benchmark.log').open('w') as log:
        child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=output, env=environment)
    if child.returncode:
        report = dict(case=case.key, dtype=case.dtype, status='ERROR', exit_code=child.returncode,
                      log=str(output / 'benchmark.log'))
    else:
        report = assess(case, output, threshold)
    save(output / 'regression.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', default=[], help='requirement, DIE/layout key or full dtype key; repeatable')
    parser.add_argument('--dtype', action='append', choices=('fp16', 'bf16'), default=[], help='recorded precision; repeatable')
    parser.add_argument('--record', action='append', type=Path, help='explicit baseline instead of document discovery; repeatable')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=int, default=1, help='independent core-only PMD processes (default: 1)')
    parser.add_argument('--max-regression-percent', type=float, default=1.0)
    parser.add_argument('--evaluate', action='store_true', help='validate saved measurements; launch no kernels')
    parser.add_argument('--list', action='store_true', help='list recorded cases without Torch/Triton imports')
    args = parser.parse_args(argv)
    if args.jobs < 1 or not math.isfinite(args.max_regression_percent) or args.max_regression_percent < 0:
        parser.error('jobs must be positive; regression threshold must be finite and nonnegative')
    try:
        cases = discover_cases(records=args.record)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    selectors = {value.lower() for value in args.case}
    def aliases(case):
        return {case.key.lower(), case.legacy_key.lower(), case.record['requirement'].lower()}
    known = set().union(*(aliases(case) for case in cases))
    if selectors - known:
        parser.error('unknown recorded cases: ' + ', '.join(sorted(selectors - known)))
    cases = [case for case in cases if (not selectors or aliases(case) & selectors)
             and (not args.dtype or case.dtype in args.dtype)]
    missing = set(args.dtype) - {case.dtype for case in cases}
    if not cases or missing:
        parser.error('no matching baseline for requested precision/case; FP16 and BF16 need separate records (--record)')
    if args.list:
        print(json.dumps([dict(case=c.key, dtype=c.dtype, shape=c.record['shape'],
            baseline_kernel_us=c.record['performance']['kernel_us'], aggregation=timing_protocol(c.record))
            for c in cases], indent=2))
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
    state = dict(operator='gemm_a16w16', max_regression_percent=args.max_regression_percent,
                 status='RUNNING', cases={c.key: dict(case=c.key, status='QUEUED') for c in cases})
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
            report = dict(case=case.key, dtype=case.dtype, status='ERROR', error=str(error))
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / 'regression.json', report)
        update(case.key, report)
        print(json.dumps(report), flush=True)
        return report
    save(output / 'report.json', state)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        reports = list(pool.map(work, cases))
    state['status'] = 'PASS' if all(row['status'] == 'PASS' for row in reports) else 'FAIL'
    save(output / 'report.json', state)
    return 0 if state['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
