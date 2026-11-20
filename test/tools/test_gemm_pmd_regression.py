# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU checks for baseline discovery, replay preparation and PMD evidence gates."""
import configparser
import hashlib
import json
import io
from pathlib import Path
from runpy import run_path
import statistics
import subprocess
import sys
import tempfile
import tarfile
import shutil
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from unittest.mock import patch

from scripts.performance import gemm_a16w16_pmd_regression as linear
from scripts.performance import qwen35_fp8_pmd_regression as fp8


class GemmRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def case(self, key='R001/1die/tn'):
        return next(c for c in fp8.discover_cases() if c.key == key)

    def evidence(self, case, times, validation=(), legacy_mean=False):
        record = case.record
        measurement = self.root / 'measurement'
        measurement.mkdir(exist_ok=True)
        environment = record['environment']
        clock = environment['clock_period_ps']['gc']
        cu_count, dies = environment['cu_count'], environment['dies']
        model = configparser.RawConfigParser()
        model.optionxform = str
        sections = deepcopy(environment.get('runtime_configuration', {}).get('sections', {}))
        gpu = dict(sections.get('gpu_configuration', {}))
        gpu.update(environment.get('gpu_configuration', {}))
        gpu.update(device_id=0x66b7, ipc_conn=False, skip_cal=False, skip_kernel_exec=False)
        sections['gpu_configuration'] = gpu
        for section, fields in sections.items():
            model[section] = {key: str(value).lower() for key, value in fields.items()}
        cu_paths = []
        for die in range(dies):
            shader = 'system.shaders' + (str(die) if dies > 1 else '')
            model[shader] = {}
            model[shader + '.clk_domain'] = dict(clock=str(clock))
            for cu in range(cu_count // dies):
                cu_paths.append(f'{shader}.CUs{cu}')
                model[cu_paths[-1]] = {}
        (measurement / 'm5out').mkdir(exist_ok=True)
        with (measurement / 'm5out/config.ini').open('w') as stream:
            model.write(stream)
        tgs = record['performance']['launched_TGs']
        samples = []
        for index, us in enumerate(list(validation) + list(times)):
            path = measurement / f'm5out/0/{index}/stats.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            cycles = us / (clock / 1e6)
            raw = [f'system.gcSimCycles {cycles}', 'simFreq 1000000000000']
            raw += [f'{name}.completedWGs {tgs // cu_count + (i < tgs % cu_count)}'
                    for i, name in enumerate(cu_paths)]
            raw.append('---------- End Simulation Statistics ----------')
            path.write_text('\n'.join(raw))
            accuracy = dict(elements=record['shape'][0]*record['shape'][1],
                            mismatches=0, nonfinite=0, max_abs=0., canary_ok=True)
            reference = record.get('validation', {})
            if isinstance(reference, dict):
                accuracy.update({k: reference[k] for k in ('atol', 'rtol', 'inputs_byte_identical') if k in reference})
            samples.append(dict(kernel_us=us, completed_TGs=tgs, accuracy=accuracy,
                                stats_path=str(path.relative_to(measurement))))
        result = dict(status='PASS', single_tg=False, requirement=record['requirement'],
                      performance=dict(kernel_us=statistics.median(times)),
                      public_api_verified=True, benchmark_sha256=record['benchmark'].get('sha256'))
        if legacy_mean:
            result['job'] = dict(shape=record['shape'])
            result['grid'] = [tgs]
            result['public_auto_exact_match'] = True
            result['measurements'] = [dict(warmup=i == 0,
                accuracy=dict(mismatches=0, nonfinite=0, guards_ok=True),
                stats=dict(cycles=s['kernel_us']/(clock/1e6), clock_ps=clock,
                           path=s['stats_path'], completed_TGs=tgs)) for i, s in enumerate(samples)]
            result['performance']['kernel_us'] = statistics.mean(times[1:])
        elif record['benchmark'].get('timing_mode') == 'validated_warm_median':
            result.update(samples=samples[len(validation):], validation=samples[:len(validation)])
        elif record['benchmark'].get('timing_mode') in ('legacy_median', 'existing_first_timed'):
            checked, timed = samples[:len(validation)], samples[len(validation):]
            result['timing_input'] = dict(seed=record['benchmark']['seed'], input_kind='normal_random')
            for index, sample in enumerate(checked):
                sample.update(seed=record['benchmark']['seed'] + 1000*index,
                              phase='correctness', input_kind='ones' if index == 1 else 'uniform_random')
            for index, sample in enumerate(timed):
                sample.update(index=index, phase='warmup' if index == 0 else 'timed')
            for sample in samples:
                sample['accuracy'].update(atol=.02, rtol=.01, mean_abs_vs_fp32=0., rounding_floor=0.)
            result.update(samples=timed, validation=checked)
        else:
            result['launches'] = samples
        path = measurement / 'result.json'
        path.write_text(json.dumps(result))
        return path

    def test_paired_baseline_retains_published_route_and_sample_median(self):
        path = linear.REPO / 'docs/ops/gemm/tuning/r003/2die/tn/record.json'
        if not path.is_file():
            self.skipTest('paired-route baseline is supplied by the BF16 MR')
        original = linear.load(path)
        measurement = linear.load(path.parent / original['regression']['measurement'])
        chosen = measurement['variants']['tuned_auto']
        normalized = linear.normalize_record(original, path)
        self.assertNotIn('performance', original)
        self.assertEqual(normalized['regression']['config'], chosen['config'])
        self.assertEqual(normalized['performance']['kernel_us'],
                         statistics.median(s['kernel_us'] for s in chosen['samples']))
        self.assertEqual(normalized['benchmark']['repeats'], 4)
        self.assertEqual(normalized['benchmark']['validation_count'], 3)
        local = self.root / 'measurement.json'
        local.write_text(json.dumps(measurement))
        with self.assertRaisesRegex(ValueError, 'measurement changed'):
            linear.normalize_record(original, self.root / 'record.json')

    def test_compiler_comparison_cannot_replace_production_gate(self):
        case = self.case('R009/1die/tn')
        document = self.root / fp8.DOCUMENT
        document.parent.mkdir(parents=True)
        for label in ('production', 'comparison'):
            path = document.parent / label / 'record.json'
            path.parent.mkdir()
            record = deepcopy(case.record)
            if label == 'comparison':
                record['regression'] = dict(role='compiler_comparison')
                record['performance']['kernel_us'] *= 1.028
            path.write_text(json.dumps(record))
        document.write_text('### ' + fp8.MODEL + '\n[production](production/record.json)\n'
                            '[comparison](comparison/record.json)\n')
        cases = fp8.discover_cases(self.root)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].record['performance']['kernel_us'], case.record['performance']['kernel_us'])

    def test_public_job_gate_checks_seed_sampling_and_regression(self):
        cases = {c.key: c for c in fp8.discover_cases()}
        if 'R005/2die/tn' not in cases:
            self.skipTest('public-job evidence is supplied by the FP8 MR')
        case = cases['R005/2die/tn']
        count = case.record['benchmark']['repeats']
        baseline = case.record['performance']['kernel_us']
        for factor in (1., 1.02):
            with self.subTest(factor=factor):
                path = self.evidence(case, [baseline * factor] * count)
                result = json.loads(path.read_text())
                result.update(job=case.record['benchmark']['job'],
                              public_config=case.record['config']['launch_options'],
                              public_kernel=case.record['config']['launch_options']['kernel'])
                result['measurements'] = [dict(cold=i == 0, us=sample['kernel_us'],
                    accuracy=dict(sample['accuracy'], guards_ok=True),
                    stages=[dict(path=sample['stats_path'], completed_TGs=sample['completed_TGs'])])
                    for i, sample in enumerate(result.pop('launches'))]
                path.write_text(json.dumps(result))
                self.assertEqual(fp8.evaluate(case, self.root)['status'],
                                 'PASS' if factor == 1 else 'REGRESSION')
                for mutation in ('seed', 'cold', 'config', 'kernel', 'stats'):
                    bad = deepcopy(result)
                    if mutation == 'seed':
                        bad['job']['seed'] += 1
                    elif mutation == 'cold':
                        bad['measurements'][1]['cold'] = True
                    elif mutation == 'config':
                        bad['public_config']['BLOCK_SIZE_N'] *= 2
                    elif mutation == 'kernel':
                        bad['public_kernel'] = 'unexpected_kernel'
                    else:
                        bad['measurements'][1]['stages'] = bad['measurements'][0]['stages']
                    path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        fp8.evaluate(case, self.root)

    def test_all_current_cases_are_discovered_without_modifying_records(self):
        paths = list(linear.REPO.glob('docs/ops/gemm/tuning/**/record.json'))
        hashes = {p: linear.digest(p) for p in paths}
        model = fp8.discover_cases()
        bf16 = linear.discover_cases()
        has = lambda r: (linear.REPO / f'docs/ops/gemm/tuning/{r}/record.json').is_file()
        self.assertEqual(sum(not linear.is_linear_record(c.record) for c in model),
                         16 + has('r005/2die/tn'))
        self.assertTrue({'R004/2die/tn', 'R004/2die/tt'}.issubset(
            c.key for c in model if linear.is_linear_record(c.record)))
        self.assertEqual(len(bf16), 25 + has('r001/2die/tn') + has('r003/2die/tn'))
        self.assertTrue(all(c.dtype == 'bf16' for c in bf16))
        self.assertEqual({c.key for c in model if c.record.get('public_api', {}).get('path') == linear.OPERATOR},
                         {'R001/1die/tn', 'R001/1die/tt', 'R002/1die/tt',
                          'R002/2die/tt', 'R003/1die/tt', 'R004/1die/tt'}
                         | ({'R001/2die/tn'} if has('r001/2die/tn') else set()))
        self.assertEqual(hashes, {p: linear.digest(p) for p in paths})

    def test_experimental_candidates_are_not_production_baselines(self):
        source = self.case('R005/1die/tn').record_path
        document = self.root / fp8.DOCUMENT
        document.parent.mkdir(parents=True)
        regular = document.parent / 'cases/production/record.json'
        experimental = document.parent / 'cases/experimental/record.json'
        regular.parent.mkdir(parents=True);experimental.parent.mkdir(parents=True)
        regular.write_bytes(source.read_bytes())
        experimental.write_text(json.dumps(dict(requirement='R999', kernel_symbol='candidate_kernel',
            implementation='experimental direct Gluon ring; public defaults unchanged')))
        document.write_text('### ' + fp8.MODEL + '\n[production](cases/production/record.json)\n'
                            '[experimental](cases/experimental/record.json)\n')
        self.assertEqual([c.key for c in fp8.discover_cases(self.root)], ['R005/1die/tn'])

    def test_every_replay_prepares_current_source_and_unchanged_protocol(self):
        with patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
            for suite, cases in [(fp8, fp8.discover_cases()), (linear, linear.discover_cases())]:
                for case in cases:
                    with self.subTest(suite=suite.__name__, case=case.key):
                        folder = self.root / suite.__name__ / case.directory
                        folder.parent.mkdir(parents=True, exist_ok=True)
                        before = linear.digest(case.record_path)
                        if case.record['benchmark'].get('interface') == 'job_json':
                            runtime = self.root / 'test-runtime'
                            library = runtime / 'triton/_C/libtriton.so'
                            library.parent.mkdir(parents=True, exist_ok=True)
                            library.write_bytes(b'test toolchain')
                            case.record['environment']['environment_variables']['PYTHONPATH'] = str(runtime)
                            case.record['environment']['triton_library_sha256'] = linear.digest(library)
                        result = suite.run_case(case, folder)
                        self.assertEqual(result['status'], 'ERROR')  # Mocked child, no GPU execution.
                        replay = linear.load(folder / 'replay-record.json')
                        self.assertEqual(replay['performance'], case.record['performance'])
                        self.assertEqual(replay['benchmark'], case.record['benchmark'])
                        self.assertEqual(replay['source']['sha256'], linear.digest(linear.REPO / replay['source']['path']))
                        if case.record['benchmark'].get('interface') == 'variant_blocks':
                            invocation = linear.load(folder / 'invocation.json')
                            self.assertEqual(invocation['command'][-2:], [case.record['layout'], str(linear.REPO)])
                            self.assertTrue((folder / 'measurement').is_symlink())
                            self.assertEqual((folder / 'runner/kernel.py').read_bytes(),
                                             (linear.REPO / replay['source']['path']).read_bytes())
                            staged = linear.load(folder / 'runner' / case.record['layout'].lower() / 'record.json')
                            self.assertEqual(staged, replay)
                        self.assertEqual(before, linear.digest(case.record_path))

    def test_all_validated_warm_samples_contribute_to_median(self):
        case = self.case('R002/2die/tt')
        case.record['benchmark'].update(repeats=2, validation_count=1)
        case.record['performance']['kernel_us'] = 5.
        self.evidence(case, [1., 9.], validation=[99.])
        for suite, current in [(fp8, case), (linear, linear.Case(case.record_path, case.record))]:
            report = suite.evaluate(current, self.root)
            self.assertEqual(report['kernel_us'], 5.)
            self.assertEqual(report['selected_indexes'], [0, 1])
            self.assertEqual(report['status'], 'PASS')

    def test_one_percent_is_inclusive_and_larger_regression_fails(self):
        for time, expected in [(5.05, 'PASS'), (5.0501, 'REGRESSION')]:
            with self.subTest(time=time):
                case = self.case()
                case.record['benchmark'].update(repeats=2, validation_count=1)
                case.record['performance']['kernel_us'] = 5.
                self.evidence(case, [time, time], validation=[8.])
                self.assertEqual(fp8.evaluate(case, self.root)['status'], expected)

    def test_bad_evidence_cannot_pass(self):
        changes = [lambda r: r.update(single_tg=True),
                   lambda r: r.update(public_api_verified=False),
                   lambda r: r.update(benchmark_sha256='wrong'),
                   lambda r: r['validation'].clear(),
                   lambda r: r['samples'][0]['accuracy'].update(mismatches=1),
                   lambda r: r['samples'][0]['accuracy'].update(guards_ok=False),
                   lambda r: r['samples'][0].update(stats_path='../outside/stats.txt'),
                   lambda r: r['samples'][1].update(stats_path=r['samples'][0]['stats_path'])]
        for change in changes:
            with self.subTest(change=change):
                case = self.case()
                case.record['benchmark'].update(repeats=2, validation_count=1)
                case.record['performance']['kernel_us'] = 5.
                path = self.evidence(case, [5., 5.], validation=[8.])
                result = json.loads(path.read_text());change(result);path.write_text(json.dumps(result))
                self.assertEqual(fp8.assess(case, self.root, 1.)['status'], 'INVALID')

    def test_fp8_legacy_mean_keeps_cold_launch_exclusion(self):
        case = self.case('R005/1die/tn')
        case.record['performance']['kernel_us'] = 5.
        self.evidence(case, [100., 3., 5., 7.], legacy_mean=True)
        self.assertEqual(fp8.evaluate(case, self.root)['kernel_us'], 5.)

    def test_fp8_cold_record_remains_one_complete_dispatch(self):
        case = self.case('R006/1die/tn')
        case.record['performance']['kernel_us'] = 5.
        self.evidence(case, [5.])
        self.assertEqual(fp8.evaluate(case, self.root)['kernel_us'], 5.)

    def test_bf16_cold_record_keeps_first_dispatch(self):
        case = next(c for c in linear.discover_cases() if c.record['requirement'] == 'R020')
        case.record['performance']['kernel_us'] = 5.
        self.evidence(case, [5.])
        self.assertEqual(linear.evaluate(case, self.root)['kernel_us'], 5.)

    def test_existing_bf16_sampling_protocols_are_preserved(self):
        for mode, expected in [('existing_first_timed', 2.), ('legacy_median', 5.)]:
            with self.subTest(mode=mode):
                model = self.case()
                case = linear.Case(model.record_path, model.record)
                case.record['benchmark'].update(timing_mode=mode, repeats=3)
                case.record['performance']['kernel_us'] = expected
                path = self.evidence(case, [100., 2., 8.], validation=[11., 12., 13.])
                result = json.loads(path.read_text());result['performance']['kernel_us'] = expected
                path.write_text(json.dumps(result))
                self.assertAlmostEqual(linear.evaluate(case, self.root)['kernel_us'], expected)

    def test_archived_pmd_measurement_passes_both_suites(self):
        cases = {c.key: c for c in fp8.discover_cases()}
        if 'R001/2die/tn' not in cases:
            self.skipTest('dual-die TN archived evidence is supplied by the BF16 MR')
        case = cases['R001/2die/tn']
        record = case.record
        evidence = record['measurement']
        measurement = self.root / 'measurement';measurement.mkdir()
        shutil.copyfile(case.record_path.parent / evidence['path'], measurement / 'result.json')
        with tarfile.open(case.record_path.parent / evidence['stats_archive']) as archive:
            for member in archive.getmembers():
                (measurement / member.name).resolve().relative_to(measurement.resolve())
            archive.extractall(measurement)
        for suite, current in [(fp8, case), (linear, linear.Case(case.record_path, record))]:
            with self.subTest(suite=suite.__name__):
                result = suite.evaluate(current, self.root)
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(result['kernel_us'], record['performance']['kernel_us'])
                output = self.root / ('cli-fp8' if suite is fp8 else 'cli-bf16')
                case_folder = output / current.directory
                case_folder.mkdir(parents=True)
                (case_folder / 'measurement').symlink_to(measurement, target_is_directory=True)
                command = [sys.executable, suite.__file__, '--evaluate', '--case', case.key,
                           '--output', str(output)]
                if suite is linear:
                    command.extend(['--dtype', 'bf16'])
                child = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
                self.assertEqual(linear.load(output / 'report.json')['status'], 'PASS')

    def test_selected_tt_replay_rejects_binary_or_resource_changes(self):
        tuning = linear.REPO / 'docs/ops/gemm/tuning'
        runner = tuning / 'r024/2die/tt/benchmark.py'
        verify_assembly = run_path(str(runner))['verify_assembly']
        for req, roles in [('r024', ('main', 'reduce')), ('r028', ('main',))]:
            record = linear.load(tuning / req / '2die/tt/record.json')
            for role in roles:
                actual = deepcopy(record['assembly' if role == 'main' else 'reduce_assembly'])
                verify_assembly(role, actual, record)
                for key in ('executable_sections', 'n_regs', 'n_spills', 'compiler_shared_bytes', 'num_warps'):
                    with self.subTest(requirement=req, role=role, field=key):
                        with self.assertRaisesRegex(ValueError, key):
                            verify_assembly(role, dict(actual, **{key: None}), record)

    def test_invalid_raw_stats_cannot_pass(self):
        changes = [lambda text: text.replace('End Simulation Statistics', 'incomplete'),
                   lambda text: text.replace('completedWGs 1', 'completedWGs 0', 1),
                   lambda text: text + '\nsystem.gcSimCycles 12345\n']
        for change in changes:
            with self.subTest(change=change):
                case = self.case()
                case.record['benchmark'].update(repeats=2, validation_count=1)
                case.record['performance']['kernel_us'] = 5.
                path = self.evidence(case, [5., 5.], validation=[8.])
                result = json.loads(path.read_text());stats = path.parent / result['samples'][0]['stats_path']
                stats.write_text(change(stats.read_text()))
                self.assertEqual(fp8.assess(case, self.root, 1.)['status'], 'INVALID')

    def variant_evidence(self, case, factor=1.):
        root = self.root
        measurement = root / 'measurement';measurement.mkdir(exist_ok=True)
        target = case.record['regression']['variant']
        other = case.record['interleaved_comparison']['best']
        variants = {}
        configurations = [('baseline', case.record['baseline']['config']),
                          (target, case.record['config']),
                          (other, case.record['interleaved_comparison']['config'])]
        for group, (name, cfg) in enumerate(configurations):
            current = deepcopy(case.record)
            m, n, _ = current['shape']
            grid = m // cfg['BLOCK_SIZE_M'] * (n // cfg['BLOCK_SIZE_N'])
            current['performance']['launched_TGs'] = grid
            us = case.record['performance']['kernel_us'] * factor if name == target else 1.
            stage = root / ('stage-' + str(group));stage.mkdir(exist_ok=True)
            self.root = stage
            try:
                path = self.evidence(linear.Case(case.record_path, current), [us]*9, validation=[us])
            finally:
                self.root = root
            result = json.loads(path.read_text())
            if group == 0:
                (measurement / 'm5out').mkdir(exist_ok=True)
                shutil.copyfile(path.parent / 'm5out/config.ini', measurement / 'm5out/config.ini')
            samples = result['launches']
            for index, sample in enumerate(samples):
                number = group if index == 0 else 3 + group*9 + index - 1
                relative = f'm5out/0/{number}/stats.txt'
                destination = measurement / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path.parent / sample['stats_path'], destination)
                sample['stats_path'] = relative
            variants[name] = dict(status='PASS', family='candidate', config=cfg, grid=grid,
                resources=dict(SGPR_spills=0, VGPR_spills=0),
                confirmation=dict(validation=samples[:1], samples=samples[1:]),
                performance=dict(kernel_us=us))
        path = measurement / 'result.json'
        path.write_text(json.dumps(dict(status='COMPLETE', stage='complete',
            shape=case.record['shape'], layout=case.record['layout'], best=other, variants=variants)))
        return path

    def test_r004_both_layouts_are_enrolled_without_changing_baselines(self):
        cases = {c.key: c for c in linear.discover_cases()}
        for layout, baseline in [('tn', 14.208740), ('tt', 14.625520)]:
            case = cases[f'R004/bf16/2die/{layout}']
            self.assertEqual(case.record['performance']['kernel_us'], baseline)
            self.assertEqual(case.record['performance']['launched_TGs'], 52)
            self.assertEqual(linear.timing_protocol(case.record), 'variant_block_median')
            self.assertEqual(case.record['benchmark']['repeats'], 9)

    def test_r004_gate_uses_fixed_variant_and_all_nine_samples(self):
        for layout in ['tn', 'tt']:
            with self.subTest(layout=layout):
                root = self.root / layout;root.mkdir();original = self.root;self.root = root
                try:
                    case = self.case(f'R004/2die/{layout}')
                    path = self.variant_evidence(case)
                    result = json.loads(path.read_text())
                    for suite, current in [(fp8, case), (linear, linear.Case(case.record_path, case.record))]:
                        report = suite.evaluate(current, root)
                        self.assertEqual(report['status'], 'PASS')
                        self.assertEqual(report['selected_indexes'], list(range(9)))
                        self.assertAlmostEqual(report['kernel_us'], case.record['performance']['kernel_us'])
                        output = root / ('cli-fp8' if suite is fp8 else 'cli-bf16')
                        folder = output / current.directory
                        folder.mkdir(parents=True)
                        (folder / 'measurement').symlink_to(root / 'measurement', target_is_directory=True)
                        command = [sys.executable, suite.__file__, '--evaluate', '--case', f'R004/2die/{layout}',
                                   '--output', str(output)]
                        if suite is linear:
                            command.extend(['--dtype', 'bf16'])
                        child = subprocess.run(command, capture_output=True, text=True)
                        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
                        self.assertEqual(linear.load(output / 'report.json')['status'], 'PASS')
                    self.assertNotEqual(result['best'], case.record['regression']['variant'])
                finally:
                    self.root = original

    def test_r004_slow_fixed_variant_cannot_hide_behind_faster_candidate(self):
        case = self.case('R004/2die/tn')
        self.variant_evidence(case, factor=1.02)
        self.assertEqual(fp8.evaluate(case, self.root)['status'], 'REGRESSION')

    def test_r004_invalid_blocks_cannot_pass(self):
        changes = [lambda r, v: v['confirmation']['samples'].pop(0),
                   lambda r, v: v['config'].update(CHUNKS=2),
                   lambda r, v: v['resources'].update(VGPR_spills=1),
                   lambda r, v: r.update(layout='TT'),
                   lambda r, v: v['confirmation']['samples'][0]['accuracy'].update(mismatches=1),
                   lambda r, v: v['confirmation']['samples'][0].update(
                       stats_path=r['variants']['baseline']['confirmation']['samples'][0]['stats_path'])]
        case = self.case('R004/2die/tn')
        for change in changes:
            with self.subTest(change=change):
                path = self.variant_evidence(case)
                result = json.loads(path.read_text())
                change(result, result['variants'][case.record['regression']['variant']])
                path.write_text(json.dumps(result))
                self.assertEqual(fp8.assess(case, self.root, 1.)['status'], 'INVALID')

    def test_requested_fp16_cannot_silently_use_bf16(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            linear.main(['--list', '--dtype', 'fp16'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
