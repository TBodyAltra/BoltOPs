# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""CPU-only tests for evidence validation and performance regression decisions."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('qwen35_pmd_report', ROOT / 'scripts/performance/qwen35_fp8_pmd_regression.py')
reg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reg
SPEC.loader.exec_module(reg)


def evidence(tmp_path, cycles=(100000,), protocol='single'):
    record = dict(requirement='R009', shape=[128, 256, 128], layout='TN',
                  environment=dict(type='PMD', dies=1, cu_count=48, arch='gfx946', clock_period_ps={'gc':455}),
                  performance=dict(kernel_us=45.5, launched_TGs=2), benchmark=dict(repeats=len(cycles)))
    if protocol == 'median':
        record['performance']['aggregation'] = 'median of repeats 1..4; repeat 0 excluded'
    path = tmp_path / 'record.json'
    path.write_text(json.dumps(record))
    root = tmp_path / 'run'
    model = root / 'measurement/m5out'
    model.mkdir(parents=True)
    nodes = [dict(path='system.shaders', CUs=[dict(path=f'system.shaders.CUs{i}') for i in range(48)]),
             dict(path='system.shaders.clk_domain', clock=[455]),
             dict(path='gpu_configuration', device_id=0x66b7, ipc_conn=False, skip_cal=False, skip_kernel_exec=False)]
    (model / 'config.json').write_text(json.dumps(nodes))
    samples = []
    for i, value in enumerate(cycles):
        stats = model / '0' / str(i) / 'stats.txt'
        stats.parent.mkdir(parents=True)
        stats.write_text(f'simFreq 1000000000000\nsystem.gcSimCycles {value}\nsystem.shaders.clk_domain.clock 455\n'
                         'system.shaders.CUs0.completedWGs 1\nsystem.shaders.CUs1.completedWGs 1\n')
        samples.append(dict(kernel_us=value*.000455, stats_path=str(stats.relative_to(root/'measurement')), completed_TGs=2,
                            accuracy=dict(elements=128*256, outside_tolerance=0, nonfinite=0, canary_ok=True)))
    result = dict(status='PASS', shape=record['shape'], launches=samples)
    if protocol == 'median':
        result = dict(status='PASS', samples=samples, validation=[samples[0]])
    (root/'measurement/result.json').write_text(json.dumps(result))
    return reg.Case(path, record), root, result


def change(root, result):
    (root/'measurement/result.json').write_text(json.dumps(result))


@pytest.mark.parametrize('cycles,status', [(99000,'PASS'),(100500,'PASS'),(101000,'PASS'),(101001,'REGRESSION')])
def test_default_one_percent_boundary(tmp_path, cycles, status):
    case, root, _ = evidence(tmp_path, [cycles])
    assert reg.evaluate(case, root)['status'] == status


def test_threshold_is_configurable(tmp_path):
    case, root, _ = evidence(tmp_path, [103000])
    assert reg.evaluate(case, root, 2)['status'] == 'REGRESSION'
    assert reg.evaluate(case, root, 3)['status'] == 'PASS'


def test_warm_median_excludes_cold_launch(tmp_path):
    case, root, _ = evidence(tmp_path, [300000,99000,100000,100000,150000], 'median')
    assert reg.evaluate(case, root)['kernel_us'] == pytest.approx(45.5)


@pytest.mark.parametrize('mutation', ['accuracy','nonfinite','guards','elements','shape','TG','clock','missing','partial','time','frequency','escaped_stats'])
def test_invalid_evidence_cannot_pass(tmp_path, mutation):
    case, root, data = evidence(tmp_path)
    sample = data['launches'][0]
    if mutation == 'accuracy': sample['accuracy']['outside_tolerance'] = 1
    if mutation == 'nonfinite': sample['accuracy']['nonfinite'] = 1
    if mutation == 'guards': sample['accuracy']['canary_ok'] = False
    if mutation == 'elements': sample['accuracy']['elements'] = 32767
    if mutation == 'shape': data['shape'] = [128,256,64]
    if mutation == 'TG': sample['completed_TGs'] = 1
    if mutation == 'clock':
        p=root/'measurement/m5out/config.json';p.write_text(p.read_text().replace('[455]','[500]'))
    if mutation == 'missing': data['launches'] = []
    if mutation == 'partial': data['single_tg'] = True
    if mutation == 'time': sample['kernel_us'] = .01
    if mutation == 'frequency':
        p=root/'measurement'/sample['stats_path'];p.write_text(p.read_text().replace('1000000000000','2000000000000'))
    if mutation == 'escaped_stats': sample['stats_path'] = '../../outside.txt'
    change(root, data)
    assert reg.assess(case, root, 1)['status'] == 'INVALID'


def test_legacy_r005_warm_mean_and_guards(tmp_path):
    case, root, data = evidence(tmp_path, [300000,99000,100000,101000])
    data['job'] = {'shape': case.record['shape']}
    data['grid'] = [1,2]
    data['public_auto_exact_match'] = True
    data['measurements'] = [dict(warmup=i==0, stats=dict(cycles=cycles,clock_ps=455,completed_TGs=2,path=sample['stats_path']),
                                accuracy=dict(mismatches=0,nonfinite=0,guards_ok=True,max_abs=.1))
                            for i,(cycles,sample) in enumerate(zip([300000,99000,100000,101000],data.pop('launches')))]
    change(root, data)
    assert reg.evaluate(case, root)['kernel_us'] == pytest.approx(45.5)
    data['measurements'][2]['accuracy']['guards_ok'] = False
    change(root,data)
    assert reg.assess(case, root, 1)['status']=='INVALID'


def test_failed_child_is_not_a_performance_pass(tmp_path, monkeypatch):
    case, _, _ = evidence(tmp_path)
    script = case.record_path.parent/'benchmark.py';script.write_text('')
    case.record.update(benchmark=dict(script='benchmark.py',command='python benchmark.py',repeats=1))
    case.record['environment']['environment_variables'] = {}
    monkeypatch.setattr(reg.subprocess,'run',lambda *args,**kwargs: type('Child',(),{'returncode':7})())
    report=reg.run_case(case,tmp_path/'new-run')
    assert report['status']=='ERROR' and report['exit_code']==7


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan'), True])
def test_invalid_baseline_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        reg.positive(value,'baseline')


def test_discovery_keeps_die_and_layout_records_separate(tmp_path):
    document=tmp_path/reg.DOCUMENT;document.parent.mkdir(parents=True)
    links=[]
    for die,layout in [(1,'tn'),(1,'tt'),(2,'tn')]:
        record=dict(requirement='R009',shape=[128,256,128],layout=layout.upper(),
                    environment=dict(type='PMD',dies=die),performance=dict(kernel_us=1,launched_TGs=2))
        path=document.parent/f'tuning/r009/{die}die/{layout}/record.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(record))
        links.append(f'<a href="{path.relative_to(document.parent)}">record</a>')
    document.write_text('### '+reg.MODEL+'\n'+'\n'.join(links)+'\n### Other model\n<a href="missing/record.json">ignore</a>')
    assert {c.key for c in reg.discover_cases(tmp_path)}=={'R009/1die/tn','R009/1die/tt','R009/2die/tn'}
