"""Close original short gate without turning rejected fast allocation into a pass."""
import pathlib,json,hashlib,re,copy,sys,collections
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));import scan
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def code(p):
    lines=[]
    for l in p.read_text().splitlines():
        if re.match(r'\s*\.Lfunc_end\d+:',l):break
        l=l.split(';',1)[0].strip()
        if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(l)
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
e=load(R/'job-exit.json');assert e['returncode']==1 and not pathlib.Path('/proc',str(e['pid'])).exists()
assert not (R/'mixed-short-gate-audit.json').exists()
act=load(R/'active-manifest.json');m=load(R/('manifest-'+act['manifest_id']+'.json'));assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
controls=load(R/'default_controls-qualification.json');pilot=load(R/'liveness_pilot-qualification.json');mixed=load(R/'liveness_features_short-qualification.json');assert controls['passed'] and pilot['passed'] and not mixed['passed']
refs={x['id']:dict(x,control=True) for x in controls['checks']};assert len(refs)==2
for x in mixed['checks']:assert x['id'] not in refs;refs[x['id']]=dict(x,control=False)
assert len(refs)==8 and {x['id'] for x in pilot['checks']}<={x['id'] for x in mixed['checks']}
rows=[];intervals=[];pids=[]
for p in sorted((R/'runs').glob('*/result.json')):
    x=load(p);ref=refs[x['id']];cr=load(pathlib.Path(ref['CPU_reference'])/'result.json')
    assert x['measurement_context_id']==act['measurement_context_id'] and x['status'] in ['PASS','ACCURACY_FAIL'] and x['grid']==[1] and x['shape']==[256,128,2048] and x['pipeline_loops']==16
    assert x['input_seed']==29 and x['input_scale']==1 and x['accuracy']['elements']==32768 and x['accuracy']['nonfinite']==0
    assert x['loaded_native']['sha256']=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990' and x['asm_sha256']==sha(p.parent/'kernel.amdgcn') and code(p.parent/'kernel.amdgcn')==code(pathlib.Path(ref['CPU_reference'])/'kernel.amdgcn')
    assert x['metadata']['shared']==cr['metadata']['shared']==65536 and x['metadata']['num_warps']==cr['metadata']['num_warps']==8 and x['metadata']['hcu_sgpr_liveness_strategy']==cr['metadata']['hcu_sgpr_liveness_strategy']
    isa=(p.parent/'kernel.amdgcn').read_text();physical={k:int(re.search(r'\.'+k+r'\s*[:=]?\s*(\d+)',isa)[1]) for k in cr['fields']};assert physical==cr['fields'] and physical['private_segment_fixed_size']==physical['vgpr_spill_count']==0
    y=copy.deepcopy(x);scan.stats_for(p.parent,y);assert y['dispatches']==x['dispatches'] and len(x['dispatches'])==4
    expected=2*256*128*128*16/8192;assert all(d['instructions']['MMOP']==expected and d['active_mmac_cus']==[0] for d in y['dispatches'])
    ex=load(p.parent/'executor.json');assert not pathlib.Path('/proc',str(ex['child_pid'])).exists();pids.append(ex['child_pid'])
    serial=x['compile_serialization'];assert serial['acquired_monotonic_ns']<=serial['completed_monotonic_ns'];intervals.append((serial['acquired_monotonic_ns'],serial['completed_monotonic_ns'],x['id']))
    snapshot=None
    if x['status']=='PASS':assert ref['passed'] and x['accuracy']['mismatches']==0
    else:
        assert not ref['control'] and not ref['passed'] and x['config']['hcu_sgpr_liveness_strategy']==5 and x['accuracy']['mismatches']==1414
        f=p.parent/'precision-failure.pt';assert f.is_file();snapshot=dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size)
    rows.append(dict(id=x['id'],control=ref['control'],tile=x['tile'],layout=x['layout'],config=x['config'],status=x['status'],accuracy=x['accuracy'],CPU_reference=ref['CPU_reference'],CPU_ISA_resources_equal=True,physical=physical,raw_stats_hashes={d['stats']:sha(pathlib.Path(d['stats'])) for d in y['dispatches']},result_sha256=sha(p),asm_sha256=x['asm_sha256'],failure_snapshot=snapshot))
intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])) and len(set(pids))==8
counts=collections.Counter(x['status'] for x in rows if not x['control']);assert dict(counts)==dict(PASS=4,ACCURACY_FAIL=2)
assert not(R/'liveness_features_loops64-plan.json').exists() and not(R/'liveness_features_loops32-plan.json').exists()
assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
out=dict(passed=True,scope='Audit includes original numerical failures; not all-feature precision qualification',measurement_context_id=act['measurement_context_id'],points=8,raw_dispatches=32,exact_controls_passed=2,feature_status_counts=dict(counts),all_executed_MMOP_ratios=1,all_active_mmac_cus=[0],all_feature_scratch_and_VGPR_spills_zero=True,actual_SGPR_spills_preserved=True,source_hashes_unchanged=True,maximum_concurrent_compilers=1,compiler_intervals_ns=intervals,job_and_children_missing=True,long_calibration_not_started=True,rows=rows,accepted_feature_ids=[x['id'] for x in rows if not x['control'] and x['status']=='PASS'],rejected_feature_ids=[x['id'] for x in rows if x['status']=='ACCURACY_FAIL'],no_performance_map_promotion=True,script_sha256=sha(pathlib.Path(__file__)))
(R/'mixed-short-gate-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=True,points=8,raw_dispatches=32,feature_status_counts=dict(counts),long_calibration_not_started=True,failed_snapshots_preserved=True)))
