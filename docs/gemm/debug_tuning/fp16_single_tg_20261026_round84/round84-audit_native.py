"""Audit raw R84 execution, including independent re-reading of each MMOP counter.
No map mutation. Ranking keeps the original >=3% initial/fresh and steady gates.
"""
import pathlib,json,hashlib,math,re,statistics,configparser
from stats_io import read_stats
R=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
steady=lambda x:len(x['loop_counts'])>=3 and x.get('intermediate_residual_fraction_of_long_time',1)<=.02 and 0<x['mmac_theoretical_fraction']<=1
active=json.loads((R/'active-manifest.json').read_text());mp=R/('manifest-'+active['manifest_id']+'.json');manifest=json.loads(mp.read_text())
assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
assert json.loads((R/'job-exit.json').read_text())['returncode']==0
for n in ['qualification.json','feature-qualification.json','cpu-native-equality.json','liveness-pilot-qualification.json']:assert json.loads((R/n).read_text())['passed']
admission=json.loads((R/'admission.json').read_text());N=admission['admitted'];assert N==40
initial=json.loads((R/'analysis/pipeline-map.json').read_text())['candidates'];assert len(initial)==N
confirmation=json.loads((R/'confirmation.json').read_text());plan=json.loads((R/'confirmation-plan.json').read_text());nfresh=plan['points'];assert nfresh==3*len(plan['selected'])
fresh=json.loads((R/'analysis/confirmation/pipeline-map.json').read_text())['candidates'] if nfresh else []
assert len(fresh)==nfresh//3
raws={};pids={};checks=[];controls=[]
for label,count in [('runs',8+3*N),('confirmation-runs',nfresh)]:
 paths=list((R/label).glob('*/result.json'));assert len(paths)==count;pids[label]=set()
 for p in paths:
  q=json.loads(p.read_text());exe=json.loads((p.parent/'executor.json').read_text());pids[label].add(exe['child_pid']);assert q['status']=='PASS' and q['measurement_context_id']==active['measurement_context_id']
  assert q['input_seed']==29 and q['input_scale']==1 and q['atol']==q['rtol']==1e-3 and q['warmup_launches']==1 and q['measured_launches']==3
  assert q['accuracy']['mismatches']==q['accuracy']['nonfinite']==0 and q['accuracy']['elements']==q['tile'][0]*q['tile'][1]
  assert q['grid']==[1] and q['scope']=='single_cta' and q['shape'][:2]==q['tile'][:2]
  if not q.get('config',{}).get('hcu_wasp_first_operand_early'):
   assert label=='runs';controls.append(dict(id=p.parent.name,PID=exe['child_pid'],result_sha256=sha(p)));continue
  assert q['tile'] in ([128,256,128],[256,128,128]) and q['shape'][2]==q['tile'][2]*q['pipeline_loops'] and q['pipeline_loops'] in (16,32,64)
  assert q['n_spills']==0 and q['metadata']['shared']==98304 and q['effective_num_warps']==12
  assert q['metadata']['hcu_first_operand_early_capture']==('B' if q['config'].get('LOAD_B_FIRST',False) else 'A') and q['config']['hcu_wasp_merge_ab'] is False
  assert q['metadata']['hcu_full_BK_local_prefetch'] is True and q['metadata']['hcu_partial_k_local_reads']==8 and q['metadata']['hcu_partial_k_fragments_per_bk']==2
  assert bool(q['metadata'].get('hcu_merged_AB_data_barriers',False))==q['config']['hcu_wasp_merge_ab']
  assert q['config']['MMA_K_SPLIT']==2 and q['config']['empty_arrive_after_mmac'] is False and q['config']['hcu_wasp_single_buffer'] is True
  assert q['metadata'].get('profile_scratch_size',0)==0 and sum(q['actual_partition_regs'])<=512
  asm=(p.parent/'kernel.amdgcn').read_text();fields={}
  for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count']:
   m=re.search(r'\.'+k+r'\s*[:=]?\s*(\d+)',asm);assert m and int(m[1])==0;fields[k]=int(m[1])
  assert not re.search(r'\bscratch_(?:load|store)',asm) and sha(p.parent/'kernel.amdgcn')==q['asm_sha256']
  expected=2*math.prod(q['shape'])/8192;dispatches=q['dispatches'];assert len(dispatches)==4
  cfg=configparser.ConfigParser(strict=False);cfg.read(p.parent/'m5out/config.ini');assert cfg['gpu_configuration']['mmop_f16_i8_f8']=='8' and cfg['gpu_configuration']['simds_per_cu']=='4'
  raw_checks=[]
  for d in dispatches:
   raw=read_stats(d['stats']);vals={}
   for line in raw.decode().splitlines():
    parts=line.split()
    if len(parts)<2:continue
    try:vals[parts[0]]=float(parts[1])
    except ValueError:pass
   mmops={int(re.search(r'CUs(\d+)',k)[1]):v for k,v in vals.items() if re.fullmatch(r'system.shaders.CUs\d+.numInstrTypeExecuted::MMOP',k)}
   active_cus=sorted(k for k,v in mmops.items() if v)
   assert sum(mmops.values())==expected==d['instructions']['MMOP'] and active_cus==d['active_mmac_cus'] and len(active_cus)==1
   assert vals['system.gcSimCycles']==d['cycles'] and vals['gcFrequency']==d['frequency_ghz']
   raw_checks.append(dict(logical_path=d['stats'],raw_sha256=hashlib.sha256(raw).hexdigest(),raw_bytes=len(raw),MMOP=sum(mmops.values()),cycles=d['cycles'],active_cus=active_cus))
  assert statistics.median(d['cycles'] for d in dispatches[1:])==q['median_cycles']
  record=dict(root=label,id=p.parent.name,tile=q['tile'],shape=q['shape'],layout=q['layout'],PID=exe['child_pid'],elements=q['accuracy']['elements'],mismatches=0,nonfinite=0,expected_MMOP_per_dispatch=expected,executed_to_logical_MMAC_ratio=1,physical_spills=fields,result_sha256=sha(p),asm_sha256=q['asm_sha256'],executor_sha256=sha(p.parent/'executor.json'),raw_stats=raw_checks)
  checks.append(record);raws[(label,p.parent.name)]=q
assert len(controls)==8 and len(checks)==3*N+nfresh and len(pids['runs'])==8+3*N and len(pids['confirmation-runs'])==nfresh and not pids['runs']&pids['confirmation-runs']
for label,groups in [('runs',initial),('confirmation-runs',fresh)]:
 for x in groups:
  assert len(x['source_ids'])==3 and x['loop_counts']==[16,32,64] and 0<x['mmac_theoretical_fraction']<=1
  rows=[raws[(label,i)] for i in x['source_ids']];assert len({q['asm_sha256'] for q in rows})==1
  assert all(q['metadata']['shared']==x['shared_bytes'] and q['actual_partition_regs']==x['actual_partition_regs'] for q in rows)
for x in fresh:
 y=next(y for y in initial if y['tile']==x['tile'] and y['layout']==x['layout'] and y['config']==x['config']);assert x['asm_sha256']==y['asm_sha256']
for x in confirmation['profiles']:
 for comparison in x['comparisons']:
  if comparison['confirmed']:
   assert x['repeat'] and steady(x['initial']) and steady(x['repeat'])
   assert comparison['coverage'] or (comparison['initial_gain']>=.03 and comparison['repeat_gain']>=.03)
assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
proof=dict(passed=True,CPU_only=False,scope='8 default-off controls + all40 actual-ISA qualified first-operand early capture rectangular recipes x3 lengths, plus predeclared fresh points. Raw MMOP counters independently re-read, full-output precision, physical spill, immutable code and independent PID audit. Steady and performance gates are reported, never waived.',new_points=3*N+nfresh,control_points=8,initial_recipes=N,initial_steady=sum(steady(x) for x in initial),fresh_recipes=len(fresh),checks=checks,controls=controls,manifest_sha256=sha(mp),source_hashes_unchanged=True,initial_map_sha256=sha(R/'analysis/pipeline-map.json'),repeat_map_sha256=sha(R/'analysis/confirmation/pipeline-map.json') if nfresh else None,confirmation_sha256=sha(R/'confirmation.json'),no_map_mutation=True,audit_source_sha256=sha(pathlib.Path(__file__)))
(R/'native-audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(passed=True,new_points=proof['new_points'],controls=8,initial=N,steady=proof['initial_steady'],fresh=len(fresh),all_actual_MMOP_ratio=1,all_physical_spills=0)))
