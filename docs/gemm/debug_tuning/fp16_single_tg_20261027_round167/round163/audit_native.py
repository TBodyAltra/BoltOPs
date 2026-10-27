"""Terminal raw audit for nowasp zero-spill features; original controls retain their genuine SGPR spills."""
import pathlib,json,hashlib,sys,copy,re,statistics
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));import scan
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exit=json.loads((R/'job-exit.json').read_text());assert exit['returncode']==0 and json.loads((R/'state.json').read_text())['phase']=='FINISHED'
assert not pathlib.Path('/proc/'+str(exit['pid'])).exists(),'Job still live: audit only after actual terminal'
a=json.loads((R/'active-manifest.json').read_text());manifest=json.loads((R/('manifest-'+a['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
assert not(R/'native-raw-audit.json').exists()
controls=json.loads((R/'default_controls-qualification.json').read_text());assert controls['passed'];control_ids={x['id']:x for x in controls['checks']};assert len(control_ids)==5
rows=[];byid={}
for p in sorted((R/'runs').glob('*/result.json')):
 x=json.loads(p.read_text());assert x['measurement_context_id']==a['measurement_context_id'] and x['status']=='PASS' and x['grid']==[1] and x['shape'][:2]==x['tile'][:2] and x['input_seed']==29 and x['input_scale']==1 and x['accuracy']['elements']==x['tile'][0]*x['tile'][1] and x['accuracy']['mismatches']==x['accuracy']['nonfinite']==x['n_spills']==0
 y=copy.deepcopy(x);scan.stats_for(p.parent,y);assert y['dispatches']==x['dispatches'] and len(x['dispatches'])==4 and x['asm_sha256']==sha(p.parent/'kernel.amdgcn')
 expected=2*x['tile'][0]*x['tile'][1]*x['tile'][2]/8192*x['pipeline_loops'];assert all(d['instructions']['MMOP']==expected and d['active_mmac_cus']==[0] for d in y['dispatches'])
 isa=(p.parent/'kernel.amdgcn').read_text();physical={}
 for name in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']:
  m=re.search(r'\.'+name+r'\s*[:=]?\s*(\d+)',isa);assert m;physical[name]=int(m[1]);assert physical[name]==(control_ids[x['id']]['physical'][name] if x['id'] in control_ids else 0)
 if x['id'] not in control_ids:
  assert x['config']['USE_MATRIX_STORE'] is True and x['config']['mmac_layout_force']==3 and x['config']['MMA_K_SPLIT'] in [1,2] and 'matrix_store_32x32_b16' in isa
 row=dict(id=x['id'],tile=x['tile'],layout=x['layout'],feature=x['id'] not in control_ids,pipeline_loops=x['pipeline_loops'],median_cycles=x['median_cycles'],MMOP_per_dispatch=expected,executed_to_logical_mmac_ratio=1,physical=physical,raw_stats_hashes={d['stats']:sha(pathlib.Path(d['stats'])) for d in y['dispatches']},result_sha256=sha(p),asm_sha256=x['asm_sha256']);rows.append(row);byid[x['id']]=x
n=len(json.loads((R/'cpu-admission.json').read_text())['results']);assert 1<=n<=24;assert len(rows)==5+3*n and sum(x['feature'] for x in rows)==3*n and sum(not x['feature'] for x in rows)==5
assert all(not x['config']['WARP_SPECIALIZE'] and not x['config']['wasp_wdra'] and x['metadata']['shared']<=65536 and x['loaded_native']['sha256']=='7eb9fbc909873af0faaf37002362bde031b406530473095d104ba86df3926193' for x in byid.values())
candidates=json.loads((R/'pipeline-candidates.json').read_text());assert candidates['no_map_promotion'] and len(candidates['candidates'])==n;reference=json.loads((R/'reference-map.json').read_text());comparisons=[]
for c in candidates['candidates']:
 points=[byid[i] for i in c['source_ids']];assert [q['pipeline_loops'] for q in points]==[16,32,64] and len({q['asm_sha256'] for q in points})==1
 slope=(points[2]['median_cycles']-points[0]['median_cycles'])/48;intercept=points[0]['median_cycles']-16*slope;res=abs(points[1]['median_cycles']-(intercept+32*slope))/points[2]['median_cycles'];frac=2*c['tile'][0]*c['tile'][1]*128/(slope*points[0]['peak_fp16_flops_per_cu_cycle']);assert c['cycles_per_bk_slope']==slope and c['intermediate_residual_fraction_of_long_time']==res and c['mmac_theoretical_fraction']==frac and c['steady']==(res<=.02 and 0<frac<=1)
 pids=[json.loads((R/'runs'/i/'executor.json').read_text())['child_pid'] for i in c['source_ids']];assert len(set(pids))==3 and all(not pathlib.Path('/proc/'+str(pid)).exists() for pid in pids)
 for choice in ['best','best_no_spill','best_64k','best_64k_no_spill']:
  matches=[x for x in reference[choice] if x['tile']==c['tile'] and x['layout']==c['layout']]
  if not matches:continue
  old=matches[0];comparisons.append(dict(choice=choice,prior_metric_steady=bool(old.get('valid_metric') and old.get('identical_binary_across_loop_counts') and len(old.get('loop_counts',[]))>=3 and old.get('intermediate_residual_fraction_of_long_time',1)<=.02),tile=c['tile'],layout=c['layout'],unroll=c['config']['LOOP_UNROLL_FACTOR'],ILP=c['config']['amdgpu_enable_max_ilp_scheduling_strategy'],schedV2=c['config']['hcu_use_gfx946_sched_model_v2'],new_fraction=frac,prior_fraction=old['mmac_theoretical_fraction'],relative_initial_gain=frac/old['mmac_theoretical_fraction']-1,new_residual=res,steady=c['steady'],independent_confirmation_required=True,old_source_ids=old['source_ids'],new_source_ids=c['source_ids']))
assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
out=dict(passed=True,points=len(rows),raw_dispatches=4*len(rows),all_MMOP_ratios=1,all_full_output_gold=True,all_feature_physical_spills_zero=True,control_SGPR_spills_preserved=True,source_hashes_unchanged=True,rows=rows,comparisons=comparisons,no_map_promotion=True,independent_confirmation_still_required=True,candidate_sha256=sha(R/'pipeline-candidates.json'),script_sha256=sha(pathlib.Path(__file__)))
(R/'native-raw-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,points=len(rows),raw_dispatches=4*len(rows),comparisons=comparisons)))
