"""Read-only terminal audit; do not run until confirmation103 job and models drain."""
import pathlib,json,hashlib,sys,copy,re,statistics
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));import scan
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exit=json.loads((R/'job-exit.json').read_text());assert exit['returncode']==0 and json.loads((R/'state.json').read_text())['phase']=='FINISHED'
assert not pathlib.Path('/proc/'+str(exit['pid'])).exists(),'Job still live: audit only after actual terminal'
a=json.loads((R/'active-manifest.json').read_text());manifest=json.loads((R/('manifest-'+a['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
rows=[];byid={}
for p in sorted((R/'runs').glob('*/result.json')):
 x=json.loads(p.read_text());assert x['measurement_context_id']==a['measurement_context_id'] and x['status']=='PASS' and x['grid']==[1] and x['shape'][:2]==x['tile'][:2] and x['input_seed']==29 and x['input_scale']==1 and x['accuracy']['elements']==x['tile'][0]*x['tile'][1] and x['accuracy']['mismatches']==x['accuracy']['nonfinite']==x['n_spills']==0
 y=copy.deepcopy(x);scan.stats_for(p.parent,y);assert y['dispatches']==x['dispatches'] and len(x['dispatches'])==4 and x['asm_sha256']==sha(p.parent/'kernel.amdgcn')
 expected=2*x['tile'][0]*x['tile'][1]*x['tile'][2]/8192*x['pipeline_loops'];assert all(d['instructions']['MMOP']==expected and d['active_mmac_cus']==[0] for d in y['dispatches'])
 isa=(p.parent/'kernel.amdgcn').read_text();physical={}
 for name in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']:
  m=re.search(r'\.'+name+r'\s*[:=]?\s*(\d+)',isa);assert m and int(m[1])==0;physical[name]=0
 row=dict(id=x['id'],tile=x['tile'],layout=x['layout'],feature=bool(x['compiler_options'].get('hcu_wasp_split_b_slot_turn',0)),pipeline_loops=x['pipeline_loops'],median_cycles=x['median_cycles'],MMOP_per_dispatch=expected,executed_to_logical_mmac_ratio=1,physical=physical,raw_stats_hashes={d['stats']:sha(pathlib.Path(d['stats'])) for d in y['dispatches']},result_sha256=sha(p),asm_sha256=x['asm_sha256']);rows.append(row);byid[x['id']]=x
assert len(rows)==10 and sum(x['feature'] for x in rows)==6 and sum(not x['feature'] for x in rows)==4
confirm=json.loads((R/'confirmation.json').read_text());plan=json.loads((R/'confirmation-plan.json').read_text());assert confirm['plan_sha256']==sha(R/'confirmation-plan.json') and len(confirm['results'])==len(plan['selected'])==2 and plan['one_independent_series_per_initial_eligible_recipe']
comparisons=[]
for c in confirm['results']:
 points=[byid[i] for i in c['fresh_source_ids']];assert [q['pipeline_loops'] for q in points]==[16,32,64] and len({q['asm_sha256'] for q in points})==1 and c['initial_context']!=c['fresh_context']==a['measurement_context_id']
 slope=(points[2]['median_cycles']-points[0]['median_cycles'])/48;intercept=points[0]['median_cycles']-16*slope;res=abs(points[1]['median_cycles']-(intercept+32*slope))/points[2]['median_cycles'];frac=2*c['tile'][0]*c['tile'][1]*128/(slope*points[0]['peak_fp16_flops_per_cu_cycle']);gain=frac/c['baseline']['mmac_theoretical_fraction']-1
 assert c['cycles_per_bk_slope']==slope and c['intermediate_residual_fraction_of_long_time']==res and c['mmac_theoretical_fraction']==frac and c['relative_fresh_gain']==gain
 assert c['steady']==(res<=.02 and 0<frac<=1) and c['confirmed']==(c['steady'] and gain>=.03)
 pids=[json.loads((R/'runs'/i/'executor.json').read_text())['child_pid'] for i in c['fresh_source_ids']]+[json.loads((pathlib.Path(plan['initial_root'])/'runs'/i/'executor.json').read_text())['child_pid'] for i in c['initial_source_ids']];assert len(set(pids))==6
 comparisons.append(dict(tile=c['tile'],layout=c['layout'],initial_fraction=c['initial']['mmac_theoretical_fraction'],fresh_fraction=frac,prior_fraction=c['baseline']['mmac_theoretical_fraction'],relative_fresh_gain=gain,fresh_residual=res,confirmed=c['confirmed'],initial_source_ids=c['initial_source_ids'],fresh_source_ids=c['fresh_source_ids'],six_distinct_model_process_ids=pids))
assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
out=dict(passed=True,points=10,raw_dispatches=40,all_MMOP_ratios=1,all_full_output_gold=True,all_physical_spills_zero=True,source_hashes_unchanged=True,rows=rows,comparisons=comparisons,no_map_promotion=True,independent_confirmation_complete=True,confirmation_sha256=sha(R/'confirmation.json'),script_sha256=sha(pathlib.Path(__file__)))
(R/'native-raw-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,points=10,raw_dispatches=40,comparisons=comparisons)))
