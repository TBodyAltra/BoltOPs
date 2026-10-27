"""Once-only terminal audit: single-TG calibration and excluded edge correctness."""
import pathlib,json,hashlib,sys,copy,re,statistics
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));import scan
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(R/'job-exit.json');assert e['returncode']==0 and load(R/'state.json')['phase']=='FINISHED' and not pathlib.Path('/proc',str(e['pid'])).exists();assert not(R/'native-raw-audit.json').exists()
a=load(R/'active-manifest.json');manifest=load(R/('manifest-'+a['manifest_id']+'.json'));assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
labels=['default_controls','fragment_singleTG_pilot_TT','fragment_singleTG_pilot_TN','fragment_features_short','default_edge_control','fragment_edge_odd17_pilot_TT','fragment_edge_odd17_pilot_TN','fragment_edge_odd17_features','fragment_features_loops64','fragment_features_loops32'];expected={};control_ids=set()
for label in labels:
 q=load(R/(label+'-qualification.json'));assert q['passed']
 for c in q['checks']:
  ref=load(pathlib.Path(c['CPU_reference'])/'result.json');expected[c['id']]=ref
  if label=='default_controls':control_ids.add(c['id'])
assert len(control_ids)==2
rows=[];byid={};intervals=[]
for p in sorted((R/'runs').glob('*/result.json')):
 x=load(p);assert x['id'] in expected and x['measurement_context_id']==a['measurement_context_id'] and x['status']=='PASS' and x['input_seed']==29 and x['input_scale']==1 and x['atol']==x['rtol']==1e-3 and x['accuracy']['elements']==x['shape'][0]*x['shape'][1] and x['accuracy']['mismatches']==x['accuracy']['nonfinite']==x['n_spills']==0
 edge=x.get('correctness_only',False);bm,bn,bk=x['tile'];m,n,k=x['shape'];assert bk==128 and x['grid']==([4] if edge else [1]) and ([m,n]==[bm+16,bn+16] if edge else [m,n]==[bm,bn]) and x['scope']==('full_gemm' if edge else 'single_cta')
 assert not x['config']['WARP_SPECIALIZE'] and not x['config']['wasp_wdra'] and x['metadata']['shared']<=131072 and x['loaded_native']['sha256']=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
 if edge:assert x['not_for_performance_map'] is True and not any(f in x for f in ['tops','peak_fp16_flops_per_cu_cycle','mmac_ideal_fraction_including_fill_drain','cycles_per_bk_including_fill_drain'])
 else:assert x['peak_fp16_flops_per_cu_cycle']==4096 and x['benchmark_kind']=='single_TG_streaming_BK_pipeline'
 y=copy.deepcopy(x);scan.stats_for(p.parent,y);assert y['dispatches']==x['dispatches'] and y['median_cycles']==x['median_cycles'] and len(y['dispatches'])==4 and x['asm_sha256']==sha(p.parent/'kernel.amdgcn')
 mmops=2*bm*bn*bk/8192*x['pipeline_loops']*(4 if edge else 1)
 assert all(d['instructions']['MMOP']==mmops and (0<len(d['active_mmac_cus'])<=4 if edge else d['active_mmac_cus']==[0]) for d in y['dispatches'])
 isa=(p.parent/'kernel.amdgcn').read_text();physical={}
 for name in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
  z=re.search(r'\.'+name+r'\s*[:=]?\s*(\d+)',isa);assert z;physical[name]=int(z[1]);assert physical[name]==expected[x['id']]['fields'][name]
 assert physical['private_segment_fixed_size']==physical['vgpr_spill_count']==0
 ex=load(p.parent/'executor.json');assert 1<=ex['parallelism_limit']<=8;assert not pathlib.Path('/proc',str(ex['child_pid'])).exists();cs=x['compile_serialization'];assert cs['completed_monotonic_ns']>=cs['acquired_monotonic_ns'];intervals.append((cs['acquired_monotonic_ns'],cs['completed_monotonic_ns'],x['id']))
 if x['id'] not in control_ids and not edge:assert x['config']['STREAM_K_FRAGMENTS'] in [2,4] and x['config']['LOOP_UNROLL_FACTOR']==1 and all(physical[k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
 rows.append(dict(id=x['id'],tile=x['tile'],layout=x['layout'],correctness_only=edge,grid=x['grid'],shape=x['shape'],feature=x['id'] not in control_ids,pipeline_loops=x['pipeline_loops'],median_cycles=x['median_cycles'],MMOP_per_dispatch=mmops,executed_to_logical_MMAC_ratio=1 if not edge else None,physical=physical,raw_stats_hashes={d['stats']:sha(pathlib.Path(d['stats'])) for d in y['dispatches']},result_sha256=sha(p),asm_sha256=x['asm_sha256']));byid[x['id']]=x
intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
n=len(load(R/'cpu-admission.json')['results']);assert n==12 and len(rows)==2+3*n+1+n and len(expected)==len(rows);assert sum(x['correctness_only'] for x in rows)==13
candidates=load(R/'pipeline-candidates.json');reference=load(R/'reference-map.json');assert candidates['no_map_promotion'] and len(candidates['candidates'])==n;comparisons=[]
for c in candidates['candidates']:
 points=[byid[i] for i in c['source_ids']];assert [x['pipeline_loops'] for x in points]==[16,32,64] and len({x['asm_sha256'] for x in points})==1 and all(not x['correctness_only'] for x in points)
 slope=(points[2]['median_cycles']-points[0]['median_cycles'])/48;intercept=points[0]['median_cycles']-16*slope;res=abs(points[1]['median_cycles']-(intercept+32*slope))/points[2]['median_cycles'];frac=2*c['tile'][0]*c['tile'][1]*128/(slope*4096);assert c['cycles_per_bk_slope']==slope and c['intermediate_residual_fraction_of_long_time']==res and c['mmac_theoretical_fraction']==frac and c['steady']==(res<=.02 and 0<frac<=1)
 assert len({load(R/'runs'/i/'executor.json')['child_pid'] for i in c['source_ids']})==3
 edge_matches=[x for x in byid.values() if x['correctness_only'] and x['config']==points[0]['config'] and x['layout']==c['layout'] and x['tile']==c['tile']];assert len(edge_matches)==1 and edge_matches[0]['asm_sha256']==points[0]['asm_sha256'] and edge_matches[0]['pipeline_loops']==17
 for choice in ['best','best_no_spill','best_64k','best_64k_no_spill']:
  old=next((x for x in reference[choice] if x['tile']==c['tile'] and x['layout']==c['layout']),None)
  if old is None:continue
  comparisons.append(dict(choice=choice,tile=c['tile'],layout=c['layout'],config=c['config'],fraction=frac,prior_fraction=old['mmac_theoretical_fraction'],relative_initial_gain=frac/old['mmac_theoretical_fraction']-1,residual=res,steady=c['steady'],physical=c['physical'],source_ids=c['source_ids'],old_source_ids=old['source_ids'],independent_confirmation_required=True))
assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
out=dict(passed=True,points=len(rows),raw_dispatches=4*len(rows),singleTG_points=sum(not x['correctness_only'] for x in rows),off_origin_edge_correctness_only_points=13,edge_grid_TGs=4,edge_data_excluded_from_performance_map=True,all_feature_odd_K_over_BK17=True,all_full_output_gold=True,all_singleTG_MMOP_ratios=1,all_scratch_and_VGPR_spills_zero=True,actual_SGPR_spills_preserved=True,physical_zero_candidates=sum(all(c['physical'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']) for c in candidates['candidates']),compiler_intervals_nonoverlap=True,max_compilers=1,source_hashes_unchanged=True,rows=rows,comparisons=comparisons,candidate_sha256=sha(R/'pipeline-candidates.json'),script_sha256=sha(pathlib.Path(__file__)),no_map_promotion=True,independent_confirmation_still_required=True)
(R/'native-raw-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,points=len(rows),raw_dispatches=4*len(rows),singleTG=out['singleTG_points'],edge=13,steady=sum(c['steady'] for c in candidates['candidates']),zero=out['physical_zero_candidates'])))
