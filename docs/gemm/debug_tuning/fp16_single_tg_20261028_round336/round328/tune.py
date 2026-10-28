"""PMD MLS recenter qualification, off-origin edge checks and single-TG slopes."""
import pathlib,json,hashlib,re,fcntl,os,time,sys
import scan
R=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def code(p):
 lines=[]
 for line in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',line):break
  line=line.split(';',1)[0].strip()
  if line and not line.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(line)
 return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
def template(row):
 c=row['config'];mls=row.get('mls',True);wasp=c['WARP_SPECIALIZE'];assert not c['wasp_wdra'];allowed={'hcu_sgpr_liveness_strategy','misched_regpressure','amdgpu_enable_max_ilp_scheduling_strategy','hcu_use_gfx946_sched_model_v2','schedule_hint','mls_global_prefetch','hcu_fragment_mls_layout'};opts={k:v for k,v in c.items() if k in allowed}
 return dict(scan.spec(tile=row['tile'],layout=row['layout'],mode='wasp44' if wasp else 'nowasp',stages=c['num_stages'],regs=None,warps=c.get('num_warps',4),empty=c['empty_arrive_after_mmac'],cluster=c['enable_v_mmac_cluster'],mmac=c['mmac_layout_force'],mls=mls,matrix_store=c.get('USE_MATRIX_STORE',False)),compiler_options=opts,wasp_single_buffer=False,wasp_global_k_fragments=c.get('WASP_GLOBAL_K_FRAGMENTS',1),dot_k_split=c.get('MMA_K_SPLIT',1),pipeline_unroll_factor=c.get('LOOP_UNROLL_FACTOR',1),load_b_first=c.get('LOAD_B_FIRST',False),mixed_operand=None,mixed_vector_full=False,partition_warps=c.get('wasp_partition_warps') if wasp else None,use_async_copy=c.get('use_async_copy',False),use_block_pingpong=c.get('use_block_pingpong',False),async_global_prefetch=c.get('block_pingpong_async_global_prefetch'),async_local_prefetch=c.get('block_pingpong_async_local_prefetch'),shape=[row['tile'][0],row['tile'][1],128*16],pipeline_loops=16,optimize_epilogue=c['optimize_epilogue'],simplify_group1=c.get('SIMPLIFY_GROUP1',False),rebase_mls_tile=c.get('REBASE_MLS_TILE',0),correctness_only=False,stream_k_fragments=1,stream_fragment_prefetch=False,recipe='large_global_k_fragment307')
def fields(p):
 s=p.read_text();r={}
 for n in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count','amdhsa_next_free_vgpr']:
  m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',s);r[n]=int(m[1]) if m else None
 return r
with (R/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((R/'prepare-proof.json').read_text())['passed'] and json.loads((R/'layout3-review.json').read_text())['passed']
 cpu=json.loads((R/'cpu-admission.json').read_text());assert all(sha(pathlib.Path(p))==h for p,h in cpu['source_hashes'].items())
 manifest=scan.capture_manifest();mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16];ctx=scan.measurement_context(manifest);scan.write_json(R/('manifest-'+mid+'.json'),manifest);scan.write_json(R/'active-manifest.json',dict(manifest_id=mid,measurement_context_id=ctx))
 def check():
  assert all(scan.digest(p)==v for p,v in manifest['hashes'].items())
  assert all(os.environ.get(k)==v for k,v in manifest['env'].items())
 def measure(refs,label,loops=16,maximum=8,edge=False):
  specs=[dict(template(x),scope='full_gemm' if edge else 'single_cta',correctness_only=edge,shape=[x['tile'][0]+(16 if edge else 0),x['tile'][1]+(16 if edge else 0),128*loops],pipeline_loops=loops) for x in refs];scan.write_json(R/(label+'-plan.json'),dict(specs=specs,CPU_directories=[x['directory'] for x in refs],loops=loops,maximum_models=maximum,full_gold=True,promotion_allowed=False))
  scan.write_json(R/'state.json',dict(phase=label,pid=os.getpid(),measurement_context_id=ctx));check();scan.execute_points(specs,R/'runs',mid,ctx,'validate',maximum);check();checks=[]
  for spec,ref in zip(specs,refs):
   sid=scan.point_key(spec,ctx);p=R/'runs'/sid;q=json.loads((p/'result.json').read_text());actual=fields(p/'kernel.amdgcn') if (p/'kernel.amdgcn').exists() else None
   ok=bool(q.get('status')=='PASS' and q.get('loaded_native',{}).get('sha256')==cpu['build']['sha256'] and q['accuracy']['mismatches']==q['accuracy']['nonfinite']==0 and q['accuracy']['elements']==spec['shape'][0]*spec['shape'][1] and q['n_spills']==0 and q['config'].get('SIMPLIFY_GROUP1',False)==ref['config'].get('SIMPLIFY_GROUP1',False) and q['config'].get('REBASE_MLS_TILE',0)==ref['config'].get('REBASE_MLS_TILE',0) and q['config'].get('STREAM_K_FRAGMENTS',1)==ref['config'].get('STREAM_K_FRAGMENTS',1) and q['config'].get('STREAM_FRAGMENT_PREFETCH',False)==ref['config'].get('STREAM_FRAGMENT_PREFETCH',False) and all(d['instructions']['MMOP']==2*spec['tile'][0]*spec['tile'][1]*128/8192*loops*(4 if edge else 1) and (0<len(d['active_mmac_cus'])<=4 if edge else d['active_mmac_cus']==[0]) for d in q['dispatches']) and actual==ref['fields'] and q['metadata']['shared']==ref['metadata']['shared'] and q['metadata']['num_warps']==ref['metadata']['num_warps'] and q['metadata']['misched_regpressure']==ref['metadata']['misched_regpressure'] and q['metadata'].get('hcu_sgpr_liveness_strategy',0)==ref['metadata'].get('hcu_sgpr_liveness_strategy',0) and q['metadata'].get('hcu_split_b_slot_mmac_turn')==ref['metadata'].get('hcu_split_b_slot_mmac_turn') and q['metadata'].get('hcu_alias_ab_slots')==ref['metadata'].get('hcu_alias_ab_slots') and q['metadata'].get('wasp_partition_regs_actual')==ref['metadata'].get('wasp_partition_regs_actual') and q['metadata'].get('wasp_partition_regs')==ref['metadata'].get('wasp_partition_regs') and q['metadata'].get('wasp_partition_regs_requested')==ref['metadata'].get('wasp_partition_regs_requested') and code(p/'kernel.amdgcn')==code(pathlib.Path(ref['directory'])/'kernel.amdgcn'))
   if edge:ok=bool(ok and q.get('not_for_performance_map') is True and q['grid']==[4])
   checks.append(dict(id=sid,tile=spec['tile'],layout=spec['layout'],correctness_only=edge,grid=q.get('grid'),passed=ok,status=q.get('status'),accuracy=q.get('accuracy'),physical=actual,CPU_reference=ref['directory']))
  scan.write_json(R/(label+'-qualification.json'),dict(passed=all(x['passed'] for x in checks),checks=checks));assert all(x['passed'] for x in checks),checks
  return checks
 controls=measure(json.loads((R/'cpu-controls.json').read_text())['controls'],'fresh_controls')
 assert len(cpu['results'])==3 and all(x['all_physical_spills_zero'] for x in cpu['results'])
 measure(cpu['results'],'fresh_recipes16',16)
 measure(cpu['results'],'fresh_recipes64',64)
 measure(cpu['results'],'fresh_recipes32',32)
 candidates=[]
 for ref in cpu['results']:
  points=[]
  for loops in [16,32,64]:
   spec=dict(template(ref),shape=[ref['tile'][0],ref['tile'][1],128*loops],pipeline_loops=loops);sid=scan.point_key(spec,ctx);p=R/'runs'/sid;q=json.loads((p/'result.json').read_text());points.append(q)
  assert len({x['asm_sha256'] for x in points})==1
  a,b,c=points;slope=(c['median_cycles']-a['median_cycles'])/48;assert slope>0
  intercept=a['median_cycles']-16*slope;residual=abs(b['median_cycles']-(intercept+32*slope))/c['median_cycles']
  volume=2*ref['tile'][0]*ref['tile'][1]*128;peak=a['peak_fp16_flops_per_cu_cycle'];fraction=volume/(slope*peak)
  mmops=[(x['pipeline_loops'],__import__('statistics').median(d['instructions']['MMOP'] for d in x['dispatches'][1:])) for x in points]
  ratio=(mmops[-1][1]-mmops[0][1])/48*8192/volume;assert ratio==1
  candidates.append(dict(tile=ref['tile'],layout=ref['layout'],config=points[0]['config'],source_ids=[x['id'] for x in points],loop_counts=[16,32,64],median_cycles=[x['median_cycles'] for x in points],cycles_per_bk_slope=slope,fill_drain_intercept_cycles=intercept,intermediate_residual_fraction_of_long_time=residual,mmac_theoretical_fraction=fraction,executed_to_logical_mmac_ratio=ratio,steady=residual<=.02 and 0<fraction<=1,shared_bytes=a['metadata']['shared'],actual_partition_regs=a['actual_partition_regs'],physical=fields(R/'runs'/a['id']/'kernel.amdgcn'),independent_confirmation_required=True))
 scan.write_json(R/'pipeline-candidates.json',dict(measurement_context_id=ctx,candidates=candidates,no_map_promotion=True,independent_confirmation_required=True,source_hashes_unchanged=True))
 check()
 scan.write_json(R/'state.json',dict(phase='FINISHED',measurement_context_id=ctx,pid=os.getpid()))
