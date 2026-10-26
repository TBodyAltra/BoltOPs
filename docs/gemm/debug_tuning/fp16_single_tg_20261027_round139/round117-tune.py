"""PMD full-output qualification and slope calibration of independent B0/B1 slots."""
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
 c=row['config'];assert c['WARP_SPECIALIZE'] and c['wasp_wdra'] and c['hcu_wasp_prefetch_one_operand']==1 and not c.get('LOAD_B_FIRST')
 opts={k:v for k,v in c.items() if k in ['amdgpu_enable_max_ilp_scheduling_strategy','hcu_use_gfx946_sched_model_v2','schedule_hint','mls_global_prefetch','hcu_fragment_mls_layout','hcu_wasp_partial_k_local_read','hcu_wasp_mmac_turn','hcu_wasp_merge_ab','hcu_wasp_prefetch_full_k','hcu_wasp_first_operand_early','hcu_wasp_prefetch_one_operand','hcu_wasp_split_b_global_probe','hcu_wasp_split_b_slots','hcu_wasp_split_b_slot_turn','hcu_wasp_split_b_slot_turn_wait_mask','hcu_wasp_alias_ab_slots']}
 return dict(scan.spec(tile=row['tile'],layout=row['layout'],mode='wdra48',stages=1,regs=c['wasp_partition_regs'],warps=None,empty=False,cluster=c['enable_v_mmac_cluster'],mmac=c['mmac_layout_force'],mls=True,matrix_store=c.get('USE_MATRIX_STORE',False)),compiler_options=opts,wasp_single_buffer=True,dot_k_split=2,pipeline_unroll_factor=1,load_b_first=False,mixed_operand=None,mixed_vector_full=False,partition_warps=None,use_async_copy=False,use_block_pingpong=False,async_global_prefetch=None,async_local_prefetch=None,shape=[row['tile'][0],row['tile'][1],128*16],pipeline_loops=16,recipe='alias_A_B_64KiB_native117')
def fields(p):
 s=p.read_text();r={}
 for n in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count','amdhsa_next_free_vgpr']:
  m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',s);r[n]=int(m[1]) if m else None
 return r
with (R/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((R/'prepare-proof.json').read_text())['passed'] and json.loads((R/'parent-address-audit.json').read_text())['passed'] and json.loads((R/'ownership-isa-audit.json').read_text())['passed']
 cpu=json.loads((R/'cpu-admission.json').read_text());assert all(sha(pathlib.Path(p))==h for p,h in cpu['source_hashes'].items())
 manifest=scan.capture_manifest();mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16];ctx=scan.measurement_context(manifest);scan.write_json(R/('manifest-'+mid+'.json'),manifest);scan.write_json(R/'active-manifest.json',dict(manifest_id=mid,measurement_context_id=ctx))
 def check():
  assert all(scan.digest(p)==v for p,v in manifest['hashes'].items())
  assert all(os.environ.get(k)==v for k,v in manifest['env'].items())
 def measure(refs,label,loops=16):
  specs=[dict(template(x),shape=[x['tile'][0],x['tile'][1],128*loops],pipeline_loops=loops) for x in refs];scan.write_json(R/(label+'-plan.json'),dict(specs=specs,CPU_directories=[x['directory'] for x in refs],loops=loops,maximum_models=8,full_gold=True,promotion_allowed=False))
  scan.write_json(R/'state.json',dict(phase=label,pid=os.getpid(),measurement_context_id=ctx));check();scan.execute_points(specs,R/'runs',mid,ctx,'validate',8);check();checks=[]
  for spec,ref in zip(specs,refs):
   sid=scan.point_key(spec,ctx);p=R/'runs'/sid;q=json.loads((p/'result.json').read_text());actual=fields(p/'kernel.amdgcn') if (p/'kernel.amdgcn').exists() else None
   ok=bool(q.get('status')=='PASS' and q.get('loaded_native',{}).get('sha256')==cpu['build']['sha256'] and q['accuracy']['mismatches']==q['accuracy']['nonfinite']==0 and q['accuracy']['elements']==spec['tile'][0]*spec['tile'][1] and q['n_spills']==0 and actual==ref['fields'] and q['metadata']['shared']==ref['metadata']['shared'] and q['metadata']['num_warps']==ref['metadata']['num_warps'] and q['metadata'].get('hcu_split_b_slot_mmac_turn')==ref['metadata'].get('hcu_split_b_slot_mmac_turn') and q['metadata'].get('hcu_alias_ab_slots')==ref['metadata'].get('hcu_alias_ab_slots') and q['metadata'].get('wasp_partition_regs_actual')==ref['metadata'].get('wasp_partition_regs_actual') and code(p/'kernel.amdgcn')==code(pathlib.Path(ref['directory'])/'kernel.amdgcn'))
   checks.append(dict(id=sid,tile=spec['tile'],layout=spec['layout'],passed=ok,status=q.get('status'),accuracy=q.get('accuracy'),physical=actual,CPU_reference=ref['directory']))
  scan.write_json(R/(label+'-qualification.json'),dict(passed=all(x['passed'] for x in checks),checks=checks));assert all(x['passed'] for x in checks),checks
  return checks
 controls=measure(json.loads((R/'cpu-controls.json').read_text())['controls'],'default_controls')
 pilots=list(cpu['results']);assert len(pilots)==6 and all(x['config'].get('hcu_wasp_alias_ab_slots') is True and x['metadata']['shared']==65536 for x in pilots)
 measure(pilots,'alias_liveness_full_gold_pilots')
 features=measure(cpu['results'],'independent_B_slot_turn_features');check()
 scan.write_json(R/'native-qualification.json',dict(passed=True,measurement_context_id=ctx,controls=controls,features=features,source_hashes_unchanged=True,full_output_seed29_atol_rtol_1e3=True,single_TG=True,loop_counts=[16],full_split_slot_ring=True,performance_map_promotion_allowed=False))
 # All four short true-ring cases must pass before longer calibration lengths.
 measure(cpu['results'],'independent_B_slots_loops64',64)
 measure(cpu['results'],'independent_B_slots_loops32',32)
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
