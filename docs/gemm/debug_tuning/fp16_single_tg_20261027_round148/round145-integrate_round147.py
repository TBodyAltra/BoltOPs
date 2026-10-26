"""Record audited quota candidates, one-shot confirmations and resource alternatives."""
import pathlib,json,hashlib,copy,re,fcntl,os,time
R=pathlib.Path(__file__).resolve().parent;B=R.parent;I=B/'single-tg-pipeline-20261027-round141-alias-vgpr-quota-native';C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='8569d130b7bcbc013b6b551b96452a83fe825d58e0c90bae1faa2df9c5b33a17';choices=['best','best_no_spill','best_64k','best_64k_no_spill']
with (C/'map-integration.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(R/'integration-round147.json').exists() and sha(P)==parent
 analyses={}
 for root,points,nc in [(I,91,26),(R,31,6)]:
  e=json.loads((root/'job-exit.json').read_text());a=json.loads((root/'native-raw-audit.json').read_text())
  assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and a['passed'] and a['points']==points and a['raw_dispatches']==4*points and a['all_full_output_gold'] and a['all_MMOP_ratios']==1 and (a.get('all_physical_spills_zero') or a.get('all_feature_physical_spills_zero'))
  active=json.loads((root/'active-manifest.json').read_text());m=json.loads((root/('manifest-'+active['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
  analysis=json.loads((root/'analysis/pipeline-map.json').read_text());assert len(analysis['candidates'])==nc and analysis['measurement_context_id']==active['measurement_context_id'];analyses[str(root)]=analysis
 conf=json.loads((R/'confirmation.json').read_text());plan=json.loads((R/'confirmation-plan.json').read_text());assert conf['plan_sha256']==sha(R/'confirmation-plan.json') and len(conf['results'])==len(plan['selected'])==6
 assert plan['one_independent_series_per_initial_eligible_recipe'] and plan['no_extra_repeats_on_failure'] and plan['fresh_loop_order']==[32,16,64] and plan['resource_comparison_separate_from_performance_baseline']
 assert plan['initial_raw_audit_sha256']==sha(I/'native-raw-audit.json') and plan['initial_candidates_sha256']==sha(I/'pipeline-candidates.json')
 old=json.loads(P.read_text());new=copy.deepcopy(old);rows=[];changed=[];accepted=[];rejected=[]
 for c in conf['results']:
  selected=next(s for s in plan['selected'] if s['initial']['source_ids']==c['initial_source_ids']);seed=pathlib.Path(selected['resource_reference_CPU_directory']);seed_q=json.loads((seed/'result.json').read_text());seed_regs=seed_q['metadata']['wasp_partition_regs']
  assert c['aggregate_quota']==3*c['physical']['amdhsa_next_free_vgpr']<sum(seed_regs)==c['baseline_aggregate_quota']==504
  assert c['initial']['steady'] and c['initial']['mmac_theoretical_fraction']>=.98*c['baseline']['mmac_theoretical_fraction']
  valid=c['steady'] and c['mmac_theoretical_fraction']>=.98*c['baseline']['mmac_theoretical_fraction'];assert valid==c['resource_alternative_validated']
  alt=dict(c);alt.update(resource_reference_CPU_result_sha256=sha(seed/'result.json'),resource_reference_ISA_sha256=sha(seed/'kernel.amdgcn'),source_root_container=str(R),evidence_root_absolute=str(R/'runs'),scope='Single-TG compact-quota alternative with >=98% prior64KiB throughput in initial/fresh steady measurements. Quota compared to exact CPU seed, NOT a NoWASP map winner. No measured multiTG occupancy.',replaces_map_winner_as_resource_alternative=False)
  (accepted if valid else rejected).append(alt)
 for row in analyses[str(I)]['candidates']:
  x=copy.deepcopy(row);first=I/'runs'/x['source_ids'][0];physical={}
  for name in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
   m=re.search(r'\.'+name+r'\s*[:=]?\s*(\d+)',(first/'kernel.amdgcn').read_text());assert m;physical[name]=int(m[1])
  assert all(physical[k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
  x.update(measurement_context_id=analyses[str(I)]['measurement_context_id'],source_root_container=str(I),evidence_root_absolute=str(I/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(I/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(I/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=x['config']['LOOP_UNROLL_FACTOR'],executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,continuation_eligible_choices=[],native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',physical_resource_fields=physical,kernel_contract='Natural-stride FP16 singleTG fullBK128/FP32acc/FP16out. Frozen CPU133 GPU source/native95; compact quota configs qualified CPU140 with actual LLVM/ISA register descriptor, addresses/all64lane reads/data+MMACturn ownership. AliasAB64KiB, splitB slots, slot_turn2, waitmask per config, originaldata guards retained; WASP/WDRA4/4/4 stage1/privateSB/UF1/split2/Afirst/noasyncAPI/noBPP. Rawgold/MMOP1/CU0/allphysicalspill0/fullASM16/32/64; one independent fresh series before promotion. Quota resource alternatives separately qualified vs exact CPUseed and performance-map throughput, not multiTG occupancy.')
  c=next((v for v in conf['results'] if v['initial_source_ids']==x['source_ids']),None)
  if c:
   assert c['initial_context']==x['measurement_context_id'] and c['fresh_context']!=c['initial_context'] and c['initial']['mmac_theoretical_fraction']==x['mmac_theoretical_fraction']
   eligible=[]
   for key in c['eligible_choices']:
    prior=next(v for v in old[key] if v['tile']==x['tile'] and v['layout']==x['layout']);assert prior==c['baselines'][key]
    if c['steady'] and x['valid_metric'] and x['identical_binary_across_loop_counts'] and x['intermediate_residual_fraction_of_long_time']<=.02 and x['mmac_theoretical_fraction']/prior['mmac_theoretical_fraction']-1>=.03 and c['mmac_theoretical_fraction']/prior['mmac_theoretical_fraction']-1>=.03:eligible.append(key)
   x.update(independent_repeat_fraction=c['mmac_theoretical_fraction'],independent_repeat_residual=c['intermediate_residual_fraction_of_long_time'],confirmation_source_ids=c['fresh_source_ids'],confirmation_root_absolute=str(R/'runs'),confirmation_measurement_context_id=c['fresh_context'],independent_confirmation_sha256=sha(R/'confirmation.json'),independent_confirmation_raw_audit_sha256=sha(R/'native-raw-audit.json'),independent_resource_alternative_validated=c['resource_alternative_validated'],continuation_selection_eligible=bool(eligible),continuation_eligible_choices=eligible)
   for key in eligible:
    assert key in ['best_64k','best_64k_no_spill'] and x['tile']==[64,256,128] and x['layout']=='TT' and x['shared_bytes']<=65536
    idx=next(i for i,v in enumerate(new[key]) if v['tile']==x['tile'] and v['layout']==x['layout']);prior=new[key][idx];new[key][idx]=copy.deepcopy(x)
    changed.append(dict(choice=key,tile=x['tile'],layout=x['layout'],prior_fraction=prior['mmac_theoretical_fraction'],initial_fraction=x['mmac_theoretical_fraction'],fresh_fraction=c['mmac_theoretical_fraction'],fresh_residual=c['intermediate_residual_fraction_of_long_time'],source_ids=x['source_ids'],confirmation_source_ids=c['fresh_source_ids']))
  if not x['continuation_selection_eligible']:x['selection_rejection_reason']='Not an independently confirmed >=3% performance improvement; resource alternatives recorded separately; no extra repeats'
  rows.append(x)
 assert len(rows)==26 and new['best']==old['best'] and new['best_no_spill']==old['best_no_spill'] and sum(len(new[k]) for k in choices)==503
 assert all(x['tile']!=[128,128,128] for key in choices for x in new[key])
 altpath=C/'published/compact-quota-resource-alternatives-round145.json';assert not altpath.exists()
 alt_report=dict(passed=True,parent_map_sha256=parent,validated_recipes=accepted,rejected_recipes=rejected,source_initial_root=str(I),source_fresh_root=str(R),initial_raw_audit_sha256=sha(I/'native-raw-audit.json'),fresh_raw_audit_sha256=sha(R/'native-raw-audit.json'),confirmation_sha256=sha(R/'confirmation.json'),initial_and_fresh_min_relative_throughput=.98,all_physical_spills_zero=True,resource_reference='Exact original CPU seed quota; performance reference is prior best64 map row; separate baselines',one_series_per_recipe=True,no_multi_TG_occupancy_claim=True,script_sha256=sha(pathlib.Path(__file__)))
 altpath.write_text(json.dumps(alt_report,indent=2,allow_nan=False)+'\n')
 new['candidates']+=rows;new['paired_candidates']=len(new['candidates'])
 for root in [I,R]:
  context=analyses[str(root)]['measurement_context_id'];assert context not in new['measurement_contexts'];new['measurement_contexts'].append(context)
  assert not any(s['container']==str(root) for s in new['source_roots']);new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
 qual=dict(passed=True,parent_map_sha256=parent,initial_root=str(I),fresh_root=str(R),raw_dispatches_audited=488,candidate_rows_added=26,changed_choices=changed,primary_choices_unchanged=True,all_physical_spills_zero=True,resource_alternatives_path=str(altpath),resource_alternatives_sha256=sha(altpath),validated_resource_alternatives=len(accepted),rejected_resource_alternatives=len(rejected),one_fresh_series_only=True,confirmation_sha256=sha(R/'confirmation.json'),script_sha256=sha(pathlib.Path(__file__)))
 new['continuation_round147_qualification']=qual;new['generated_remote_epoch']=time.time()
 backup=C/'published/natural-stride-map-round139-before-round147.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent
 tmp=P.with_name(P.name+'.round147.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P)
 proof=dict(qual,new_map_sha256=sha(P),unchanged_choice_records=503-len(changed),deferred_128cube_preserved=True)
 (R/'integration-round147.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
