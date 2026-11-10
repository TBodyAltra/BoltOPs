"""Integrate audited130/135 negatives and136/137 independently confirmed choices once."""
import pathlib,json,hashlib,copy,re,fcntl,os,time
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent='716e900cf10d72748f19858b9d10ccb43e035d9d917f394ad7b3d2a17c572612'
I=B/'single-tg-pipeline-20261027-round136-alias-late-turn-native'
inputs=[(B/'single-tg-pipeline-20261027-round130-alias-b1-early-native',12,3),(B/'single-tg-pipeline-20261027-round135-bm64-slot-geometry-native',24,7),(I,68,20),(R,23,5)]
choices=['best','best_no_spill','best_64k','best_64k_no_spill']
with (C/'map-integration.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(R/'integration-round139.json').exists() and sha(P)==parent
 audits={};analyses={}
 for root,npoints,ncandidates in inputs:
  e=json.loads((root/'job-exit.json').read_text());a=json.loads((root/'native-raw-audit.json').read_text())
  assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and a['passed'] and a['points']==npoints and a['raw_dispatches']==4*npoints
  assert a['all_full_output_gold'] and a['all_MMOP_ratios']==1 and (a.get('all_physical_spills_zero') or a.get('all_feature_physical_spills_zero'))
  active=json.loads((root/'active-manifest.json').read_text());m=json.loads((root/('manifest-'+active['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
  analysis=json.loads((root/'analysis/pipeline-map.json').read_text());assert len(analysis['candidates'])==ncandidates and analysis['measurement_context_id']==active['measurement_context_id']
  audits[str(root)]=a;analyses[str(root)]=analysis
 conf=json.loads((R/'confirmation.json').read_text());plan=json.loads((R/'confirmation-plan.json').read_text())
 assert conf['plan_sha256']==sha(R/'confirmation-plan.json') and len(conf['results'])==len(plan['selected'])==5
 assert plan['one_independent_series_per_initial_eligible_recipe'] and plan['no_extra_repeats_on_failure'] and plan['fresh_loop_order']==[32,16,64]
 assert plan['initial_raw_audit_sha256']==sha(I/'native-raw-audit.json') and plan['initial_candidates_sha256']==sha(I/'pipeline-candidates.json')
 old=json.loads(P.read_text());new=copy.deepcopy(old);rows=[];changed=[]
 for root,_,_ in inputs[:-1]:
  analysis=analyses[str(root)]
  for row in analysis['candidates']:
   x=copy.deepcopy(row);first=root/'runs'/x['source_ids'][0];physical={}
   for name in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
    m=re.search(r'\.'+name+r'\s*[:=]?\s*(\d+)',(first/'kernel.amdgcn').read_text());assert m;physical[name]=int(m[1])
   assert all(physical[k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
   x.update(measurement_context_id=analysis['measurement_context_id'],source_root_container=str(root),evidence_root_absolute=str(root/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(root/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(root/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=x['config']['LOOP_UNROLL_FACTOR'],executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,continuation_eligible_choices=[],native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',physical_resource_fields=physical,native_raw_audit_sha256=sha(root/'native-raw-audit.json'),kernel_contract='Natural-stride FP16 singleTG fullBK128/FP32acc/FP16out; frozen source worktree and native95 required. MLS private WASP/WDRA4/4/4 stage1 UF1 split2, A-first, no asyncAPI/BPP. Source-specific fullB/splitB/alias/earlyB1/late-turn options are explicit in config. Fullgold, raw actual MMOP ratio1/CU0, same fullASM16/32/64, physicalscratch/SGPR/VGPRspill0. Alias late-turn initial136 and independent fresh137 must both be steady and improve eligible choice>=3%; no MNK tile selection or multiTG occupancy claim.')
   c=next((c for c in conf['results'] if root==I and c['initial_source_ids']==x['source_ids']),None)
   if c:
    assert c['initial_context']==analysis['measurement_context_id'] and c['fresh_context']!=c['initial_context']
    assert c['initial']['mmac_theoretical_fraction']==x['mmac_theoretical_fraction']
    assert c['steady']==(c['intermediate_residual_fraction_of_long_time']<=.02 and 0<c['mmac_theoretical_fraction']<=1)
    eligible=[]
    for choice in c['eligible_choices']:
     assert choice in choices
     prior=next(v for v in old[choice] if v['tile']==x['tile'] and v['layout']==x['layout'])
     assert prior==c['baselines'][choice]
     if c['steady'] and x['valid_metric'] and x['identical_binary_across_loop_counts'] and len(x['loop_counts'])==3 and x['intermediate_residual_fraction_of_long_time']<=.02 and x['mmac_theoretical_fraction']/prior['mmac_theoretical_fraction']-1>=.03 and c['mmac_theoretical_fraction']/prior['mmac_theoretical_fraction']-1>=.03:eligible.append(choice)
    x.update(independent_repeat_fraction=c['mmac_theoretical_fraction'],independent_repeat_residual=c['intermediate_residual_fraction_of_long_time'],confirmation_source_ids=c['fresh_source_ids'],confirmation_root_absolute=str(R/'runs'),confirmation_measurement_context_id=c['fresh_context'],independent_confirmation_sha256=sha(R/'confirmation.json'),independent_confirmation_raw_audit_sha256=sha(R/'native-raw-audit.json'),continuation_selection_eligible=bool(eligible),continuation_eligible_choices=eligible)
    for choice in eligible:
     assert choice in ['best_64k','best_64k_no_spill'] and x['shared_bytes']<=65536
     idx=next(i for i,v in enumerate(new[choice]) if v['tile']==x['tile'] and v['layout']==x['layout']);prior=new[choice][idx]
     new[choice][idx]=copy.deepcopy(x);changed.append(dict(choice=choice,tile=x['tile'],layout=x['layout'],prior_fraction=prior['mmac_theoretical_fraction'],initial_fraction=x['mmac_theoretical_fraction'],fresh_fraction=c['mmac_theoretical_fraction'],fresh_residual=c['intermediate_residual_fraction_of_long_time'],source_ids=x['source_ids'],confirmation_source_ids=c['fresh_source_ids']))
   if not x['continuation_selection_eligible']:x['selection_rejection_reason']='No initial steady >=3% eligible gain, or predeclared independent series did not meet steady >=3% gain; no extra repeats'
   rows.append(x)
 assert len(rows)==30
 assert new['best']==old['best'] and new['best_no_spill']==old['best_no_spill']
 assert sum(len(new[k]) for k in choices)==503
 assert all(x['tile']!=[128,128,128] for k in choices for x in new[k])
 new['candidates']+=rows;new['paired_candidates']=len(new['candidates'])
 for root,_,_ in inputs:
  context=analyses[str(root)]['measurement_context_id'];assert context not in new['measurement_contexts'];new['measurement_contexts'].append(context)
  assert not any(v['container']==str(root) for v in new['source_roots']);new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
 qualification=dict(passed=True,parent_map_sha256=parent,initial_roots=[str(x[0]) for x in inputs[:-1]],fresh_root=str(R),raw_dispatches_audited=sum(4*x[1] for x in inputs),candidate_rows_added=30,changed_choices=changed,primary_choices_unchanged=True,all_feature_physical_spills_zero=True,independent_series_once_each=True,rejected_evidence_retained=True,raw_audit_sha256={str(root):sha(root/'native-raw-audit.json') for root,_,_ in inputs},confirmation_sha256=sha(R/'confirmation.json'),script_sha256=sha(pathlib.Path(__file__)))
 new['continuation_round139_qualification']=qualification;new['generated_remote_epoch']=time.time()
 backup=C/'published/natural-stride-map-round125-before-round139.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent
 tmp=P.with_name(P.name+'.round139.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P)
 proof=dict(qualification,new_map_sha256=sha(P),unchanged_choice_records=503-len(changed),deferred_128cube_preserved=True)
 (R/'integration-round139.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n');print(json.dumps(proof))
