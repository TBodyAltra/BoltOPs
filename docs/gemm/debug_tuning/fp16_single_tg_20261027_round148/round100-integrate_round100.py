"""One-shot merge of the independently confirmed B0/B1 pipeline into the map."""
import pathlib,json,hashlib,copy,fcntl,time,re,os
R=pathlib.Path(__file__).resolve().parent;B=R.parent;I=B/'single-tg-pipeline-20261027-round99-split-b-slots-native';C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent='197a4de4d116178cabaf7c76f0babac6465b70b25f8172e1c85e7ce77351cc49'
with (C/'map-integration.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(R/'integration-round100.json').exists() and sha(P)==parent
 for root in [I,R]:
  assert json.loads((root/'job-exit.json').read_text())['returncode']==0 and json.loads((root/'native-raw-audit.json').read_text())['passed']
  manifest=json.loads((root/('manifest-'+json.loads((root/'active-manifest.json').read_text())['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
 conf=json.loads((R/'confirmation.json').read_text());raw=json.loads((R/'native-raw-audit.json').read_text());assert conf['confirmed']==1 and len(conf['results'])==1 and raw['comparisons'][0]['confirmed'];c=conf['results'][0]
 assert c['tile']==[256,256,128] and c['layout']=='TN' and c['steady'] and c['relative_fresh_gain']>=.03 and c['executed_to_logical_mmac_ratio']==1
 assert c['initial']['steady'] and c['initial']['mmac_theoretical_fraction']/c['baseline']['mmac_theoretical_fraction']>=1.03
 old=json.loads(P.read_text());new=copy.deepcopy(old);initial=json.loads((I/'analysis/pipeline-map.json').read_text());fresh=json.loads((R/'analysis/pipeline-map.json').read_text());assert len(initial['candidates'])==4 and len(fresh['candidates'])==1
 assert fresh['candidates'][0]['mmac_theoretical_fraction']==c['mmac_theoretical_fraction']
 def key(x):return tuple(x['tile']),x['layout']
 changes=[];target=([256,256,128],'TN');newrows=[]
 for x in initial['candidates']:
  x=copy.deepcopy(x);x.update(measurement_context_id=initial['measurement_context_id'],source_root_container=str(I),evidence_root_absolute=str(I/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(I/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(I/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=1,executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',kernel_contract='Natural-stride singleTG FP16/FP16 FP32acc FP16out, BK128 full A capture; B global K64+K64 copies into disjoint parent-layout subviews with independent empty/ready ownership A0/1 B0 2/3 B1 4/5. Same full parent LDS allocation and actual MLS global work; data releases drain LDS reads. One producer and two consumers4/4/4, WDRA stage1 private singlebuffer, A-first, no MMAC turns, no BPP/async_copy API. Parent-view MLS lowering requires native95 fix3fc2 (not legacy622e). Full output seed29 gold, fullASM16/32/64, <=2% residual, rawMMOP ratio1, independentfresh>=3% before promotion.')
  cases=[json.loads((I/'runs'/sid/'result.json').read_text()) for sid in x['source_ids']]
  assert all(q['status']=='PASS' and q['accuracy']['mismatches']==q['accuracy']['nonfinite']==q['n_spills']==0 for q in cases)
  isa=(I/'runs'/x['source_ids'][0]/'kernel.amdgcn').read_text();phys={}
  for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
   m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',isa);assert m;phys[n]=int(m[1])
  assert all(phys[n]==0 for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']);x['physical_resource_fields']=phys
  if x['tile']==target[0] and x['layout']==target[1]:
   assert x['source_ids']==c['initial_source_ids'] and x['config']==c['config'] and x['mmac_theoretical_fraction']==c['initial']['mmac_theoretical_fraction']
   x.update(continuation_selection_eligible=True,independent_repeat_fraction=c['mmac_theoretical_fraction'],independent_repeat_residual=c['intermediate_residual_fraction_of_long_time'],confirmation_source_ids=c['fresh_source_ids'],confirmation_root_absolute=str(R/'runs'),confirmation_measurement_context_id=c['fresh_context'],independent_confirmation_sha256=sha(R/'confirmation.json'),independent_confirmation_raw_audit_sha256=sha(R/'native-raw-audit.json'))
  else:x['selection_rejection_reason']='Initial residual exceeds2%' if x['intermediate_residual_fraction_of_long_time']>.02 else 'Initial material gain below3%; no independent repeat requested'
  newrows.append(x)
 winner=next(x for x in newrows if x['tile']==target[0] and x['layout']==target[1]);assert winner['shared_bytes']==131072 and winner['n_spills']==0
 for name in ['best','best_no_spill']:
  matches=[i for i,x in enumerate(old[name]) if x['tile']==target[0] and x['layout']==target[1]];assert len(matches)==1;i=matches[0];assert old[name][i]['config']==c['baseline']['config'] and old[name][i]['cycles_per_bk_slope']==c['baseline']['cycles_per_bk_slope'];new[name][i]=copy.deepcopy(winner);changes.append(dict(choice=name,tile=target[0],layout=target[1],old_fraction=old[name][i]['mmac_theoretical_fraction'],new_fraction=winner['mmac_theoretical_fraction'],independent_fraction=c['mmac_theoretical_fraction']))
 # All 503 selected choice records are compared, including constrained fallbacks.
 unchanged=0
 for name in ['best','best_no_spill','best_64k','best_64k_no_spill']:
  assert len(old[name])==len(new[name])
  for a,b in zip(old[name],new[name]):
   if a==b:unchanged+=1
   else:assert name in ['best','best_no_spill'] and key(a)==key(winner)==key(b)
 assert unchanged==501 and len(changes)==2
 new['candidates']+=newrows;new['paired_candidates']=len(new['candidates'])
 for context in [initial['measurement_context_id'],fresh['measurement_context_id']]:
  assert context not in new['measurement_contexts'];new['measurement_contexts'].append(context)
 for root in [I,R]:new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
 new['continuation_round100_qualification']=dict(passed=True,initial_context=initial['measurement_context_id'],fresh_context=fresh['measurement_context_id'],native_sha256=winner['native_artifact_sha256'],old_full_root_controls='CPU95 eleven normalizedISA/resources exact; native97 four full-root gold controls; CPU96/97 repeated controls exact',subview_and_ring_controls='CPU96 full-parent source/destination equality; native97 eight gold+rawMMOP; CPU98 four root controls; native99 four coupled-copy controls and twelve ring lengths; native100 four coupled-copy controls and independent three ring lengths',CPU_address_and_ownership_audit=str(B/'single-tg-pipeline-20261027-round98-split-b-slots-cpu/ownership-isa-audit.json'),initial_native_audit=str(I/'native-raw-audit.json'),fresh_native_audit=str(R/'native-raw-audit.json'))
 new['continuation_round100_confirmation']=conf;new['round100_selection_policy']='All four initial profiles evaluated. Require same fullASM16/32/64, no physicalspill, fullgold, rawMMOP ratio1, residual<=2%, >=3% initial+one predeclared independentfresh gain; no extra repeats for failing residual. Only cube256x256x128TN confirmed; initialrows for otherprofiles retained as rejected evidence. Preserve128cube deferred and all64KiB alternatives.'
 new['round100_parent_map_sha256']=parent;new['round100_statuses']=dict(initial={'PASS':16},fresh={'PASS':7});new['generated_remote_epoch']=time.time()
 assert len(new['best'])==126 and len(new['best_64k'])+len(new['best_64k_no_spill'])+len(new['best'])+len(new['best_no_spill'])==503
 data=(json.dumps(new,indent=2,allow_nan=False)+'\n').encode();backup=C/'published/natural-stride-map-round93-before-round100.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent
 t=P.with_name(P.name+'.round100.tmp');t.write_bytes(data);os.replace(t,P)
 proof=dict(passed=True,parent_map_sha256=parent,new_map_sha256=sha(P),initial_context=initial['measurement_context_id'],fresh_context=fresh['measurement_context_id'],changed_choices=changes,unchanged_choices=501,profiles=126,total_choices=503,main_steady_count=sum(len(x.get('loop_counts',[]))>=3 and x.get('intermediate_residual_fraction_of_long_time',1)<=.02 and 0<x['mmac_theoretical_fraction']<=1 for x in new['best']),deferred_128cube_preserved=True,raw_native_dispatches_audited=92,source_rows_added=4,script_sha256=sha(pathlib.Path(__file__)))
 (R/'integration-round100.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
