"""One-shot promotion of independently confirmed single-TG late-wait pipelines."""
import pathlib,json,hashlib,copy,fcntl,time,re,os
R=pathlib.Path(__file__).resolve().parent;B=R.parent;I=B/'single-tg-pipeline-20261027-round111-late-slot-turn-native';C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='e19a025cc5b98810ebc1e312e90d65dc0ac25740fe148bdc3565b0d39b7f90f0'
with (C/'map-integration.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(R/'integration-round113.json').exists() and sha(P)==parent
 for root in [I,R]:
  e=json.loads((root/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and json.loads((root/'state.json').read_text())['phase']=='FINISHED' and json.loads((root/'native-raw-audit.json').read_text())['passed']
  manifest=json.loads((root/('manifest-'+json.loads((root/'active-manifest.json').read_text())['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in manifest['hashes'].items())
 conf=json.loads((R/'confirmation.json').read_text());plan=json.loads((R/'confirmation-plan.json').read_text());assert conf['plan_sha256']==sha(R/'confirmation-plan.json') and plan['parent_map_sha256']==parent and len(conf['results'])==len(plan['selected'])==5
 confirmed=[x for x in conf['results'] if x['confirmed']];assert confirmed and conf['confirmed']==len(confirmed)
 old=json.loads(P.read_text());new=copy.deepcopy(old);initial=json.loads((I/'analysis/pipeline-map.json').read_text());fresh=json.loads((R/'analysis/pipeline-map.json').read_text());assert len(initial['candidates'])==14 and len(fresh['candidates'])==5
 def key(x):return tuple(x['tile']),x['layout']
 cmap={key(x):x for x in confirmed};assert len(cmap)==len(confirmed)
 for c in confirmed:
  assert c['steady'] and c['relative_fresh_gain']>=.03 and c['intermediate_residual_fraction_of_long_time']<=.02 and c['executed_to_logical_mmac_ratio']==1 and c['initial']['steady'] and c['initial']['mmac_theoretical_fraction']/c['baseline']['mmac_theoretical_fraction']>=1.03
  f=next(x for x in fresh['candidates'] if x['source_ids']==c['fresh_source_ids']);assert f['mmac_theoretical_fraction']==c['mmac_theoretical_fraction']
 changes=[];newrows=[];winners={}
 for row in initial['candidates']:
  x=copy.deepcopy(row);x.update(measurement_context_id=initial['measurement_context_id'],source_root_container=str(I),evidence_root_absolute=str(I/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(I/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(I/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=1,executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',kernel_contract='Natural-stride singleTG FP16/FP16 FP32acc FP16out, BK128 OneA full A capture/A-first, B K64+K64 copies in independent parent-layout slots A0/1 B0 2/3 B1 4/5. MMAC turns6/7 twice perBK; wait_mask bits move corresponding token wait after B-ready/read/fullLGKMdrain/emptyrelease, before first fragment MMAC; other waits remain before B-ready. Same copies/read mapping/math; all12join2 before8IDinvalidate. WDRA4/4/4 stage1 private singlebuffer UF1 split2, no BPP/async API. native95 parent-view MLS fix3fc2 required. Full-output seed29 gold, fullASM16/32/64, rawMMOP1, all physical spills0, residual<=2%, initial+independentfresh gain>=3% before promotion.')
  cases=[json.loads((I/'runs'/sid/'result.json').read_text()) for sid in x['source_ids']];assert all(q['status']=='PASS' and q['accuracy']['mismatches']==q['accuracy']['nonfinite']==q['n_spills']==0 for q in cases)
  isa=(I/'runs'/x['source_ids'][0]/'kernel.amdgcn').read_text();phys={}
  for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
   m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',isa);assert m;phys[n]=int(m[1])
  assert all(phys[n]==0 for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']);x['physical_resource_fields']=phys
  c=cmap.get(key(x))
  if c and x['source_ids']==c['initial_source_ids']:
   assert x['config']==c['config'] and x['mmac_theoretical_fraction']==c['initial']['mmac_theoretical_fraction']
   x.update(continuation_selection_eligible=True,independent_repeat_fraction=c['mmac_theoretical_fraction'],independent_repeat_residual=c['intermediate_residual_fraction_of_long_time'],confirmation_source_ids=c['fresh_source_ids'],confirmation_root_absolute=str(R/'runs'),confirmation_measurement_context_id=c['fresh_context'],independent_confirmation_sha256=sha(R/'confirmation.json'),independent_confirmation_raw_audit_sha256=sha(R/'native-raw-audit.json'));winners[key(x)]=x
  else:x['selection_rejection_reason']='Nonsteady, insufficient material initial gain, not the highest eligible recipe for this block/layout, or independent confirmation gate failed; no promotion'
  newrows.append(x)
 assert winners.keys()==cmap.keys()
 for ident,winner in winners.items():
  c=cmap[ident]
  for name in ['best','best_no_spill']:
   matches=[i for i,x in enumerate(old[name]) if key(x)==ident];assert len(matches)==1;i=matches[0];assert old[name][i]['config']==c['baseline']['config'] and old[name][i]['cycles_per_bk_slope']==c['baseline']['cycles_per_bk_slope'];new[name][i]=copy.deepcopy(winner);changes.append(dict(choice=name,tile=winner['tile'],layout=winner['layout'],old_fraction=old[name][i]['mmac_theoretical_fraction'],new_fraction=winner['mmac_theoretical_fraction'],independent_fraction=c['mmac_theoretical_fraction']))
 unchanged=0
 for name in ['best','best_no_spill','best_64k','best_64k_no_spill']:
  assert len(old[name])==len(new[name])
  for a,b in zip(old[name],new[name]):
   if a==b:unchanged+=1
   else:assert name in ['best','best_no_spill'] and key(a)==key(b) and key(b) in winners
 assert unchanged==503-2*len(winners) and len(changes)==2*len(winners)
 new['candidates']+=newrows;new['paired_candidates']=len(new['candidates'])
 for context in [initial['measurement_context_id'],fresh['measurement_context_id']]:
  assert context not in new['measurement_contexts'];new['measurement_contexts'].append(context)
 for root in [I,R]:new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
 new['continuation_round113_qualification']=dict(passed=True,initial_context=initial['measurement_context_id'],fresh_context=fresh['measurement_context_id'],native_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',CPU_source=str(B/'single-tg-pipeline-20261027-round110-late-slot-turn-screen'),CPU_controls=6,CPU_attempted_features=18,CPU_accepted_features=14,CPU_VGPR_budget_exclusions=4,CPU_address_and_ownership_audit=str(B/'single-tg-pipeline-20261027-round110-late-slot-turn-screen/ownership-isa-audit.json'),initial_native_audit=str(I/'native-raw-audit.json'),fresh_native_audit=str(R/'native-raw-audit.json'))
 new['continuation_round113_confirmation']=conf;new['round113_selection_policy']='Late token waits after B capture/emptyrelease before corresponding fragment MMAC. 14 initial recipes and6controls, fullgold/MMOP1/allphysicalspill0/fullASM16/32/64. Highest initial steady >=3% perblock/layout only; five predeclared independentfresh32/16/64series, no extra repeats. Promote only fresh steady>=3%. Preserve every other choice, all64KiB alternatives and128cube deferred.';new['round113_parent_map_sha256']=parent;new['round113_statuses']=dict(initial={'PASS':48},fresh={'PASS':6+3*len(plan['selected'])});new['generated_remote_epoch']=time.time()
 assert len(new['best'])==126 and sum(len(new[k]) for k in ['best','best_no_spill','best_64k','best_64k_no_spill'])==503
 backup=C/'published/natural-stride-map-round108-before-round113.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent
 data=(json.dumps(new,indent=2,allow_nan=False)+'\n').encode();tmp=P.with_name(P.name+'.round113.tmp');tmp.write_bytes(data);os.replace(tmp,P)
 proof=dict(passed=True,parent_map_sha256=parent,new_map_sha256=sha(P),initial_context=initial['measurement_context_id'],fresh_context=fresh['measurement_context_id'],changed_choices=changes,unchanged_choices=unchanged,profiles=126,total_choices=503,confirmed_profiles=len(winners),main_steady_count=sum(len(x.get('loop_counts',[]))>=3 and x.get('intermediate_residual_fraction_of_long_time',1)<=.02 and 0<x['mmac_theoretical_fraction']<=1 for x in new['best']),main_at_least80percent=sum(x['mmac_theoretical_fraction']>=.8 for x in new['best']),deferred_128cube_preserved=True,raw_native_dispatches_audited=4*(48+6+3*len(plan['selected'])),source_rows_added=14,script_sha256=sha(pathlib.Path(__file__)))
 (R/'integration-round113.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
