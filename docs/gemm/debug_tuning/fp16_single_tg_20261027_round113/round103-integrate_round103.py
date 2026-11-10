"""One-shot integrate independently confirmed skinny per-K64 MMAC turns."""
import pathlib,json,hashlib,copy,fcntl,re,os,time
R=pathlib.Path(__file__).resolve().parent;B=R.parent;I=B/'single-tg-pipeline-20261027-round102-slot-turns-native';C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='57ebc467b839bcfc53ca6052cd4e833a20c54bb294bfa9b9d3b0aee7a46427f5'
with (C/'map-integration.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert sha(P)==parent and not(R/'integration-round103.json').exists()
 for root in [I,R]:
  assert json.loads((root/'job-exit.json').read_text())['returncode']==0 and json.loads((root/'native-raw-audit.json').read_text())['passed'];m=json.loads((root/('manifest-'+json.loads((root/'active-manifest.json').read_text())['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
 conf=json.loads((R/'confirmation.json').read_text());assert conf['confirmed']==2 and len(conf['results'])==2
 targets={ (tuple(c['tile']),c['layout']):c for c in conf['results']};assert set(targets)=={((32,256,128),'TT'),((32,256,128),'TN')}
 for c in targets.values():assert c['confirmed'] and c['steady'] and c['relative_fresh_gain']>=.03 and c['executed_to_logical_mmac_ratio']==1 and c['config']['hcu_wasp_split_b_slot_turn']==2
 old=json.loads(P.read_text());new=copy.deepcopy(old);initial=json.loads((I/'analysis/pipeline-map.json').read_text());fresh=json.loads((R/'analysis/pipeline-map.json').read_text());assert len(initial['candidates'])==8 and len(fresh['candidates'])==2
 rows=[];winners={}
 for row in initial['candidates']:
  x=copy.deepcopy(row);key=(tuple(x['tile']),x['layout']);c=targets.get(key);confirmed=bool(c and x['source_ids']==c['initial_source_ids'])
  x.update(measurement_context_id=initial['measurement_context_id'],source_root_container=str(I),evidence_root_absolute=str(I/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(I/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(I/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=1,executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=confirmed,native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',kernel_contract='Natural-stride singleTG FP16/FP16 FP32acc FP16out. OneA fullBK128 register capture and independent B0/B1 globalK64 slots, data abarriers0..5. Separate consumer MMAC issue turns on6/7: mode2 waits before matching B0/B1 ready, signals after each K64 dot with scheduler fences. Full parent LDS allocation/global MLS work/local read mapping/math unchanged; data release drains LGKM, all12-wavejoin2 before8-ID teardown. One producer/two consumers4/4/4 WDRA stage1 earlyFalse private singlebuffer; A-first; no BPP/async_copy API. Native95 parent-view MLS lowering3fc2 required. Fullgold, samefullASM16/32/64, residual<=2%, rawMMOP ratio1, >=3%initial+one predeclared independentfresh gate beforepromotion.')
  phys={};isa=(I/'runs'/x['source_ids'][0]/'kernel.amdgcn').read_text()
  for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
   z=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',isa);assert z;phys[n]=int(z[1])
  assert x['n_spills']==0 and all(phys[n]==0 for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']);x['physical_resource_fields']=phys
  if confirmed:
   assert x['config']==c['config'] and x['mmac_theoretical_fraction']==c['initial']['mmac_theoretical_fraction'] and x['shared_bytes']==73728 and x['intermediate_residual_fraction_of_long_time']<=.02
   x.update(independent_repeat_fraction=c['mmac_theoretical_fraction'],independent_repeat_residual=c['intermediate_residual_fraction_of_long_time'],confirmation_source_ids=c['fresh_source_ids'],confirmation_root_absolute=str(R/'runs'),confirmation_measurement_context_id=c['fresh_context'],independent_confirmation_sha256=sha(R/'confirmation.json'),independent_confirmation_raw_audit_sha256=sha(R/'native-raw-audit.json'));winners[key]=x
  else:x['selection_rejection_reason']='Nonsteady initial residual or no material gain over current selected baseline; no fresh repeat requested'
  rows.append(x)
 assert len(winners)==2;changes=[]
 for name in ['best','best_no_spill']:
  for i,a in enumerate(old[name]):
   key=(tuple(a['tile']),a['layout'])
   if key not in winners:continue
   c=targets[key];assert a['config']==c['baseline']['config'] and a['cycles_per_bk_slope']==c['baseline']['cycles_per_bk_slope'];new[name][i]=copy.deepcopy(winners[key]);changes.append(dict(choice=name,tile=a['tile'],layout=a['layout'],old_fraction=a['mmac_theoretical_fraction'],old_residual=a['intermediate_residual_fraction_of_long_time'],new_fraction=winners[key]['mmac_theoretical_fraction'],independent_fraction=c['mmac_theoretical_fraction']))
 same=0
 for name in ['best','best_no_spill','best_64k','best_64k_no_spill']:
  assert len(new[name])==len(old[name])
  for a,b in zip(old[name],new[name]):
   if a==b:same+=1
   else:assert name in ['best','best_no_spill'] and (tuple(a['tile']),a['layout']) in winners
 assert len(changes)==4 and same==499 and len(new['best'])==126
 new['candidates']+=rows;new['paired_candidates']=len(new['candidates'])
 for context in [initial['measurement_context_id'],fresh['measurement_context_id']]:assert context not in new['measurement_contexts'];new['measurement_contexts'].append(context)
 for root in [I,R]:new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
 new['continuation_round103_qualification']=dict(passed=True,initial_context=initial['measurement_context_id'],fresh_context=fresh['measurement_context_id'],native_sha256=next(iter(winners.values()))['native_artifact_sha256'],CPU_screen_and_protocol=str(B/'single-tg-pipeline-20261027-round101-slot-turns-cpu/ownership-isa-audit.json'),CPU_defaultFalse_controls=4,CPU_turn_features=8,individual_wave_model='Data0..5,turn6/7 counts4/role; finite8consumerwaves2generations plus phase-wrap controls and injected negative protocols; actualLLVM/ISA before corresponding Bready waits and MMAC issue before each turn signal; drains+all12join2/8-ID teardown.',initial_native_audit=str(I/'native-raw-audit.json'),fresh_native_audit=str(R/'native-raw-audit.json'))
 new['continuation_round103_confirmation']=conf;new['round103_selection_policy']='Eight initial turn candidates, allrawMMOP1/zeroallphysical/fullgold/samefullASM16/32/64; initialresidual<=2% and>=3%gain nominateonly32x256TT/TNturn2. One predeclared independent32/16/64 series pernominee, no extra repeats; bothconfirmed. Preservecube256TN round100 winner, all64KiBchoices and128cube deferred.'
 new['round103_parent_map_sha256']=parent;new['round103_statuses']=dict(initial={'PASS':28},fresh={'PASS':10});new['generated_remote_epoch']=time.time();steady=sum(len(x.get('loop_counts',[]))>=3 and x.get('intermediate_residual_fraction_of_long_time',1)<=.02 and 0<x['mmac_theoretical_fraction']<=1 for x in new['best']);assert steady==126
 backup=C/'published/natural-stride-map-round100-before-round103.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent;t=P.with_name(P.name+'.round103.tmp');t.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(t,P)
 proof=dict(passed=True,parent_map_sha256=parent,new_map_sha256=sha(P),changed_choices=changes,unchanged_choices=499,profiles=126,total_choices=503,main_steady_count=126,initial_context=initial['measurement_context_id'],fresh_context=fresh['measurement_context_id'],raw_dispatches_audited=152,candidates_added=8,all64k_choices_unchanged=True,round100_cube_winner_unchanged=True,script_sha256=sha(pathlib.Path(__file__)));(R/'integration-round103.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
