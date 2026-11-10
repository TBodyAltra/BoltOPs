"""Promote independently confirmed64KiB alias recipes, preserving main choices."""
import pathlib,json,hashlib,copy,fcntl,time,re,os
R=pathlib.Path(__file__).resolve().parent;B=R.parent;I=B/'single-tg-pipeline-20261027-round117-alias-ab-slots-native-retry';C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='642f0afce2f7a214933aa64a357df85b7d0486f0906cd748a8a5d0b467049374'
with (C/'map-integration.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(R/'integration-round119.json').exists() and sha(P)==parent
 for root in [I,R]:
  e=json.loads((root/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and json.loads((root/'state.json').read_text())['phase']=='FINISHED' and json.loads((root/'native-raw-audit.json').read_text())['passed']
  m=json.loads((root/('manifest-'+json.loads((root/'active-manifest.json').read_text())['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
 conf=json.loads((R/'confirmation.json').read_text());plan=json.loads((R/'confirmation-plan.json').read_text());assert conf['plan_sha256']==sha(R/'confirmation-plan.json') and plan['parent_map_sha256']==parent and conf['confirmed']==2 and len(conf['results'])==2
 old=json.loads(P.read_text());new=copy.deepcopy(old);initial=json.loads((I/'analysis/pipeline-map.json').read_text());fresh=json.loads((R/'analysis/pipeline-map.json').read_text());assert len(initial['candidates'])==6 and len(fresh['candidates'])==2
 def key(x):return tuple(x['tile']),x['layout']
 cmap={key(x):x for x in conf['results']};assert set(cmap)=={((32,256,128),'TT'),((128,256,128),'TT')}
 for c in conf['results']:
  assert c['confirmed'] and c['steady'] and c['relative_fresh_gain']>=.03 and c['intermediate_residual_fraction_of_long_time']<=.02 and c['executed_to_logical_mmac_ratio']==1 and c['initial']['steady'] and c['initial']['mmac_theoretical_fraction']/c['baseline']['mmac_theoretical_fraction']>=1.03
  assert c['eligible_choices']==['best_64k','best_64k_no_spill'] and c['shared_bytes']==65536 and c['config']['hcu_wasp_alias_ab_slots'] is True
  f=next(x for x in fresh['candidates'] if x['source_ids']==c['fresh_source_ids']);assert f['mmac_theoretical_fraction']==c['mmac_theoretical_fraction']
 newrows=[];winners={};changes=[]
 for row in initial['candidates']:
  x=copy.deepcopy(row);x.update(measurement_context_id=initial['measurement_context_id'],source_root_container=str(I),evidence_root_absolute=str(I/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(I/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(I/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=1,executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,continuation_eligible_choices=[],native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',kernel_contract='Natural-stride singleTG FP16/FP16 FP32acc FP16out; BK128 full OneA register capture/Afirst; Atypedmemdesc view reinterprets full contiguousBroot64KiB, originalA/BMLSencodings. B0/B1K64+K64 copy/read mappings preserved after separateoperandbases->0. BeforeAwrite waitall physicallyoverlappedBempty slots; TT B1prefix32768B,TN8192B, soTN128protectsB1too. AfterAready waitcurrentgenerationAempty phase(oldAemptyphase xor1) untilall8consumerAreadsLGKMdrained, beforeBoverwrite. Data0..5,MMACturn2tokens6/7,oldturn0,latewaitmaskdefault0,all12join2before8IDinvalidate. WDRA4/4/4 stage1 privateSB UF1split2/noBPP/asyncAPI. Native95 parent-viewMLS fix3fc2 plus recordedaliashelper/compilerrequired. Fullgold/rawMMOP1/allphysical0/fullASM16/32/64,independentinitial+freshsteady>=3%gainbeforepromotion. Eligibility is64KiBchoices only, not primary128KiB replacements.')
  isa=(I/'runs'/x['source_ids'][0]/'kernel.amdgcn').read_text();phys={}
  for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
   m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',isa);assert m;phys[n]=int(m[1])
  assert all(phys[n]==0 for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']) and x['shared_bytes']==65536;x['physical_resource_fields']=phys
  c=cmap.get(key(x))
  if c:
   assert x['source_ids']==c['initial_source_ids'] and x['config']==c['config'] and x['mmac_theoretical_fraction']==c['initial']['mmac_theoretical_fraction']
   x.update(continuation_selection_eligible=True,continuation_eligible_choices=c['eligible_choices'],independent_repeat_fraction=c['mmac_theoretical_fraction'],independent_repeat_residual=c['intermediate_residual_fraction_of_long_time'],confirmation_source_ids=c['fresh_source_ids'],confirmation_root_absolute=str(R/'runs'),confirmation_measurement_context_id=c['fresh_context'],independent_confirmation_sha256=sha(R/'confirmation.json'),independent_confirmation_raw_audit_sha256=sha(R/'native-raw-audit.json'));winners[key(x)]=x
  else:x['selection_rejection_reason']='Nonsteady or below3%gain under applicable choice constraints; no independentnomination and no promotion'
  newrows.append(x)
 for ident,w in winners.items():
  c=cmap[ident]
  for name in c['eligible_choices']:
   i=next(i for i,v in enumerate(old[name]) if key(v)==ident);base=c['baselines'][name];assert old[name][i]['config']==base['config'] and old[name][i]['cycles_per_bk_slope']==base['cycles_per_bk_slope']
   assert w['mmac_theoretical_fraction']/base['mmac_theoretical_fraction']>=1.03 and c['mmac_theoretical_fraction']/base['mmac_theoretical_fraction']>=1.03;new[name][i]=copy.deepcopy(w);changes.append(dict(choice=name,tile=w['tile'],layout=w['layout'],old_fraction=base['mmac_theoretical_fraction'],new_fraction=w['mmac_theoretical_fraction'],independent_fraction=c['mmac_theoretical_fraction']))
 unchanged=0
 for name in ['best','best_no_spill','best_64k','best_64k_no_spill']:
  assert len(old[name])==len(new[name])
  for a,b in zip(old[name],new[name]):
   if a==b:unchanged+=1
   else:assert name in ['best_64k','best_64k_no_spill'] and key(a)==key(b) and key(b) in winners
 assert unchanged==499 and len(changes)==4 and old['best']==new['best'] and old['best_no_spill']==new['best_no_spill']
 new['candidates']+=newrows;new['paired_candidates']=len(new['candidates'])
 for context in [initial['measurement_context_id'],fresh['measurement_context_id']]:
  assert context not in new['measurement_contexts'];new['measurement_contexts'].append(context)
 for root in [I,R]:new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
 new['continuation_round119_qualification']=dict(passed=True,CPU_source=str(B/'single-tg-pipeline-20261027-round115-alias-ab-slots-cpu-fix'),CPU_alias_ownership_audit=str(B/'single-tg-pipeline-20261027-round115-alias-ab-slots-cpu-fix/alias-ownership-isa-audit.json'),initial_native_audit=str(I/'native-raw-audit.json'),fresh_native_audit=str(R/'native-raw-audit.json'),controller_only_retry_proof=str(I/'retry-controller-proof.json'),initial_context=initial['measurement_context_id'],fresh_context=fresh['measurement_context_id'],native_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990')
 new['continuation_round119_confirmation']=conf;new['round119_selection_policy']='Six64KiBaliasedA/Bslotrecipes.24initial+12freshmodelsfullgold/MMOP1/allphysical0/fullASM16/32/64. TT32x256andTT128x256>=3%gain relativeboth64KiBcategories andsteadyinbothpredeclaredfresh32/16/64series;onefreshserieseach,noextrarepeats. Onlyfour64KiBchoicesupdated;all499otherchoices,primary128KiBchoicesand128cube deferredpreserved. TNaliasesnonsteadyorbelowold64Kgainsnotpromoted;zeroSGPRcubealternativeat37.85%notusedtoreplacefaster45.14%SGPRspillingchoice.';new['round119_parent_map_sha256']=parent;new['round119_statuses']=dict(initial={'PASS':24},fresh={'PASS':12});new['generated_remote_epoch']=time.time()
 assert len(new['best'])==126 and sum(len(new[k]) for k in ['best','best_no_spill','best_64k','best_64k_no_spill'])==503
 backup=C/'published/natural-stride-map-round113-before-round119.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent;tmp=P.with_name(P.name+'.round119.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P)
 proof=dict(passed=True,parent_map_sha256=parent,new_map_sha256=sha(P),changed_choices=changes,unchanged_choices=499,profiles=126,total_choices=503,primary_choices_identical=True,confirmed64KiBprofiles=2,main_steady_count=126,deferred_128cube_preserved=True,raw_native_dispatches_audited=144,controller_retry_extra_control_dispatches_audited=24,source_rows_added=6,script_sha256=sha(pathlib.Path(__file__)))
 (R/'integration-round119.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
