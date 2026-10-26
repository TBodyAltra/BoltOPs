"""Record audited candidates and a confirmed resource alternative; preserve all winners."""
import pathlib,json,hashlib,copy,re,fcntl,os,time
R=pathlib.Path(__file__).resolve().parent;B=R.parent;I=B/'single-tg-pipeline-20261027-round122-nowasp-sgpr-native';C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='16d1c90b9210d7d2806a15dd7f727e88600913dfc1160597e293f9936265d6c1'
with (C/'map-integration.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(R/'integration-round125.json').exists() and sha(P)==parent
 for root,points in [(I,38),(R,8)]:
  e=json.loads((root/'job-exit.json').read_text());audit=json.loads((root/'native-raw-audit.json').read_text())
  assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and audit['passed'] and audit['points']==points and audit['all_feature_physical_spills_zero']
  m=json.loads((root/('manifest-'+json.loads((root/'active-manifest.json').read_text())['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
 conf=json.loads((R/'confirmation.json').read_text());assert conf['plan_sha256']==sha(R/'confirmation-plan.json') and conf['confirmed']==0 and len(conf['results'])==2
 assert all(not x['eligible_choices'] and x['resource_alternative_only'] for x in conf['results'])
 verified=[x for x in conf['results'] if x['resource_alternative_validated']];assert len(verified)==1 and verified[0]['tile']==[256,256,128]
 old=json.loads(P.read_text());new=copy.deepcopy(old);initial=json.loads((I/'analysis/pipeline-map.json').read_text());fresh=json.loads((R/'analysis/pipeline-map.json').read_text());assert len(initial['candidates'])==12 and len(fresh['candidates'])==2
 rows=[]
 for row in initial['candidates']:
  x=copy.deepcopy(row);x.update(measurement_context_id=initial['measurement_context_id'],source_root_container=str(I),evidence_root_absolute=str(I/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(I/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(I/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=x['config']['LOOP_UNROLL_FACTOR'],executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,continuation_eligible_choices=[],native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',kernel_contract='Natural-stride FP16 singleTG/fullBK128/FP32acc/FP16out. NoWASP/NoWDRA; stage1 warps8 split4 UF1/2 optimize_epilogueFalse; MLS/fragment layoutTrue, current per-tile load order; asyncAPI/BPPFalse. CPU121 explicitly binds JIT default constexprs; controls exact prior ISA/resources. Native95+recorded frozen CPU121 kernel/backend required. All feature scratch/SGPR/VGPR spill0, LDS65536; fullgold/rawMMOP1/CU0/fullASM16/32/64. Slower than old map winners; no selection replacement.')
  first=I/'runs'/x['source_ids'][0];phys={}
  for n in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
   m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',(first/'kernel.amdgcn').read_text());assert m;phys[n]=int(m[1])
  assert all(phys[k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']) and x['shared_bytes']==65536;x['physical_resource_fields']=phys
  selected=next((c for c in conf['results'] if c['initial_source_ids']==x['source_ids']),None)
  if selected:
   x.update(independent_repeat_fraction=selected['mmac_theoretical_fraction'],independent_repeat_residual=selected['intermediate_residual_fraction_of_long_time'],confirmation_source_ids=selected['fresh_source_ids'],confirmation_root_absolute=str(R/'runs'),confirmation_measurement_context_id=selected['fresh_context'],independent_resource_alternative_validated=selected['resource_alternative_validated'],independent_confirmation_sha256=sha(R/'confirmation.json'))
  x['selection_rejection_reason']='Initial nonsteady or no3percent gain; resource-only confirmation cannot replace a faster legacy scratch-only winner'
  rows.append(x)
 accepted=[];rejected=[]
 for c in conf['results']:
  x=dict(c);x.update(source_root_container=str(R),evidence_root_absolute=str(R/'runs'),native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',source_worktree=str(B/'single-tg-pipeline-20261027-round121-nowasp-sgpr-cpu-defaults'),raw_audit_sha256=sha(R/'native-raw-audit.json'),scope='Resource alternative from the focused CPU121/native122/123 screen; not a claim of globally fastest zero-spill recipe',replaces_map_winner=False)
  (accepted if c['resource_alternative_validated'] else rejected).append(x)
 alt=C/'published/zero-physical-spill-alternatives-round123.json';assert not alt.exists()
 report=dict(schema_version=1,passed=True,parent_map_sha256=parent,validated_recipes=accepted,rejected_fresh_nonsteady_recipes=rejected,all_best_choices_preserved=True,initial_points=38,fresh_points=8,raw_dispatches_audited=184,initial_context=initial['measurement_context_id'],fresh_context=fresh['measurement_context_id'],initial_raw_audit_sha256=sha(I/'native-raw-audit.json'),fresh_raw_audit_sha256=sha(R/'native-raw-audit.json'),confirmation_sha256=sha(R/'confirmation.json'),script_sha256=sha(pathlib.Path(__file__)))
 alt.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 for name in ['best','best_no_spill','best_64k','best_64k_no_spill']:assert old[name]==new[name]
 new['candidates']+=rows;new['paired_candidates']=len(new['candidates'])
 for context in [initial['measurement_context_id'],fresh['measurement_context_id']]:assert context not in new['measurement_contexts'];new['measurement_contexts'].append(context)
 for root in [I,R]:new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
 new['continuation_round125_qualification']=dict(passed=True,initial_root=str(I),fresh_root=str(R),initial_raw_audit=str(I/'native-raw-audit.json'),fresh_raw_audit=str(R/'native-raw-audit.json'),raw_dispatches_audited=184,zero_physical_spill_initial_candidates=12,initial_steady_candidates=11,all503_choice_records_unchanged=True,zero_physical_spill_resource_alternatives=str(alt),resource_alternatives_sha256=sha(alt),confirmed_resource_alternatives=1,rejected_fresh_nonsteady_resource_alternatives=1,no_performance_promotion=True)
 new['generated_remote_epoch']=time.time()
 assert sum(len(new[k]) for k in ['best','best_no_spill','best_64k','best_64k_no_spill'])==503
 backup=C/'published/natural-stride-map-round119-before-round125.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent
 tmp=P.with_name(P.name+'.round125.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P)
 proof=dict(passed=True,parent_map_sha256=parent,new_map_sha256=sha(P),changed_choices=[],unchanged_choice_records=503,candidate_rows_added=12,validated_resource_alternatives=1,rejected_fresh_nonsteady_resource_alternatives=1,resource_alternatives_file=str(alt),resource_alternatives_sha256=sha(alt),raw_dispatches_audited=184,deferred_128cube_preserved=True,script_sha256=sha(pathlib.Path(__file__)))
 (R/'integration-round125.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
