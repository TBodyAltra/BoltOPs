"""Record all eight audited no-turn results; preserve every chosen recipe."""
import pathlib,json,hashlib,copy,re,fcntl,os,time
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='85fca38d5a4fbc9c6b689a47127b6fdd6fcb1b83f4844de0d05c594a2f12af13';keys=['best','best_no_spill','best_64k','best_64k_no_spill']
with (C/'map-integration.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(R/'integration-round148.json').exists() and sha(P)==parent
 e=json.loads((R/'job-exit.json').read_text());a=json.loads((R/'native-raw-audit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and a['passed'] and a['points']==32 and a['raw_dispatches']==128 and a['all_full_output_gold'] and a['all_MMOP_ratios']==1 and a['all_physical_spills_zero']
 active=json.loads((R/'active-manifest.json').read_text());m=json.loads((R/('manifest-'+active['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
 analysis=json.loads((R/'analysis/pipeline-map.json').read_text());initial=json.loads((R/'pipeline-candidates.json').read_text());assert len(analysis['candidates'])==len(initial['candidates'])==8 and analysis['measurement_context_id']==active['measurement_context_id']
 old=json.loads(P.read_text());new=copy.deepcopy(old);rows=[]
 for row in analysis['candidates']:
  x=copy.deepcopy(row);c=next(v for v in initial['candidates'] if v['source_ids']==x['source_ids']);assert c['steady']==(x['valid_metric'] and x['intermediate_residual_fraction_of_long_time']<=.02)
  eligible=[k for k in keys if c['steady'] and any(v['tile']==c['tile'] and v['layout']==c['layout'] and c['mmac_theoretical_fraction']/v['mmac_theoretical_fraction']-1>=.03 for v in old[k])];assert eligible==[]
  physical={}
  for name in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count','amdhsa_next_free_vgpr']:
   match=re.search(r'\.'+name+r'\s*[:=]?\s*(\d+)',(R/'runs'/x['source_ids'][0]/'kernel.amdgcn').read_text());assert match;physical[name]=int(match[1])
  assert all(physical[k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
  x.update(measurement_context_id=analysis['measurement_context_id'],source_root_container=str(R),evidence_root_absolute=str(R/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(R/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=x['config']['LOOP_UNROLL_FACTOR'],executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,continuation_eligible_choices=[],native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',physical_resource_fields=physical,native_raw_audit_sha256=sha(R/'native-raw-audit.json'),kernel_contract='Natural-stride FP16 singleTG fullBK128/FP32acc/FP16out. Frozen CPU133 GPU/native95; CPU143 actual-address/all64lane-read/ISA ownership and finite DATA-only no-turn model142 qualified. AliasAB64KiB/splitBslots, no cross-role MMACturn, all actual data ownership guards retained, six data signal IDs0..5; WASP/WDRA4/4/4 stage1/privateSB/UF1/split2/Afirst/noasyncAPI/noBPP. Fullgold/MMOP1/CU0/allphysicalspill0/fullASM16/32/64. No initial steady>=3% winner; no fresh series or promotion.',selection_rejection_reason='Initial nonsteady or steady but slower than existing choices; no independent repeat nominated')
  rows.append(x)
 assert sum(c['steady'] for c in initial['candidates'])==2
 new['candidates']+=rows;new['paired_candidates']=len(new['candidates']);assert analysis['measurement_context_id'] not in new['measurement_contexts'];new['measurement_contexts'].append(analysis['measurement_context_id']);new['source_roots'].append(dict(container=str(R),host=str(R).replace('/xukang/code/','/home/hygon/xukang/code/')))
 for key in keys:assert new[key]==old[key]
 qual=dict(passed=True,parent_map_sha256=parent,source_root=str(R),native_raw_audit_sha256=sha(R/'native-raw-audit.json'),raw_dispatches_audited=128,candidate_rows_added=8,initial_steady_but_slower_candidates=2,initial_nonsteady_candidates=6,all503_choice_records_unchanged=True,no_fresh_nominees=True,no_performance_promotion=True,script_sha256=sha(pathlib.Path(__file__)))
 new['continuation_round148_qualification']=qual;new['generated_remote_epoch']=time.time()
 backup=C/'published/natural-stride-map-round147-before-round148.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent
 tmp=P.with_name(P.name+'.round148.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P)
 out=dict(qual,new_map_sha256=sha(P),changed_choices=[],deferred_128cube_preserved=True);(R/'integration-round148.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
