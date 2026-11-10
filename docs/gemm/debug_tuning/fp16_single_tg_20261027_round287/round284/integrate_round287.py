from pathlib import Path
import json,hashlib,copy,fcntl,os,time
R=Path(__file__).resolve().parent;K=R.parent/'single-tg-continuation-20261023';P=K/'published/natural-stride-map.json';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();parent='f5d4d5bff3d345ef6d70e7f096c8b3bc0f4845f5d192619bc1ff5e8a7121ebab'
with (K/'map-integration.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(P)==parent and not(R/'integration-round287.json').exists()
 assert load(R/'job-exit.json')['returncode']==load(R/'supervisor-exit.json')['returncode']==0
 a=load(R/'native-raw-audit.json');cs=load(R/'pipeline-candidates.json')['candidates'];N=len(cs)
 assert N==30 and a['passed'] and a['points']==2+4*N and a['raw_dispatches']==4*(2+4*N) and a['off_origin_edge_correctness_only_points']==N
 selection=load(R/'selection-audit.json');assert not selection['nominees']
 act=load(R/'active-manifest.json');manifest=load(R/('manifest-'+act['manifest_id']+'.json'));assert all(sha(p)==v for p,v in manifest['hashes'].items())
 old=load(P);new=copy.deepcopy(old);rows=load(R/'analysis/pipeline-map.json')['candidates'];assert len(rows)==N
 refs={}
 for p in R.glob('*-qualification.json'):
  q=load(p)
  for check in q.get('checks',[]):
   assert q['passed'];refs[check['id']]=Path(check['CPU_reference'])
 for x0 in rows:
  x=copy.deepcopy(x0);c=next(z for z in cs if z['source_ids']==x['source_ids']);assert c['tile']==[64,64,128] and c['executed_to_logical_mmac_ratio']==1 and not c['config']['WARP_SPECIALIZE'] and not c['config']['wasp_wdra'];assert all(c['physical'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
  ref=refs[c['source_ids'][0]];actual_BPP='hcu.block_pingpong.applied' in (ref/'kernel.ttgir').read_text()
  x.update(measurement_context_id=act['measurement_context_id'],source_root_container=str(R),evidence_root_absolute=str(R/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),compiler_source_absolute=str(R/'triton/third_party/hcu/backend/compiler.py'),physical_resource_fields=c['physical'],all_physical_spills_zero=True,native_artifact_sha256='e559a4aba702980b7e7275644a8b0be73e6b03c48432fd1bbae6a20c6ec4a424',logical_block_K=128,dot_fragment_K=128//c['config']['MMA_K_SPLIT'],executed_mmop_instructions_per_bk=2*64*64*128//8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=c['mmac_theoretical_fraction'],continuation_selection_eligible=False,continuation_eligible_choices=[],independent_repeat=False,native_raw_audit_sha256=sha(R/'native-raw-audit.json'),block_pingpong_actual_TTGIR_applied=actual_BPP,block_pingpong_actual_marker_CPU_reference=str(ref),kernel_contract='Natural-stride FP16 single-TG64x64x128 TT/TN,vector tl.load,NoWASP/WDRA,async/BPP requested;actual BPP marker separately recorded. Stable first-use BlockPingpong allocation ordering. FullGOLD,zero physicalspills,MMOPwork1,CU0,identicalshader16/32/64 andoff-origin4TG odd17. Resource-budget comparisons and steady criteria required;no independent repeat or promotion.',selection_rejection_reason='Nonsteady initial slope;no repeat' if not c['steady'] else 'No >=3% gain over current same-budget best;no independent repeat or promotion.')
  new['candidates'].append(x)
 for k in ['best','best_no_spill','best_64k','best_64k_no_spill']:assert old[k]==new[k]
 assert sum(len(new[k]) for k in ['best','best_no_spill','best_64k','best_64k_no_spill'])==503
 new['paired_candidates']=len(new['candidates']);assert act['measurement_context_id'] not in new['measurement_contexts'];new['measurement_contexts'].append(act['measurement_context_id']);new['source_roots'].append(dict(container=str(R),host=str(R).replace('/xukang/code/','/home/hygon/xukang/code/')))
 C=next(R.parent.glob('*round280-*'))
 proof=dict(passed=True,parent_map_sha256=parent,candidate_rows_added=N,steady=sum(x['steady'] for x in cs),models=a['points'],raw_dispatches=a['raw_dispatches'],all503_choices_unchanged=True,no_promotion=True,native_raw_audit_sha256=sha(R/'native-raw-audit.json'),selection_audit_sha256=sha(R/'selection-audit.json'),new_native_sha256=sha(R/'native/libtriton.so'),CPU_determinism_qualification_sha256=sha(C/'determinism-qualification.json'),MLS_control_invariance_sha256=sha(C/'control-native209-invariance.json'),complete_source_patch_sha256=sha(C/'triton-complete-source.patch'),native_build_proof_sha256=sha(C/'build-result.json'),script_sha256=sha(Path(__file__)))
 new['continuation_round287_qualification']=proof;new['generated_remote_epoch']=time.time();backup=K/'published/natural-stride-map-round276-before-round287.json';assert not backup.exists();backup.write_bytes(P.read_bytes());tmp=P.with_name(P.name+'.round287.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P);out=dict(proof,new_map_sha256=sha(P),candidate_rows=len(new['candidates']));(R/'integration-round287.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
