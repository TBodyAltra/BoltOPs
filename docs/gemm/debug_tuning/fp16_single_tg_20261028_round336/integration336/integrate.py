from pathlib import Path
import json,hashlib,copy,fcntl,os,time
R=Path(__file__).resolve().parent;B=R.parent;K=B/'single-tg-continuation-20261023';P=K/'published/natural-stride-map.json'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent='0eb9d58f1f7c13ce5714ed910f37c223bcdf5320167812fb13dad15ab2bf77c7';keys=['best','best_no_spill','best_64k','best_64k_no_spill']
with (K/'map-integration.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert sha(P)==parent and not(R/'integration.json').exists()
 old=load(P);new=copy.deepcopy(old);assert len(old['candidates'])==9043 and sum(len(old[k]) for k in keys)==503
 identity=load(next(B.glob('*round321-*'))/'point-identity-regression.json');assert identity['passed'] and identity['contracts']==identity['unique_ids']==242
 changes=[];cohorts=[];alladded=[]
 for ni,nf,count,points,freshpoints in [(321,328,60,242,11)]:
  I=next(B.glob(f'*round{ni}-*'));F=next(B.glob(f'*round{nf}-*'));assert load(I/'physical-fragment-contract-proof.json')['passed']
  conf=load(F/'performance-confirmation.json');assert conf['passed'] and conf['one_independent_series_per_recipe'] and conf['no_retry_or_fallback_after_fresh_failure']
  initial=load(I/'selection-audit.json');assert len(conf['confirmed_recipes'])+len(conf['rejected_recipes'])==len(initial['nominees'])
  for root,expected in [(I,points),(F,freshpoints)]:
   assert load(root/'job-exit.json')['returncode']==load(root/'supervisor-exit.json')['returncode']==0
   act=load(root/'active-manifest.json');manifest=load(root/('manifest-'+act['manifest_id']+'.json'));assert all(sha(Path(p))==h for p,h in manifest['hashes'].items())
   audit=load(root/'native-raw-audit.json');assert audit['passed'] and audit['points']==expected and audit['raw_dispatches']==expected*4 and audit['all_singleTG_MMOP_ratios']==1 and audit['physical_zero_candidates']==(count if root==I else len(initial['nominees']))
  proof=load(F/'executable-relocation-proof.json');assert proof['passed'] and proof['pairs']==3*len(initial['nominees']) and proof['all_allocated_sections_and_AMDGPU_notes_identical'] and proof['all_non_debug_ELF_sections_identical']
  analyzed=load(I/'analysis/pipeline-map.json')['candidates'];candidates=load(I/'pipeline-candidates.json')['candidates'];assert len(analyzed)==len(candidates)==count
  confirmations={tuple(n['initial']['source_ids']):n for n in conf['confirmed_recipes']+conf['rejected_recipes']};added=[]
  for x0 in analyzed:
   x=copy.deepcopy(x0);c=next(z for z in candidates if z['source_ids']==x['source_ids']);cfg=c['config'];f=cfg['WASP_GLOBAL_K_FRAGMENTS'];assert x['config']==cfg and x['mmac_theoretical_fraction']==c['mmac_theoretical_fraction'] and f in [2,4] and cfg['MMA_K_SPLIT']==1 and cfg['WARP_SPECIALIZE'] and c['tile'][2]==128 and c['tile']!=[128,128,128] and c['executed_to_logical_mmac_ratio']==1
   assert not cfg['wasp_wdra'] and all(c['physical'][k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count'])
   n=confirmations.get(tuple(c['source_ids']));eligible=[ch['choice'] for ch in n['choices'] if ch['validated']] if n else []
   x.update(measurement_context_id=load(I/'active-manifest.json')['measurement_context_id'],source_root_container=str(I),evidence_root_absolute=str(I/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(I/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(I/'triton/third_party/hcu/backend/compiler.py'),native_artifact_sha256='e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a',physical_resource_fields=c['physical'],all_physical_spills_zero=True,native_raw_audit_sha256=sha(I/'native-raw-audit.json'),pipeline_unroll_factor=cfg['LOOP_UNROLL_FACTOR'],logical_block_K=128,dot_fragment_K=128//f//cfg['MMA_K_SPLIT'],physical_global_K=128//f,global_fragments_per_logical_BK=f,physical_loop_counts=[v*f for v in c['loop_counts']],executed_mmop_instructions_per_bk=2*c['tile'][0]*c['tile'][1]*128//8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=bool(eligible),continuation_eligible_choices=eligible,requested_num_stages=cfg['num_stages'],requested_partition_warps=cfg['wasp_partition_warps'],requested_partition_regs=cfg.get('wasp_partition_regs'),kernel_contract='Natural-stride FP16 singleTG logicalBK128; internal global and dot K=BK/WASP_GLOBAL_K_FRAGMENTS, physical iteration count multiplied by fragments; FP32acc/FP16out. Frozen isolated Bolt297 kernel/native209. FullGOLD,physicalspills0,MMOPwork1,CU0; logical16/32/64 slope and4TG off-origin odd17 correctness. Independent sole fresh series with full allocated/nondebug ELF,metadata/resources and runtime agreement. Separate actual64KiB/128KiB LDS rankings. No runtimeMN specialization or TG rearrangement.')
   first=load(I/'runs'/c['source_ids'][0]/'result.json');x['available_partition_regs']=first['metadata'].get('wasp_partition_regs');x['backend_required_partition_regs']=first['metadata'].get('wasp_partition_regs_actual');assert first['metadata'].get('wasp_partition_regs_requested')==cfg.get('wasp_partition_regs')
   if n:
    x.update(independent_repeat_fraction=n['fresh']['mmac_theoretical_fraction'],independent_repeat_residual=n['fresh']['intermediate_residual_fraction_of_long_time'],confirmation_source_ids=n['fresh']['source_ids'],confirmation_root_absolute=str(F/'runs'),confirmation_measurement_context_id=n['fresh_context'],independent_confirmation_sha256=sha(F/'performance-confirmation.json'),independent_confirmation_raw_audit_sha256=sha(F/'native-raw-audit.json'),cross_root_executable_proof_sha256=sha(F/'executable-relocation-proof.json'),raw_full_ASM_identical_across_roots=False,executable_and_non_debug_ELF_sections_identical_across_roots=True,independent_choice_validation=n['choices'])
   if not eligible:x['selection_rejection_reason']='Not selected by initial fastest steady >=3% policy, or sole independent fresh did not qualify; no retry or fallback.'
   added.append(x)
  new['candidates']+=added;alladded+=added
  for key in keys:
   for index,prev in enumerate(new[key]):
    selected=[x for x in added if key in x['continuation_eligible_choices'] and (x['tile'],x['layout'])==(prev['tile'],prev['layout'])];assert len(selected)<=1
    if not selected:continue
    x=selected[0];n=confirmations[tuple(x['source_ids'])];ch=next(z for z in n['choices'] if z['choice']==key);assert ch['prior']==prev and ch['validated'] and min(ch['initial_relative_throughput'],ch['fresh_relative_throughput'])>=1.03 and x['shared_bytes']<=(65536 if '64k' in key else 131072)
    new[key][index]=copy.deepcopy(x);changes.append(dict(choice=key,tile=x['tile'],layout=x['layout'],prior_fraction=prev['mmac_theoretical_fraction'],initial_fraction=x['mmac_theoretical_fraction'],fresh_fraction=x['independent_repeat_fraction'],initial_gain=ch['initial_relative_throughput']-1,fresh_gain=ch['fresh_relative_throughput']-1,shared_bytes=x['shared_bytes'],config=x['config'],source_ids=x['source_ids'],confirmation_source_ids=x['confirmation_source_ids']))
  for root in (I,F):
   ctx=load(root/'active-manifest.json')['measurement_context_id'];assert ctx not in new['measurement_contexts'];new['measurement_contexts'].append(ctx);new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
  cohorts.append(dict(initial_root=str(I),fresh_root=str(F),candidates=count,confirmed_recipes=len(conf['confirmed_recipes']),rejected_recipes=len(conf['rejected_recipes']),confirmed_choices=conf['confirmed_choices'],initial_raw_audit_sha256=sha(I/'native-raw-audit.json'),fresh_raw_audit_sha256=sha(F/'native-raw-audit.json'),confirmation_sha256=sha(F/'performance-confirmation.json'),physical_fragment_contract_sha256=sha(I/'physical-fragment-contract-proof.json')))
 assert len(alladded)==60 and len(new['candidates'])==9103 and len(changes)==sum(c['confirmed_choices'] for c in cohorts)
 for key in keys:
  before={(*x['tile'],x['layout']):x for x in old[key]};after={(*x['tile'],x['layout']):x for x in new[key]};assert before.keys()==after.keys();assert {k for k in before if before[k]!=after[k]}=={(*c['tile'],c['layout']) for c in changes if c['choice']==key}
 assert sum(len(new[k]) for k in keys)==503
 new['paired_candidates']=len(new['candidates']);qual=dict(passed=True,parent_map_sha256=parent,initial_candidates_added=60,candidate_rows=9103,changed_choices=len(changes),unchanged_choices=503-len(changes),cohorts=cohorts,deferred128cube_preserved=True,one_fresh_per_recipe=True,script_sha256=sha(Path(__file__)));new['continuation_round336_qualification']=qual;new['generated_remote_epoch']=time.time()
 backup=K/'published/natural-stride-map-before-round336.json';assert not backup.exists();backup.write_bytes(P.read_bytes());tmp=P.with_name(P.name+'.round336.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P);out=dict(qual,new_map_sha256=sha(P),changes=changes);(R/'integration.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
