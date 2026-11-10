"""Record audited GROUP1 candidates and the one independent resource verdict."""
import pathlib,json,hashlib,fcntl,copy,time,os
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json';F=B/'single-tg-pipeline-20261027-round177-group1-index-confirm'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
parent='1f59672b149ac413bb7eb894780d1e4d50d33e21925a29fe3f3cbfbd22f064d3';choices=['best','best_no_spill','best_64k','best_64k_no_spill']
with (C/'map-integration.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert sha(P)==parent and not (R/'integration-round178.json').exists()
    a=load(R/'native-raw-audit.json');f=load(F/'native-raw-audit.json');v=load(F/'confirmation.json');sel=load(R/'selection-audit.json')
    assert a['passed'] and a['points']==26 and a['raw_dispatches']==104 and f['passed'] and f['points']==5 and f['raw_dispatches']==20 and v['passed'] and v['one_fresh_series_only'] and sel['performance_nominees']==0
    assert v['initial_raw_audit_sha256']==sha(R/'native-raw-audit.json') and v['fresh_raw_audit_sha256']==sha(F/'native-raw-audit.json') and v['full_assembly_and_runtime_contract_equal'] and v['source_hashes_unchanged']
    for root in [R,F]:
        e=load(root/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
        act=load(root/'active-manifest.json');m=load(root/('manifest-'+act['manifest_id']+'.json'));assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
    analysis=load(R/'analysis/pipeline-map.json');initial=load(R/'pipeline-candidates.json');assert len(analysis['candidates'])==len(initial['candidates'])==8
    old=load(P);new=copy.deepcopy(old);rows=[]
    resource_path=C/'published/zero-physical-spill-alternatives-round178.json';assert not resource_path.exists()
    resource=dict(schema_version=1,passed=True,parent_map_sha256=parent,source_initial_root=str(R),source_fresh_root=str(F),validated_recipes=[v] if v['resource_alternative_validated'] else [],rejected_recipes=[] if v['resource_alternative_validated'] else [v],all503_best_choices_preserved=True,initial_raw_audit_sha256=sha(R/'native-raw-audit.json'),fresh_raw_audit_sha256=sha(F/'native-raw-audit.json'),confirmation_sha256=sha(F/'confirmation.json'),initial_and_fresh_min_relative_throughput=.98,full_assembly_equal=True,one_series_per_recipe=True,no_multi_TG_occupancy_claim=True,kernel_source_worktree=str((R/'bolt_ops').resolve()),kernel_mapping='SIMPLIFY_GROUP1=True only; same legal TG mapping for every M,N. No workload M/N specialization. FullBK128 transport and MMA unchanged.',script_sha256=sha(pathlib.Path(__file__)))
    for row in analysis['candidates']:
        x=copy.deepcopy(row);c=next(z for z in initial['candidates'] if z['source_ids']==x['source_ids'])
        assert c['steady']==bool(x['valid_metric'] and x['intermediate_residual_fraction_of_long_time']<=.02)
        for key in choices:
            baselines=[z for z in old[key] if z['tile']==c['tile'] and z['layout']==c['layout']]
            assert not (c['steady'] and baselines and c['mmac_theoretical_fraction']/baselines[0]['mmac_theoretical_fraction']>=1.03)
        physical=c['physical'];zero=all(physical[k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count'])
        nominated=x['source_ids']==v['initial']['source_ids']
        x.update(measurement_context_id=analysis['measurement_context_id'],source_root_container=str(R),evidence_root_absolute=str(R/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(R/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=x['config']['LOOP_UNROLL_FACTOR'],executed_mmop_instructions_per_bk=2*x['tile'][0]*x['tile'][1]*128/8192,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,continuation_eligible_choices=[],native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',physical_resource_fields=physical,all_physical_spills_zero=zero,native_raw_audit_sha256=sha(R/'native-raw-audit.json'),resource_alternative_requested=nominated,resource_alternative_only=nominated,resource_alternative_validated=bool(nominated and v['resource_alternative_validated']),kernel_contract='Natural-stride FP16 singleTG fullBK128/FP32acc/FP16out. Isolated Bolt175 and frozen Triton121/native95. SIMPLIFY_GROUP1=True specializes GROUP_SIZE_M=1 mapping for all legal TG ids, preserving exact arrangement for arbitrary positive M,N; no MNK workload specialization. DefaultFalse old CPU/native controls exact. NoWASP/noWDRA,warps8/stage1/MLSfragment/Bfirst/split4,UF1/2/4/8,EPbool. K transport, math, launch grid, full-output masks unchanged. Fullgold/MMOP1/CU0/fullASM16/32/64. Seven initial steady,one nonsteady; no>=3% performance winner. Exactly one initial strict-zero UF2/EPFalse resource nominee received one independent16/32/64 series.',selection_rejection_reason='No>=3% performance gain; best503 preserved. Resource alternative verdict recorded separately.')
        if nominated:
            x.update(resource_alternative_report=str(resource_path),resource_confirmation_root_absolute=str(F),resource_confirmation_source_ids=v['fresh']['source_ids'],resource_confirmation_fraction=v['fresh']['mmac_theoretical_fraction'],resource_confirmation_residual=v['fresh']['intermediate_residual_fraction_of_long_time'])
        rows.append(x)
    assert sum(x['steady'] for x in initial['candidates'])==7
    new['candidates']+=rows;new['paired_candidates']=len(new['candidates'])
    for root in [R,F]:
        ctx=load(root/'active-manifest.json')['measurement_context_id'];assert ctx not in new['measurement_contexts'];new['measurement_contexts'].append(ctx);new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
    for key in choices:assert old[key]==new[key]
    qual=dict(passed=True,parent_map_sha256=parent,initial_root=str(R),fresh_root=str(F),initial_raw_audit_sha256=sha(R/'native-raw-audit.json'),fresh_raw_audit_sha256=sha(F/'native-raw-audit.json'),candidate_rows_added=8,initial_steady_candidates=7,initial_nonsteady_candidates=1,raw_dispatches_audited=124,all503_choice_records_unchanged=True,no_performance_promotion=True,independent_resource_nominees=1,independent_resource_validated=int(v['resource_alternative_validated']),resource_alternatives_path=str(resource_path),script_sha256=sha(pathlib.Path(__file__)))
    backup=C/'published/natural-stride-map-round167-before-round178.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent
    resource_path.write_text(json.dumps(resource,indent=2,allow_nan=False)+'\n');qual['resource_alternatives_sha256']=sha(resource_path)
    new['continuation_round178_qualification']=qual;new['generated_remote_epoch']=time.time()
    tmp=P.with_name(P.name+'.round178.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P)
    out=dict(qual,new_map_sha256=sha(P),changed_choices=[],deferred_128cube_preserved=True);(R/'integration-round178.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
