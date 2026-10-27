"""Record four calibrated recipes and two immutable rejected fast-allocator cases."""
import pathlib,json,hashlib,copy,fcntl,time,os
R=pathlib.Path(__file__).resolve().parent;B=R.parent;S=B/'single-tg-pipeline-20261027-round181-sgpr-liveness-native';C=B/'single-tg-continuation-20261023';P=C/'published/natural-stride-map.json'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent='d46000e6bcd22424e6752301ff055b10870a2deba419d24c266983bfeffaf456';keys=['best','best_no_spill','best_64k','best_64k_no_spill']
with (C/'map-integration.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert sha(P)==parent and not (R/'integration-round183.json').exists()
    a=load(S/'mixed-short-gate-audit.json');b=load(R/'native-raw-audit.json');sel=load(R/'selection-audit.json');h=load(R/'handoff-proof.json')
    assert a['passed'] and a['points']==8 and a['raw_dispatches']==32 and a['feature_status_counts']==dict(PASS=4,ACCURACY_FAIL=2)
    assert b['passed'] and b['points']==14 and b['raw_dispatches']==56 and b['imported_short_points']==6 and b['new_long_points']==8 and b['maximum_concurrent_compilers']==1
    assert sel['passed'] and sel['performance_nominees']==sel['resource_alternative_nominees']==0 and sha(S/'mixed-short-gate-audit.json')==h['parent_mixed_audit_sha256']
    aa={x['id']:x for x in a['rows']};bb={x['id']:x for x in b['rows']};common=aa.keys()&bb.keys();assert len(common)==6 and len(aa.keys()|bb.keys())==16
    for ident in common:
        for field in ['result_sha256','asm_sha256','raw_stats_hashes']:assert aa[ident][field]==bb[ident][field]
    for root,expected in [(S,1),(R,0)]:
        e=load(root/'job-exit.json');assert e['returncode']==expected and not pathlib.Path('/proc',str(e['pid'])).exists()
        act=load(root/'active-manifest.json');m=load(root/('manifest-'+act['manifest_id']+'.json'));assert all(sha(pathlib.Path(p))==v for p,v in m['hashes'].items())
    assert load(S/'active-manifest.json')['measurement_context_id']==load(R/'active-manifest.json')['measurement_context_id']==a['measurement_context_id']
    rejected=[x for x in a['rows'] if x['status']=='ACCURACY_FAIL'];assert len(rejected)==2
    for x in rejected:
        assert x['config']['hcu_sgpr_liveness_strategy']==5 and x['accuracy']['mismatches']==1414 and x['accuracy']['nonfinite']==0
        assert sha(pathlib.Path(x['failure_snapshot']['path']))==x['failure_snapshot']['sha256']
    reject_path=C/'published/rejected-sgpr-fast-strategy-round183.json';assert not reject_path.exists()
    reject_report=dict(passed=True,scope='Original precision rejections; no performance eligibility or retry',source_root=str(S),mixed_short_gate_audit_sha256=sha(S/'mixed-short-gate-audit.json'),strategy=5,clang_flag='-mllvm=-sgpr-regalloc=fast',recipes=rejected,source_hashes_unchanged=True,original_precision_failure_snapshots_preserved=True,no_reset_or_kill=True)
    analysis=load(R/'analysis/pipeline-map.json');initial=load(R/'pipeline-candidates.json');assert len(analysis['candidates'])==len(initial['candidates'])==4
    old=load(P);new=copy.deepcopy(old);rows=[]
    for row in analysis['candidates']:
        x=copy.deepcopy(row);c=next(z for z in initial['candidates'] if z['source_ids']==x['source_ids'])
        assert c['steady']==bool(x['valid_metric'] and x['intermediate_residual_fraction_of_long_time']<=.02)
        for key in keys:
            baseline=next(z for z in old[key] if z['tile']==c['tile'] and z['layout']==c['layout']);assert not (c['steady'] and c['mmac_theoretical_fraction']/baseline['mmac_theoretical_fraction']>=1.03)
        physical=c['physical'];assert physical['private_segment_fixed_size']==physical['vgpr_spill_count']==0 and physical['sgpr_spill_count'] in [2,4,18]
        x.update(measurement_context_id=analysis['measurement_context_id'],source_root_container=str(R),evidence_root_absolute=str(R/'runs'),evidence_root_relative='runs',kernel_source_absolute=str(S/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),kernel_source_relative='bolt_ops/boltops/gemm/triton/gemm_fp16.py',compiler_source_absolute=str(S/'triton/third_party/hcu/backend/compiler.py'),pipeline_unroll_factor=4,executed_mmop_instructions_per_bk=1024,executed_to_logical_mmac_ratio=1.0,executed_mmac_throughput_fraction=x['mmac_theoretical_fraction'],continuation_selection_eligible=False,continuation_eligible_choices=[],native_artifact_sha256='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990',physical_resource_fields=physical,all_physical_spills_zero=False,native_raw_audit_sha256=sha(R/'native-raw-audit.json'),short_gate_audit_sha256=sha(S/'mixed-short-gate-audit.json'),initial_series_continuation=True,independent_repeat=False,controller_handoff_proof=str(R/'handoff-proof.json'),kernel_contract='Natural-stride FP16 singleTG fullBK128/FP32acc/FP16out. Frozen kernel175 and private Triton180 Python-backend liveness option, native95 unchanged. NoWASP/noWDRA/8warps/stage1/UF4/split4/EPTrue/Bfirst/fragmentMLS. Strategies1(arg-loweringFalse) and3(SGPR allocator basic), generic or exactly equivalent GROUP1 mapping. Default0 two CPU/native controls exact. Original short gate retained two strategy5 fast-allocator accuracy failures; four passing short cases and two controls reused without redispatch. Unchanged parent worker/scanner/env and exact context proved before8 new long points. Fullgold/MMOP1/CU0/fullASM16/32/64, one compiler lock. Two steady below3%gain,two nonsteady. No independent fresh nominee, strict-zero resource alternative or performance promotion.',selection_rejection_reason='Initial nonsteady or gain below3%; genuine SGPR spills retained. Fast allocation separately rejected for accuracy.')
        rows.append(x)
    assert sum(x['steady'] for x in initial['candidates'])==2
    new['candidates']+=rows;new['paired_candidates']=len(new['candidates']);ctx=a['measurement_context_id'];assert ctx not in new['measurement_contexts'];new['measurement_contexts'].append(ctx)
    for root in [S,R]:new['source_roots'].append(dict(container=str(root),host=str(root).replace('/xukang/code/','/home/hygon/xukang/code/')))
    for key in keys:assert old[key]==new[key]
    backup=C/'published/natural-stride-map-round178-before-round183.json';assert not backup.exists();backup.write_bytes(P.read_bytes());assert sha(backup)==parent
    reject_path.write_text(json.dumps(reject_report,indent=2,allow_nan=False)+'\n')
    qual=dict(passed=True,parent_map_sha256=parent,short_gate_root=str(S),calibration_root=str(R),short_gate_audit_sha256=sha(S/'mixed-short-gate-audit.json'),calibration_raw_audit_sha256=sha(R/'native-raw-audit.json'),handoff_proof_sha256=sha(R/'handoff-proof.json'),candidate_rows_added=4,steady_candidates=2,nonsteady_candidates=2,precision_rejections=2,imported_short_points=6,new_long_points=8,unique_model_points=16,unique_raw_dispatches_audited=64,per_dataset_counts_include_shared_cases=True,all503_choice_records_unchanged=True,no_fresh_nominees=True,no_performance_promotion=True,strict_zero_eligible=0,rejection_report=str(reject_path),rejection_report_sha256=sha(reject_path),script_sha256=sha(pathlib.Path(__file__)))
    new['continuation_round183_qualification']=qual;new['generated_remote_epoch']=time.time()
    tmp=P.with_name(P.name+'.round183.tmp');tmp.write_text(json.dumps(new,indent=2,allow_nan=False)+'\n');os.replace(tmp,P)
    out=dict(qual,new_map_sha256=sha(P),changed_choices=[],deferred_128cube_preserved=True);(R/'integration-round183.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
