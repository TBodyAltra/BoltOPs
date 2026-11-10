"""Continue the original initial series using only four original passing recipes."""
import pathlib,json,hashlib,shlex
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round181-sgpr-liveness-native'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(C/'job-exit.json');assert e['returncode']==1 and not pathlib.Path('/proc',str(e['pid'])).exists()
a=load(C/'mixed-short-gate-audit.json');assert a['passed'] and a['points']==8 and a['raw_dispatches']==32 and a['feature_status_counts']==dict(PASS=4,ACCURACY_FAIL=2) and a['long_calibration_not_started']
act=load(C/'active-manifest.json');m=load(C/('manifest-'+act['manifest_id']+'.json'));assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
accepted={x['id']:x for x in a['rows'] if not x['control'] and x['status']=='PASS'};assert len(accepted)==4
short=load(C/'liveness_features_short-qualification.json');checks=[x for x in short['checks'] if x['id'] in accepted];assert len(checks)==4 and all(x['passed'] for x in checks)
directories={x['CPU_reference'] for x in checks};admission=load(C/'cpu-admission.json');refs=[x for x in admission['results'] if x['directory'] in directories];assert len(refs)==4 and all(x['config']['hcu_sgpr_liveness_strategy'] in [1,3] for x in refs)
for n in ['runs','logs']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock','native','scan.py','worker.py','stats_io.py']:
    assert not(R/n).exists();(R/n).symlink_to((C/n).resolve())
imports=[]
for row in a['rows']:
    if row['status']!='PASS':continue
    case=C/'runs'/row['id'];assert sha(case/'result.json')==row['result_sha256'] and sha(case/'kernel.amdgcn')==row['asm_sha256']
    ex=load(case/'executor.json');assert not pathlib.Path('/proc',str(ex['child_pid'])).exists()
    for p,h in row['raw_stats_hashes'].items():assert sha(pathlib.Path(p))==h
    dst=R/'runs'/row['id'];assert not dst.exists();dst.symlink_to(case)
    imports.append(dict(id=row['id'],source=str(case),result_sha256=row['result_sha256'],asm_sha256=row['asm_sha256'],control=row['control']))
assert len(imports)==6
for n in ['cpu-controls.json','default_controls-qualification.json','control.json','reference-map.json','triton-source.patch','bolt_ops-source.patch','native-reuse.json','liveness-review.json','index-mapping-proof.json','kernel-index-fix.patch','backend-liveness.patch','clang-help-proof.json']:(R/n).write_bytes((C/n).read_bytes())
admission['results']=refs;(R/'cpu-admission.json').write_text(json.dumps(admission,indent=2)+'\n')
(R/'liveness_features_short-qualification.json').write_text(json.dumps(dict(passed=True,checks=checks,source_parent=str(C),source_original_failed_gate_sha256=sha(C/'liveness_features_short-qualification.json'),scope='Only original four passing feature cases; no new short dispatch and no change to original failed gate'),indent=2)+'\n')
(R/'env.sh').write_text((C/'env.sh').read_text()+'\n# Preserve every parent measured environment value for initial-series continuation.\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in sorted(m['env'].items())))
controller_files={str(R/n):sha(R/n) for n in ['prepare.py','tune.py','job.py','env.sh','cpu-admission.json','liveness_features_short-qualification.json']}
proof=dict(passed=True,parent=str(C),parent_exit_sha256=sha(C/'job-exit.json'),parent_mixed_audit_sha256=sha(C/'mixed-short-gate-audit.json'),parent_manifest_id=act['manifest_id'],parent_context=act['measurement_context_id'],parent_manifest_sha256=sha(C/('manifest-'+act['manifest_id']+'.json')),unchanged_parent_scanner_worker_used=True,parent_measurement_environment_preserved=True,controller_files=controller_files,imported_cases=imports,original_failed_ids=a['rejected_feature_ids'],new_dispatches='Only4x64 and4x32 loops; original two controls and four16-loop passing cases reused once',independent_repeat=False,no_performance_map_promotion=True)
(R/'handoff-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,accepted_source=str(C),source_and_native_frozen=True,features=4,controls_reused=2,short_features_reused=4,short_points_repeated=0,new_long_points=8,maximum_models=4,maximum_compilers=1,handoff_proof_sha256=sha(R/'handoff-proof.json'),original_failed_cases_not_retried=True,no_automatic_map_promotion=True),indent=2)+'\n')
print('Prepared182: reuse6 original passing short points; unchanged181 scanner/worker/env/context; only8 new long points, strategy1/3. Fast failures preserved and not retried.')
