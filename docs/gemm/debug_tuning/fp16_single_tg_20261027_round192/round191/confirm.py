"""Close the one independent strict-zero resource series, including any rejection."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(R/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
assert not (R/'confirmation.json').exists()
fresh_audit=load(R/'native-raw-audit.json');assert fresh_audit['passed'] and fresh_audit['points']==5 and fresh_audit['raw_dispatches']==20
p=load(R/'confirmation-plan.json');C=pathlib.Path(p['initial_root']);assert sha(C/'native-raw-audit.json')==p['initial_raw_audit_sha256'] and sha(C/'selection-audit.json')==p['selection_audit_sha256']
assert load(C/'native-raw-audit.json')['off_origin_edge_correctness_only_points']==13
a=load(C/'active-manifest.json');b=load(R/'active-manifest.json');assert a['measurement_context_id']!=b['measurement_context_id']
for root,active in [(C,a),(R,b)]:
    m=load(root/('manifest-'+active['manifest_id']+'.json'));assert all(sha(pathlib.Path(n))==h for n,h in m['hashes'].items())
initial=p['recipe'];series=load(R/'pipeline-candidates.json');assert len(series['candidates'])==1
fresh=series['candidates'][0];assert fresh['config']==initial['config'] and fresh['loop_counts']==initial['loop_counts']==[16,32,64] and fresh['shared_bytes']==initial['shared_bytes']==65536
checks=[];all_pids=[]
for i,j in zip(initial['source_ids'],fresh['source_ids']):
    old=load(C/'runs'/i/'result.json');new=load(R/'runs'/j/'result.json')
    assert old['asm_sha256']==new['asm_sha256']==sha(C/'runs'/i/'kernel.amdgcn')==sha(R/'runs'/j/'kernel.amdgcn')
    assert old['config']==new['config'] and old['shape']==new['shape'] and old['strides']==new['strides'] and old['loaded_native']['sha256']==new['loaded_native']['sha256']
    for root,ident in [(C,i),(R,j)]:
        pid=load(root/'runs'/ident/'executor.json')['child_pid'];assert not pathlib.Path('/proc',str(pid)).exists();all_pids.append(pid)
    checks.append(dict(initial_id=i,fresh_id=j,loops=old['pipeline_loops'],full_assembly_equal=True,full_gold=True))
assert len(set(all_pids))==6
zero=all(fresh['physical'][k]==initial['physical'][k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count']);assert zero
baseline=p['baseline'];initial_ratio=initial['mmac_theoretical_fraction']/baseline['mmac_theoretical_fraction'];fresh_ratio=fresh['mmac_theoretical_fraction']/baseline['mmac_theoretical_fraction']
ok=bool(initial['steady'] and fresh['steady'] and initial_ratio>=.98 and fresh_ratio>=.98 and zero)
controls=[]
oldctl=load(C/'default_controls-qualification.json')['checks'];newctl=load(R/'default_controls-qualification.json')['checks']
for x,y in zip(oldctl,newctl):
    old=load(C/'runs'/x['id']/'result.json');new=load(R/'runs'/y['id']/'result.json');assert x['passed'] and y['passed'] and old['config']==new['config'] and old['asm_sha256']==new['asm_sha256']
    controls.append(dict(initial_id=x['id'],fresh_id=y['id'],full_assembly_equal=True,initial_cycles=old['median_cycles'],fresh_cycles=new['median_cycles']))
out=dict(passed=True,resource_alternative_validated=ok,resource_alternative_only=True,no_performance_best_promotion=True,initial_root=str(C),fresh_root=str(R),initial_context=a['measurement_context_id'],fresh_context=b['measurement_context_id'],initial_raw_audit_sha256=sha(C/'native-raw-audit.json'),fresh_raw_audit_sha256=sha(R/'native-raw-audit.json'),one_fresh_series_only=True,initial_off_origin_edge_gates_preserved=True,initial=initial,fresh=fresh,baseline=baseline,initial_relative_throughput=initial_ratio,fresh_relative_throughput=fresh_ratio,all_physical_spills_zero=zero,full_assembly_and_runtime_contract_equal=True,model_PIDs_all_distinct=True,checks=checks,controls=controls,source_hashes_unchanged=True,rejection_reason=None if ok else 'Fresh nonsteady or throughput below98%: no resource admission, no additional repeat',script_sha256=sha(pathlib.Path(__file__)))
(R/'confirmation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=True,resource_alternative_validated=ok,initial_fraction=initial['mmac_theoretical_fraction'],fresh_fraction=fresh['mmac_theoretical_fraction'],initial_ratio=initial_ratio,fresh_ratio=fresh_ratio,fresh_residual=fresh['intermediate_residual_fraction_of_long_time'],physical_zero=zero,no_performance_promotion=True)))
