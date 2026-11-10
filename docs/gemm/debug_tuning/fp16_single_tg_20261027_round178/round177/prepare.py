"""One independent series for the single preselected strict-zero resource recipe."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round176-group1-index-native'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
e=load(C/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
a=load(C/'native-raw-audit.json');s=load(C/'selection-audit.json');assert a['passed'] and a['points']==26 and a['raw_dispatches']==104 and s['passed'] and s['performance_nominees']==0 and s['resource_alternative_nominees']==1
c=s['nominee']['candidate'];assert c['steady'] and s['nominee']['relative_to_64KiB_best']>=.98 and all(c['physical'][k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count'])
for ident in c['source_ids']:
    ex=load(C/'runs'/ident/'executor.json');assert not pathlib.Path('/proc',str(ex['child_pid'])).exists()
act=load(C/'active-manifest.json');m=load(C/('manifest-'+act['manifest_id']+'.json'));assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
admission=load(C/'cpu-admission.json');refs=[x for x in admission['results'] if x['config']==c['config']];assert len(refs)==1
for n in ['logs','runs','native']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
    assert not (R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['worker.py','scan.py','stats_io.py','triton-source.patch','bolt_ops-source.patch','native-reuse.json','index-review.json','index-mapping-proof.json','kernel-index-fix.patch','cpu-controls.json','reference-map.json','control.json']:(R/n).write_bytes((C/n).read_bytes())
admission['results']=refs;(R/'cpu-admission.json').write_text(json.dumps(admission,indent=2)+'\n')
# Keep original import paths so independent model runs can compare complete ISA bytes.
env=(C/'env.sh').read_text()
env=env.replace('export FP16_MAP_ROOT='+str(C),'export FP16_MAP_ROOT='+str(R))
for suffix in ['cache/home','cache/kernels','build']:env=env.replace(str(C/suffix),str(R/suffix))
env=env.replace('176group1indexnative','177group1indexconfirm')
assert 'export PYTHONPATH='+str(C/'triton/python')+':'+str(C/'bolt_ops') in env
(R/'env.sh').write_text(env)
(R/'confirmation-plan.json').write_text(json.dumps(dict(initial_root=str(C),initial_context=act['measurement_context_id'],initial_raw_audit_sha256=sha(C/'native-raw-audit.json'),selection_audit_sha256=sha(C/'selection-audit.json'),resource_alternative_only=True,recipe=c,baseline=s['nominee']['baseline'],minimum_relative_throughput=.98,maximum_residual=.02,one_series_only=True,no_automatic_map_promotion=True),indent=2)+'\n')
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,independent_model_processes=True,private_results_and_cache=True,GPU_sources_read_only_frozen=True,original_GPU_import_paths_preserved=True,controls=2,features=1,native_sha256=sha(R/'native/libtriton.so'),initial_manifest_hashes_unchanged=True,confirmation_plan_sha256=sha(R/'confirmation-plan.json'),independent16_32_64=True,resource_alternative_only=True,no_automatic_map_promotion=True),indent=2)+'\n')
print('Prepared177:one independent physical-zero UF2/EPFalse GROUP1 recipe,2 exact controls,one16/64/32 series; preserve complete ISA and canonical lease.')
