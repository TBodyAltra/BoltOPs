from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;B=R.parent;P=B/'single-tg-pipeline-20261028-round321-small-work1-fragment-native';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(P/'job-exit.json');assert e['returncode']==0 and not Path('/proc',str(e['pid'])).exists();a=load(P/'native-raw-audit.json');sel=load(P/'selection-audit.json');assert a['passed'] and a['points']==242 and len(sel['nominees'])==3 and sel['performance_nominees']==3;q=load(P/'cpu-admission.json');assert all(sha(Path(p))==v for p,v in q['source_hashes'].items());refs=[]
for n in sel['nominees']:
 c=n['candidate'];sid=c['source_ids'][0];ch=next(x for x in load(P/'features_short-qualification.json')['checks'] if x['id']==sid);ref=next(x for x in q['results'] if x['directory']==ch['CPU_reference']);assert ref['tile']==c['tile'] and c['tile'][2]==128 and ref['layout']==c['layout'] and c['layout'] in ['TT','TN'];refs.append(ref)
assert len(refs)==3 and shutil.disk_usage(R).free>24*2**30
for d in ['runs','native']:(R/d).mkdir(exist_ok=True)
for d in ['triton','bolt_ops','deps','session.lock']:
 assert not (R/d).exists();(R/d).symlink_to((P/d).resolve())
(R/'native/libtriton.so').symlink_to((P/'native/libtriton.so').resolve());assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
h=dict(q['source_hashes'])
for f in ['native-raw-audit.json','selection-audit.json','pipeline-candidates.json','features_short-qualification.json','cpu-admission.json','active-manifest.json','physical-fragment-contract-proof.json']:h[str(P/f)]=sha(P/f)
(R/'cpu-admission.json').write_text(json.dumps(dict(q,results=refs,source_hashes=h),indent=2)+'\n');(R/'cpu-controls.json').write_bytes((P/'cpu-controls.json').read_bytes());(R/'initial-selection.json').write_bytes((P/'selection-audit.json').read_bytes());(R/'layout3-review.json').write_text(json.dumps(dict(passed=True,initial_raw_audit_sha256=sha(P/'native-raw-audit.json'),one_independent_series_only=True,no_retry_nonsteady=True),indent=2)+'\n')
for f in ['triton-source.patch','bolt_ops-source.patch','control.json','reference-map.json']:(R/f).write_bytes((P/f).read_bytes())
(R/'env.sh').write_text((P/'env.sh').read_text().replace(str(P),str(R)).replace('321smallwork1fragmentnative','328smallfragmentconfirm'));(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,features=3,controls=2,initial_context=load(P/'active-manifest.json')['measurement_context_id'],one_independent_series=True,no_pilot_or_edge_retries=True,initial_odd17_edges_preserved=True,no_map_promotion_before_fresh_audit_and_executable_proof=True,max_models=8,max_compilers=1),indent=2)+'\n');print('Prepared294:128x32 TT/TN,one independent16/32/64 series each,initial edge proof preserved')
