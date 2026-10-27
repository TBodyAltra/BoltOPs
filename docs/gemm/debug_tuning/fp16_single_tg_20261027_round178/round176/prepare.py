"""Freeze qualified GROUP1 source and prepare an isolated PMD controller."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round175-group1-index-cpu';P=B/'single-tg-pipeline-20261027-round166-nowasp-scratch0-performance-native'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=json.loads((C/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
q=json.loads((C/'cpu-screen.json').read_text());review=json.loads((C/'index-review.json').read_text());assert review['passed'] and review['CPU_screen_sha256']==sha(C/'cpu-screen.json') and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
refs=review['selected'];assert len(refs)==8 and len(review['physical_zero'])==4 and len(review['scratch0_SGPR_spilling'])==4
for x in refs:assert x['fields']['private_segment_fixed_size']==x['fields']['vgpr_spill_count']==0 and x['metadata']['shared']<=65536 and x['config']['SIMPLIFY_GROUP1'] is True and x['config']['GROUP_SIZE_M']==1
for n in ['logs','runs','native']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
    assert not(R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve())
assert sha(R/'native/libtriton.so')==review['native_sha256']=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['worker.py','scan.py','stats_io.py']:(R/n).write_bytes((P/n).read_bytes())
p=R/'worker.py';s=p.read_text();anchor='    cfg.update(compiler_options)';assert s.count(anchor)==1
s=s.replace(anchor,'''    simplify = spec.get("simplify_group1", False)
    assert type(simplify) is bool
    if simplify:
        assert spec["mls"] and cfg["GROUP_SIZE_M"] == 1
    cfg["SIMPLIFY_GROUP1"] = simplify
'''+anchor);p.write_text(s)
p=R/'scan.py';s=p.read_text();a='"wasp_single_buffer", "optimize_epilogue")';assert s.count(a)==1;s=s.replace(a,'"wasp_single_buffer", "optimize_epilogue", "simplify_group1")')
s=s.replace('"scratch0-performance-admission.json"','"index-review.json"').replace("files.append(ROOT/'strict-zero-rejection-audit.json')","files.extend(ROOT/n for n in ['index-mapping-proof.json','kernel-index-fix.patch'])").replace('scratch0/VGPRspill0 SGPR-spilling performance exploration, no physical-zero claim','GROUP1 index mapping source experiment; per-candidate physical spills preserved')
p.write_text(s)
for n in ['triton-source.patch','bolt_ops-source.patch','native-reuse.json','index-review.json','index-mapping-proof.json','kernel-index-fix.patch']:(R/n).write_bytes((C/n).read_bytes())
(R/'cpu-admission.json').write_text(json.dumps(dict(q,results=refs,build=dict(sha256=review['native_sha256'])),indent=2)+'\n')
(R/'cpu-controls.json').write_bytes((C/'control-qualification.json').read_bytes())
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('175group1indexcpu','176group1indexnative'))
(R/'control.json').write_bytes((P/'control.json').read_bytes())
(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_source=str(C),CPU_screen_sha256=sha(C/'cpu-screen.json'),index_review_sha256=sha(C/'index-review.json'),native_sha256=review['native_sha256'],GPU_sources_frozen=True,controls=2,features=8,physical_zero_candidates=4,SGPR_spilling_candidates=4,worker_only_changes='Typed simplify_group1 constexpr; false default; included in point identity',all_legal_TG_index_mapping_proved=True,logical_BK_and_transport_unchanged=True,full_gold_pilot_before_remaining_features=True,independent_confirmation_required=True,no_automatic_map_promotion=True),indent=2)+'\n')
print('Prepared176:2 default controls,one GROUP1 pilot,8 source variants; per-candidate spill reporting, canonical lease, max8 followup.')
