"""Admit deterministic MLS build only after repeated CPU compiler qualification."""
from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round209-deterministic-mls-cpu';P=B/'single-tg-pipeline-20261027-round207-fragment-layout3-native-retry'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(C/'cpu-complete.json')['passed'] and load(C/'determinism-qualification.json')['passed']
q=load(C/'cpu-screen.json');assert q['passed'] and all(sha(Path(p))==h for p,h in q['source_hashes'].items())
refs=q['eligible'];assert len(refs)==24
assert len({(x['layout'],x['config']['num_warps'],x['metadata']['shared'],x['normalized_ISA_sha256']) for x in refs})==24
assert all(x['all_physical_spills_zero'] and x['metadata']['shared']<=131072 for x in refs)
assert shutil.disk_usage(R).free>24*2**30
for n in ['logs','runs','native']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
 assert not (R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());pin=load(C/'build-result.json')['sha256'];assert sha(R/'native/libtriton.so')==pin
assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['worker.py','scan.py','stats_io.py','audit_native.py','select.py']:(R/n).write_bytes((P/n).read_bytes())
p=R/'worker.py';s=p.read_text();old='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990';assert s.count(old)==1;p.write_text(s.replace(old,pin))
p=R/'tune.py';s=(P/'tune.py').read_text().replace('fragment_layout3_native207','deterministic_fragment_layout3_native210');p.write_text(s)
for n in ['triton-source.patch','bolt_ops-source.patch','address-bounds-proof.json','kernel-fragment-stream.patch','fragment-coverage-proof.json']:(R/n).write_bytes((C/n).read_bytes())
(R/'kernel-rebase.patch').write_bytes((P/'kernel-rebase.patch').read_bytes());(R/'initial-stream-audit.json').write_bytes((P/'initial-stream-audit.json').read_bytes())
review=dict(passed=True,CPU_screen_sha256=sha(C/'cpu-screen.json'),native_sha256=pin,selected=refs,duplicates=[],new_native=True,determinism_repeats=4,physical_zero=True,CPU_only=True)
(R/'layout3-review.json').write_text(json.dumps(review,indent=2)+'\n');(R/'cpu-admission.json').write_text(json.dumps(dict(q,results=refs,build=dict(sha256=pin)),indent=2)+'\n');(R/'cpu-controls.json').write_text(json.dumps(dict(passed=True,controls=q['controls'],new_native=True,unchanged_from_native95=[x['unchanged_from_native95_normalized_ISA'] for x in q['controls']]),indent=2)+'\n')
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('209deterministicmlscpu','210deterministiclayout3native'));(R/'control.json').write_bytes((P/'control.json').read_bytes());(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
(R/'native-reuse.json').write_text(json.dumps(dict(native_sha256=pin,source=str(C),CPU_determinism=True,PMD_qualification_pending=True),indent=2)+'\n')
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,source_CPU=str(C),GPU_source_frozen=True,source_hashes_unchanged=True,controls=2,features=24,native_sha256=pin,native_change='Stable geometric insertion order for MLS descriptor groups',no_gate_relaxation=True,TT_TN_pilots_before_remaining=True,full_gold_and_odd17_edges_before_long=True,maximum_models=8,maximum_compilers=1,canonical_lease=True,no_automatic_map_promotion=True),indent=2)+'\n')
print('Prepared210:new pinned deterministic native; unchanged strict CPU/PMD ASM,resources,GOLD and useful-work gates')
