"""Freeze backend180 and admit only actually novel, bounded-resource recipes."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round180-sgpr-liveness-cpu';P=B/'single-tg-pipeline-20261027-round176-group1-index-native'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
e=load(C/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
q=load(C/'cpu-screen.json');review=load(C/'liveness-review.json');assert review['passed'] and review['CPU_screen_sha256']==sha(C/'cpu-screen.json') and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
refs=review['selected'];assert len(refs)==6 and len(review['duplicates'])==4 and review['strict_physical_zero']==0
for x in refs:assert x['fields']['private_segment_fixed_size']==x['fields']['vgpr_spill_count']==0 and x['metadata']['shared']==65536 and x['config']['hcu_sgpr_liveness_strategy'] in [1,3,5]
for n in ['logs','runs','native']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
    assert not(R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')==review['native_sha256']=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['worker.py','scan.py','stats_io.py']:(R/n).write_bytes((P/n).read_bytes())
p=R/'worker.py';s=p.read_text();s=s.replace('import hashlib\n','import hashlib\nimport fcntl\n');a='assert set(compiler_options) <= {';assert s.count(a)==1;s=s.replace(a,'assert set(compiler_options) <= {"hcu_sgpr_liveness_strategy", ')
a='    cfg.update(compiler_options)';assert s.count(a)==1;s=s.replace(a,'''    if "hcu_sgpr_liveness_strategy" in compiler_options:
        strategy = compiler_options["hcu_sgpr_liveness_strategy"]
        assert type(strategy) is int and strategy in range(6)
        if strategy:
            assert spec["mode"] == "nowasp" and spec["mls"]
'''+a)
a='        kernel = fn.warmup(*args, grid=grid, **cfg)';assert s.count(a)==1;s=s.replace(a,'''        compiler_lock_path = Path(__file__).resolve().parent.parent / "single-tg-continuation-20261023/compiler.lock"
        with compiler_lock_path.open("a") as compiler_lock:
            fcntl.flock(compiler_lock, fcntl.LOCK_EX)
            result["compile_serialization"] = dict(lock=str(compiler_lock_path), acquired_remote_epoch=time.time(), acquired_monotonic_ns=time.monotonic_ns(), pid=os.getpid())
            save()
            kernel = fn.warmup(*args, grid=grid, **cfg)
            result["compile_serialization"]["completed_remote_epoch"] = time.time()
            result["compile_serialization"]["completed_monotonic_ns"] = time.monotonic_ns()
''');p.write_text(s);compile(s,str(p),'exec')
p=R/'scan.py';s=p.read_text().replace('"index-review.json"','"liveness-review.json"').replace("['index-mapping-proof.json','kernel-index-fix.patch']","['index-mapping-proof.json','kernel-index-fix.patch','backend-liveness.patch','clang-help-proof.json']").replace('GROUP1 index mapping source experiment; per-candidate physical spills preserved','backend SGPR liveness strategies1/3/5 with generic/GROUP1 mappings; actual SGPR spills retained; serial compiler lock')
p.write_text(s);compile(s,str(p),'exec')
for n in ['triton-source.patch','bolt_ops-source.patch','native-reuse.json','liveness-review.json','index-mapping-proof.json','kernel-index-fix.patch','backend-liveness.patch','clang-help-proof.json']:(R/n).write_bytes((C/n).read_bytes())
(R/'cpu-admission.json').write_text(json.dumps(dict(q,results=refs,build=dict(sha256=review['native_sha256'])),indent=2)+'\n');(R/'cpu-controls.json').write_bytes((C/'control-qualification.json').read_bytes())
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('180sgprlivenesscpu','181sgprlivenessnative'))
(R/'control.json').write_bytes((P/'control.json').read_bytes());(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_source=str(C),CPU_screen_sha256=sha(C/'cpu-screen.json'),liveness_review_sha256=sha(C/'liveness-review.json'),native_sha256=review['native_sha256'],GPU_sources_frozen=True,controls=2,features=6,strict_zero_candidates=0,actual_SGPR_spills=[x['fields']['sgpr_spill_count'] for x in refs],worker_only_changes='Typed backend-strategy option; serialize compilation under common compiler.lock; measured GPU code frozen',pilot='minimumSGPR2 generic mapping argument-lowering strategy1',max_models=8,max_compilers=1,full_gold_before_long_calibration=True,independent_fresh_for_new_winners_required=True,no_automatic_map_promotion=True),indent=2)+'\n')
print('Prepared181:2 exact defaults,one SGPR2 pilot,6 genuinely novel strategies; max8 models/1 compiler; actual spills2/4/18/140/141 retained.')
