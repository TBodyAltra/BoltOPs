import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round143-alias-no-turn-cpu';P=B/'single-tg-pipeline-20261027-round136-alias-late-turn-native'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=json.loads((C/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists()
q=json.loads((C/'cpu-screen.json').read_text());assert len(q['eligible'])==8 and q['default_controls_exact'] and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
for filename,script in [('parent-address-audit.json','audit_addresses.py'),('consumer-read-audit.json','audit_reads.py'),('ownership-isa-audit.json','audit_ownership.py')]:
 a=json.loads((C/filename).read_text());assert a['passed'] and a['source_hashes_unchanged'] and len(a['rows'])==8 and a.get('script_sha256',a.get('audit_sha256'))==sha(C/script)
model=json.loads((C/'alias-no-turn-protocol-model.json').read_text());assert model['passed'] and len(model['controls'])==10 and len(model['positive_cases'])==6 and len(model['negative_cases'])==54 and len(model['equivalent_guard_controls'])==32 and model['script_sha256']==sha(C/'protocol_alias_no_turn.py')
assert model['turn_ids']==[] and model['data_ids']==list(range(6))
assert json.loads((C/'ownership-isa-audit.json').read_text())['protocol_model_sha256']==sha(C/'alias-no-turn-protocol-model.json')
for n in ['logs','runs','native','controller-sources']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
 assert not(R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['triton-source.patch','bolt_ops-source.patch','native-reuse.json','parent-address-audit.json','consumer-read-audit.json','ownership-isa-audit.json','alias-no-turn-protocol-model.json','protocol_alias_no_turn.py','model-v1-classification-note.json']:(R/n).write_bytes((C/n).read_bytes())
for n in ['worker.py','scan.py','stats_io.py']:(R/n).write_bytes((P/n).read_bytes())
p=R/'scan.py';s=p.read_text();start=s.index('def capture_manifest():');end=s.index('\ndef spec(',start)
s=s[:start]+'''def capture_manifest():
    def cmd(*args):return subprocess.check_output(args,text=True).strip()
    repos={n:dict(commit=cmd("git","-C",str(ROOT/n),"rev-parse","HEAD"),status=cmd("git","-C",str(ROOT/n),"status","--porcelain"),tracked_diff_sha256=hashlib.sha256(subprocess.check_output(["git","-C",str(ROOT/n),"diff","HEAD","--binary"])).hexdigest()) for n in ["triton","bolt_ops"]}
    q=json.loads((ROOT/"cpu-admission.json").read_text());files=[Path(p) for p in q['source_hashes']]
    files.extend(ROOT/n for n in ["scan.py","worker.py","stats_io.py","tune.py","env.sh","control.json","native/libtriton.so","deps/pmd/libgem5.so","cpu-controls.json","cpu-admission.json","prepare-proof.json","triton-source.patch","bolt_ops-source.patch","reference-map.json","job.py","prepare.py","parent-address-audit.json","consumer-read-audit.json","ownership-isa-audit.json","alias-no-turn-protocol-model.json","protocol_alias_no_turn.py","model-v1-classification-note.json","triton/third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp"])
    files.append(Path(os.environ['TRITON_HIP_CLANG_PATH']))
    return dict(repos=repos,host=cmd('hostname'),execution='PMD',arch='gfx946',cu_per_die=48,statistics='gcSimCycles median3 excluding warmup; aliased singleTG no cross-role MMAC turn; data-only finite model;16/32/64 calibration; no automatic promotion',hashes={str(p):digest(p) for p in files},env={k:v for k,v in sorted(os.environ.items()) if k.startswith(("TRITON","HCU","HIP","GPU","PMD","AMD","OMP","OPENBLAS")) or k in ("PYTHONPATH","LD_LIBRARY_PATH","PATH","LLVM_SYSPATH")})

'''+s[end:];p.write_text(s)
admission=dict(q,results=q['eligible'],build=dict(sha256=sha(R/'native/libtriton.so')));(R/'cpu-admission.json').write_text(json.dumps(admission,indent=2)+'\n');(R/'cpu-controls.json').write_bytes((C/'control-qualification.json').read_bytes())
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('143aliasnoturncpu','146aliasnoturnnative'))
(R/'control.json').write_bytes((P/'control.json').read_bytes());(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
for p in [R/'scan.py',R/'worker.py',R/'tune.py']:compile(p.read_text(),str(p),'exec')
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_source=str(C),actual_source_worktree=str((C/'triton').resolve()),GPU_sources_and_native_read_only_frozen=True,native_sha256=sha(R/'native/libtriton.so'),CPU_screen_sha256=sha(C/'cpu-screen.json'),producer_address_audit_sha256=sha(C/'parent-address-audit.json'),consumer_read_audit_sha256=sha(C/'consumer-read-audit.json'),ownership_ISA_audit_sha256=sha(C/'ownership-isa-audit.json'),finite_data_only_model_sha256=sha(C/'alias-no-turn-protocol-model.json'),controller_only_changes='Same BM64-capable partial-K runner with original8/248/248 quotas; no cross-role MMACturn option; all data ownership guards retained',controls=8,eligible=8,no_GPU_source_modifications=True),indent=2)+'\n')
print('Prepared native146 frozenCPU133/qualifiedCPU143 no-MMAC-turn recipes, eight controls+eight candidates.')
