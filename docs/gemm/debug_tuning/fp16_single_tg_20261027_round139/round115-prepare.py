import pathlib,json,hashlib,subprocess
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round114-alias-ab-slots-cpu';O=B/'single-tg-pipeline-20261026-round95-mls-destination-view-cpu';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((C/'destination-view-screen.json').read_text());assert json.loads((C/'job-exit.json').read_text())['returncode']==0 and json.loads((C/'cpu-complete.json').read_text())['passed'] and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());assert json.loads((C/'alias-overlap-counterexample.json').read_text())['counterexample_detected']
assert json.loads((B/'single-tg-pipeline-20261027-round102-slot-turns-native/native-raw-audit.json').read_text())['passed']
for name,commit in [('triton','46f0bed3561364b63ba4046d9c8c0cb70380ea30'),('bolt_ops','840d816b671983527e5b1acc831c5e67618fe981')]:
 assert not(R/name).exists();subprocess.run(['git','-C',str(C/name),'worktree','add','-b','perf/single-tg-alias-ab-slots-cpu-fix-round115-'+name,str(R/name),commit],check=True);subprocess.run(['git','-C',str(R/name),'apply',str(C/(name+'-source.patch'))],check=True)
(R/'native').mkdir(exist_ok=True);(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
old=C/'triton'
for p in (old/'python/triton').rglob('*'):
 if not p.is_symlink():continue
 target=p.readlink()
 try:target=R/'triton'/target.relative_to(old)
 except ValueError:target=p.resolve()
 if p.name=='libtriton.so':target=R/'native/libtriton.so'
 d=R/'triton'/p.relative_to(old);d.parent.mkdir(parents=True,exist_ok=True);d.symlink_to(target)
for p in (old/'third_party/hcu/backend').glob('_cy_utils*.so'):(R/'triton/third_party/hcu/backend'/p.name).symlink_to(p.resolve())
p=R/'triton/python/triton/backends/hcu';assert p.is_symlink();p.unlink();p.symlink_to(R/'triton/third_party/hcu/backend')
for n in ['deps','session.lock']:(R/n).symlink_to((C/n).resolve())
subprocess.run(['git','-C',str(R/'triton'),'add','-N','third_party/hcu/backend/split_b_global_probe.py','third_party/hcu/backend/split_b_slots.py','third_party/hcu/backend/split_b_slot_turn.py','third_party/hcu/backend/partial_k_local.py','third_party/hcu/backend/mmac_turn.py','third_party/hcu/backend/merge_ab.py','third_party/hcu/backend/prefetch_full_k.py','third_party/hcu/backend/first_operand_early.py','third_party/hcu/backend/prefetch_one_operand.py','third_party/hcu/backend/data_barrier_map.py','native-overlays'],check=True)
be=R/'triton/third_party/hcu/backend';(be/'alias_ab_slots.py').write_bytes((R/'alias_ab_slots.py').read_bytes());subprocess.run(['git','-C',str(R/'triton'),'add','-N','third_party/hcu/backend/alias_ab_slots.py'],check=True)
for name in ['triton','bolt_ops']:(R/(name+'-source.patch')).write_bytes(subprocess.check_output(['git','-C',str(R/name),'diff','HEAD','--binary']))
for n in ['compile_probe.py','llvm_addresses.py','audit_addresses.py','native-reuse.json']:(R/n).write_bytes((C/n).read_bytes())
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('114aliasabslotscpu','115aliasabslotscpufix'));(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
refs=json.loads((C/'qualification.json').read_text())['controls'];assert len(refs)==6
(R/'old-control-references.json').write_text(json.dumps(refs,indent=2)+'\n');(R/'predecessor-overlap-counterexample.json').write_bytes((C/'alias-overlap-counterexample.json').read_bytes());print('CPU115 fresh isolated worktrees: correct TN interleavedB1 ownership; CPU-only, no PMD admission.')
