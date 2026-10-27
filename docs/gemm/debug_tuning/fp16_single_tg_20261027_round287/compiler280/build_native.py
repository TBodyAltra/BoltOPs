from pathlib import Path
import json,hashlib,subprocess,shlex,time,fcntl,shutil,os
R=Path(__file__).resolve().parent;B=R.parent;P=next(B.glob('*round209-*'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
with (B/'single-tg-continuation-20261023/compiler.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert shutil.disk_usage(R).free>24*2**30
 parent=P/'native/libtriton.so';assert sha(parent)=='e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a'
 plan=json.loads((P/'build-plan.json').read_text());link=list(plan['link']);cwd=Path(plan['link_cwd'])
 stable=dict(plan['unchanged_link_inputs']);stable[plan['replacement']['new_object']]=sha(plan['replacement']['new_object']);assert len(stable)==504;assert all(sha(p)==h for p,h in stable.items())
 old=next(p for p in stable if '/third_party/hcu/' in p and p.endswith('/BlockPingpong.cpp.o'))
 src=R/'triton/third_party/hcu/lib/TritonHCUTransforms/BlockPingpong.cpp';obj=R/'native/BlockPingpong.cpp.o';so=R/'native/libtriton.so'
 commands=json.loads((B/'fp16-map-20261020/build/compile_commands.json').read_text());cc=next(c for c in commands if '/third_party/hcu/' in c['file'] and c['file'].endswith('/BlockPingpong.cpp'));args=shlex.split(cc['command']);args[args.index('-o')+1]=str(obj);args[args.index('-c')+1]=str(src);args.extend(['-MMD','-MF',str(R/'native/BlockPingpong.cpp.d')])
 link[link.index('-o')+1]=str(so);replaced=0
 for i,a in enumerate(link):
  if a.endswith('.o') and str((cwd/a).resolve())==old:link[i]=str(obj);replaced+=1
 assert replaced==1
 source_sha=sha(src);save('build-plan.json',dict(compile=args,compile_cwd=cc['directory'],link=link,link_cwd=str(cwd),parent_native_sha256=sha(parent),replacement=dict(original_object=old,original_sha256=stable[old],new_object=str(obj),source=str(src),source_sha256=source_sha),stable_parent_inputs=stable,unchanged_link_inputs={p:h for p,h in stable.items() if p!=old}))
 save('build-state.json',dict(status='COMPILING',pid=os.getpid()))
 with (R/'native/compile.log').open('w') as f:subprocess.run(args,cwd=cc['directory'],stdout=f,stderr=subprocess.STDOUT,check=True)
 deps=(R/'native/BlockPingpong.cpp.d').read_text().split(':',1)[1].replace('\\\n',' ');paths=[str((Path(cc['directory'])/p).resolve()) for p in shlex.split(deps)];dh={p:sha(p) for p in sorted(set(paths))};save('native/compile-dependency-hashes.json',dh)
 assert shutil.disk_usage(R).free>20*2**30;save('build-state.json',dict(status='LINKING',pid=os.getpid()))
 with (R/'native/link.log').open('w') as f:subprocess.run(link,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert all(sha(p)==h for p,h in stable.items()) and all(sha(p)==h for p,h in dh.items()) and sha(src)==source_sha and sha(parent)=='e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a'
 save('build-result.json',dict(passed=True,sha256=sha(so),parent_native_sha256=sha(parent),original_link_inputs=504,unchanged_link_inputs=503,replaced_object_sha256=sha(obj),source_sha256=source_sha,compile_dependency_files=len(dh),no_PMD_execution=True,not_a_full_rebuild=True))
 save('build-state.json',dict(status='NATIVE_BUILD_COMPLETE',pid=os.getpid()))
