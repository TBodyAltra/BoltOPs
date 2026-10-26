"""Replace exactly one MLS codegen object; preserve qualified ABI, parser and E4 inputs."""
import pathlib,json,hashlib,subprocess,shlex,time,fcntl,shutil,os
R=pathlib.Path(__file__).resolve().parent;B=R.parent;P=B/'single-tg-pipeline-20261026-round68-partial-k-parser-prototype'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,q):(R/n).write_text(json.dumps(q,indent=2)+'\n')
with (R/'native/build.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();assert shutil.disk_usage(R).free>24*2**30
 parent_so=P/'native/libtriton.so';assert sha(parent_so)=='622e3233d6b7d9f1153d75346d09d175f26606ef410389b3ab4fa290bb22e285'
 plan=json.loads((P/'build-plan.json').read_text());stable=json.loads((P/'native/link-input-hashes.json').read_text());assert len(stable)==504 and stable==plan['stable_link_inputs']
 save('build-state.json',dict(status='VERIFYING_504_ORIGINAL_LINK_INPUTS',pid=os.getpid()))
 assert all(sha(p)==h for p,h in stable.items())
 oldobj=next(p for p in stable if p.endswith('/MlsOpToLLVM.cpp.o'));assert stable[oldobj]=='f36afdcc70473dd1267f32ca1c632e3bb2ef2276849b148d962246fdc33945ca'
 src=R/'triton/third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp';obj=R/'native/MlsOpToLLVM.cpp.o';so=R/'native/libtriton.so';assert not obj.exists() and not so.exists()
 commands=json.loads((B/'fp16-map-20261020/build/compile_commands.json').read_text());cc=next(x for x in commands if x['file'].endswith('/MlsOpToLLVM.cpp'));args=shlex.split(cc['command']);oi=args.index('-o');ci=args.index('-c');args[oi+1]=str(obj);args[ci+1]=str(src);args.extend(['-MMD','-MF',str(R/'native/MlsOpToLLVM.cpp.d')])
 link=list(plan['link']);li=link.index('-o');link[li+1]=str(so);replaced=0
 for i,a in enumerate(link):
  if a.endswith('.o') and str((pathlib.Path(plan['link_cwd'])/a).resolve())==str(pathlib.Path(oldobj).resolve()):link[i]=str(obj);replaced+=1
 assert replaced==1
 out=dict(compile=args,compile_cwd=cc['directory'],link=link,link_cwd=plan['link_cwd'],qualified_base_native_sha256=sha(parent_so),replacement=dict(original_object=oldobj,original_sha256=stable[oldobj],new_object=str(obj),source=str(src),source_sha256=sha(src)),unchanged_link_inputs={p:h for p,h in stable.items() if p!=oldobj},original_qualified_plan_sha256=sha(P/'build-plan.json'),original_qualified_inputs_sha256=sha(P/'native/link-input-hashes.json'))
 save('build-plan.json',out);save('build-state.json',dict(status='COMPILING_ONLY_MLS_DESTINATION_FIX',pid=os.getpid()))
 with (R/'native/compile.log').open('w') as f:subprocess.run(args,cwd=cc['directory'],stdout=f,stderr=subprocess.STDOUT,check=True)
 deps=(R/'native/MlsOpToLLVM.cpp.d').read_text().split(':',1)[1].replace('\\\n',' ');dep_paths=[str((pathlib.Path(cc['directory'])/p).resolve()) for p in shlex.split(deps)];dep_hashes={p:sha(p) for p in sorted(set(dep_paths))};save('native/compile-dependency-hashes.json',dep_hashes)
 assert shutil.disk_usage(R).free>20*2**30;save('build-state.json',dict(status='LINKING_PRIVATE_NATIVE_WITH_503_UNCHANGED_INPUTS',pid=os.getpid()))
 with (R/'native/link.log').open('w') as f:subprocess.run(link,cwd=plan['link_cwd'],stdout=f,stderr=subprocess.STDOUT,check=True)
 assert all(sha(p)==h for p,h in stable.items()) and all(sha(p)==h for p,h in dep_hashes.items()) and sha(src)==out['replacement']['source_sha256']
 assert sha(parent_so)==out['qualified_base_native_sha256'];assert sha(so)!=sha(parent_so)
 proof=dict(status='BUILT_PRIVATE_MLS_PARENT_DESTINATION_FIX',sha256=sha(so),qualified_base_sha256=sha(parent_so),original_link_inputs=504,unchanged_link_inputs=503,replaced_object_sha256=sha(obj),source_sha256=sha(src),compile_dependency_files=len(dep_hashes),source_and_link_inputs_unchanged=True,parser_and_E4_objects_unchanged=True,seconds=time.time()-start,plan_sha256=sha(R/'build-plan.json'),output=str(so))
 save('build-result.json',proof);save('build-state.json',dict(status='NATIVE_BUILD_COMPLETE',pid=os.getpid()));print(json.dumps(proof))
