from pathlib import Path
import json,subprocess,fcntl,shutil,hashlib,re,os,time
R=Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round203-fragment-layout3-cpu'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,q):(R/n).write_text(json.dumps(q,indent=2)+'\n')
def code(p):
 a=[]
 for l in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',l):break
  l=l.split(';',1)[0].strip()
  if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):a.append(l)
 return hashlib.sha256('\n'.join(a).encode()).hexdigest()
sourcefiles=[R/'compile_probe.py',R/'triton/third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp',R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py',R/'native/libtriton.so']+list((R/'triton/third_party/hcu/backend').glob('*.py'))
frozen={str(p):sha(p) for p in sourcefiles};rows=[]
with (B/'single-tg-continuation-20261023/compiler.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert shutil.disk_usage(R).free>24*2**30
 refs=json.loads((R/'references.json').read_text());features=json.loads((R/'feature-plan.json').read_text())['features'];pilot=json.loads((C/'probes/features/4c7c95c875d07b7f/job.json').read_text())
 jobs=[('repeat'+str(i),pilot,str(i%2)) for i in range(4)]+[('control'+str(i),json.loads((Path(ref['directory'])/'job.json').read_text()),'0') for i,ref in enumerate(refs)]+[('feature'+str(i),j,'0') for i,j in enumerate(features)]
 for label,job,preimport in jobs:
  p=R/'probes'/label;p.mkdir(parents=True,exist_ok=False);save(str(p.relative_to(R)/'job.json'),job)
  cmd='source '+str(R/'env.sh')+'; export TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR='+str(p/'cache')+' PREIMPORT_TORCH='+preimport+'; python3 '+str(R/'compile_probe.py')+' '+str(p/'job.json')
  with (p/'compiler.log').open('w') as f:rc=subprocess.run(['bash','-lc',cmd],stdout=f,stderr=subprocess.STDOUT).returncode
  x=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else {};assert rc==0 and x.get('status')=='COMPILED',dict(label=label,result=x,rc=rc)
  assert x['loaded_native']['sha256']==json.loads((R/'build-result.json').read_text())['sha256'];x.update(directory=str(p),probe_label=label,preimport_torch=preimport,normalized_ISA_sha256=code(p/'kernel.amdgcn'),all_physical_spills_zero=all(x['fields'][k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count']))
  rows.append(x);save('screen-progress.json',dict(completed=len(rows),total=len(jobs),results=rows));print(label,x['fields'],x['metadata']['shared'],x['normalized_ISA_sha256'],flush=True)
  if len(rows)==4:
   assert len({x['normalized_ISA_sha256'] for x in rows})==1,'New native still nondeterministic'
   save('determinism-qualification.json',dict(passed=True,CPU_only=True,no_PMD_execution=True,independent_processes=4,torch_preimport_variants=[False,True],normalized_ISA_sha256=rows[0]['normalized_ISA_sha256'],all_resources_equal=all(x['fields']==rows[0]['fields'] for x in rows),results=rows))
 assert all(sha(Path(p))==h for p,h in frozen.items())
 controls=rows[4:6]
 for ref,x in zip(refs,controls):x['unchanged_from_native95_normalized_ISA']=code(Path(ref['directory'])/'kernel.amdgcn')==x['normalized_ISA_sha256'];x['unchanged_from_native95_resources']=ref['fields']==x['fields'] and ref['metadata']['shared']==x['metadata']['shared']
 eligible=[x for x in rows[6:] if x['all_physical_spills_zero'] and x['metadata']['shared']<=131072]
 save('cpu-screen.json',dict(passed=True,CPU_only=True,no_PMD_execution=True,new_native_requires_PMD_qualification=True,determinism_processes=4,controls=controls,results=rows[6:],eligible=eligible,source_hashes=frozen,source_hashes_unchanged=True))
 save('cpu-complete.json',dict(passed=True,features=24,eligible=len(eligible),controls=2,determinism_repeats=4,native_sha256=sha(R/'native/libtriton.so')))
