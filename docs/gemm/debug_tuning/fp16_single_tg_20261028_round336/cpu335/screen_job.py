from pathlib import Path
import subprocess,json,hashlib,re,fcntl,shutil,time,os,traceback
R=Path(__file__).resolve().parent;B=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,q):(R/n).write_text(json.dumps(q,indent=2)+'\n')
def code(p):
 a=[]
 for l in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',l):break
  l=l.split(';',1)[0].strip()
  if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):a.append(l)
 return hashlib.sha256('\n'.join(a).encode()).hexdigest()
with (R/'job.lock').open('a+') as joblock:
 fcntl.flock(joblock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert not (R/'cpu-screen.json').exists();assert shutil.disk_usage(R).free>24*2**30
  save('job-state.json',dict(pid=os.getpid(),phase='CPU_ONLY_SCREENING',epoch=time.time()))
  files=[R/n for n in ['screen_job.py','compile_probe.py','env.sh','feature-plan.json','control-jobs.json','native/libtriton.so','bolt_ops/boltops/gemm/triton/gemm_fp16.py','triton/third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp']]+list((R/'triton/third_party/hcu/backend').glob('*.py'));frozen={str(p):sha(p) for p in files};rows=[]
  controls=json.loads((R/'control-jobs.json').read_text());features=json.loads((R/'feature-plan.json').read_text())['features'];jobs=controls+features
  for i,j in enumerate(jobs):
   p=R/'probes'/('controls' if i<2 else 'features')/hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()[:16];p.mkdir(parents=True,exist_ok=False);save(str(p.relative_to(R)/'job.json'),j)
   with (B/'single-tg-continuation-20261023/compiler.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);assert shutil.disk_usage(R).free>20*2**30
    cmd='source '+str(R/'env.sh')+'; export TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR='+str(p/'cache')+'; python3 '+str(R/'compile_probe.py')+' '+str(p/'job.json')
    acquired=time.monotonic_ns()
    with (p/'compiler.log').open('w') as f:rc=subprocess.run(['bash','-lc',cmd],stdout=f,stderr=subprocess.STDOUT).returncode
    completed=time.monotonic_ns()
   x=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else dict(status='ERROR',returncode=rc)
   x.update(directory=str(p),compile_serial_interval=dict(acquired_monotonic_ns=acquired,completed_monotonic_ns=completed))
   if x['status']=='COMPILED':
    assert x['loaded_native']['sha256']=='e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a';x['normalized_ISA_sha256']=code(p/'kernel.amdgcn');x['all_physical_spills_zero']=all(x['fields'][k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count'])
   rows.append(x);save('screen-progress.json',dict(completed=len(rows),total=len(jobs),results=rows));print(json.dumps(dict(index=i,status=x['status'],variant=j['variant'],fields=x.get('fields'),shared=x.get('metadata',{}).get('shared'))),flush=True)
   if i<2:assert x['status']=='COMPILED',x
  assert all(sha(Path(p))==v for p,v in frozen.items());eligible=[x for x in rows[2:] if x['status']=='COMPILED' and x['all_physical_spills_zero'] and x['metadata']['shared']<=131072];seen={};selected=[];duplicates=[]
  for x in rows[:2]:
   if x['status']=='COMPILED':seen[(tuple(x['tile']),x['layout'],x['metadata']['num_warps'],x['metadata']['shared'],x['normalized_ISA_sha256'])]=x['directory']
  for x in eligible:
   key=(tuple(x['tile']),x['layout'],x['metadata']['num_warps'],x['metadata']['shared'],x['normalized_ISA_sha256'])
   if key in seen:duplicates.append(dict(directory=x['directory'],duplicate_of=seen[key]))
   else:selected.append(x);seen[key]=x['directory']
  excluded=[dict(directory=x['directory'],status=x['status'],fields=x.get('fields'),shared=x.get('metadata',{}).get('shared')) for x in rows[2:] if x not in eligible]
  save('cpu-screen.json',dict(passed=True,CPU_only=True,no_PMD_execution=True,controls=rows[:2],results=rows[2:],eligible=eligible,source_hashes=frozen,source_hashes_unchanged=True))
  save('small-cta-screen-review.json',dict(passed=True,CPU_only=True,CPU_screen_sha256=sha(R/'cpu-screen.json'),native_sha256=sha(R/'native/libtriton.so'),selected=selected,duplicates=duplicates,excluded=excluded,resource_budget_bytes=131072,no_PMD_admission=True))
  save('cpu-complete.json',dict(passed=True,features=len(features),controls=2,eligible=len(eligible),novel=len(selected),duplicates=len(duplicates),excluded=len(excluded),no_PMD_execution=True));save('job-state.json',dict(pid=os.getpid(),phase='CPU_COMPLETE',epoch=time.time()));save('job-exit.json',dict(pid=os.getpid(),returncode=0,epoch=time.time()))
 except BaseException:
  save('job-exit.json',dict(pid=os.getpid(),returncode=1,error=traceback.format_exc(),epoch=time.time()));raise
