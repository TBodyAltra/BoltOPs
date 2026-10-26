import pathlib,json,hashlib,re,subprocess,sys,os,shutil
R=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,q):p.write_text(json.dumps(q,indent=2,default=str)+'\n')
def fields(s):
 out={}
 for n in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count','amdhsa_next_free_vgpr']:
  m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',s);out[n]=int(m.group(1)) if m else None
 return out
def code(s):
 lines=[]
 for l in s.splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',l):break
  l=l.split(';',1)[0].strip()
  if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(l)
 return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
plan=json.loads((R/'plan.json').read_text());refs=json.loads((R/'references.json').read_text())
files=[R/n for n in ['prepare.py','screen.py','job.py','env.sh','plan.json','reference-map.json','references.json','compile_probe.py','native/libtriton.so','triton-source.patch','bolt_ops-source.patch','bolt_ops/boltops/gemm/triton/gemm_fp16.py']]
files+=list((R/'triton/third_party/hcu/backend').glob('*.py'))
hashes={str(p):sha(p) for p in files}
def run(job):
 assert shutil.disk_usage(R).free>20*2**30
 ident=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:16];p=R/'probes'/job['label']/ident;p.mkdir(parents=True,exist_ok=False);save(p/'job.json',job)
 with (p/'compiler.log').open('w') as f:rc=subprocess.run([sys.executable,str(R/'compile_probe.py'),str(p/'job.json')],stdout=f,stderr=subprocess.STDOUT).returncode
 x=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else dict(status='ERROR',returncode=rc)
 x['directory']=str(p);x['compiler_returncode']=rc
 if x['status']=='COMPILED':
  assert x['loaded_native']['sha256']=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
  x['all_physical_spills_zero']=all(x['fields'][k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count'])
 print(json.dumps(dict(label=job['label'],tile=job['tile'],status=x['status'],fields=x.get('fields'),shared=x.get('metadata',{}).get('shared'),config={k:job['config'][k] for k in ['LOOP_UNROLL_FACTOR','optimize_epilogue','amdgpu_enable_max_ilp_scheduling_strategy','hcu_use_gfx946_sched_model_v2']})),flush=True)
 return x
controls=[]
for ref in refs:
 x=run(dict(tile=ref['tile'],layout=ref['layout'],config=ref['config'],label='controls'))
 old=(R/('baseline-'+str(ref['tile'][1])+'.amdgcn')).read_text()
 x['normalized_ISA_equal']=x['status']=='COMPILED' and code((pathlib.Path(x['directory'])/'kernel.amdgcn').read_text())==code(old)
 x['resource_equal']=x.get('fields')==fields(old) and x.get('metadata',{}).get('shared')==ref['shared_bytes']
 controls.append(x);save(R/'control-qualification.json',dict(passed=all(z['normalized_ISA_equal'] and z['resource_equal'] for z in controls),controls=controls))
 assert x['normalized_ISA_equal'] and x['resource_equal'],x
rows=[]
for job in plan['features']:
 rows.append(run(job));save(R/'screen-progress.json',dict(completed=len(rows),planned=len(plan['features']),results=rows))
assert all(sha(pathlib.Path(p))==h for p,h in hashes.items())
eligible=[x for x in rows if x['status']=='COMPILED' and x['all_physical_spills_zero'] and x['metadata']['shared']<=65536]
save(R/'cpu-screen.json',dict(passed=True,CPU_only=True,no_PMD_execution=True,default_controls_exact=True,source_hashes=hashes,source_hashes_unchanged=True,results=rows,eligible=eligible,not_a_performance_result=True))
save(R/'cpu-complete.json',dict(passed=True,scope='CPU_ONLY_DEFAULT_CONTROLS_EXACT_RESOURCE_SCREEN',controls=2,features=len(rows),eligible=len(eligible),screen_sha256=sha(R/'cpu-screen.json'),native_admission=False))
