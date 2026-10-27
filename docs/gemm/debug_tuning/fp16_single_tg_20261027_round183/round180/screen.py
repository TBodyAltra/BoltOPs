import pathlib,json,hashlib,subprocess,sys,os,re,shutil
R=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,q):p.write_text(json.dumps(q,indent=2,default=str)+'\n')
def code(p):
 lines=[]
 for l in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',l):break
  l=l.split(';',1)[0].strip()
  if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(l)
 return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
refs=json.loads((R/'references.json').read_text());assert len(refs)==2
files=[R/n for n in ['prepare.py','screen.py','job.py','env.sh','compile_probe.py','feature-plan.json','references.json','reference-map.json','triton-source.patch','bolt_ops-source.patch','native/libtriton.so','bolt_ops/boltops/gemm/triton/gemm_fp16.py','index-mapping-proof.json','kernel-index-fix.patch','backend-liveness.patch','clang-help-proof.json']]+list((R/'triton/third_party/hcu/backend').glob('*.py'));hashes={str(p):sha(p) for p in files}
def run(ref,label):
 assert shutil.disk_usage(R).free>20*2**30
 job=json.loads((pathlib.Path(ref['directory'])/'job.json').read_text()) if label=='controls' else json.loads(json.dumps(ref));job['label']=label
 ident=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:16];p=R/'probes'/label/ident;p.mkdir(parents=True,exist_ok=False);save(p/'job.json',job)
 with (p/'compiler.log').open('w') as f:rc=subprocess.run([sys.executable,str(R/'compile_probe.py'),str(p/'job.json')],stdout=f,stderr=subprocess.STDOUT).returncode
 x=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else dict(status='ERROR',returncode=rc);x['directory']=str(p)
 if x['status']=='COMPILED':
  assert x['metadata']['misched_regpressure']==job['config'].get('misched_regpressure',True)
  assert x['metadata'].get('hcu_sgpr_liveness_strategy')==job['config'].get('hcu_sgpr_liveness_strategy',0)
  isa=(p/'kernel.amdgcn').read_text();x['scalar_resources']={}
  for name in ['amdhsa_next_free_sgpr','sgpr_count']:
   match=re.search(r'\.'+name+r'\s*[:=]?\s*(\d+)',isa);x['scalar_resources'][name]=int(match[1]) if match else None
  x['scalar_resources']['writelane_instructions']=len(re.findall(r'^\s*v_writelane_b32',isa,re.M))
  x['scalar_resources']['readlane_instructions']=len(re.findall(r'^\s*v_readlane_b32',isa,re.M))
  assert x['metadata'].get('sched_latency','none')==job['config'].get('sched_latency','none')
  assert x['metadata'].get('max_memory_cluster_dwords')==job['config'].get('max_memory_cluster_dwords')
  assert x['loaded_native']['sha256']=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990';x['all_physical_spills_zero']=all(x['fields'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']);x['normalized_ISA_sha256']=code(p/'kernel.amdgcn')
 if label=='controls':
  x['exact_default_control']=x['status']=='COMPILED' and x['fields']==ref['fields'] and x['metadata']['shared']==ref['metadata']['shared'] and x['metadata']['num_warps']==ref['metadata']['num_warps'] and code(p/'kernel.amdgcn')==code(pathlib.Path(ref['directory'])/'kernel.amdgcn');assert x['exact_default_control'],x
 print(json.dumps(dict(label=label,variant=job.get('variant'),status=x['status'],fields=x.get('fields'),shared=x.get('metadata',{}).get('shared'))),flush=True);return x
controls=[run(x,'controls') for x in refs];save(R/'control-qualification.json',dict(passed=True,controls=controls));rows=[]
for x in json.loads((R/'feature-plan.json').read_text())['features']:
 rows.append(run(x,'features'));save(R/'screen-progress.json',dict(completed=len(rows),total=len(json.loads((R/'feature-plan.json').read_text())['features']),results=rows))
assert all(sha(pathlib.Path(p))==h for p,h in hashes.items());eligible=[x for x in rows if x['status']=='COMPILED' and x['all_physical_spills_zero'] and x['metadata']['shared']<=65536]
save(R/'cpu-screen.json',dict(passed=True,CPU_only=True,default_controls_exact=True,results=rows,eligible=eligible,source_hashes=hashes,source_hashes_unchanged=True,no_PMD_execution=True,native_precision_and_performance_still_required=True));save(R/'cpu-complete.json',dict(passed=True,controls=2,features=len(rows),eligible=len(eligible),eligible_unique_normalized_ISA=len({x['normalized_ISA_sha256'] for x in eligible}),screen_sha256=sha(R/'cpu-screen.json'),native_admission=False))
