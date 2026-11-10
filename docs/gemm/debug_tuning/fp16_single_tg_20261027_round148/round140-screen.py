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
refs=json.loads((R/'references.json').read_text());assert len(refs)==13
files=[R/n for n in ['prepare.py','screen.py','job.py','env.sh','compile_probe.py','protocol_alias_late_turn.py','protocol_full_B.py','feature-plan.json','references.json','reference-map.json','triton-source.patch','bolt_ops-source.patch','native/libtriton.so','triton/third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp','bolt_ops/boltops/gemm/triton/gemm_fp16.py']]+list((R/'triton/third_party/hcu/backend').glob('*.py'))
hashes={str(p):sha(p) for p in files}
def run(ref,label):
 assert shutil.disk_usage(R).free>20*2**30
 job=json.loads((pathlib.Path(ref['directory'])/'job.json').read_text()) if label=='controls' else json.loads(json.dumps(ref))
 job['label']=label;ident=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:16];p=R/'probes'/label/ident;p.mkdir(parents=True,exist_ok=False);save(p/'job.json',job)
 with (p/'compiler.log').open('w') as f:rc=subprocess.run([sys.executable,str(R/'compile_probe.py'),str(p/'job.json')],env=dict(os.environ,TRITON_ALIAS_AB_SLOTS_DUMP_DIR=str(p/'alias-ab-proof'),TRITON_ALIAS_B1_EARLY_DUMP_DIR=str(p/'early-b1-proof'),TRITON_SLOT_MMAC_TURN_DUMP_DIR=str(p/'slot-turn-proof'),TRITON_SPLIT_B_GLOBAL_PROBE_DUMP_DIR=str(p/'split-b-global-proof'),TRITON_SPLIT_B_SLOTS_DUMP_DIR=str(p/'split-b-slots-proof')),stdout=f,stderr=subprocess.STDOUT).returncode
 x=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else dict(status='ERROR',returncode=rc);x['directory']=str(p);x['quota_seed_directory']=job.get('quota_seed_directory');x['quota_requested']=job.get('quota_requested');x['quota_headroom']=job.get('quota_headroom')
 if x['status']=='COMPILED':
  assert x['loaded_native']['sha256']=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
  x['all_physical_spills_zero']=all(x['fields'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
  regs=x['metadata']['wasp_partition_regs'];assert len(regs)==3 and sum(regs)%12==0 and sum(regs)<=504
  assert x['metadata']['wasp_partition_warps']==[4,4,4] and x['metadata']['num_warps']==12
  llir=(p/'kernel.llir').read_text();isa=(p/'kernel.amdgcn').read_text()
  init=re.findall(r'call\s+void\s+@llvm\.hcu\.wdra\.init\(([^)]*)\)',llir);assert len(init)==1
  llvm_init=list(map(int,re.findall(r'i16\s+(\d+)',init[0])));assert llvm_init==regs+[0]
  llvm_sets=list(map(int,re.findall(r'call\s+void\s+@llvm\.hcu\.s\.set\.vgpr\.size\(i16\s+(\d+)\)',llir)))
  isa_sets=list(map(int,re.findall(r'^\s*s_set_vgpr_size\s+(\d+)',isa,re.M)))
  assert llvm_sets==isa_sets==regs and x['fields']['amdhsa_next_free_vgpr']==sum(regs)//3
  x['quota_descriptor_audit']=dict(passed=True,requested=x['metadata']['wasp_partition_regs_requested'],effective=regs,actual_demand=x['metadata']['wasp_partition_regs_actual'],LLVM_init=llvm_init,LLVM_partition_sets=llvm_sets,ISA_partition_sets=isa_sets,next_free_vgpr=x['fields']['amdhsa_next_free_vgpr'],aggregate_quota_per_SIMD=sum(regs),quota_smaller_than_504=sum(regs)<504,arithmetic_two_TG_VGPR_budget_fits=sum(regs)*2<=512,multi_TG_occupancy_not_measured=True)

 if label=='controls':
  x['exact_default_control']=x['status']=='COMPILED' and x['fields']==ref['fields'] and x['metadata']['shared']==ref['metadata']['shared'] and x['metadata'].get('wasp_partition_regs_actual')==ref['metadata'].get('wasp_partition_regs_actual') and code(p/'kernel.amdgcn')==code(pathlib.Path(ref['directory'])/'kernel.amdgcn')
  assert x['exact_default_control'],x
 elif x['status']=='COMPILED':
  proof=json.loads((p/'slot-turn-proof/proof.json').read_text());assert proof['mode']==2 and proof['wait_mask']==job['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0) and proof['data_and_math_unchanged']
  assert x['metadata']['shared']==65536
 print(json.dumps(dict(label=label,tile=job['tile'],layout=job['layout'],status=x['status'],fields=x.get('fields'),shared=x.get('metadata',{}).get('shared'))),flush=True)
 return x
controls=[run(x,'controls') for x in refs];save(R/'control-qualification.json',dict(passed=True,controls=controls))
plan=json.loads((R/'feature-plan.json').read_text());rows=[run(x,'features') for x in plan['features']];assert all(sha(pathlib.Path(p))==h for p,h in hashes.items())
eligible=[x for x in rows if x['status']=='COMPILED' and x['all_physical_spills_zero'] and x['metadata']['shared']==65536]
save(R/'cpu-screen.json',dict(passed=True,CPU_only=True,default_controls_exact=True,results=rows,eligible=eligible,source_hashes=hashes,source_hashes_unchanged=True,no_PMD_execution=True,actual_address_and_ISA_ownership_audits_still_required=True))
save(R/'cpu-complete.json',dict(passed=True,scope='COMPILE_SCREEN_ONLY_PENDING_ACTUAL_ADDRESS_AND_ISA_PROOFS',controls=13,features=26,eligible=len(eligible),screen_sha256=sha(R/'cpu-screen.json'),native_admission=False))
