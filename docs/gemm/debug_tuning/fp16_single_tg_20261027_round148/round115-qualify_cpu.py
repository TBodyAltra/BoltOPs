import pathlib,json,hashlib,subprocess,sys,os,re,shutil
R=pathlib.Path(__file__).resolve().parent;C=R.parent/'single-tg-pipeline-20261027-round114-alias-ab-slots-cpu'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,q):(R/n).write_text(json.dumps(q,indent=2,default=str)+'\n')
def code(p):
 lines=[]
 for l in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',l):break
  l=l.split(';',1)[0].strip()
  if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(l)
 return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
old=json.loads((C/'destination-view-screen.json').read_text());refs=json.loads((R/'old-control-references.json').read_text());assert len(refs)==6
be=R/'triton/third_party/hcu/backend'
files=[R/n for n in ['prepare.py','qualify_cpu.py','job.py','native/libtriton.so','compile_probe.py','llvm_addresses.py','audit_addresses.py','env.sh','old-control-references.json','reference-map.json','native-reuse.json','triton-source.patch','bolt_ops-source.patch','triton/third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp','bolt_ops/boltops/gemm/triton/gemm_fp16.py']]
files+=[be/n for n in ['compiler.py','wasp.py','partial_k_local.py','data_barrier_map.py','prefetch_one_operand.py','mmac_turn.py','prefetch_full_k.py','first_operand_early.py','merge_ab.py','split_b_global_probe.py','split_b_slots.py','split_b_slot_turn.py','alias_ab_slots.py']]
files += [pathlib.Path(p) for p in old['source_hashes']]+[C/'alias-overlap-counterexample.json'];hashes={str(p):sha(p) for p in files}
native=json.loads((R/'native-reuse.json').read_text())
def execute(ref,label,mode=0):
 assert shutil.disk_usage(R).free>20*2**30
 job=json.loads((pathlib.Path(ref['directory'])/'job.json').read_text())
 if label=='features':job['config'].update(hcu_wasp_alias_ab_slots=True)
 ident=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:16];p=R/'probes'/label/ident;p.mkdir(parents=True,exist_ok=False);save(str(p/'job.json'),job)
 with (p/'compiler.log').open('w') as f:rc=subprocess.run([sys.executable,str(R/'compile_probe.py'),str(p/'job.json')],env=dict(os.environ,TRITON_SPLIT_B_GLOBAL_PROBE_DUMP_DIR=str(p/'split-b-global-proof'),TRITON_SPLIT_B_SLOTS_DUMP_DIR=str(p/'split-b-slots-proof'),TRITON_SLOT_MMAC_TURN_DUMP_DIR=str(p/'slot-turn-proof'),TRITON_ALIAS_AB_SLOTS_DUMP_DIR=str(p/'alias-ab-proof')),stdout=f,stderr=subprocess.STDOUT).returncode
 x=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else dict(status='ERROR',rc=rc);x['directory']=str(p)
 equal=bool(rc==0 and x['status']=='COMPILED' and x['loaded_native']['sha256']==native['sha256'] and x['metadata']['shared']==(65536 if label=='features' else ref['metadata']['shared']) and x['metadata']['num_warps']==ref['metadata']['num_warps'] and all(x['fields'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']))
 if label=='controls':equal=bool(equal and x['fields']==ref['fields'] and x['metadata'].get('wasp_partition_regs_actual')==ref['metadata'].get('wasp_partition_regs_actual') and code(p/'kernel.amdgcn')==code(pathlib.Path(ref['directory'])/'kernel.amdgcn'))
 if label=='features' and equal:
  proof=json.loads((p/'alias-ab-proof/proof.json').read_text());assert proof['passed'] and proof['module_verified'] and proof['expected_shared_bytes']==65536
  if mode:
   turn=json.loads((p/'slot-turn-proof/proof.json').read_text());assert turn['mode']==2 and turn['wait_mask']==0 and turn['turn_ids']==[6,7] and turn['data_and_math_unchanged']

 x['control_equal']=equal;x['reference_directory']=ref['directory'];print(json.dumps(dict(label=label,tile=x['tile'],layout=x['layout'],status=x['status'],equal=equal)),flush=True);assert equal,x
 return x
controls=[execute(x,'controls') for x in refs];save('qualification.json',dict(passed=True,CPU_only=True,controls=controls,native_sha256=native['sha256']))
rows=[execute(x,'features',mode) for x in refs for mode in [1]];assert all(sha(pathlib.Path(p))==h for p,h in hashes.items())
save('destination-view-screen.json',dict(default_qualified=True,source_hashes_unchanged=True,source_hashes=hashes,CPU_only=True,build=native,results=rows,full_split_slot_ring=True))
save('cpu-complete.json',dict(passed=True,scope='COMPILE_SCREEN_ONLY_PENDING_ALIAS_PROTOCOL_AND_ACTUAL_ADDRESSES',controls=6,features=6,screen_sha256=sha(R/'destination-view-screen.json'),native_admission=False))
