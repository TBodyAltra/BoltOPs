"""CPU-only extension of the unchanged qualified reader to skinny block geometry."""
import pathlib,json,hashlib,subprocess,sys,os,re,fcntl,shutil,traceback,itertools
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261026-round89-one-operand-no-turn-cpu'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def code(p):
 lines=[]
 for s in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',s):break
  s=s.split(';',1)[0].strip()
  if s and not s.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(s)
 return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
with (R/'job.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert shutil.disk_usage(R).free>24*2**30
  q=json.loads((C/'one-operand-no-turn-screen.json').read_text());assert q['default_qualified'] and q['source_hashes_unchanged'] and json.loads((C/'job-exit.json').read_text())['returncode']==0
  assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
  assert json.loads((C/'qualification.json').read_text())['passed'] and json.loads((C/'no-turn-static-audit.json').read_text())['passed']
  save('job-state.json',dict(status='PREPARING_IDENTICAL_SOURCE_WORKTREES',pid=os.getpid()))
  for name,commit in [('triton','46f0bed3561364b63ba4046d9c8c0cb70380ea30'),('bolt_ops','840d816b671983527e5b1acc831c5e67618fe981')]:
   assert not(R/name).exists();subprocess.run(['git','-C',str(C/name),'worktree','add','-b','perf/single-tg-skinny-one-operand-round92-'+name,str(R/name),commit],check=True);subprocess.run(['git','-C',str(R/name),'apply',str(C/(name+'-source.patch'))],check=True)
  old=C/'triton'
  for p in (old/'python/triton').rglob('*'):
   if not p.is_symlink():continue
   target=p.readlink()
   try:target=R/'triton'/target.relative_to(old)
   except ValueError:target=p.resolve()
   if p.name=='libtriton.so':target=p.resolve()
   d=R/'triton'/p.relative_to(old);d.parent.mkdir(parents=True,exist_ok=True);d.symlink_to(target)
  for p in (old/'third_party/hcu/backend').glob('_cy_utils*.so'):(R/'triton/third_party/hcu/backend'/p.name).symlink_to(p.resolve())
  p=R/'triton/python/triton/backends/hcu'
  if p.is_symlink():p.unlink()
  else:assert not p.exists()
  p.symlink_to(R/'triton/third_party/hcu/backend')
  for n in ['deps','session.lock']:(R/n).symlink_to(C/n)
  names=['compiler.py','wasp.py','data_barrier_map.py','partial_k_local.py','mmac_turn.py','merge_ab.py','prefetch_full_k.py','first_operand_early.py','prefetch_one_operand.py']
  for n in names:assert sha(R/'triton/third_party/hcu/backend'/n)==sha(C/'triton/third_party/hcu/backend'/n)
  assert sha(R/'triton/python/triton/_C/libtriton.so')=='622e3233d6b7d9f1153d75346d09d175f26606ef410389b3ab4fa290bb22e285'
  assert sha(R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py')==sha(C/'bolt_ops/boltops/gemm/triton/gemm_fp16.py')
  for n in ['compile_probe.py','triton-source.patch','bolt_ops-source.patch','protocol-proof.json']:(R/n).write_bytes((C/n).read_bytes())
  (R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
  files=[pathlib.Path(__file__),R/'compile_probe.py',R/'reference-map.json',R/'triton-source.patch',R/'bolt_ops-source.patch',R/'protocol-proof.json',R/'triton/python/triton/_C/libtriton.so',R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py',C/'one-operand-no-turn-screen.json',C/'no-turn-static-audit.json',C/'qualification.json'];files += [R/'triton/third_party/hcu/backend'/n for n in names]+[pathlib.Path(p) for p in q['source_hashes']];hashes={str(p):sha(p) for p in files}
  env=dict(os.environ,PYTHONPATH=str(R/'triton/python')+':'+str(R/'bolt_ops'),TRITON_HOME=str(R/'cache/home'))
  def execute(job,label):
   assert shutil.disk_usage(R).free>20*2**30
   ident=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:16];p=R/'probes'/label/ident;p.mkdir(parents=True,exist_ok=False);save(str(p/'job.json'),job)
   with (p/'compiler.log').open('w') as f:rc=subprocess.run([sys.executable,str(R/'compile_probe.py'),str(p/'job.json')],env=dict(env,TRITON_CACHE_DIR=str(R/'cache'/label),TRITON_PARTIAL_K_DUMP_DIR=str(p/'partial-k-proof'),TRITON_MMAC_TURN_DUMP_DIR=str(p/'mmac-turn-proof'),TRITON_MERGE_AB_DUMP_DIR=str(p/'merge-ab-proof'),TRITON_FULL_K_PREFETCH_DUMP_DIR=str(p/'full-k-prefetch-proof'),TRITON_FIRST_OPERAND_EARLY_DUMP_DIR=str(p/'first-operand-early-proof'),TRITON_ONE_OPERAND_PREFETCH_DUMP_DIR=str(p/'one-operand-prefetch-proof')),stdout=f,stderr=subprocess.STDOUT).returncode
   x=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else dict(status='PROCESS_FAILED',returncode=rc);x['directory']=str(p)
   if x['status']=='COMPILED':
    m=x['metadata'];x['fits_128k_and_vgpr_budget']=m['shared']<=131072 and sum(m.get('wasp_partition_regs') or [])<=512 and sum(m.get('wasp_partition_regs_actual') or [])<=512;x['zero_all_spills']=all(x['fields'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
   print(json.dumps(dict(label=label,tile=job['tile'],layout=job['layout'],mode=job['config'].get('hcu_wasp_mmac_turn'),status=x['status'],shared=x.get('metadata',{}).get('shared'),fields=x.get('fields'),error=x.get('exception'))),flush=True);return x
  refs=[]
  for tile in ([256,256,128],[128,256,128]):
   for lay in ['TT','TN']:
    for mode in [0,2]:
     pool=[x for x in q['results'] if x['status']=='COMPILED'] if tile!=[256,256,128] or mode==0 else json.loads((C/'qualification.json').read_text())['controls']
     refs.append(next(x for x in pool if x['tile']==tile and x['layout']==lay and x['config'].get('hcu_wasp_prefetch_one_operand')==1 and x['config'].get('hcu_wasp_mmac_turn')==mode and not x['config'].get('LOAD_B_FIRST',False)))
  assert len(refs)==8
  save('job-state.json',dict(status='EIGHT_IDENTICAL_SOURCE_CONTROL_REPEATS',pid=os.getpid()));controls=[]
  for ref in refs:
   job=json.loads((pathlib.Path(ref['directory'])/'job.json').read_text());x=execute(job,'controls');m=x.get('metadata',{});x['control_equal']=bool(x['status']=='COMPILED' and x['fields']==ref['fields'] and m['shared']==ref['metadata']['shared'] and m['num_warps']==ref['metadata']['num_warps'] and m.get('wasp_partition_regs_actual')==ref['metadata'].get('wasp_partition_regs_actual') and code(pathlib.Path(x['directory'])/'kernel.amdgcn')==code(pathlib.Path(ref['directory'])/'kernel.amdgcn'));x['reference_directory']=ref['directory'];controls.append(x)
  qualified=all(x['control_equal'] for x in controls);save('qualification.json',dict(passed=qualified,controls=controls,compiler_kernel_native_unchanged=True,previous68_controls_reused_by_source_hash=True));assert qualified
  base=next(x['config'] for x in q['results'] if x['status']=='COMPILED' and x['tile']==[128,256,128] and x['layout']=='TN' and x['config'].get('hcu_wasp_mmac_turn')==0 and x['config'].get('hcu_wasp_prefetch_one_operand')==1 and not x['config'].get('LOAD_B_FIRST',False))
  jobs=[]
  for tile,lay,order,pref,turn,mmac in itertools.product(([32,256,128],[256,32,128]),('TT','TN'),(False,True),(1,2),(0,2),(-1,3)):
   cfg=dict(base,BLOCK_SIZE_M=tile[0],BLOCK_SIZE_N=tile[1],BLOCK_SIZE_K=128,LOAD_B_FIRST=order,hcu_wasp_prefetch_one_operand=pref,hcu_wasp_mmac_turn=turn,mmac_layout_force=mmac)
   jobs.append(dict(tile=tile,layout=lay,config=cfg,no_pmd_execution=True,no_kernel_loading=True,policy='Skinny32/256 and256/32 singleTG; unchangedqualified reader/compiler/native. Source transformations must verify; physical resource+actual ownership/MMAC audit still required.'))
  assert len(jobs)==64;save('plan.json',dict(CPU_only=True,no_pmd_execution=True,no_kernel_loading=True,maximum_compilers=1,jobs=jobs,controls=8,source_changes=False))
  pilots=[j for j in jobs if not j['config']['LOAD_B_FIRST'] and j['config']['hcu_wasp_mmac_turn']==0 and j['config']['mmac_layout_force']==3];assert len(pilots)==8
  save('job-state.json',dict(status='SKINNY_GEOMETRY_CPU_PILOTS',pid=os.getpid()));first=[execute(j,'features') for j in pilots];done={json.dumps([x['tile'],x['layout'],x['config']],sort_keys=True) for x in first}
  viable=any(x['status']=='COMPILED' and x['zero_all_spills'] and x['fits_128k_and_vgpr_budget'] for x in first)
  save('geometry-pilot.json',dict(passed=viable,results=first,policy='If no pilot meets compiler/resource constraints, preserve failures and do not expand or waive assertions.'))
  rows=list(first)
  if viable:
   save('job-state.json',dict(status='SKINNY_CPU_SCREEN',pid=os.getpid()))
   rows += [execute(j,'features') for j in jobs if json.dumps([j['tile'],j['layout'],j['config']],sort_keys=True) not in done]
  assert all(sha(pathlib.Path(p))==h for p,h in hashes.items())
  save('skinny-one-operand-screen.json',dict(default_qualified=qualified,source_hashes_unchanged=True,source_hashes=hashes,CPU_only=True,no_pmd_execution=True,no_kernel_loading=True,maximum_compilers=1,results=rows,geometry_pilot_passed=viable,planned=64,executed=len(rows)))
  save('job-state.json',dict(status='CPU_SCREEN_COMPLETE',pid=os.getpid()));save('job-exit.json',dict(returncode=0,pid=os.getpid()))
 except BaseException:save('job-exit.json',dict(returncode=1,pid=os.getpid(),error=traceback.format_exc()));raise
