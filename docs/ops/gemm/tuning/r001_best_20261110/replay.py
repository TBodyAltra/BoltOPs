from pathlib import Path
import os
os.environ.setdefault('TRITON_CACHE_DIR',str(Path(__file__).resolve().parent/'cache'))
import json,sys,importlib.util,time,re,traceback,configparser,hashlib
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));j=json.loads(Path(sys.argv[1]).read_text());out=dict(j,status='STARTING',validation=[],benchmarks=[])
def save():Path('result.tmp').write_text(json.dumps(out,indent=2,default=str));Path('result.tmp').replace('result.json')
def module(name,file):
 s=importlib.util.spec_from_file_location(name,R/file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def stats():
 files=sorted(Path('m5out/0').glob('*/stats.txt'),key=lambda p:int(p.parent.name));p=files[-1];v={}
 for l in p.read_text().splitlines():
  a=l.split()
  if len(a)>1 and a[0] in ['system.gcSimCycles','system.simTicks','system.finalTick','system.shaders.firstWaveStartTick','system.shaders.lastWaveEndTick']:
   v[a[0]]=float(a[1])
 completed=sum(int(x) for x in re.findall(r'^system\.shaders\.CUs\d+\.completedWGs\s+(\d+)',p.read_text(),re.M))
 assert completed==grid, f'incomplete grid: {completed} != {grid}'
 parser=configparser.RawConfigParser(strict=False);parser.read('m5out/config.ini');period=int(parser.get('system.shaders.clk_domain','clock'))
 return dict(completed_TGs=completed,dispatch_cycles=v['system.gcSimCycles'],dispatch_us=v['system.gcSimCycles']*period/1e6,shader_cycles=(v['system.shaders.lastWaveEndTick']-v['system.shaders.firstWaveStartTick'])/period,shader_us=(v['system.shaders.lastWaveEndTick']-v['system.shaders.firstWaveStartTick'])/1e6,clock_ps=period,path=str(p))
save()
try:
 import torch,triton
 torch.set_num_threads(1);m,n,k=j['shape'];fused=j.get('mode','fused')=='fused';cfg=dict(j['config']);ks=cfg['NUM_KSPLIT'];bm,bn=cfg['BLOCK_SIZE_M'],cfg['BLOCK_SIZE_N'];tiles=triton.cdiv(m,bm)*triton.cdiv(n,bn)
 a=torch.empty((m,k),dtype=torch.bfloat16,device='cuda');b=torch.empty((n,k),dtype=torch.bfloat16,device='cuda').T;storage=torch.full((256+m*n+256,),123.,dtype=torch.bfloat16,device='cuda');c=storage[256:256+m*n].view(m,n)
 partial=torch.empty((ks,m,n),dtype=torch.float32,device='cuda') if fused else None
 counters=torch.zeros(tiles,dtype=torch.int32,device='cuda') if fused else None
 sentinel=torch.full((m,n),float('nan'),dtype=torch.bfloat16);p_sentinel=torch.full((ks,m,n),float('nan'),dtype=torch.float32) if fused else None
 if fused:assert cfg['num_warps']==1
 fn=module('kernel_impl',j['source'])._gemm_bf16_fused_splitk_kernel
 args=(a,b,c,partial,counters,m,n,k,a.stride(0),a.stride(1),b.stride(0),b.stride(1),m*n,n,1,n,1);grid=tiles*cfg['NUM_KSPLIT']

 handle=fn.warmup(*args,grid=(grid,),**cfg)
 if j.get('asm_patch'):
  from triton.compiler.compiler import make_backend
  rewrite=module('asm_patch_impl',j['asm_patch_source']).rewrite
  original=handle.asm['amdgcn'];Path('original.amdgcn').write_text(original);Path('original.hsaco').write_bytes(handle.kernel)
  newasm,proof=rewrite(original,j['asm_patch']);out['asm_patch_proof']=proof
  if 'shared_bytes' in proof:handle.metadata=handle.metadata._replace(shared=proof['shared_bytes'])
  backend=make_backend(handle.metadata.target);options=backend.parse_options(cfg)
  handle.packed_metadata=backend.pack_metadata(handle.metadata)
  assert handle.packed_metadata[2]==handle.metadata.shared
  out['packed_launch_metadata']=list(handle.packed_metadata)
  hsaco=backend.make_hsaco(newasm,handle.metadata._asdict(),options)
  assert handle.module is None,'binary already loaded'
  cache=Path('patched-cache').resolve();cache.mkdir(exist_ok=True)
  import shutil
  group={}
  for key,path in handle.metadata_group.items():
   dst=cache/Path(path).name;shutil.copy2(path,dst);group[key]=str(dst)
  for key,dst in group.items():
   if dst.endswith('.amdgcn'):Path(dst).write_text(newasm)
   elif dst.endswith('.hsaco'):Path(dst).write_bytes(hsaco)
   elif dst.endswith('.json') and 'shared_bytes' in proof:
    md=json.loads(Path(dst).read_text());md['shared']=proof['shared_bytes'];Path(dst).write_text(json.dumps(md))
  handle.metadata_group=group;handle.perf_ir_path=str(cache)
  handle.asm['amdgcn']=newasm;handle.asm['hsaco']=hsaco;handle.kernel=hsaco
  out['parent_hsaco_sha256']=hashlib.sha256(Path('original.hsaco').read_bytes()).hexdigest() if Path('original.hsaco').exists() else None
 handle._init_handles()
 for ext in ['ttgir','llir','amdgcn']:Path('kernel.'+ext).write_text(handle.asm[ext])
 blob=handle.asm['hsaco'];Path('kernel.hsaco').write_bytes(blob if isinstance(blob,bytes) else Path(blob).read_bytes())
 asm=handle.asm['amdgcn']
 ins=[l.strip().split()[0] for l in asm.splitlines() if l.startswith('\t') and l.strip() and not l.lstrip().startswith(('.', ';'))]
 from collections import Counter
 hist=Counter(ins)
 out['isa_histogram']=dict(hist)
 out['static_isa_counts']=dict(VALU_non_MMAC=sum(v for k,v in hist.items() if k.startswith('v_') and not k.startswith('v_mmac')),SALU=sum(v for k,v in hist.items() if k.startswith('s_')),MMAC=sum(v for k,v in hist.items() if k.startswith('v_mmac')),v_mov_zero=sum(bool(re.match(r'\s*v_mov\S*\s+v\d+, 0$',l)) for l in asm.splitlines()),v_rcp=sum(v for k,v in hist.items() if k.startswith('v_rcp')),v_cndmask=sum(v for k,v in hist.items() if k.startswith('v_cndmask')))
 out.update(metadata=handle.metadata._asdict(),n_regs=handle.n_regs,n_spills=handle.n_spills,grid_TGs=grid,tensor_strides=dict(a=list(a.stride()),b=list(b.stride())),workspace_bytes=ks*m*n*4 if fused else 0,counter_bytes=tiles*4 if fused else 0,kernel_sha256=hashlib.sha256(Path('kernel.hsaco').read_bytes()).hexdigest(),effective_config=cfg,phase='execute');save()
 assert handle.n_spills==0,'SPILL_REJECTED: avoid spilled candidates in performance ranking'
 assert handle.metadata.shared<=131072,'LDS budget: 128 KiB'
 lines=[l.strip() for l in handle.asm['amdgcn'].splitlines() if l.startswith('\t') and l.strip() and not l.strip().startswith(('.', ';'))]
 mls=[i for i,l in enumerate(lines) if l.startswith('matrix_load_')]
 ds=[i for i,l in enumerate(lines) if l.startswith('ds_read_matrix')]
 mma=[i for i,l in enumerate(lines) if l.startswith('v_mmac')]
 out['issue_order']=dict(MLS_total=len(mls),MLS_before_first_DS=sum(i<ds[0] for i in mls) if ds else 0,all_MLS_before_DS=bool(ds and mls and max(mls)<min(ds)),DS_matrix_total=len(ds),DS_after_first_MMAC=sum(i>mma[0] for i in ds) if mma else 0)
 save()
 if j.get('require_front'):
  assert out['issue_order']['all_MLS_before_DS'],'ORDER_REJECTED: MLS remain after first DS read'
  assert out['issue_order']['DS_after_first_MMAC']>0,'ORDER_REJECTED: no batched DS read/MMAC'

 if k>0:assert 'v_mmac_f32_16x16x16_bf16' in handle.asm['amdgcn']
 def check(gold):
  assert bool((storage[:256].cpu()==123).all()) and bool((storage[-256:].cpu()==123).all()), 'output guard corrupted'
  actual=c.cpu().float();diff=(actual-gold).abs();bad=(~torch.isfinite(actual))|(diff>.001+.01*gold.abs());cnt=counters.cpu().tolist() if fused else []
  acc=dict(elements=actual.numel(),mismatches=int(bad.sum()),nonfinite=int((~torch.isfinite(actual)).sum()),max_abs=float(diff.max()),counters=cnt)
  if bad.any() or any(cnt):
   torch.save(dict(a=ac,b=bc,actual=actual,gold=gold,counters=cnt),'failure.pt');raise RuntimeError('precision/counter failure '+str(acc))
  return acc
 for i in range(3):
  gen=torch.Generator().manual_seed(j['seed']+1000*i);ac=(torch.rand((m,k),generator=gen)-.5).bfloat16();bc=(torch.rand((n,k),generator=gen)-.5).bfloat16().T;
  if i==1:ac=torch.ones((m,k),dtype=torch.bfloat16);bc=torch.ones((k,n),dtype=torch.bfloat16)
  a.copy_(ac);b.copy_(bc);c.copy_(sentinel)
  if fused:partial.copy_(p_sentinel) # Counters initialized once, never reset by host during reuse validation.
  gold=(ac.float()@bc.float()).bfloat16().float();launched=fn[(grid,)](*args,**cfg);assert launched is handle,'dispatch used another handle';assert hashlib.sha256(launched.kernel).hexdigest()==out['kernel_sha256'],'loaded binary changed';out['verified_dispatch_handle']=True;torch.cuda.synchronize()
  out['validation'].append(dict(input_seed=j['seed']+1000*i,accuracy=check(gold),stats=stats()));save()
 for rep in range(j.get("repeats",3)):
  launched=fn[(grid,)](*args,**cfg);assert launched is handle,'dispatch used another handle';assert hashlib.sha256(launched.kernel).hexdigest()==out['kernel_sha256'],'loaded binary changed';out['verified_dispatch_handle']=True;torch.cuda.synchronize();out['benchmarks'].append(dict(repeat=rep,accuracy=check(gold),stats=stats()));save()
 out.update(status='PASS',phase='complete')
except BaseException:out.update(status='ERROR',error=traceback.format_exc());traceback.print_exc()
finally:save();print(json.dumps(out,default=str),flush=True)
if out['status']!='PASS':raise SystemExit(1)
