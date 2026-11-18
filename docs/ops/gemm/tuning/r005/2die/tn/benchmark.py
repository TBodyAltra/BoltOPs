"""Replay configured FP8 TN blockscale GEMM with full-output validation."""
from pathlib import Path
import sys,json,time,traceback,hashlib,re,os
r=Path.cwd();spec=json.loads(Path(sys.argv[1]).read_text());job=spec['benchmark']['job'] if 'benchmark' in spec else spec;result={'job':job,'status':'STARTING','measurements':[]}
def save():
 p=r/'result.tmp';p.write_text(json.dumps(result,indent=2,default=str));p.replace(r/'result.json')
def stats_files():return {p for p in (r/'m5out/0').glob('*/stats.txt')}
def read_stats(p):
 values={}
 for l in p.read_text().splitlines():
  f=l.split()
  if len(f)>1:
   try:values[f[0]]=float(f[1])
   except ValueError:pass
 counts={k:int(v) for k,v in values.items() if re.search(r'[.]shaders[0-9]*[.]CUs[0-9]+[.]completedWGs$',k)}
 periods=[v for k,v in values.items() if re.search(r'[.]shaders[0-9]*[.]clk_domain[.]clock$',k)]
 assert counts and periods and set(periods)=={455.},(p,periods)
 dies={}
 for k,v in counts.items():
  die=re.search(r'[.]shaders([0-9]*)[.]',k)[1] or '0';dies[die]=dies.get(die,0)+v
 return {'cycles':values['system.gcSimCycles'],'us':values['system.gcSimCycles']*455/1e6,'sim_ticks':values['system.simTicks'],'final_tick':values['system.finalTick'],'active_CUs':sum(v>0 for v in counts.values()),'completed_TGs':sum(counts.values()),'completed_TGs_per_die':dies,'path':str(p),'counts':counts}
def artifacts(handle,name):
 for ext in ['amdgcn','ttgir','llir']:
  if ext in handle.asm:(r/(name+'.'+ext)).write_text(handle.asm[ext])
 (r/(name+'.hsaco')).write_bytes(handle.asm['hsaco'])
 from triton.backends.hcu.wasp import read_partition_registers
 return {'metadata':handle.metadata._asdict(),'VGPR':handle.n_regs,'spill':handle.n_spills,'partition_registers':read_partition_registers(handle.asm['amdgcn']),'ISA_sha256':hashlib.sha256(handle.asm['amdgcn'].encode()).hexdigest()}
save()
try:
 import torch,triton,importlib
 from boltops.gemm import gemm_a8w8_blockscale
 from boltops.gemm.config import get_gemm_config
 from boltops.utility.arch_info import get_device_info
 torch.set_num_threads(1)
 m,n,k=job['shape'];device=get_device_info()
 gen=torch.Generator().manual_seed(job.get('seed',0))
 ac=(torch.randn((m,k),generator=gen)*.5).to(torch.float8_e4m3fn)
 bc=(torch.randn((n,k),generator=gen)*.5).to(torch.float8_e4m3fn)
 asc=(torch.rand((m,k//128),generator=gen)*1.5+.25).float()
 bsc=(torch.rand(((n+127)//128,k//128),generator=gen)*1.5+.25).bfloat16().float()
 golden=torch.zeros((m,n))
 for i in range(k//128):
  dot=ac[:,i*128:(i+1)*128].float()@bc[:,i*128:(i+1)*128].float().T
  golden+=dot*asc[:,i,None]*bsc[:,i].repeat_interleave(128)[:n][None,:]
 a,b,sa,sb=[t.cuda() for t in [ac,bc,asc,bsc]]
 storage=torch.full((m*n+512,),123.,dtype=torch.bfloat16,device='cuda')
 out=storage[256:-256].view(m,n)
 profile='DTYPE=float8_e4m3fn-AS=float32-BS=float32-OUT=bfloat16-RECIPE_A=1x128-RECIPE_B=128x128'
 cfg,tuned=get_gemm_config('GEMM-A8W8_BLOCKSCALE',m,n,k,profile=profile,backend='gluon',arch=device.arch,cu_num=device.cu_count)
 assert tuned
 resources={};module=importlib.import_module('boltops.gemm.gluon.gemm_a8w8_blockscale')
 name=cfg['kernel'];original_fn=getattr(module,name)
 class Capture:
  def __getitem__(self,grid):
   def call(*args,**kwargs):
    h=original_fn[grid](*args,**kwargs)
    if 'gemm' not in resources:resources['gemm']=artifacts(h,'gemm')
    result['public_kernel']=name
    return h
   return call
 setattr(module,name,Capture())
 result.update(status='COMPILED',public_config=cfg,resources=resources,
  environment={key:os.environ.get(key) for key in ['GPU_CHIP','GPU_DFLAGS','GPU_ARGS','PYTHONPATH','TRITON_HIP_CLANG_PATH','TRITON_HIP_LLD_PATH']},
  libtriton_sha256=hashlib.sha256(Path(triton.__file__).parent.joinpath('_C/libtriton.so').read_bytes()).hexdigest())
 save()
 for rep in range(job.get('repeats',3)+1):
  out.fill_(float('nan'));torch.cuda.synchronize();before=stats_files();t=time.time()
  gemm_a8w8_blockscale(a,b,sa,sb,out=out);torch.cuda.synchronize();elapsed=time.time()-t
  paths=sorted(stats_files()-before,key=lambda p:int(p.parent.name));stats=[read_stats(p) for p in paths]
  assert len(stats)==1 and stats[0]['completed_TGs']==triton.cdiv(m,cfg['BLOCK_SIZE_M'])*triton.cdiv(n,cfg['BLOCK_SIZE_N'])
  actual=out.cpu().float();err=(actual-golden).abs()
  bad=(~torch.isfinite(actual))|(err>.01+.02*golden.abs())
  guards=bool((storage[:256].cpu()==123).all() and (storage[-256:].cpu()==123).all())
  row={'repeat':rep,'cold':rep==0,'cycles':stats[0]['cycles'],'us':stats[0]['us'],
   'wall_seconds':elapsed,'stages':stats,'accuracy':{'max_abs':float(err.max()),'mismatches':int(bad.sum()),
   'nonfinite':int((~torch.isfinite(actual)).sum()),'guards_ok':guards,'elements':m*n},
   'output_sha256':hashlib.sha256(actual.numpy().tobytes()).hexdigest()}
  result['measurements'].append(row);save();assert not bad.any() and guards,row
 assert all(not v['spill'] for v in resources.values())
 result['status']='PASS'
except BaseException:
 result.update(status='ERROR',error=traceback.format_exc());traceback.print_exc()
finally:save()
if result['status']=='ERROR':raise SystemExit(1)
