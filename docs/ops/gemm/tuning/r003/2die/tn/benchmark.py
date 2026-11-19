from pathlib import Path
import argparse,json,os,sys,statistics,hashlib,re,configparser,shutil,itertools
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--record',required=True);p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=4);a=p.parse_args();record=json.loads(Path(a.record).read_text())
if 'R003_SBX2_TN_TUNE' not in os.environ:
 env=dict(os.environ)
 for k,v in record['environment']['environment_variables'].items():
  if v is None:env.pop(k,None)
  else:env[k]=v
 env.update(R003_SBX2_TN_TUNE='1',GPU_DFLAGS="['StatLog','TT0','Perf']",HSA_TOOLS_LIB='/opt/rocm/lib/xprofiler/libperf_gen_helper.so')
 os.execve(sys.executable,[sys.executable,*sys.argv],env)
repo=Path(a.repo);out=Path(a.output);out.mkdir(exist_ok=False);os.chdir(out);os.environ['TRITON_CACHE_DIR']=str(out/'cache');os.environ['TRITON_HOME']=str(out/'triton-home');sys.path.insert(0,str(repo))
import torch,importlib
public=importlib.import_module('boltops.gemm.gemm_a16w16');kernels=importlib.import_module('boltops.gemm.gluon.gemm_a16w16');torch.set_num_threads(1)
M,N,K=record['shape'];props=torch.cuda.get_device_properties(0);assert props.multi_processor_count==96;assert os.environ['GPU_CHIP']=='sbx2';cfg=json.loads((repo/'boltops/gemm/configs/gfx946/96cu/gluon/gemm_a16w16/GEMM-A16W16-N=64-K=2048-TN.json').read_text())['profiles']['DTYPE=bfloat16-OUT=bfloat16']['M_EQ_2560'];symbol=cfg['kernel'];fn=getattr(kernels,symbol)
# Pin relative A/B/C placement and 16-MiB alignment for an independent cohort.
common=torch.empty((64*1024*1024//2,),dtype=torch.bfloat16,device='cuda')
base=common.data_ptr();align=16*1024*1024;start=((base+align-1)//align*align-base)//2
A=common[start:start+M*K].view(M,K)
bstart=start+16*1024*1024//2;B=common[bstart:bstart+K*N].view(N,K)
cstart=start+18*1024*1024//2;storage=common[cstart:cstart+256+M*N+256];storage.fill_(123.);C=storage[256:-256].view(M,N)
assert A.data_ptr()%(16*1024*1024)==0 and B.data_ptr()-A.data_ptr()==16*1024*1024
assert A.data_ptr()+A.numel()*2<=B.data_ptr() and B.data_ptr()+B.numel()*2<=storage.data_ptr()
addresses=dict(A=A.data_ptr(),B=B.data_ptr(),C=C.data_ptr());variants={};current=None;events=[];phase="setup"
class Observe:
 def __getitem__(self,grid):
  launch=fn[grid]
  def invoke(*args,**kwargs):
   resolved=grid(kwargs) if callable(grid) else grid;assert tuple(resolved)==(variants[current]['grid'],)
   h=launch(*args,**kwargs);assert h is variants[current]['handle'];return h
  return invoke
setattr(kernels,symbol,Observe())
def save(status):
 result=dict(status=status,shape=[M,N,K],input_addresses=addresses,scope='R003 sbx2 TN: A[M,K], W[N,K] both K contiguous; no transpose/copy; shared aligned buffers; full grid',environment=record['environment'],rounds=a.rounds,phase=phase,events=events,variants={})
 for name,v in variants.items():
  row={k:x for k,x in v.items() if k!='handle'}
  if v['samples']:
   row['performance']={key:statistics.median(s[key] for s in v['samples']) for key in ['kernel_us','shader_us']}
  result['variants'][name]=row
 (out/'progress.json').write_text(json.dumps(result,indent=2)+'\n');return result
save('RUNNING')
def run(name):
 global current
 current=name;C.fill_(float('nan'));torch.cuda.synchronize();result=public.gemm_a16w16(A,B,out=C,backend='triton') if name=='default_triton' else public.gemm_a16w16(A,B,out=C,backend='auto');assert result is C;torch.cuda.synchronize()
 assert addresses==dict(A=A.data_ptr(),B=B.data_ptr(),C=C.data_ptr())
 paths=sorted(Path('m5out/0').glob('*/stats.txt'),key=lambda p:int(p.parent.name));path=paths[-1];text=path.read_text()
 cycles=float(re.search(r'^system.gcSimCycles\s+([\d.eE+-]+)',text,re.M)[1])
 cu_rows=re.findall(r'^(system.shaders\d+)[.]CUs\d+[.]completedWGs\s+(\d+)',text,re.M);assert len(cu_rows)==96,len(cu_rows)
 tg=sum(int(v) for _,v in cu_rows);assert tg==variants[name]['grid'],(tg,variants[name]['grid'])
 per_die={};active_per_die={}
 for die,v in cu_rows:per_die[die]=per_die.get(die,0)+int(v);active_per_die[die]=active_per_die.get(die,0)+(int(v)>0)
 assert len(per_die)==2 and all(v>0 for v in per_die.values()),per_die
 starts=[float(v) for v in re.findall(r'^system.shaders\d+[.]firstWaveStartTick\s+([\d.eE+-]+)',text,re.M)]
 ends=[float(v) for v in re.findall(r'^system.shaders\d+[.]lastWaveEndTick\s+([\d.eE+-]+)',text,re.M)];assert len(starts)==len(ends)==2
 ini=configparser.RawConfigParser(strict=False);ini.read('m5out/config.ini')
 for die in per_die:assert int(ini.get(die+'.clk_domain','clock'))==455
 actual=storage.cpu();assert bool((actual[:256]==123).all()) and bool((actual[-256:]==123).all());actual=actual[256:-256].view(M,N).float();diff=(actual-golden).abs();bad=(~torch.isfinite(actual))|(diff>.001+.01*golden.abs());accuracy=dict(elements=actual.numel(),mismatches=int(bad.sum()),nonfinite=int((~torch.isfinite(actual)).sum()),max_abs=float(diff.max()),canary_ok=True);assert not bool(bad.any()),accuracy
 perf=list(path.parent.glob('[0-9]*_*.perf'));assert len(perf)==1
 return dict(kernel_us=cycles*.000455,shader_us=(max(ends)-min(starts))/1e6,completed_TGs_per_die=per_die,active_CUs_per_die=active_per_die,completed_TGs=tg,stats_path=str(path),perf_path=str(perf[0]),accuracy=accuracy)

def make(bm,bn,pw,cw,bk=64,chunks=2,slots=3,mode='defer'):
 return dict(cfg,BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,CHUNKS=chunks,SLOTS=slots,PW=pw,CW=cw,wasp_partition_warps=[pw,cw],LTS=False,DEFER_RELEASE=mode=='defer')
def ident(c):
 mode='defer' if c['DEFER_RELEASE'] else 'original'
 return f"m{c['BLOCK_SIZE_M']}n{c['BLOCK_SIZE_N']}k{c['BLOCK_SIZE_K']}_p{c['PW']}c{c['CW']}_ch{c['CHUNKS']}s{c['SLOTS']}_{mode}"
def compile_config(config):
 name=ident(config)
 if name in variants:return name
 estimated=2*(config['BLOCK_SIZE_M']+config['BLOCK_SIZE_N'])*config['BLOCK_SIZE_K']*config['CHUNKS']*config['SLOTS']
 if estimated>131072:
  events.append(dict(phase=phase,name=name,status='filtered',reason='logical LDS exceeds 128 KiB',logical_shared=estimated));save('RUNNING');return None
 grid=(M//config['BLOCK_SIZE_M'])*(N//config['BLOCK_SIZE_N']);opts={k:tuple(v) if isinstance(v,list) else v for k,v in config.items() if k!='kernel'};opts['B_K_CONTIGUOUS']=True
 try:
  h=fn.warmup(A,B,C,M=M,N=N,K=K,grid=(grid,),**opts);h._init_handles();asm=h.asm['amdgcn'];resources={key:int(re.search(r'\.'+native+r':\s*(\d+)',asm)[1]) for key,native in [('VGPR','vgpr_count'),('SGPR','sgpr_count'),('VGPR_spills','vgpr_spill_count'),('SGPR_spills','sgpr_spill_count')]}
  if h.n_spills or resources['VGPR_spills'] or resources['SGPR_spills']:raise ValueError('register spill')
 except Exception as e:
  events.append(dict(phase=phase,name=name,status='compile_rejected',reason=repr(e)));save('RUNNING');print(json.dumps(events[-1]),flush=True);return None
 d=out/name;d.mkdir();(d/'kernel.amdgcn').write_text(asm);(d/'kernel.hsaco').write_bytes(h.kernel)
 variants[name]=dict(config=config,grid=grid,handle=h,resources=resources,binary_sha256=hashlib.sha256(h.kernel).hexdigest(),validation=[],samples=[],screening=[],warmup=[])
 events.append(dict(phase=phase,name=name,status='compiled'));save('RUNNING');return name

def inputs(seed):
 global golden
 gen=torch.Generator().manual_seed(seed);ac=(torch.rand((M,K),generator=gen)-.5).bfloat16();bc=(torch.rand((N,K),generator=gen)-.5).bfloat16()
 if seed==-1:ac.fill_(1);bc.fill_(1)
 golden=(ac.float()@bc.float().T).bfloat16().float();A.copy_(ac);B.copy_(bc)

def screen(config):
 name=compile_config(config)
 if name is None or variants[name]['screening']:return
 warm=run(name);variants[name]['warmup'].append(warm);save('RUNNING');sample=run(name);variants[name]['screening'].append(sample);events.append(dict(phase=phase,name=name,status='screened',kernel_us=sample['kernel_us']));save('RUNNING');print(json.dumps(events[-1]),flush=True)


# Capture and retain the unmodified public auto->Triton default as the control.
from boltops.gemm.config import get_gemm_config
from boltops.gemm.triton.gemm_config_utils import compute_splitk_params
control_config,_=get_gemm_config('GEMM-A16W16',M,backend='triton',arch='gfx946',cu_num=96)
control_config=compute_splitk_params(control_config,K);assert control_config['NUM_KSPLIT']==1
control_grid=((M+control_config['BLOCK_SIZE_M']-1)//control_config['BLOCK_SIZE_M'])*((N+control_config['BLOCK_SIZE_N']-1)//control_config['BLOCK_SIZE_N'])
tmod=importlib.import_module('boltops.gemm.triton.gemm_a16w16');tf=tmod._gemm_a16_w16_kernel
class ObserveDefault:
 def __getitem__(self,grid):
  launch=tf[grid]
  def invoke(*args,**kwargs):
   assert current=='default_triton'
   resolved=grid(kwargs) if callable(grid) else grid;assert tuple(resolved)==(control_grid,)
   h=launch(*args,**kwargs);v=variants['default_triton']
   if v['handle'] is None:
    v['handle']=h;h._init_handles();asm=h.asm['amdgcn'];d=out/'default_triton';d.mkdir()
    (d/'kernel.amdgcn').write_text(asm);(d/'kernel.hsaco').write_bytes(h.kernel)
    v['binary_sha256']=hashlib.sha256(h.kernel).hexdigest();v['resources']={key:int(re.search(r'\.'+native+r':\s*(\d+)',asm)[1]) for key,native in [('VGPR','vgpr_count'),('SGPR','sgpr_count'),('VGPR_spills','vgpr_spill_count'),('SGPR_spills','sgpr_spill_count')]}
   assert v['handle'] is h;return h
  return invoke
setattr(tmod,'_gemm_a16_w16_kernel',ObserveDefault())
variants['default_triton']=dict(config=control_config,grid=control_grid,handle=None,resources={},binary_sha256=None,validation=[],samples=[],screening=[],warmup=[],backend='forced triton control')

best=cfg
from boltops.gemm.config import get_gemm_config as lookup
resolved,tuned=lookup('GEMM-A16W16',M,N,K,backend='gluon',arch='gfx946',cu_num=96,layout='TN',profile='DTYPE=bfloat16-OUT=bfloat16');assert tuned and resolved==best
name=compile_config(best);assert name is not None
variants['tuned_auto']=variants.pop(name);(out/name).rename(out/'tuned_auto')
variants['tuned_auto']['backend']='auto->gluon tuned TN'
route_checks={'target_TN':'public auto uses exact warmed profile'}
(out/'actual_environment.json').write_text(json.dumps(dict(pid=os.getpid(),environment_variables={k:os.environ.get(k) for k in record['environment']['environment_variables']}),indent=2)+'\n')
phase='api_confirm';inputs(2059)
for name in ['default_triton','tuned_auto']:
 sample=run(name);variants[name]['warmup'].append(sample);save('RUNNING')
for seed in [59,-1,2059]:
 inputs(seed)
 for name in ['default_triton','tuned_auto']:
  sample=run(name);sample['seed']=seed;variants[name]['validation'].append(sample);save('RUNNING');print(json.dumps(dict(phase=phase,name=name,status='validated',seed=seed)),flush=True)
for i in range(a.rounds):
 for name in (['default_triton','tuned_auto'] if i%2==0 else ['tuned_auto','default_triton']):
  sample=run(name);sample['round']=i;variants[name]['samples'].append(sample);save('RUNNING');print(json.dumps(dict(phase=phase,name=name,status='timed',round=i,kernel_us=sample['kernel_us'])),flush=True)
phase='complete';result=save('PASS');result['finalists']=['default_triton','tuned_auto'];result['ranking']=sorted(result['finalists'],key=lambda n:result['variants'][n]['performance']['kernel_us']);result['profile_config']=best;result['route_checks']=route_checks
for name in result['finalists']:
 v=variants[name];med=result['variants'][name]['performance']['kernel_us'];selected=min(v['samples'],key=lambda s:abs(s['kernel_us']-med));dst=out/name/'representative_full_grid.perf';shutil.copyfile(selected['perf_path'],dst);result['variants'][name]['selected_perf']=dict(path=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),sample=selected)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',performance={n:result['variants'][n]['performance'] for n in result['finalists']})),flush=True)
