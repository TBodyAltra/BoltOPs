import os,sys,json,time,traceback,re,hashlib,importlib
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import torch,triton
root=Path.cwd();session=root.parent.parent;sys.path.insert(0,str(session));sys.path.insert(0,'/xukang/code/pmd-triton-sessions')
from perf_metrics_20261019 import mean_stats
cfg=json.loads((root/'request.json').read_text());row=dict(request=cfg,status='STARTING',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());start=time.monotonic()
def progress(phase):
 (root/'progress.json').write_text(json.dumps(dict(phase=phase,epoch=time.time())))
def save():
 row['wall_seconds']=time.monotonic()-start;(root/'result.json').write_text(json.dumps(row,indent=2,default=str))
try:
 torch.set_num_threads(1);torch.manual_seed(cfg.get('seed',20));m,n,k=[cfg[x] for x in ['M','N','K']];bm,bn,bk=[cfg[x] for x in ['BM','BN','BK']]
 assert k%128==0 and bk in [32,64,128] and m<=256 and n<=512 and k<=8192,'bounded pilot only'
 assert (m,n,k) not in [(6,4096,2048),(80,512,7168),(4608,3072,6144),(5120,4096,8192)],'full requirements excluded'
 grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),);assert grid[0]<=16
 g=k//128;a=(torch.randn(m,k)*.5).to(torch.float8_e4m3fn);w=(torch.randn(n,k)*.5).to(torch.float8_e4m3fn)
 ag=1 if cfg.get('a_per_row') else g
 sa=torch.randint(125,130,(m,ag),dtype=torch.uint8) if cfg.get('a_encoded') else torch.rand(m,ag)*.2+.05
 sb=torch.randint(125,130,(n,g),dtype=torch.uint8) if cfg.get('b_encoded',True) else torch.rand(n,g)*.2+.05
 saf=torch.pow(2.,sa.float()-127) if cfg.get('a_encoded') else sa;sbf=torch.pow(2.,sb.float()-127) if cfg.get('b_encoded',True) else sb
 A=a.cuda();W=w.cuda();SA=sa.T.contiguous().cuda().T if cfg.get('a_layout')=='M' else sa.cuda();SB=sb.T.contiguous().cuda().T if cfg.get('b_layout')=='N' else sb.cuda();C=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
 row.update(scale_a_stride=list(SA.stride()),scale_b_stride=list(SB.stride()),grid=grid,actual_CTA_count=grid[0],logical_flops=2*m*n*k,output_elements=m*n)
 source=cfg.get('source','trial_aiter_dotfirst');module=importlib.import_module(source);kernel=getattr(module,'_gemm_a8w8_blockscale_kernel' if cfg['api']=='aiter' else '_w8a8_triton_block_scaled_mm');row['kernel_source']=str(session/(source+'.py'));row['kernel_source_sha256']=hashlib.sha256((session/(source+'.py')).read_bytes()).hexdigest()
 mode=cfg['mode'];special=mode in ['wasp44','wasp_custom','wdra44','wdra48'];opts=dict(BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,GROUP_SIZE_M=1,num_stages=cfg.get('stages',2),num_warps=cfg.get('warps',4),A_SCALE_PER_ROW=cfg.get('a_per_row',False),WARP_SPECIALIZE=special)
 if special:opts.update(wasp_partition_warps=tuple(cfg.get('partition_warps',(4,4,4) if mode=='wdra48' else (4,4))),wasp_wdra=not mode.startswith('wasp'))
 if cfg.get('regs'):opts['wasp_partition_regs']=tuple(cfg['regs'])
 if cfg.get('compiler_options'):opts.update(cfg['compiler_options'])
 if cfg.get('async'):opts['use_async_copy']=True
 if cfg.get('bpp'):opts['use_block_pingpong']=True
 if cfg['api']=='aiter':
  opts.update(GROUP_K=128,GROUP_N=1,NUM_KSPLIT=1,SPLITK_BLOCK_SIZE=k,cache_modifier=cfg.get('cache_modifier',''))
  args=(A,W.T,C,SA,SB.T,m,n,k,A.stride(0),A.stride(1),W.stride(1),W.stride(0),0,C.stride(0),C.stride(1),SA.stride(0),SA.stride(1),SB.stride(1),SB.stride(0))
 else:
  args=(A,W,C,SA,SB,m,n,k,1,128,A.stride(0),A.stride(1),W.stride(1),W.stride(0),C.stride(0),C.stride(1),SA.stride(0),SA.stride(1),SB.stride(1),SB.stride(0))
 row['effective_options']=opts;progress('JIT compile only');t=time.monotonic();compiled=kernel.warmup(*args,grid=grid,**opts);row['compile_seconds']=time.monotonic()-t;row['metadata']=compiled.metadata._asdict();row['lds_bytes']=compiled.metadata.shared
 ir=root/'ir';ir.mkdir(exist_ok=True)
 for key in ['ttir','ttgir','llir','amdgcn']:
  if key in compiled.asm:(ir/('kernel.'+key)).write_text(compiled.asm[key])
 row['binary_sha256']=hashlib.sha256(compiled.asm['hsaco']).hexdigest();(root/'executed.hsaco').write_bytes(compiled.asm['hsaco']);compiled._init_handles();row.update(spill=compiled.n_spills,vgpr_descriptor=compiled.n_regs)
 asm=compiled.asm['amdgcn'];row['native_counts']={key:len(re.findall(p,asm)) for key,p in dict(mmac=r'\bv_mmac',buffer_load=r'\bbuffer_load',buffer_load_dword=r'\bbuffer_load_dword\b',buffer_load_dwordx4=r'\bbuffer_load_dwordx4\b',async_lds=r'\b(?:buffer_load[^\n]*lds|buffer_load_lds[^\n]*)',lds_read=r'\bds_read',abarrier=r'\bs_abarrier',scratch=r'\bscratch_',packed_mul=r'\bv_pk_mul_f32',packed_fma=r'\bv_pk_fma_f32').items()}
 row['dynamic_partition_vgprs']=[int(x) for x in re.findall(r's_set_vgpr_size (\d+)',asm)];row['features']=dict(wasp=bool(compiled.metadata.wasp_enabled),num_warps=compiled.metadata.num_warps)
 row['runtime_properties']=triton.runtime.driver.active.utils.get_device_properties(0);row['pmd_lds_capacity']=131072
 row['status']='COMPILED';save()
 if row['lds_bytes']>min(row['runtime_properties']['max_shared_mem'] if 'max_shared_mem' in row['runtime_properties'] else row['runtime_properties']['max_shared_mem_per_block'] if 'max_shared_mem_per_block' in row['runtime_properties'] else row['runtime_properties'].get('max_shared_mem',262144),131072):
  row['status']='EXCLUDED_LDS_CAPACITY';save();sys.exit(0)
 if special and not row['features']['wasp']:
  row['status']='NOT_SELECTED_WASP';save();sys.exit(0)
 if row['spill']>0:
  row['status']='EXCLUDED_SPILL';save();sys.exit(0)
 progress('CPU golden');ref=torch.zeros(m,n)
 for j in range(g):ref+=(a.float()[:,j*128:(j+1)*128]@w.float()[:,j*128:(j+1)*128].T)*saf[:,0 if cfg.get('a_per_row') else j,None]*sbf[:,j][None,:]
 ref=ref.bfloat16().float();progress('warmup and accuracy');kernel[grid](*args,**opts);torch.cuda.synchronize();got=C.cpu().float();diff=(got-ref).abs();row.update(max_abs=float(diff.max()),mismatch=int((diff!=0).sum()),violations=int((diff>.01+.01*ref.abs()).sum()),finite=bool(torch.isfinite(got).all()));torch.save(got,root/'output.pt');assert row['finite'] and row['violations']==0,'numerical check failed'
 row['accuracy']='PASS';save();progress('modeled timings')
 for _ in range(cfg.get('repeats',3)):kernel[grid](*args,**opts);torch.cuda.synchronize()
 row.update(mean_stats(root,list(range(1,1+cfg.get('repeats',3)))));row['status']='PASS';row['cycles_per_output']=row['cycles']/(m*n);progress('COMPLETE')
except Exception as e:
 row.update(status='FAILED' if row.get('accuracy') or row.get('status')=='COMPILED' else 'COMPILE_FAILED',error_type=type(e).__name__,error=traceback.format_exc());print(row['error'],flush=True)
finally:save()
print(json.dumps({k:row.get(k) for k in ['status','cycles','cycle_samples','max_abs','mismatch','violations','spill','lds_bytes','native_counts','error']},default=str),flush=True)
