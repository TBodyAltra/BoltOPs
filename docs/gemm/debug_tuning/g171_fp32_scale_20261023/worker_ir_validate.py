from pathlib import Path
import os,sys,json,time,re,hashlib,traceback,importlib
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import torch,triton
r=Path.cwd();S=r.parent;B=S.parent;sys.path.insert(0,str(S));sys.path.insert(0,str(B))
from perf_metrics_20261019 import mean_stats
from guard import safety
cfg=json.loads((r/'request.json').read_text());result=dict(request=cfg,status='STARTING');t0=time.monotonic()
def save(phase):
 result['phase']=phase;result['wall_seconds']=time.monotonic()-t0
 p=r/'result.json.tmp';p.write_text(json.dumps(result,indent=2,default=str));p.replace(r/'result.json')
try:
 safety('start '+cfg['name']);torch.set_num_threads(1);torch.manual_seed(cfg.get('seed',20))
 m,n,k=128,256,6144;bm,bn,bk=128,256,128;gn,gk=128,128;groups=k//gk;assert cfg.get('M',m)==m and cfg.get('N',n)==n and cfg.get('K',k)==k
 a=(torch.randn(m,k)*.5).to(torch.float8_e4m3fn);b=(torch.randn(n,k)*.5).to(torch.float8_e4m3fn)
 sa=torch.rand(m,groups)*.2+.05;sb=torch.rand(triton.cdiv(n,gn),groups)*.2+.05
 assert sa.dtype==sb.dtype==torch.float32 and list(sb.shape)==[2,48]
 result['inputs']={'M':m,'N':n,'K':k,'A_dtype':str(a.dtype),'B_dtype':str(b.dtype),'output_dtype':'torch.bfloat16','A_scale_dtype':str(sa.dtype),'B_scale_dtype':str(sb.dtype),'A_scale_shape':list(sa.shape),'B_scale_shape':list(sb.shape),'A_scale_block_MK':[1,128],'B_scale_block_NK':[128,128],'A_scale_stride':list(sa.stride()),'B_scale_stride':list(sb.stride()),'B_scale_is_physical_compact':True,'fingerprints':{key:hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for key,t in dict(A=a,B=b,As=sa,Bs=sb).items()}}
 from boltops.gemm.triton.gemm_a8w8_blockscale_op import _get_config
 baseline_config=dict(_get_config(m,n,k)[0]);result['builtin_config']=baseline_config
 opts={**baseline_config,**dict(BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,GROUP_SIZE_M=1,NUM_KSPLIT=1,SPLITK_BLOCK_SIZE=k,GROUP_K=gk,GROUP_N=gn,cache_modifier='',num_warps=cfg.get('warps',4),num_stages=cfg.get('stages',2),A_SCALE_PER_ROW=False,WARP_SPECIALIZE=cfg.get('wdra',False))}
 if cfg.get('wdra'):opts.update(wasp_partition_warps=(4,4,4),wasp_partition_regs=(8,248,248),wasp_wdra=True)
 opts.update(cfg.get('compiler_options',{}));result['effective_options']=opts
 A=a.cuda();W=b.cuda();AS=sa.cuda();BS=sb.cuda();C=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
 mod=importlib.import_module(cfg.get('module','boltops.gemm.triton.gemm_a8w8_blockscale_op'));kernel=mod._gemm_a8w8_blockscale_kernel
 if cfg.get('module','boltops.gemm.triton.gemm_a8w8_blockscale_op').startswith('boltops.'):
  assert not cfg.get('wdra');opts.pop('WARP_SPECIALIZE',None)
 result['module_path']=mod.__file__;result['module_sha256']=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
 grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),);assert grid==(1,)
 args=(A,W.T,C,AS,BS.T,m,n,k,A.stride(0),A.stride(1),W.stride(1),W.stride(0),0,C.stride(0),C.stride(1),AS.stride(0),AS.stride(1),BS.stride(1),BS.stride(0))
 result['grid']=grid;save('compile');t=time.monotonic();from triton.backends.hcu.compiler import HIPBackend
 backend=HIPBackend(triton.runtime.driver.active.get_current_target());backend_opts=vars(backend.parse_options(opts));ir_source=S/cfg['ir_source'];result['executed_IR_source']=str(ir_source);result['executed_IR_sha256']=hashlib.sha256(ir_source.read_bytes()).hexdigest();compiled=triton.compile(str(ir_source),options=backend_opts);result['JIT_seconds']=time.monotonic()-t;result['metadata']=compiled.metadata._asdict();compiled._init_handles();result.update(spill=compiled.n_spills,vgpr_descriptor=compiled.n_regs,lds_bytes=compiled.metadata.shared,actual_warps=compiled.metadata.num_warps)
 ir=r/'ir';ir.mkdir(exist_ok=True)
 for key in ['ttir','ttgir','llir','amdgcn']:
  if key in compiled.asm:(ir/f'kernel.{key}').write_text(compiled.asm[key])
 (r/'executed.hsaco').write_bytes(compiled.asm['hsaco']);result['binary_sha256']=hashlib.sha256(compiled.asm['hsaco']).hexdigest();asm=compiled.asm['amdgcn'];result['dynamic_vgpr_sizes']=[int(x) for x in re.findall(r's_set_vgpr_size (\d+)',asm)];result['static_instructions']={k:len(re.findall(v,asm)) for k,v in {'MMAC':r'\bv_mmac','LDS_READ':r'\bds_read','LDS_MATRIX_READ':r'\bds_read_matrix','BUFFER_LOAD_DWORD':r'\bbuffer_load_dword\b','CVT_E8M0':r'\bv_cvt_pk_f32_e8m0','ASYNC_MLS':r'\bbuffer_load[^\n]*lds','EBARRIER':r'\bs_ebarrier','ABARRIER':r'\bs_abarrier','VMEM_SCRATCH':r'\bscratch_'}.items()};save('compiled')
 if result['spill'] or result['lds_bytes']>131072:
  result['status']='EXCLUDED_SPILL' if result['spill'] else 'EXCLUDED_LDS';save('excluded');sys.exit(0)
 if cfg.get('wdra'):assert compiled.metadata.wasp_enabled,'requested WDRA not selected'
 # CPU golden indexes the compact B scale by output channel group, never by channel.
 ref=torch.zeros(m,n);scale_n=torch.arange(n)//gn;tg=time.monotonic()
 for j in range(groups):ref+=(a.float()[:,j*gk:(j+1)*gk]@b.float()[:,j*gk:(j+1)*gk].T)*sa[:,j,None]*sb[scale_n,j][None,:]
 ref=ref.bfloat16().float();result['CPU_golden_seconds']=time.monotonic()-tg;save('accuracy_warmup')
 compiled[(1,1,1)](*[args[i] for i in [0,1,2,3,4,5,6,7,8,11,12,13,15,18]]);torch.cuda.synchronize();got=C.cpu().float();diff=(got-ref).abs();result.update(max_abs=float(diff.max()),mismatch=int((diff!=0).sum()),violations=int((diff>.01+.01*ref.abs()).sum()),finite=bool(torch.isfinite(got).all()));torch.save(got,r/'output.pt');torch.save(ref,r/'golden.pt');assert result['finite'] and result['violations']==0,'CPU accuracy failure'
 baseline=S/f"production-output-seed{cfg.get('seed',20)}.pt"
 input_record=S/('production-inputs.json' if cfg.get('seed',20)==20 else f"production-inputs-seed{cfg.get('seed')}.json")
 if cfg.get('reference'):
  assert not baseline.exists(),'do not overwrite production baseline';torch.save(got,baseline);input_record.write_text(json.dumps(result['inputs'],indent=2))
 elif baseline.exists():
  previous=json.loads(input_record.read_text());assert previous['fingerprints']==result['inputs']['fingerprints'],'baseline input mismatch';base=torch.load(baseline,map_location='cpu');d=(got-base).abs();result['production_comparison']={'max_abs':float(d.max()),'mismatch':int((d!=0).sum())};assert not result['production_comparison']['mismatch'],'candidate not bit-exact to production'
 result['accuracy']='PASS';save('timing')
 if cfg.get('accuracy_only'):
  result['status']='PASS';save('accuracy_only_complete');sys.exit(0)
 for _ in range(cfg.get('repeats',3)):compiled[(1,1,1)](*[args[i] for i in [0,1,2,3,4,5,6,7,8,11,12,13,15,18]]);torch.cuda.synchronize()
 result.update(mean_stats(r,list(range(1,1+cfg.get('repeats',3)))));result['status']='PASS';save('complete')
except SystemExit:raise
except Exception as ex:
 result.update(status='FAIL' if result.get('phase') in ['accuracy_warmup','timing'] else 'COMPILE_OR_RUNTIME_FAILED',error_type=type(ex).__name__,error=traceback.format_exc());save('error');print(result['error'],flush=True)
finally:
 result['final_safety']=safety('end '+cfg['name']);save(result.get('phase','finished'))
print(json.dumps({key:result.get(key) for key in ['status','cycles','cycle_samples','max_abs','mismatch','violations','production_comparison','spill','lds_bytes','static_instructions']},default=str),flush=True)
