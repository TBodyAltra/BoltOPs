import os,sys,json,time,traceback,re
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import torch,triton
import triton.language as tl
import boltops
root=Path.cwd();cfg=json.loads(sys.argv[1]);measure='--measure' in sys.argv
sys.path.insert(0,str(root.parent));sys.path.insert(0,'/xukang/code/pmd-triton-sessions')
sys.path.insert(0,'/xukang/code/pmd-triton-sessions/fp8-pipeline-probes-20261020')
from trial_aiter import _gemm_a8w8_blockscale_kernel as ka
from trial_vllm import _w8a8_triton_block_scaled_mm as kv
sys.path.insert(0,'/xukang/code/pmd-triton-sessions/fp8-pipeline-probes-20261020')
from perf_metrics_20261019 import mean_stats
row={'request':cfg,'phase':'measure' if measure else 'compile','source':boltops.__file__};start=time.monotonic()
try:
 torch.set_num_threads(1);torch.manual_seed(20)
 m,n,k=cfg.get('M',cfg['BM']),cfg.get('N',cfg['BN']),cfg.get('K',1024);full=cfg.get('full',False);g=triton.cdiv(k,128)
 a=(torch.randn(m,k)*.5).to(torch.float8_e4m3fn);w=(torch.randn(n,k)*.5).to(torch.float8_e4m3fn)
 sa=torch.rand(m,1 if full else g)*.2+.05
 if cfg.get('a_encoded'):sa=torch.randint(125,130,sa.shape,dtype=torch.uint8)
 sa_float=torch.pow(2.,sa.float()-127) if cfg.get('a_encoded') else sa
 sb=torch.randint(125,130,(n,g),dtype=torch.uint8)
 if cfg.get('b_encoded',True)==False:sb=torch.pow(2.,sb.float()-127)*1.125
 sb_float=torch.pow(2.,sb.float()-127) if cfg.get('b_encoded',True) else sb
 A=a.cuda();W=w.cuda();SA=sa.cuda();SB=sb.cuda();C=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
 mode=cfg['mode'];special=mode in ('wdra88','wdra48','wdra48_async','wdra44','wasp44')
 opts=dict(BLOCK_SIZE_M=cfg['BM'],BLOCK_SIZE_N=cfg['BN'],BLOCK_SIZE_K=cfg.get('BK',128),GROUP_SIZE_M=1,num_stages=cfg.get('stages',2),num_warps=8 if mode in ('base8','bpp','bpp_async','async8') else 4,A_SCALE_PER_ROW=full,WARP_SPECIALIZE=special)
 if special:opts.update(wasp_partition_warps=tuple(cfg['partition_warps']),wasp_wdra=True)
 if 'regs' in cfg:opts['wasp_partition_regs']=tuple(cfg['regs'])
 if 'compiler_options' in cfg:opts.update(cfg['compiler_options'])
 if mode in ('bpp','bpp_async'):opts['use_block_pingpong']=True
 if mode in ('async','async8','bpp_async','wdra48_async'):opts['use_async_copy']=True
 if cfg['api']=='aiter':
  kernel=ka
  if cfg.get('use_mls'):
   import importlib
   kernel=importlib.import_module(cfg['module'])._gemm_a8w8_blockscale_kernel
  if cfg.get('control'):
   from control_aiter import _gemm_a8w8_blockscale_kernel as kernel
  opts.update(GROUP_K=128,GROUP_N=1,NUM_KSPLIT=1,SPLITK_BLOCK_SIZE=k,cache_modifier='')
  args=(A,W.T,C,SA,SB.T,m,n,k,A.stride(0),A.stride(1),W.stride(1),W.stride(0),0,C.stride(0),C.stride(1),SA.stride(0),SA.stride(1),SB.stride(1),SB.stride(0))
 else:
  kernel=kv
  if cfg.get('constexpr_groups'):
   from trial_vllm_constexpr import _w8a8_triton_block_scaled_mm as kernel
  if cfg.get('control'):
   from control_vllm import _w8a8_triton_block_scaled_mm as kernel
  args=(A,W,C,SA,SB,m,n,k,1,128,A.stride(0),A.stride(1),W.stride(1),W.stride(0),C.stride(0),C.stride(1),SA.stride(0),SA.stride(1),SB.stride(1),SB.stride(0))
 import inspect,hashlib
 kernel_object=kernel
 while hasattr(kernel_object,'fn'):kernel_object=kernel_object.fn
 row['actual_kernel_source']=str(root.parent/cfg['ir_source'])
 row['actual_kernel_source_sha256']=hashlib.sha256(Path(row['actual_kernel_source']).read_bytes()).hexdigest()
 grid=(triton.cdiv(m,cfg['BM'])*triton.cdiv(n,cfg['BN']),)
 assert grid==(1,) and m==cfg['BM'] and n==cfg['BN'] and k%cfg['BK']==0,'aligned natural single TG only'
 t=time.monotonic();compiled=triton.compile(str(root.parent/cfg['ir_source']),options={k:v for k,v in opts.items() if k in ['num_warps','num_stages','wasp_partition_warps','wasp_partition_regs','wasp_wdra','optimize_epilogue','mmac_layout_force']});row['compile_seconds']=time.monotonic()-t
 if cfg.get('batch_pairs'):
  import runpy
  from triton.backends.compiler import GPUTarget
  from triton.backends.hcu.compiler import HIPBackend
  md=compiled.metadata._asdict();backend=HIPBackend(compiled.metadata.target);options=backend.parse_options({**md,'num_warps':opts['num_warps']})
  changed,records=runpy.run_path(str(root.parent/'batch-transform.py'))['transform'](compiled.asm['llir'],cfg['batch_pairs'])
  from triton.backends.hcu import wasp
  before=tuple(int(n) for n in re.findall(r'i16 (\d+)',re.search(r'@llvm.hcu.wdra.init\(([^)]*)\)',changed)[1]))[:3]
  changed=wasp.update_llir_regs(changed,before,(8,248,248));md['wasp_partition_regs']=(8,248,248)
  (root/'scheduled.llir').write_text(changed)
  (root/'scheduling.json').write_text(json.dumps(records,indent=2))
  gcn=backend.make_amdgcn(changed,md,options);binary=backend.make_hsaco(gcn,md,options)
  compiled.asm['llir']=changed;compiled.asm['amdgcn']=gcn;compiled.asm['hsaco']=binary;compiled.kernel=binary
  compiled.metadata=compiled.metadata._replace(**{k:md[k] for k in compiled.metadata._fields if k in md});compiled.packed_metadata=backend.pack_metadata(compiled.metadata)
  row['native_schedule_override']='external LLIR order + native scheduler fences only; normal native backend and quota checks'
  row['executed_hsaco_sha256']=hashlib.sha256(binary).hexdigest();(root/'executed.hsaco').write_bytes(binary)
 if cfg.get('defer_scale_prefetch_wait'):
  import runpy
  from triton.backends.hcu.compiler import HIPBackend
  backend=HIPBackend(compiled.metadata.target);md=compiled.metadata._asdict();options=backend.parse_options({**md,'num_warps':opts['num_warps']})
  changed,proof=runpy.run_path(str(root.parent/'defer-prefetch-wait.py'))['transform'](compiled.asm['amdgcn'])
  (root/'native-before-defer.amdgcn').write_text(compiled.asm['amdgcn']);(root/'deferred-wait-proof.json').write_text(json.dumps(proof,indent=2))
  binary=backend.make_hsaco(changed,md,options);compiled.asm['amdgcn']=changed;compiled.asm['hsaco']=binary;compiled.kernel=binary;compiled.packed_metadata=backend.pack_metadata(compiled.metadata)
  row['native_schedule_override']='next scale-only prefetch: relocate full LGKM wait from before MMAC turn signal to first pending scale register use after current scale math; all hardware waits/barriers retained'
 if cfg.get('native_idea'):
  import runpy
  from triton.backends.hcu.compiler import HIPBackend
  backend=HIPBackend(compiled.metadata.target);md=compiled.metadata._asdict();options=backend.parse_options({**md,'num_warps':opts['num_warps']})
  changed,proof=runpy.run_path(str(root.parent/'native-transform.py'))['transform'](compiled.asm['amdgcn'],cfg['native_idea'])
  (root/'native-original.amdgcn').write_text(compiled.asm['amdgcn']);(root/'native-proof.json').write_text(json.dumps(proof,indent=2))
  binary=backend.make_hsaco(changed,md,options);compiled.asm['amdgcn']=changed;compiled.asm['hsaco']=binary;compiled.kernel=binary;compiled.packed_metadata=backend.pack_metadata(compiled.metadata)
  row['native_schedule_override']=cfg['native_idea']
 row['executed_hsaco_sha256']=hashlib.sha256(compiled.asm['hsaco']).hexdigest();(root/'executed.hsaco').write_bytes(compiled.asm['hsaco'])
 row['metadata']=compiled.metadata._asdict();row['lds_bytes']=compiled.metadata.shared;row['compile']='PASS'
 ir=root/'ir';ir.mkdir(exist_ok=True)
 for key in ['ttir','ttgir','llir','amdgcn']:
  if key in compiled.asm:(ir/f'kernel.{key}').write_text(compiled.asm[key])
 row['effective_launch_options']=opts
 row['launch_grid']=grid
 (root/'compile-metadata.json').write_text(json.dumps(row,indent=2,default=str))
 compiled._init_handles();row.update(vgpr=compiled.n_regs,spill=compiled.n_spills)
 asm=compiled.asm.get('amdgcn','');ttgir=compiled.asm.get('ttgir','');llir=compiled.asm.get('llir','')
 row['features']={key:len(re.findall(pattern,asm)) for key,pattern in {'mmac':r'\bv_mmac','buffer_load':r'\bbuffer_load','async_lds':r'\b(?:buffer_load[^\n]*lds|buffer_load_lds[^\n]*)','lds_read':r'\bds_read','lds_write':r'\bds_write','abarrier':r'\bs_abarrier','wdra':r'\bs_(?:set_vgpr_size|sendmsg_rtn)','waitcnt':r'\bs_waitcnt','scratch':r'\bscratch_'}.items()}
 row['features']['warp_specialize_ir']='warp_specialize' in ttgir
 row['features']['block_pingpong_ir']='block_pingpong' in ttgir.lower()
 row['features']['wdra_intrinsic']='llvm.hcu.wdra.init' in llir
 props=triton.runtime.driver.active.utils.get_device_properties(0)
 row['runtime_device_properties']=props
 row['arch_info_lds_capacity']=__import__('boltops.utility.arch_info',fromlist=['get_lds_capacity_bytes']).get_lds_capacity_bytes()
 (root/'device-capacity.json').write_text(json.dumps({'runtime':props,'arch_info_lds_capacity':row['arch_info_lds_capacity'],'compiled_lds_bytes':row['lds_bytes']},indent=2,default=str))
 row['pmd_lds_capacity']=131072
 row['eligible']=row['lds_bytes']<=min(props['max_shared_mem'],row['pmd_lds_capacity'])
 (root/'progress.json').write_text(json.dumps({'phase':'compiled; device capacity checked','eligible':row['eligible'],'metadata':row['metadata'],'device_properties':props},indent=2,default=str))
 row['status']='COMPILED' if row['eligible'] else 'EXCLUDED_LDS_CAPACITY'
 if special and not row['metadata'].get('wasp_enabled',False):row['status']='NOT_SELECTED_WASP';row['eligible']=False
 if measure and row['eligible']:
  (root/'progress.json').write_text(json.dumps({'phase':'CPU golden','compiled_lds_bytes':row['lds_bytes'],'device_properties':props},indent=2))
  ref=torch.zeros(m,n)
  for j in range(g):ref+=(a.float()[:,j*128:(j+1)*128]@w.float()[:,j*128:(j+1)*128].T)*sa_float[:,0 if full else j,None]*sb_float[:,j][None,:]
  (root/'progress.json').write_text(json.dumps({'phase':'warmup launch','grid':grid,'timestamp':time.time()},indent=2))
  compiled[(grid[0],1,1)](*[args[i] for i in [0,1,2,3,4,5,6,7,8,11,12,13,15,18]]);torch.cuda.synchronize()
  got=C.cpu().float();ref=ref.bfloat16().float();diff=(got-ref).abs();row.update(max_abs=diff.max().item(),mismatch=int((diff!=0).sum()),violations=int((diff>.01+.01*ref.abs()).sum()),finite=bool(torch.isfinite(got).all()))
  torch.save(got,root/'output.pt')
  assert row['finite'] and row['violations']==0,'numerical check failed'
  (root/'accuracy.json').write_text(json.dumps({k:row[k] for k in ['max_abs','mismatch','violations','finite']},indent=2))
  (root/'progress.json').write_text(json.dumps({'phase':'timed launch','grid':grid,'timestamp':time.time()},indent=2))
  for _ in range(cfg.get('repeats',1)):compiled[(grid[0],1,1)](*[args[i] for i in [0,1,2,3,4,5,6,7,8,11,12,13,15,18]]);torch.cuda.synchronize()
  row.update(mean_stats(root,list(range(1,1+cfg.get('repeats',1)))));row['status']='PASS'
except Exception as e:
 tb=e.__traceback__; diag={}
 while tb:
  loc=tb.tb_frame.f_locals
  if tb.tb_frame.f_code.co_name in ('check_and_adjust_wasp_wdra_regs','make_amdgcn'):
   for key in ['metadata','regs','required','spilling','budget','values','demand']:
    if key in loc:diag[key]=loc[key]
   if 'gcn' in loc:(root/'failed-kernel.amdgcn').write_text(loc['gcn'])
   if 'options' in loc:diag['effective_options']=vars(loc['options'])
  tb=tb.tb_next
 (root/'compile-diagnostics.json').write_text(json.dumps(diag,indent=2,default=str))
 row.update(status='COMPILE_FAILED' if 'compile' not in row else 'FAILED',error_type=type(e).__name__,error=traceback.format_exc());print(row['error'],flush=True)
row['wall_seconds']=time.monotonic()-start
(root/'result.json').write_text(json.dumps(row,indent=2,default=str));print(json.dumps(row,default=str),flush=True)
