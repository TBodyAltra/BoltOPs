import pathlib,json,sys,time,re,hashlib,resource,importlib,os
R=pathlib.Path(os.environ["DIAGNOSTIC_SOURCE_ROOT"]);job=json.loads(pathlib.Path(sys.argv[1]).read_text());out=pathlib.Path(sys.argv[1]).parent;start=time.time()
result=dict(job,status='STARTED',no_pmd_execution=True,no_kernel_loading=True)
try:
 import triton
 import triton._C.libtriton as native
 assert pathlib.Path(native.__file__).resolve()==(R/'native/libtriton.so').resolve()
 result['loaded_native']=dict(path=str(pathlib.Path(native.__file__).resolve()),sha256=hashlib.sha256(pathlib.Path(native.__file__).read_bytes()).hexdigest())
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 mod=importlib.import_module('boltops.gemm.triton.gemm_fp16');fn=mod._gemm_fp16_mls_kernel
 cfg=job['config']
 import inspect
 defaults={name:param.default for name,param in inspect.signature(fn.fn).parameters.items() if param.default is not inspect.Parameter.empty and fn.arg_names.index(name) in fn.constexprs}
 constants=dict(defaults);constants.update({k:v for k,v in cfg.items() if k in fn.arg_names})
 bm,bn,bk=job['tile'];k=bk*16
 # Specialize unit strides as the runtime JIT does; all other dimensions remain runtime.
 constants.update(stride_ak=1,stride_cn=1)
 constants['stride_bn' if job['layout']=='TT' else 'stride_bk']=1
 signature={n:('*fp16' if n.endswith('_ptr') else 'i32') for n in fn.arg_names if n not in constants}
 opts={n:v for n,v in cfg.items() if n not in fn.arg_names}
 attrs={}
 for name in signature:
  if name.endswith('_ptr') or name in ['M','N','K','stride_am','stride_cm'] or (name=='stride_bk' and job['layout']=='TT') or (name=='stride_bn' and job['layout']=='TN'):
   attrs[(fn.arg_names.index(name),)]=[['tt.divisibility',16]]
   if name.endswith('_ptr'):attrs[(fn.arg_names.index(name),)].append(['tt.pointer_range',32])
 cap=resource.getrlimit(resource.RLIMIT_AS);vms=int(next(x.split()[1] for x in pathlib.Path('/proc/self/status').read_text().splitlines() if x.startswith('VmSize:')))*1024
 resource.setrlimit(resource.RLIMIT_AS,(vms+8*1024**3,cap[1]))
 kernel=triton.compile(ASTSource(fn,signature,constexprs=constants,attrs=attrs),target=GPUTarget('hip','gfx946',64),options=opts)
 resource.setrlimit(resource.RLIMIT_AS,cap)
 for kind in ['ttir','ttgir','llir','amdgcn']:(out/('kernel.'+kind)).write_text(kernel.asm[kind])
 isa=kernel.asm['amdgcn'];fields={}
 for n in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count','amdhsa_next_free_vgpr']:
  m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',isa);fields[n]=int(m.group(1)) if m else None
 result.update(status='COMPILED',metadata=kernel.metadata._asdict(),fields=fields,asm_sha256=hashlib.sha256(isa.encode()).hexdigest())
except Exception as e:
 import traceback;traceback.print_exc();result.update(status='ERROR',exception_type=type(e).__name__,exception=str(e))
result['seconds']=time.time()-start;(out/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps(result,default=str))
