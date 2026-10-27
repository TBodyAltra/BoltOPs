"""Execute all real worker admission/config code, stopping before RNG/tensor allocation."""
from pathlib import Path
import ast,json,hashlib,inspect,sys,dataclasses
import scan
from triton.backends.hcu.compiler import HIPBackend
from triton.backends.compiler import GPUTarget
from boltops.gemm.triton.gemm_fp16 import _gemm_fp16_mls_kernel as fn, _gemm_fp16_load_kernel as vector_fn
mls_fn=fn
R=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text());p=R/'worker-prefix-review.json';assert not p.exists()
worker=R/'worker.py';source=worker.read_text();cut='    torch.manual_seed(29)';assert source.count(cut)==1;prefix=source[:source.index(cut)];original_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest();native_hash=sha(R/'native/libtriton.so')
# Hash the immutable888MB native once before/after, rather than152 times.
anchor='native_sha=hashlib.sha256(native_path.read_bytes()).hexdigest()';assert prefix.count(anchor)==1;prefix=prefix.replace(anchor,'native_sha=_review_native_hash');prefix+='    _review_built_config = cfg\nexcept BaseException:\n    raise\n';compiled=compile(prefix,str(worker),'exec')
func=next(x for x in ast.parse((R/'tune.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='template');g={'scan':scan};exec(compile(ast.Module(body=[func],type_ignores=[]),str(R/'tune.py'),'exec'),g)
refs=load(R/'cpu-controls.json')['controls']+load(R/'cpu-admission.json')['results'];assert len(refs)==len(load(R/'cpu-admission.json')['results'])+2;defaults={n:v.default for n,v in inspect.signature(fn.fn).parameters.items() if v.default is not inspect.Parameter.empty and fn.arg_names.index(n) in fn.constexprs};backend=HIPBackend(GPUTarget('hip','gfx946',64));rows=[];argv=list(sys.argv)
for index,ref in enumerate(refs):
 fn=mls_fn if ref.get('mls',True) else vector_fn
 defaults={n:v.default for n,v in inspect.signature(fn.fn).parameters.items() if v.default is not inspect.Parameter.empty and fn.arg_names.index(n) in fn.constexprs}
 for loops,edge in ([(16,False),(32,False),(64,False),(17,True)] if index>=2 else [(16,False),(32,False),(64,False)]):
  bm,bn,bk=ref['tile'];s=dict(g['template'](ref),scope='full_gemm' if edge else 'single_cta',correctness_only=edge,shape=[bm+(16 if edge else 0),bn+(16 if edge else 0),bk*loops],pipeline_loops=loops);case=R/'worker-prefix-cases'/f'{index}-{loops}';case.mkdir(parents=True,exist_ok=False);sp=case/'spec.json';sp.write_text(json.dumps(s,indent=2)+'\n');sys.argv=[str(worker),str(sp)];namespace={'__file__':str(worker),'__name__':'worker_prefix_review','_review_native_hash':native_hash};exec(compiled,namespace);c=namespace['_review_built_config'];expected=ref['config'];a=dict(defaults);a.update({k:v for k,v in c.items() if k in fn.arg_names});b=dict(defaults);b.update({k:v for k,v in expected.items() if k in fn.arg_names});assert a==b,(index,loops,a,b);actual=backend.parse_options({k:v for k,v in c.items() if k not in fn.arg_names});reference=backend.parse_options({k:v for k,v in expected.items() if k not in fn.arg_names});assert actual==reference,(index,loops,actual,reference)
  rows.append(dict(CPU_reference=ref['directory'],loops=loops,correctness_only=edge,all_actual_worker_preallocation_guards_passed=True,constexpr_and_backend_options_equal_to_CPU=True,grid=namespace['grid'],requested_quotas=s['regs']))
sys.argv=argv;assert len(rows)==4*len(refs)-2 and sha(worker)==hashlib.sha256(source.encode()).hexdigest() and sha(R/'native/libtriton.so')==native_hash
out=dict(passed=True,CPU_only=True,no_PMD_execution=True,no_kernel_compilation=True,no_kernel_loading=True,no_gpu_allocation=True,actual_worker_prefix_executed=True,prefix_stops_before='torch.manual_seed(29),all randn/cuda/tensor operations',immutable_native_hash_cached_for_review_only=True,source_worker_sha256=sha(worker),original_prefix_sha256=original_prefix_sha256,tune_sha256=sha(R/'tune.py'),cases=len(rows),controls=2,recipes=len(refs)-2,loops=[16,32,64],edge_loops=17,rows=rows,script_sha256=sha(Path(__file__)))
p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(rows),no_GPU_allocation=True)))
