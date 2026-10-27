import pathlib,json,sys,hashlib,importlib
from triton._C.libtriton import ir
from triton.backends.hcu.compiler import HIPBackend
from triton.backends.compiler import GPUTarget
R=pathlib.Path(__file__).resolve().parent
cpu=R.parent/'single-tg-pipeline-20261027-round203-fragment-layout3-cpu/probes/features/4c7c95c875d07b7f'
native=R.parent/'single-tg-pipeline-20261027-round207-fragment-layout3-native-retry/runs/8ba2a10add81b095'
label=sys.argv[1];inp=cpu if label.startswith('cpu') else native
out=R/label;out.mkdir(exist_ok=False)
job=json.loads((cpu/'job.json').read_text());metadata=json.loads((cpu/'result.json').read_text())['metadata']
fn=importlib.import_module('boltops.gemm.triton.gemm_fp16')._gemm_fp16_mls_kernel
backend=HIPBackend(GPUTarget('hip','gfx946',64));opts=backend.parse_options({k:v for k,v in job['config'].items() if k not in fn.arg_names})
ctx=ir.context();ir.load_dialects(ctx);backend.load_dialects(ctx)
mod=ir.parse_mlir_module(str(inp/'kernel.ttgir'),ctx);mod.context=ctx
llir=backend.make_llir(mod,metadata,opts)
(out/'kernel.llir').write_text(llir);(out/'lowered.mlir').write_text(str(mod))
(out/'result.json').write_text(json.dumps(dict(passed=True,input=str(inp/'kernel.ttgir'),input_sha256=hashlib.sha256((inp/'kernel.ttgir').read_bytes()).hexdigest(),metadata=metadata),indent=2)+'\n')
print(label,'PASS')
