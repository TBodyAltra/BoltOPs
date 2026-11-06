"""Run unmodified xpu-perf operators on CPU and capture their tensor contracts.

This is a reference-path probe, not a GPU benchmark or vendor implementation.
"""
import argparse,datetime,hashlib,json,platform,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.repo/'src'),str(a.repo/'projects/micro_perf')]
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from xpu_perf.micro_perf.core import utils
from op_defs.basic_ops.tensor_gemm_ops import GemmOp
from op_defs.llm_ops.quant_matmul import QuantMatmulOp
source=json.loads(a.source.read_text());torch.set_num_threads(1);torch.manual_seed(20261105)
class CPUBackend:
    def get_torch_device_name(self):return 'cpu'
backend=CPUBackend()
def ts(x):
    if isinstance(x,torch.Tensor):return dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),device=str(x.device),bytes=x.numel()*x.element_size())
    if isinstance(x,(list,tuple)):return [ts(t) for t in x]
    if isinstance(x,dict):return {k:ts(v) for k,v in x.items()}
    return str(x)
class Trace(TorchDispatchMode):
    def __init__(self):super().__init__();self.ops=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        self.ops.append(dict(op=str(func),inputs=ts(args),output=ts(result)))
        return result
def args(kind,d,K,N,M=4):
    if kind=='gemm':return dict(arg_type='default',M=M,K=K,N=N,dtype=d)
    return dict(arg_type='llm',sp_size=1,num_tokens=M,hidden_size=K,new_hidden_size=N,dtype=d[0],w_dtype=d[1],compute_dtype=d[2],dst_dtype=d[3])
records=[];executions=[];mappings=[]
for name in ['bfloat16','int8','float8','float4','mxfloat8','mxfloat4','float8_e8m0']:
    r=dict(name=name)
    try:
        dtype=utils.get_torch_dtype(name);r.update(torch_dtype=str(dtype),logical_element_bytes=utils.get_torch_dtype_size(dtype))
        t=utils.default_creator((2,8),dtype,'cpu');r['creator_tensor']=ts(t)
    except Exception as e:r['error']=type(e).__name__+': '+str(e)
    mappings.append(r)
for kind,dtypes,geometry,cls in [('gemm',source['basic']['dtypes'],source['basic']['KN'],GemmOp),('quant_matmul',source['quant']['dtypes'],[(x[1],x[2]) for x in source['quant']['sp_K_N']],QuantMatmulOp)]:
    for d in dtypes:
        for K,N in geometry:
            arg=args(kind,d,K,N);r=dict(operator=kind,args=arg)
            try:
                op=cls(arg,backend)
                r.update(accepted=True,inputs={k:dict(shape=v.shape,dtype=str(v.dtype),creator=getattr(v.creator,'__name__',str(v.creator))) for k,v in op.input_tensor_info.items()},outputs={k:dict(shape=v.shape,dtype=str(v.dtype)) for k,v in op.output_tensor_info.items()},operator_attributes={k:v for k,v in op.__dict__.items() if ('block' in k or 'quant' in k or 'scale' in k) and isinstance(v,(str,int,float,bool,type(None)))})
                # Execute one real Excel geometry for each accepted operator, using its own tensor creators.
                if K==4096 and N==4096:
                    tensors={k:v.creator(size=v.shape,dtype=v.dtype,device=v.device) for k,v in op.input_tensor_info.items()}
                    scales={k:dict(**ts(t),unique_values=torch.unique(t).tolist()) for k,t in tensors.items() if 'scale' in k}
                    with Trace() as trace:out=op.core_run(tensors)
                    executions.append(dict(operator=kind,args=arg,inputs=ts(tensors),scales=scales,output=ts(out),output_finite=bool(torch.isfinite(out).all()),trace=trace.ops,input_pattern='upstream creators unchanged'))
            except Exception as e:r.update(accepted=False,error=type(e).__name__+': '+str(e))
            records.append(r)
paths=['projects/micro_perf/op_defs/basic_ops/tensor_gemm_ops.py','projects/micro_perf/op_defs/llm_ops/quant_matmul.py','projects/micro_perf/op_defs/llm_ops/quant_matmul.md','src/xpu_perf/micro_perf/core/utils.py','src/xpu_perf/micro_perf/core/op.py']
report=dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),capture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_snapshot_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),environment=dict(hostname=platform.node(),platform=platform.platform(),python=sys.version,torch=torch.__version__,device='CPU',cuda_available=torch.cuda.is_available()),repository_commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),source_file_sha256={f:hashlib.sha256((a.repo/f).read_bytes()).hexdigest() for f in paths},dtype_mapping=mappings,prepare_records=records,executions=executions,GPU_tests_run=False,scope='40 dtype/KN groups prepared; one original Excel MNK executed per accepted operator; unsupported paths not bypassed')
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(torch=torch.__version__,groups=len(records),accepted=sum(r['accepted'] for r in records),rejected=sum(not r['accepted'] for r in records),executed=[r['operator'] for r in executions],output=str(a.output)),ensure_ascii=False))
