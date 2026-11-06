"""Expand the Excel workload with upstream parser, inspect every quant_matmul case,
and execute each supported K/N on CPU without bypassing precision validation.
"""
import argparse,collections,csv,datetime,hashlib,json,platform,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--workload',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(a.repo/'src'),str(a.repo/'projects/micro_perf')]
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from xpu_perf.micro_perf.core.common_utils import parse_json_file
from xpu_perf.micro_perf.core.utils import get_torch_dtype,get_torch_dtype_size
from op_defs.llm_ops.quant_matmul import QuantMatmulOp
class CPUDevice:
    def get_torch_device_name(self):return 'cpu'
class Trace(TorchDispatchMode):
    def __init__(self):super().__init__();self.records=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}));self.records.append(dict(op=str(func),inputs=describe(args),output=describe(out)));return out

def describe(t):
    if isinstance(t,torch.Tensor):return dict(shape=list(t.shape),dtype=str(t.dtype),stride=list(t.stride()),bytes=t.numel()*t.element_size())
    if isinstance(t,(list,tuple)):return [describe(x) for x in t]
    if isinstance(t,dict):return {k:describe(v) for k,v in t.items()}
    return str(t)

torch.set_num_threads(1);torch.manual_seed(20261105)
cases=parse_json_file(a.workload)['quant_matmul'];assert len(cases)==1104
counts=collections.defaultdict(collections.Counter);rows=[];runs=[]
for i,args in enumerate(cases):
    rec=dict(case=i+1,dtype=args['dtype'],w_dtype=args['w_dtype'],compute_dtype=args['compute_dtype'],dst_dtype=args['dst_dtype'],M=args['num_tokens']//args['sp_size'],N=args['new_hidden_size'],K=args['hidden_size'],sp_size=args['sp_size'],prepare_status='',A_scale_dtype='',A_scale_shape='',B_scale_dtype='',B_scale_shape='',error='')
    try:op=QuantMatmulOp(args,CPUDevice())
    except Exception as e:
        rec.update(prepare_status='REJECTED',error=type(e).__name__+': '+str(e));counts[args['dtype']]['rejected']+=1
    else:
        rec.update(prepare_status='ACCEPTED',A_scale_dtype=str(op.input_tensor_info['per_token_scale'].dtype),A_scale_shape=json.dumps(op.input_tensor_info['per_token_scale'].shape),B_scale_dtype=str(op.input_tensor_info['expert_scale'].dtype),B_scale_shape=json.dumps(op.input_tensor_info['expert_scale'].shape))
        counts[args['dtype']]['accepted']+=1
        if args['num_tokens']==1:
            entry=dict(args=args)
            try:
                ts={k:v.creator(size=v.shape,dtype=v.dtype,device=v.device) for k,v in op.input_tensor_info.items()}
                with Trace() as trace:out=op.core_run(ts)
                entry.update(status='EXECUTED',inputs=describe(ts),scales={k:dict(**describe(v),values=torch.unique(v).tolist()) for k,v in ts.items() if 'scale' in k},output=describe(out),trace=trace.records,calc_flops=op.calc_flops,read_bytes=op.read_bytes,write_bytes=op.write_bytes,input_pattern='upstream zeros inputs, ones scales')
            except Exception as e:entry.update(status='EXECUTION_ERROR',error=type(e).__name__+': '+str(e))
            runs.append(entry)
    rows.append(rec)
with (a.output_dir/'six-precision-cases.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
report=dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository='https://github.com/bytedance/xpu-perf',commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),workload_sha256=hashlib.sha256(a.workload.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),environment=dict(hostname=platform.node(),platform=platform.platform(),torch=torch.__version__,python=sys.version,device='CPU',cuda_available=torch.cuda.is_available()),expanded_case_count=len(cases),per_precision={k:dict(v) for k,v in counts.items()},executions=runs,GPU_benchmarks_run=False,full_launch_run=False)
(a.output_dir/'six-precision-runtime.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(cases=len(rows),per_precision=report['per_precision'],executed_real_MNK=[[1,r['args']['new_hidden_size'],r['args']['hidden_size']] for r in runs if r['status']=='EXECUTED']),ensure_ascii=False))
