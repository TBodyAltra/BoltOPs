from pathlib import Path
import os,json,subprocess,hashlib,time
r=Path(__file__).resolve().parent;B=r.parent
selected={k:v for k,v in os.environ.items() if k.startswith(('TRITON_','GPU_','AMDGCN_','PMD_','HSA_')) or k in ['PATH','PYTHONPATH','LD_LIBRARY_PATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE']}
d={'captured_before_worker_epoch':time.time(),'environment':selected,'die_count':1,'cu_count':48,'clock_period_ps':455,'gc_frequency_ghz':1000/455,'triton_commit':subprocess.check_output(['git','-C',str(B/'wdra48-fix-20261020/triton'),'rev-parse','HEAD'],text=True).strip(),'boltops_commit':subprocess.check_output(['git','-C',str(B/'gemm-optimization-20261020/bolt_ops'),'rev-parse','HEAD'],text=True).strip(),'private_parser_library_sha256':hashlib.sha256((B/'fp8-G171-wdra88-ab-20261021/libtriton.import.so').read_bytes()).hexdigest(),'canonical_library_sha256':hashlib.sha256((B/'mmac-descriptor-20261020/libtriton.descriptor.so').read_bytes()).hexdigest()}
(Path.cwd()/'environment.json').write_text(json.dumps(d,indent=2)+'\n')
