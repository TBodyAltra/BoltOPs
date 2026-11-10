from pathlib import Path
import subprocess,json,sys,shlex,time,runpy
r=Path(__file__).resolve().parent;name=sys.argv[1];cfg=json.loads((r/'plan.json').read_text())[name];d=r/name;d.mkdir(exist_ok=True);assert not (d/'result.json').exists(),'case already exists; use new name'
(r/'active.json').write_text(json.dumps({'case':name,'phase':'launching','epoch':time.time()}));(d/'request.json').write_text(json.dumps(cfg,indent=2))
runpy.run_path(str(r/'guard.py'))['safety']('before independent point '+name)
launch=f'''source {shlex.quote(str(r/'env.sh'))}
export GPU_CHIP=sb GPU_ARGS='[]' GPU_DFLAGS="['StatLog','TT0','Perf']"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR={shlex.quote(str(r/'cache'))}
ulimit -c 0
export HSA_TOOLS_LIB=/opt/rocm/lib/xprofiler/libperf_gen_helper.so
cd {shlex.quote(str(d))}
python {shlex.quote(str(r/'capture-env.py'))}
exec python {shlex.quote(str(r/cfg.get('worker','worker.py')))}
''';(d/'launch.sh').write_text(launch)
with (d/'run.log').open('wb') as h:ret=subprocess.call(['bash',str(d/'launch.sh')],stdout=h,stderr=subprocess.STDOUT)
res=json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else {'status':'PROCESS_FAILED','returncode':ret};res['returncode']=ret;(d/'result.json').write_text(json.dumps(res,indent=2,default=str));(r/'active.json').write_text(json.dumps({'case':None,'last_case':name,'status':res['status'],'epoch':time.time()}));print(json.dumps({k:res.get(k) for k in ['status','cycles','max_abs','mismatch','production_comparison','spill','lds_bytes','error']},default=str),flush=True)
