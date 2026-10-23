from pathlib import Path
import argparse,json,runpy,subprocess,time,shlex
p=argparse.ArgumentParser();p.add_argument('plan');p.add_argument('--limit',type=int);args=p.parse_args()
r=Path(__file__).resolve().parent;rows=json.loads((r/args.plan).read_text());safety=runpy.run_path(str(r/'guard.py'))['safety'];rounddir=r/Path(args.plan).stem;rounddir.mkdir(exist_ok=True);out=[]
for cfg in rows[:args.limit] if args.limit else rows:
 d=rounddir/cfg['name'];d.mkdir(exist_ok=True)
 if (d/'result.json').exists():out.append(json.loads((d/'result.json').read_text()));continue
 safety('before '+cfg['name']);(d/'request.json').write_text(json.dumps(cfg,indent=2));(r/'progress.json').write_text(json.dumps(dict(active=cfg['name'],round=Path(args.plan).stem,done=len(out),total=len(rows),phase='compile/warmup/timing',epoch=time.time())))
 launch=f'''source {shlex.quote(str(r/'env.sh'))}
export GPU_CHIP=sb GPU_ARGS='[]' GPU_DFLAGS="['StatLog']"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR={shlex.quote(str(r/'cache'))}
ulimit -c 0
unset HSA_TOOLS_LIB
cd {shlex.quote(str(d))}
python {shlex.quote(str(r/'capture-env.py'))}
exec python {shlex.quote(str(r/'worker-scale-variants.py'))}
''';(d/'launch.sh').write_text(launch)
 with (d/'run.log').open('wb') as f:ret=subprocess.call(['bash',str(d/'launch.sh')],stdout=f,stderr=subprocess.STDOUT)
 if (d/'result.json').exists():x=json.loads((d/'result.json').read_text())
 else:
  x=dict(request=cfg,status='PROCESS_FAILED',returncode=ret,error_log=str(d/'run.log'));(d/'result.json').write_text(json.dumps(x,indent=2))
 x['returncode']=ret;out.append(x);(rounddir/'results.json').write_text(json.dumps(out,indent=2,default=str));safety('after '+cfg['name']);print(cfg['name'],{k:x.get(k) for k in ['status','cycles','max_abs','mismatch','violations','spill','lds_bytes','error_type']},flush=True)
(r/'final-safety.json').write_text(json.dumps(safety('round checkpoint'),indent=2));(r/'progress.json').write_text(json.dumps(dict(active=None,round=Path(args.plan).stem,phase='round complete',done=len(out),total=len(rows))))
