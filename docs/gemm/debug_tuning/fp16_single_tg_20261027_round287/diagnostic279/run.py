from pathlib import Path
import subprocess,fcntl,shutil,json,os,time
R=Path(__file__).resolve().parent
B=R.parent
CPU=next(B.glob('*round274-*'))
NATIVE=next(B.glob('*round277-*'))
rows=[]
with (B/'single-tg-continuation-20261023/compiler.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert shutil.disk_usage(R).free>24*2**30
 for label,root in [('cpu1',CPU),('native1',NATIVE),('cpu2',CPU),('native2',NATIVE)]:
  out=R/label;out.mkdir(exist_ok=False);(out/'job.json').write_bytes((R/'pilot-job.json').read_bytes())
  with (R/(label+'.log')).open('w') as f:
   cmd='source '+str(root/'env.sh')+'; export MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR='+str(out/'cache')+' DIAGNOSTIC_SOURCE_ROOT='+str(root)+'; python3 '+str(R/'compile_dump.py')+' '+str(out/'job.json')
   rc=subprocess.run(['bash','-lc',cmd],stdout=f,stderr=subprocess.STDOUT).returncode
  result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
  rows.append(dict(label=label,returncode=rc,status=result.get('status')))
  (R/'progress.json').write_text(json.dumps(rows,indent=2))
  if rc or result.get('status')!='COMPILED': break
(R/'complete.json').write_text(json.dumps(dict(CPU_only=True,no_PMD_execution=True,rows=rows),indent=2))
