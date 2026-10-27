from pathlib import Path
import subprocess,fcntl,shutil,json,time
R=Path(__file__).resolve().parent
with (R.parent/'single-tg-continuation-20261023/compiler.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert shutil.disk_usage(R).free>24*2**30
 rows=[]
 for label in ['cpu1','native1','cpu2','native2']:
  with (R/(label+'.log')).open('w') as f:
   rc=subprocess.run(['bash','-lc','source '+str(R.parent/'single-tg-pipeline-20261027-round203-fragment-layout3-cpu/env.sh')+'; export MLIR_ENABLE_DUMP=1; python3 '+str(R/'replay.py')+' '+label],stdout=f,stderr=subprocess.STDOUT).returncode
  rows.append(dict(label=label,returncode=rc));print(rows[-1],flush=True)
  if rc:break
 (R/'replay-run.json').write_text(json.dumps(dict(CPU_only=True,no_PMD_execution=True,rows=rows),indent=2)+'\n')
