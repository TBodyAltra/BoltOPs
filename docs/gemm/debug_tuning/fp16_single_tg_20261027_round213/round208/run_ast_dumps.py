from pathlib import Path
import subprocess,fcntl,shutil,json,os
R=Path(__file__).resolve().parent
cpu=R.parent/'single-tg-pipeline-20261027-round203-fragment-layout3-cpu'
native=R.parent/'single-tg-pipeline-20261027-round207-fragment-layout3-native-retry'
with (R.parent/'single-tg-continuation-20261023/compiler.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert shutil.disk_usage(R).free>24*2**30
 rows=[]
 for label,root in [('ast_cpu1',cpu),('ast_native1',native),('ast_cpu2',cpu),('ast_native2',native)]:
  out=R/label;out.mkdir(exist_ok=False);(out/'job.json').write_bytes((cpu/'probes/features/4c7c95c875d07b7f/job.json').read_bytes())
  with (R/(label+'.log')).open('w') as f:
   cmd='source '+str(root/'env.sh')+'; export MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR='+str(out/'cache')+' DIAGNOSTIC_SOURCE_ROOT='+str(root)+'; python3 '+str(R/'compile_dump.py')+' '+str(out/'job.json')
   rc=subprocess.run(['bash','-lc',cmd],stdout=f,stderr=subprocess.STDOUT).returncode
  result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
  rows.append(dict(label=label,returncode=rc,status=result.get('status')));print(rows[-1],flush=True)
  if rc or result.get('status')!='COMPILED':break
 (R/'ast-dump-run.json').write_text(json.dumps(dict(CPU_only=True,no_PMD_execution=True,rows=rows),indent=2)+'\n')
