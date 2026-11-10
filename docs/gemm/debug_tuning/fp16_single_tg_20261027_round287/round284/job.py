import pathlib,json,subprocess,sys,time,os,fcntl,traceback,shutil
R=pathlib.Path(__file__).resolve().parent
def save(n,q):q.update(remote_epoch=time.time(),pid=os.getpid());(R/n).write_text(json.dumps(q,indent=2)+'\n')
with (R/'job.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert shutil.disk_usage(R).free>24*2**30
  save('job-state.json',dict(status='PREPARING_FROZEN_SMALL_CTA_PIPELINES'))
  with (R/'logs/prepare.py.log').open('w') as f:subprocess.run([sys.executable,str(R/'prepare.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
  with (R/'logs/worker-preflight.log').open('w') as f:subprocess.run(['bash','-lc',f'source {R}/env.sh; python3 {R}/review_worker_prefix.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
  save('job-state.json',dict(status='TWO_CONTROLS_SMALL_TT_TN_PILOTS_EDGES_THEN_16_64_32'))
  with (R/'logs/tune.py.log').open('w') as f:subprocess.run(['bash','-lc',f'source {R}/env.sh; cd {R}; python3 -u tune.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
  assert json.loads((R/'state.json').read_text())['phase']=='FINISHED'
  save('job-state.json',dict(status='CALIBRATION_COMPLETE_PENDING_RAW_AUDIT_AND_INDEPENDENT_CONFIRMATION'));save('job-exit.json',dict(returncode=0))
 except BaseException:save('job-exit.json',dict(returncode=1,error=traceback.format_exc()));raise
