import json,re,statistics,subprocess,hashlib,datetime
from pathlib import Path
def stats(p):
 v={}
 for l in Path(p).read_text().splitlines():
  t=l.split()
  if len(t)>1:
   try:v[t[0]]=float(t[1])
   except ValueError:pass
 q=dict(cycles=v['system.gcSimCycles'],hostSeconds=v['system.hostSeconds'],wave_start=v['system.shaders.firstWaveStartTick'],wave_end=v['system.shaders.lastWaveEndTick'],stats_path=str(p))
 for c in ['VALU','MMOP','LDS','VMEM','SCA','MISC','FLAT']:q[c]=sum(x for k,x in v.items() if re.fullmatch(r'system.shaders.CUs[0-9]+.numInstrTypeExecuted::'+c,k))
 for c in ['waitVmCounter','waitLgkmCounter']:q[c]=sum(x for k,x in v.items() if re.fullmatch(r'system.shaders.CUs[0-9]+.SIMD[0-9]+.'+c,k))
 for key in ['totalTCCRequests','hitTCCRequests','missTCCRequests']:q[key]=v.get('system.shaders.cache_system.'+key)
 if q['missTCCRequests'] is None and q['totalTCCRequests'] is not None and q['hitTCCRequests'] is not None:q['missTCCRequests']=q['totalTCCRequests']-q['hitTCCRequests'];q['miss_source']='derived_total_minus_hit'
 else:q['miss_source']='raw'
 return q
def mean_stats(d,indices):
 rows=[stats(Path(d)/'m5out/0'/str(i)/'stats.txt') for i in indices]
 q={k:statistics.mean(v[k] for v in rows) for k in rows[0] if all(isinstance(v[k],(int,float)) for v in rows)}
 q['cycle_samples']=[v['cycles'] for v in rows];q['jitter_pct']=(max(q['cycle_samples'])-min(q['cycle_samples']))/q['cycles']*100;q['raw']=rows
 return q
def fit(x,y):
 mx=statistics.mean(x);my=statistics.mean(y);a=sum((u-mx)*(v-my) for u,v in zip(x,y))/sum((u-mx)**2 for u in x);b=my-a*mx
 den=sum((v-my)**2 for v in y)
 return dict(a=a,b=b,R2=1-sum((v-a*u-b)**2 for u,v in zip(x,y))/den if den else None)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,v))+' |' for v in rows])
def safety(baseline,out):
 base=json.loads(Path(baseline).read_text());root=Path(baseline).parent.parent;proof=dict(captured_at=datetime.datetime.now().isoformat(),processes={},repos={},hashes={})
 for pid,ticks in base['protected_ticks'].items():
  p=Path('/proc')/pid;v=(p/'stat').read_text().split();q=dict(state=v[2],cpu_ticks=int(v[13])+int(v[14]),all_threads_T=all((x/'stat').read_text().split()[2]=='T' for x in (p/'task').iterdir()))
  assert q['state']=='T' and q['all_threads_T'] and q['cpu_ticks']==ticks,(pid,q);proof['processes'][pid]=q
 for name,old in base['repos'].items():
  q={k:subprocess.check_output(['git','-C',str(root/name),*args],text=True).strip() for k,args in [('commit',['rev-parse','HEAD']),('status',['status','--porcelain']),('diff_stat',['diff','--stat'])]}
  assert q==old,(name,q,old);proof['repos'][name]=q
 for p,h in base['hashes'].items():
  new=hashlib.sha256(Path(p).read_bytes()).hexdigest();assert new==h,p;proof['hashes'][p]=new
 Path(out).write_text(json.dumps(proof,indent=2));return proof
