from pathlib import Path
import json,subprocess,time,hashlib
O=Path(__file__).resolve().parent;B=O.parent

def safety(tag):
 rows=[]
 for pid,ticks in [(61399,3189924),(63997,195118),(31138,1784),(54355,568),(31159,145)]:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().split();actual=int(s[13])+int(s[14]);threads=[(f/'stat').read_text().split()[2] for f in (p/'task').iterdir()];assert s[2]=='T' and actual==ticks and set(threads)=={'T'},(pid,s[2],actual,threads);rows.append(dict(pid=pid,state=s[2],cpu_ticks=actual))
 repos={}
 for r in ['gemm-optimization-20261020/bolt_ops','wdra48-fix-20261020/triton','fp8fp4-support-20261017/bolt_ops','fp8-e8m0-fix-20261017/bolt_ops','fp8-support-20261017/triton_gemm_asm']:
  status=subprocess.check_output(['git','-C',str(B/r),'status','--porcelain'],text=True);assert not status,(r,status);repos[r]=status
 d=dict(tag=tag,remote_epoch=time.time(),protected=rows,git_statuses=repos,passed=True)
 with (O/'safety.jsonl').open('a') as h:h.write(json.dumps(d)+'\n')
 return d

