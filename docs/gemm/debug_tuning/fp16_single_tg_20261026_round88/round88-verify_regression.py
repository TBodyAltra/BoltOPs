import pathlib,json,re,hashlib
R=pathlib.Path(__file__).resolve().parent;O=R.parent/'single-tg-pipeline-20261026-round86-one-operand-prefetch-native'
def norm(p):
 out=[]
 for line in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',line):break
  line=line.split(';',1)[0].strip()
  if line and not line.startswith(('.file','.loc','.Ltmp','.cfi')):out.append(line)
 return hashlib.sha256('\n'.join(out).encode()).hexdigest()
assert json.loads((R/'liveness-pilot-qualification.json').read_text())['passed'];old=json.loads((O/'failure-audit.json').read_text());assert old['status']=='REJECTED_COHORT_AUDITED'
new=[(p,json.loads(p.read_text())) for p in (R/'runs').glob('*/result.json')];checks=[]
for x in old['rows']:
 if x['status']!='ACCURACY_FAIL':continue
 original=json.loads((O/'runs'/x['id']/'result.json').read_text());matches=[(p,q) for p,q in new if q.get('status')=='PASS' and q.get('config')==original['config'] and q.get('shape')==original['shape']];assert len(matches)==1
 p,q=matches[0];assert q['accuracy']['mismatches']==q['accuracy']['nonfinite']==0 and q['metadata']['hcu_operand_barriers']=={'A':{'empty':2,'ready':3},'B':{'empty':0,'ready':1}}
 newpid=json.loads((p.parent/'executor.json').read_text())['child_pid'];assert newpid!=x['PID'];a=norm(O/'runs'/x['id']/'kernel.amdgcn');b=norm(p.parent/'kernel.amdgcn');assert a!=b
 checks.append(dict(layout=q['layout'],turn=q['config']['hcu_wasp_mmac_turn'],original_id=x['id'],original_mismatches=x['accuracy']['mismatches'],fixed_id=p.parent.name,fixed_mismatches=0,fixed_nonfinite=0,fixed_PID=newpid,actual_operand_barriers=q['metadata']['hcu_operand_barriers'],original_normalized_ISA_sha256=a,fixed_normalized_ISA_sha256=b))
assert len(checks)==4
out=dict(passed=True,scope='Exact four former failing configs at same K=BK*16 and seed29/unit scale; fresh processes, changed actualISA,0mismatch/nonfinite. Long-loop steadiness/performance/independent repeats still pending.',checks=checks,original_failure_audit_sha256=hashlib.sha256((O/'failure-audit.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
(R/'regression-fix-short-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,former_failing_configs_now_pass=4,all_b_first_actual_A_channel=[2,3],long_loop_and_fresh_pending=True)))
