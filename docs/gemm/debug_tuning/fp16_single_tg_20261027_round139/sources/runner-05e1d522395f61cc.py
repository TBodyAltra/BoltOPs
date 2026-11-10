"""Finite alias ownership model, individual consumer waves, collective MLS publication.
Producer publication abstracts qualified legacy seq/ready completion across four
producer waves. Alias guards require ALL consumer waves; physical-owner tests
also catch overwrite before a read starts, not only pending DS reads.
"""
import pathlib,json,hashlib,time,collections

def explore(overlap,per_role,gens,mutation=None,limit=20000000,wait_mask=0):
 assert wait_mask in (0,1,2,3)
 assert overlap in (1,2)
 ops=[('wait',0),('issue',0),('drain',0),('release',0)]
 for f in [0,1]:
  data=[('wait',f+1),('issue',f+1),('drain',f+1),('release',f+1)]
  ops+=(data+[('turn',f)] if wait_mask&(1<<f) else [('turn',f)]+data)+[('dot',f),('signal',f)]
 L=len(ops);issue={s:ops.index(('issue',s)) for s in range(3)};drain={s:ops.index(('drain',s)) for s in range(3)};release={s:ops.index(('release',s)) for s in range(3)};signal={f:ops.index(('signal',f)) for f in range(2)}
 def pack(st):
  p,a,b=st;v=p;assert p<256 and len(a)==len(b)==per_role and all(0<=c<256 for c in a+b)
  for c in a+b:v=(v<<8)|c
  return v
 def unpack(v):
  pcs=[]
  for _ in range(2*per_role):pcs.append(v&255);v>>=8
  pcs.reverse();return v,tuple(pcs[:per_role]),tuple(pcs[per_role:])
 def intersects(s,t):return s==t or (s==0 and 1<=t<=overlap) or (t==0 and 1<=s<=overlap)
 def ordinal(p,s):return s+3*((p-1-s)//3)
 def ready_turn(role,g,f,other,weak=False):
  previous=g*2+f-(1 if role==0 else 0)
  if previous<0:return mutation!='missing_turn_seed'
  threshold=(previous//2)*L+signal[previous%2]+1
  return sum(c>=threshold for c in other)>=(max(1,per_role//2) if weak else per_role)
 def released(cs,s,g):
  if g<0:return True
  threshold=(issue[s] if mutation=='early_slot_release' else release[s])+1
  need=per_role if mutation=='wrong_B1_empty_count' and s==2 else 2*per_role
  return sum(c>=g*L+threshold for c in cs)>=need
 def empty_phase(cs,s):
  count=per_role if mutation=='wrong_B1_empty_count' and s==2 else 2*per_role
  threshold=(issue[s] if mutation=='early_slot_release' else release[s])+1
  arrivals=2*per_role+sum(max(0,(c-threshold)//L+1) for c in cs)
  return (arrivals//count)&1
 def turn_phase_ready(role,g,f,other):
  seed=per_role if role==0 and mutation!='missing_turn_seed' else 0
  arrivals=seed+sum(max(0,(c-(signal[k]+1))//L+1) for c in other for k in range(2))
  count=max(1,per_role//2) if mutation=='half_turn_count' else per_role
  phase=(arrivals//count)&1
  target=0 if mutation=='wrong_turn_phase' else f
  return phase!=target
 start=(0,(0,)*per_role,(0,)*per_role);assert unpack(pack(start))==start;seen={pack(start)};todo=collections.deque(seen);transitions=0;terminal=0;begin=time.monotonic()
 def bad(reason,p,a,b,**extra):return dict(passed=False,reason=reason,overlap=overlap,per_role=per_role,generations=gens,mutation=mutation,state=[p,a,b],states=len(seen),**extra)
 while todo:
  p,a,b=unpack(todo.popleft());cs=a+b;successors=[]
  if p<3*gens:
   g,s=divmod(p,3);allow=empty_phase(cs,s)!=(g&1)
   if s==0 and mutation!='missing_B_empty_before_A':
    protected=range(1,overlap+1)
    if mutation=='omit_B1_empty_for_largeA':protected=[1]
    allow=allow and all(empty_phase(cs,t)!=(g&1) for t in protected)
   if s in range(1,overlap+1) and mutation!='missing_current_A_capture_wait':
    complete=max(0,(min(cs)-release[0]-1)//L+1)
    actual_phase=empty_phase(cs,0)
    target=(g&1) if mutation=='wrong_A_capture_phase' else ((g&1)^1)
    phase_ready=actual_phase!=target
    if mutation not in ['wrong_A_capture_phase','early_slot_release']:
     assert phase_ready==released(cs,0,g),(p,a,b,actual_phase,target)
    allow=allow and phase_ready
   if allow:
    for t in range(3):
     if not intersects(s,t):continue
     for c in cs:
      cg,idx=divmod(c,L)
      if issue[t]<idx<=drain[t]:return bad('OVERWRITE_PENDING_ALIASED_DS_READ',p,a,b,write_slot=s,read_slot=t,write_generation=g,read_generation=cg)
    successors.append((p+1,a,b))
  for role,own in enumerate([a,b]):
   other=b if role==0 else a
   for i,c in enumerate(own):
    if i and c==own[i-1]:continue
    if c>=L*gens:continue
    g,idx=divmod(c,L);op,s=ops[idx];allow=True
    if op=='wait':allow=ordinal(p,s)//3==g
    elif op=='issue':
     if ordinal(p,s)//3!=g:return bad('READ_WRONG_GENERATION',p,a,b,read_slot=s,generation=g)
     owner=max(ordinal(p,t) for t in range(3) if intersects(s,t))
     if owner!=3*g+s:return bad('READ_OVERWRITTEN_PHYSICAL_OWNER',p,a,b,read_slot=s,generation=g,actual_write_ordinal=owner)
    elif op=='turn':allow=turn_phase_ready(role,g,s,other)
    elif op=='dot' and not ready_turn(role,g,s,other):return bad('MMAC_ISSUE_TURN_OVERLAP',p,a,b,consumer=role,fragment=s,generation=g)
    if allow:
     cc=list(own);cc[i]+=1;cc=tuple(sorted(cc));successors.append((p,cc,b) if role==0 else (p,a,cc))
  if not successors:
   if p==3*gens and all(c==L*gens for c in cs):terminal+=1
   else:return bad('DEADLOCK',p,a,b)
  for st in successors:
   transitions+=1;v=pack(st)
   if v not in seen:
    seen.add(v);todo.append(v)
    if len(seen)>limit:raise RuntimeError('Incomplete alias model: state limit exceeded')
 assert terminal==1
 return dict(passed=True,overlap=overlap,consumer_waves_per_role=per_role,generations=gens,states=len(seen),transitions=transitions,seconds=time.monotonic()-begin,all_reachable_states_no_pending_DS_overwrite_no_stale_physical_owner_no_deadlock_no_MMAC_issue_overlap=True)


if __name__=='__main__':
 import resource
 R=pathlib.Path(__file__).resolve().parent;assert not(R/'alias-late-turn-protocol-model.json').exists()
 cap=resource.getrlimit(resource.RLIMIT_AS);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,cap[1]))
 source=R.parent/'single-tg-pipeline-20261027-round115-alias-ab-slots-cpu-fix/protocol-model-v3.json';original=json.loads(source.read_text());assert original['passed']
 controls=[];positives=[];negatives=[]
 def progress():
  (R/'alias-late-turn-model-progress.json').write_text(json.dumps(dict(controls=controls,positives=positives,negatives_completed=len(negatives),complete=False),indent=2)+'\n')
 for overlap in [1,2]:
  for w,g in [(1,4),(4,2),(4,4)]:
   q=explore(overlap,w,g,wait_mask=0);old=next(x for x in original['positive_cases'] if x['overlap']==overlap and x['consumer_waves_per_role']==w and x['generations']==g)
   assert q['passed'] and q['states']==old['states'] and q['transitions']==old['transitions'];controls.append(q);progress()
 for mask in [1,2,3]:
  for overlap in [1,2]:
   for w,g in [(1,4),(4,2),(4,4)]:
    q=explore(overlap,w,g,wait_mask=mask);q['wait_mask']=mask;assert q['passed'];positives.append(q);progress();print(json.dumps(q),flush=True)
   mutations=['missing_current_A_capture_wait','wrong_A_capture_phase','missing_B_empty_before_A','early_slot_release','wrong_B1_empty_count','missing_turn_seed','half_turn_count','wrong_turn_phase']
   if overlap==2:mutations+=['omit_B1_empty_for_largeA']
   for mutation in mutations:
    q=explore(overlap,4,2,mutation,wait_mask=mask);q['wait_mask']=mask;assert not q['passed'];negatives.append(q);progress()
 out=dict(passed=True,CPU_only=True,finite_model=True,controls_mask0_match_old_v3=controls,positive_cases=positives,negative_cases=negatives,wait_masks=[1,2,3],data_ids=list(range(6)),turn_ids=[6,7],producer_publication_order=['A','B0','B1'],producer_waves=4,consumer_waves_per_role=4,empty_count=8,ready_count=4,turn_count=4,scope='Conservative whole-slot alias overlap1/2, individual consumer waves, finite bounds; late token wait after matching data read/drain/release immediately before MMAC issue. Not physical MMAC completion or unbounded proof. Actual per-case LLVM/ISA/read and native checks required.',base_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 (R/'alias-late-turn-protocol-model.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,controls=len(controls),positives=len(positives),negatives=len(negatives))),flush=True)
