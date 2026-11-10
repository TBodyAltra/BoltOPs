"""Finite alias ownership model, individual consumer waves, collective MLS publication.
Producer publication abstracts qualified legacy seq/ready completion across four
producer waves. Alias guards require ALL consumer waves; physical-owner tests
also catch overwrite before a read starts, not only pending DS reads.
"""
import pathlib,json,hashlib,time,collections

def explore(overlap,per_role,gens,mutation=None,limit=20000000,publication_order=(0,1,2)):
 assert tuple(sorted(publication_order))==(0,1,2);rank={s:i for i,s in enumerate(publication_order)}
 assert overlap in (1,2)
 ops=[('wait',0),('issue',0),('drain',0),('release',0),('turn',0),('wait',1),('issue',1),('drain',1),('release',1),('dot',0),('signal',0),('turn',1),('wait',2),('issue',2),('drain',2),('release',2),('dot',1),('signal',1)]
 L=len(ops);issue={s:ops.index(('issue',s)) for s in range(3)};drain={s:ops.index(('drain',s)) for s in range(3)};release={s:ops.index(('release',s)) for s in range(3)};signal={f:ops.index(('signal',f)) for f in range(2)}
 def pack(st):
  p,a,b=st;v=p;assert p<256 and len(a)==len(b)==per_role and all(0<=c<256 for c in a+b)
  for c in a+b:v=(v<<8)|c
  return v
 def unpack(v):
  pcs=[]
  for _ in range(2*per_role):pcs.append(v&255);v>>=8
  pcs.reverse();return v,tuple(pcs[:per_role]),tuple(pcs[per_role:])
 def intersects(s,t):
  physical_overlap=2 if mutation=='wrong_physical_B1_alias' else overlap
  return s==t or (s==0 and 1<=t<=physical_overlap) or (t==0 and 1<=s<=physical_overlap)
 def ordinal(p,s):return rank[s]+3*((p-1-rank[s])//3)
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
   g,step=divmod(p,3);s=publication_order[step];allow=empty_phase(cs,s)!=(g&1)
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
     if owner!=3*g+rank[s]:return bad('READ_OVERWRITTEN_PHYSICAL_OWNER',p,a,b,read_slot=s,generation=g,actual_write_ordinal=owner)
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
 R=pathlib.Path(__file__).resolve().parent;assert not(R/'protocol-model.json').exists()
 positives=[];controls=[];negatives=[]
 for waves,gens in [(1,4),(4,2),(4,4)]:
  q=explore(1,waves,gens,publication_order=(0,1,2));assert q['passed'];controls.append(q)
  q=explore(1,waves,gens,publication_order=(0,2,1));assert q['passed'];positives.append(q);print(json.dumps(q),flush=True)
 for waves in [1,4]:
  for mutation in ['missing_current_A_capture_wait','wrong_A_capture_phase','missing_B_empty_before_A','early_slot_release','wrong_B1_empty_count','missing_turn_seed','half_turn_count','wrong_turn_phase','wrong_physical_B1_alias']:
   if waves==1 and mutation=='half_turn_count':continue # max(1,waves//2)==waves: no mutation exists in the one-wave abstraction
   q=explore(1,waves,2,mutation=mutation,publication_order=(0,2,1));assert not q['passed'],(mutation,waves,q);negatives.append(q);print(json.dumps(q),flush=True)
 out=dict(passed=True,scope='FINITE_REACHABLE_STATES_ONLY_NOT_ACTUAL_ISA_OR_UNBOUNDED_PROOF',producer_publication_order=['A','B1','B0'],supported_physical_overlap='A/B0 only; B1 physically disjoint',controls_original_order=controls,positives=positives,negatives=negatives,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),actual_producer_and_read_address_audits_and_ISA_phase_checks_still_required=True)
 (R/'protocol-model.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,positive_cases=len(positives),negative_cases=len(negatives))),flush=True)
