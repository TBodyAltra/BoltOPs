"""Finite data-only projection for aliased A/B0/B1, no cross-role MMAC tokens.
Each individual consumer's data operations remain ordered. Register-only math
is erased as stuttering for this DATA ownership proof; no MMAC order/completion
claim. Producer publication abstracts all four qualified MLS waves collectively.
"""
import pathlib,json,hashlib,collections,time

def explore(overlap,waves,gens,mutation=None,limit=20000000,quotient=True,collect_graph=False):
 assert overlap in (1,2) and waves in (1,2,4)
 ops=[(op,s) for s in range(3) for op in ['wait','issue','drain','release']];L=len(ops)
 issue={s:ops.index(('issue',s)) for s in range(3)};drain={s:ops.index(('drain',s)) for s in range(3)};release={s:ops.index(('release',s)) for s in range(3)}
 def canonical(st):
  p,a,b=st;v=p
  for c in sorted(a+b):assert 0<=c<256;v=(v<<8)|c
  return v
 def pack(st):
  if quotient:return canonical(st)
  p,a,b=st;v=p
  for c in a+b:assert 0<=c<256;v=(v<<8)|c
  return v
 def unpack(v):
  cs=[]
  for _ in range(2*waves):cs.append(v&255);v>>=8
  cs.reverse();return v,tuple(cs[:waves]),tuple(cs[waves:])
 def intersects(s,t):return s==t or (s==0 and 1<=t<=overlap) or (t==0 and 1<=s<=overlap)
 def ordinal(p,s):return s+3*((p-1-s)//3)
 def empty(cs,s):
  count=waves if mutation==['wrong_A_empty_count','wrong_B0_empty_count','wrong_B1_empty_count'][s] else 2*waves
  threshold=(issue[s] if mutation in ['early_all_release',['early_A_release','early_B0_release','early_B1_release'][s]] else release[s])+1
  arrivals=2*waves+sum(max(0,(c-threshold)//L+1) for c in cs)
  return (arrivals//count)&1
 def ready(p,s):
  pubs=max(0,(p-1-s)//3+1)
  # Four producer waves publish one common ready phase. No assertion that
  # a two-wave early publication would be safe; native/address/ISA proof needed.
  count=2 if mutation=='wrong_ready_count' else 4
  return ((4*pubs)//count)&1
 initial=(0,(0,)*waves,(0,)*waves);seen={pack(initial)};todo=collections.deque(seen);nodes=set();edges=set();terminal=transitions=0;started=time.monotonic()
 def bad(reason,p,a,b,**kw):return dict(passed=False,reason=reason,mutation=mutation,overlap=overlap,waves_per_role=waves,generations=gens,state=[p,a,b],states=len(seen),**kw)
 while todo:
  p,a,b=unpack(todo.popleft());cs=a+b;succ=[]
  if collect_graph:nodes.add(canonical((p,a,b)))
  if p<3*gens:
   g,s=divmod(p,3);target=(g&1)^(mutation=='wrong_producer_phase');allow=empty(cs,s)!=target
   if mutation==['missing_A_empty_wait','missing_B0_empty_wait','missing_B1_empty_wait'][s]:allow=True
   if s==0 and mutation!='missing_B_empty_before_A':
    protected=[1] if mutation=='omit_B1_empty_for_largeA' else range(1,overlap+1)
    allow=allow and all(empty(cs,t)!=(g&1) for t in protected)
   if 1<=s<=overlap and mutation!='missing_current_A_capture_wait':
    target=(g&1) if mutation=='wrong_A_capture_phase' else ((g&1)^1)
    allow=allow and empty(cs,0)!=target
   if allow:
    for t in range(3):
     if intersects(s,t):
      for c in cs:
       cg,i=divmod(c,L)
       if issue[t]<i<=drain[t]:return bad('OVERWRITE_PENDING_ALIASED_DS_READ',p,a,b,write_slot=s,read_slot=t,write_generation=g,read_generation=cg)
    succ.append((p+1,a,b))
  for role,own in enumerate([a,b]):
   for i,c in enumerate(own):
    if i and c==own[i-1]:continue
    if c>=L*gens:continue
    g,idx=divmod(c,L);op,s=ops[idx];allow=True
    if op=='wait':
     target=(g&1)^(mutation=='wrong_consumer_phase');allow=ready(p,s)!=target
     if mutation=='missing_ready_wait':allow=True
     if mutation is None:assert allow==(ordinal(p,s)//3==g)
    elif op=='issue':
     if ordinal(p,s)//3!=g:return bad('READ_WRONG_GENERATION',p,a,b,slot=s,generation=g)
     owner=max(ordinal(p,t) for t in range(3) if intersects(s,t))
     if owner!=3*g+s:return bad('READ_OVERWRITTEN_PHYSICAL_OWNER',p,a,b,slot=s,generation=g,write_ordinal=owner)
    if allow:
     cc=list(own);cc[i]+=1;cc=tuple(sorted(cc));succ.append((p,cc,b) if role==0 else (p,a,cc))
  if not succ:
   if p==3*gens and all(c==L*gens for c in cs):terminal+=1
   else:return bad('DEADLOCK',p,a,b)
  for st in succ:
   transitions+=1;v=pack(st)
   if collect_graph:edges.add((canonical((p,a,b)),canonical(st)))
   if v not in seen:
    seen.add(v);todo.append(v)
    if len(seen)>limit:raise RuntimeError('Incomplete proof: state bound exceeded')
 assert terminal==1
 out=dict(passed=True,overlap=overlap,waves_per_role=waves,total_consumer_waves=2*waves,generations=gens,states=len(seen),transitions=transitions,seconds=time.monotonic()-started,quotient=quotient)
 if collect_graph:out.update(_nodes=nodes,_edges=edges)
 return out

if __name__=='__main__':
 import resource
 R=pathlib.Path(__file__).resolve().parent;assert not(R/'alias-no-turn-protocol-model.json').exists()
 cap=resource.getrlimit(resource.RLIMIT_AS);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,cap[1]))
 controls=[];positives=[];negatives=[]
 def progress():
  (R/'model-progress.json').write_text(json.dumps(dict(complete=False,controls=controls,positives=positives,negatives_completed=len(negatives)),indent=2)+'\n')
 for overlap in [1,2]:
  for w,g in [(1,1),(1,2),(1,4),(2,1),(2,2)]:
   a=explore(overlap,w,g,quotient=False,collect_graph=True);b=explore(overlap,w,g,collect_graph=True)
   assert a['passed'] and b['passed'] and a['_nodes']==b['_nodes'] and a['_edges']==b['_edges']
   controls.append(dict(overlap=overlap,waves_per_role=w,generations=g,original_states=a['states'],quotient_states=b['states'],canonical_nodes=len(a['_nodes']),canonical_edges=len(a['_edges']),exact_canonical_node_and_edge_graphs_equal=True));progress();print(json.dumps(controls[-1]),flush=True)
 for overlap in [1,2]:
  for w,g in [(1,4),(4,2),(4,4)]:
   q=explore(overlap,w,g);assert q['passed'];positives.append(q);progress();print(json.dumps(q),flush=True)
  muts=['missing_current_A_capture_wait','wrong_A_capture_phase','missing_B_empty_before_A','missing_A_empty_wait','missing_B0_empty_wait','missing_B1_empty_wait','early_all_release','early_A_release','early_B0_release','early_B1_release','wrong_A_empty_count','wrong_B0_empty_count','wrong_B1_empty_count','wrong_producer_phase','wrong_consumer_phase','missing_ready_wait','wrong_ready_count']
  if overlap==2:muts+=['omit_B1_empty_for_largeA']
  for w in [1,4]:
   for mutation in muts:
    q=explore(overlap,w,2,mutation);assert not q['passed'],(mutation,overlap,w,q);negatives.append(q);progress()
 out=dict(passed=True,CPU_only=True,finite_model=True,scope='Data ownership/liveness only; math erasure is stuttering for data-only properties; no MMAC issue order/physical completion/unbounded proof; actual emitted address/read/ownership and native fullgold required',symmetry='All eight consumer data projections have identical rules; producer guards/counts and physical owner/read hazards depend only on their PC multiset. Global sorting is an exact orbit quotient. No role-dependent math turn exists.',data_ids=list(range(6)),turn_ids=[],empty_count=8,ready_count=4,producer_waves=4,producer_order=['A','B0','B1'],controls=controls,positive_cases=positives,negative_cases=negatives,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 (R/'alias-no-turn-protocol-model.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,controls=len(controls),positives=len(positives),negatives=len(negatives))),flush=True)
