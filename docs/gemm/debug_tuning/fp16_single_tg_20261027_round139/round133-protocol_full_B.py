"""Finite data ownership model: OneA capture and one full B ready/empty channel.
All individual consumer waves; each producer publication abstracts all four MLS
copies completing before the collective ready phase. No MMAC ordering claim.
"""
import pathlib,json,hashlib,collections,time
def explore(waves,gens,mutation=None,limit=20000000,quotient_roles=True,collect_graph=False):
 ops=[('wait',0),('issue',0),('drain',0),('release',0),('wait',1),('issue',1),('drain',1),('dot',0),('issue',2),('drain',2),('release',2),('dot',1)]
 L=len(ops);issue={s:ops.index(('issue',s)) for s in range(3)};drain={s:ops.index(('drain',s)) for s in range(3)};release={0:3,1:10,2:10}
 def canonical(st):
  p,a,b=st;v=p
  for c in sorted(a+b):assert c<256;v=(v<<8)|c
  return v
 def pack(st):
  if quotient_roles:return canonical(st)
  p,a,b=st;v=p
  for c in a+b:assert c<256;v=(v<<8)|c
  return v
 def unpack(v):
  pcs=[]
  for _ in range(2*waves):pcs.append(v&255);v>>=8
  pcs.reverse();return v,tuple(pcs[:waves]),tuple(pcs[waves:])
 def ordinal(p,s):return s+3*((p-1-s)//3)
 def empty(cs,s):
  count=waves if mutation==('wrong_A_empty_count' if s==0 else 'wrong_B_empty_count') else 2*waves
  r=issue[0] if mutation=='early_A_release' and s==0 else drain[1] if mutation=='early_B_release' and s else release[s]
  arrivals=2*waves+sum(max(0,(c-r-1)//L+1) for c in cs)
  return (arrivals//count)&1
 initial=(0,(0,)*waves,(0,)*waves);seen={pack(initial)};todo=collections.deque(seen);terminal=edges=0;start=time.monotonic();graph_nodes=set();graph_edges=set()
 def bad(reason,p,a,b,**kw):return dict(passed=False,reason=reason,mutation=mutation,waves_per_role=waves,generations=gens,state=[p,a,b],states=len(seen),**kw)
 while todo:
  p,a,b=unpack(todo.popleft());cs=a+b;succ=[]
  if collect_graph:graph_nodes.add(canonical((p,a,b)))
  if p<3*gens:
   g,s=divmod(p,3);assert s in [0,1] # B0/B1 become ready atomically.
   target=(g&1)^(mutation=='wrong_producer_phase');allow=empty(cs,s)!=target
   if s==0 and mutation=='missing_A_empty_wait':allow=True
   if s==1:
    assert empty(cs,1)==empty(cs,2)
    if mutation=='missing_B_empty_wait':allow=True
   if allow:
    for slot in ([0] if s==0 else [1,2]):
     for c in cs:
      cg,idx=divmod(c,L)
      if issue[slot]<idx<=drain[slot]:return bad('OVERWRITE_PENDING_DS',p,a,b,slot=slot,read_generation=cg,write_generation=g)
    succ.append((p+(1 if s==0 else 2),a,b))
  for role,own in enumerate([a,b]):
   for i,c in enumerate(own):
    if i and c==own[i-1]:continue
    if c>=L*gens:continue
    g,idx=divmod(c,L);op,s=ops[idx];allow=True
    if op=='wait':allow=ordinal(p,s)//3==g or mutation==('missing_A_ready_wait' if s==0 else 'missing_B_ready_wait')
    if op=='issue' and ordinal(p,s)//3!=g:return bad('READ_STALE_GENERATION',p,a,b,slot=s,generation=g)
    if allow:
     cc=list(own);cc[i]+=1;cc=tuple(sorted(cc));succ.append((p,cc,b) if role==0 else (p,a,cc))
  if not succ:
   if p==3*gens and all(c==L*gens for c in cs):terminal+=1
   else:return bad('DEADLOCK',p,a,b)
  for st in succ:
   edges+=1;v=pack(st)
   if collect_graph:graph_edges.add((canonical((p,a,b)),canonical(st)))
   if v not in seen:
    seen.add(v);todo.append(v)
    if len(seen)>limit:raise RuntimeError('State bound exceeded: no completed proof')
 assert terminal==1
 out=dict(passed=True,waves_per_role=waves,total_consumer_waves=2*waves,generations=gens,states=len(seen),transitions=edges,seconds=time.monotonic()-start,identical_data_only_consumer_roles_quotiented=quotient_roles)
 if collect_graph:out.update(_nodes=graph_nodes,_edges=graph_edges)
 return out
if __name__=='__main__':
 R=pathlib.Path(__file__).resolve().parent;assert not(R/'full-B-protocol-model.json').exists();positives=[];negatives=[];equivalence=[]
 for w,g in [(1,1),(1,2),(1,4),(2,1),(2,2)]:
  a=explore(w,g,quotient_roles=False,collect_graph=True);b=explore(w,g,quotient_roles=True,collect_graph=True);assert a['passed'] and b['passed'] and a['_nodes']==b['_nodes'] and a['_edges']==b['_edges'];equivalence.append(dict(waves_per_role=w,generations=g,original_states=a['states'],quotient_states=b['states'],canonical_nodes=len(a['_nodes']),canonical_edges=len(a['_edges']),exact_quotient_graph_equal=True));print(json.dumps(equivalence[-1]),flush=True)
 for w,g in [(1,4),(4,2),(4,4)]:
  q=explore(w,g);assert q['passed'];positives.append(q);print(json.dumps(q),flush=True)
 for w in [1,4]:
  for m in ['missing_A_empty_wait','missing_B_empty_wait','missing_A_ready_wait','missing_B_ready_wait','early_A_release','early_B_release','wrong_A_empty_count','wrong_B_empty_count','wrong_producer_phase']:
   q=explore(w,2,m);assert not q['passed'],(w,m,q);negatives.append(q)
 (R/'full-B-protocol-model.json').write_text(json.dumps(dict(passed=True,scope='Finite individual-wave data ownership/liveness only; all-four-producer publication abstraction; no MMAC issue-order or physical-completion or unbounded-proof claim',exact_quotient_graph_controls=equivalence,symmetry_reason='All8consumer waves have identical data-only transition rules; producer guards, phase counts and hazard observations depend only on their PC multiset. Global sorting is an exact orbit quotient, not fewer waves/generations. Not applicable to role-dependent MMAC tokens.',hardware_barriers=dict(A=dict(empty=0,ready=1),B=dict(empty=2,ready=3)),empty_count=8,ready_count=4,positives=positives,negatives=negatives,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
 print(json.dumps(dict(passed=True,positives=len(positives),negatives=len(negatives))),flush=True)
