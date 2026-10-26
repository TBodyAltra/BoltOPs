"""Finite exhaustive individual-wave protocol model for B slots and MMAC turns.
Sort PCs only within the two distinct consumer roles. Producer publication is
abstracted by the qualified legacy MLS seq/copy/arrive completion semantics.
Dots model issue ordering, not physical completion of the MMAC pipeline.
"""
import pathlib,collections,json,time,hashlib

def explore(mode,per_role,gens,mutation=None,limit=20000000,wait_mask=0):
 assert type(wait_mask) is int and wait_mask in (0,1,2,3)
 assert not wait_mask or mode==2
 assert mode in (1,2)
 ops=[('wait',0),('issue',0),('drain',0),('release',0),('turn',0),('wait',1),('issue',1),('drain',1),('release',1),('dot',0)]
 if mode==2:ops += [('signal',0),('turn',1)]
 ops += [('wait',2),('issue',2),('drain',2),('release',2),('dot',1),('signal',1 if mode==2 else 0)]
 for f in range(2):
  if wait_mask & (1<<f):
   ops.remove(('turn',f));ops.insert(ops.index(('dot',f)),('turn',f))
 L=len(ops);issue={s:ops.index(('issue',s)) for s in range(3)};drain={s:ops.index(('drain',s)) for s in range(3)};release={s:ops.index(('release',s)) for s in range(3)};signal={f:ops.index(('signal',f)) for f in range(mode)}
 def pack(st):
  p,a,b=st;assert p<256 and len(a)==len(b)==per_role and all(0<=c<256 for c in a+b)
  v=p
  for c in a+b:v=(v<<8)|c
  return v
 def unpack(v):
  cs=[]
  for _ in range(2*per_role):cs.append(v&255);v>>=8
  cs.reverse();return v,tuple(cs[:per_role]),tuple(cs[per_role:])
 start=(0,(0,)*per_role,(0,)*per_role);encoded=pack(start);assert unpack(encoded)==start;seen={encoded};todo=collections.deque([encoded]);transitions=0;terminal=0;started=time.monotonic()
 def latest(p,s):return (p-1-s)//3
 def required(role,g,f):
  ordinal=g*mode+f;previous=ordinal-1 if role==0 else ordinal
  return None if previous<0 else (previous//mode)*L+signal[previous%mode]+1
 def all_turn_ready(role,g,f,other,weak=False):
  threshold=required(role,g,f)
  if threshold is None:return mutation!='missing_turn_seed'
  count=sum(c>=threshold for c in other);need=per_role//2 if weak and per_role>1 else per_role
  return count>=need
 while todo:
  p,a,b=unpack(todo.popleft());groups=(a,b);cs=a+b;successors=[]
  if p<3*gens:
   g,s=divmod(p,3);threshold=issue[s]+1 if mutation=='early_slot_release' else release[s]+1
   ready=sum(c>=(g-1)*L+threshold for c in cs);need=per_role if mutation=='wrong_B1_empty_count' and s==2 else per_role*2
   if g==0 or ready>=need:
    for c in cs:
     cg,ci=divmod(c,L)
     if cg==g-1 and issue[s]<ci<=drain[s]:return dict(passed=False,reason='OVERWRITE_PENDING_DS_READ',state=[p,a,b],slot=s,generation=g,states=len(seen),mutation=mutation)
    successors.append((p+1,a,b))
  for role,own in enumerate(groups):
   other=groups[1-role]
   for i,c in enumerate(own):
    if i and c==own[i-1]:continue
    if c>=L*gens:continue
    g,idx=divmod(c,L);op,s=ops[idx];allow=True
    if op=='wait':allow=latest(p,s)==g
    elif op=='issue' and latest(p,s)!=g:return dict(passed=False,reason='READ_WRONG_GENERATION',state=[p,a,b],slot=s,generation=g,states=len(seen),mutation=mutation)
    elif op=='turn':
     ordinal=g*mode+s;phase=(g&1) if mode==1 else s
     if mutation=='wrong_turn_phase':phase=0
     allow=phase==(ordinal&1) and all_turn_ready(role,g,s,other,mutation=='half_turn_count')
    elif op=='dot':
     f=s if mode==2 else 0
     if not all_turn_ready(role,g,f,other):return dict(passed=False,reason='MMAC_ISSUE_TURN_OVERLAP',state=[p,a,b],role=role,generation=g,fragment=s,states=len(seen),mutation=mutation)
    if allow:
     cc=list(own);cc[i]+=1;cc=tuple(sorted(cc));successors.append((p,cc,b) if role==0 else (p,a,cc))
  if not successors:
   if p==3*gens and all(c==L*gens for c in cs):terminal+=1
   else:return dict(passed=False,reason='DEADLOCK',state=[p,a,b],states=len(seen),mutation=mutation)
  for st in successors:
   transitions+=1
   encoded=pack(st)
   if encoded not in seen:
    seen.add(encoded);todo.append(encoded)
    if len(seen)>limit:raise RuntimeError(f'Incomplete: state bound {limit} exceeded for {mode}/{per_role}/{gens}')
 assert terminal==1
 return dict(passed=True,mode=mode,wait_mask=wait_mask,consumer_waves_per_role=per_role,generations=gens,states=len(seen),transitions=transitions,all_reachable_states_no_deadlock_overwrite_wrong_generation_or_mmac_issue_overlap=True,seconds=time.monotonic()-started)
if __name__=='__main__':
 rows=[];negatives=[]
 for mask in [1,2,3]:
  for per_role,gens in [(1,4),(4,2)]:
   r=explore(2,per_role,gens,wait_mask=mask);print(json.dumps(r),flush=True);assert r['passed'];rows.append(r)
  for mutation in ['early_slot_release','wrong_B1_empty_count','missing_turn_seed','wrong_turn_phase','half_turn_count']:
   r=explore(2,4,2,mutation,wait_mask=mask);print(json.dumps(dict(wait_mask=mask,**r)),flush=True);assert not r['passed'];negatives.append(dict(wait_mask=mask,**r))
 p=pathlib.Path(__file__);out=dict(passed=True,CPU_only=True,finite_model=True,positive_cases=rows,negative_cases=negatives,data_barrier_ids=list(range(6)),turn_barrier_ids=[6,7],turn_count_per_role=4,producer_ready_completion_abstracted_via_legacy_MLS_protocol=True,state_encoding='Lossless fixed 8-bit fields for producer and every consumer PC; no extra state quotient',state_limit=20000000,scope='Finite all-reachable individual-wave interleavings within stated bounds; late token wait after data release before dot issue. Not unbounded execution, emitted code, physical completion, or performance. Actual LLVM/ISA and native full gold required.',script_sha256=hashlib.sha256(p.read_bytes()).hexdigest());(p.parent/'protocol-model.json').write_text(json.dumps(out,indent=2)+'\n')
