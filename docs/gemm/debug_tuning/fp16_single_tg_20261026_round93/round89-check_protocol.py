"""Enumerate aggregate-four-wave events, checking buffer ownership and math order.
A supplement to compiler/ISA checks and native full-output/liveness validation.
"""
import pathlib,json,collections,hashlib
R=pathlib.Path(__file__).resolve().parent

def events(first,mode,n,prefetch=0,bad_early=False,bad_other_release=False,bad_phase=False,skip_first_wait=False,legacy_ids=False):
 assert mode in (0,1,2) and prefetch in (0,1)
 acts=[[],[],[]];other=1-prefetch
 physical={first:0,1-first:2};bound={0:0,1:2} if legacy_ids else physical
 for k in range(n):
  p=k&1
  for operand in (first,1-first):acts[0]+=[('wait',physical[operand],p),('load',operand,k),('arrive',physical[operand]+1,4)]
  for role in (1,2):
   e=acts[role];inp,out=(4,5) if role==1 else (5,4)
   if not skip_first_wait:e.append(('wait',bound[prefetch]+1,p))
   e.append(('read',prefetch,k,0))
   if bad_early:e += [('arrive',bound[prefetch],4),('read',prefetch,k,1)]
   else:e += [('read',prefetch,k,1),('arrive',bound[prefetch],4)]
   # The first MMAC token is acquired BEFORE the other-ready wait and K64 read.
   if mode:e.append(('wait',inp,p if mode==1 or bad_phase else 0))
   e += [('wait',bound[other]+1,p),('read',other,k,0),('mmac',role,k,0)]
   if mode==2:e.append(('arrive',out,4))
   e += [('arrive',bound[other],4),('read',other,k,1)] if bad_other_release else [('read',other,k,1),('arrive',bound[other],4)]
   if mode==2:e.append(('wait',inp,p if bad_phase else 1))
   e.append(('mmac',role,k,1))
   if mode:e.append(('arrive',out,4))
 return acts

def check(first,mode,n,seed=True,**kw):
 acts=events(first,mode,n,**kw);thresholds=(8,4,8,4,4,4)
 init=((0,0,0),(1,0,1,0,1 if seed else 0,0),(0,)*6,0,(-1,-1),(15,15),(-1,)*8)
 pending=collections.deque([init]);seen={init};terminal=0;violations=[];deadlocks=[]
 expected=[(role,k,f) for k in range(n) for role,f in ([(1,0),(1,1),(2,0),(2,1)] if mode==1 else [(1,0),(2,0),(1,1),(2,1)])]
 while pending:
  pcs,phase,counts,mmac,slots,masks,regs=pending.popleft();enabled=0
  if all(pcs[i]==len(acts[i]) for i in range(3)):
   assert mmac==4*n and masks==(15,15);terminal+=1;continue
  for who in range(3):
   if pcs[who]==len(acts[who]):continue
   op=acts[who][pcs[who]]
   if op[0]=='wait' and phase[op[1]]==op[2]:continue
   enabled+=1;np=list(pcs);np[who]+=1;ph=list(phase);cs=list(counts);nm=mmac;sl=list(slots);ms=list(masks);rg=list(regs);error=None
   if op[0]=='arrive':
    ident,amount=op[1:];cs[ident]+=amount;assert cs[ident]<=thresholds[ident]
    if cs[ident]==thresholds[ident]:cs[ident]=0;ph[ident]^=1
   if op[0]=='load':
    operand,k=op[1:]
    if masks[operand]!=15:error='Producer overwrites an operand before both consumers capture both K64 fragments'
    else:sl[operand]=k;ms[operand]=0
   if op[0]=='read':
    operand,k,f=op[1:];bit=1<<((who-1)*2+f)
    if slots[operand]!=k:error='Consumer reads the wrong tile generation'
    elif masks[operand]&bit:error='Duplicate operand read'
    else:ms[operand]|=bit;rg[(who-1)*4+operand*2+f]=k
   if op[0]=='mmac':
    role,k,f=op[1:]
    if any(regs[(role-1)*4+operand*2+f]!=k for operand in (0,1)):error='MMAC uses uncaptured or wrong-generation operand'
    elif mode and op[1:]!=expected[mmac]:error='MMAC turn order violation'
    else:nm+=1
   if error:
    if len(violations)<1:violations.append(dict(pcs=pcs,event=op,reason=error,slots=slots,masks=masks,regs=regs))
    continue
   state=(tuple(np),tuple(ph),tuple(cs),nm,tuple(sl),tuple(ms),tuple(rg))
   if state not in seen:seen.add(state);pending.append(state)
  if not enabled:
   if len(deadlocks)<1:deadlocks.append(dict(pcs=pcs,phase=phase,counts=counts))
 return dict(producer_first='AB'[first],prefetched_operand='AB'[kw.get('prefetch',0)],mode=mode,loops=n,states=len(seen),terminal_states=terminal,passed=bool(terminal and not violations and not deadlocks),first_violation=violations,first_deadlock=deadlocks,scope='All aggregate-four-wave event interleavings: phase/count, LDS ownership, register generation and MMAC order; native mandatory.')
if __name__=='__main__':
 rows=[check(f,m,n,prefetch=p) for f in (0,1) for p in (0,1) for m in (0,1,2) for n in (1,2,3,4)];assert all(x['passed'] for x in rows)
 bad_seed=[check(f,m,2,seed=False,prefetch=p) for f in (0,1) for p in (0,1) for m in (1,2)];assert all(not x['passed'] and x['first_deadlock'] for x in bad_seed)
 bad_release=[check(f,2,3,prefetch=f,bad_early=True) for f in (0,1)];assert all(not x['passed'] and x['first_violation'] for x in bad_release)
 bad_other=[check(f,2,3,prefetch=1-f,bad_other_release=True) for f in (0,1)];assert all(not x['passed'] and x['first_violation'] for x in bad_other)
 bad_wait=[check(f,2,2,prefetch=p,skip_first_wait=True) for f in (0,1) for p in (0,1)];assert all(not x['passed'] and x['first_violation'] for x in bad_wait)
 bad_phase=[check(f,2,2,prefetch=p,bad_phase=True) for f in (0,1) for p in (0,1)];assert all(not x['passed'] and x['first_violation'] for x in bad_phase)
 wrong_map=[check(1,m,2,prefetch=0,legacy_ids=True) for m in (1,2)];assert all(not x['passed'] and x['first_violation'] for x in wrong_map)
 mode0_bad_wait=[check(f,0,2,prefetch=p,skip_first_wait=True) for f in (0,1) for p in (0,1)];assert all(not x['passed'] and x['first_violation'] for x in mode0_bad_wait)
 q=dict(passed=True,actual_barrier_assignment='producerfirst0/1,producersecond2/3; matching source-derived map',source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),rows=rows,negative_controls=dict(no_turn_seed=bad_seed,prefetched_release_before_all_reads=bad_release,streamed_release_before_last_read=bad_other,missing_prefetched_ready_wait=bad_wait,naive_iteration_phase=bad_phase,legacy_A0_B2_binding_on_Bfirst=wrong_map,mode0_missing_ready=mode0_bad_wait))
 (R/'protocol-proof.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(passed=True,positive_cases=len(rows),states=sum(x['states'] for x in rows),negative_cases=len(bad_seed+bad_release+bad_other+bad_wait+bad_phase+wrong_map+mode0_bad_wait))))
