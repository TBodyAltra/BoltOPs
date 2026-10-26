"""Enumerate aggregate-four-wave events, checking buffer ownership and math order.
A supplement to compiler/ISA checks and native full-output/liveness validation.
"""
import pathlib,json,collections,hashlib
R=pathlib.Path(__file__).resolve().parent

def events(first,mode,n,bad_early=False,bad_phase=False,skip_first_wait=False):
 acts=[[],[],[]];second=1-first
 for k in range(n):
  p=k&1
  for operand in (first,second):acts[0]+=[('wait',2*operand,p),('load',operand,k),('arrive',2*operand+1,4)]
  for role in (1,2):
   e=acts[role];inp,out=(4,5) if role==1 else (5,4)
   for operand in (first,second):
    if operand!=first or not skip_first_wait:e.append(('wait',2*operand+1,p))
    e.append(('read',operand,k,0))
    if bad_early and operand==first:e += [('arrive',2*operand,4),('read',operand,k,1)]
    else:e += [('read',operand,k,1),('arrive',2*operand,4)]
   if mode:e.append(('wait',inp,p if mode==1 or bad_phase else 0))
   e.append(('mmac',role,k,0))
   if mode==2:e += [('arrive',out,4),('wait',inp,p if bad_phase else 1)]
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
 return dict(first_operand='AB'[first],mode=mode,loops=n,states=len(seen),terminal_states=terminal,passed=bool(terminal and not violations and not deadlocks),first_violation=violations,first_deadlock=deadlocks,scope='All aggregate-four-wave event interleavings: phase/count, LDS ownership, register generation and MMAC order; native mandatory.')
if __name__=='__main__':
 rows=[check(f,m,n) for f in (0,1) for m in (0,1,2) for n in (1,2,3,4)];assert all(x['passed'] for x in rows)
 bad_seed=[check(f,m,2,seed=False) for f in (0,1) for m in (1,2)];assert all(not x['passed'] and x['first_deadlock'] for x in bad_seed)
 bad_release=[check(f,2,3,bad_early=True) for f in (0,1)];assert all(not x['passed'] and x['first_violation'] for x in bad_release)
 bad_wait=[check(f,2,2,skip_first_wait=True) for f in (0,1)];assert all(not x['passed'] and x['first_violation'] for x in bad_wait)
 bad_phase=[check(f,2,2,bad_phase=True) for f in (0,1)];assert all(not x['passed'] and x['first_violation'] for x in bad_phase)
 q=dict(passed=True,source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),rows=rows,negative_controls=dict(no_turn_seed=bad_seed,release_before_all_reads=bad_release,missing_first_ready_wait=bad_wait,naive_iteration_phase=bad_phase))
 (R/'protocol-proof.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(passed=True,positive_cases=len(rows),states=sum(x['states'] for x in rows),negative_cases=len(bad_seed+bad_release+bad_wait+bad_phase))))
