"""Finite exhaustive wave-interleaving ownership model; no device initialization.
Producer represents the four-wave ready publication after the legacy MLS
seq/copy/arrive protocol has completed its associated transfers. Consumers are individual waves; their equal protocol permits symmetry
reduction by sorting PCs. This checks protocol, not instruction scheduling.
"""
import collections,json,time,pathlib,hashlib
OPS=[('wait',0),('issue',0),('drain',0),('release',0),('wait',1),('issue',1),('drain',1),('release',1),('dot',0),('wait',2),('issue',2),('drain',2),('release',2),('dot',1)]
ISSUE={s:OPS.index(('issue',s)) for s in range(3)};DRAIN={s:OPS.index(('drain',s)) for s in range(3)};RELEASE={s:OPS.index(('release',s)) for s in range(3)};L=len(OPS)
def explore(waves,gens,mutation=None,limit=3000000):
 start=(0,(0,)*waves);seen={start};todo=collections.deque([start]);transitions=0;terminal=0;started=time.monotonic()
 def latest(p,s):return (p-1-s)//3
 while todo:
  p,cs=todo.popleft();successors=[]
  if p<3*gens:
   g,s=divmod(p,3)
   threshold=RELEASE[s]+1
   if mutation=='release_before_drain':threshold=ISSUE[s]+1
   ready=sum(c >= (g-1)*L+threshold for c in cs)
   required=waves//2 if mutation=='B1_empty_count4' and s==2 else waves
   allowed=(g==0 and mutation!='missing_initial_empty') or (g>0 and ready>=required)
   if allowed:
    # A refill may occur only after EVERY wave's previous read has completed.
    for c in cs:
     cg,ci=divmod(c,L)
     if cg==g-1 and ISSUE[s]<ci<=DRAIN[s]:
      return dict(passed=False,reason='OVERWRITE_WITH_PENDING_LDS_READ',state=[p,cs],slot=s,generation=g,states=len(seen),mutation=mutation)
    successors.append((p+1,cs))
  for i,c in enumerate(cs):
   if i and c==cs[i-1]:continue
   if c>=L*gens:continue
   g,opidx=divmod(c,L);op,s=OPS[opidx];allow=True
   if op=='wait':
    phase=g&1
    if mutation=='wrong_B1_ready_phase' and s==2:phase^=1
    allow=latest(p,s)==g and (latest(p,s)&1)==phase
   elif op=='issue' and latest(p,s)!=g:
    return dict(passed=False,reason='READ_WRONG_GENERATION',state=[p,cs],slot=s,generation=g,states=len(seen),mutation=mutation)
   if allow:
    cc=list(cs);cc[i]+=1;successors.append((p,tuple(sorted(cc))))
  if not successors:
   if p==3*gens and all(c==L*gens for c in cs):terminal+=1
   else:return dict(passed=False,reason='DEADLOCK',state=[p,cs],states=len(seen),mutation=mutation)
  for st in successors:
   transitions+=1
   if st not in seen:
    seen.add(st);todo.append(st)
    if len(seen)>limit:raise RuntimeError('Bound exceeded; incomplete proof')
 assert terminal==1
 return dict(passed=True,consumer_waves=waves,producer_waves=4,generations=gens,states=len(seen),transitions=transitions,terminal_states=terminal,all_reachable_states_no_pending_read_overwrite_or_wrong_generation_or_deadlock=True,seconds=time.monotonic()-started)
if __name__=='__main__':
 rows=[]
 for waves,gens in [(2,4),(4,4),(8,2)]:
  row=explore(waves,gens);print(json.dumps(row),flush=True);assert row['passed'];rows.append(row)
 negatives=[]
 for mutation in ['release_before_drain','B1_empty_count4','wrong_B1_ready_phase','missing_initial_empty']:
  x=explore(8,2,mutation);print(json.dumps(x),flush=True);assert not x['passed'];negatives.append(x)
 p=pathlib.Path(__file__);out=dict(passed=True,CPU_only=True,finite_protocol_model=True,MMAC_turns_disabled=True,positive_cases=rows,negative_cases=negatives,scope='All reachable symmetry-reduced individual-consumer-wave interleavings for the stated bounds. Four-producer ready publication abstracts the legacy MLS seq/copy/arrive completion semantics, separately paired for each slot; actual ISA pairing/teardown checks and PMD gold are required. No producer ebarrier6 is assumed. This is not an unbounded proof or performance measurement.',script_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (p.parent/'protocol-model.json').write_text(json.dumps(out,indent=2)+'\n')
