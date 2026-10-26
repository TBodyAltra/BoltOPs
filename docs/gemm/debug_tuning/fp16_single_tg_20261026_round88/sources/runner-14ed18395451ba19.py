"""Exhaustive group-event interleavings; supplementary to native liveness tests.
Each actor represents four waves after their aggregate count completes. This
checks phases, aggregate counts and program dependencies, not instruction timing.
"""
import pathlib,json,collections,hashlib
R=pathlib.Path(__file__).resolve().parent

def events(mode,n,bad_phase=False,merged=False):
    acts=[[],[],[]]
    for k in range(n):
        p=k&1
        acts[0]+=([('wait',0,p),('load','A',k),('load','B',k),('arrive',1,4)] if merged else [('wait',0,p),('load','A',k),('arrive',1,4),('wait',2,p),('load','B',k),('arrive',3,4)])
        for role in [1,2]:
            inp,out=(4,5) if role==1 else (5,4)
            e=acts[role];e += [('wait',1,p)]+([] if merged else [('wait',3,p)])
            e += [('read','A0',k),('read','B0',k),('read','A1',k),('read','B1',k),('arrive',0,4)]+([] if merged else [('arrive',2,4)])
            if mode:e += [('wait',inp,p if mode==1 or bad_phase else 0)]
            e += [('mmac',role,k,0)]
            if mode==2:e += [('arrive',out,4),('wait',inp,p if bad_phase else 1)]
            e += [('mmac',role,k,1)]+([('arrive',out,4)] if mode else [])
    return acts

def check(mode,n,seed=True,bad_phase=False,merged=False):
    acts=events(mode,n,bad_phase,merged);thresholds=(8,4,8,4,4,4)
    # Empty A/B initially full and ping4 seeded by producer's four waves.
    init=((0,0,0),(1,0,1,0,1 if seed else 0,0),(0,0,0,0,0,0),0)
    pending=collections.deque([init]);seen={init};terminal=0;violations=[];deadlocks=[]
    expected=[(role,k,f) for k in range(n) for role,f in ([(1,0),(1,1),(2,0),(2,1)] if mode==1 else [(1,0),(2,0),(1,1),(2,1)])]
    while pending:
        pcs,phase,counts,mmac=pending.popleft();enabled=0
        if all(pcs[i]==len(acts[i]) for i in range(3)):
            assert mmac==4*n;terminal+=1;continue
        for who in range(3):
            if pcs[who]==len(acts[who]):continue
            op=acts[who][pcs[who]]
            if op[0]=='wait' and phase[op[1]]==op[2]:continue
            enabled+=1;np=list(pcs);np[who]+=1;ph=list(phase);cs=list(counts);nm=mmac
            if op[0]=='arrive':
                ident,amount=op[1:];cs[ident]+=amount;assert cs[ident]<=thresholds[ident]
                if cs[ident]==thresholds[ident]:cs[ident]=0;ph[ident]^=1
            if op[0]=='mmac':
                actual=op[1:]
                if mode and actual!=expected[mmac]:
                    violations.append(dict(pcs=pcs,actual=actual,expected=expected[mmac]));continue
                nm+=1
            state=(tuple(np),tuple(ph),tuple(cs),nm)
            if state not in seen:seen.add(state);pending.append(state)
        if not enabled:deadlocks.append(dict(pcs=pcs,phase=phase,counts=counts,mmac=mmac))
    return dict(mode=mode,merged=merged,loops=n,states=len(seen),terminal_states=terminal,deadlock_states=len(deadlocks),order_violations=len(violations),passed=bool(terminal and not deadlocks and not violations),first_deadlock=deadlocks[:1],first_violation=violations[:1],scope='all aggregate-four-wave program-event interleavings; native execution remains mandatory')

if __name__=='__main__':
 rows=[check(m,n,merged=ab) for ab in (False,True) for m in (0,1,2) for n in (1,2,3,4)]
 assert all(x['passed'] for x in rows)
 bad_seed=[check(m,2,seed=False) for m in (1,2)];assert all(not x['passed'] and x['deadlock_states'] for x in bad_seed)
 bad_phase=check(2,2,bad_phase=True);assert not bad_phase['passed'] and bad_phase['order_violations']
 proof=dict(passed=True,full_K_local_prefetch=True,rows=rows,negative_controls=dict(no_seed=bad_seed,naive_iteration_phase_for_two_fragments=bad_phase),source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 (R/'protocol-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(passed=True,states=sum(x['states'] for x in rows),positive_cases=len(rows),negative_controls=3)))
