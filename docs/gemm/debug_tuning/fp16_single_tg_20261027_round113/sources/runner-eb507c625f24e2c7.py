"""Aligned G171 experiment: merge AB data barriers, reserve 4/5 for MMAC turns."""
from pathlib import Path
import re,json,hashlib

def transform(source):
    # Literal barrier values are lexically local in every partition.
    constants={};merged=[];counts={'removed_B_init':0,'removed_B_invalidate':0,'removed_B_ready_wait':0,'removed_B_empty_arrive':0,'removed_B_seq':0}
    for line in source.splitlines(True):
        if re.match(r'      partition\d+\(',line):constants={}
        m=re.search(r'(%\w+) = llvm.mlir.constant\((\d+) : i32\)',line)
        if m:constants[m[1]]=int(m[2])
        m=re.search(r'rocdl.hcu.s.abarrier.(init|inv) (%\w+)',line)
        if m and constants.get(m[2]) in [4,5,6,7]:
            counts['removed_B_init' if m[1]=='init' else 'removed_B_invalidate']+=1;continue
        # Initial B-empty pre-arrival is redundant with the retained A-empty seed.
        if re.match(r'      %b_\d+ = rocdl.hcu.s.abarrier.arrive ',line):continue
        merged.append(line)
    s=''.join(merged);a=s.index('      partition0(');b=s.index('      partition1(');producer=s[a:b];lines=producer.splitlines(True)
    aready=[l for l in lines if re.match(r'\s*%a_\d+ = rocdl.hcu.s.abarrier.arrive ',l)];bready=[l for l in lines if re.match(r'\s*%b_\d+ = rocdl.hcu.s.abarrier.arrive ',l)];assert len(aready)==len(bready)==1
    waits=[l for l in lines if re.match(r'\s*%b_\d+ = rocdl.hcu.s.abarrier.try.wait ',l)];assert len(waits)==1
    seqs=[l for l in lines if 'rocdl.hcu.s.abarrier.seq ' in l];assert len(seqs)==2
    changed=[]
    for l in lines:
        if l==aready[0] or l==waits[0] or l==seqs[1]:continue
        changed.append(aready[0] if l==bready[0] else l)
    producer=''.join(changed);assert producer.count('amdg.matrix_load_to_local ')==2 and producer.count('rocdl.hcu.s.abarrier.seq ')==1
    s=s[:a]+producer+s[b:]
    # Both consumers acquire/release one common AB stage, after BOTH reads finish.
    tail=s[s.index('      partition1('):];tail,nw=re.subn(r'^\s*%b_\d+ = rocdl.hcu.s.abarrier.try.wait [^\n]*\n','',tail,flags=re.M);tail,na=re.subn(r'^\s*%b_\d+ = rocdl.hcu.s.abarrier.arrive [^\n]*\n','',tail,flags=re.M);assert nw==na==2
    s=s[:s.index('      partition1(')]+tail;counts.update(removed_B_ready_wait=nw,removed_B_empty_arrive=na,removed_B_seq=1)
    assert s.count('rocdl.hcu.s.abarrier.init ')==12,counts
    merged=s
    init='''      %turn_ping_id = arith.constant 4 : i32
      %turn_pong_id = arith.constant 5 : i32
      rocdl.hcu.s.abarrier.init %turn_ping_id, 4
      %turn_seed = rocdl.hcu.s.abarrier.arrive i32, %turn_ping_id, 1
      rocdl.hcu.s.abarrier.init %turn_pong_id, 4
'''
    s=s.replace('      ttg.warp_specialize(',init+'      ttg.warp_specialize(',1)
    first=s.index('      partition1(');second=s.index('      partition2(');parts=[s[:first]]
    for role,part in [(1,s[first:second]),(2,s[second:])]:
        part,n=re.subn(r'^\s*rocdl.s.barrier \{hcu.consumer_pingpong\}[^\n]*\n','',part,flags=re.M);assert n==(2 if role==1 else 3),(role,n)
        # Remove C1's last-iteration conditional now containing only a scheduling fence.
        part=re.sub(r'^          scf.if %\w+ \{\n\s*rocdl.sched.barrier 0[^\n]*\n          \}[^\n]*\n','',part,flags=re.M)
        in_id,out_id=(4,5) if role==1 else (5,4);head=part.index('\n')+1
        part=part[:head]+f'        %turn_in = arith.constant {in_id} : i32\n        %turn_out = arith.constant {out_id} : i32\n'+part[head:]
        loop=re.search(r'%accumulator:9 = scf.for (%\w+) = ',part);assert loop;iv=loop[1]
        needle=re.search(r'^          %partial = tt.dot [^\n]*\n',part,re.M);assert needle
        before=f'          %turn_phase = arith.andi {iv}, %c1_i32 : i32\n          %turn_wait = rocdl.hcu.s.abarrier.try.wait i32, %turn_in, %turn_phase\n          rocdl.sched.barrier 0\n'
        after='          rocdl.sched.barrier 0\n          %turn_signal = rocdl.hcu.s.abarrier.arrive i32, %turn_out, 1\n          rocdl.sched.barrier 0\n'
        part=part[:needle.start()]+before+needle[0]+after+part[needle.end():]
        pos=part.index('        ttg.warp_return');part=part[:pos]+'        rocdl.hcu.s.abarrier.inv %turn_in\n        rocdl.hcu.s.abarrier.inv %turn_out\n'+part[pos:];parts.append(part)
    token=''.join(parts);assert token.count('rocdl.hcu.s.abarrier.init ')==14 and 'hcu.consumer_pingpong}' not in token
    assert token.count('turn_signal = rocdl.hcu.s.abarrier.arrive')==2
    # Math statements must remain unchanged by both synchronization transforms.
    math=lambda q:[l.strip() for l in q.splitlines() if any(t in l for t in [' = tt.dot ',' = arith.mulf ',' = arith.addf '])]
    assert math(merged)==math(source)==math(token)
    proof={'data_merge':{'IDs_AB':[0,1,2,3],'IDs_scale':list(range(8,16)),'init_count':12,'empty_count':8,'empty_initial_arrival_per_producer_wave':2,'ready_count':4,'producer_ready_arrival_per_wave':1,'consumer_empty_arrival_per_wave':1,'MLS_seq':'one seq covers A and B transfers, one ready arrival after B','A_B_LDS_allocations':'separate and unchanged','B_slot_and_phase':'original synchronized two-slot counters retained'},'MMAC_turn':{'IDs':[4,5],'count':4,'ping_initial_seed_per_producer_wave':1,'phase':'K_iteration & 1','ping':'wait4 -> issue MMAC -> arrive5 -> scale work','pong':'wait5 -> issue MMAC -> arrive4 -> scale work','seq_used_for_turn':False,'init_count':14,'consumer_ebarrier_removed':True,'teardown':'invalidate after existing final consumer join'},'changes':counts,'arithmetic_unchanged':True}
    return merged,token,proof

if __name__=='__main__':
    import sys
    p=Path(sys.argv[1]);r=Path(sys.argv[2]);s=p.read_text();merged,token,proof=transform(s)
    (r/'abarrier-merged-AB.ttgir').write_text(merged);(r/'abarrier-turn.ttgir').write_text(token);proof.update(source=str(p),source_sha256=hashlib.sha256(s.encode()).hexdigest(),merged_sha256=hashlib.sha256(merged.encode()).hexdigest(),turn_sha256=hashlib.sha256(token.encode()).hexdigest());(r/'abarrier-turn-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('Prepared merged AB data sync and ordinary MMAC turn IDs4/5; arithmetic unchanged')
