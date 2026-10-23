"""Batch current-iteration A/B/AS/BS LDS reads, with safe delayed slot release."""
from pathlib import Path
import re,json,hashlib
from matrix_prefetch import normalize_mls

def transform(text):
    text,aliases=normalize_mls(text);edits=[];proof=[]
    for match in re.finditer(r'^        %accumulator:9 = scf.for[^\n]+\n',text,re.M):
        a=match.end();b=text.index('        } {hcu.wasp_split_policy',a);lines=text[a:b].splitlines(True)
        defs={};names={}
        for i,l in enumerate(lines):
            m=re.match(r'\s*(%\w+) = ',l)
            if m:defs[m[1]]=i;names[i]=m[1]
        reads=[i for i,l in enumerate(lines) if ' = ttg.local_load ' in l]
        waits=[i for i,l in enumerate(lines) if ' = rocdl.hcu.s.abarrier.try.wait ' in l]
        releases=[i for i,l in enumerate(lines) if ' = rocdl.hcu.s.abarrier.arrive ' in l]
        assert len(reads)==len(waits)==len(releases)==4,(reads,waits,releases)
        assert ['xf8E4M3FN' in lines[i] for i in reads]==[True,True,False,False]
        dot=next(i for i,l in enumerate(lines) if ' = tt.dot ' in l)
        operands=re.search(r'tt.dot (%\w+), (%\w+),',lines[dot]).groups()
        assert tuple(names[i] for i in reads[:2])==operands
        selected=set()
        def visit(i):
            if i in selected:return
            selected.add(i)
            rhs=lines[i].split(' = ',1)[1]
            for n in re.findall(r'%\w+',rhs):
                if n in defs:visit(defs[n])
        for i in reads+waits+releases:visit(i)
        pure=sorted(selected-set(reads+waits+releases))
        assert all(not any(x in lines[i] for x in ['tt.dot','arith.mulf','arith.addf','scf.','ttg.local_load']) for i in pure)
        for i in releases:
            assert len(re.findall(re.escape(names[i])+r'\b',''.join(lines)))==1,'slot release result unexpectedly used'
        batch=''.join(lines[i] for i in pure+waits+reads)
        remaining=[l for i,l in enumerate(lines) if i not in selected]
        # The first scale expand is outside the second consumer's conditional handoff.
        j=next(i for i,l in enumerate(remaining) if ' = tt.expand_dims ' in l and any(names[v] in l for v in reads[2:]))
        assert any(' = tt.dot ' in l for l in remaining[:j])
        # Preserve all existing ping-pong sync and all mathematical statements/order.
        changed=batch+''.join(remaining[:j])+''.join(lines[i] for i in releases)+''.join(remaining[j:])
        def math(s):return [l.strip() for l in s.splitlines() if any(q in l for q in [' = tt.dot ',' = arith.mulf ',' = arith.addf '])]
        assert math(changed)==math(''.join(lines))
        edits.append((a,b,changed));proof.append({'consumer':len(proof),'ready_waits':4,'read_order':['A','B','A_scale','B_scale'],'all_ready_waits_before_reads':True,'slot_releases':'all four after current MMAC/handoff, before scale arithmetic; compiler must wait for LDS completion','original_math_statements_and_order_preserved':True,'across_iteration_prefetch':False,'extra_LDS_allocation':0})
    assert len(edits)==2
    for a,b,s in reversed(edits):text=text[:a]+s+text[b:]
    return text,proof

if __name__=='__main__':
    import sys
    p=Path(sys.argv[1]);r=Path(sys.argv[2]);source=p.read_text();control,_=normalize_mls(source);candidate,proof=transform(source)
    (r/'group-read-control.ttgir').write_text(control);(r/'group-read-late-release.ttgir').write_text(candidate)
    (r/'group-read-proof.json').write_text(json.dumps({'source':str(p),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'control_sha256':hashlib.sha256(control.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'consumers':proof},indent=2)+'\n');print('Prepared two consumers; batch all four reads, retain ready and safe reuse notifications')
