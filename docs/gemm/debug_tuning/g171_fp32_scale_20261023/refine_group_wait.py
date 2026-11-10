"""Refine native waits only for the verified 16-matrix/10-scale LDS batch."""
import re

def regs(text):
    out=set()
    for a,b,c in re.findall(r'v\[(\d+):(\d+)\]|\bv(\d+)\b',text):
        out.update(range(int(a),int(b)+1) if a else [int(c)])
    return out

def transform(text):
    lines=text.splitlines(True);edits={};proof=[];start=0
    while True:
        mat=next((i for i in range(start,len(lines)) if lines[i].strip().startswith('ds_read_matrix')),None)
        if mat is None:break
        first=next(i for i in range(mat,len(lines)) if lines[i].strip().startswith('v_mmac'))
        matrix=[i for i in range(mat,first) if lines[i].strip().startswith('ds_read_matrix')]
        scales=[i for i in range(mat,first) if lines[i].strip().startswith('ds_read2_b32')]
        assert len(matrix)==16 and len(scales)==10 and max(matrix)<min(scales)
        waits=[i for i in range(mat,first) if lines[i].strip().startswith('s_waitcnt')]
        assert len(waits)==1 and lines[waits[0]].strip()=='s_waitcnt lgkmcnt(0)' and waits[0]>max(scales)
        release=next(i for i in range(first,len(lines)) if lines[i].strip().startswith('s_abarrier_arrive'))
        mmac=[i for i in range(first,release) if lines[i].strip().startswith('v_mmac')]
        assert len(mmac)==64
        sr=set().union(*(regs(lines[i].split(',',1)[0]) for i in scales));mr=set().union(*(regs(lines[i]) for i in mmac));assert not sr&mr,(sr&mr)
        for i in range(mat,release):
            op=lines[i].strip()
            assert not op.startswith(('s_load','s_buffer_load','buffer_load','ds_write','flat_','global_','s_branch','s_cbranch')),op
        for i in range(max(scales)+1,first):
            assert not regs(lines[i])&sr,lines[i]
        # Only this current batch contributes outstanding LDS: previous iteration
        # drains before releases; each exact batch issues ordered AB, then scales.
        outstanding=len(scales)
        edits[waits[0]]=lines[waits[0]].replace('lgkmcnt(0)',f'lgkmcnt({outstanding})')
        edits[release]='\ts_waitcnt lgkmcnt(0)\n'+lines[release]
        proof.append({'first_matrix_line':mat+1,'matrix_read_count':len(matrix),'scale_read_count':len(scales),'partial_wait_line':waits[0]+1,'partial_lgkmcnt':outstanding,'MMAC_first_line':first+1,'MMAC_last_line':mmac[-1]+1,'full_drain_before_first_slot_release_line':release+1,'scale_destination_VGPRs':sorted(sr),'MMAC_scale_VGPR_intersection':[],'MMAC_count':len(mmac),'pre_and_post_MMAC_handoffs':'unchanged; final block may have no post handoff','no_unaccounted_async_ops_or_branches':True})
        start=release+1
    assert len(proof)==3,len(proof)
    for i,s in edits.items():lines[i]=s
    changed=''.join(lines)
    # Every non-wait native instruction remains byte-for-byte in the same order.
    strip=lambda s:[l for l in s.splitlines() if not l.strip().startswith('s_waitcnt')]
    assert strip(changed)==strip(text)
    return changed,proof
