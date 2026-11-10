"""R001-only, checked gfx946 assembly experiment: two disjoint 32-KiB LDS banks.

Preserves the compiled TN GEMM arithmetic, descriptor updates, output store,
MLS VMEM waits and LDS-to-VGPR waits. Not a general compiler transformation.
"""
import re, hashlib


def rewrite(asm, mode):
    assert mode in ('pingpong_toggle', 'pingpong_unrolled', 'pingpong_unrolled_prefetch_first', 'pingpong_unrolled_keep_wait')
    assert '.amdhsa_next_free_sgpr 57' in asm
    assert '.amdhsa_next_free_vgpr 133' in asm
    head, tail = asm.split('.LBB0_1:', 1)
    body, drain = tail.split('\ts_cbranch_scc1 .LBB0_1\n', 1)
    assert body.count('matrix_load_32x16_b16') == 32
    assert body.count('ds_read_matrix_trans_format') == 32
    assert body.count('v_mmac_f32_16x16x16_bf16') == 32
    assert '\ts_mov_b32 s16, -4\n' in head
    assert '\ts_add_i32 s16, s16, 4\n' in body
    assert '\ts_cmp_lt_u32 s16, 8\n' in body
    regs = [int(x) for x in re.findall(r'^\s*matrix_load_\S+\s+s\[0:3\]\s+s(\d+)\b', body, re.M)]
    assert len(regs) == len(set(regs)) == 32
    # Every address is a constant in bank0 before the steady-state loop.
    for r in regs:
        assert re.search(r'^\s*s_(?:mov_b32|add_i32) s'+str(r)+r', (?:0, )?(?:0|0x[0-9a-f]+)$',head,re.M)
    body, waits = re.subn(r'^\ts_waitcnt lgkmcnt\(0\)\n','',body,flags=re.M)
    assert waits == 1
    # MLS contributes to VM_CNT. LDS reads contribute to LGKM_CNT. Waiting
    # for LDS reads here leaves the next bank's global prefetch outstanding.
    mmac = body.index('\tv_mmac_')
    body = body[:mmac] + '\ts_waitcnt lgkmcnt(0)\n' + body[mmac:]
    aliases = {r:57+i for i,r in enumerate(regs)}
    def map_mls(text):
        return re.sub(r'^(\s*matrix_load_\S+\s+s\[0:3\]\s+)s(\d+)\b',
                      lambda m:m[1]+'s'+str(aliases[int(m[2])]),text,flags=re.M)
    def map_ds(text, mapping):
        return re.sub(r'^(\s*ds_read_matrix\S+\s+v\[\d+:\d+\],\s*)s(\d+)\b',
                      lambda m:m[1]+'s'+str(mapping.get(int(m[2]),int(m[2]))),text,flags=re.M)
    if mode == 'pingpong_toggle':
        # Each DS read has already consumed its scalar address operand.
        # Toggle producer addresses only after issuing all current-bank reads.
        pos = list(re.finditer(r'^\tds_read_matrix[^\n]*\n',body,re.M))[-1].end()
        toggle='\t; Explicit bank swap: 0..32767 <-> 32768..65535\n'+''.join(
            f'\ts_xor_b32 s{r}, s{r}, 0x8000\n' for r in regs)
        body=body[:pos]+toggle+body[pos:]
        assert '\ts_mov_b32 s0, 0\n' in drain
        drain=drain.replace('\ts_mov_b32 s0, 0\n','\ts_mov_b32 s0, s13\n',1)
        out=head+'.LBB0_1:'+body+'\ts_cbranch_scc1 .LBB0_1\n'+drain
        sgprs=57
    else:
        # K=2048, BK=128, unroll=4: prologue + exactly 3 refill iterations.
        # Explicitly unroll those iterations and keep fixed addresses for both
        # banks, avoiding 32 address XORs on every refill.
        head=head.replace('\ts_mov_b32 s16, -4\n','')
        init='\t; Fixed second-bank addresses for explicit three-iteration pipeline\n'+''.join(
            f'\ts_xor_b32 s{aliases[r]}, s{r}, 0x8000\n' for r in regs)
        body=body.replace('\ts_add_i32 s16, s16, 4\n','').replace('\ts_cmp_lt_u32 s16, 8\n','')
        if mode == 'pingpong_unrolled_keep_wait':
            body = body.replace('\ts_waitcnt lgkmcnt(0)\n','')
            pos = body.index('\tmatrix_load_')
            body = body[:pos]+'\ts_waitcnt lgkmcnt(0)\n'+body[pos:]
        if mode == 'pingpong_unrolled_prefetch_first':
            first_ds = body.index('\tds_read_matrix')
            last_ds = list(re.finditer(r'^\tds_read_matrix[^\n]*\n',body,re.M))[-1].end()
            read_wait = body.index('\ts_waitcnt lgkmcnt(0)\n')
            assert first_ds < last_ds < read_wait
            body = body[:first_ds]+body[last_ds:read_wait]+body[first_ds:last_ds]+body[read_wait:]
        phases=[]
        for phase in range(3):
            text=map_mls(body) if phase%2==0 else map_ds(body,aliases)
            phases.append(f'\n\t; batch {phase}: read bank{phase%2}, prefetch bank{1-phase%2}\n'+text)
        # The fourth and final batch is in bank1. No further MLS is issued.
        drain=map_ds(drain,aliases|{0:aliases[13]})
        out=head+init+''.join(phases)+drain
        sgprs=89
        out=out.replace('.amdhsa_next_free_sgpr 57','.amdhsa_next_free_sgpr 89')
        out=re.sub(r'(\.sgpr_count:\s*)59\b',r'\g<1>91',out)
    proof=dict(mode=mode,shape=[64,64,2048],block=[16,16,128],unroll=4,
               bank_bytes=32768,shared_bytes=65536,lds_banks=[[0,32768],[32768,65536]],
               next_free_sgpr=sgprs,next_MLS_before_current_DS=(mode=='pingpong_unrolled_prefetch_first'),VGPR=133,steady_state_DS_read_complete_wait_after_next_MLS=(mode!='pingpong_unrolled_keep_wait'),
               MLS_readiness_wait_retained=True,arithmetic_and_output_store_preserved=True,
               address_registers=regs,parent_asm_sha256=hashlib.sha256(asm.encode()).hexdigest(),
               patched_asm_sha256=hashlib.sha256(out.encode()).hexdigest())
    return out,proof
