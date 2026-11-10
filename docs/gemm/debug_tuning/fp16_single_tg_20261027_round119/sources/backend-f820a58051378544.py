"""Private MMAC turns on abarriers6/7 after independent B0/B1 ownership.
Data0..5 and full-parent copies/reads/math remain unchanged. wait_mask bits
move each K64 turn wait from before B-ready to immediately before its dot.
"""
import re,json,hashlib,tempfile,os
from pathlib import Path
SSA=r'%[\w.]+'
def transform(source,mode,wait_mask=0):
    assert type(wait_mask) is int and wait_mask in (0,1,2,3)
    assert not wait_mask or mode==2
    assert type(mode) is int and mode in (1,2)
    prefetch_operand=1
    assert '__mt_' not in source and 'hcu.consumer_pingpong' not in source
    markers=list(re.finditer(r'^    partition([012])\(',source,re.M))
    assert [m[1] for m in markers]==['0','1','2']
    assert source.count(' = tt.dot ')==4 and source.count(' = ttg.local_load ')==8
    assert not re.search(r'ttg.local_alloc\s+%',source)
    boundary=source.index('    ttg.warp_specialize(')
    prefix=source[:boundary]
    constants={m[1]:int(m[2]) for m in re.finditer(r'('+SSA+r') = (?:arith.constant (\d+) : i32|NOT_MATCH)',prefix)}
    for m in re.finditer(r'('+SSA+r') = llvm.mlir.constant\((\d+) : i32\)',prefix):constants[m[1]]=int(m[2])
    init=re.findall(r'^    rocdl.hcu.s.abarrier.init ('+SSA+r'), (\d+)( loc\([^\n]+\))$',prefix,re.M)
    data_init=sorted((constants[v],int(count)) for v,count,_ in init)
    assert data_init==[(0,8),(1,4),(2,8),(3,4),(4,8),(5,4)]
    data_ids=[z[0] for z in data_init]
    loc=init[0][2]
    seed=('    %__mt_ping = arith.constant 6 : i32'+loc+'\n'
          '    %__mt_pong = arith.constant 7 : i32'+loc+'\n'
          '    rocdl.hcu.s.abarrier.init %__mt_ping, 4'+loc+'\n'
          '    rocdl.hcu.s.abarrier.init %__mt_pong, 4'+loc+'\n'
          '    %__mt_seed = rocdl.hcu.s.abarrier.arrive i32, %__mt_ping, 1'+loc+'\n')
    parts=[source[:markers[1].start()]];records=[]
    for role,part in [(1,source[markers[1].start():markers[2].start()]),(2,source[markers[2].start():])]:
        lines=part.splitlines(True)
        dots=[i for i,l in enumerate(lines) if ' = tt.dot ' in l];assert len(dots)==2
        assert part.count(' = ttg.local_load ')==4 and part.count('rocdl.hcu.s.abarrier.arrive ')==len(data_ids)//2
        loop=re.search(r'scf.for ('+SSA+r') = ('+SSA+r') to ('+SSA+r') step ('+SSA+r')',part);assert loop
        vals={m[1]:int(m[2]) for m in re.finditer(r'('+SSA+r') = arith.constant (\d+) : i32',part)}
        assert vals[loop[2]]==0 and vals[loop[4]]==1
        inp,out=(6,7) if role==1 else (7,6)
        head=part.index('\n')+1
        local=(f'      %__mt_in = arith.constant {inp} : i32{loc}\n'
               f'      %__mt_out = arith.constant {out} : i32{loc}\n'
               f'      %__mt_zero = arith.constant 0 : i32{loc}\n'
               f'      %__mt_one = arith.constant 1 : i32{loc}\n')
        before_at={};after_at={}
        ids=dict(constants);ids.update(vals);stream_waits=[]
        for ready_id in [3,5]:
            waits=[]
            for j,line in enumerate(lines):
                m=re.search(r'rocdl.hcu.s.abarrier.try.wait i32, ('+SSA+r'),',line)
                if m and ids[m[1]]==ready_id:waits.append(j)
            assert len(waits)==1;stream_waits.append(waits[0])
        assert stream_waits[0]<dots[0]<stream_waits[1]<dots[1]

        for f,i in enumerate(dots):
            m=re.search(r'( loc\([^\n]+\))$',lines[i]);assert m;dl=m[1]
            before='';after=''
            if mode==2 or f==0:
                phase=f'%__mt_phase_{f}' if mode==1 else '%__mt_zero' if f==0 else '%__mt_one'
                if mode==1:before+=f'        {phase} = arith.andi {loop[1]}, %__mt_one : i32{dl}\n'
                before+=f'        %__mt_wait_{f} = rocdl.hcu.s.abarrier.try.wait i32, %__mt_in, {phase}{dl}\n        rocdl.sched.barrier 0{dl}\n'
            if mode==2 or f==1:
                after+=f'        rocdl.sched.barrier 0{dl}\n        %__mt_signal_{f} = rocdl.hcu.s.abarrier.arrive i32, %__mt_out, 1{dl}\n        rocdl.sched.barrier 0{dl}\n'
            target=i if wait_mask & (1<<f) else stream_waits[f] if mode==2 or f==0 else i
            before_at[target]=before_at.get(target,'')+before
            after_at[i]=after_at.get(i,'')+after
        part=''.join(before_at.get(i,'')+l+after_at.get(i,'') for i,l in enumerate(lines))
        head=part.index('\n')+1;part=part[:head]+local+part[head:]
        firstinv=re.search(r'^      rocdl.hcu.s.abarrier.inv ',part,re.M);assert firstinv
        # Native HCU WarpSpecializeToLLVM inserts its all12-wave teardown join
        # before the earliest invalidate. No consumer may destroy another's turn.
        teardown=f'      rocdl.hcu.s.abarrier.inv %__mt_in{loc}\n      rocdl.hcu.s.abarrier.inv %__mt_out{loc}\n'
        part=part[:firstinv.start()]+teardown+part[firstinv.start():]
        parts.append(part);records.append(dict(consumer=role,in_id=inp,out_id=out,waits_per_bk=mode,arrivals_per_bk=mode,phase='iteration & 1' if mode==1 else 'fragment 0/1; twice per BK',teardown='existing HCU all-partition join before earliest invalidate'))
    result=''.join(parts);result=result[:boundary]+seed+result[boundary:]
    for token in [' = tt.dot ',' = ttg.local_load ','ttg.memdesc_subslice ','amdg.matrix_load_to_local ','rocdl.hcu.s.abarrier.seq ','arith.xori','scf.yield']:
        assert [l.strip() for l in source.splitlines() if token in l]==[l.strip() for l in result.splitlines() if token in l],token
    assert result.count('rocdl.hcu.s.abarrier.init ')==len(data_init)+2
    assert result.count('rocdl.hcu.s.abarrier.inv ')==source.count('rocdl.hcu.s.abarrier.inv ')+4
    assert result.count(' = rocdl.hcu.s.abarrier.arrive ')==source.count(' = rocdl.hcu.s.abarrier.arrive ')+1+2*mode
    return result,dict(mode=mode,prefetch_operand=prefetch_operand,wait_mask=wait_mask,first_turn_wait_before_streamed_ready=not bool(wait_mask&1),fragment_turn_wait_before_its_B_ready=(mode==2 and wait_mask==0),late_turn_wait_before_dot_mask=wait_mask,source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(result.encode()).hexdigest(),count=4,turn_ids=[6,7],seed=dict(id=6,per_producer_wave=1,producer_waves=4),data_barrier_ids=data_ids,data_and_math_unchanged=True,no_consumer_ebarrier=True,consumers=records)

def rewrite_module(module,mode,wait_mask=0):
    from triton._C.libtriton import ir
    from .partial_k_local import normalize_mls
    text,proof=transform(str(module),mode,wait_mask);text,count=normalize_mls(text);assert count==2
    with tempfile.TemporaryDirectory(prefix='pmd-slot-mmac-turn-') as t:
        p=Path(t)/'turn.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
    dump=os.environ.get('TRITON_SLOT_MMAC_TURN_DUMP_DIR')
    if dump:
        p=Path(dump);p.mkdir(parents=True,exist_ok=True);(p/'before.ttgir').write_text(str(module));(p/'after.ttgir').write_text(text);(p/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return new,proof
