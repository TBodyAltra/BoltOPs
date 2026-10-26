"""Private post-WASP MMAC turns on ordinary abarriers 4/5.
Data abarriers 0..3 and early release stay intact. All new invalidates live
behind the compiler's existing all-partition teardown rendezvous.
"""
import re,json,hashlib,tempfile,os
from pathlib import Path
SSA=r'%[\w.]+'
def transform(source,mode):
    assert type(mode) is int and mode in (1,2)
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
    assert data_init in [[(0,8),(1,4)],[(0,8),(1,4),(2,8),(3,4)]]
    data_ids=[z[0] for z in data_init]
    loc=init[0][2]
    seed=('    %__mt_ping = arith.constant 4 : i32'+loc+'\n'
          '    %__mt_pong = arith.constant 5 : i32'+loc+'\n'
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
        inp,out=(4,5) if role==1 else (5,4)
        head=part.index('\n')+1
        local=(f'      %__mt_in = arith.constant {inp} : i32{loc}\n'
               f'      %__mt_out = arith.constant {out} : i32{loc}\n'
               f'      %__mt_zero = arith.constant 0 : i32{loc}\n'
               f'      %__mt_one = arith.constant 1 : i32{loc}\n')
        updates={}
        for f,i in enumerate(dots):
            m=re.search(r'( loc\([^\n]+\))$',lines[i]);assert m;dl=m[1]
            before='';after=''
            if mode==2 or f==0:
                phase=f'%__mt_phase_{f}' if mode==1 else '%__mt_zero' if f==0 else '%__mt_one'
                if mode==1:before+=f'        {phase} = arith.andi {loop[1]}, %__mt_one : i32{dl}\n'
                before+=f'        %__mt_wait_{f} = rocdl.hcu.s.abarrier.try.wait i32, %__mt_in, {phase}{dl}\n        rocdl.sched.barrier 0{dl}\n'
            if mode==2 or f==1:
                after+=f'        rocdl.sched.barrier 0{dl}\n        %__mt_signal_{f} = rocdl.hcu.s.abarrier.arrive i32, %__mt_out, 1{dl}\n        rocdl.sched.barrier 0{dl}\n'
            updates[i]=before+lines[i]+after
        part=''.join(updates.get(i,l) for i,l in enumerate(lines))
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
    return result,dict(mode=mode,source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(result.encode()).hexdigest(),count=4,turn_ids=[4,5],seed=dict(id=4,per_producer_wave=1,producer_waves=4),data_barrier_ids=data_ids,data_and_math_unchanged=True,no_consumer_ebarrier=True,consumers=records)

def rewrite_module(module,mode):
    from triton._C.libtriton import ir
    from .partial_k_local import normalize_mls
    text,proof=transform(str(module),mode);text,count=normalize_mls(text);assert count==2
    with tempfile.TemporaryDirectory(prefix='pmd-mmac-turn-') as t:
        p=Path(t)/'turn.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
    dump=os.environ.get('TRITON_MMAC_TURN_DUMP_DIR')
    if dump:
        p=Path(dump);p.mkdir(parents=True,exist_ok=True);(p/'before.ttgir').write_text(str(module));(p/'after.ttgir').write_text(text);(p/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return new,proof
