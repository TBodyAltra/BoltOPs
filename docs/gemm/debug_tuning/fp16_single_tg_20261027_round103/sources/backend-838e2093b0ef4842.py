"""Experimental two K64 global copies into one BK128 parent.
Original coupled B-ready/empty ownership remains; this is not a split-slot ring.
Native qualification requires prior full-parent address equivalence and CPU ISA gates.
"""
import pathlib,re,json,hashlib,tempfile,os
SSA=r'%[\w.]+'
def transform(source):
 from .data_barrier_map import derive
 mapping,binding=derive(source)
 assert mapping=={'A':{'empty':0,'ready':1},'B':{'empty':2,'ready':3}},mapping
 marks=list(re.finditer(r'^    partition([012])\(',source,re.M));assert [m[1] for m in marks]==['0','1','2']
 producer=source[marks[0].start():marks[1].start()];lines=source.splitlines(True)
 indices=[i for i,l in enumerate(lines) if ' = amdg.matrix_load_to_local ' in l and 'opIdx = 1,' in l];assert len(indices)==1
 i=indices[0];line=lines[i].rstrip('\n')
 pat=r'^(\s*)('+SSA+r') = amdg.matrix_load_to_local ('+SSA+r'), ('+SSA+r'), shape = (\[[^\]]+\]), strides = (\[[^\]]+\]), tensorShape = \[128,\s*(\d+)\], indices = \[('+SSA+r'), ('+SSA+r')\] (\{MlsEncoding = .*\}) : <f16> -> <(.*?)>( loc\(.*\))$'
 m=re.match(pat,line);assert m,line
 indent,lhs,ptr,dest,shape,strides,n,koff,noff,attrs,payload,loc=m.groups();n=int(n)
 # Four producer waves cover distinct N tiles in each K64 transfer. Keeping
 # the full-B [2,2] layout would duplicate each K64 tile across two K waves.
 assert n==256 and ('warpsPerCTA = [2, 2]' in attrs or 'warpsPerCTA = [1, 4]' in attrs)
 attrs=attrs.replace('warpsPerCTA = [2, 2]','warpsPerCTA = [1, 4]')
 assert len(re.findall(re.escape(lhs)+r'(?![\w.])',source))==1
 assert '__sbg_' not in source
 parts=[p.strip() for p in payload.split(',')];assert parts[0]==f'128x{n}xf16' and parts[-1]=='mutable' and len(parts)==4,parts
 vals={m[1]:int(m[2]) for m in re.finditer('('+SSA+r') = arith.constant (\d+) : i32',producer)}
 mul=re.search(re.escape(koff)+r' = arith.muli '+SSA+r', ('+SSA+r') : i32',producer);assert mul and vals[mul[1]]==128
 parent='!ttg.memdesc<'+', '.join(parts)+'>';view_parts=[f'64x{n}xf16',*parts[1:],f'128x{n}'];view_payload=', '.join(view_parts);view='!ttg.memdesc<'+view_payload+'>'
 out=[f'{indent}%__sbg_c64 = arith.constant 64 : i32{loc}\n',f'{indent}%__sbg_k1 = arith.addi {koff}, %__sbg_c64 : i32{loc}\n']
 for slot,offset,index in [(0,0,koff),(1,64,'%__sbg_k1')]:
  out.append(f'{indent}%__sbg_view{slot} = ttg.memdesc_subslice {dest}[{offset}, 0] : {parent} -> {view}{loc}\n')
  out.append(f'{indent}%__sbg_copy{slot} = amdg.matrix_load_to_local {ptr}, %__sbg_view{slot}, shape = {shape}, strides = {strides}, tensorShape = [64, {n}], indices = [{index}, {noff}] {attrs} : <f16> -> <{view_payload}>{loc}\n')
 result=''.join(lines[:i]+out+lines[i+1:]);assert result.count(' = amdg.matrix_load_to_local ')==source.count(' = amdg.matrix_load_to_local ')+1==3
 for marker in ['rocdl.hcu.s.abarrier.', 'ttg.local_alloc : ()',' = tt.dot ','scf.yield']:
  assert [l.strip() for l in source.splitlines() if marker in l]==[l.strip() for l in result.splitlines() if marker in l]
 assert result[:marks[0].start()]==source[:marks[0].start()]
 assert result[result.index('    partition1('):]==source[source.index('    partition1('):]
 return result,dict(coupled_copy_subview_probe=True,split_slot_pipeline_NOT_implemented=True,source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(result.encode()).hexdigest(),parent_shape=[128,n],global_transfer_shapes=[[64,n],[64,n]],K_offsets=[0,64],global_loop_step=128,producer_warp_layout=[1,4],logical_parent_layout_preserved=True,parent_allocation_unchanged=True,total_B_bytes_per_iteration=128*n*2,consumer_and_math_unchanged=True,original_barriers_unchanged=True,original_B_ready_after_both_copies=True,operand_barriers=mapping,producer_binding_proof=binding)
def rewrite_module(module):
 from triton._C.libtriton import ir
 from .partial_k_local import normalize_mls
 text,proof=transform(str(module));text,n=normalize_mls(text);assert n==2
 dump=os.environ.get('TRITON_SPLIT_B_GLOBAL_PROBE_DUMP_DIR');dumper=pathlib.Path(dump) if dump else None
 if dumper:
  dumper.mkdir(parents=True,exist_ok=True);(dumper/'before.ttgir').write_text(str(module));(dumper/'after.ttgir').write_text(text);(dumper/'proof.json').write_text(json.dumps(dict(proof,module_verified=False),indent=2)+'\n')
 with tempfile.TemporaryDirectory(prefix='pmd-split-b-global-probe-') as d:
  p=pathlib.Path(d)/'probe.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
 proof['module_verified']=True
 if dumper:(dumper/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
 return new,proof
