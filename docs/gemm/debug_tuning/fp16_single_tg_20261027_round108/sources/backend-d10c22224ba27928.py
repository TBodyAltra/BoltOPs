"""Private OneA pipeline: independent B K64 ownership within a BK128 parent.
A remains one full-K slot; B0 and B1 have distinct empty/ready barriers. No MMAC
turns are accepted here, leaving abarrier 4/5 exclusively owned by B1.
"""
import pathlib,re,json,hashlib,tempfile,os,collections
SSA=r'%[\w.]+'
def constants(text):
 d={m[1]:int(m[2]) for m in re.finditer('('+SSA+r') = arith.constant (\d+) : i32',text)}
 d.update({m[1]:int(m[2]) for m in re.finditer('('+SSA+r') = llvm.mlir.constant\((\d+) : i32\)',text)})
 return d

def transform(source):
 assert '__sbs_' not in source and 'hcu.consumer_pingpong' not in source
 marks=list(re.finditer(r'^    partition([012])\(',source,re.M));assert [m[1] for m in marks]==['0','1','2']
 pre=source[:marks[0].start()];producer=source[marks[0].start():marks[1].start()];consumers=[source[marks[1].start():marks[2].start()],source[marks[2].start():]]
 initvals=constants(pre)
 inits=[(initvals[m[1]],int(m[2])) for m in re.finditer(r'rocdl.hcu.s.abarrier.init ('+SSA+r'), (\d+)',pre)]
 assert sorted(inits)==[(0,8),(1,4),(2,8),(3,4)],inits
 # Two new barriers are initialized by exactly the same pre-partition waves as
 # the existing four. Match the old seeded-empty operation, including arrive2.
 seed=re.search(r'^(\s*)('+SSA+r') = rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 2( loc\([^\n]+\))$',pre,re.M)
 assert seed and initvals[seed[3]]==0
 loc=seed[4];ind=seed[1];newinit=f'{ind}%__sbs_init4 = llvm.mlir.constant(4 : i32) : i32{loc}\n{ind}%__sbs_init5 = llvm.mlir.constant(5 : i32) : i32{loc}\n{ind}rocdl.hcu.s.abarrier.init %__sbs_init4, 8{loc}\n{ind}%__sbs_seed = rocdl.hcu.s.abarrier.arrive i32, %__sbs_init4, 2{loc}\n{ind}rocdl.hcu.s.abarrier.init %__sbs_init5, 4{loc}\n'
 anchor=next(l for l in pre.splitlines(True) if '    ttg.warp_specialize(' in l);pre=pre.replace(anchor,newinit+anchor)
 ls=producer.splitlines(True);v=constants(producer);loads=[i for i,l in enumerate(ls) if ' = amdg.matrix_load_to_local ' in l and 'opIdx = 1,' in l];assert len(loads)==2 and all('tensorShape = [64, 256]' in ls[i] and 'warpsPerCTA = [1, 4]' in ls[i] for i in loads)
 waits=[];arr=[];seq=[]
 for i,l in enumerate(ls):
  m=re.search(r'rocdl.hcu.s.abarrier.try.wait i32, ('+SSA+r'), ('+SSA+r')',l)
  if m:waits.append((i,v[m[1]],m[2]))
  m=re.search(r'rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 1',l)
  if m:arr.append((i,v[m[1]]))
  m=re.search(r'rocdl.hcu.s.abarrier.seq ('+SSA+r')',l)
  if m:seq.append((i,v[m[1]]))
 assert [x[1] for x in waits]==[0,2] and [x[1] for x in arr]==[1,3] and [x[1] for x in seq]==[1,3]
 phase=waits[1][2];ready=ls[arr[1][0]];indent=re.match(r'\s*',ready)[0];loc=re.search(r'( loc\([^\n]+\))\n$',ready)[1]
 # Existing B0 wait/seq stays before first copy; readiness now follows copy0.
 # The B1 copy cannot overwrite its own previous-generation slot until wait4.
 newready=re.sub(r'^\s*'+SSA+' = ',indent+'%__sbs_ready0 = ',ready)
 insertion={loads[0]:newready,loads[1]:''}
 before1=f'{indent}%__sbs_wait1 = rocdl.hcu.s.abarrier.try.wait i32, %__sbs_p4, {phase}{loc}\n{indent}rocdl.hcu.s.abarrier.seq %__sbs_p5{loc}\n'
 after1=f'{indent}%__sbs_ready1 = rocdl.hcu.s.abarrier.arrive i32, %__sbs_p5, 1{loc}\n'
 out=[]
 for i,l in enumerate(ls):
  if i==arr[1][0]:continue
  if i==loads[1]:out.append(before1)
  out.append(l)
  if i==loads[0]:out.append(newready)
  if i==loads[1]:out.append(after1)
 producer=''.join(out);head=producer.index('\n')+1;producer=producer[:head]+f'      %__sbs_p4 = arith.constant 4 : i32{loc}\n      %__sbs_p5 = arith.constant 5 : i32{loc}\n'+producer[head:]
 records=[];newcons=[]
 for role,part in enumerate(consumers,1):
  ls=part.splitlines(True);v=constants(part);dots=[i for i,l in enumerate(ls) if ' = tt.dot ' in l];assert len(dots)==2
  loads=[i for i,l in enumerate(ls) if ' = ttg.local_load ' in l and 'opIdx = 1,' in l];assert len(loads)==2 and loads[0]<dots[0]<loads[1]<dots[1]
  waits=[];releases=[]
  for i,l in enumerate(ls):
   m=re.search(r'rocdl.hcu.s.abarrier.try.wait i32, ('+SSA+r'), ('+SSA+r')',l)
   if m:waits.append((i,v[m[1]],m[2]))
   m=re.search(r'rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 1',l)
   if m:releases.append((i,v[m[1]]))
  assert [x[1] for x in waits]==[1,3] and [x[1] for x in releases]==[0,2]
  assert releases[0][0]<waits[1][0]<loads[0] and loads[1]<releases[1][0]<dots[1]
  phase=waits[1][2];oldrelease=ls[releases[1][0]];indent=re.match(r'\s*',oldrelease)[0];loc=re.search(r'( loc\([^\n]+\))\n$',oldrelease)[1]
  # Keep old B0 release name (unused), introduce B1 release with barrier4.
  oldlhs=re.match(r'\s*('+SSA+r') = ',oldrelease)[1];assert len(re.findall(re.escape(oldlhs)+r'(?![\w.])',part))==1
  p4=f'%__sbs_c4_{role}';p5=f'%__sbs_c5_{role}'
  wait1=f'{indent}%__sbs_b1wait_{role} = rocdl.hcu.s.abarrier.try.wait i32, {p5}, {phase}{loc}\n'
  release1=f'{indent}%__sbs_b1free_{role} = rocdl.hcu.s.abarrier.arrive i32, {p4}, 1{loc}\n'
  drain=f'{indent}amdg.memory_counter_wait ds(0){loc}\n'
  out=[]
  for i,l in enumerate(ls):
   if i==releases[1][0]:out.extend([drain,release1]);continue
   if i==loads[1]:out.append(wait1)
   if i==releases[0][0]:out.append(drain)
   out.append(l)
   if i==loads[0]:out.extend([drain,oldrelease])
  result=''.join(out);head=result.index('\n')+1;result=result[:head]+f'      {p4} = arith.constant 4 : i32{loc}\n      {p5} = arith.constant 5 : i32{loc}\n'+result[head:]
  inv=next(l for l in result.splitlines(True) if 'rocdl.hcu.s.abarrier.inv' in l);result=result.replace(inv,f'      rocdl.hcu.s.abarrier.inv {p5}{loc}\n      rocdl.hcu.s.abarrier.inv {p4}{loc}\n'+inv,1)
  assert [l.strip() for l in part.splitlines() if ' = tt.dot ' in l]==[l.strip() for l in result.splitlines() if ' = tt.dot ' in l]
  assert [l.strip() for l in part.splitlines() if ' = ttg.local_load ' in l]==[l.strip() for l in result.splitlines() if ' = ttg.local_load ' in l]
  assert [l.strip() for l in part.splitlines() if 'arith.xori' in l or 'scf.yield' in l]==[l.strip() for l in result.splitlines() if 'arith.xori' in l or 'scf.yield' in l]
  newcons.append(result);records.append(dict(consumer=role,A_full_capture_then_release=True,B0_wait_read_drain_release_before_dot0=True,B1_wait_read_drain_release_before_dot1=True,phase_from_existing_B_generation=phase,explicit_ds_drains=3))
 output=pre+producer+''.join(newcons)
 for marker in [' = amdg.matrix_load_to_local ','ttg.local_alloc : ()',' = tt.dot ',' = ttg.local_load ']:assert [l.strip() for l in source.splitlines() if marker in l]==[l.strip() for l in output.splitlines() if marker in l]
 return output,dict(passed=True,split_slot_ring_implemented=True,operand_barriers={'A':{'empty':0,'ready':1},'B0':{'empty':2,'ready':3},'B1':{'empty':4,'ready':5}},empty_counts=8,ready_counts=4,producer_waves=4,consumer_waves=[4,4],parent_allocation_unchanged=True,global_transfer_and_read_mapping_unchanged=True,math_unchanged=True,loop_generation_and_phase_unchanged=True,consumers=records,source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(output.encode()).hexdigest())

def rewrite_module(module):
 from triton._C.libtriton import ir
 from .partial_k_local import normalize_mls
 text,proof=transform(str(module));text,n=normalize_mls(text);assert n==2
 dump=os.environ.get('TRITON_SPLIT_B_SLOTS_DUMP_DIR');d=pathlib.Path(dump) if dump else None
 if d:d.mkdir(parents=True,exist_ok=True);(d/'before.ttgir').write_text(str(module));(d/'after.ttgir').write_text(text);(d/'proof.json').write_text(json.dumps(dict(proof,module_verified=False),indent=2)+'\n')
 with tempfile.TemporaryDirectory(prefix='pmd-split-b-slots-') as t:
  p=pathlib.Path(t)/'slots.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
 proof['module_verified']=True
 if d:(d/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
 return new,proof
