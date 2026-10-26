"""Capture/release the producer's first operand before waiting for its second.
Requires separate A/B data barriers and full-K direct-view prefetch. Moves only
existing SSA operations; producer, local allocation and FP32 math are unchanged.
"""
import pathlib,re,json,hashlib,tempfile,os,collections
SSA=r'%[\w.]+'
def definitions(lines):
 d={}
 for i,line in enumerate(lines):
  m=re.match(r'\s*('+SSA+r') = (.*)',line)
  if m:assert m[1] not in d;d[m[1]]=(i,m[2])
 return d
def transform(source):
 assert '__mt_' not in source and 'hcu.consumer_pingpong' not in source
 markers=list(re.finditer(r'^    partition([012])\(',source,re.M));assert [m[1] for m in markers]==['0','1','2']
 prefix=source[:markers[0].start()];producer=source[markers[0].start():markers[1].start()]
 const={m[1]:int(m[2]) for m in re.finditer(r'('+SSA+r') = arith.constant (\d+) : i32',prefix)}
 for m in re.finditer(r'('+SSA+r') = llvm.mlir.constant\((\d+) : i32\)',prefix):const[m[1]]=int(m[2])
 init=re.findall(r'rocdl.hcu.s.abarrier.init ('+SSA+r'), (\d+)',prefix)
 assert sorted((const[v],int(n)) for v,n in init)==[(0,8),(1,4),(2,8),(3,4)]
 ml=[int(re.search(r'opIdx = ([01])',l)[1]) for l in producer.splitlines() if 'amdg.matrix_load_to_local ' in l];assert sorted(ml)==[0,1]
 first=ml[0];ready=1+2*first;empty=ready-1;second=1-first
 pieces=[source[:markers[1].start()]];proof=[]
 for role,part in [(1,source[markers[1].start():markers[2].start()]),(2,source[markers[2].start():])]:
  lines=part.splitlines(True);defs=definitions(lines);vals=dict(const)
  for v,(i,rhs) in defs.items():
   m=re.fullmatch(r'arith.constant (\d+) : i32(?: loc\([^\n]+\))?',rhs)
   if m:vals[v]=int(m[1])
  waits={};releases={};reads={0:[],1:[]}
  for i,l in enumerate(lines):
   m=re.search(r'rocdl.hcu.s.abarrier.try.wait i32, ('+SSA+r'),',l)
   if m:assert vals[m[1]] not in waits;waits[vals[m[1]]]=i
   m=re.search(r'rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 1',l)
   if m:assert vals[m[1]] not in releases;releases[vals[m[1]]]=i
   if ' = ttg.local_load ' in l:
    op=int(re.search(r'dot_op<\{opIdx = ([01])',l)[1]);reads[op].append(i)
  assert sorted(waits)==[1,3] and sorted(releases)==[0,2] and [len(reads[o]) for o in [0,1]]==[2,2]
  dots=[i for i,l in enumerate(lines) if ' = tt.dot ' in l];assert len(dots)==2
  assert max(reads[0]+reads[1])<min(releases.values()) and max(releases.values())<dots[0]
  pivot=min(waits.values());move={waits[ready],releases[empty]}
  def closure(i):
   if i<pivot or i in move:return
   rhs=re.match(r'\s*'+SSA+r' = (.*)',lines[i])[1]
   assert rhs.startswith(('ttg.local_load ','ttg.memdesc_index ','ttg.memdesc_subslice ')),rhs
   move.add(i)
   for v in re.findall(SSA,rhs):
    if v in defs and defs[v][0]>=pivot:closure(defs[v][0])
  for i in reads[first]:closure(i)
  assert all(i<dots[0] for i in move) and waits[1+2*second] not in move
  for i in move:
   lhs=re.match(r'\s*('+SSA+r') = ',lines[i])[1]
   uses=[j for j,l in enumerate(lines) if j!=i and lhs in re.findall(SSA,l)]
   assert all(j in move or j>=dots[0] for j in uses),(lhs,uses)
  ordered=sorted(move)
  result=''.join((''.join(lines[j] for j in ordered) if i==pivot else '')+(l if i not in move else '') for i,l in enumerate(lines))
  rl=result.splitlines();fwait=next(i for i,l in enumerate(rl) if lines[waits[ready]].strip()==l.strip());swait=next(i for i,l in enumerate(rl) if lines[waits[1+2*second]].strip()==l.strip());release=next(i for i,l in enumerate(rl) if lines[releases[empty]].strip()==l.strip());rfirst=[i for i,l in enumerate(rl) if any(l.strip()==lines[j].strip() for j in reads[first])];rsecond=[i for i,l in enumerate(rl) if any(l.strip()==lines[j].strip() for j in reads[second])]
  assert fwait<min(rfirst)<=max(rfirst)<release<swait<min(rsecond)
  assert [l.strip() for l in part.splitlines() if ' = tt.dot ' in l]==[l.strip() for l in rl if ' = tt.dot ' in l]
  assert collections.Counter(l.strip() for l in part.splitlines())==collections.Counter(l.strip() for l in rl)
  # Every newly early operand's SSA dependency is either already defined or moved before it.
  for i in ordered:
   rhs=defs[re.match(r'\s*('+SSA+r') = ',lines[i])[1]][1]
   for v in re.findall(SSA,rhs):
    if v in defs:assert defs[v][0]<pivot or (defs[v][0] in move and defs[v][0]<i),(i,v,defs[v])
  pieces.append(result);proof.append(dict(consumer=role,first_operand='AB'[first],first_ready_id=ready,first_empty_id=empty,moved_nodes=len(move),both_K64_views_captured_before_empty=True,first_empty_before_second_ready_wait=True,math_order_unchanged=True))
 output=''.join(pieces)
 assert output[:output.index('    partition1(')]==source[:source.index('    partition1(')]
 return output,dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(output.encode()).hexdigest(),producer_unchanged=True,no_new_memory_or_math_ops=True,first_operand='AB'[first],consumers=proof)
def rewrite_module(module):
 from triton._C.libtriton import ir
 from .partial_k_local import normalize_mls
 text,proof=transform(str(module));text,n=normalize_mls(text);assert n==2
 with tempfile.TemporaryDirectory(prefix='pmd-first-operand-early-') as t:
  p=pathlib.Path(t)/'early.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
 dump=os.environ.get('TRITON_FIRST_OPERAND_EARLY_DUMP_DIR')
 if dump:
  p=pathlib.Path(dump);p.mkdir(parents=True,exist_ok=True);(p/'before.ttgir').write_text(str(module));(p/'after.ttgir').write_text(text);(p/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
 return new,proof
