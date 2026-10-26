"""Full-register capture of one BK128 operand; stream the other as two K64 views.
Only reorders existing memory/SSA operations. Pair with a first MMAC-turn wait
before the streamed operand's ready wait, so its LDS reads can interleave MMAC.
"""
import pathlib,re,json,hashlib,collections,tempfile,os
SSA=r'%[\w.]+'
def transform(source,operand):
 from .prefetch_full_k import transform as full_prefetch
 from .first_operand_early import transform as capture_first
 assert type(operand) is int and operand in (1,2);first=operand-1;other=1-first
 from .data_barrier_map import derive
 mapping,mapping_proof=derive(source)
 full,full_proof=full_prefetch(source);ordered,early_proof=capture_first(full,operand=first)
 markers=list(re.finditer(r'^    partition([012])\(',ordered,re.M));assert [m[1] for m in markers]==['0','1','2']
 pieces=[ordered[:markers[1].start()]];records=[]
 for role,part in [(1,ordered[markers[1].start():markers[2].start()]),(2,ordered[markers[2].start():])]:
  lines=part.splitlines(True);defs={}
  for i,l in enumerate(lines):
   m=re.match(r'\s*('+SSA+r') = (.*)',l)
   if m:assert m[1] not in defs;defs[m[1]]=(i,m[2])
  dots=[i for i,l in enumerate(lines) if ' = tt.dot ' in l];assert len(dots)==2
  loads=[i for i,l in enumerate(lines) if ' = ttg.local_load ' in l and re.search(r'dot_op<\{opIdx = '+str(other)+r',',l)];assert len(loads)==2
  assert loads[1]<dots[0]
  desc=re.search(r'ttg.local_load ('+SSA+r')',lines[loads[1]])[1];di,rhs=defs[desc];assert rhs.startswith('ttg.memdesc_subslice ')
  root=re.match(r'ttg.memdesc_subslice ('+SSA+r')',rhs)[1];assert defs[root][0]<dots[0]
  vals={m[1]:int(m[2]) for m in re.finditer(r'('+SSA+r') = arith.constant (\d+) : i32',ordered[:markers[0].start()]+part)}
  for m in re.finditer(r'('+SSA+r') = llvm.mlir.constant\((\d+) : i32\)',ordered[:markers[0].start()]):vals[m[1]]=int(m[2])
  releases=[]
  for i,l in enumerate(lines):
   m=re.search(r'rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 1',l)
   if m and vals[m[1]]==mapping['AB'[other]]['empty']:releases.append(i)
  assert len(releases)==1 and loads[1]<releases[0]<dots[0]
  move={di,loads[1],releases[0]};assert len(move)==3
  for i in move:
   lhs=re.match(r'\s*('+SSA+r') = ',lines[i])[1];uses=[j for j,l in enumerate(lines) if j!=i and lhs in re.findall(SSA,l)]
   assert all(j in move or j==dots[1] for j in uses),(lhs,uses)
  result=''.join(l+(''.join(lines[j] for j in sorted(move)) if i==dots[0] else '') for i,l in enumerate(lines) if i not in move)
  out=result.splitlines();ds=[i for i,l in enumerate(out) if ' = tt.dot ' in l];ld=[i for i,l in enumerate(out) if ' = ttg.local_load ' in l and re.search(r'dot_op<\{opIdx = '+str(other)+r',',l)]
  assert ld[0]<ds[0]<ld[1]<ds[1]
  assert collections.Counter(l.strip() for l in lines)==collections.Counter(l.strip() for l in out)
  assert [l.strip() for l in lines if ' = tt.dot ' in l]==[l.strip() for l in out if ' = tt.dot ' in l]
  pieces.append(result);records.append(dict(consumer=role,prefetched_operand='AB'[first],streamed_operand='AB'[other],streamed_K_fragments=[64,64],moved_existing_nodes=3,other_second_fragment_after_first_dot=True,math_order_unchanged=True))
 output=''.join(pieces);assert output[:output.index('    partition1(')]==source[:source.index('    partition1(')]
 return output,dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(output.encode()).hexdigest(),producer_unchanged=True,no_new_allocations_or_memory_ops=True,full_global_BK=128,operand_barriers=mapping,barrier_map_proof=mapping_proof,prefetched_operand='AB'[first],streamed_operand='AB'[other],full_prefetch_intermediate=full_proof,early_capture_intermediate=early_proof,consumers=records)
def rewrite_module(module,operand):
 from triton._C.libtriton import ir
 from .partial_k_local import normalize_mls
 text,proof=transform(str(module),operand);text,n=normalize_mls(text);assert n==2
 with tempfile.TemporaryDirectory(prefix='pmd-one-operand-') as t:
  p=pathlib.Path(t)/'prefetch.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
 dump=os.environ.get('TRITON_ONE_OPERAND_PREFETCH_DUMP_DIR')
 if dump:
  p=pathlib.Path(dump);p.mkdir(parents=True,exist_ok=True);(p/'before.ttgir').write_text(str(module));(p/'after.ttgir').write_text(text);(p/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
 return new,proof
