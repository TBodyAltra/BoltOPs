"""Prefetch both direct K64 operand views before the first dot; no new copies.
Only qualified two-dot/OnePTwoC/early-release input is accepted. This is a
register-lifetime experiment, not a change to the full BK128 global transport.
"""
import pathlib,re,hashlib,json,tempfile,os
SSA=r'%[\w.]+'
def definitions(lines):
 d={}
 for i,line in enumerate(lines):
  m=re.match(r'\s*('+SSA+r'(?:,\s*'+SSA+r')*) = (.*)',line)
  if m:
   for v in re.findall(SSA,m[1]):assert v not in d;d[v]=(i,m[2])
 return d
def transform(source):
 assert '__mt_' not in source and 'hcu.consumer_pingpong' not in source
 markers=list(re.finditer(r'^    partition([012])\(',source,re.M));assert [m[1] for m in markers]==['0','1','2']
 assert source.count(' = tt.dot ')==4 and source.count(' = ttg.local_load ')==8 and not re.search(r'ttg.local_alloc\s+%',source)
 parts=[source[:markers[1].start()]];proof=[]
 for role,part in [(1,source[markers[1].start():markers[2].start()]),(2,source[markers[2].start():])]:
  lines=part.splitlines(True);defs=definitions(lines)
  dots=[i for i,l in enumerate(lines) if ' = tt.dot ' in l];loads=[i for i,l in enumerate(lines) if ' = ttg.local_load ' in l];releases=[i for i,l in enumerate(lines) if ' = rocdl.hcu.s.abarrier.arrive ' in l]
  assert len(dots)==2 and len(loads)==4 and len(releases)==2
  assert loads[0]<loads[1]<dots[0]<loads[2]<loads[3]<min(releases)<=max(releases)<dots[1]
  move=set(releases)
  for i in loads[2:]:
   m=re.search(r'ttg.local_load ('+SSA+r')',lines[i]);assert m;desc=m[1];di,rhs=defs[desc]
   assert rhs.startswith('ttg.memdesc_subslice ');root=re.match(r'ttg.memdesc_subslice ('+SSA+r')',rhs)[1]
   assert defs[root][0]<dots[0] and dots[0]<di<i<dots[1]
   move.update([di,i])
  assert len(move)==6
  for i in move:
   lhs=re.match(r'\s*('+SSA+r') = ',lines[i])[1]
   if i in releases:assert len(re.findall(re.escape(lhs)+r'(?![\w.])',part))==1
   else:
    uses=[j for j,l in enumerate(lines) if j!=i and lhs in re.findall(SSA,l)]
    assert all(j in move or j==dots[1] for j in uses),(lhs,uses)
  ordered=sorted(move);assert ordered[-1]<dots[1]
  result=''.join((''.join(lines[j] for j in ordered)+l) if i==dots[0] else l for i,l in enumerate(lines) if i not in move)
  rl=result.splitlines();ds=[i for i,l in enumerate(rl) if ' = tt.dot ' in l];ld=[i for i,l in enumerate(rl) if ' = ttg.local_load ' in l];ar=[i for i,l in enumerate(rl) if ' = rocdl.hcu.s.abarrier.arrive ' in l]
  assert len(ld)==4 and len(ar)==2 and max(ld)<min(ar)<=max(ar)<ds[0]<ds[1]
  for token in [' = tt.dot ',' = ttg.local_load ','ttg.memdesc_subslice ','rocdl.hcu.s.abarrier.try.wait ','arith.xori','scf.yield']:
   assert [l.strip() for l in part.splitlines() if token in l]==[l.strip() for l in result.splitlines() if token in l],token
  parts.append(result);proof.append(dict(consumer=role,moved_nodes=6,local_reads=4,all_reads_before_empty_release=True,empty_release_before_first_dot=True,math_accumulation_order_unchanged=True))
 output=''.join(parts)
 for token in ['amdg.matrix_load_to_local ','rocdl.hcu.s.abarrier.seq ','rocdl.hcu.s.abarrier.init ','rocdl.hcu.s.abarrier.inv ','ttg.local_alloc']:
  assert [l.strip() for l in source.splitlines() if token in l]==[l.strip() for l in output.splitlines() if token in l]
 return output,dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(output.encode()).hexdigest(),full_global_BK=128,operand_local_K_fragments=[64,64],no_new_allocations_or_memory_ops=True,producer_unchanged=True,consumers=proof)
def rewrite_module(module):
 from triton._C.libtriton import ir
 from .partial_k_local import normalize_mls
 text,proof=transform(str(module));text,n=normalize_mls(text);assert n==2
 with tempfile.TemporaryDirectory(prefix='pmd-full-k-prefetch-') as t:
  p=pathlib.Path(t)/'prefetch.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
 dump=os.environ.get('TRITON_FULL_K_PREFETCH_DUMP_DIR')
 if dump:
  p=pathlib.Path(dump);p.mkdir(parents=True,exist_ok=True);(p/'before.ttgir').write_text(str(module));(p/'after.ttgir').write_text(text);(p/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
 return new,proof
