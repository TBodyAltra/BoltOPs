"""Merge synchronized single-slot A/B data barriers; preserve all data and math.
One producer seq1 covers both full BK128 MLS transfers; one ready1 publication
after B. Empty0 is released after both operands' final K64 reads complete.
"""
import re,json,hashlib,tempfile,os
from pathlib import Path
SSA=r'%[\w.]+'
def constants(s):
 d={m[1]:int(m[2]) for m in re.finditer(r'('+SSA+r') = arith.constant (\d+) : i32',s)}
 d.update({m[1]:int(m[2]) for m in re.finditer(r'('+SSA+r') = llvm.mlir.constant\((\d+) : i32\)',s)});return d
def used_once(value,part):return len(re.findall(re.escape(value)+r'(?![\w.])',part))==1
def transform(source):
 assert 'hcu.consumer_pingpong' not in source and '__mt_' not in source
 markers=list(re.finditer(r'^    partition([012])\(',source,re.M));assert [m[1] for m in markers]==['0','1','2']
 assert source.count(' = tt.dot ')==4 and source.count(' = ttg.local_load ')==8
 first=markers[0].start();pre=source[:first];c=constants(pre);lines=pre.splitlines(True);out=[];removed=[]
 for line in lines:
  m=re.search(r'rocdl.hcu.s.abarrier.init ('+SSA+r'), (\d+)',line)
  if m and c[m[1]] in (2,3):removed.append('init'+str(c[m[1]]));continue
  m=re.search(r'('+SSA+r') = rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 2',line)
  if m and c[m[2]]==2:assert used_once(m[1],pre);removed.append('Bemptyseed');continue
  out.append(line)
 assert sorted(removed)==['Bemptyseed','init2','init3']
 producer=source[first:markers[1].start()];pc=constants(producer);pl=producer.splitlines(True)
 waits=[(i,re.search(r'('+SSA+r') = rocdl.hcu.s.abarrier.try.wait i32, ('+SSA+r'),',l)) for i,l in enumerate(pl) if 'rocdl.hcu.s.abarrier.try.wait' in l]
 assert len(waits)==2 and sorted(pc[m[2]] for _,m in waits)==[0,2]
 arrivals=[(i,re.search(r'('+SSA+r') = rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 1',l)) for i,l in enumerate(pl) if 'rocdl.hcu.s.abarrier.arrive' in l]
 assert len(arrivals)==2 and sorted(pc[m[2]] for _,m in arrivals)==[1,3]
 ai,am=next((i,m) for i,m in arrivals if pc[m[2]]==1);bi,bm=next((i,m) for i,m in arrivals if pc[m[2]]==3)
 assert ai<bi and used_once(am[1],producer) and used_once(bm[1],producer)
 wi,wm=next((i,m) for i,m in waits if pc[m[2]]==2);assert used_once(wm[1],producer)
 seqs=[(i,re.search(r'rocdl.hcu.s.abarrier.seq ('+SSA+r')',l)) for i,l in enumerate(pl) if 'rocdl.hcu.s.abarrier.seq' in l];assert len(seqs)==2 and sorted(pc[m[1]] for _,m in seqs)==[1,3]
 si=next(i for i,m in seqs if pc[m[1]]==3)
 producer=''.join(pl[ai] if i==bi else l for i,l in enumerate(pl) if i not in (ai,wi,si));assert producer.count('amdg.matrix_load_to_local ')==2 and producer.count('rocdl.hcu.s.abarrier.seq ')==1
 parts=[''.join(out),producer];proof=[]
 for role,part in [(1,source[markers[1].start():markers[2].start()]),(2,source[markers[2].start():])]:
  cs=constants(part);kept=[];removed=[]
  for line in part.splitlines(True):
   m=re.search(r'('+SSA+r') = rocdl.hcu.s.abarrier.try.wait i32, ('+SSA+r'),',line)
   if m and cs[m[2]]==3:assert used_once(m[1],part);removed.append('Breadywait');continue
   m=re.search(r'('+SSA+r') = rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 1',line)
   if m and cs[m[2]]==2:assert used_once(m[1],part);removed.append('Bemptyarrive');continue
   m=re.search(r'rocdl.hcu.s.abarrier.inv ('+SSA+r')',line)
   if m and cs[m[1]] in (2,3):removed.append('inv'+str(cs[m[1]]));continue
   kept.append(line)
  assert sorted(removed)==['Bemptyarrive','Breadywait','inv2','inv3']
  parts.append(''.join(kept));proof.append(dict(consumer=role,removed=removed,empty0_after_all_original_local_reads=True))
 result=''.join(parts)
 assert result.count('rocdl.hcu.s.abarrier.init ')==2 and result.count('rocdl.hcu.s.abarrier.inv ')==4
 for token in [' = tt.dot ',' = ttg.local_load ','ttg.memdesc_subslice ','amdg.matrix_load_to_local ','ttg.local_alloc','arith.xori','scf.yield']:
  assert [l.strip() for l in source.splitlines() if token in l]==[l.strip() for l in result.splitlines() if token in l],token
 return result,dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(result.encode()).hexdigest(),data_ids=[0,1],empty_count=8,ready_count=4,initial_empty_arrival_per_producer_wave=2,producer_seq=1,full_BK128_MLS_both_operands_unchanged=True,all_local_reads_dot_math_allocations_unchanged=True,Aready_moved_after_B_transfer=True,consumers=proof)
def rewrite_module(module):
 from triton._C.libtriton import ir
 from .partial_k_local import normalize_mls
 text,proof=transform(str(module));text,n=normalize_mls(text);assert n==2
 with tempfile.TemporaryDirectory(prefix='pmd-merge-ab-') as t:
  p=Path(t)/'ab.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
 dump=os.environ.get('TRITON_MERGE_AB_DUMP_DIR')
 if dump:
  p=Path(dump);p.mkdir(parents=True,exist_ok=True);(p/'before.ttgir').write_text(str(module));(p/'after.ttgir').write_text(text);(p/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
 return new,proof
