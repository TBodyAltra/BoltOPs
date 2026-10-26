"""Bind math operands to their producer-assigned data barrier channels."""
import re,hashlib
SSA=r'%[\w.]+'
def derive(source):
 markers=list(re.finditer(r'^    partition([012])\(',source,re.M));assert [m[1] for m in markers]==['0','1','2']
 prefix=source[:markers[0].start()];producer=source[markers[0].start():markers[1].start()];lines=producer.splitlines()
 vals={m[1]:int(m[2]) for m in re.finditer(r'('+SSA+r') = arith.constant (\d+) : i32',prefix+producer)}
 for m in re.finditer(r'('+SSA+r') = llvm.mlir.constant\((\d+) : i32\)',prefix+producer):vals[m[1]]=int(m[2])
 init=re.findall(r'rocdl.hcu.s.abarrier.init ('+SSA+r'), (\d+)',prefix)
 initial=sorted((vals[v],int(n)) for v,n in init)
 assert [x for x in initial if x[0]<4]==[(0,8),(1,4),(2,8),(3,4)]
 assert [x for x in initial if x[0]>=4] in ([],[(4,4),(5,4)])
 copies=[i for i,l in enumerate(lines) if 'amdg.matrix_load_to_local ' in l];assert len(copies)==2
 maps={};records=[]
 for pos,i in enumerate(copies):
  lo=copies[pos-1]+1 if pos else 0;hi=copies[pos+1] if pos+1<len(copies) else len(lines)
  op=int(re.search(r'MlsEncoding opIdx = ([01])',lines[i])[1]);assert 'AB'[op] not in maps
  w=next(j for j in range(i-1,lo-1,-1) if 'rocdl.hcu.s.abarrier.try.wait ' in lines[j]);seq=next(j for j in range(i-1,lo-1,-1) if 'rocdl.hcu.s.abarrier.seq ' in lines[j]);arr=next(j for j in range(i+1,hi) if 'rocdl.hcu.s.abarrier.arrive ' in lines[j])
  e=vals[re.search(r'abarrier.try.wait i32, ('+SSA+r'),',lines[w])[1]];r=vals[re.search(r'abarrier.seq ('+SSA+r')',lines[seq])[1]];a=vals[re.search(r'abarrier.arrive i32, ('+SSA+r'), 1',lines[arr])[1]]
  assert w<seq<i<arr and r==a==e+1 and e in (0,2)
  dest=re.search(r'amdg.matrix_load_to_local '+SSA+r', ('+SSA+r'),',lines[i])[1]
  root_line=next(l for l in lines[:i] if re.match(r'\s*'+re.escape(dest)+r' = ttg.memdesc_index ',l));root=re.search(r'ttg.memdesc_index ('+SSA+r')',root_line)[1]
  maps['AB'[op]]=dict(empty=e,ready=r);records.append(dict(operand='AB'[op],empty=e,ready=r,producer_root=root,wait_line=w+1,sequence_line=seq+1,copy_line=i+1,arrival_line=arr+1))
 assert sorted(v['empty'] for v in maps.values())==[0,2]
 # The local-load math encoding must trace to the same shared root as its MLS.
 for role,part in [(1,source[markers[1].start():markers[2].start()]),(2,source[markers[2].start():])]:
  defs={}
  for l in part.splitlines():
   m=re.match(r'\s*('+SSA+r') = (.*)',l)
   if m:defs[m[1]]=m[2]
  seen={0:0,1:0}
  for l in part.splitlines():
   if ' = ttg.local_load ' not in l:continue
   op=int(re.search(r'dot_op<\{opIdx = ([01])',l)[1]);v=re.search(r'ttg.local_load ('+SSA+r')',l)[1]
   while v in defs:
    rhs=defs[v];m=re.match(r'ttg.memdesc_(?:index|subslice) ('+SSA+r')',rhs);assert m,(v,rhs);v=m[1]
   expect=next(x['producer_root'] for x in records if x['operand']=='AB'[op]);assert v==expect,(role,op,v,expect)
   seen[op]+=1
  assert seen=={0:2,1:2}
 return maps,dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),producer_order=[x['operand'] for x in records],channels=records,consumer_shared_roots_match_math_operands=True)
