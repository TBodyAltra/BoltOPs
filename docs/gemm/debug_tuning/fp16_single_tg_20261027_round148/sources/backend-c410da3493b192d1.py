"""Private full-A register capture followed by B reuse of the same LDS allocation.
Only a full contiguous root B allocation is reinterpreted; both operand encodings
remain unchanged. Producer consumes overlapping B empty phases before A writes,
then waits for current A capture before B writes. No barrier IDs are added.
"""
import pathlib,re,json,hashlib,tempfile,os
SSA=r'%[\w.]+'
def constants(s):
 d={m[1]:int(m[2]) for m in re.finditer('('+SSA+r') = arith.constant (\d+) : i32',s)}
 d.update({m[1]:int(m[2]) for m in re.finditer('('+SSA+r') = llvm.mlir.constant\((\d+) : i32\)',s)});return d

def transform(source):
 assert '__aas_' not in source and '__mt_' not in source and 'hcu.consumer_pingpong' not in source
 marks=list(re.finditer(r'^    partition([012])\(',source,re.M));assert [m[1] for m in marks]==['0','1','2']
 pre=source[:marks[0].start()];producer=source[marks[0].start():marks[1].start()];consumers=source[marks[1].start():]
 allocs=list(re.finditer(r'^    ('+SSA+r') = ttg.local_alloc : \(\) -> (!ttg.memdesc<[^\n]+>)( loc\([^\n]+\))\n',pre,re.M));assert len(allocs)==2
 aa=next(m for m in allocs if re.match(r'!ttg.memdesc<1x(?:32|128|256)x128xf16,',m[2]));bb=next(m for m in allocs if m is not aa);assert re.match(r'!ttg.memdesc<1x128x256xf16,',bb[2]);bm=int(re.match(r'!ttg.memdesc<1x(\d+)x128xf16,',aa[2])[1]);abytes=2*bm*128
 assert pre.count(aa[1])==2 and pre.count(bb[1])==2
 replacement=bb[0]+f'    {aa[1]} = ttg.memdesc_reinterpret {bb[1]} : {bb[2]} -> {aa[2]}{aa[3]}\n'
 pre=pre.replace(aa[0],'').replace(bb[0],replacement)
 v=constants(producer);ls=producer.splitlines(True);waits={};arr={}
 for i,l in enumerate(ls):
  m=re.search(r'('+SSA+r') = rocdl.hcu.s.abarrier.try.wait i32, ('+SSA+r'), ('+SSA+r')( loc\([^\n]+\))',l)
  if m:waits[v[m[2]]]=(i,m)
  m=re.search(r'rocdl.hcu.s.abarrier.arrive i32, ('+SSA+r'), 1',l)
  if m:arr[v[m[1]]]=i
 assert sorted(waits)==[0,2,4] and sorted(arr)==[1,3,5]
 assert waits[0][0]<arr[1]<waits[2][0]<arr[3]<waits[4][0]<arr[5]
 benc=re.match(r'!ttg.memdesc<1x128x256xf16, (#[\w.]+),',bb[2])[1]
 enc=next(l for l in source.splitlines() if l.startswith(benc+' = #ttg.mls_shared<'))
 order=list(map(int,re.search(r'order = \[(\d+), (\d+)\]',enc).groups()));tile=list(map(int,re.search(r'mlsTile = \[(\d+), (\d+)\]',enc).groups()));assert order in ([0,1],[1,0])
 # K-minor TN interleaves its K64 halves per N tile: B1 starts after8192B,
 # not after an entire32768B half. Full A view is a contiguous prefix.
 if order==[0,1]:assert tile==[64,64]
 safe_prefix=2*tile[0]*tile[1] if order==[0,1] else 32768
 overlap=[2] if abytes<=safe_prefix else [2,4];move={waits[k][0] for k in overlap}
 for k in overlap:
  assert len(re.findall(re.escape(waits[k][1][1])+r'(?![\w.])',producer))==1
 aempty=waits[0][1][2];aphase=waits[0][1][3];loc=waits[0][1][4];one=next(k for k,n in v.items() if n==1)
 add=f'        %__aas_nextAphase = arith.xori {aphase}, {one} : i32{loc}\n        %__aas_Acaptured = rocdl.hcu.s.abarrier.try.wait i32, {aempty}, %__aas_nextAphase{loc}\n'
 out=[]
 for i,l in enumerate(ls):
  if i==waits[0][0]:out.extend(ls[waits[k][0]] for k in overlap)
  if i not in move:out.append(l)
  if i==arr[1]:out.append(add)
 result=pre+''.join(out)+consumers
 for marker in [' = tt.dot ',' = ttg.local_load ',' = amdg.matrix_load_to_local ','rocdl.hcu.s.abarrier.seq ','rocdl.hcu.s.abarrier.arrive ','rocdl.hcu.s.abarrier.init ','rocdl.hcu.s.abarrier.inv ','scf.yield']:
  assert [l.strip() for l in source.splitlines() if marker in l]==[l.strip() for l in result.splitlines() if marker in l],marker
 assert result.count(' = ttg.local_alloc ')==1 and result.count(' = ttg.memdesc_reinterpret ')==1 and result.count('rocdl.hcu.s.abarrier.try.wait ')==source.count('rocdl.hcu.s.abarrier.try.wait ')+1
 return result,dict(passed=True,tile=[bm,256,128],old_shared_bytes=65536+abytes,expected_shared_bytes=65536,A_bytes=abytes,B_bytes=65536,B_layout_order=order,B_mls_tile=tile,A_contiguous_bytes_before_B1=safe_prefix,A_view_of_full_contiguous_B_root=True,operand_layouts_and_shapes_unchanged=True,overlapped_B_empty_ids=overlap,overlapped_B_empty_waits_moved_before_A_write=True,current_A_capture_wait_before_B_write=True,A_capture_wait_phase='producer A empty phase xor1; current generation completion, not seeded previous-generation empty',data_ids=list(range(6)),no_added_barrier_ids=True,consumer_and_math_unchanged=True,source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(result.encode()).hexdigest())

def rewrite_module(module):
 from triton._C.libtriton import ir
 from .partial_k_local import normalize_mls
 text,proof=transform(str(module));text,n=normalize_mls(text);assert n==2
 with tempfile.TemporaryDirectory(prefix='pmd-alias-ab-slots-') as d:
  p=pathlib.Path(d)/'alias.ttgir';p.write_text(text);new=ir.parse_mlir_module(str(p),module.context);new.context=module.context;assert new.verify()
 proof['module_verified']=True
 dump=os.environ.get('TRITON_ALIAS_AB_SLOTS_DUMP_DIR')
 if dump:
  p=pathlib.Path(dump);p.mkdir(parents=True,exist_ok=True);(p/'before.ttgir').write_text(str(module));(p/'after.ttgir').write_text(text);(p/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
 return new,proof
