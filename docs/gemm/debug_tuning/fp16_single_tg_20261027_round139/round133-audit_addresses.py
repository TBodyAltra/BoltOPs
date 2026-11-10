"""BM64 producer coverage and effective address/flag checks for all geometries."""
import pathlib,json,re,hashlib,collections,sys
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));from llvm_addresses import Addresses
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not(R/'parent-address-audit.json').exists()
e=json.loads((R/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists()
q=json.loads((R/'cpu-screen.json').read_text());assert q['default_controls_exact'] and len(q['eligible'])==7 and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
rows=[]
for x in q['eligible']:
 p=pathlib.Path(x['directory']);tile=x['tile'];lay=x['layout'];assert tile==[64,256,128]
 ref=next(y for y in q['eligible'] if y['layout']==lay and y['variant']=='oneA_full_B');cp=pathlib.Path(ref['directory']);ll=(p/'kernel.llir').read_text();cll=(cp/'kernel.llir').read_text();tt=(p/'kernel.ttgir').read_text()
 shapes={(int(o),int(m),int(n)) for o,m,n in re.findall(r'opIdx = ([01]), mlsTile = \[(\d+), (\d+)\]',tt)};assert shapes=={(0,16,32),(1,64,64)},shapes
 def rec(text,loops,it):
  out=[]
  for w in range(4):out+=Addresses(text,tile,lay,loops,w,it).records()
  return out
 def bases(text):
  rr=rec(text,16,0);return {op:min(v[2] for v in rr if v[0]==op) for op in ['A','B']}
 cb=bases(cll);ab=bases(ll);sizes=dict(A=16384,B=65536)
 spans=sorted((cb[o],cb[o]+sizes[o],o) for o in ['A','B']);assert spans[0][0]==0 and spans[0][1]==spans[1][0] and spans[1][1]==81920
 alias=x['config'].get('hcu_wasp_alias_ab_slots',False)
 if alias:assert ab==dict(A=0,B=0) and x['metadata']['shared']==65536
 else:assert sorted((ab[o],ab[o]+sizes[o],o) for o in ['A','B'])==spans and x['metadata']['shared']==81920
 digest=hashlib.sha256();total=0;halves={0:set(),1:set()}
 for loops in [16,32,64]:
  K=128*loops
  for it in range(loops):
   aa=rec(ll,loops,it);cc=rec(cll,loops,it);expected=[]
   for m in range(4):
    for k in range(4):expected.append(('A',Addresses.A+2*((m*16)*K+it*128+k*32),cb['A']+(m*4+k)*1024))
   for k in range(2):
    for n in range(4):
     for sub in range(4):
      g=Addresses.B+2*((it*128+k*64+sub*16)*256+n*64) if lay=='TT' else Addresses.B+2*(it*128+k*64+(n*64+sub*16)*K)
      dest=cb['B']+((k*4+n) if lay=='TT' else (n*2+k))*8192+sub*1024;expected.append(('B',g,dest))
   # Independent logical MLS tile coverage: sixteen A16x32 and eight B64x64,
   # each exactly once. Each B64x64 lowers to four64x16 intrinsics with source steps16 and dsByteOffsets0/1024/2048/3072, per MlsGroup.cpp version2/fallback3.
   assert collections.Counter(v[:3] for v in cc)==collections.Counter(expected),(lay,it,loops)
   norm=lambda rr,bs:collections.Counter((v[0],v[1],v[2]-bs[v[0]],*v[3:]) for v in rr)
   actual=norm(aa,ab);control=norm(cc,cb);assert actual==control,dict(layout=lay,variant=x['variant'],loops=loops,iteration=it,unexpected=list((actual-control).items())[:3],missing=list((control-actual).items())[:3])
   assert len(aa)==48 and all(0<=v[2]-ab[v[0]]<sizes[v[0]] for v in aa)
   for v in aa:
    if v[0]!='B':continue
    off=(v[1]-Addresses.B)//2;k=off//256 if lay=='TT' else off%K;halves[(k%128)//64].add(v[2]-ab['B'])
   assert not halves[0]&halves[1]
   digest.update(json.dumps(sorted(actual.elements()),separators=(',',':')).encode());total+=len(aa)
 assert min(halves[0])==0 and min(halves[1])==(32768 if lay=='TT' else 8192)
 if alias:
  proof=json.loads((p/'alias-ab-proof/proof.json').read_text());assert proof['overlapped_B_empty_ids']==([2] if lay=='TT' else [2,4]) and proof['A_bytes']==16384 and proof['A_contiguous_bytes_before_B1']==min(halves[1])
 if x['config'].get('hcu_wasp_alias_b1_early',False):assert lay=='TT' and 16384<=min(halves[1])
 rows.append(dict(tile=tile,layout=lay,variant=x['variant'],feature=str(p),reference=str(cp),full_original_parent_spans=spans,feature_bases=ab,shared_bytes=x['metadata']['shared'],independent_logical_coverage='16 A16x32 macros + 8 B64x64 macros lowered to32 B micro-intrinsics, exactly once per BK128',checked_iterations=112,checked_loop_counts=[16,32,64],producer_waves=4,effective_global_and_local_addresses_and_all_flags_match_full_parent_after_base_translation=True,total_matrix_load_events=total,B1_min_start=min(halves[1]),events_sha256=digest.hexdigest(),llir_sha256=sha(p/'kernel.llir'),asm_sha256=sha(p/'kernel.amdgcn')))
 print(json.dumps(dict(layout=lay,variant=x['variant'],coverage_and_addresses_passed=True)),flush=True)
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
(R/'parent-address-audit.json').write_text(json.dumps(dict(passed=True,CPU_only=True,rows=rows,source_hashes_unchanged=True,scope='BM64 actual LLVM producer logical tile coverage and all4wave address/flag evaluation for112iterations; consumer/ownership/native precision and liveness remain unproven',MlsGroup_source_sha256=sha(R/'triton/third_party/hcu/lib/TritonHCUTransforms/MlsGroup.cpp'),producer_lowering_source_sha256=sha(R/'triton/third_party/hcu/lib/TritonHCUToLLVM/MlsOpToLLVM.cpp'),evaluator_sha256=sha(R/'llvm_addresses.py'),script_sha256=sha(pathlib.Path(__file__))),indent=2)+'\n')
