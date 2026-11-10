"""Evaluate each consumer's A/B0/B1 read regions after independent base translation."""
import pathlib,json,hashlib,re,sys,collections
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));from llvm_addresses import Addresses,split
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();q=json.loads((R/'destination-view-screen.json').read_text());qual=json.loads((R/'qualification.json').read_text());assert len(q['results'])==6 and qual['passed'] and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
def reads(text,tile,layout,loops,warp,iteration,bases):
 a=Addresses(text,tile,layout,loops,warp,iteration);role=-1;slot=None;out=collections.defaultdict(collections.Counter)
 for l in text.splitlines():
  m=re.search(r'call i32 @llvm.hcu.s.abarrier.try.wait\(i32 ([135]),',l)
  if m:
   ready=int(m[1]);slot=(ready-1)//2
   if ready==1:role+=1
  m=re.search(r'call <\d+ x half> @(llvm.hcu.ds.read.matrix[\w.]+)\((.*?)\)(?:, !|$)',l)
  if m:
   assert role in (0,1) and slot in (0,1,2),(role,slot,l)
   vals=[a.tv(v)[1] for v in split(m[2])];assert len(vals)>=3
   address=vals[0]+vals[1];operand='A' if slot==0 else 'B';address-=bases[operand]
   assert 0<=address<tile[0]*128*2 if slot==0 else 0<=address<65536
   out[(role,slot)][(m[1],address,*vals[2:])]+=1
 assert set(out)=={(r,s) for r in range(2) for s in range(3)},set(out)
 return dict(out)
rows=[]
for x in q['results']:
 p=pathlib.Path(x['directory']);c=next(c for c in qual['controls'] if c['tile']==x['tile'] and c['layout']==x['layout']);cp=pathlib.Path(c['directory']);text=(p/'kernel.llir').read_text();ref=(cp/'kernel.llir').read_text();events=[]
 for w in range(4):events+=Addresses(ref,x['tile'],x['layout'],16,w,0).records()
 bases={o:min(v[2] for v in events if v[0]==o) for o in ['A','B']};tests=[]
 for loops in [16,32,64]:
  for iteration in sorted(set([0,loops//2,loops-1])):
   for warp in range(4,12):
    actual=reads(text,x['tile'],x['layout'],loops,warp,iteration,dict(A=0,B=0));expected=reads(ref,x['tile'],x['layout'],loops,warp,iteration,bases);assert actual==expected,(x['tile'],x['layout'],loops,iteration,warp)
   tests.append([loops,iteration])
 rows.append(dict(tile=x['tile'],layout=x['layout'],all_six_role_operand_regions_separately_matched=True,addresses_and_all_format_flags_equal_after_base_translation=True,wave_ids_checked=list(range(4,12)),loop_and_iteration_tests=tests,reference_operand_bases=bases,shared_bytes=x['metadata']['shared'],llir_sha256=sha(p/'kernel.llir'),asm_sha256=sha(p/'kernel.amdgcn')));print(json.dumps(dict(tile=x['tile'],layout=x['layout'],read_translation_passed=True)),flush=True)
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());(R/'alias-consumer-read-audit.json').write_text(json.dumps(dict(passed=True,CPU_only=True,rows=rows,scope='Static actual LLVM ds.read.matrix total address and flags for both consumer regions separately A/B0/B1, eight consumer wave IDs, first/mid/last iterations16/32/64; not native execution or performance.',source_hashes_unchanged=True,script_sha256=sha(pathlib.Path(__file__))),indent=2)+'\n')
