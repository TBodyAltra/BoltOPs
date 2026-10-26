"""Compare all consumer lanes and K64 regions with the original full-B geometry."""
import pathlib,json,re,hashlib,collections,sys
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));from llvm_addresses import Addresses,split
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class LaneAddresses(Addresses):
 def expr(self,r,name=None):
  if r.startswith('call ') and 'workitem.id.x' in r:return self.warp*64+self.lane
  if r.startswith('call ') and 'readfirstlane' in r and self.lane:
   m=re.search(r'@[\w.]+\((.*)\)',r);return self.zero.tv(split(m[1])[0])[1]
  return super().expr(r,name)
def lane(seed,lane_id,zero=None):
 a=LaneAddresses.__new__(LaneAddresses);a.__dict__=dict(seed.__dict__);a.cache=dict(seed.cache);a.busy=set();a.lane=lane_id;a.zero=a if zero is None else zero;return a
def parse(text):
 role=-1;operand=None;out=[]
 for l in text.splitlines():
  m=re.search(r'call i32(?: \(i32, \.\.\.\))? @llvm.hcu.s.abarrier.try.wait\(i32 ([135]),',l)
  if m:
   if m[1]=='1':role+=1;operand='A'
   else:operand='B'
  m=re.search(r'call <\d+ x half> @(llvm.hcu.ds.read.matrix[\w.]+)\((.*?)\)(?:, !|$)',l)
  if m:assert role in (0,1) and operand in ['A','B'];out.append((role,operand,m[1],split(m[2])))
 assert {r for r,_,_,_ in out}=={0,1};return out
def evaluate(a,parsed,bases,layout,Abytes):
 out=collections.defaultdict(collections.Counter)
 for role,operand,op,args in parsed:
  vals=[a.tv(v)[1] for v in args];assert len(vals)>=3;address=vals[0]+vals[1]-bases[operand]
  assert 0<=address<(Abytes if operand=='A' else 65536)
  slot=0 if operand=='A' else 1+(int(address>=32768) if layout=='TT' else (address//8192)%2)
  out[(role,slot)][(op,address,*vals[2:])]+=1
 assert set(out)=={(r,s) for r in range(2) for s in range(3)}
 return dict(out)
assert not(R/'consumer-read-audit.json').exists()
q=json.loads((R/'cpu-screen.json').read_text());pa=json.loads((R/'parent-address-audit.json').read_text());assert pa['passed'] and len(q['eligible'])==8
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());rows=[]
for x in q['eligible']:
 p=pathlib.Path(x['directory']);cp=pathlib.Path(x['control_seed_directory']);ref=json.loads((cp/'result.json').read_text());ll=(p/'kernel.llir').read_text();cll=(cp/'kernel.llir').read_text();parsed=parse(ll);cparsed=parse(cll);ab=cb=dict(A=0,B=0);tests=[]
 for loops in [16,32,64]:
  for it in sorted({0,loops//2,loops-1}):
   for wave in range(4,12):
    seed=Addresses(ll,x['tile'],x['layout'],loops,wave,it);cseed=Addresses(cll,x['tile'],x['layout'],loops,wave,it);zero=lane(seed,0);czero=lane(cseed,0)
    for lid in range(64):
     actual=evaluate(zero if lid==0 else lane(seed,lid,zero),parsed,ab,x['layout'],x['tile'][0]*128*2);expected=evaluate(czero if lid==0 else lane(cseed,lid,czero),cparsed,cb,x['layout'],x['tile'][0]*128*2);assert actual==expected,(x['layout'],x['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0),loops,it,wave,lid)
   tests.append([loops,it])
 rows.append(dict(layout=x['layout'],wait_mask=x['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0),tile=x['tile'],wave_ids=list(range(4,12)),lane_ids=list(range(64)),loop_iteration_tests=tests,all_six_role_A_B0_B1_regions_separately_match=True,total_address_and_all_format_flags_equal_after_base_translation=True,readfirstlane_uses_lane0_value=True,feature_llir_sha256=sha(p/'kernel.llir'),control_llir_sha256=sha(cp/'kernel.llir')))
 print(json.dumps(dict(layout=x['layout'],wait_mask=x['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0),all8waves64lanes_read_checks_passed=True)),flush=True)
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
(R/'consumer-read-audit.json').write_text(json.dumps(dict(passed=True,CPU_only=True,rows=rows,source_hashes_unchanged=True,producer_address_audit_sha256=sha(R/'parent-address-audit.json'),scope='Actual LLVM DS-read addresses/flags for all64 lanes of8consumer waves, both roles A/B0/B1, first/middle/last iterations16/32/64; no native execution/precision or ownership claim',evaluator_sha256=sha(R/'llvm_addresses.py'),script_sha256=sha(pathlib.Path(__file__))),indent=2)+'\n')
