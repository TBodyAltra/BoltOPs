"""Check actual early-B1 producer addresses against qualified aliased controls."""
import pathlib,json,hashlib,re,collections,sys
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));from llvm_addresses import Addresses
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not(R/'parent-address-audit.json').exists()
e=json.loads((R/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists()
q=json.loads((R/'cpu-screen.json').read_text());qual=json.loads((R/'control-qualification.json').read_text());assert q['default_controls_exact'] and qual['passed'] and len(q['eligible'])==26
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
rows=[]
for x in q['eligible']:
 p=pathlib.Path(x['directory']);tile=x['tile'];lay=x['layout'];cp=pathlib.Path(x['quota_seed_directory']);c=json.loads((cp/'result.json').read_text());text=(p/'kernel.llir').read_text();ref=(cp/'kernel.llir').read_text()
 assert x['metadata']['shared']==c['metadata']['shared']==65536
 assert all(x['fields'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
 proof=json.loads((p/'slot-turn-proof/proof.json').read_text());assert proof['mode']==2 and proof['wait_mask']==x['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0) and proof['data_and_math_unchanged']
 assert text.count('call void @llvm.hcu.matrix.load.')==ref.count('call void @llvm.hcu.matrix.load.')
 counts=[];digest=hashlib.sha256();halves={0:set(),1:set()};amin=[]
 for loops in [16,32,64]:
  for iteration in range(loops):
   actual=[];expected=[]
   for warp in range(4):actual+=Addresses(text,tile,lay,loops,warp,iteration).records();expected+=Addresses(ref,tile,lay,loops,warp,iteration).records()
   aa=collections.Counter(actual);ee=collections.Counter(expected)
   assert aa==ee,dict(tile=tile,layout=lay,loops=loops,iteration=iteration,unexpected=list((aa-ee).items())[:4],missing=list((ee-aa).items())[:4])
   assert all(0<=v[2]<(tile[0]*128*2 if v[0]=='A' else 65536) for v in actual)
   bb=[v for v in actual if v[0]=='B'];assert len(set(bb))==len(bb)
   amin.extend(v[2] for v in actual if v[0]=='A')
   for v in bb:
    offset=(v[1]-Addresses.B)//2;K=128*loops;k=offset//tile[1] if lay=='TT' else offset%K;half=(k%128)//64;halves[half].add(v[2])
   assert halves[0] and halves[1] and not halves[0]&halves[1]
   digest.update(json.dumps(sorted(actual),separators=(',',':')).encode());counts.append(len(actual))
 assert min(amin)==min(halves[0])==0
 alias=json.loads((p/'alias-ab-proof/proof.json').read_text());required=[2]+([4] if tile[0]*128*2>min(halves[1]) else []);assert alias['overlapped_B_empty_ids']==required and alias['A_contiguous_bytes_before_B1']==min(halves[1])
 rows.append(dict(tile=tile,layout=lay,feature=str(p),reference=str(cp),shared_bytes=65536,all4producer_waves=True,checked_loop_counts=[16,32,64],checked_iterations=112,total_matrix_load_events=sum(counts),same_per_operand_actual_global_sources_and_LDS_destinations_and_flags=True,no_duplicate_B_transfers=True,B0_B1_disjoint=True,A_contiguous_prefix_bytes=tile[0]*128*2,actual_B1_min_start=min(halves[1]),A_and_B1_physically_disjoint=tile[0]*128*2<=min(halves[1]),wait_mask=x['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0),events_sha256=digest.hexdigest(),feature_llir_sha256=sha(p/'kernel.llir'),control_llir_sha256=sha(cp/'kernel.llir'),feature_asm_sha256=sha(p/'kernel.amdgcn')))
badroot=R.parent/'single-tg-pipeline-20261026-round94-b-global-subview-cpu';bad=json.loads((badroot/'split-b-global-lowering-screen.json').read_text())['results'][0];badtext=(pathlib.Path(bad['directory'])/'kernel.llir').read_text();good=next(x for x in json.loads((R.parent/'single-tg-pipeline-20261026-round96-global-subview-warp-cpu/qualification.json').read_text())['controls'] if x['tile']==[256,256,128] and x['layout']=='TN');goodtext=(pathlib.Path(good['directory'])/'kernel.llir').read_text();ba=[];rr=[]
for w in range(4):ba+=Addresses(badtext,[256,256,128],'TN',16,w,0).records();rr+=Addresses(goodtext,[256,256,128],'TN',16,w,0).records()
assert collections.Counter(ba)!=collections.Counter(rr) and sum(v[0]=='B' for v in ba)==2*sum(v[0]=='B' for v in rr)
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
out=dict(passed=True,CPU_only=True,scope='Actual LLVM matrix-load intrinsic operand/flag evaluation for all4producer waves and112iterations per case; operation multiset equality to exact alias controls. Does not prove consumer reads or actual emitted synchronization.',rows=rows,source_hashes_unchanged=True,old_double_transfer_counterexample_rejected=True,native_precision_and_ISA_ownership_still_required=True,screen_sha256=sha(R/'cpu-screen.json'),evaluator_sha256=sha(R/'llvm_addresses.py'),audit_sha256=sha(pathlib.Path(__file__)))
(R/'parent-address-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=26,iterations_per_case=112,producer_waves=4,old_counterexample_rejected=True)))
