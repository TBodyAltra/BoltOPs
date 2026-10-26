"""Compare every executed producer warp's effective source/LDS tuple to full-B reference."""
import pathlib,json,hashlib,re,collections,sys,time
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));from llvm_addresses import Addresses
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();q=json.loads((R/'destination-view-screen.json').read_text());qual=json.loads((R/'qualification.json').read_text());assert q['default_qualified'] and q['source_hashes_unchanged'] and qual['passed'] and len(q['results'])==4
assert json.loads((R/'job-exit.json').read_text())['returncode']==0 and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
rows=[]
for x in q['results']:
 assert x['status']=='COMPILED';p=pathlib.Path(x['directory']);tile=x['tile'];lay=x['layout'];c=next(c for c in qual['controls'] if c['tile']==tile and c['layout']==lay);cp=pathlib.Path(c['directory']);text=(p/'kernel.llir').read_text();ref=(cp/'kernel.llir').read_text()
 assert x['metadata']['shared']==c['metadata']['shared']==(tile[0]+tile[1])*128*2
 assert all(x['fields'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])
 proof=json.loads((p/'split-b-global-proof/proof.json').read_text());assert proof['module_verified'] and proof['parent_shape']==[128,256] and proof['K_offsets']==[0,64] and proof['producer_warp_layout']==[1,4]
 assert text.count('call void @llvm.hcu.matrix.load.')==ref.count('call void @llvm.hcu.matrix.load.')
 # Infer allocation order from the qualified full-parent reference, rather than
 # assuming A precedes B. The skinny parent puts B at zero and A at 65536.
 baseline=[]
 for warp in range(4): baseline+=Addresses(ref,tile,lay,16,warp,0).records()
 bases={o:min(v[2] for v in baseline if v[0]==o) for o in ['A','B']}
 sizes={'A':tile[0]*128*2,'B':tile[1]*128*2}
 spans=sorted((bases[o],bases[o]+sizes[o],o) for o in ['A','B'])
 assert spans[0][0]==0 and spans[0][1]==spans[1][0] and spans[1][1]==x['metadata']['shared'],spans
 for v in baseline: assert bases[v[0]]<=v[2]<bases[v[0]]+sizes[v[0]],v
 counts=[];digest=hashlib.sha256();halves={0:set(),1:set()}
 for loops in [16,32,64]:
  for iteration in range(loops):
   actual=[];expected=[]
   for warp in range(4):
    actual+=Addresses(text,tile,lay,loops,warp,iteration).records();expected+=Addresses(ref,tile,lay,loops,warp,iteration).records()
   aa=collections.Counter(actual);ee=collections.Counter(expected)
   assert aa==ee,dict(tile=tile,layout=lay,loops=loops,iteration=iteration,unexpected=list((aa-ee).items())[:4],missing=list((ee-aa).items())[:4])
   bb=[v for v in actual if v[0]=='B'];assert len(set(bb))==len(bb)
   for v in bb:
    offset=(v[1]-Addresses.B)//2;K=tile[2]*loops;k=offset//tile[1] if lay=='TT' else offset%K
    half=(k%128)//64;halves[half].add(v[2]);assert bases['B']<=v[2]<bases['B']+sizes['B']
   assert halves[0] and halves[1] and not halves[0]&halves[1]
   digest.update(json.dumps(sorted(actual),separators=(',',':')).encode());counts.append(len(actual))
 rows.append(dict(tile=tile,layout=lay,feature=str(p),reference=str(cp),checked_loop_counts=[16,32,64],checked_iterations=112,producer_warps_per_iteration=4,total_matrix_load_events=sum(counts),same_effective_global_source_and_LDS_mapping=True,no_duplicate_B_transfer_events=True,B_K64_destination_start_sets_disjoint=True,B_first_LDS_start=min(halves[0]),B_second_LDS_start=min(halves[1]),B_half_LDS_start_delta=min(halves[1])-min(halves[0]),all_B_start_addresses_within_allocated_parent=True,reference_parent_allocation_spans=spans,allocation_order_from_qualified_reference=True,evaluated_events_sha256=digest.hexdigest(),matrix_load_static_count=text.count('call void @llvm.hcu.matrix.load.'),reference_matrix_load_static_count=ref.count('call void @llvm.hcu.matrix.load.'),llir_sha256=sha(p/'kernel.llir'),asm_sha256=sha(p/'kernel.amdgcn')))
# The independent evaluator must also reject the old real counterexample.
old=R.parent/'single-tg-pipeline-20261026-round94-b-global-subview-cpu';bad=json.loads((old/'split-b-global-lowering-screen.json').read_text())['results'][0];badtext=(pathlib.Path(bad['directory'])/'kernel.llir').read_text();good=next(r for r in rows if r['tile']==[256,256,128] and r['layout']=='TN');ref=(pathlib.Path(good['reference'])/'kernel.llir').read_text();ba=[];rr=[]
for w in range(4):ba+=Addresses(badtext,[256,256,128],'TN',16,w,0).records();rr+=Addresses(ref,[256,256,128],'TN',16,w,0).records()
assert collections.Counter(ba)!=collections.Counter(rr) and sum(v[0]=='B' for v in ba)==2*sum(v[0]=='B' for v in rr)
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
out=dict(passed=True,CPU_only=True,native_precision_and_liveness_still_required=True,full_split_slot_pipeline_implemented=False,native_sha256=q['build']['sha256'],rows=rows,old_alias_and_double_transfer_counterexample_rejected=True,scope='Actual LLVM intrinsic operand evaluation for all4 producer waves and EVERY iteration of each16/32/64 calibration length; compare operation multisets to full-B128 qualified reference. This proves address/transfer equivalence, not GPU execution correctness or performance.',source_hashes_unchanged=True,screen_sha256=sha(R/'destination-view-screen.json'),evaluator_sha256=sha(R/'llvm_addresses.py'),audit_sha256=sha(pathlib.Path(__file__)))
(R/'parent-address-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(rows),iterations_per_case=112,producer_warps=4,old_counterexample_rejected=True,half_LDS_deltas=[(r['layout'],r['B_half_LDS_start_delta']) for r in rows])))
