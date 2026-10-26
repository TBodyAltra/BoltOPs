import pathlib,json,hashlib,re,sys,collections
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));from llvm_addresses import Addresses,split
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((R/'destination-view-screen.json').read_text());qual=json.loads((R/'qualification.json').read_text());model=json.loads((R/'protocol-model.json').read_text());assert model['passed'] and model['script_sha256']==sha(R/'protocol_model.py') and json.loads((R/'parent-address-audit.json').read_text())['passed'] and json.loads((R/'cpu-complete.json').read_text())['passed']
base_model=json.loads((R/'base-protocol-model.json').read_text());assert base_model['passed'] and base_model['script_sha256']==sha(R/'base_protocol_model.py')
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());rows=[]
def asmops(text):
 return [line.split(';',1)[0].strip() for line in text.splitlines() if re.match(r'^\s+(?:s_|v_|ds_|matrix_|buffer_)',line)]
def reads(text,tile,layout,loops,warp,iteration):
 a=Addresses(text,tile,layout,loops,warp,iteration);out=[]
 for l in text.splitlines():
  m=re.search(r'call <\d+ x half> @(llvm.hcu.ds.read.matrix[\w.]+)\((.*?)\)(?:, !|$)',l)
  if m:out.append((m[1],*(a.tv(v)[1] for v in split(m[2]))))
 assert out
 return collections.Counter(out)
for x in q['results']:
 p=pathlib.Path(x['directory']);c=next(v for v in qual['controls'] if v['tile']==x['tile'] and v['layout']==x['layout']);cp=pathlib.Path(c['directory']);mode=x['config']['hcu_wasp_split_b_slot_turn'];mask=x['config']['hcu_wasp_split_b_slot_turn_wait_mask'];proof=json.loads((p/('slot-turn-proof/proof.json' if mode else 'split-b-slots-proof/proof.json')).read_text());mode=x['config']['hcu_wasp_split_b_slot_turn'];assert (proof['wait_mask']==mask and proof['mode']==mode and proof['turn_ids']==[6,7] and proof['data_barrier_ids']==list(range(6)) and proof['data_and_math_unchanged']) if mode else proof['module_verified'] and proof['split_slot_ring_implemented']
 asm=(p/'kernel.amdgcn').read_text();refasm=(cp/'kernel.amdgcn').read_text();ops=asmops(asm);ll=(p/'kernel.llir').read_text();rl=(cp/'kernel.llir').read_text()
 assert collections.Counter(re.findall(r'\bv_mmac_\w+',asm))==collections.Counter(re.findall(r'\bv_mmac_\w+',refasm))
 assert collections.Counter(re.findall(r'\bmatrix_load_\w+',asm))==collections.Counter(re.findall(r'\bmatrix_load_\w+',refasm))
 init=[tuple(map(int,m)) for m in re.findall(r'@llvm.hcu.s.abarrier.init\(i32 (\d+), i32 (\d+)\)',ll)];assert sorted(init)==[(0,8),(1,4),(2,8),(3,4),(4,8),(5,4)]+([(6,4),(7,4)] if mode else []),init
 releases=[]
 for i,op in enumerate(ops):
  m=re.match(r's_abarrier_arrive\s+s\d+, ([024]), 1$',op)
  if m:
   # Scheduling may leave completed math between drain and release. Require
   # the latest DS operation to precede the latest full drain, and no control
   # transfer between that drain and the release (same straight-line region).
   drained=max(j for j in range(i) if ops[j].startswith('s_waitcnt ') and 'lgkmcnt(0)' in ops[j])
   latest_ds=max(j for j in range(i) if ops[j].startswith('ds_'))
   assert drained>latest_ds and not any(re.match(r's_(?:cbranch|branch|setpc)',v) for v in ops[drained+1:i]),(i,drained,latest_ds,ops[drained:i+1])
   releases.append(int(m[1]))
 assert sorted(releases)==[0,0,2,2,4,4],releases
 assert sum(bool(re.fullmatch(r's_ebarrier_sync\s+2, 0',op)) for op in ops)==3
 inv=[int(re.fullmatch(r's_abarrier_inv (\d+)',op)[1]) for op in ops if op.startswith('s_abarrier_inv')];assert sorted(inv)==sorted(list(range(6+2*bool(mode)))*2)
 for i,op in enumerate(ops):
  if op in ['s_abarrier_inv 6','s_abarrier_inv 7'] and re.fullmatch(r's_ebarrier_sync\s+2, 0',ops[i-1]):assert re.fullmatch(r's_ebarrier_sync\s+2, 0',ops[i-1])
 for i,op in enumerate(ops):
  if op.startswith('s_abarrier_inv') and not ops[i-1].startswith('s_abarrier_inv'):assert re.fullmatch(r's_ebarrier_sync\s+2, 0',ops[i-1])
 assert not any(re.match(r's_ebarrier_sync\s+1,',op) for op in ops)
 # Each actual consumer must acquire its role's token BEFORE the corresponding
 # B ready wait; all required MMAC issue instructions precede its token signal.
 starts=[i for i,op in enumerate(ops) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, 1, .+',op)];assert len(starts)==2
 turn_records=[]
 for role,start in (enumerate(starts) if mode else []):
  end=starts[role+1] if role+1<len(starts) else len(ops);seg=ops[start:end];inp,out=(6,7) if role==0 else (7,6)
  tw=[i for i,op in enumerate(seg) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, '+str(inp)+r', .+',op)]
  ts=[i for i,op in enumerate(seg) if re.fullmatch(r's_abarrier_arrive\s+s\d+, '+str(out)+r', 1',op)]
  bw=[next(i for i,op in enumerate(seg) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, '+str(ready)+r', .+',op)) for ready in [3,5]]
  mm=[i for i,op in enumerate(seg) if op.startswith('v_mmac_')];assert len(tw)==len(ts)==mode and len(mm)>0 and len(mm)%2==0
  assert mode==2 and mm[-1]<ts[-1] and sum(i<ts[0] for i in mm)==len(mm)//2
  br=[next(i for i,op in enumerate(seg) if re.fullmatch(r's_abarrier_arrive\s+s\d+, '+str(empty)+r', 1',op)) for empty in [2,4]]
  for f in range(2):
   first_math=mm[f*(len(mm)//2)]
   if mask&(1<<f):assert bw[f]<br[f]<tw[f]<first_math,(mask,f,seg)
   else:assert tw[f]<bw[f]<first_math,(mask,f,seg)
  assert ts[0]<tw[1]
  turn_records.append(dict(consumer=role+1,in_id=inp,out_id=out,turns_per_BK=mode,wait_mask=mask,MMAC_static_count=len(mm),late_wait_after_B_release_before_first_fragment_math=True,math_issue_before_turn_signal=True))
 # Ready publication is still the legacy MLS seq/arrive protocol, separately
 # paired for each slot; do not invent an ebarrier6 which actual ISA lacks.
 for ready in [1,3,5]:
  seq=[i for i,op in enumerate(ops) if re.fullmatch(r's_abarrier_seq '+str(ready),op)];arr=[i for i,op in enumerate(ops) if re.fullmatch(r's_abarrier_arrive\s+s\d+, '+str(ready)+r', 1',op)];assert len(seq)==len(arr)==1 and seq[0]<arr[0] and any(op.startswith('matrix_load_') for op in ops[seq[0]:arr[0]])
 # Evaluate static read regions from BOTH consumers. We compare each region's
 # operands at all eight consumer wave IDs; this is not a claim that both
 # consumer regions execute on the same wave.
 tests=[]
 for loops in [16,32,64]:
  for iteration in sorted(set([0,loops//2,loops-1])):
   for warp in range(4,12):assert reads(ll,x['tile'],x['layout'],loops,warp,iteration)==reads(rl,x['tile'],x['layout'],loops,warp,iteration),(x['tile'],x['layout'],loops,iteration,warp)
   tests.append([loops,iteration])
 rows.append(dict(tile=x['tile'],layout=x['layout'],slot_turn_mode=mode,wait_mask=mask,actual_turns=turn_records,all_six_consumer_slot_releases_preceded_by_actual_lgkmcnt0=True,three_partition_exit_all_wave_join2=True,each_consumer_invalidates_expected_barrier_count_after_join=6+2*bool(mode),no_consumer_ebarrier1=True,producer_legacy_seq_copy_arrive_protocol_independent_per_slot=True,static_matrix_load_and_mmac_work_unchanged=True,actual_LLVM_static_consumer_read_address_and_flags_equal_reference=True,consumer_wave_ids_checked=list(range(4,12)),loop_and_iteration_tests=tests,physical=x['fields'],llir_sha256=sha(p/'kernel.llir'),asm_sha256=sha(p/'kernel.amdgcn'),proof_sha256=sha(p/('slot-turn-proof/proof.json' if mode else 'split-b-slots-proof/proof.json'))))
 print(json.dumps(dict(tile=x['tile'],layout=x['layout'],slot_turn_mode=mode,passed=True)),flush=True)
assert len(rows)==q['accepted_features'] and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
(R/'ownership-isa-audit.json').write_text(json.dumps(dict(passed=True,CPU_only=True,rows=rows,protocol_model_sha256=sha(R/'protocol-model.json'),base_protocol_model_sha256=sha(R/'base-protocol-model.json'),native_precision_liveness_and_performance_still_required=True,scope='Finite per-role individual-wave protocol model plus actual LLVM/ISA copy/read mapping, data release drain, 6/7 MMAC token issue order, barrier counts and teardown checks. Math can interleave with reads under scheduling; this does not prove a particular cycle schedule or performance.',source_hashes_unchanged=True,script_sha256=sha(pathlib.Path(__file__))),indent=2)+'\n')
