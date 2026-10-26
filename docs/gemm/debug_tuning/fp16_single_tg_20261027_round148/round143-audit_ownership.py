"""Alias ownership gates in actual LLVM/ISA, tied to explicit-counter finite model."""
import pathlib,json,hashlib,re,collections
R=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();q=json.loads((R/'cpu-screen.json').read_text());qual=json.loads((R/'control-qualification.json').read_text());model=json.loads((R/'alias-no-turn-protocol-model.json').read_text());assert model['passed'] and len(model['controls'])==10 and len(model['positive_cases'])==6 and len(model['negative_cases'])==54 and len(model['equivalent_guard_controls'])==32 and model['script_sha256']==sha(R/'protocol_alias_no_turn.py')
for n in ['parent-address-audit.json','consumer-read-audit.json']:assert json.loads((R/n).read_text())['passed']
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());rows=[]
def ops(text):return [l.split(';',1)[0].strip() for l in text.splitlines() if re.match(r'^\s+(?:s_|v_|ds_|matrix_|buffer_)',l)]
def phi_phase(ll,value):
 m=re.search(r'^  '+re.escape(value)+r' = phi i32 ([^\n]+)',ll,re.M);assert m,value
 arms=re.findall(r'\[\s*([^,]+), [^\]]+\]',m[1]);assert len(arms)==2 and '0' in arms,arms;back=next(x.strip() for x in arms if x.strip()!='0')
 assert re.search(r'^  '+re.escape(back)+r' = xor i32 '+re.escape(value)+r', 1(?:,|$)',ll,re.M),(value,back)
 return back

def actual_phase_xor(producer,first_index,last_index,phase_first,phase_last,bphases):
 env={phase_first:('g',0)}
 for p in bphases:env[p]=('g',0)
 def val(s):
  if s in env:return env[s]
  try:return int(s,0)
  except ValueError:return None
 for line in producer[first_index+1:last_index]:
  m=re.fullmatch(r's_mov_b32\s+(s\d+), (.+)',line)
  if m:env[m[1]]=val(m[2]);continue
  m=re.fullmatch(r's_xor_b32\s+(s\d+), ([^,]+), ([^,]+)',line)
  if m:
   a,b=val(m[2]),val(m[3]);x=None
   if isinstance(a,tuple) and isinstance(b,int):x=('g',a[1]^b)
   elif isinstance(b,tuple) and isinstance(a,int):x=('g',b[1]^a)
   elif isinstance(a,int) and isinstance(b,int):x=a^b
   env[m[1]]=x;continue
  m=re.match(r's_\w+\s+(s\d+),',line)
  if m:env[m[1]]=None
 assert env.get(phase_last)==('g',1),(phase_first,phase_last,env.get(phase_last))

assert not(R/'ownership-isa-audit.json').exists()
for x in q['eligible']:
 p=pathlib.Path(x['directory']);proof=json.loads((p/'alias-ab-proof/proof.json').read_text());ll=(p/'kernel.llir').read_text();asm=(p/'kernel.amdgcn').read_text();oo=ops(asm);cp=pathlib.Path(x['control_seed_directory']);refasm=(cp/'kernel.amdgcn').read_text()
 for pattern in [r'\bv_mmac_\w+',r'\bmatrix_load_\w+']:assert collections.Counter(re.findall(pattern,asm))==collections.Counter(re.findall(pattern,refasm))
 assert x['config']['hcu_wasp_split_b_slot_turn']==0 and x['config']['hcu_wasp_split_b_slot_turn_wait_mask']==0
 ids=list(map(int,re.findall(r'@llvm\.hcu\.s\.abarrier\.(?:try\.wait|init|seq|arrive|inv)\(i32 (\d+)',ll)));assert set(ids)==set(range(6))
 for o in oo:
  m=re.fullmatch(r's_abarrier_(?:try_wait|arrive)\s+s\d+, (\d+), .+',o) or re.fullmatch(r's_abarrier_(?:seq|inv)\s+(\d+)',o)
  if m:assert int(m[1])<6
 starts=[i for i,o in enumerate(oo) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, 1, .+',o)];assert len(starts)==2;prod=oo[:starts[0]]
 waits=[]
 for i,o in enumerate(prod):
  m=re.fullmatch(r's_abarrier_try_wait\s+s\d+, ([024]), (.+)',o)
  if m:waits.append((i,int(m[1]),m[2]))
 assert sorted(w[1] for w in waits)==[0,0,2,4];aw=[w for w in waits if w[1]==0];bw={w[1]:w for w in waits if w[1] in [2,4]}
 seq={s:next(i for i,o in enumerate(prod) if o==f's_abarrier_seq {s}') for s in [1,3,5]};arr={s:next(i for i,o in enumerate(prod) if re.fullmatch(r's_abarrier_arrive\s+s\d+, '+str(s)+r', 1',o)) for s in [1,3,5]}
 assert aw[0][0]<seq[1]<arr[1]<aw[1][0]<seq[3]<arr[3]<seq[5]<arr[5]
 for bid in proof['overlapped_B_empty_ids']:assert bw[bid][0]<aw[0][0]
 if 4 not in proof['overlapped_B_empty_ids']:assert arr[3]<bw[4][0]<seq[5]
 matrix=[i for i,o in enumerate(prod) if o.startswith('matrix_load_')];assert matrix and all(any(seq[s]<i<arr[s] for s in [1,3,5]) for i in matrix)
 actual_phase_xor(prod,aw[0][0],aw[1][0],aw[0][2],aw[1][2],[w[2] for w in bw.values()])
 # LLVM exact phase recurrence: old empty wait g&1, new current capture wait g&1 xor1.
 first_ready=re.search(r'call i32(?: \(i32, \.\.\.\))? @llvm.hcu.s.abarrier.try.wait\(i32 1,',ll);assert first_ready;pl=ll[:first_ready.start()]
 ww=re.findall(r'call i32(?: \(i32, \.\.\.\))? @llvm.hcu.s.abarrier.try.wait\(i32 ([024]), i32 (%[\w.]+)\)',pl);assert sorted(int(i) for i,_ in ww)==[0,0,2,4]
 ap=[v for i,v in ww if i=='0'];assert phi_phase(pl,ap[0])==ap[1]
 for i,v in ww:
  if i in ['2','4']:phi_phase(pl,v)
 init=sorted(tuple(map(int,m)) for m in re.findall(r'@llvm.hcu.s.abarrier.init\(i32 (\d+), i32 (\d+)\)',ll));assert init==[(0,8),(1,4),(2,8),(3,4),(4,8),(5,4)]
 releases=[]
 for i,o in enumerate(oo):
  m=re.fullmatch(r's_abarrier_arrive\s+s\d+, ([024]), 1',o)
  if m:
   drain=max(j for j in range(i) if oo[j].startswith('s_waitcnt ') and 'lgkmcnt(0)' in oo[j]);ds=max(j for j in range(i) if oo[j].startswith('ds_'));assert ds<drain and not any(re.match(r's_(?:cbranch|branch|setpc)',z) for z in oo[drain+1:i]);releases.append(int(m[1]))
 assert sorted(releases)==[0,0,2,2,4,4]
 # Consumer ready phase recurrences and exact MMAC phase constants.
 rr=re.findall(r'call i32(?: \(i32, \.\.\.\))? @llvm.hcu.s.abarrier.try.wait\(i32 ([135]), i32 (%[\w.]+)\)',ll);assert len(rr)==6
 for ready,v in rr:phi_phase(ll,v)
 seeds=re.findall(r'@llvm.hcu.s.abarrier.arrive\(i32 ([0246]), i32 ([12])\)',pl);assert sorted(seeds)==[('0','2'),('2','2'),('4','2')]
 # Every emitted matrix read stays inside its ready-to-empty ownership
 # window. Off-mask math may interleave reads before the final drain/release;
 # the token remains held until all fragment MMACs and its empty release.
 lr=collections.Counter();role_idx=-1;slot_idx=None
 for line in ll.splitlines():
  m=re.search(r'call i32(?: \(i32, \.\.\.\))? @llvm.hcu.s.abarrier.try.wait\(i32 ([135]),',line)
  if m:
   slot_idx=(int(m[1])-1)//2
   if m[1]=='1':role_idx+=1
  if re.search(r'call <\d+ x half> @llvm.hcu.ds.read.matrix',line):lr[(role_idx,slot_idx)]+=1
 for role_idx,start in enumerate(starts):
  seg=oo[start:starts[role_idx+1] if role_idx+1<len(starts) else len(oo)];windows=[]
  for slot_idx,ready,empty in [(0,1,0),(1,3,2),(2,5,4)]:
   lo=next(i for i,o in enumerate(seg) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, '+str(ready)+r', .+',o));hi=next(i for i,o in enumerate(seg) if re.fullmatch(r's_abarrier_arrive\s+s\d+, '+str(empty)+r', 1',o));assert lo<hi
   actual_reads=sum(o.startswith('ds_read_matrix') for o in seg[lo+1:hi]);assert actual_reads==lr[(role_idx,slot_idx)],(x['tile'],x['layout'],0,role_idx,slot_idx,actual_reads,lr[(role_idx,slot_idx)]);windows.append((lo,hi))
  assert windows[0][1]<windows[1][0] and windows[1][1]<windows[2][0]
  for j,o in enumerate(seg):
   if o.startswith('ds_read_matrix'):assert any(lo<j<hi for lo,hi in windows)
 assert sum(bool(re.fullmatch(r's_ebarrier_sync\s+2, 0',o)) for o in oo)==3
 inv=[int(re.fullmatch(r's_abarrier_inv (\d+)',o)[1]) for o in oo if o.startswith('s_abarrier_inv')];assert sorted(inv)==sorted(list(range(6))*2)
 for i,o in enumerate(oo):
  if o.startswith('s_abarrier_inv') and not oo[i-1].startswith('s_abarrier_inv'):assert re.fullmatch(r's_ebarrier_sync\s+2, 0',oo[i-1])
 assert not any(re.match(r's_ebarrier_sync\s+1,',o) for o in oo)
 rows.append(dict(tile=x['tile'],layout=x['layout'],shared_bytes=x['metadata']['shared'],required_overlapping_B_empty_ids=proof['overlapped_B_empty_ids'],all_overlapped_B_empty_waits_before_Awrite=True,current_A_capture_wait_after_Aready_before_Bwrite=True,LLVM_phase_recurrence_and_actual_SALU_xor1_valid=True,all_A_read_drain_release_before_Bready=True,all_six_data_releases_after_actual_LGKM0=True,no_cross_role_MMAC_turn_instructions=True,data_read_windows_match_finite_projection=True,wait_mask=x['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0),all12join2_before_each_6IDinvalidate=True,producer_exec_barriers6=[o for o in prod if re.match(r's_ebarrier_sync\s+6,',o)],consumer_exec_barriers6=[o for o in oo[starts[0]:] if re.match(r's_ebarrier_sync\s+6,',o)],physical=x['fields'],asm_sha256=sha(p/'kernel.amdgcn'),llir_sha256=sha(p/'kernel.llir')));print(json.dumps(dict(tile=x['tile'],layout=x['layout'],ownership_ISA_passed=True)),flush=True)
assert len(rows)==8 and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());(R/'ownership-isa-audit.json').write_text(json.dumps(dict(passed=True,CPU_only=True,rows=rows,protocol_model_sha256=sha(R/'alias-no-turn-protocol-model.json'),protocol_script_sha256=sha(R/'protocol_alias_no_turn.py'),producer_address_audit_sha256=sha(R/'parent-address-audit.json'),consumer_read_audit_sha256=sha(R/'consumer-read-audit.json'),source_hashes_unchanged=True,native_full_gold_and_liveness_and_performance_still_required=True,script_sha256=sha(pathlib.Path(__file__))),indent=2)+'\n')
