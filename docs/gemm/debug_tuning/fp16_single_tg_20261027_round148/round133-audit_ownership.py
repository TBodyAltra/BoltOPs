"""Actual BM64 data ownership/phase checks, binding existing protocol classes."""
import pathlib,json,hashlib,re,collections
from ownership_helpers import ops,phi_phase,actual_phase_xor
R=pathlib.Path(__file__).resolve().parent;B=R.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((R/'cpu-screen.json').read_text());assert len(q['eligible'])==7 and not(R/'ownership-isa-audit.json').exists()
for name in ['parent-address-audit.json','consumer-read-audit.json']:assert json.loads((R/name).read_text())['passed']
models={
 'oneA_full_B':(B/'single-tg-pipeline-20261026-round89-one-operand-no-turn-cpu','protocol-proof.json','check_protocol.py','source_sha256'),
 'split_B_slots':(B/'single-tg-pipeline-20261027-round101-slot-turns-cpu','protocol-model.json','protocol_model.py','script_sha256'),
 'alias_B_slots':(B/'single-tg-pipeline-20261027-round115-alias-ab-slots-cpu-fix','protocol-model-v3.json','protocol_model_v3.py','script_sha256'),
 'alias_early_B1':(B/'single-tg-pipeline-20261027-round129-alias-b1-early-cpu-driver','protocol-model.json','protocol_model.py','script_sha256')}
modelproofs={}
for variant,(root,name,script,key) in models.items():
 m=json.loads((root/name).read_text());assert m['passed'] and m[key]==sha(root/script)
 modelproofs[variant]=dict(proof=str(root/name),proof_sha256=sha(root/name),script=str(root/script),script_sha256=sha(root/script))
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());rows=[]
for x in q['eligible']:
 p=pathlib.Path(x['directory']);variant=x['variant'];full=variant=='oneA_full_B';alias=x['config'].get('hcu_wasp_alias_ab_slots',False);early=x['config'].get('hcu_wasp_alias_b1_early',False);assert not x['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0)
 ll=(p/'kernel.llir').read_text();asm=(p/'kernel.amdgcn').read_text();oo=ops(asm);ref=next(z for z in q['eligible'] if z['layout']==x['layout'] and z['variant']=='oneA_full_B');rasm=(pathlib.Path(ref['directory'])/'kernel.amdgcn').read_text()
 for pat in [r'\bv_mmac_\w+',r'\bmatrix_load_\w+']:assert collections.Counter(re.findall(pat,asm))==collections.Counter(re.findall(pat,rasm))
 starts=[i for i,o in enumerate(oo) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, 1, .+',o)];assert len(starts)==2;prod=oo[:starts[0]];waits=[]
 for i,o in enumerate(prod):
  m=re.fullmatch(r's_abarrier_try_wait\s+s\d+, ([024]), (.+)',o)
  if m:waits.append((i,int(m[1]),m[2]))
 aw=[w for w in waits if w[1]==0];bw={w[1]:w for w in waits if w[1] in [2,4]};ids=[1,3] if full else [1,3,5]
 seq={s:next(i for i,o in enumerate(prod) if o==f's_abarrier_seq {s}') for s in ids};arr={s:next(i for i,o in enumerate(prod) if re.fullmatch(r's_abarrier_arrive\s+s\d+, '+str(s)+r', 1',o)) for s in ids}
 if full:
  assert sorted(w[1] for w in waits)==[0,2] and aw[0][0]<seq[1]<arr[1]<bw[2][0]<seq[3]<arr[3]
 elif not alias:
  assert sorted(w[1] for w in waits)==[0,2,4] and aw[0][0]<seq[1]<arr[1]<bw[2][0]<seq[3]<arr[3]<bw[4][0]<seq[5]<arr[5]
 else:
  assert sorted(w[1] for w in waits)==[0,0,2,4]
  ap=json.loads((p/'alias-ab-proof/proof.json').read_text());required=[2] if x['layout']=='TT' else [2,4];assert ap['overlapped_B_empty_ids']==required
  for k in required:assert bw[k][0]<aw[0][0]
  if early:assert x['layout']=='TT' and aw[0][0]<seq[1]<arr[1]<bw[4][0]<seq[5]<arr[5]<aw[1][0]<seq[3]<arr[3]
  else:
   assert aw[0][0]<seq[1]<arr[1]<aw[1][0]<seq[3]<arr[3]<seq[5]<arr[5]
   if 4 not in required:assert arr[3]<bw[4][0]<seq[5]
  actual_phase_xor(prod,aw[0][0],aw[1][0],aw[0][2],aw[1][2],[w[2] for w in bw.values()])
 matrix=[i for i,o in enumerate(prod) if o.startswith('matrix_load_')];assert matrix and all(any(seq[s]<i<arr[s] for s in ids) for i in matrix)
 first_ready=re.search(r'call i32(?: \(i32, \.\.\.\))? @llvm.hcu.s.abarrier.try.wait\(i32 1,',ll);assert first_ready;pl=ll[:first_ready.start()]
 ww=re.findall(r'call i32(?: \(i32, \.\.\.\))? @llvm.hcu.s.abarrier.try.wait\(i32 ([024]), i32 (%[\w.]+)\)',pl)
 assert sorted(int(i) for i,_ in ww)==sorted(w[1] for w in waits)
 ap=[v for i,v in ww if i=='0'];back=phi_phase(pl,ap[0])
 if alias:assert back==ap[1]
 for i,v in ww:
  if i in ['2','4']:phi_phase(pl,v)
 init=sorted(tuple(map(int,m)) for m in re.findall(r'@llvm.hcu.s.abarrier.init\(i32 (\d+), i32 (\d+)\)',ll));expected=[(0,8),(1,4),(2,8),(3,4)]+([] if full else [(4,8),(5,4),(6,4),(7,4)]);assert init==expected
 releases=[]
 for i,o in enumerate(oo):
  m=re.fullmatch(r's_abarrier_arrive\s+s\d+, ([024]), 1',o)
  if m:
   drain=max(j for j in range(i) if oo[j].startswith('s_waitcnt ') and 'lgkmcnt(0)' in oo[j]);ds=max(j for j in range(i) if oo[j].startswith('ds_'));assert ds<drain and not any(re.match(r's_(?:cbranch|branch|setpc)',z) for z in oo[drain+1:i]);releases.append(int(m[1]))
 assert sorted(releases)==([0,0,2,2] if full else [0,0,2,2,4,4])
 for role,start in enumerate(starts):
  seg=oo[start:starts[role+1] if role+1<len(starts) else len(oo)];ar=next(i for i,o in enumerate(seg) if re.fullmatch(r's_abarrier_arrive\s+s\d+, 0, 1',o));b0=next(i for i,o in enumerate(seg) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, 3, .+',o));assert ar<b0 and not any(o.startswith('ds_') for o in seg[ar+1:b0]);mm=[i for i,o in enumerate(seg) if o.startswith('v_mmac_')];assert mm and b0<mm[0]
  if full:
   br=next(i for i,o in enumerate(seg) if re.fullmatch(r's_abarrier_arrive\s+s\d+, 2, 1',o));assert not any(o.startswith('ds_read_matrix') for o in seg[br+1:mm[-1]+1]);assert not any(re.fullmatch(r's_abarrier_try_wait\s+s\d+, [67], .+',o) for o in seg)
  else:
   inp,out=(6,7) if role==0 else (7,6);tw=[i for i,o in enumerate(seg) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, '+str(inp)+r', .+',o)];ts=[i for i,o in enumerate(seg) if re.fullmatch(r's_abarrier_arrive\s+s\d+, '+str(out)+r', 1',o)];b1=next(i for i,o in enumerate(seg) if re.fullmatch(r's_abarrier_try_wait\s+s\d+, 5, .+',o));assert len(tw)==len(ts)==2 and len(mm)%2==0
   assert ar<tw[0]<b0<mm[0] and sum(i<ts[0] for i in mm)==len(mm)//2 and ts[0]<tw[1]<b1<mm[len(mm)//2] and mm[-1]<ts[-1]
 assert sum(bool(re.fullmatch(r's_ebarrier_sync\s+2, 0',o)) for o in oo)==3
 inv=[int(re.fullmatch(r's_abarrier_inv (\d+)',o)[1]) for o in oo if o.startswith('s_abarrier_inv')];assert sorted(inv)==sorted(list(range(4 if full else 8))*2)
 for i,o in enumerate(oo):
  if o.startswith('s_abarrier_inv') and not oo[i-1].startswith('s_abarrier_inv'):assert re.fullmatch(r's_ebarrier_sync\s+2, 0',oo[i-1])
 assert not any(re.match(r's_ebarrier_sync\s+1,',o) for o in oo)
 rows.append(dict(tile=x['tile'],layout=x['layout'],variant=variant,protocol=modelproofs[variant],actual_producer_wait_copy_ready_order_valid=True,actual_LLVM_phi_phase_recurrence_valid=True,alias_current_A_capture_phase_xor1_valid=alias,all_data_empty_releases_after_actual_DS_drain=True,all12join2_before_signal_reclamation=True,physical=x['fields'],producer_exec_barriers6=[o for o in prod if re.match(r's_ebarrier_sync\s+6,',o)],asm_sha256=sha(p/'kernel.amdgcn'),llir_sha256=sha(p/'kernel.llir')))
 print(json.dumps(dict(layout=x['layout'],variant=variant,ownership_ISA_passed=True)),flush=True)
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
(R/'ownership-isa-audit.json').write_text(json.dumps(dict(passed=True,CPU_only=True,rows=rows,protocol_sources=modelproofs,source_hashes_unchanged=True,producer_address_audit_sha256=sha(R/'parent-address-audit.json'),consumer_read_audit_sha256=sha(R/'consumer-read-audit.json'),scope='Actual LLVM/ISA ownership, phases, read drains and synchronization for7BM64 variants; shape-independent existing finite protocol classes bound separately. Full-B class uses original aggregate-four-wave model89; additional individual-wave model remains separate. Native full-output precision/liveness/performance still required.',helpers_sha256=sha(R/'ownership_helpers.py'),script_sha256=sha(pathlib.Path(__file__))),indent=2)+'\n')
