"""Record the rejected cohort and the verified operand/channel mismatch."""
import pathlib,json,re,hashlib,sys,subprocess
R=pathlib.Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'single-tg-pipeline-20261026-round87-derived-barrier-map-cpu'))
from data_barrier_map import derive
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def code(p):
 out=[]
 for line in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',line):break
  line=line.split(';',1)[0].strip()
  if line and not line.startswith(('.file','.loc','.Ltmp','.cfi')):out.append(line)
 return hashlib.sha256('\n'.join(out).encode()).hexdigest()
assert json.loads((R/'job-exit.json').read_text())['returncode']==1 and json.loads((R/'state.json').read_text())['phase']=='LIVENESS_PILOT_REJECTED'
act=json.loads((R/'active-manifest.json').read_text());mp=R/('manifest-'+act['manifest_id']+'.json');man=json.loads(mp.read_text());assert all(sha(pathlib.Path(p))==h for p,h in man['hashes'].items())
refs={x['source_id']:pathlib.Path(x['CPU_reference']) for x in json.loads((R/'liveness-pilot-qualification.json').read_text())['checks']}
rows=[];controls=[];pids=set()
for p in sorted((R/'runs').glob('*/result.json')):
 q=json.loads(p.read_text());ex=json.loads((p.parent/'executor.json').read_text());assert ex['child_pid'] not in pids;pids.add(ex['child_pid'])
 if not q.get('config',{}).get('hcu_wasp_prefetch_one_operand'):
  assert q['status']=='PASS';controls.append(p.parent.name);continue
 assert q['config']['hcu_wasp_prefetch_one_operand']==1 and q['accuracy']['nonfinite']==0 and q['n_spills']==0
 cp=refs[p.parent.name];cpu=json.loads((cp/'result.json').read_text());assert code(p.parent/'kernel.amdgcn')==code(cp/'kernel.amdgcn') and q['metadata']['shared']==cpu['metadata']['shared'] and q['metadata']['wasp_partition_regs_actual']==cpu['metadata']['wasp_partition_regs_actual']
 ir=(p.parent/'kernel.ttgir').read_text();mapping,proof=derive(ir);markers=list(re.finditer(r'^    partition([012])\(',ir,re.M));details=[]
 prefix=ir[:markers[0].start()]
 for role,part in [(1,ir[markers[1].start():markers[2].start()]),(2,ir[markers[2].start():])]:
  vals={m[1]:int(m[2]) for m in re.finditer(r'(%[\w.]+) = arith.constant (\d+) : i32',prefix+part)}
  for m in re.finditer(r'(%[\w.]+) = llvm.mlir.constant\((\d+) : i32\)',prefix+part):vals[m[1]]=int(m[2])
  ls=part.splitlines();loadA=[i for i,l in enumerate(ls) if ' = ttg.local_load ' in l and 'opIdx = 0,' in l];assert len(loadA)==2
  wa=next(i for i,l in enumerate(ls) if (m:=re.search(r'abarrier.try.wait i32, (%[\w.]+),',l)) and vals[m[1]]==mapping['A']['ready'])
  details.append(dict(consumer=role,A_load_lines=[i+1 for i in loadA],actual_A_ready_wait_line=wa+1,reads_A_before_actual_A_ready=any(i<wa for i in loadA)))
 bfirst=q['config'].get('LOAD_B_FIRST',False);assert all(x['reads_A_before_actual_A_ready']==bfirst for x in details)
 if bfirst:
  assert q['status']=='ACCURACY_FAIL' and q['accuracy']['mismatches']>65000 and (p.parent/'precision-failure.pt').is_file()
 else:assert q['status']=='PASS' and q['accuracy']['mismatches']==0
 rows.append(dict(id=p.parent.name,layout=q['layout'],B_first=bfirst,turn=q['config']['hcu_wasp_mmac_turn'],status=q['status'],accuracy=q['accuracy'],config=q['config'],actual_operand_barriers=mapping,consumer_checks=details,CPU_native_ISA_and_resources_equal=True,PID=ex['child_pid'],result_sha256=sha(p),asm_sha256=sha(p.parent/'kernel.amdgcn'),failure_payload_sha256=sha(p.parent/'precision-failure.pt') if bfirst else None))
assert len(controls)==8 and len(rows)==8 and len(pids)==16 and sum(x['status']=='ACCURACY_FAIL' for x in rows)==4
head=B/'single-tg-continuation-20261023/published/natural-stride-map.json';m=json.loads(head.read_text());choices=[x for k in ['best','best_no_spill','best_64k','best_64k_no_spill'] for x in m[k]]
assert sha(head)==sha(R/'reference-map.json')=='2a53480f78a9f5c81e1bef2aada9a8254566f67f6d22b1c9c7d070f79229c640'
assert not any(x['config'].get('hcu_wasp_prefetch_one_operand') for x in choices)
assert all(not x['config'].get('LOAD_B_FIRST',False) for x in choices if x['config'].get('hcu_wasp_first_operand_early'))
out=dict(status='REJECTED_COHORT_AUDITED',source_hashes_unchanged=True,controls=controls,rows=rows,canonical_unchanged=True,canonical_sha256=sha(head),all_published_first_early_choices_A_first=True,root_cause='Data channels follow producer allocation order. Legacy A->0/1,B->2/3 binding reads A before Aready and returns Bempty early under B-first. Derive bindings from producer MLS/wait/seq/ready and validate consumer shared roots.',scope='Only16 short points; no long-loop/fresh promotion.4B-first precision failures,4A-first pilot passes; no nonfinite or reset.',manifest_sha256=sha(mp),audit_source_sha256=sha(pathlib.Path(__file__)))
(R/'failure-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],controls=8,pilot_pass=4,pilot_accuracy_fail=4,CPU_native_equal=True,canonical_unchanged=True)))
