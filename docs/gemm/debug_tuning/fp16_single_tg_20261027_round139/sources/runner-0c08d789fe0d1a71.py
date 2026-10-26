"""Check real producer channels, source ownership order and actual ISA release points."""
import pathlib,json,re,hashlib,sys
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));from data_barrier_map import derive
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((R/'derived-barrier-map-screen.json').read_text());assert q['default_qualified'] and q['source_hashes_unchanged'] and len(q['results'])==56
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
rows=[];exclusions=[]
for x in q['results']:
 p=pathlib.Path(x['directory']);c=x['config'];one=c.get('hcu_wasp_prefetch_one_operand',0);mode=c.get('hcu_wasp_mmac_turn',0);bm,bn,bk=x['tile'];assert not c.get('hcu_wasp_merge_ab')
 if x['status']!='COMPILED' or not x.get('zero_all_spills') or not x.get('fits_128k_and_vgpr_budget'):
  exclusions.append(dict(directory=str(p),tile=x['tile'],layout=x['layout'],config=c,reason=x.get('exception','resource/physicalspill exclusion')));continue
 m=x['metadata'];assert m['shared']==(131072 if one else 98304) and m['num_warps']==12 and bk==128
 ir=(p/'kernel.ttgir').read_text();mapping,producer_proof=derive(ir);assert m['hcu_operand_barriers']==mapping
 proofdir='one-operand-prefetch-proof' if one else 'first-operand-early-proof';proof=json.loads((p/proofdir/'proof.json').read_text());assert proof['operand_barriers']==mapping
 first='AB'.index(proof['prefetched_operand'] if one else proof['first_operand']);other=1-first;pe=mapping['AB'[first]]['empty'];pr=mapping['AB'[first]]['ready'];oe=mapping['AB'[other]]['empty'];orr=mapping['AB'[other]]['ready']
 assert m['hcu_partial_k_local_reads']==8 and ir.count(' = tt.dot ')==4 and ir.count(' = ttg.local_load ')==8 and 'hcu.consumer_pingpong' not in ir and not re.search(r'ttg.local_alloc\s+%',ir)
 if one:assert mode in (1,2) and m['hcu_prefetch_one_operand']=='AB'[first] and m['hcu_turn_wait_before_streamed_ready'] is True
 else:assert c.get('hcu_wasp_first_operand_early') and m['hcu_first_operand_early_capture']=='AB'[first] and m['hcu_full_BK_local_prefetch'] is True
 # Independently bind each local read to the producer's actual ready/empty IDs.
 marks=list(re.finditer(r'^    partition([012])\(',ir,re.M));prefix=ir[:marks[0].start()]
 for part in [ir[marks[1].start():marks[2].start()],ir[marks[2].start():]]:
  vals={z[1]:int(z[2]) for z in re.finditer(r'(%[\w.]+) = arith.constant (\d+) : i32',prefix+part)}
  for z in re.finditer(r'(%[\w.]+) = llvm.mlir.constant\((\d+) : i32\)',prefix+part):vals[z[1]]=int(z[2])
  ls=part.splitlines();wait={};release={};loads={0:[],1:[]}
  for i,l in enumerate(ls):
   z=re.search(r'abarrier.try.wait i32, (%[\w.]+),',l)
   if z and vals[z[1]]<4:wait[vals[z[1]]]=i
   z=re.search(r'abarrier.arrive i32, (%[\w.]+), 1',l)
   if z and vals[z[1]]<4:release[vals[z[1]]]=i
   if ' = ttg.local_load ' in l:loads[int(re.search(r'dot_op<\{opIdx = ([01])',l)[1])].append(i)
  for op in [0,1]:assert len(loads[op])==2 and wait[mapping['AB'[op]]['ready']]<min(loads[op])<=max(loads[op])<release[mapping['AB'[op]]['empty']]
  assert release[pe]<wait[orr]
 ins=[];block=''
 for ln,l in enumerate((p/'kernel.amdgcn').read_text().splitlines(),1):
  if re.match(r'\s*\.Lfunc_end\d+:',l):break
  s=l.split(';',1)[0].strip()
  if re.match(r'\.LBB\w+:',s):block=s.split(':')[0]
  if s and not s.startswith(('.','#')) and not s.endswith(':'):ins.append((ln,s,block))
 expected=bm*bn*bk//(8192*4);assert expected in (128,256)
 assert sum(s.startswith('v_mmac_') for _,s,_ in ins)==2*expected and all(s.startswith('v_mmac_f32_16x16x16_f16') for _,s,_ in ins if s.startswith('v_mmac_'))
 init=[re.fullmatch(r's_abarrier_init\s+(\d+),\s*(\d+)',s) for _,s,_ in ins if s.startswith('s_abarrier_init')];assert sorted((int(z[1]),int(z[2])) for z in init)==[(0,8),(1,4),(2,8),(3,4)]+([(4,4),(5,4)] if mode else [])
 tw=[(i,re.fullmatch(r's_abarrier_try_wait\s+\w+,\s*([45]),\s*\w+',s)) for i,(_,s,_) in enumerate(ins) if re.match(r's_abarrier_try_wait\s+\w+,\s*[45],',s)];assert len(tw)==2*mode;groups=[]
 for i,z in tw:
  out=5 if z[1]=='4' else 4;j=next(j for j in range(i+1,len(ins)) if re.fullmatch(r's_abarrier_arrive\s+\w+,\s*'+str(out)+r',\s*1',ins[j][1]))
  assert not any(re.match(r's_abarrier_try_wait\s+\w+,\s*[45],',s) for _,s,_ in ins[i+1:j]);n=sum(s.startswith('v_mmac_') for _,s,_ in ins[i+1:j]);assert n==expected//mode;groups.append(dict(wait_line=ins[i][0],signal_line=ins[j][0],MMAs=n))
 arrival=lambda s,i:bool(re.fullmatch(r's_abarrier_arrive\s+\w+,\s*'+str(i)+r',\s*1',s));ready=lambda s,i:bool(re.fullmatch(r's_abarrier_try_wait\s+\w+,\s*'+str(i)+r',\s*\w+',s))
 empties=[i for i,(_,s,_) in enumerate(ins) if arrival(s,pe)];assert len(empties)==2;placements=[];realized=True
 for i in empties:
  b=ins[i][2];h=next(j for j in range(i,-1,-1) if j==0 or ins[j-1][2]!=b);t=next(j for j in range(i+1,len(ins)) if ins[j][1].startswith(('s_cbranch','s_branch')))
  fw=next(j for j in range(i-1,h-1,-1) if ready(ins[j][1],pr));ow=next(j for j in range(h,t) if ready(ins[j][1],orr));end=next(j for j in range(i+1,t) if arrival(ins[j][1],oe));drains=[]
  for e in [i,end]:
   w=next(j for j in range(e-1,h-1,-1) if ins[j][1]=='s_waitcnt lgkmcnt(0)');assert not any(re.match(r'(?:ds_|buffer_load|flat_load|global_load|s_cbranch|s_branch)',s) for _,s,_ in ins[w+1:e]);drains.append(ins[w][0])
  early=fw<i<ow<end and any(s.startswith('ds_read') for _,s,_ in ins[fw+1:i]) and any(s.startswith('ds_read') for _,s,_ in ins[ow+1:end]) and not any(s.startswith(('ds_read','v_mmac_')) for _,s,_ in ins[i+1:ow])
  pre=sum(s.startswith('v_mmac_') for _,s,_ in ins[h:i]);total=sum(s.startswith('v_mmac_') for _,s,_ in ins[i+1:t]);post=sum(s.startswith('ds_read') for _,s,_ in ins[end+1:t]);before_other=sum(s.startswith('v_mmac_') for _,s,_ in ins[i+1:end]);actualturn=[j for j,z in tw if h<=j<t]
  if one:early &= bool(actualturn and i<actualturn[0]<ow)
  realized &= early and pre==0 and total==expected and post==0
  placements.append(dict(prefetched_operand='AB'[first],prefetched_ready_id=pr,prefetched_empty_id=pe,other_ready_id=orr,other_empty_id=oe,prefetched_ready_line=ins[fw][0],prefetched_empty_line=ins[i][0],other_ready_line=ins[ow][0],other_empty_line=ins[end][0],LGKM_full_wait_lines=drains,MMAs_before_other_empty=before_other,other_operand_fully_captured_before_math=before_other==0,actual_turn_wait_lines=[ins[j][0] for j in actualturn],source_and_actual_channel_order_correct=early))
 invs=[i for i,(_,s,_) in enumerate(ins) if re.fullmatch(r's_abarrier_inv\s+(?:4|5)' if mode else r's_abarrier_inv\s+(?:0|1)',s)];assert len(invs)==4;joins=set()
 for i in invs:
  j=next(j for j in range(i-1,-1,-1) if re.fullmatch(r's_ebarrier_sync\s+2(?:,\s*(?:0|12))?',ins[j][1]));assert not any(re.match(r'(?:s_abarrier_try_wait|v_mmac_|ds_|s_cbranch|s_branch)',s) for _,s,_ in ins[j+1:i]);joins.add(ins[j][0])
 assert len(joins)==2
 row=dict(directory=str(p),tile=x['tile'],layout=x['layout'],config=c,kind='one_operand_stream' if one else 'first_operand_capture',operand_barriers=mapping,prefetched_operand='AB'[first],fields=x['fields'],shared=m['shared'],actual_vgprs=m['wasp_partition_regs_actual'],source_ownership_order_valid=True,actual_binding_and_early_capture_realized=bool(realized),placements=placements,groups=groups,join_lines=sorted(joins),asm_sha256=sha(p/'kernel.amdgcn'),proof_sha256=sha(p/proofdir/'proof.json'))
 if realized:rows.append(row)
 else:exclusions.append(dict(row,reason='Actual code does not realize the producer-derived binding/capture boundary; not admitted.'))
assert rows and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
out=dict(passed=True,CPU_only=True,native_full_gold_and_liveness_still_required=True,screen_sha256=sha(R/'derived-barrier-map-screen.json'),admitted=len(rows),excluded=len(exclusions),rows=rows,exclusions=exclusions,audit_source_sha256=sha(pathlib.Path(__file__)))
(R/'derived-bindings-static-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,admitted=len(rows),excluded=len(exclusions),one_operand=sum(x['kind']=='one_operand_stream' for x in rows),full_first=sum(x['kind']=='first_operand_capture' for x in rows))))
