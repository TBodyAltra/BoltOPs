"""Admit only actually realized full-register prefetch, not compiler fallbacks."""
import pathlib,json,re,hashlib
R=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((R/'rect-full-k-prefetch-screen.json').read_text());assert q['default_qualified'] and q['source_hashes_unchanged'] and len(q['results'])==96
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
checks=[];excluded=[]
for x in q['results']:
 p=pathlib.Path(x['directory']);c=x['config'];mode=c['hcu_wasp_mmac_turn'];ab=c['hcu_wasp_merge_ab'];bm,bn,bk=x['tile']
 if x['status']!='COMPILED' or not x.get('zero_all_spills') or not x.get('fits_128k_and_vgpr_budget'):
  excluded.append(dict(directory=str(p),tile=x['tile'],layout=x['layout'],reason=x.get('exception','resource/physicalspill exclusion')));continue
 m=x['metadata'];assert m['hcu_full_BK_local_prefetch'] and m['shared']==98304 and m['num_warps']==12 and m['hcu_partial_k_local_reads']==8 and bk==128
 proof=json.loads((p/'full-k-prefetch-proof/proof.json').read_text());assert proof['no_new_allocations_or_memory_ops'] and proof['producer_unchanged'] and len(proof['consumers'])==2 and all(z['moved_nodes']==6 and z['all_reads_before_empty_release'] and z['empty_release_before_first_dot'] for z in proof['consumers'])
 ir=(p/'kernel.ttgir').read_text();assert ir.count(' = tt.dot ')==4 and ir.count(' = ttg.local_load ')==8 and not re.search(r'ttg.local_alloc\s+%',ir) and 'hcu.consumer_pingpong' not in ir
 assert f'tensorShape = [{bm}, {bk}]' in ir and f'tensorShape = [{bk}, {bn}]' in ir
 ins=[];block=''
 for ln,line in enumerate((p/'kernel.amdgcn').read_text().splitlines(),1):
  if re.match(r'\s*\.Lfunc_end\d+:',line):break
  s=line.split(';',1)[0].strip()
  if re.match(r'\.LBB\w+:',s):block=s.split(':')[0]
  if s and not s.startswith(('.','#')) and not s.endswith(':'):ins.append((ln,s,block))
 expected=(bm*bn*bk)//(8192*4);assert expected==128
 assert sum(s.startswith('v_mmac_') for _,s,_ in ins)==2*expected
 assert all(s.startswith('v_mmac_f32_16x16x16_f16') for _,s,_ in ins if s.startswith('v_mmac_'))
 init=[re.fullmatch(r's_abarrier_init\s+(\d+),\s*(\d+)',s) for _,s,_ in ins if s.startswith('s_abarrier_init')]
 expected_init=([(0,8),(1,4)] if ab else [(0,8),(1,4),(2,8),(3,4)])+([(4,4),(5,4)] if mode else [])
 assert sorted((int(z[1]),int(z[2])) for z in init)==expected_init
 waits=[(i,re.fullmatch(r's_abarrier_try_wait\s+\w+,\s*([45]),\s*(\w+)',s)) for i,(_,s,_) in enumerate(ins) if re.match(r's_abarrier_try_wait\s+\w+,\s*[45],',s)];assert len(waits)==2*mode
 groups=[]
 for i,z in waits:
  out=5 if z[1]=='4' else 4
  j=next(j for j in range(i+1,len(ins)) if re.fullmatch(r's_abarrier_arrive\s+\w+,\s*'+str(out)+r',\s*1',ins[j][1]))
  assert not any(re.match(r's_abarrier_try_wait\s+\w+,\s*[45],',s) for _,s,_ in ins[i+1:j])
  count=sum(s.startswith('v_mmac_') for _,s,_ in ins[i+1:j]);assert count==expected//mode
  groups.append(dict(wait_line=ins[i][0],signal_line=ins[j][0],static_MMAs=count))
 empties=[i for i,(_,s,_) in enumerate(ins) if re.fullmatch(r's_abarrier_arrive\s+\w+,\s*0,\s*1',s)];assert len(empties)==2
 placements=[];realized=True
 for i in empties:
  block=ins[i][2];h=next(j for j in range(i,-1,-1) if j==0 or ins[j-1][2]!=block)
  before=sum(s.startswith('v_mmac_') for _,s,_ in ins[h:i]);w=next(j for j in range(i-1,max(-1,i-150),-1) if ins[j][1]=='s_waitcnt lgkmcnt(0)')
  assert not any(re.match(r'(?:ds_|buffer_load|flat_load|global_load|s_cbranch|s_branch)',s) for _,s,_ in ins[w+1:i])
  t=next(j for j in range(i+1,len(ins)) if ins[j][1].startswith(('s_cbranch','s_branch')))
  after=sum(s.startswith('v_mmac_') for _,s,_ in ins[i+1:t]);postreads=sum(s.startswith('ds_read') for _,s,_ in ins[i+1:t])
  realized &= before==0 and after==expected and postreads==0
  placements.append(dict(LGKM_full_wait_line=ins[w][0],empty_A_line=ins[i][0],MMAs_before_Aempty=before,MMAs_after_Aempty_before_branch=after,DS_reads_after_Aempty_before_branch=postreads))
  if not ab:
   j=next(j for j in range(i+1,min(len(ins),i+60)) if re.fullmatch(r's_abarrier_arrive\s+\w+,\s*2,\s*1',ins[j][1]))
   assert not any(re.match(r'(?:ds_|buffer_load|flat_load|global_load|s_cbranch|s_branch)',s) for _,s,_ in ins[i+1:j]);placements[-1]['empty_B_line']=ins[j][0]
 invs=[i for i,(_,s,_) in enumerate(ins) if re.fullmatch(r's_abarrier_inv\s+(?:4|5)' if mode else r's_abarrier_inv\s+(?:0|1)',s)];assert len(invs)==4;joins=set()
 for i in invs:
  j=next(j for j in range(i-1,-1,-1) if re.fullmatch(r's_ebarrier_sync\s+2(?:,\s*(?:0|12))?',ins[j][1]))
  assert not any(re.match(r'(?:s_abarrier_try_wait|v_mmac_|ds_|s_cbranch|s_branch)',s) for _,s,_ in ins[j+1:i]);joins.add(ins[j][0])
 assert len(joins)==2
 record=dict(directory=str(p),tile=x['tile'],layout=x['layout'],mode=mode,merged=ab,config=c,fields=x['fields'],shared=m['shared'],actual_vgprs=m['wasp_partition_regs_actual'],groups=groups,placements=placements,join_lines=sorted(joins),full_prefetch_realized=bool(realized),asm_sha256=sha(p/'kernel.amdgcn'),proof_sha256=sha(p/'full-k-prefetch-proof/proof.json'))
 if realized:checks.append(record)
 else:excluded.append(dict(record,reason='Backend moved MMAC/data reads across requested full-register prefetch boundary; not admitted as this pipeline.'))
assert checks and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
out=dict(passed=True,CPU_only=True,native_full_gold_and_liveness_still_required=True,admitted=len(checks),excluded=len(excluded),screen_sha256=sha(R/'rect-full-k-prefetch-screen.json'),rows=checks,exclusions=excluded,audit_source_sha256=sha(pathlib.Path(__file__)))
(R/'rect-prefetch-static-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,admitted=len(checks),excluded=len(excluded),modes={str(k):sum(x['mode']==k for x in checks) for k in [0,1,2]})))
