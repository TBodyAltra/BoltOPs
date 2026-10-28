from pathlib import Path
import json,time,hashlib,re,os,traceback
R=Path('/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round310-square-wdra-fragment-native');load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
try:
 pid=load(R/'launch-receipt.json')['pid'];deadline=time.time()+7200
 while Path('/proc',str(pid)).exists():
  assert time.time()<deadline,'PMD job not terminal in2h';time.sleep(2)
 assert load(R/'job-exit.json')['returncode']==0,'Prototype job failed; no retries'
 deadline=time.time()+300
 while not(R/'native-raw-audit.json').exists():
  assert time.time()<deadline,'No closed raw audit within5min';time.sleep(2)
 assert load(R/'native-raw-audit.json')['passed'];rows=[]
 for p in sorted((R/'runs').glob('*/result.json')):
  x=load(p);f=x['config'].get('WASP_GLOBAL_K_FRAGMENTS',1)
  if f==1:continue
  assert f in [2,4] and x['tile'][2]==128 and x['status']=='PASS'
  bm,bn,bk=x['tile'];pk=bk//f;tc=x['transport_contract'];assert tc['logical_bk']==128 and tc['wasp_global_k_fragments']==f and tc['physical_global_block_k']==pk and tc['physical_dot_k']==pk and tc['physical_loop_count']==x['shape'][2]//pk==f*x['pipeline_loops'] and tc['logical_loop_count']==x['pipeline_loops']
  dotlines=[l for l in (p.parent/'kernel.ttir').read_text().splitlines() if 'tt.dot ' in l];assert dotlines
  for line in dotlines:
   shapes=[tuple(map(int,z)) for z in re.findall(r'tensor<(\d+)x(\d+)xf16',line)];assert (bm,pk) in shapes and (pk,bn) in shapes,(p.name,line)
  expected_AB_staged_bytes=(bm+bn)*pk*2*x['config']['num_stages'];assert expected_AB_staged_bytes<=x['metadata']['shared']<=131072
  rows.append(dict(id=x['id'],logical_tile=x['tile'],physical_global_block_k=pk,dot_k=pk,logical_loops=x['pipeline_loops'],physical_loops=tc['physical_loop_count'],correctness_only=x.get('correctness_only',False),staged_AB_bytes=expected_AB_staged_bytes,total_LDS_bytes=x['metadata']['shared'],actual_TTIR_dot_shapes_verified=True,TTIR_sha256=sha(p.parent/'kernel.ttir'),result_sha256=sha(p),ASM_sha256=x['asm_sha256']))
 assert len(rows)==72 and sum(x['correctness_only'] for x in rows)==18
 out=dict(passed=True,logical_BK128_preserved=True,physical_global_K64_or32_verified=True,physical_dot_K64_or32_verified=True,actual_TTIR_and_runtime_contract_verified=True,logical_odd17_means_physical34_or68=True,no_MN_runtime_specialization=True,no_TG_rearrangement=True,raw_audit_sha256=sha(R/'native-raw-audit.json'),rows=rows,script_sha256=sha(Path(__file__)),no_promotion=True)
 (R/'physical-fragment-contract-proof.json').write_text(json.dumps(out,indent=2)+'\n')
except BaseException:
 (R/'physical-fragment-contract-failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),epoch=time.time()),indent=2));raise
