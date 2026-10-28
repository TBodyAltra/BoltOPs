from pathlib import Path
import json,hashlib,re,statistics,importlib.util
R=Path(__file__).resolve().parent;P=next(R.parent.glob('*round321-*'));load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sp=importlib.util.spec_from_file_location('statsio',R.parent/'single-tg-pipeline-20261021/stats_io.py');reader=importlib.util.module_from_spec(sp);sp.loader.exec_module(reader)
rows=[]
for p in sorted((P/'runs').glob('*/result.json')):
 x=load(p)
 if x.get('status')!='PASS' or x.get('correctness_only') or x['tile']!=[64,64,128] or x['layout']!='TN' or x['pipeline_loops'] not in [16,64]:continue
 if x['config']['wasp_partition_warps'] not in [[1,4],[2,4]]:continue
 assert x['config']['LOOP_UNROLL_FACTOR']==1 and x['config']['WASP_GLOBAL_K_FRAGMENTS'] in [2,4] and x['accuracy']['mismatches']==x['accuracy']['nonfinite']==0
 assert x['asm_sha256']==sha(p.parent/'kernel.amdgcn') and all(d['instructions']['MMOP']==2*64*64*128*x['pipeline_loops']/8192 for d in x['dispatches'])
 dispatches=[]
 for d in x['dispatches'][1:]:
  path=Path(d['stats'])
  try:raw=reader.read_stats(path)
  except FileNotFoundError:raw=reader.read_stats(path)
  values={}
  for line in raw.decode().splitlines():
   parts=line.split()
   if len(parts)<2:continue
   name=parts[0]
   if name in ['system.shaders.CUs00.numCyclesWithNoIssue','system.shaders.CUs00.numBranchTaken','system.shaders.CUs00.numCyclesWaitIFetchAfterBranchTaken'] or re.fullmatch(r'system\.shaders\.CUs00\.SIMD[0-3]\.(waitLgkmCounter|barrierCounter|depCounter)',name):values[name]=float(parts[1])
  assert len(values)==15
  dispatches.append(dict(stats=str(path),raw_stats_sha256=hashlib.sha256(raw).hexdigest(),counters=values))
 rows.append(dict(id=x['id'],tile=x['tile'],layout=x['layout'],global_fragments=x['config']['WASP_GLOBAL_K_FRAGMENTS'],partition_warps=x['config']['wasp_partition_warps'],logical_loops=x['pipeline_loops'],physical_K_fragments=x['transport_contract']['physical_loop_count'],median_cycles=x['median_cycles'],ASM_sha256=x['asm_sha256'],result_sha256=sha(p),MMOP_work_ratio=1,dispatches=dispatches))
assert len(rows)==8
pairs=[]
for producer in [1,2]:
 for f in [2,4]:
  pair=sorted([x for x in rows if x['partition_warps'][0]==producer and x['global_fragments']==f],key=lambda x:x['logical_loops']);assert len(pair)==2 and pair[0]['logical_loops']==16 and pair[1]['logical_loops']==64 and pair[0]['ASM_sha256']==pair[1]['ASM_sha256']
  names=pair[0]['dispatches'][0]['counters'];deltas={name:(statistics.median(d['counters'][name] for d in pair[1]['dispatches'])-statistics.median(d['counters'][name] for d in pair[0]['dispatches']))/48 for name in names}
  pairs.append(dict(producer_warps=producer,global_fragments=f,source_ids=[x['id'] for x in pair],counter_slope_per_logical_BK128=deltas,cycles_slope_preliminary=(pair[1]['median_cycles']-pair[0]['median_cycles'])/48))
out=dict(passed=True,diagnostic_only=True,points=8,dispatches=24,initial_root=str(P),rows=rows,pairs=pairs,interpretation='Counters motivate UF2/4 experiments only. IFetch/LGKM/barrier counters are wave/qcycles,overlap and do not establish critical-path CU cost;numCyclesWithNoIssue is issue-stage inactivity,not MMAC execution idle. Two-point slopes are preliminary;32-loop residual and independent fresh still required. No ranking or causal claim.',source_kernel_support='Frozen297 already forwards LOOP_UNROLL_FACTOR to tl.range;no kernel/backend source edit required. Existing worker global-fragment guard remainsUF1 until exact candidate CPU admission and later full work/GOLD/odd17 gates.',script_sha256=sha(Path(__file__)),no_map_promotion=True)
(R/'control-overhead-evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,points=8,pairs=4,preliminary_only=True)))
