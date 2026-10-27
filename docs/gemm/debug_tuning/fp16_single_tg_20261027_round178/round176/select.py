"""Select once: performance requires3%; strict-zero resource alternatives retain98%."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(R/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
assert not (R/'selection-audit.json').exists()
a=load(R/'native-raw-audit.json');assert a['passed'] and a['points']==26 and a['raw_dispatches']==104
q=load(R/'pipeline-candidates.json');ref=load(R/'reference-map.json');analysis=load(R/'analysis/pipeline-map.json');assert len(q['candidates'])==len(analysis['candidates'])==8
rows=[];perf=[];resources=[]
for c in q['candidates']:
    old=next(x for x in ref['best_64k'] if x['tile']==c['tile'] and x['layout']==c['layout'])
    ratio=c['mmac_theoretical_fraction']/old['mmac_theoretical_fraction']
    zero=all(c['physical'][k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count'])
    rows.append(dict(config=c['config'],fraction=c['mmac_theoretical_fraction'],residual=c['intermediate_residual_fraction_of_long_time'],steady=c['steady'],physical=c['physical'],relative_to_64KiB_best=ratio))
    if c['steady'] and ratio>=1.03:perf.append(c)
    if c['steady'] and zero and ratio>=.98:resources.append(dict(candidate=c,baseline=old,relative_to_64KiB_best=ratio))
assert perf==[] and len(resources)==2
nominee=max(resources,key=lambda x:x['candidate']['mmac_theoretical_fraction'])
assert nominee['candidate']['config']['LOOP_UNROLL_FACTOR']==2 and nominee['candidate']['config']['optimize_epilogue'] is False
out=dict(passed=True,raw_audit_sha256=sha(R/'native-raw-audit.json'),candidates_sha256=sha(R/'pipeline-candidates.json'),analysis_sha256=sha(R/'analysis/pipeline-map.json'),rows=rows,performance_nominees=0,strict_zero_resource_eligible=2,resource_alternative_nominees=1,resource_alternative_only=True,nominee=nominee,policy='One fastest initial steady physical-zero recipe at>=98% of64KiB best; one independent16/32/64 series required. No repeat of rejected/nonsteady recipes and no performance-best promotion.',map_unchanged=True)
(R/'selection-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=True,performance_nominees=0,resource_alternative_nominees=1,nominee_fraction=nominee['candidate']['mmac_theoretical_fraction'],ratio=nominee['relative_to_64KiB_best'],UF=2,optimize_epilogue=False,physical_zero=True)))
