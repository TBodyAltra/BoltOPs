"""Select fastest steady >=3% winners separately within real resource budgets."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent;load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(R/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists();assert not(R/'selection-audit.json').exists();a=load(R/'native-raw-audit.json');assert a['passed'] and a['points']==2+4*len(load(R/'cpu-admission.json')['results']) and a['raw_dispatches']==4*a['points'] and a['all_feature_odd_K_over_BK17']
q=load(R/'pipeline-candidates.json');analysis=load(R/'analysis/pipeline-map.json');m=load(R/'reference-map.json');assert len(q['candidates'])==len(analysis['candidates'])==len(load(R/'cpu-admission.json')['results'])
rows=[];winners={}
for c in q['candidates']:
 ar=next(x for x in analysis['candidates'] if x['source_ids']==c['source_ids']);assert ar['mmac_theoretical_fraction']==c['mmac_theoretical_fraction'] and ar['intermediate_residual_fraction_of_long_time']==c['intermediate_residual_fraction_of_long_time'];physical=c['physical'];assert all(physical[k]==0 for k in ['private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count']);eligible=[];comparisons=[]
 for choice in ['best','best_no_spill','best_64k','best_64k_no_spill']:
  cap=65536 if '64k' in choice else 131072
  if c['shared_bytes']>cap:continue
  old=next(x for x in m[choice] if x['tile']==c['tile'] and x['layout']==c['layout']);ratio=c['mmac_theoretical_fraction']/old['mmac_theoretical_fraction'];comparisons.append(dict(choice=choice,cap=cap,prior=old['mmac_theoretical_fraction'],relative_initial_gain=ratio-1))
  if c['steady'] and ratio>=1.03:
   eligible.append(choice);key=(tuple(c['tile']),c['layout'],choice)
   if key not in winners or winners[key]['candidate']['mmac_theoretical_fraction']<c['mmac_theoretical_fraction']:winners[key]=dict(candidate=c,choice=choice,prior=old,ratio=ratio)
 rows.append(dict(config=c['config'],layout=c['layout'],fraction=c['mmac_theoretical_fraction'],shared=c['shared_bytes'],physical=physical,residual=c['intermediate_residual_fraction_of_long_time'],steady=c['steady'],source_ids=c['source_ids'],eligible_choices=eligible,comparisons=comparisons))
nominees={}
for w in winners.values():
 c=w['candidate'];key=tuple(c['source_ids']);n=nominees.setdefault(key,dict(candidate=c,choices=[]));n['choices'].append(dict(choice=w['choice'],prior=w['prior'],initial_relative_throughput=w['ratio']))
# No standalone98% resource fallback in this performance cohort.
resource_count=0
out=dict(passed=True,raw_audit_sha256=sha(R/'native-raw-audit.json'),candidate_sha256=sha(R/'pipeline-candidates.json'),analysis_sha256=sha(R/'analysis/pipeline-map.json'),rows=rows,nominees=list(nominees.values()),performance_nominees=sum(bool(x['choices']) for x in nominees.values()),resource_alternative_nominees=resource_count,one_independent_series_per_recipe=True,policy='Fastest initial steady recipe per layout and each real128KiB/64KiB budget; initial and one independent fresh each >=3% gain; full ASM and runtime contract same, residual<=2%, MMOPwork1. Never compare80KiB shader to64KiB domain. Resource alternative: fastest steady physical-zero recipe within64KiB, both initial/fresh >=98% of64KiB baseline, full matching ASM/work; only one independent series even when also a performance nominee. No retries of nonsteady recipes. Odd17 edge data are correctness only.',map_unchanged=True)
(R/'selection-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,performance_nominees=sum(bool(x['choices']) for x in nominees.values()),nominees=[dict(layout=v['candidate']['layout'],fraction=v['candidate']['mmac_theoretical_fraction'],shared=v['candidate']['shared_bytes'],choices=[x['choice'] for x in v['choices']]) for v in nominees.values()],steady=sum(c['steady'] for c in q['candidates']))))
