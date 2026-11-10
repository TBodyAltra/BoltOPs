"""Predeclare one performance nominee and five compact-quota resource alternatives."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;P=B/'single-tg-pipeline-20261027-round141-alias-vgpr-quota-native'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=json.loads((P/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists()
a=json.loads((P/'native-raw-audit.json').read_text());assert a['passed'] and a['points']==91 and a['raw_dispatches']==364 and a['all_physical_spills_zero']
q=json.loads((P/'cpu-admission.json').read_text());assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
for n in ['logs','runs','native','controller-sources']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
 assert not(R/n).exists();(R/n).symlink_to((P/n).resolve())
(R/'native/libtriton.so').symlink_to((P/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['worker.py','scan.py','stats_io.py','triton-source.patch','bolt_ops-source.patch','native-reuse.json','cpu-admission.json','cpu-controls.json','parent-address-audit.json','consumer-read-audit.json','ownership-isa-audit.json','alias-late-turn-protocol-model.json','protocol_alias_late_turn.py','predecessor-failure.json']:(R/n).write_bytes((P/n).read_bytes())
p=R/'scan.py';s=p.read_text();old="files.append(Path(os.environ['TRITON_HIP_CLANG_PATH']))";assert s.count(old)==1
s=s.replace(old,"files.extend(ROOT/n for n in ['confirmation-plan.json','initial-candidates.json','initial-raw-audit.json'])\n    "+old);p.write_text(s)
(R/'env.sh').write_text((P/'env.sh').read_text().replace(str(P),str(R)).replace('141aliasvgprquotanative','144aliasvgprquotaconfirm'))
for n in ['control.json','reference-map.json']:(R/n).write_bytes((P/n).read_bytes())
(R/'initial-candidates.json').write_bytes((P/'pipeline-candidates.json').read_bytes());(R/'initial-raw-audit.json').write_bytes((P/'native-raw-audit.json').read_bytes())
initial=json.loads((R/'initial-candidates.json').read_text());reference=json.loads((R/'reference-map.json').read_text());selected=[];keys=['best','best_no_spill','best_64k','best_64k_no_spill']
def quota(x):return 3*x['physical']['amdhsa_next_free_vgpr']
for tile,layout in sorted({(tuple(c['tile']),c['layout']) for c in initial['candidates']}):
 cs=[c for c in initial['candidates'] if c['tile']==list(tile) and c['layout']==layout and c['steady'] and c['shared_bytes']<=65536 and all(c['physical'][k]==0 for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'])]
 if not cs:continue
 bases={k:next(x for x in reference[k] if x['tile']==list(tile) and x['layout']==layout) for k in keys};best=max(cs,key=lambda c:c['mmac_theoretical_fraction'])
 eligible=[k for k in keys if best['mmac_theoretical_fraction']/bases[k]['mmac_theoretical_fraction']-1>=.03]
 baseline_choice=eligible[0] if eligible else 'best_64k';baseline=bases[baseline_choice]
 prior_quota=sum(baseline.get('effective_partition_regs') or baseline['config']['wasp_partition_regs'])
 if eligible:c=best
 else:
  near=[c for c in cs if c['mmac_theoretical_fraction']>=.98*baseline['mmac_theoretical_fraction'] and quota(c)<prior_quota]
  if not near:continue
  c=min(near,key=lambda c:(quota(c),-c['mmac_theoretical_fraction']))
 assert quota(c)<prior_quota
 selected.append(dict(initial=c,baseline=baseline,baseline_choice=baseline_choice,baselines=bases,eligible_choices=eligible,resource_alternative_requested=True,resource_alternative_only=not eligible,aggregate_quota=quota(c),baseline_aggregate_quota=prior_quota,selection='Fastest steady >=3% performance nominee if any; otherwise smallest effective quota among steady >=98% prior64KiB throughput alternatives, throughput tie-break. One series per profile.',no_multi_TG_occupancy_claim=True))
assert len(selected)==6 and sum(bool(x['eligible_choices']) for x in selected)==1
assert next(x for x in selected if x['eligible_choices'])['initial']['tile']==[64,256,128]
plan=dict(passed=True,initial_root=str(P),initial_context=initial['measurement_context_id'],initial_raw_audit_sha256=sha(P/'native-raw-audit.json'),initial_candidates_sha256=sha(P/'pipeline-candidates.json'),selected=selected,fresh_loop_order=[32,16,64],one_independent_series_per_initial_eligible_recipe=True,no_extra_repeats_on_failure=True,performance_promotion_requires_three_percent_initial_and_fresh=True,resource_alternative_validation_requires_steady_full_gold_zero_physical_spills=True,resource_initial_and_fresh_minimum_relative_fraction=.98,resource_alternative_requires_reduced_effective_quota=True,slower_resource_alternatives_never_replace_faster_map_choices=True,no_multi_TG_occupancy_claim=True,maximum_models=4)
(R/'confirmation-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,initial_root=str(P),actual_source_worktree=str((P/'triton').resolve()),GPU_sources_and_native_read_only_frozen=True,native_sha256=sha(R/'native/libtriton.so'),initial_raw_audit_sha256=sha(P/'native-raw-audit.json'),confirmation_plan_sha256=sha(R/'confirmation-plan.json'),selected=len(selected),no_extra_repeats=True),indent=2)+'\n')
print(json.dumps(dict(prepared=True,selected=[dict(tile=x['initial']['tile'],layout=x['initial']['layout'],regs=x['initial']['config']['wasp_partition_regs'],mask=x['initial']['config'].get('hcu_wasp_split_b_slot_turn_wait_mask',0),eligible=x['eligible_choices'],aggregate_quota=x['aggregate_quota'],resource_only=x['resource_alternative_only']) for x in selected])))
