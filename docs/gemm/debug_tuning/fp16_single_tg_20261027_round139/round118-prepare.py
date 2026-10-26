import pathlib,json,hashlib,subprocess
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round115-alias-ab-slots-cpu-fix';P=B/'single-tg-pipeline-20261027-round111-late-slot-turn-native';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((C/'destination-view-screen.json').read_text());audit=json.loads((C/'alias-ownership-isa-audit.json').read_text());e=json.loads((C/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and len(q['results'])==6 and audit['passed'] and audit['source_hashes_unchanged']
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());assert audit['script_sha256']==sha(C/'audit_alias_ownership.py') and audit['protocol_model_sha256']==sha(C/'protocol-model-v3.json') and audit['protocol_script_sha256']==sha(C/'protocol_model_v3.py')
for n in ['alias-parent-address-audit.json','alias-consumer-read-audit.json']:assert json.loads((C/n).read_text())['passed']
assert json.loads((P/'native-raw-audit.json').read_text())['passed'] and json.loads((B/'single-tg-pipeline-20261027-round112-late-slot-turn-confirm/native-raw-audit.json').read_text())['passed']
for n in ['logs','runs','native','controller-sources']:(R/n).mkdir(exist_ok=True)
# Read-only use of the now-frozen isolated CPU115 source worktrees; no GPU edits.
for n in ['triton','bolt_ops','deps','session.lock']:
 assert not (R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['triton','bolt_ops']:(R/(n+'-source.patch')).write_bytes((C/(n+'-source.patch')).read_bytes())
for n in ['worker.py','scan.py','stats_io.py']:(R/n).write_bytes((P/n).read_bytes())
p=R/'worker.py';s=p.read_text();anchor='spec = json.loads(Path(sys.argv[1]).read_text())';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'''
_alias_ab=(spec.get("compiler_options") or {}).get("hcu_wasp_alias_ab_slots",False)
assert type(_alias_ab) is bool
if _alias_ab:
    o=spec["compiler_options"]
    assert o.get("hcu_wasp_split_b_slots") is True and o.get("hcu_wasp_prefetch_one_operand")==1 and o.get("hcu_wasp_mmac_turn")==0
''');needle='"hcu_wasp_split_b_slot_turn_wait_mask"}';assert s.count(needle)==1;s=s.replace(needle,'"hcu_wasp_split_b_slot_turn_wait_mask", "hcu_wasp_alias_ab_slots"}');p.write_text(s)
p=R/'scan.py';s=p.read_text().replace('single-tg-pipeline-20261027-round110-late-slot-turn-screen',C.name)
s=s.replace('"protocol_model.py","encoding-equivalence.json","excluded-candidates.json","CPU110-predecessor-proof.json",','"protocol_model_v3.py","alias-consumer-read-audit.json","predecessor-overlap-counterexample.json",')
s=s.replace('"split_b_slots.py","split_b_slot_turn.py"]','"split_b_slots.py","split_b_slot_turn.py","alias_ab_slots.py"]')
s=s.replace('short full-output precision qualification only','alias A/B single-TG full-output and16/32/64 slope qualification, no automatic promotion');p.write_text(s)
for src,dst in [('destination-view-screen.json','cpu-admission.json'),('qualification.json','cpu-controls.json'),('alias-parent-address-audit.json','parent-address-audit.json'),('alias-consumer-read-audit.json','alias-consumer-read-audit.json'),('alias-ownership-isa-audit.json','ownership-isa-audit.json'),('protocol-model-v3.json','protocol-model.json'),('protocol_model_v3.py','protocol_model_v3.py'),('predecessor-overlap-counterexample.json','predecessor-overlap-counterexample.json'),('native-reuse.json','native-reuse.json')]:(R/dst).write_bytes((C/src).read_bytes())
oldbase=B/'single-tg-pipeline-20261027-round104-rect-split-slots-cpu'
for n in ['base-protocol-model.json','base_protocol_model.py']:(R/n).write_bytes((oldbase/n).read_bytes())
assert json.loads((R/'base-protocol-model.json').read_text())['script_sha256']==sha(R/'base_protocol_model.py')
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('115aliasabslotscpufix','118aliasabslotsfresh'));(R/'control.json').write_bytes((P/'control.json').read_bytes());(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_source=str(C),GPU_sources_and_native_read_only_frozen=True,source_patches_identical=True,native_sha256=sha(R/'native/libtriton.so'),CPU_alias_ownership_audit_sha256=sha(C/'alias-ownership-isa-audit.json'),alias_protocol_script_sha256=sha(C/'protocol_model_v3.py'),alias_protocol_model_sha256=sha(C/'protocol-model-v3.json'),TN128_interleaved_B1_counterexample_preserved=True,no_GPU_source_modifications=True),indent=2)+'\n');print('Native116 exact qualified CPU115 alias sources: six controls + six64KiB alias full-gold short pilots, then64/32loops.')

I=B/'single-tg-pipeline-20261027-round117-alias-ab-slots-native-retry';e=json.loads((I/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and json.loads((I/'native-raw-audit.json').read_text())['passed']
initial=json.loads((I/'pipeline-candidates.json').read_text());reference=json.loads((R/'reference-map.json').read_text());assert sha(R/'reference-map.json')=='642f0afce2f7a214933aa64a357df85b7d0486f0906cd748a8a5d0b467049374';selected=[]
for x in initial['candidates']:
 if not x['steady'] or x['shared_bytes']>65536 or any(x['physical'][k] for k in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']):continue
 choices=[];baselines={}
 for name in ['best','best_no_spill','best_64k','best_64k_no_spill']:
  old=next((r for r in reference[name] if r['tile']==x['tile'] and r['layout']==x['layout']),None)
  if old and x['mmac_theoretical_fraction']/old['mmac_theoretical_fraction']>=1.03:choices.append(name);baselines[name]=old
 if choices:
  chosen=choices[0];selected.append(dict(initial=x,eligible_choices=choices,baselines=baselines,baseline_choice=chosen,baseline=baselines[chosen]))
assert len(selected)==2 and {(tuple(x['initial']['tile']),x['initial']['layout']) for x in selected}=={((32,256,128),'TT'),((128,256,128),'TT')}
assert all(x['eligible_choices']==['best_64k','best_64k_no_spill'] for x in selected)
for src,dst in [('pipeline-candidates.json','initial-candidates.json'),('native-raw-audit.json','initial-raw-audit.json')]:(R/dst).write_bytes((I/src).read_bytes())
plan=dict(passed=True,initial_root=str(I),selected=selected,one_independent_series_per_initial_eligible_recipe=True,no_extra_repeats_on_failure=True,fresh_loop_order=[32,16,64],initial_candidate_sha256=sha(I/'pipeline-candidates.json'),initial_raw_audit_sha256=sha(I/'native-raw-audit.json'),parent_map_sha256=sha(R/'reference-map.json'),no_automatic_map_promotion=True)
(R/'confirmation-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
p=R/'scan.py';s=p.read_text();needle='"protocol-model.json","prepare-proof.json"';assert s.count(needle)==1;p.write_text(s.replace(needle,'"protocol-model.json","confirmation-plan.json","initial-candidates.json","initial-raw-audit.json","prepare-proof.json"'))
print('Predeclared118 onefresh32/16/64series each for two TT64KiB winners; primary128KiB choices are NOT targets.')
