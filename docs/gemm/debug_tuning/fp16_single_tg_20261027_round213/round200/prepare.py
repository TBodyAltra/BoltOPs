"""Admit two frozen CPU cohorts; preserve overcapacity and compile rejections."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round197-fragment-stream-cpu';D=B/'single-tg-pipeline-20261027-round199-fragment-stage2-cpu';P=B/'single-tg-pipeline-20261027-round195-rebase-argument-lowering-native'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[];hashes={};reviews=[]
for root,reviewname in [(C,'fragment-stream-review.json'),(D,'stage2-review.json')]:
 e=load(root/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists();q=load(root/'cpu-screen.json');v=load(root/reviewname);assert q['passed'] and v['passed'] and v['CPU_screen_sha256']==sha(root/'cpu-screen.json') and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());rows+=v['selected'];reviews.append(dict(root=str(root),screen_sha256=sha(root/'cpu-screen.json'),review_sha256=sha(root/reviewname),selected=len(v['selected']),excluded=len(v['excluded'])))
 for p,h in q['source_hashes'].items():
  if p in hashes:assert hashes[p]==h
  hashes[p]=h
assert len(rows)==12
for x in rows:assert x['tile']==[64,256,128] and x['config']['LOOP_UNROLL_FACTOR']==1 and x['config']['MMA_K_SPLIT']==x['config']['STREAM_K_FRAGMENTS'] in [2,4] and x['all_physical_spills_zero'] and x['metadata']['shared']<=131072 and not x['config']['use_async_copy'] and not x['config']['use_block_pingpong']
for n in ['logs','runs','native']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
 assert not(R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990';assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['worker.py','scan.py','stats_io.py']:(R/n).write_bytes((P/n).read_bytes())
p=R/'worker.py';s=p.read_text();a='    cfg["MMA_K_SPLIT"] = dot_split';assert s.count(a)==1;s=s.replace(a,a+'''
    stream_fragments = spec.get("stream_k_fragments", 1)
    stream_prefetch = spec.get("stream_fragment_prefetch", False)
    assert type(stream_fragments) is int and stream_fragments in (1, 2, 4)
    assert type(stream_prefetch) is bool
    if stream_fragments > 1:
        assert spec["mode"] == "nowasp" and spec["mls"] and not spec.get("mixed_operand")
        assert dot_split == stream_fragments and unroll == 1
    else:
        assert not stream_prefetch
    cfg["STREAM_K_FRAGMENTS"] = stream_fragments
    cfg["STREAM_FRAGMENT_PREFETCH"] = stream_prefetch
''');p.write_text(s);compile(s,str(p),'exec')
p=R/'scan.py';s=p.read_text();a='"rebase_mls_tile", "correctness_only")';assert s.count(a)==1;s=s.replace(a,'"rebase_mls_tile", "correctness_only", "stream_k_fragments", "stream_fragment_prefetch")');assert s.count('"argument-lowering-review.json"')==1;s=s.replace('"argument-lowering-review.json"','"fragment-stream-review.json"');a="['address-bounds-proof.json','kernel-rebase.patch','backend-liveness.patch','clang-help-proof.json']";assert s.count(a)==1;s=s.replace(a,"['address-bounds-proof.json','kernel-rebase.patch','kernel-fragment-stream.patch','fragment-coverage-proof.json','stage2-review.json','combined-admission.json']");p.write_text(s);compile(s,str(p),'exec')
for n in ['triton-source.patch','bolt_ops-source.patch','native-reuse.json','fragment-stream-review.json','address-bounds-proof.json','kernel-rebase.patch','kernel-fragment-stream.patch','fragment-coverage-proof.json']:(R/n).write_bytes((C/n).read_bytes())
(R/'stage2-review.json').write_bytes((D/'stage2-review.json').read_bytes());(R/'combined-admission.json').write_text(json.dumps(dict(passed=True,CPU_cohorts=reviews,combined_features=12,overcapacity_excluded=8,compile_failures_excluded=4,logical_BK128_preserved=True,shared_GPU_source_worktree=str((R/'bolt_ops').resolve()),no_GPU_source_changes_after_CPU=True),indent=2)+'\n')
(R/'cpu-admission.json').write_text(json.dumps(dict(passed=True,CPU_only=True,results=rows,source_hashes=hashes,source_hashes_unchanged=True,build=dict(sha256=sha(R/'native/libtriton.so'))),indent=2)+'\n');(R/'cpu-controls.json').write_bytes((C/'control-qualification.json').read_bytes());(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('197fragmentstreamcpu','200fragmentstreamnative'));(R/'control.json').write_bytes((P/'control.json').read_bytes());(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_sources=reviews,GPU_sources_frozen=True,controls=2,features=12,logical_BK128_preserved=True,physical_fragments_K=[64,32],requested_stages=[1,2],shared_bytes=sorted(set(x['metadata']['shared'] for x in rows)),all_candidate_physical_spills_zero=True,native_sha256=sha(R/'native/libtriton.so'),max_models=8,max_compilers=1,TT_and_TN_single_model_pilots_before_remaining=True,full_gold_and_odd17_off_origin_edge_gate_before_long_calibration=True,edge_data_excluded_from_performance_map=True,independent_fresh_for_new_winners_required=True,no_automatic_map_promotion=True),indent=2)+'\n');print('Prepared200:two exact controls,12 legal fragment-stream variants,TT/TN pilots,odd17 edge gates before long calibration; native95,canonical lease,max8/one compiler')
