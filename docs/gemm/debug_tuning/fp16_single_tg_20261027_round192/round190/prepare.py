"""Freeze isolated MLS rebase source; admit distinct shaders with explicit edge gates."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round188-mls-rebase-cpu';P=B/'single-tg-pipeline-20261027-round181-sgpr-liveness-native'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(C/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists();q=load(C/'cpu-screen.json');review=load(C/'rebase-review.json');assert review['passed'] and review['CPU_screen_sha256']==sha(C/'cpu-screen.json') and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
refs=review['selected'];assert len(refs)==12 and review['strict_physical_zero']==6
for x in refs:assert x['fields']['private_segment_fixed_size']==x['fields']['vgpr_spill_count']==0 and x['metadata']['shared']==65536 and x['config']['REBASE_MLS_TILE'] in [1,2,3]
for n in ['logs','runs','native']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
 assert not(R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')==review['native_sha256']=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990';assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
for n in ['worker.py','scan.py','stats_io.py']:(R/n).write_bytes((P/n).read_bytes())
p=R/'worker.py';s=p.read_text();a='    cfg["SIMPLIFY_GROUP1"] = simplify';assert s.count(a)==1;s=s.replace(a,a+'''
    rebase = spec.get("rebase_mls_tile", 0)
    assert type(rebase) is int and rebase in (0, 1, 2, 3)
    if rebase:
        assert spec["mode"] == "nowasp" and spec["mls"] and not spec.get("mixed_operand")
    cfg["REBASE_MLS_TILE"] = rebase
''')
a='    assert (m, n) == (bm, bn) and k % bk == 0\n';assert s.count(a)==1;s=s.replace(a,'''    assert k % bk == 0
    correctness_only = spec.get("correctness_only", False)
    assert type(correctness_only) is bool
    if correctness_only:
        assert spec["scope"] == "full_gemm" and (m, n) == (bm + 16, bn + 16)
    else:
        assert spec["scope"] == "single_cta" and (m, n) == (bm, bn)
''')
a='    result["kernel_contract"] = "Runtime M/N interface, full K blocks; benchmark grid1 and M=BM/N=BN; K divisible by BK; no K-tail loads"';assert s.count(a)==1;s=s.replace(a,'''    result["kernel_contract"] = ("Correctness-only full grid with off-origin and M/N edge tiles; same full-K contract; excluded from performance map" if correctness_only else "Runtime M/N full-K, singleTG grid1 and M=BM/N=BN; no K-tail loads")''')
a='    assert (m, n) == (bm, bn) and k % bk == 0 and grid == (1,)';assert s.count(a)==1;s=s.replace(a,'''    if correctness_only:
        assert grid == (4,)
        result["benchmark_kind"] = "correctness_only_off_origin_and_edge_tiles"
    else:
'''+a.replace('    ','        ',1));p.write_text(s);compile(s,str(p),'exec')
p=R/'scan.py';s=p.read_text();a='"simplify_group1")';assert s.count(a)==1;s=s.replace(a,'"simplify_group1", "rebase_mls_tile", "correctness_only")')
s=s.replace('"liveness-review.json"','"rebase-review.json"').replace("['index-mapping-proof.json','kernel-index-fix.patch','backend-liveness.patch','clang-help-proof.json']","['address-bounds-proof.json','kernel-rebase.patch']").replace('backend SGPR liveness strategies1/3/5 with generic/GROUP1 mappings; actual SGPR spills retained; serial compiler lock','MLS non-K tile base recenter, actual spill counts retained; separate off-origin edge correctness gate; serial compiler lock')
a='            assert result["scope"] == "single_cta" and result["grid"] == [1]';assert s.count(a)==1;s=s.replace(a,'''            if result.get("correctness_only", False):
                assert result["scope"] == "full_gemm" and result["grid"] == [4]
                result["not_for_performance_map"] = True
                result.pop("tops", None)
                result.pop("cycles_per_output", None)
                return
'''+a);p.write_text(s);compile(s,str(p),'exec')
for n in ['triton-source.patch','bolt_ops-source.patch','native-reuse.json','rebase-review.json','address-bounds-proof.json','kernel-rebase.patch']:(R/n).write_bytes((C/n).read_bytes())
(R/'cpu-admission.json').write_text(json.dumps(dict(q,results=refs,build=dict(sha256=review['native_sha256'])),indent=2)+'\n');(R/'cpu-controls.json').write_bytes((C/'control-qualification.json').read_bytes());(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('188mlsrebasecpu','190mlsrebasenative'));(R/'control.json').write_bytes((P/'control.json').read_bytes());(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_source=str(C),CPU_screen_sha256=sha(C/'cpu-screen.json'),rebase_review_sha256=sha(C/'rebase-review.json'),native_sha256=review['native_sha256'],GPU_sources_frozen=True,controls=2,features=12,strict_zero_candidates=6,actual_SGPR_spills=[x['fields']['sgpr_spill_count'] for x in refs],worker_changes='Typed rebase constexpr; narrowly scoped full-grid edge correctness opt-in; same common compiler.lock',max_models=8,max_compilers=1,singleTG_pilot_before_remaining=True,full_gold_and_off_origin_edge_gate_before_long_calibration=True,edge_data_excluded_from_performance_map=True,independent_fresh_for_new_winners_required=True,no_automatic_map_promotion=True),indent=2)+'\n');print('Prepared189:2 exact defaults,12 novel shaders,6 physicalzero; singleTG pilot then separate fullgrid edge gold; max8 models/1 compiler')
