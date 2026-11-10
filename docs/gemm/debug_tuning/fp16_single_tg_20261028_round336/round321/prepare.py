from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261028-round320-small-work1-admission-cpu';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(C/'job-exit.json');assert e['returncode']==0 and (not Path('/proc',str(e['pid'])).exists() or any(l.startswith('State:') and 'Z (zombie)' in l for l in Path('/proc',str(e['pid']),'status').read_text().splitlines()));q=load(C/'cpu-screen.json');rev=load(C/'small-cta-admission-review.json');assert q['passed'] and rev['passed'] and rev['CPU_screen_sha256']==sha(C/'cpu-screen.json');assert all(sha(Path(p))==v for p,v in q['source_hashes'].items());refs=rev['selected'];assert refs and all(x['tile'] in [[32,32,128],[32,64,128],[64,32,128],[32,128,128],[128,32,128],[64,64,128],[64,128,128],[128,64,128]] and x['mls'] is True and x['config']['WARP_SPECIALIZE'] and not x['config']['wasp_wdra'] and x['config']['num_stages']==2 and x['config']['wasp_partition_warps'] in [[1,4],[2,4]] for x in refs);controls=q['controls'];assert refs and len(controls)==2;assert shutil.disk_usage(R).free>24*2**30
for n in ['runs','native']:(R/n).mkdir(exist_ok=True)
for n in ['triton','bolt_ops','deps','session.lock']:
 assert not (R/n).exists();(R/n).symlink_to((C/n).resolve())
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert (R/'session.lock').stat().st_ino==179052475 and (R/'session.lock').stat().st_dev==66310
import ast,types,math
scan_source=(R/'scan.py').read_text();tree=ast.parse(scan_source);nodes=[x for x in tree.body if (isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SPEC_KEYS' for t in x.targets)) or (isinstance(x,ast.FunctionDef) and x.name in ['spec','point_key'])];g={'json':json,'hashlib':hashlib};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(R/'scan.py'),'exec'),g)
tree=ast.parse((R/'tune.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='template');tg={'scan':types.SimpleNamespace(spec=g['spec'])};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(R/'tune.py'),'exec'),tg)
contracts=[]
for x in controls:contracts.append(tg['template'](x))
for x in refs:
 for loops in (16,32,64,17):
  edge=loops==17;contracts.append(dict(tg['template'](x),scope='full_gemm' if edge else 'single_cta',correctness_only=edge,shape=[x['tile'][0]+(16 if edge else 0),x['tile'][1]+(16 if edge else 0),128*loops],pipeline_loops=loops))
ids=[g['point_key'](x,'guarded-preflight321') for x in contracts];assert len(ids)==len(set(ids))==2+4*len(refs)
a={'tile':[32,32,128]};assert g['point_key'](a,'x')==g['point_key'](dict(a,wasp_global_k_fragments=1),'x')
(R/'point-identity-regression.json').write_text(json.dumps(dict(passed=True,actual_template_spec_key_functions=True,contracts=len(contracts),unique_ids=len(set(ids)),scan_sha256=sha(R/'scan.py'),rows=[dict(id=i,spec=x) for i,x in zip(ids,contracts)]),indent=2)+'\n')
vgpr=[]
for x in refs:
 waves=sum(x['config']['wasp_partition_warps']);v=x['fields']['amdhsa_next_free_vgpr'];reserve=((v+3)//4)*4;bound=((waves+3)//4)*reserve;assert bound<=512 and x['metadata']['shared']<=131072
 vgpr.append(dict(directory=x['directory'],waves=waves,nextfree=v,rounded_per_wave=reserve,max_per_SIMD_bound=bound,capacity=512))
(C/'SIMD-VGPR-admission-proof.json').write_text(json.dumps(dict(passed=True,capacity=512,SIMDs=4,minalloc=4,rows=vgpr),indent=2)+'\n')

h=dict(q['source_hashes']);h[str(R/'point-identity-regression.json')]=sha(R/'point-identity-regression.json');h[str(R/'point-identity-global-k-fragments.patch')]=sha(R/'point-identity-global-k-fragments.patch')
h[str(C/'work-overcoverage-evidence.json')]=sha(C/'work-overcoverage-evidence.json')
for x in refs+controls:
 for f in ['kernel.amdgcn','result.json','job.json']:
  p=Path(x['directory'])/f;h[str(p)]=sha(p)
for n in ['cpu-screen.json','small-cta-admission-review.json','review_admission.py','map-snapshot-for-alias-review.json','default-source-compatibility.json','wasp-global-k-fragment.patch','bolt_ops-complete-source.patch','SIMD-VGPR-admission-proof.json']:h[str(C/n)]=sha(C/n)
(R/'cpu-admission.json').write_text(json.dumps(dict(q,results=refs,source_hashes=h,build=dict(sha256=sha(R/'native/libtriton.so'))),indent=2)+'\n');(R/'cpu-controls.json').write_text(json.dumps(dict(passed=True,controls=controls),indent=2)+'\n');(R/'layout3-review.json').write_bytes((C/'small-cta-admission-review.json').read_bytes())
(R/'triton-source.patch').write_bytes((B/'single-tg-pipeline-20261027-round226-complete-source-export/triton-parent209-complete-source.patch').read_bytes());(R/'bolt_ops-source.patch').write_bytes((C/'bolt_ops-complete-source.patch').read_bytes());(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('299largewapsfragmentcpu','321smallwork1fragmentnative'));(R/'control.json').write_bytes((B/'single-tg-pipeline-20261027-round225-wdra-tight-vgpr-admitted-native/control.json').read_bytes());(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,features=len(refs),controls=2,scope='LogicalBK128,physicalglobalK64/32;8 small tiles TT/TN;NoWDRA producers1/2 consumers4,stage2;novel physical zero-spill actual LDS/SIMD VGPR capacity proven;128cube deferred',native_sha256=sha(R/'native/libtriton.so'),source_worktrees_frozen=True,source_edits=1,no_runtime_MN_specialization=True,no_TG_rearrangement=True,all_feature_odd17_edges_required_before_long=True,full_GOLD_CPU_ASM_resources_and_work_required=True,max_models=8,max_compilers=1,no_automatic_promotion=True),indent=2)+'\n');print(json.dumps(dict(prepared=True,features=len(refs),controls=2)))
