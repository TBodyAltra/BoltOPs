"""Bind exact controls, mapping proof, actual ISA novelty and physical spills."""
import pathlib,json,hashlib,re
R=pathlib.Path(__file__).resolve().parent;B=R.parent
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def code(p):
    lines=[]
    for l in p.read_text().splitlines():
        if re.match(r'\s*\.Lfunc_end\d+:',l):break
        l=l.split(';',1)[0].strip()
        if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(l)
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
e=load(R/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
assert not (R/'index-review.json').exists()
q=load(R/'cpu-screen.json');assert q['passed'] and q['default_controls_exact'] and len(q['results'])==8
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
proof=load(R/'index-mapping-proof.json');assert proof['passed'] and proof['valid_program_ids_checked']==2483776
prep=load(R/'prepare-proof.json');C=B/'single-tg-pipeline-20261027-round121-nowasp-sgpr-cpu-defaults'
assert sha(C/'bolt_ops/boltops/gemm/triton/gemm_fp16.py')==prep['parent_kernel_sha256']
assert sha(R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py')==prep['new_kernel_sha256']
old=load(C/'cpu-screen.json');assert all(sha(pathlib.Path(p))==h for p,h in old['source_hashes'].items())
known={};prior=old['results']+load(B/'single-tg-pipeline-20261027-round165-nowasp-register-pressure-cpu/cpu-screen.json')['results']+load(B/'single-tg-pipeline-20261027-round173-nowasp-scheduler-budget-cpu/cpu-screen.json')['results']
def key(x):return (code(pathlib.Path(x['directory'])/'kernel.amdgcn'),json.dumps(x['fields'],sort_keys=True),x['metadata']['shared'],x['metadata']['num_warps'])
for x in prior:
    if x['status']=='COMPILED' and x['tile']==[256,128,128] and x['layout']=='TT':known.setdefault(key(x),x['directory'])
selected=[];zero=[];scratch0=[];duplicates=[];excluded=[]
for x in q['results']:
    if x['status']!='COMPILED':excluded.append(x);continue
    c=x['config'];f=x['fields'];assert c['SIMPLIFY_GROUP1'] is True and c['GROUP_SIZE_M']==1 and c['num_stages']==1 and c['num_warps']==8 and c['MMA_K_SPLIT']==4 and c['LOOP_UNROLL_FACTOR'] in [1,2,4,8] and not c['WARP_SPECIALIZE'] and not c['wasp_wdra'] and not c['USE_MATRIX_STORE']
    assert x['normalized_ISA_sha256']==code(pathlib.Path(x['directory'])/'kernel.amdgcn')
    k=key(x)
    if k in known:duplicates.append(dict(variant=x['variant'],known_directory=known[k]));continue
    known[k]=x['directory']
    if x['metadata']['shared']>65536 or f['private_segment_fixed_size']!=0 or f['vgpr_spill_count']!=0:excluded.append(x);continue
    selected.append(x)
    (zero if f['sgpr_spill_count']==0 else scratch0).append(x)
out=dict(passed=True,CPU_only=True,CPU_screen_sha256=sha(R/'cpu-screen.json'),kernel_index_patch_sha256=sha(R/'kernel-index-fix.patch'),index_mapping_proof_sha256=sha(R/'index-mapping-proof.json'),native_sha256=sha(R/'native/libtriton.so'),source_hashes_unchanged=True,parent_GPU_sources_unchanged=True,default_controls_exact=2,selected=selected,physical_zero=zero,scratch0_SGPR_spilling=scratch0,duplicates=duplicates,excluded=excluded,actual_PMD_LDS_capacity=131072,native_gold_and_ISA_resource_check_required=True,independent_fresh_for_new_winners_required=True,no_performance_claim=True,no_automatic_map_promotion=True)
(R/'index-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=True,features=8,compiled=sum(x['status']=='COMPILED' for x in q['results']),selected=len(selected),physical_zero=len(zero),scratch0_SGPR_spilling=len(scratch0),duplicates=len(duplicates),excluded=len(excluded))))
