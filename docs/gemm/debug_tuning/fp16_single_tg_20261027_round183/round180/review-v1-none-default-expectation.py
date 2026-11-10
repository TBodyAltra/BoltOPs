"""Bind actual clang arguments, unchanged controls, ISA novelty and real spills."""
import pathlib,json,hashlib,re,collections
R=pathlib.Path(__file__).resolve().parent;B=R.parent
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def code(p):
    lines=[]
    for l in p.read_text().splitlines():
        if re.match(r'\s*\.Lfunc_end\d+:',l):break
        l=l.split(';',1)[0].strip()
        if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(l)
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
def scalar(p):
    s=p.read_text();out={}
    for n in ['amdhsa_next_free_sgpr','sgpr_count']:
        m=re.search(r'\.'+n+r'\s*[:=]?\s*(\d+)',s);out[n]=int(m[1]) if m else None
    return out
def key(x):
    p=pathlib.Path(x['directory'])/'kernel.amdgcn'
    return (code(p),json.dumps(x['fields'],sort_keys=True),json.dumps(scalar(p),sort_keys=True),x['metadata']['shared'],x['metadata']['num_warps'])
e=load(R/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists();assert not (R/'liveness-review.json').exists()
q=load(R/'cpu-screen.json');assert q['passed'] and q['default_controls_exact'] and len(q['results'])==10
assert all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
C=B/'single-tg-pipeline-20261027-round175-group1-index-cpu';parent=load(C/'cpu-screen.json');assert all(sha(pathlib.Path(p))==h for p,h in parent['source_hashes'].items())
from triton.backends.hcu.compiler import HIPBackend
from triton.backends.compiler import GPUTarget
import importlib
fn=importlib.import_module('boltops.gemm.triton.gemm_fp16')._gemm_fp16_mls_kernel
backend=HIPBackend(GPUTarget('hip','gfx946',64))
for bad in [True,-1,6,'1',None]:
    try:backend.parse_options(dict(hcu_sgpr_liveness_strategy=bad))
    except ValueError:pass
    else:raise AssertionError(('Invalid strategy accepted',bad))
known={};prior_count=0
for name in ['single-tg-pipeline-20261027-round121-nowasp-sgpr-cpu-defaults','single-tg-pipeline-20261027-round165-nowasp-register-pressure-cpu','single-tg-pipeline-20261027-round173-nowasp-scheduler-budget-cpu','single-tg-pipeline-20261027-round175-group1-index-cpu']:
    p=B/name;rows=load(p/'cpu-screen.json')['results']
    if (p/'control-qualification.json').exists():rows+=load(p/'control-qualification.json')['controls']
    for x in rows:
        if x['status']=='COMPILED' and x['tile']==[256,128,128] and x['layout']=='TT':known.setdefault(key(x),x['directory']);prior_count+=1
selected=[];duplicates=[];excluded=[];args_records=[]
for x in q['results']:
    if x['status']!='COMPILED':excluded.append(x);continue
    c=x['config'];assert c['num_stages']==1 and c['num_warps']==8 and c['LOOP_UNROLL_FACTOR']==4 and c['MMA_K_SPLIT']==4 and c['optimize_epilogue'] is True and not c['WARP_SPECIALIZE'] and not c['wasp_wdra'] and c['USE_MATRIX_STORE'] is False
    opts={k:v for k,v in c.items() if k not in fn.arg_names};option=backend.parse_options(opts)
    actual=HIPBackend._get_clang_args(x['metadata'],option);default=HIPBackend._get_clang_args(x['metadata'],backend.parse_options(dict(opts,hcu_sgpr_liveness_strategy=0)))
    extra=list((collections.Counter(actual)-collections.Counter(default)).elements());assert extra and all(a.startswith('-mllvm=') for a in extra)
    assert not any('max-sgpr' in a or 'vgpr-limit' in a for a in extra)
    assert x['metadata']['hcu_sgpr_liveness_strategy']==c['hcu_sgpr_liveness_strategy'];args_records.append(dict(variant=x['variant'],strategy=c['hcu_sgpr_liveness_strategy'],added_clang_args=extra,config=c))
    assert x['normalized_ISA_sha256']==code(pathlib.Path(x['directory'])/'kernel.amdgcn')
    k=key(x)
    if k in known:duplicates.append(dict(variant=x['variant'],same_ISA_resources_as=known[k]));continue
    known[k]=x['directory'];f=x['fields']
    if x['metadata']['shared']>65536 or f['private_segment_fixed_size']!=0 or f['vgpr_spill_count']!=0 or f['amdhsa_next_free_vgpr']>256:excluded.append(x);continue
    selected.append(x)
out=dict(passed=True,CPU_only=True,CPU_screen_sha256=sha(R/'cpu-screen.json'),native_sha256=sha(R/'native/libtriton.so'),backend_patch_sha256=sha(R/'backend-liveness.patch'),source_hashes_unchanged=True,parent_sources_unchanged=True,default_controls_exact=2,previous_actual_ISA_resource_records=prior_count,added_clang_arguments=args_records,invalid_strategy_types_rejected=True,selected=selected,strict_physical_zero=sum(x['all_physical_spills_zero'] for x in selected),duplicates=duplicates,excluded=excluded,actual_PMD_LDS_capacity=131072,per_recipe_SGPR_spills_preserved=True,native_full_gold_and_actual_workload_check_required=True,no_automatic_map_promotion=True,no_performance_claim=True)
(R/'liveness-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=True,features=10,compiled=sum(x['status']=='COMPILED' for x in q['results']),novel_selected=len(selected),duplicates=len(duplicates),excluded=len(excluded),strict_zero=out['strict_physical_zero'],selected_spills=[(x['variant'],x['fields']['sgpr_spill_count']) for x in selected])))
