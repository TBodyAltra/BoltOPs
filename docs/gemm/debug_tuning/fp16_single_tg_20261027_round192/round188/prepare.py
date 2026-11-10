"""Isolated, bounds-preserving matrix-load base recentering for BK128 SGPR pressure."""
import pathlib,json,hashlib,subprocess,difflib,itertools
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round175-group1-index-cpu';D=B/'single-tg-pipeline-20261027-round180-sgpr-liveness-cpu'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for root in [C,D]:
 e=load(root/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
 q=load(root/'cpu-screen.json');assert q['passed'] and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items())
assert not (R/'bolt_ops').exists()
for n in ['triton','deps','session.lock']:(R/n).symlink_to((C/n).resolve())
subprocess.run(['git','-C',str(C/'bolt_ops'),'worktree','add','-b','perf/mls-rebase-round188-bolt_ops',str(R/'bolt_ops'),'840d816b671983527e5b1acc831c5e67618fe981'],check=True)
subprocess.run(['git','-C',str(R/'bolt_ops'),'apply',str(C/'bolt_ops-source.patch')],check=True)
p=R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py';assert sha(p)==sha(C/'bolt_ops/boltops/gemm/triton/gemm_fp16.py');old=p.read_text();start=old.index('def _gemm_fp16_mls_kernel(');prefix,body=old[:start],old[start:]
a='    SIMPLIFY_GROUP1: tl.constexpr = False,\n';assert body.count(a)==1;body=body.replace(a,a+'    REBASE_MLS_TILE: tl.constexpr = 0,\n')
a='''    mls_offs_am = pid_m * BLOCK_SIZE_M
    mls_offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
''';assert body.count(a)==1
repl='''    mls_offs_am = pid_m * BLOCK_SIZE_M
    mls_offs_bn = pid_n * BLOCK_SIZE_N
    tl.static_assert(REBASE_MLS_TILE >= 0 and REBASE_MLS_TILE <= 3)
    if REBASE_MLS_TILE:
        tl.static_assert(not WARP_SPECIALIZE)
        tl.static_assert(MIXED_OPERAND == 0)
    # Rebase only the non-K tile coordinates. Keep the exact remaining shape
    # so every original edge predicate is preserved for arbitrary M/N.
    if REBASE_MLS_TILE & 1:
        mls_a_ptr = a_ptr + mls_offs_am.to(tl.int64) * stride_am
        mls_shape_m = M - mls_offs_am
        mls_load_m = 0
    else:
        mls_a_ptr = a_ptr
        mls_shape_m = M
        mls_load_m = mls_offs_am
    if REBASE_MLS_TILE & 2:
        mls_b_ptr = b_ptr + mls_offs_bn.to(tl.int64) * stride_bn
        mls_shape_n = N - mls_offs_bn
        mls_load_n = 0
    else:
        mls_b_ptr = b_ptr
        mls_shape_n = N
        mls_load_n = mls_offs_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
'''
body=body.replace(a,repl)
x=body.index('        if MIXED_OPERAND == 0:');y=body.index('        elif MIXED_OPERAND == 1:');region=body[x:y]
for a,z in [('                    a_ptr,','                    mls_a_ptr,'),('                    b_ptr,','                    mls_b_ptr,'),('shape=[M, K]','shape=[mls_shape_m, K]'),('shape=[K, N]','shape=[K, mls_shape_n]'),('offsets=[mls_offs_am, mls_offs_k]','offsets=[mls_load_m, mls_offs_k]'),('offsets=[mls_offs_k, mls_offs_bn]','offsets=[mls_offs_k, mls_load_n]')]:
 assert region.count(a)==2,(a,region.count(a));region=region.replace(a,z)
body=body[:x]+region+body[y:];p.write_text(prefix+body);compile(p.read_text(),str(p),'exec')
(R/'kernel-rebase.patch').write_text(''.join(difflib.unified_diff(old.splitlines(keepends=True),p.read_text().splitlines(keepends=True),fromfile='a/boltops/gemm/triton/gemm_fp16.py',tofile='b/boltops/gemm/triton/gemm_fp16.py')))
subprocess.run(['git','-C',str(R/'bolt_ops'),'diff','--check'],check=True);(R/'bolt_ops-source.patch').write_bytes(subprocess.check_output(['git','-C',str(R/'bolt_ops'),'diff','HEAD','--binary']))
(R/'triton-source.patch').write_bytes((C/'triton-source.patch').read_bytes());(R/'native').mkdir();(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
for n in ['compile_probe.py','native-reuse.json']:(R/n).write_bytes((C/n).read_bytes())
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('175group1indexcpu','188mlsrebasecpu'))
refs=load(C/'control-qualification.json')['controls'];assert len(refs)==2;(R/'references.json').write_text(json.dumps(refs,indent=2)+'\n');seed=load(pathlib.Path(refs[0]['directory'])/'job.json');assert seed['tile']==[256,128,128] and seed['layout']=='TT'
jobs=[]
for uf,simple,mask in itertools.product([2,4],[False,True],[1,2,3]):
 j=load(pathlib.Path(refs[0]['directory'])/'job.json');j.update(label='features',variant=f'rebase{mask}_simple{int(simple)}_UF{uf}_EP1');j['config'].update(REBASE_MLS_TILE=mask,SIMPLIFY_GROUP1=simple,LOOP_UNROLL_FACTOR=uf,optimize_epilogue=True,num_warps=8,MMA_K_SPLIT=4,misched_regpressure=True,USE_MATRIX_STORE=False);jobs.append(j)
(R/'feature-plan.json').write_text(json.dumps(dict(CPU_only=True,controls=2,features=jobs,scope='256x128x128 TT LDS64KiB NoWASP, same transport and MMAC, base recenter A/B/both, remaining bounds preserved, no new M/N/K assumptions',no_PMD_admission=True),indent=2)+'\n');(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
checks=0
for size,block in itertools.product([1,17,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1023],[128,256]):
 for tile in range((size+block-1)//block):
  off=tile*block
  for i in range(block):
   assert (off+i<size)==(i<size-off)
   for stride in [1,16,128,272,1040]:assert (off+i)*stride==off*stride+i*stride
   checks+=1
proof=dict(passed=True,domain='Every valid GEMM TG, positive M/N, K divisible BK as before; no change to grid, GT ordering, K loop, accumulator, or C masks',A_address='a + (m0+i)*sam + k*sak == (a+m0*sam) + i*sam + k*sak',A_bounds='m0+i<M iff i<M-m0, since 0<=m0<M for every valid TG',B_address='b + k*sbk + (n0+j)*sbn == (b+n0*sbn) + k*sbk+j*sbn',B_bounds='n0+j<N iff j<N-n0, since 0<=n0<N for every valid TG',sample_coordinates_checked=checks,rebase_disabled_preserves_original_expressions=True,NoWASP_and_MIXED0_only=True,native_full_output_and_off_origin_edge_checks_required=True)
(R/'address-bounds-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
assert all(sha(pathlib.Path(p))==h for p,h in load(C/'cpu-screen.json')['source_hashes'].items())
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_only=True,isolated_Bolt_worktree=True,frozen_Triton=str((R/'triton').resolve()),native95_unchanged=True,controls=2,features=12,parent_GPU_source_hashes_unchanged=True,new_kernel_sha256=sha(R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),address_proof_sha256=sha(R/'address-bounds-proof.json'),native_precision_and_independent_performance_required=True,no_PMD_admission=True),indent=2)+'\n');print('Prepared188: isolated base recenter,2 exact controls+12 variants; no models')
