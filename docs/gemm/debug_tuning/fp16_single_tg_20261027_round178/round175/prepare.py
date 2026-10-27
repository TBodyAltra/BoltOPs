"""Isolated kernel-only GROUP_SIZE_M=1 mapping experiment; parent stays frozen."""
import pathlib,json,hashlib,subprocess,itertools,difflib
R=pathlib.Path(__file__).resolve().parent;B=R.parent
C=B/'single-tg-pipeline-20261027-round121-nowasp-sgpr-cpu-defaults'
P=B/'single-tg-pipeline-20261027-round173-nowasp-scheduler-budget-cpu'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=json.loads((P/'job-exit.json').read_text());assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
review=json.loads((P/'scheduler-review.json').read_text());assert review['passed'] and len(review['duplicates'])==22
old=json.loads((C/'cpu-screen.json').read_text());assert old['passed'] and all(sha(pathlib.Path(p))==h for p,h in old['source_hashes'].items())
for n in ['triton','deps','session.lock']:
    assert not(R/n).exists();(R/n).symlink_to((C/n).resolve())
assert not(R/'bolt_ops').exists()
subprocess.run(['git','-C',str(C/'bolt_ops'),'worktree','add','-b','perf/group1-index-round175-bolt_ops',str(R/'bolt_ops'),'840d816b671983527e5b1acc831c5e67618fe981'],check=True)
subprocess.run(['git','-C',str(R/'bolt_ops'),'apply',str(C/'bolt_ops-source.patch')],check=True)
kernel=R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'
assert sha(kernel)==sha(C/'bolt_ops/boltops/gemm/triton/gemm_fp16.py')
before=kernel.read_text();start=before.index('def _gemm_fp16_mls_kernel(')
prefix,body=before[:start],before[start:]
anchor='    MIXED_VECTOR_FULL: tl.constexpr = False,\n'
assert body.count(anchor)==1
body=body.replace(anchor,anchor+'    SIMPLIFY_GROUP1: tl.constexpr = False,\n')
anchor='''    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
'''
assert body.count(anchor)==1
replacement='''    if SIMPLIFY_GROUP1:
        tl.static_assert(GROUP_SIZE_M == 1)
        # For every valid flattened GEMM program id, group_size_m is one.
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
'''+''.join('    '+line for line in anchor.splitlines(keepends=True))
body=body.replace(anchor,replacement);kernel.write_text(prefix+body)
(R/'kernel-index-fix.patch').write_text(''.join(difflib.unified_diff(before.splitlines(keepends=True),kernel.read_text().splitlines(keepends=True),fromfile='a/boltops/gemm/triton/gemm_fp16.py',tofile='b/boltops/gemm/triton/gemm_fp16.py')))
subprocess.run(['git','-C',str(R/'bolt_ops'),'diff','--check'],check=True)
(R/'bolt_ops-source.patch').write_bytes(subprocess.check_output(['git','-C',str(R/'bolt_ops'),'diff','HEAD','--binary']))
(R/'triton-source.patch').write_bytes((C/'triton-source.patch').read_bytes())
(R/'native').mkdir();(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve())
assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
for n in ['compile_probe.py','native-reuse.json']:(R/n).write_bytes((C/n).read_bytes())
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('121nowaspsgprcpudefaults','175group1indexcpu'))
refs=json.loads((P/'control-qualification.json').read_text())['controls'];assert len(refs)==2
(R/'references.json').write_text(json.dumps(refs,indent=2)+'\n')
seed=json.loads((pathlib.Path(refs[0]['directory'])/'job.json').read_text());assert seed['config']['GROUP_SIZE_M']==1
jobs=[]
for uf,ep in itertools.product([1,2,4,8],[False,True]):
    job=json.loads(json.dumps(seed));job.update(label='features',variant=f'simple_group1_UF{uf}_EP{int(ep)}')
    job['config'].update(SIMPLIFY_GROUP1=True,LOOP_UNROLL_FACTOR=uf,optimize_epilogue=ep,num_warps=8,MMA_K_SPLIT=4,misched_regpressure=True,USE_MATRIX_STORE=False)
    jobs.append(job)
(R/'feature-plan.json').write_text(json.dumps(dict(CPU_only=True,controls=2,features=jobs,scope='256x128x128 TT NoWASP noWDRA; exact GROUP1 mapping specialization; K transport, full-output masks, GEMM launch grid, arithmetic and pipeline unchanged',no_PMD_admission=True),indent=2)+'\n')
(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
counts=list(range(1,17))+[31,32,33,63,64,65,127,128,129,255,256,257]
checked=0
for pm,pn in itertools.product(counts,repeat=2):
    for pid in range(pm*pn):
        group=pid//pn;size=min(pm-group,1)
        assert size==1
        original=(group+(pid%pn)%size,(pid%pn)//size)
        simple=(pid//pn,pid%pn)
        assert original==simple and 0<=simple[0]<pm and 0<=simple[1]<pn
        checked+=1
proof=dict(passed=True,domain='M,N>0; grid=ceil(M/BM)*ceil(N/BN),0<=pid<grid,GROUP_SIZE_M=1; no M/N specialization',algebra='Let p=ceil(M/BM),q=ceil(N/BN). pid<p*q implies floor(pid/q)<p, so min(p-floor(pid/q),1)=1. Original mapping reduces exactly to (floor(pid/q),pid mod q).',tested_tile_counts=counts,valid_program_ids_checked=checked,defaultFalse_preserves_generic_mapping=True,feature_rejects_other_group_sizes=True)
(R/'index-mapping-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_only=True,isolated_Bolt_worktree=True,frozen_Triton_source=str((R/'triton').resolve()),new_kernel_sha256=sha(kernel),parent_kernel_sha256=sha(C/'bolt_ops/boltops/gemm/triton/gemm_fp16.py'),native_sha256=sha(R/'native/libtriton.so'),parent_GPU_sources_unchanged=True,controls=2,features=8,index_mapping_proof_sha256=sha(R/'index-mapping-proof.json'),full_gold_and_independent_fresh_perf_required=True,no_PMD_admission=True),indent=2)+'\n')
print('Prepared175: isolated GROUP1 mapping kernel,2 default controls and8 source variants; mapping proof',checked,'valid ids; no model execution.')
