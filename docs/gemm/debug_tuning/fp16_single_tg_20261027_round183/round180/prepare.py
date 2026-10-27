"""Isolated backend-only SGPR liveness experiments; fixed GPU kernel and native95."""
import pathlib,json,hashlib,subprocess,itertools,difflib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round175-group1-index-cpu';F=B/'single-tg-pipeline-20261027-round177-group1-index-confirm'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=load(F/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
assert load(F/'confirmation.json')['resource_alternative_validated']
parent=load(C/'cpu-screen.json');assert parent['passed'] and all(sha(pathlib.Path(p))==h for p,h in parent['source_hashes'].items())
assert not(R/'triton').exists()
subprocess.run(['git','-C',str(C/'triton'),'worktree','add','-b','perf/sgpr-liveness-round180-triton',str(R/'triton'),'46f0bed3561364b63ba4046d9c8c0cb70380ea30'],check=True)
subprocess.run(['git','-C',str(R/'triton'),'apply',str(C/'triton-source.patch')],check=True)
(R/'native').mkdir()
(R/'native/libtriton.so').symlink_to((C/'native/libtriton.so').resolve());assert sha(R/'native/libtriton.so')=='3fc2d388816202805f711d621b27231f3a103e94d67d54982340248caa537990'
for p in (C/'triton/python/triton').rglob('*'):
    if not p.is_symlink():continue
    target=p.readlink()
    try:target=R/'triton'/target.relative_to(C/'triton')
    except ValueError:target=p.resolve()
    if p.name=='libtriton.so':target=R/'native/libtriton.so'
    d=R/'triton'/p.relative_to(C/'triton');d.parent.mkdir(parents=True,exist_ok=True)
    if d.is_symlink():d.unlink()
    else:assert not d.exists(),str(d)
    d.symlink_to(target)
for p in (C/'triton/third_party/hcu/backend').glob('_cy_utils*.so'):
    d=R/'triton/third_party/hcu/backend'/p.name
    if d.is_symlink():assert d.resolve()==p.resolve()
    else:assert not d.exists();d.symlink_to(p.resolve())
p=R/'triton/python/triton/backends/hcu';assert p.is_symlink();p.unlink();p.symlink_to(R/'triton/third_party/hcu/backend')
for n in ['bolt_ops','deps','session.lock']:
    assert not(R/n).exists();(R/n).symlink_to((C/n).resolve())
p=R/'triton/third_party/hcu/backend/compiler.py';before=p.read_text();s=before
a='    misched_regpressure: bool = True\n';assert s.count(a)==1
s=s.replace(a,a+'    # Experimental gfx946 codegen-only liveness strategy. Zero preserves all flags.\n    hcu_sgpr_liveness_strategy: int = 0\n')
a='        mem_cluster = args.get("max_memory_cluster_dwords")\n';assert s.count(a)==1
s=s.replace(a,'''        sgpr_strategy = args.get("hcu_sgpr_liveness_strategy", 0)
        if type(sgpr_strategy) is not int or sgpr_strategy not in range(6):
            raise ValueError("hcu_sgpr_liveness_strategy must be an integer in [0,5]")
        args["hcu_sgpr_liveness_strategy"] = sgpr_strategy

'''+a)
a='        if not options.misched_regpressure:\n';assert s.count(a)==1
s=s.replace(a,'''        sgpr_strategy = options.hcu_sgpr_liveness_strategy
        if sgpr_strategy and options.arch != "gfx946":
            raise ValueError("Experimental SGPR liveness strategies require gfx946")
        # Discovered in the exact deployed clang18 --help-hidden. These change
        # argument lowering, sinking, or allocation, never resource limits.
        sgpr_flags = {
            0: [],
            1: ["-mllvm=-amdgpu-ir-lower-kernel-arguments=false"],
            2: ["-mllvm=-sink-insts-to-avoid-spills=true",
                "-mllvm=-machine-sink-cycle-limit=4096",
                "-mllvm=-machine-sink-load-instrs-threshold=4096",
                "-mllvm=-machine-sink-load-blocks-threshold=128"],
            3: ["-mllvm=-sgpr-regalloc=basic"],
            4: ["-mllvm=-enable-subreg-liveness=true"],
            5: ["-mllvm=-sgpr-regalloc=fast"],
        }
        options_args.extend(sgpr_flags[sgpr_strategy])

'''+a);p.write_text(s);compile(s,str(p),'exec')
(R/'backend-liveness.patch').write_text(''.join(difflib.unified_diff(before.splitlines(keepends=True),s.splitlines(keepends=True),fromfile='a/third_party/hcu/backend/compiler.py',tofile='b/third_party/hcu/backend/compiler.py')))
subprocess.run(['git','-C',str(R/'triton'),'diff','--check'],check=True)
subprocess.run(['git','-C',str(R/'triton'),'add','-N','.'],check=True)
(R/'triton-source.patch').write_bytes(subprocess.check_output(['git','-C',str(R/'triton'),'diff','HEAD','--binary']))
for n in ['bolt_ops-source.patch','native-reuse.json','compile_probe.py','index-mapping-proof.json','kernel-index-fix.patch']:(R/n).write_bytes((C/n).read_bytes())
(R/'env.sh').write_text((C/'env.sh').read_text().replace(str(C),str(R)).replace('175group1indexcpu','180sgprlivenesscpu'))
refs=load(C/'control-qualification.json')['controls'];assert len(refs)==2
(R/'references.json').write_text(json.dumps(refs,indent=2)+'\n');seed=load(pathlib.Path(refs[0]['directory'])/'job.json');jobs=[]
for strategy,simple in itertools.product(range(1,6),[False,True]):
    job=json.loads(json.dumps(seed));job.update(label='features',variant=f'sgpr_strategy{strategy}_simple{int(simple)}_UF4_EP1')
    job['config'].update(hcu_sgpr_liveness_strategy=strategy,SIMPLIFY_GROUP1=simple,LOOP_UNROLL_FACTOR=4,MMA_K_SPLIT=4,optimize_epilogue=True,num_warps=8,misched_regpressure=True,USE_MATRIX_STORE=False)
    jobs.append(job)
(R/'feature-plan.json').write_text(json.dumps(dict(CPU_only=True,controls=2,features=jobs,scope='256x128x128 TT,NoWASP/noWDRA/stage1/UF4/split4/warps8/EPTrue. Five targeted LLVM liveness strategies crossed with unchanged generic or qualified GROUP1 mapping. No raised SGPR/VGPR/LDS limits; no native binary changes. Strategy0 default must match old controls; actual ISA/resource novelty required before PMD.',no_PMD_admission=True),indent=2)+'\n')
(R/'reference-map.json').write_bytes((B/'single-tg-continuation-20261023/published/natural-stride-map.json').read_bytes())
H=B/'single-tg-pipeline-20261027-round174-sgpr-liveness-review';(R/'clang-help-proof.json').write_bytes((H/'clang-help-proof.json').read_bytes())
assert sha(R/'bolt_ops/boltops/gemm/triton/gemm_fp16.py')==load(C/'prepare-proof.json')['new_kernel_sha256'] and all(sha(pathlib.Path(p))==h for p,h in parent['source_hashes'].items())
(R/'prepare-proof.json').write_text(json.dumps(dict(passed=True,CPU_only=True,isolated_Triton_worktree=True,branch='perf/sgpr-liveness-round180-triton',kernel_source_unchanged=True,frozen_kernel_worktree=str((R/'bolt_ops').resolve()),native95_unchanged=True,native_sha256=sha(R/'native/libtriton.so'),backend_patch_sha256=sha(R/'backend-liveness.patch'),clang_help_proof_sha256=sha(R/'clang-help-proof.json'),parent_source_hashes_unchanged=True,controls=2,features=10,no_PMD_admission=True,actual_ISA_novelty_and_resource_checks_required=True),indent=2)+'\n')
print('Prepared180: isolated Python backend only;5 LLVM SGPR liveness strategies x2 GROUP1 modes;2 default controls; native95 and kernel175 frozen.')
