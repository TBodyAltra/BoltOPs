# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Replay R005 with BF16-valued B scales promoted to FP32 before launch.

Run in an empty artifact directory under the recorded PMD environment:
python benchmark.py job.json
job.json: {"shape": [64,2048,4096], "config": {"BN":64}, "repeats":3}
Use shape [64,64,4096] and PERF environment flags for a genuine single TG.
"""
from pathlib import Path
import json,sys,time,traceback,hashlib,configparser,re,os

job=json.loads(Path(sys.argv[1]).read_text())
result=dict(job=job,status='STARTING',measurements=[])

def save():
    Path('result.tmp').write_text(json.dumps(result,indent=2,default=str))
    Path('result.tmp').replace('result.json')

def read_stats(expected):
    paths=sorted(Path('m5out/0').glob('*/stats.txt'),key=lambda p:int(p.parent.name))
    p=paths[-1];values={}
    for line in p.read_text().splitlines():
        parts=line.split()
        if len(parts)>1:
            try:values[parts[0]]=float(parts[1])
            except ValueError:pass
    completed=sum(v for k,v in values.items() if re.fullmatch(r'system.shaders.CUs[0-9]+.completedWGs',k))
    assert completed==expected,(completed,expected)
    config=configparser.RawConfigParser(strict=False);config.read('m5out/config.ini')
    period=int(config.get('system.shaders.clk_domain','clock'))
    data=dict(cycles=values['system.gcSimCycles'],shader_cycles=(values['system.shaders.lastWaveEndTick']-values['system.shaders.firstWaveStartTick'])/period,hostSeconds=values['system.hostSeconds'],clock_ps=period,completed_TGs=int(completed),path=str(p))
    for typ in ('VALU','MMOP','LDS','VMEM','FLAT','SCA','MISC'):
        data[typ]=sum(v for k,v in values.items() if re.fullmatch(r'system.shaders.CUs[0-9]+.numInstrTypeExecuted::'+typ,k))
    for key in ('waitVmCounter','waitLgkmCounter'):
        data[key]=sum(v for k,v in values.items() if re.fullmatch(r'system.shaders.CUs[0-9]+.SIMD[0-9]+.'+key,k))
    data['memory_counters']={k:v for k,v in values.items() if any(s in k.lower() for s in ('l2','tcc')) and v}
    return data

save()
try:
    import torch,triton
    torch.set_num_threads(1)
    m,n,k=job.get('shape',[64,2048,4096]);cfg=job['config']
    gen=torch.Generator().manual_seed(job.get('seed',37))
    ac=(torch.randn((m,k),generator=gen)*.5).to(torch.float8_e4m3fn)
    bc=(torch.randn((n,k),generator=gen)*.5).to(torch.float8_e4m3fn)
    asc=(torch.rand((m,k//128),generator=gen)*1.5+.25).float()
    bsc=(torch.rand(((n+127)//128,k//128),generator=gen)*1.5+.25).bfloat16().float()
    reference=torch.zeros((m,n),dtype=torch.float32)
    for group in range(k//128):
        dot=ac[:,group*128:(group+1)*128].float()@bc[:,group*128:(group+1)*128].float().T
        reference+=dot*asc[:,group,None]*bsc[:,group].float().repeat_interleave(128)[:n][None,:]
    a,b,sa,sb=[t.cuda() for t in (ac,bc,asc,bsc)]
    storage=torch.full((m*n+512,),123.,dtype=torch.bfloat16,device='cuda');out=storage[256:-256].view(m,n)
    grid=(m//64,n//cfg['BN']);total=grid[0]*grid[1]
    from boltops.gemm.gluon.gemm_a8w8_blockscale import _gemm_a8w8_blockscale_as_fp32_bs_fp32_recipe_a_1x128_recipe_b_128x128_bm64_bn64_bk128_kernel as fn
    args=(a,b,sa,sb,out)
    opts=dict(M=m,N=n,K=k,GROUP_N=128,GROUP_K=128,BLOCK_SIZE_M=64,BLOCK_SIZE_N=64,BLOCK_SIZE_K=128,num_warps=1,num_stages=2,wasp_partition_warps=(1,4),wasp_wdra=False)
    t=time.time();handle=fn.warmup(*args,grid=grid,**opts)
    result.update(compile_seconds=time.time()-t,launch_options=opts,grid=list(grid))
    for ext in ('ttgir','llir','amdgcn'):
        Path('kernel.'+ext).write_text(handle.asm[ext])
    Path('kernel.hsaco').write_bytes(handle.asm['hsaco']);handle._init_handles()
    props=torch.cuda.get_device_properties(0)
    from triton.backends.hcu.wasp import read_partition_registers
    result['partition_registers']=read_partition_registers(handle.asm['amdgcn'])
    result['branch_spill']=any(v['vgpr_spill_count'] for v in (result['partition_registers'] or {}).values())
    result.update(status='COMPILED',metadata=handle.metadata._asdict(),VGPR=handle.n_regs,spill=handle.n_spills,LDS=handle.metadata.shared,ISA_sha256=hashlib.sha256(handle.asm['amdgcn'].encode()).hexdigest(),environment=dict(triton_file=triton.__file__,torch_version=torch.__version__,num_cus=props.multi_processor_count,variables={k:v for k,v in os.environ.items() if k.startswith(('TRITON_','GPU_','HIP_','AMDGCN_','LLVM_','OMP_','MKL_','OPENBLAS_'))}))
    save()
    if job.get('compile_only'):result['status']='COMPILE_PASS'
    elif handle.n_spills or result['branch_spill']:result['status']='SKIP_SPILL'
    else:
        for rep in range(job.get('repeats',1)+1):
            out.fill_(float('nan'));torch.cuda.synchronize();t=time.time()
            h=fn[grid](*args,**opts);assert h is handle
            torch.cuda.synchronize();wall=time.time()-t
            guards=bool((storage[:256].cpu()==123).all()) and bool((storage[-256:].cpu()==123).all())
            actual=out.cpu().float();err=(actual-reference).abs()
            bad=(~torch.isfinite(actual))|(err>.01+.02*reference.abs())
            result['measurements'].append(dict(repeat=rep,warmup=rep==0,wall_seconds=wall,stats=read_stats(total),accuracy=dict(max_abs=float(err.max()),mismatches=int(bad.sum()),nonfinite=int((~torch.isfinite(actual)).sum()),guards_ok=guards)))
            save()
            if bad.any() or not guards:
                torch.save(dict(a=ac,b=bc,a_scale=asc,b_scale=bsc,reference=reference,actual=actual,bad=bad),'failure.pt')
                raise AssertionError(result['measurements'][-1]['accuracy'])
        if n == 2048:
            from boltops.gemm import gemm_a8w8_blockscale
            public = gemm_a8w8_blockscale(a, b, sa, sb)
            torch.cuda.synchronize()
            torch.testing.assert_close(public.cpu().float(), actual, atol=0, rtol=0)
            result['public_auto_exact_match'] = True
        result['status']='PASS'
except BaseException:
    result.update(status='ERROR',error=traceback.format_exc());traceback.print_exc()
finally:
    save();print(json.dumps(result,default=str),flush=True)
if result['status']=='ERROR':raise SystemExit(1)
