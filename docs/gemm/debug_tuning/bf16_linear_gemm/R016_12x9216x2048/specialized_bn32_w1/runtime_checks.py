"""Full-output checks for every dispatch, optionally using a fresh B each time.
Unique B is not a full cache flush. No globally installed code is modified.
"""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
import time


def worker(c,T,dest):
    import torch
    from triton.runtime import driver
    from triton.backends.hcu.compiler import HIPOptions
    assert driver.active.get_current_target().arch=='gfx946'
    assert c['layout']=='TT' and not c.get('poison_pad')
    m,n,k=(c[key] for key in ('M','N','K'))
    const,opts=T.split_config(c['config'])
    assert not(set(opts)-set(HIPOptions.__dataclass_fields__))
    for key,val in dict(GROUP_SIZE_M=1,NUM_KSPLIT=1,SPLITK_BLOCK_SIZE=0,
                        USE_MLS=False,USE_MATRIX_STORE=False,WARP_SPECIALIZE=False).items():
        const.setdefault(key,val)
    assert const['NUM_KSPLIT']==1
    spec=importlib.util.spec_from_file_location('r016_checked',str(T.OP_PATH))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    kernel=mod._gemm_bf16_generic_kernel
    torch.set_num_threads(1)
    gen=torch.Generator().manual_seed(c.get('seed',29))
    ac=torch.randn((m,k),generator=gen).to(torch.bfloat16)
    a=ac.cuda()
    count=c['dispatches'] if c.get('rotate_b') else 1
    host_b=[torch.randn((k,n),generator=gen).to(torch.bfloat16) for _ in range(count)]
    device_b=[b.cuda() for b in host_b]
    assert len({b.data_ptr() for b in device_b})==count
    h=torch.full((128+m*n+128,),123.,dtype=torch.bfloat16)
    h[128:-128]=float('nan')
    storage=h.cuda()
    out=storage[128:-128].view(m,n)
    grid=(T.num_tg(m,n,const),)
    def args(b):
        return (a,b,out,a,m,n,k,*a.stride(),*b.stride(),0,*out.stride())
    start=time.monotonic()
    compiled=kernel.warmup(*args(device_b[0]),grid=grid,**const,**opts)
    compiled._init_handles()
    asm=compiled.asm['amdgcn']
    (dest/'kernel.amdgcn').write_text(asm)
    norm='\n'.join(line for line in re.sub(r';[^\n]*','',asm).splitlines() if line.strip() and '.file' not in line)
    info=dict(grid=grid,compile_wall_s=time.monotonic()-start,shared=compiled.metadata.shared,
              n_regs=compiled.n_regs,n_spills=compiled.n_spills,num_warps=compiled.metadata.num_warps,
              asm_sha=hashlib.sha256(norm.encode()).hexdigest())
    T.dump(dest/'compile.json',info)
    if c.get('compile_only'):
        return
    checks=[]
    addresses=[]
    guard_tests=[]
    if c.get('checked_entry'):
        try:
            mod.gemm_r016(a,device_b[0],device_b[0][:m,:])
        except ValueError as error:
            assert 'overlap' in str(error),str(error)
            guard_tests.append('output/input alias rejected before launch')
        else:
            raise AssertionError('checked entry accepted aliased output')
    for i in range(c['dispatches']):
        j=i if c.get('rotate_b') else 0
        b=device_b[j]
        addresses.append(b.data_ptr())
        storage.copy_(h)
        torch.cuda.synchronize()
        if c.get('checked_entry'):
            assert mod.gemm_r016(a,b,out) is out
        else:
            kernel[grid](*args(b),**const,**opts)
        torch.cuda.synchronize()
        actual=out.cpu()
        raw=storage.cpu()
        finite=bool(torch.isfinite(actual).all())
        guard=bool((raw[:128]==123).all() and (raw[-128:]==123).all())
        ref=ac.float()@host_b[j].float()
        gold=ref.to(torch.bfloat16)
        mism=int((~torch.isclose(actual,gold,atol=.02,rtol=.01)).sum())
        floor=float((gold.float()-ref).abs().mean())
        mean=float((actual.float()-ref).abs().mean())
        passed=finite and guard and mism==0 and mean<=1.05*floor+1e-6
        checks.append(dict(dispatch=i,status='PASS' if passed else 'FAIL',finite=finite,guard=guard,
                           sampled_points=m*n,mismatches=mism,mean_abs=mean,rounding_floor=floor,
                           bitwise_equal_to_bf16_gold=bool(torch.equal(actual,gold))))
    T.dump(dest/'dispatch_checks.json',checks)
    final=dict(checks[-1],**info)
    final['status']='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL'
    final['all_dispatches_checked']=len(checks)
    T.dump(dest/'check.json',final)
    T.dump(dest/'cache_protocol.json',dict(rotate_b=bool(c.get('rotate_b')),
        independent_b_buffers=count,b_device_addresses=addresses,checked_entry=bool(c.get('checked_entry')),
        entry_guard_tests=guard_tests,
        protocol='unique B per dispatch' if c.get('rotate_b') else 'same B repeated',
        caveat='No explicit cache flush. All B buffers remain alive and are initialized before first dispatch.'))
