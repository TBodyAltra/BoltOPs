"""Measure full-size FP8 GEMMs through the public API on two-die PMD."""
import hashlib
import importlib
import json
from pathlib import Path
import re
import statistics
import sys
import time


def read_stats(path):
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) > 1:
            try:
                values[fields[0]] = float(fields[1])
            except ValueError:
                pass
    counts = {}
    for key, value in values.items():
        match = re.search(r'\.shaders(\d*)\.CUs(\d+)\.completedWGs$', key)
        if match:
            counts.setdefault(match[1] or '0', {})[match[2]] = int(value)
    clocks = [v for k, v in values.items() if re.search(r'\.shaders\d*\.clk_domain\.clock$', k)]
    if sorted(map(len,counts.values())) != [48,48] or not clocks or set(clocks) != {455}:
        raise RuntimeError('expected PMD sbx2 with 48 CU per die and 455 ps GC')
    return dict(kernel_cycles=values['system.gcSimCycles'],
                kernel_us=values['system.gcSimCycles']*clocks[0]/values['simFreq']*1e6,
                active_CUs=sum(v>0 for cu in counts.values() for v in cu.values()),
                completed_TGs_per_die={d:sum(cu.values()) for d,cu in counts.items()},
                stats_path=str(path))


def main():
    import torch
    from boltops.gemm.config import get_gemm_config
    from boltops.gemm.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    from boltops.utility.arch_info import get_device_info

    job = json.loads(Path(sys.argv[1]).read_text())
    m,n,k = job['shape']
    device = get_device_info()
    if device.arch != 'gfx946' or device.cu_count != 96:
        raise RuntimeError('expected gfx946 96 CU')
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(job['seed'])
    a = (torch.randn((m,k),generator=g)*.5).to(torch.float8_e4m3fn)
    w = (torch.randn((n,k),generator=g)*.5).to(torch.float8_e4m3fn)
    sa = torch.rand((m,k//128),generator=g)*.2+.05
    sw = (torch.rand((n//128,k//128),generator=g)*.2+.05).bfloat16().float()
    golden = torch.zeros((m,n),dtype=torch.float32)
    for group in range(k//128):
        s = slice(group*128,(group+1)*128)
        dot = (a[:,s].float() @ w[:,s].float().T)*sa[:,group,None]
        golden.addcmul_(dot,sw[:,group].repeat_interleave(128)[None,:])
    golden = golden.bfloat16()
    # Upload physical storage before taking the TT view.
    wg = w.T.contiguous().cuda().T if job['layout']=='tt' else w.cuda()
    inputs = (a.cuda(),wg,sa.cuda(),sw.cuda())
    strides = {name:list(value.stride()) for name,value in zip(('x','w','x_scale','w_scale'),inputs)}
    assert tuple(wg.stride()) == ((1,n) if job['layout']=='tt' else (k,1))
    sentinel = -4096.
    storage = torch.full((m+2,n),sentinel,dtype=torch.bfloat16).cuda()
    out = storage[1:-1]
    torch.cuda.synchronize()
    cfg = job['config']
    if cfg is None:
        cfg,_ = get_gemm_config('GEMM-A8W8_BLOCKSCALE',m,n,k,backend='triton',arch='gfx946',cu_num=96)
    if job['backend'] in ('gluon', 'gluon-tt-small'):
        mod = importlib.import_module('boltops.gemm.gluon.gemm_a8w8_blockscale')
        kernel = getattr(mod,cfg['kernel'])
    else:
        mod = importlib.import_module('boltops.gemm.triton.gemm_a8w8_blockscale_fp8')
        kernel = mod._w8a8_triton_block_scaled_mm
    launches = []
    from boltops.gemm.gluon.gemm_a8w8_blockscale import gemm_a8w8_blockscale as gluon_launch
    from boltops.gemm.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale as triton_launch

    def launch():
        options = dict(a_scale=inputs[2], b_scale=inputs[3], block_size=(128,128), device=device)
        if job['config'] is None:
            assert gemm_a8w8_blockscale(*inputs,out=out,backend='triton') is out
        elif job['backend'] == 'gluon-tt-small':
            parameters = {k: tuple(v) if isinstance(v, list) else v for k,v in cfg.items() if k!='kernel'}
            kernel[(m//64,n//cfg['BLOCK_SIZE_N'])](*inputs,out,M=m,N=n,K=k,GROUP_N=128,GROUP_K=128,
                                                   B_K_CONTIGUOUS=False,**parameters)
        elif job['backend'] == 'gluon':
            gluon_launch(*inputs[:2],out,config=cfg.copy(),gluon_layout=job['layout'].upper(),**options)
        else:
            triton_launch(*inputs[:2],out,config=cfg,**options)

    original = kernel.run
    compiled = []
    def track(*args,**kwargs):
        handle = original(*args,**kwargs)
        compiled.append(handle)
        return handle
    kernel.run = track
    expected_tgs = cfg.get('NUM_SMS') or ((m+cfg['BLOCK_SIZE_M']-1)//cfg['BLOCK_SIZE_M']*
                                       ((n+cfg.get('BLOCK_SIZE_N',128)-1)//cfg.get('BLOCK_SIZE_N',128)))
    report = dict(job=job,config=cfg,operand_strides=strides,status='RUNNING',launched_TGs=expected_tgs,
                  launches=launches)
    report_path = Path('result.json')
    try:
        for repeat in range(job['repeats']):
            out.copy_(torch.full((m,n), torch.nan, dtype=torch.bfloat16))
            torch.cuda.synchronize()
            prior = set(Path('m5out/0').glob('*/stats.txt'))
            start = time.monotonic()
            launch()
            torch.cuda.synchronize()
            wall = time.monotonic()-start
            new = set(Path('m5out/0').glob('*/stats.txt'))-prior
            if len(new)!=1 or len(compiled)!=repeat+1:
                raise RuntimeError('expected one dispatch of the selected kernel')
            handle = compiled[-1]
            if repeat==0:
                hsaco = handle.asm['hsaco']
                Path('kernel.hsaco').write_bytes(hsaco)
                Path('kernel.amdgcn').write_text(handle.asm['amdgcn'])
                report.update(kernel=handle.metadata.name,hsaco_sha256=hashlib.sha256(hsaco).hexdigest(),
                              shared_bytes=handle.metadata.shared,spills=handle.n_spills,
                              partition_registers_actual=getattr(handle.metadata,'wasp_partition_regs_actual',None))
            actual = storage.cpu()
            # Validate in row chunks to bound memory usage on large prefill cases.
            mismatch=nonfinite=outside=0
            max_abs=0.
            for row in range(0,m,128):
                ref=golden[row:row+128].float()
                got=actual[1+row:1+min(m,row+128)].float()
                error=(got-ref).abs()
                mismatch+=int((got!=ref).sum())
                nonfinite+=int((~got.isfinite()).sum())
                outside+=int((error>.001+.01*ref.abs()).sum())
                max_abs=max(max_abs,float(error.max()))
            canary=bool((actual[0]==sentinel).all() and (actual[-1]==sentinel).all())
            sample=read_stats(new.pop())
            launches.append(dict(sample,repeat=repeat,host_wall_seconds=wall,
                accuracy=dict(elements=m*n,max_abs=max_abs,mismatch=mismatch,nonfinite=nonfinite,
                              outside_tolerance=outside,canary_ok=canary)))
            report_path.write_text(json.dumps(report,indent=2))
            if (sum(sample['completed_TGs_per_die'].values()) != expected_tgs
                    or not all(sample['completed_TGs_per_die'].values()) or nonfinite or outside or not canary):
                raise RuntimeError('full-size grid, numeric or guard validation failed')
        times=[s['kernel_us'] for s in launches]
        times=times[1:] if len(times)>1 else times
        us=statistics.median(times)
        report.update(status='PASS',kernel_us=us,kernel_tops=2*m*n*k/(us*1e6))
    except BaseException:
        report['status']='ERROR'
        raise
    finally:
        report_path.write_text(json.dumps(report,indent=2))
        kernel.run=original


if __name__=='__main__':
    main()
