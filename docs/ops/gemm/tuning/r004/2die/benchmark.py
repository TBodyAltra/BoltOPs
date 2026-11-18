from pathlib import Path
import os,sys,json,re,statistics,hashlib,traceback,time,configparser,importlib
def _perf_sections(data):
    import struct
    assert data[:4] == b'\x7fELF'
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, index = struct.unpack_from('<HHH', data, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size) for i in range(count)]
    table = headers[index]
    names = data[table[4]:table[4] + table[5]]
    result = {}
    for header in headers:
        name = names[header[0]:].split(b'\0', 1)[0].decode()
        if name in ('.text', '.rodata'):
            result[name] = hashlib.sha256(data[header[4]:header[4] + header[5]]).hexdigest()
    assert '.text' in result
    return result


def _perf_embedded_hsaco(data):
    import struct
    import zlib
    assert data[1:8] == b'HYGPERF'
    position = 24
    while position + 24 <= len(data):
        kind, offset, size = struct.unpack_from('<QQQ', data, position)
        assert size > 0
        if kind == 8:
            compressed_size = struct.unpack_from('<Q', data, position + 48)[0]
            return zlib.decompress(data[position + size:position + size + compressed_size])
        position += size
    raise RuntimeError('PERF contains no embedded HSACO')


def capture_single_tg_perf(root, layout, record, repository):
    """Three validated CU0 captures, original MNK/strides and recorded HSACO."""
    output = root / (layout + '-single-tg-perf')
    output.mkdir(exist_ok=False)
    os.chdir(output)
    sys.path[:0] = [str(root), str(repository)]
    os.environ['TRITON_CACHE_DIR'] = str(root / ('cache_' + layout))
    os.environ['TRITON_HOME'] = str(output / 'triton-home')
    report = dict(status='RUNNING', layout=layout, shape=record['shape'], grid=[1],
                  scope='single TG on CU0 of the scheduled die; original complete MNK/strides and full K; first output tile only; separate from full-grid timing',
                  seed=59, launches=[], started=time.time(),
                  environment_overrides=dict(GPU_DFLAGS="['StatLog','TT0','Perf']", HSA_TOOLS_LIB='/opt/rocm/lib/xprofiler/libperf_gen_helper.so'),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  kernel_source_sha256=hashlib.sha256((root / 'kernel.py').read_bytes()).hexdigest())
    def save():
        (output / 'perf_measurement.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    save()
    try:
        import torch
        from kernel import candidate_kernel
        torch.set_num_threads(1)
        assert os.environ['GPU_CHIP'] == 'sbx2'
        assert torch.cuda.get_device_properties(0).multi_processor_count == 96
        m, n, k = record['shape']
        cfg = {key: tuple(value) if isinstance(value, list) else value for key, value in record['config'].items() if key != 'kernel'}
        cfg['B_K_CONTIGUOUS'] = layout == 'TN'
        bm, bn = cfg['BLOCK_SIZE_M'], cfg['BLOCK_SIZE_N']
        generator = torch.Generator().manual_seed(59)
        ac = (torch.rand((m, k), generator=generator) - .5).bfloat16()
        bc = (torch.rand((k, n), generator=generator) - .5).bfloat16()
        a = ac.cuda()
        b = bc.T.contiguous().cuda().T if layout == 'TN' else bc.cuda()
        template = torch.full((256 + m * n + 256,), 123., dtype=torch.bfloat16)
        template[256:-256].fill_(float('nan'))
        storage = torch.empty(template.shape, dtype=torch.bfloat16, device='cuda')
        c = storage[256:-256].view(m, n)
        selected = torch.zeros((m, n), dtype=torch.bool)
        selected[:bm, :bn] = True
        reference = (ac[:bm].float() @ bc[:, :bn].float()).bfloat16().float()
        handle = candidate_kernel.warmup(a, b, c, M=m, N=n, K=k, grid=(1,), **cfg)
        handle._init_handles()
        binary = handle.kernel
        digest = hashlib.sha256(binary).hexdigest()
        whole_binary_equal = digest == record['binary_sha256']
        expected_sections = record.get('single_tg_perf', {}).get('sections_sha256')
        assert whole_binary_equal or (expected_sections and _perf_sections(binary) == expected_sections), ('PERF executable differs from measured full-grid kernel', digest, record['binary_sha256'])
        assembly = handle.asm['amdgcn']
        if whole_binary_equal:
            assert hashlib.sha256(assembly.encode()).hexdigest() == record['assembly_sha256']
        resources = {name: int(re.search(r'\.' + name + r':\s*(\d+)', assembly)[1]) for name in ['sgpr_count', 'vgpr_count', 'sgpr_spill_count', 'vgpr_spill_count']}
        assert resources['sgpr_spill_count'] == resources['vgpr_spill_count'] == 0
        assert handle.metadata.shared <= 131072
        (output / 'kernel.hsaco').write_bytes(binary)
        (output / 'kernel.amdgcn').write_text(assembly)
        report.update(config=record['config'], resources=resources, shared_bytes=handle.metadata.shared,
                      binary_sha256=digest, full_grid_binary_sha256=record['binary_sha256'],
                      full_and_single_hsaco_equal=whole_binary_equal, full_and_single_text_rodata_equal=True,
                      sections_sha256=_perf_sections(binary), output_tile=[bm, bn],
                      input_strides=dict(A=list(a.stride()), B=list(b.stride()), C=list(c.stride())))
        save()
        for repeat in range(3):
            storage.copy_(template)
            torch.cuda.synchronize()
            before = set(Path('m5out/0').glob('*/stats.txt'))
            start = time.time()
            launched = candidate_kernel[(1,)](a, b, c, M=m, N=n, K=k, **cfg)
            assert launched is handle
            torch.cuda.synchronize()
            elapsed = time.time() - start
            paths = set(Path('m5out/0').glob('*/stats.txt')) - before
            assert len(paths) == 1, paths
            path = paths.pop()
            stats = path.read_text()
            counts = {key: int(value) for key, value in re.findall(r'^(system.shaders\d+[.]CUs\d+[.]completedWGs)\s+(\d+)', stats, re.M)}
            assert len(counts) == 96 and sum(counts.values()) == 1
            active = [key for key, count in counts.items() if count]
            assert len(active) == 1 and re.fullmatch(r'system[.]shaders[01][.]CUs0+[.]completedWGs', active[0]), active
            ini = configparser.RawConfigParser(strict=False)
            ini.read('m5out/config.ini')
            assert ini.getint('gpu_configuration', 'num_chip') == 2
            assert ini.getint('gpu_configuration', 'gpu_clock') == 455
            for die in ['0', '1']:
                assert ini.getint('system.shaders' + die + '.clk_domain', 'clock') == 455
            host = storage.cpu()
            guards = bool((host[:256] == 123).all() and (host[-256:] == 123).all())
            actual = host[256:-256].view(m, n).float()
            untouched = bool(torch.isnan(actual[~selected]).all())
            tile = actual[:bm, :bn]
            error = (tile - reference).abs()
            bad = (~tile.isfinite()) | (error > .001 + .01 * reference.abs())
            accuracy = dict(elements=tile.numel(), max_abs=float(error.max()), mismatches=int(bad.sum()),
                            nonfinite=int((~tile.isfinite()).sum()), guards_ok=guards,
                            untouched_elements=int((~selected).sum()), untouched_ok=untouched)
            assert not accuracy['mismatches'] and not accuracy['nonfinite'] and guards and untouched, accuracy
            perfs = sorted(path.parent.glob('*_python*.perf'))
            assert len(perfs) == 1, perfs
            embedded = _perf_embedded_hsaco(perfs[0].read_bytes())
            assert embedded == binary, 'PERF embedded HSACO differs from full-grid HSACO'
            cycles = float(re.search(r'^system.gcSimCycles\s+([\d.eE+-]+)', stats, re.M)[1])
            sample = dict(repeat=repeat, kernel_cycles=cycles, kernel_us=cycles * .000455,
                          completed_TGs=1, active_CUs=1, per_die_TGs={die: int(('system.shaders' + die + '.') in active[0]) for die in ['0', '1']},
                          stats_path=str(path), perf_path=str(perfs[0]),
                          perf_sha256=hashlib.sha256(perfs[0].read_bytes()).hexdigest(),
                          embedded_hsaco_verified=True, accuracy=accuracy, simulator_wall_seconds=elapsed)
            report['launches'].append(sample)
            save()
            print(json.dumps(sample), flush=True)
        selected_sample = min(report['launches'][1:], key=lambda sample: sample['kernel_cycles'])
        import shutil
        shutil.copyfile(output / selected_sample['perf_path'], output / 'single_tg.perf')
        report.update(status='PASS', selected_repeat=selected_sample['repeat'],
                      selection='minimum cycles of repeats 1 and 2; repeat 0 excluded',
                      selected=selected_sample, finished=time.time())
        save()
    except BaseException:
        report.update(status='ERROR', error=traceback.format_exc(), finished=time.time())
        save()
        raise


R=Path(__file__).resolve().parent;layout=sys.argv[1];record=json.loads((R/layout.lower()/'record.json').read_text())
if 'R004_TUNE' not in os.environ:
 env=dict(os.environ)
 for k,v in record['environment']['environment_variables'].items():
  if v is None:env.pop(k,None)
  else:env[k]=v

 if '--single-tg-perf' in sys.argv:
  env['GPU_DFLAGS']="['StatLog','TT0','Perf']";env['HSA_TOOLS_LIB']='/opt/rocm/lib/xprofiler/libperf_gen_helper.so'
 env['R004_TUNE']='1';os.execve(sys.executable,[sys.executable,*sys.argv],env)
repo=Path(sys.argv[2]).resolve() if len(sys.argv)>2 and sys.argv[2]!='--single-tg-perf' else next((p for p in R.parents if (p/'boltops/gemm/config.py').exists()),R/'repo')
if '--single-tg-perf' in sys.argv:
 capture_single_tg_perf(R,layout,record,repo)
 raise SystemExit(0)
out=R/(layout+'-standalone');out.mkdir(exist_ok=False);os.chdir(out);sys.path[:0]=[str(R),str(repo)];os.environ['TRITON_CACHE_DIR']=str(R/('cache_'+layout));os.environ['TRITON_HOME']=str(out/'triton-home')
result=dict(status='RUNNING',stage='initialize',shape=[6656,64,2048],layout=layout,environment=record['environment'],variants={},started=time.time(),scope='complete MNK/grid, no split K; identical A/B/C addresses throughout each layout; contiguous block of9 dispatches per config, median; first sample may be cold after config switch')
def save():
 p=out/'progress.tmp';p.write_text(json.dumps(result,indent=2,default=str)+'\n');p.replace(out/'progress.json')
save()
try:
 import torch
 from kernel import candidate_kernel
 from boltops.gemm.gluon.gemm_a16w16 import _gemm_a16w16_tt_kernel
 public=importlib.import_module('boltops.gemm.gemm_a16w16')
 torch.set_num_threads(1);assert torch.cuda.get_device_properties(0).multi_processor_count==96
 M,N,K=record['shape'];gen=torch.Generator().manual_seed(59);ac=(torch.rand((M,K),generator=gen)-.5).bfloat16();bc=(torch.rand((K,N),generator=gen)-.5).bfloat16();golden=(ac.float()@bc.float()).bfloat16().float()
 A=ac.cuda();B=bc.T.contiguous().cuda().T if layout=='TN' else bc.cuda();storage=torch.full((256+M*N+256,),123.,dtype=torch.bfloat16,device='cuda');C=storage[256:-256].view(M,N);result['input_strides']=dict(A=list(A.stride()),B=list(B.stride()));result['input_addresses']=dict(A=A.data_ptr(),B=B.data_ptr(),C=C.data_ptr());handles={};runners={}
 variants=[dict(name='baseline',family='candidate' if layout=='TN' else 'production',config=record['baseline']['config']),dict(name=record['best'],family='candidate',config=record['config']),dict(name=record['interleaved_comparison']['best'],family='candidate',config=record['interleaved_comparison']['config'])]
 # Replay the no-spill ring baseline and the two recorded finalists.
 def setup(v):
  name=v['name'];cfg=dict(v['config']);d=out/name;d.mkdir();r=dict(config=v['config'],family=v['family'],validation=[],samples=[]);result['variants'][name]=r
  if v['family']=='public_auto':
   r['family']='public_auto';r['config']=None;r['grid']=None
   def launch():public.gemm_a16w16(A,B.T,out=C,backend='auto')
   runners[name]=launch;return
  symbol=cfg.pop('kernel',None)
  fn=_gemm_a16w16_tt_kernel if v['family']=='production' else candidate_kernel
  cfg={k:tuple(x) if isinstance(x,list) else x for k,x in cfg.items()};cfg['B_K_CONTIGUOUS']=layout=='TN';grid=M//cfg['BLOCK_SIZE_M']*(N//cfg['BLOCK_SIZE_N']);r['grid']=grid
  h=fn.warmup(A,B,C,M=M,N=N,K=K,grid=(grid,),**cfg);h._init_handles();asm=h.asm['amdgcn'];native=lambda s:int(re.search(r'\.'+s+r':\s*(\d+)',asm)[1]);r['resources']=dict(SGPR=native('sgpr_count'),VGPR=native('vgpr_count'),SGPR_spills=native('sgpr_spill_count'),VGPR_spills=native('vgpr_spill_count'),LDS=h.metadata.shared,waves=h.metadata.num_warps);r['binary_sha256']=hashlib.sha256(h.kernel).hexdigest();r['metadata']=h.metadata._asdict();(d/'kernel.amdgcn').write_text(asm);(d/'kernel.hsaco').write_bytes(h.kernel)
  if r['resources']['SGPR_spills'] or r['resources']['VGPR_spills']:r['status']='SPILL_REJECTED';return
  assert h.metadata.shared<=131072
  handles[name]=h
  def launch():
   z=fn[(grid,)](A,B,C,M=M,N=N,K=K,**cfg);assert z is h
  runners[name]=launch
 def run(name):
  torch.cuda.synchronize();r=result['variants'][name];before=set(Path('m5out/0').glob('*/stats.txt'));runners[name]();torch.cuda.synchronize();paths=sorted(set(Path('m5out/0').glob('*/stats.txt'))-before,key=lambda p:int(p.parent.name));assert len(paths)==1,('dispatch count',paths)
  path=paths[0];s=path.read_text();rows=re.findall(r'^(system.shaders\d+)[.]CUs\d+[.]completedWGs\s+(\d+)',s,re.M);assert len(rows)==96
  per_die={d:dict(TGs=sum(int(v) for k,v in rows if k==d),active_CUs=sum(int(v)>0 for k,v in rows if k==d)) for d in {k for k,v in rows}};grid=sum(x['TGs'] for x in per_die.values());assert r['grid'] is None or grid==r['grid'];assert len(per_die)==2 and all(x['TGs']>0 for x in per_die.values())
  ini=configparser.RawConfigParser(strict=False);ini.read('m5out/config.ini');assert ini.getint('gpu_configuration','num_chip')==2
  for d in per_die:assert ini.getint(d+'.clk_domain','clock')==455
  starts=[float(v) for v in re.findall(r'^system.shaders\d+[.]firstWaveStartTick\s+([\d.eE+-]+)',s,re.M)];ends=[float(v) for v in re.findall(r'^system.shaders\d+[.]lastWaveEndTick\s+([\d.eE+-]+)',s,re.M)]
  raw=storage.cpu();assert bool((raw[:256]==123).all()) and bool((raw[-256:]==123).all());x=raw[256:-256].view(M,N).float();diff=(x-golden).abs();bad=(~torch.isfinite(x))|(diff>.001+.01*golden.abs());accuracy=dict(elements=x.numel(),mismatches=int(bad.sum()),nonfinite=int((~torch.isfinite(x)).sum()),max_abs=float(diff.max()));assert not bool(bad.any()),accuracy
  sample=dict(kernel_us=float(re.search(r'^system.gcSimCycles\s+([\d.eE+-]+)',s,re.M)[1])*.000455,shader_us=(max(ends)-min(starts))/1e6,per_die=per_die,active_CUs=sum(x['active_CUs'] for x in per_die.values()),completed_TGs=grid,accuracy=accuracy,stats_path=str(path));print(json.dumps(dict(name=name,sample=sample)),flush=True);return sample
 for v in variants:
  result.update(stage='compile',current=v['name']);save();setup(v);assert v['name'] in runners;result['variants'][v['name']]['status']='PASS';save()
 selected=[v['name'] for v in variants];result.update(stage='confirm',selected=selected);save()
 for name in selected:result['variants'][name]['confirmation']=dict(validation=[],samples=[])
 # Fresh random input and a full NaN reset before validation.
 for vi,seed in enumerate([59]):
  gen=torch.Generator().manual_seed(seed);ac=(torch.rand((M,K),generator=gen)-.5).bfloat16();bc=(torch.rand((K,N),generator=gen)-.5).bfloat16()
  if vi==1:ac.fill_(1);bc.fill_(1)
  A.copy_(ac);B.copy_(bc);golden=(ac.float()@bc.float()).bfloat16().float()
  for name in selected:
   result['current']=dict(name=name,phase='validation',index=vi);save();C.fill_(float('nan'));result['variants'][name]['confirmation']['validation'].append(run(name));save()
 # Restore the timing input before nine consecutive dispatches per config.
 gen=torch.Generator().manual_seed(59);ac=(torch.rand((M,K),generator=gen)-.5).bfloat16();bc=(torch.rand((K,N),generator=gen)-.5).bfloat16();A.copy_(ac);B.copy_(bc);golden=(ac.float()@bc.float()).bfloat16().float()
 for name in selected:
  for i in range(9):
   result['current']=dict(name=name,phase='timing',index=i);save();result['variants'][name]['confirmation']['samples'].append(run(name));save()
 for name in selected:
  r=result['variants'][name];us=statistics.median(s['kernel_us'] for s in r['confirmation']['samples']);r['performance']=dict(kernel_us=us,kernel_tops=2*M*N*K/(us*1e6),bandwidth_TB_s=2*(M*K+K*N+M*N)/(us*1e6),mmac_util_percent=100*2*M*N*K/(us*1e6)/864.2109890109891)
 best=min(selected,key=lambda n:result['variants'][n]['performance']['kernel_us']);baseline=result['variants']['baseline']['performance']['kernel_us'];result.update(status='COMPLETE',stage='complete',best=best,improvement_percent=100*(baseline-result['variants'][best]['performance']['kernel_us'])/baseline,finished=time.time());save();(out/'result.json').write_text((out/'progress.json').read_text())
except BaseException:
 result.update(status='ERROR',error=traceback.format_exc(),finished=time.time());save();traceback.print_exc()
