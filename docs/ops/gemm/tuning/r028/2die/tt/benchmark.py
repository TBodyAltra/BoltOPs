# Publication contains only record['artifacts']; runtime files stay outside it.
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
_COMPACT_PARSE_ONLY = False
def _compact_find_repo():
    anchors = [Path(__file__).resolve().parent, Path.cwd()]
    anchors.extend(Path(p).resolve() for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p)
    for anchor in anchors:
        for candidate in (anchor, *anchor.parents):
            if (candidate/'boltops').is_dir():
                return candidate
    return None
_COMPACT_FAMILY = 'compute'

"""Reproducible full-grid BF16 TT PMD experiments. Source path.sh before use."""
import argparse,builtins,getpass,hashlib,importlib.util,json,math,os,re,statistics,subprocess,sys,tempfile,time,traceback
from pathlib import Path
RUNNER_SOURCE=Path(__file__).read_bytes()
RUNNER_SHA256=hashlib.sha256(RUNNER_SOURCE).hexdigest()
HERE=Path(__file__).resolve().parent
REPO=_compact_find_repo()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(job):
 assert job['requirement']=='R028' and job['shape']==[512,4096,12288]
 assert job['dispatches']==3 and job['seed']==2059 and job['timed_seed']==59
 save('worker-process.json',dict(pid=os.getpid(),ppid=os.getppid(),sid=os.getsid(0),argv=sys.argv,cwd=str(Path.cwd()),lease=os.environ.get('OPTALL_PMD_LEASE'),preflight=os.environ.get('OPTALL_PREFLIGHT')))
 Path('worker.pid').write_text(str(os.getpid())+'\n')
 Path('runner.snapshot.py').write_bytes(RUNNER_SOURCE)
 save('runner-snapshot.json',dict(sha256=RUNNER_SHA256,path='runner.snapshot.py'))
 original=builtins.open; lock='/tmp/'+getpass.getuser()+'.history.log.lock'
 def private_open(file,*args,**kw):
  if isinstance(file,(str,bytes,os.PathLike)) and os.fsdecode(file)==lock:file=Path.cwd()/'history.lock'
  return original(file,*args,**kw)
 builtins.open=private_open
 sys.path.insert(0,str(REPO))
 import torch,triton
 from triton import knobs
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert os.environ['GPU_CHIP']=='sbx2'
 assert torch.cuda.get_device_properties(0).multi_processor_count==96
 assert triton.runtime.driver.active.get_current_target().arch=='gfx946'
 source_bytes=Path(job['source']).read_bytes();snapshot=Path.cwd()/'candidate.snapshot.py';snapshot.write_bytes(source_bytes)
 spec=importlib.util.spec_from_file_location('compute_candidate',snapshot);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 dependency_hashes={}
 for imported in list(sys.modules.values()):
  file=getattr(imported,'__file__',None)
  if file and Path(file).is_file() and Path(file).resolve().is_relative_to(REPO/'boltops'):
   path=Path(file).resolve();relative=path.relative_to(REPO);target=Path.cwd()/'source-dependencies'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());dependency_hashes[str(relative)]=sha(target)
 save('source-snapshot.json',dict(original=job['source'],snapshot=str(snapshot),sha256=sha(snapshot),dependencies=dependency_hashes))
 kernel=getattr(module,job['symbol']);options={k:tuple(v) if isinstance(v,list) else v for k,v in job['options'].items()};options.pop('kernel',None);options.pop('requires_16_byte_alignment',None)
 m,n,k=job['shape'];family=job['family'];guard=65536
 def guarded(shape,fill):
  host=torch.full((math.prod(shape)+2*guard,),fill,dtype=torch.bfloat16);raw=host.cuda();return host,raw,raw[guard:-guard].view(shape)
 ai,ar,a=guarded((m,k),float('nan'));bi,br,b=guarded((k,n),float('nan'));ci,cr,c=guarded((m,n),123.);ci[guard:-guard]=float('nan')
 compiled=[]
 def listener(src,metadata,metadata_group,times,cache_hit):
  group={Path(k).suffix:v for k,v in metadata_group.items()}
  if '.hsaco' in group:
   name=metadata['name'];Path(name+'.amdgcn').write_bytes(Path(group['.amdgcn']).read_bytes());Path(name+'.hsaco').write_bytes(Path(group['.hsaco']).read_bytes());compiled.append(dict(name=name,shared=metadata['shared'],hsaco_sha256=sha(group['.hsaco'])))
 knobs.compilation.listener=listener
 if family=='persistent':
  opts=dict(options,M=m,N=n,K=k,B_K_CONTIGUOUS=False);grid=min(math.ceil(m/options['BLOCK_SIZE_M'])*math.ceil(n/options['BLOCK_SIZE_N']),options.get('NUM_SMS',10**9));launch=lambda:kernel[(grid,)](a,b.T,c,**opts)
 elif family=='stream_legacy':
  grid=96;launch=lambda:module._launch_gemm_a16w16_streamk(kernel,module._gemm_a16w16_bf16_tt_prepare_edges_kernel,a,b.T,c,cu_count=96,config=options.copy())
 elif family=='stream':
  # Original split-K launcher body, using this directory's two kernel definitions.
  options.pop('NUM_SMS',None)
  workspace=torch.empty((3,m,n),dtype=torch.float32,device=a.device)
  split_grid=(triton.cdiv(m,options['BLOCK_SIZE_M'])*triton.cdiv(n,options['BLOCK_SIZE_N']),3)
  handle=kernel.warmup(a,b.T,workspace,b.T,m,n,k,k,1,n,1,n,1,SPLIT_K=3,grid=split_grid,**options)
  _compact_prepare_handle(handle,job['_compact_record'],job['symbol'],Path(job['_compact_source_dir']),job.get('_compact_archived',False))
  reduce_handle=module._reduce_split3.warmup(workspace,c,ELEMENTS=m*n,BLOCK=4096,num_warps=8,grid=(triton.cdiv(m*n,4096),))
  _compact_prepare_handle(reduce_handle,job['_compact_record'],'_reduce_split3',Path(job['_compact_source_dir']),job.get('_compact_archived',False))
  def launch():
   kernel[split_grid](a,b.T,workspace,b.T,m,n,k,k,1,n,1,n,1,SPLIT_K=3,**options)
   module._reduce_split3[(triton.cdiv(m*n,4096),)](workspace,c,ELEMENTS=m*n,BLOCK=4096,num_warps=8)
  grid=96
 elif family=='wasp':
  grid=math.ceil(m/options['BLOCK_SIZE_M'])*math.ceil(n/options['BLOCK_SIZE_N']);launch=lambda:kernel[(grid,)](a,b,c,b,m,n,k,*a.stride(),*b.stride(),*c.stride(),**options)
 else:raise ValueError(family)
 import triton._C.libtriton as lib
 save('environment.json',dict(job=job,source_sha256=sha(snapshot),torch=torch.__version__,triton=triton.__version__,libtriton_sha256=sha(lib.__file__),clang_sha256=sha(os.environ['TRITON_HIP_CLANG_PATH']),environment={k:os.environ.get(k) for k in ['GPU_CHIP','GPU_ARGS','GPU_DFLAGS','AMDGCN_USE_BUFFER_OPS','TRITON_BUFFER_CACHE_SWIZZLE','TRITON_MLS_B_INTERLEAVE','PMD_PATH','PYTHONPATH','LD_LIBRARY_PATH','TRITON_HIP_CLANG_PATH','GPU_OUT','TRITON_CACHE_DIR','TRITON_HOME','RPY_HISTORY_LOG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OPTALL_PMD_LEASE','OPTALL_PREFLIGHT']}))
 seed=job.get('seed',21)
 def prepare(seed):
  gen=torch.Generator().manual_seed(seed)
  ah=torch.empty((m,k),dtype=torch.bfloat16);bh=torch.empty((k,n),dtype=torch.bfloat16)
  for tensor in [ah,bh]:
   for rows in tensor.split(256):rows.copy_((torch.randn(rows.shape,generator=gen)*.5).bfloat16())
  ai[guard:-guard].copy_(ah.flatten());bi[guard:-guard].copy_(bh.flatten());ar.copy_(ai);br.copy_(bi);return ah.float(),bh.float()
 ah,bh=prepare(seed);samples=[]
 def input_fingerprint():return dict(seed=seed,A_sha256=hashlib.sha256(ai.view(torch.uint8).numpy()).hexdigest(),B_sha256=hashlib.sha256(bi.view(torch.uint8).numpy()).hexdigest(),guard_elements=guard,layout='TT',A_stride=list(a.stride()),B_logical_stride=list(b.T.stride()))
 for idx in range(job.get('dispatches',1)):
  if idx==1 and 'timed_seed' in job:seed=job['timed_seed'];del ah,bh;ah,bh=prepare(seed)
  save('input-fingerprints-'+str(idx)+'.json',input_fingerprint())
  cr.copy_(ci);torch.cuda.synchronize();before=set(Path('g').glob('*/*/stats.txt'));start=time.monotonic();launch();torch.cuda.synchronize();elapsed=time.monotonic()-start
  files=sorted(set(Path('g').glob('*/*/stats.txt'))-before,key=lambda p:(int(p.parent.parent.name),int(p.parent.name)));assert files
  stages=[]
  for p in files:
   content=p.read_text();assert 'End Simulation Statistics' in content
   cycles=re.findall(r'^system.gcSimCycles\s+(\d+)',content,re.M);assert len(cycles)==1
   groups=re.findall(r'^(system\d*\.shaders\d*)\.CUs\d+\.completedWGs\s+(\d+)',content,re.M);assert len(groups)==96
   assert len({d for d,_ in groups})==2 and all(sum(d==key for key,_ in groups)==48 for d,_ in groups)
   freq=re.findall(r'^gcFrequency\s+([\d.]+)',content,re.M);assert freq==['2.197802']
   mmops=sum(map(int,re.findall(r'^system\d*\.shaders\d*\.CUs\d+\.numInstrTypeExecuted::MMOP\s+(\d+)',content,re.M)))
   stages.append(dict(stats=str(p),stats_sha256=sha(p),cycles=int(cycles[0]),us=int(cycles[0])*455/1e6,tgs=sum(int(v) for _,v in groups),active_cus=sum(int(v)>0 for _,v in groups),per_die={d:sum(int(v) for key,v in groups if key==d) for d,_ in groups},mmops=mmops))
  sample=dict(index=idx,phase='cold' if job['dispatches']==1 else 'warmup' if idx==0 else 'timed',seed=seed,us=sum(x['us'] for x in stages),stages=stages,simulation_wall_seconds=elapsed);save('timing-'+str(idx)+'.json',sample);print('TIMING',sample,flush=True)
  host=cr.cpu();actual=host[guard:-guard].view(m,n);canary=bool((host[:guard]==123).all() and (host[-guard:]==123).all());unchanged=torch.equal(ar.cpu().view(torch.uint8),ai.view(torch.uint8)) and torch.equal(br.cpu().view(torch.uint8),bi.view(torch.uint8));outside=nonfinite=0;max_abs=error=rounding=0.
  for row in range(0,m,64):
   fp32=ah[row:row+64]@bh;gold=fp32.bfloat16().float();obs=actual[row:row+64].float();outside+=int((~torch.isclose(obs,gold,atol=.001,rtol=.01)).sum());nonfinite+=int((~torch.isfinite(obs)).sum());max_abs=max(max_abs,float((obs-gold).abs().max()));error+=float((obs-fp32).abs().sum(dtype=torch.float64));rounding+=float((gold-fp32).abs().sum(dtype=torch.float64))
  torch.save(actual, 'output-'+str(idx)+'.pt')
  baseline_path=Path(job.get('baseline_dir',str(HERE/'baseline')))/('output-'+str(idx)+'.pt')
  baseline_comparison=None
  if job['name']!='baseline' and baseline_path.is_file():
   baseline=torch.load(baseline_path,weights_only=True).float();observed=actual.float();baseline_comparison=dict(elements=m*n,bitwise_differences=int((baseline!=observed).sum()),outside_tolerance=int((~torch.isclose(baseline,observed,atol=.001,rtol=.01)).sum()),max_abs=float((baseline-observed).abs().max()),baseline_sha256=sha(baseline_path),output_sha256=sha('output-'+str(idx)+'.pt'),atol=.001,rtol=.01)
   baseline_inputs=json.loads((baseline_path.parent/('input-fingerprints-'+str(idx)+'.json')).read_text());own_inputs=json.loads(Path('input-fingerprints-'+str(idx)+'.json').read_text());assert baseline_inputs==own_inputs,(baseline_inputs,own_inputs)
  sample['baseline_comparison']=baseline_comparison
  sample['validation']=dict(elements=m*n,outside_tolerance=outside,nonfinite=nonfinite,max_abs=max_abs if math.isfinite(max_abs) else None,canary=canary,inputs_byte_identical=unchanged,atol=.001,rtol=.01,mean_abs_vs_fp32=error/(m*n) if math.isfinite(error) else None,rounding_floor=rounding/(m*n));save('sample-'+str(idx)+'.json',sample);print('VALIDATION',sample['validation'],flush=True)
  assert canary and unchanged and outside==nonfinite==0 and error<=1.05*rounding+1e-6*m*n,sample
  if baseline_comparison is not None:assert baseline_comparison['outside_tolerance']==0,baseline_comparison
  samples.append(sample)
 timed=samples if job['dispatches']==1 else samples[1:];us=statistics.median(x['us'] for x in timed);save('result.json',dict(status='PASS',kernel_us=us,job=job,samples=samples,compiled=compiled,runner_sha256=RUNNER_SHA256))
def main():
 p=argparse.ArgumentParser();p.add_argument('job',type=Path,nargs='?',default=HERE/'record.json');p.add_argument('--dry-run',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--archived',action='store_true');p.add_argument('--out',type=Path);p.add_argument('--baseline-dir',type=Path);args=p.parse_args()
 if _COMPACT_PARSE_ONLY:return args
 record_or_job=json.loads(args.job.read_text())
 job=record_or_job.get('benchmark',{}).get('job',record_or_job)
 job['source']=str((args.job.resolve().parent/job['source']).resolve())
 if args.dry_run:
  print(json.dumps(job,indent=2));return
 if args.baseline_dir:job['baseline_dir']=str(args.baseline_dir.resolve())
 if args.worker:
  try:return worker(job)
  except Exception as e:save('failure.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc()));raise
 out=(args.out or args.job.parent/job['name']).resolve();out.mkdir(parents=True,exist_ok=False);(out/'g').mkdir();save(out/'job.json',job);env=dict(os.environ,GPU_CHIP='sbx2',GPU_DFLAGS="['StatLog']",GPU_ARGS="['--abs-max-tick=10000000000']",AMDGCN_USE_BUFFER_OPS='1',TRITON_BUFFER_CACHE_SWIZZLE='1',TRITON_MLS_B_INTERLEAVE='2',TRITON_CACHE_DIR=str(out/'cache'),TRITON_HOME=str(out/'triton-home'),RPY_HISTORY_LOG=str(out/'history.log'),RPY_HISTORY_SIZE='1024',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env.pop('HSA_TOOLS_LIB',None);env.pop('BOLTOPS_GEMM_CONFIG_DIR',None)
 for key,value in job.get('compiler_environment',{}).items():
  if value is None:env.pop(key,None)
  else:env[key]=value
 with tempfile.TemporaryDirectory(prefix='optall-compute-') as d:
  link=Path(d)/'g';link.symlink_to(out/'g',target_is_directory=True);env['GPU_OUT']=str(link)
  with (out/'worker.log').open('w') as log:
   command=[sys.executable,'-u',str(Path(__file__).resolve()),str(out/'job.json'),'--worker']
   process=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   save(out/'launch-process.json',dict(pid=process.pid,ppid=os.getpid(),command=command,cwd=str(out),lease=env.get('OPTALL_PMD_LEASE'),preflight=env.get('OPTALL_PREFLIGHT')))
   (out/'worker.pid').write_text(str(process.pid)+'\n')
   code=process.wait()
 if code:raise SystemExit(code)


_measurement_main = main

"""Appended to each measured runner: isolate work and publish its exact manifest."""
import copy as _compact_copy
import gzip as _compact_gzip
import shutil as _compact_shutil
import struct as _compact_struct
import subprocess as _compact_subprocess
import tempfile as _compact_tempfile
import uuid as _compact_uuid
import zlib as _compact_zlib


def _compact_sha(data):
    return hashlib.sha256(data).hexdigest()


def _compact_sections(data):
    assert data[:5] == b'\x7fELF\x02'
    offset, = _compact_struct.unpack_from('<Q', data, 40)
    size, count, names = _compact_struct.unpack_from('<HHH', data, 58)
    entries = [_compact_struct.unpack_from('<IIQQQQIIQQ', data, offset+i*size) for i in range(count)]
    strings = data[entries[names][4]:entries[names][4]+entries[names][5]]
    result = {}
    for entry in entries:
        name = strings[entry[0]:].split(b'\0', 1)[0].decode()
        if name in ('.text', '.rodata'):
            raw = data[entry[4]:entry[4]+entry[5]]
            result[name] = dict(size=len(raw), sha256=_compact_sha(raw))
    assert '.text' in result
    return result


def _compact_manifest(record):
    names = record['artifacts']
    assert isinstance(names, list) and len(names) == len(set(names)), names
    assert all(isinstance(name, str) and name not in ('.', '..') and Path(name).name == name for name in names), names
    assert {'benchmark.py', 'kernel.py', 'kernel.amdgcn', 'record.json'} <= set(names)
    return names


def _compact_read_record(path):
    record = json.loads(path.read_text())
    _compact_manifest(record)
    return record


def _compact_write(output, record, source, replacements=None):
    """Stage all bytes before touching output; never publish runtime intermediates."""
    regenerate = replacements is None
    replacements = replacements or {}
    names = _compact_manifest(record)
    assert not output.exists(), 'output directory must be new: ' + str(output)
    if regenerate:
        expected_hashes = record['artifact_sha256']
        assert set(expected_hashes) == set(names)-{'record.json'}, 'incomplete artifact checksum manifest'
        for name, expected in expected_hashes.items():
            assert _compact_sha((source/name).read_bytes()) == expected, 'input artifact checksum mismatch: ' + name
        _compact_assert_bundle(record, (source/'kernel.amdgcn').read_bytes())
    payload = {}
    for name in names:
        if name == 'record.json':
            continue
        path = Path(__file__) if name == 'benchmark.py' else source/name
        payload[name] = replacements[name] if name in replacements else path.read_bytes()
    record = _compact_copy.deepcopy(record)
    hashes = {name: _compact_sha(data) for name, data in payload.items()}
    record['artifact_sha256'] = hashes
    record['source']['sha256'] = hashes['kernel.py']
    record['assembly']['sha256'] = hashes['kernel.amdgcn']
    record['benchmark']['sha256'] = hashes['benchmark.py']
    for item in record.get('evidence', {}).values():
        if isinstance(item, dict) and item.get('path') in hashes:
            item['sha256'] = hashes[item['path']]
    def refresh_trace_hashes(value):
        if isinstance(value, dict):
            path = value.get('path')
            if path in hashes and (path.endswith('.perf') or path == 'perf_stats.txt.gz'):
                value['sha256'] = hashes[path]
            for child in value.values():
                refresh_trace_hashes(child)
        elif isinstance(value, list):
            for child in value:
                refresh_trace_hashes(child)
    # Embedded historical records keep their original same-named asset hashes.
    refresh_trace_hashes(record.get('single_tg_perf'))
    refresh_trace_hashes(record.get('auxiliary_single_tg_perf'))
    _compact_assert_bundle(record, payload['kernel.amdgcn'])
    payload['record.json'] = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with _compact_tempfile.TemporaryDirectory(prefix='compact-publish-', dir=output.parent) as temporary:
        stage = Path(temporary)/'artifacts'
        stage.mkdir()
        for name, data in payload.items():
            (stage/name).write_bytes(data)
        assert set(p.name for p in stage.iterdir()) == set(names)
        stage.rename(output)
    print(json.dumps(dict(status='PASS', output=str(output), artifacts=names, count=len(names)), ensure_ascii=False))


def _compact_units(record):
    units = record['assembly'].get('compilation_units')
    if units:
        return units
    return [dict(stage='gemm', symbol=record['source']['symbol'],
                 executable_sections=record['assembly'].get('executable_sections'))]


def _compact_assert_bundle(record, bundle):
    for unit in record['assembly'].get('compilation_units', []):
        offset, size = unit['offset_bytes'], unit['size_bytes']
        assert offset >= 0 and size > 0 and offset+size <= len(bundle), unit
        assert _compact_sha(bundle[offset:offset+size]) == unit['sha256'], unit['stage']


def _compact_perf_binaries(data):
    assert data[:8] == b'\xe4HYGPERF'
    assert _compact_struct.unpack_from('<2Q', data, 8) == (0, 24)
    lengths = {6: 48, 7: 72, 8: 56, 9: 160, 10: 64, 11: 80, 13: 56, 15: 72}
    compressed = {6: (24, 32, 40), 8: (32, 40, 48), 11: (48, 56, 64)}
    position, objects = 24, []
    while position < len(data):
        kind, offset, length = _compact_struct.unpack_from('<3Q', data, position)
        assert offset == position-24 and length == lengths[kind]
        end = position+length
        if kind in compressed:
            pointer_at, raw_at, size_at = compressed[kind]
            assert _compact_struct.unpack_from('<Q', data, position+pointer_at)[0] == offset+length
            raw_size = _compact_struct.unpack_from('<Q', data, position+raw_at)[0]
            packed_size = _compact_struct.unpack_from('<Q', data, position+size_at)[0]
            if kind == 8:
                binary = _compact_zlib.decompress(data[end:end+packed_size])
                assert len(binary) == raw_size
                objects.append(binary)
            end += packed_size
        assert position < end <= len(data)
        position = end
    assert position == len(data) and objects, 'PERF contains no embedded code object'
    return objects


def _compact_prepare_handle(handle, record, symbol, source, archived=False):
    """Validate before dispatch; archived replay uses the measured PERF's code object."""
    unit = next(unit for unit in _compact_units(record) if unit['symbol'] == symbol)
    expected = {key: value['sha256'] if isinstance(value, dict) else value
                for key, value in unit['executable_sections'].items()}
    if archived:
        found = None
        for name in _compact_manifest(record):
            if not name.endswith('.perf'):
                continue
            trace = (source/name).read_bytes()
            assert _compact_sha(trace) == record['artifact_sha256'][name], name
            matches = {}
            for binary in _compact_perf_binaries(trace):
                sections = {key: value['sha256'] for key, value in _compact_sections(binary).items()}
                if sections == expected:
                    matches[_compact_sha(binary)] = binary
            if unit.get('hsaco_sha256') in matches:
                found = matches[unit['hsaco_sha256']]
                break
            if matches:
                assert len(matches) == 1, 'PERF has ambiguous code objects for '+symbol
                found = next(iter(matches.values()))
                break
        assert found is not None, 'no archived PERF matches compilation unit: ' + symbol
        assert handle.module is None, 'archived binding must precede code-object loading'
        bundle = (source/'kernel.amdgcn').read_bytes()
        _compact_assert_bundle(record, bundle)
        assembly = bundle[unit['offset_bytes']:unit['offset_bytes']+unit['size_bytes']]
        handle.kernel = found
        handle.asm['hsaco'] = found
        handle.asm['amdgcn'] = assembly.decode()
    actual = {key: value['sha256'] for key, value in _compact_sections(handle.asm['hsaco']).items()}
    assert actual == expected, (
        'fresh compilation differs from measured machine code; use --archived to replay the PERF-embedded code object', symbol)
    handle._init_handles()
    resources = record.get('resources', {})
    if unit['stage'] in ('reduce', 'reduction'):
        resources = resources.get('reduction', resources)
    elif unit['stage'] in ('pack', 'packing'):
        resources = resources.get('packing', resources)
    actual_resources = dict(VGPR=handle.n_regs, spills=handle.n_spills, LDS_bytes=handle.metadata.shared)
    for key, expected_value in resources.items():
        if key in actual_resources:
            assert actual_resources[key] == expected_value, (symbol, key, actual_resources, resources)
    Path(symbol+'.amdgcn').write_text(handle.asm['amdgcn'])
    Path(symbol+'.hsaco').write_bytes(handle.asm['hsaco'])
    return handle


def _compact_compile_outputs(work, record):
    """Bind every compiled dispatch to the machine code represented by its PERF."""
    compiled = []
    for unit in _compact_units(record):
        symbol = unit['symbol']
        candidates = []
        for path in work.rglob('*.amdgcn'):
            binary = path.with_suffix('.hsaco')
            if binary.is_file() and re.search(r'(?m)^\s*\.globl\s+'+re.escape(symbol)+r'\b', path.read_text()):
                candidates.append((path, binary))
        assert candidates, 'compiled dispatch is missing: ' + symbol
        path, binary = sorted(candidates, key=lambda pair: (len(pair[0].parts), str(pair[0])))[0]
        actual = _compact_sections(binary.read_bytes())
        expected = unit.get('executable_sections')
        assert expected, 'record lacks executable-section evidence for ' + symbol
        expected_hashes = {key: value['sha256'] if isinstance(value, dict) else value for key, value in expected.items()}
        assert {key: value['sha256'] for key, value in actual.items()} == expected_hashes, (
            'compiled dispatch differs from archived PERF; recapture PERF before publication', symbol, actual, expected)
        compiled.append((unit, path.read_bytes(), binary.read_bytes(), actual))
    return compiled


def _compact_assembly(record, compiled):
    bundle = b''
    units = []
    for old, assembly, binary, sections in compiled:
        if len(compiled) > 1:
            bundle += ('// Compilation unit: '+old['stage']+' / '+old['symbol']+'\n').encode()
        unit = _compact_copy.deepcopy(old)
        unit.update(offset_bytes=len(bundle), size_bytes=len(assembly), sha256=_compact_sha(assembly),
                    hsaco_sha256=_compact_sha(binary), executable_sections=sections)
        bundle += assembly
        if not bundle.endswith(b'\n'):
            bundle += b'\n'
        units.append(unit)
    record['assembly']['compilation_units'] = units
    record['assembly']['sha256'] = _compact_sha(bundle)
    def refresh_unit_references(value):
        if isinstance(value, dict):
            if value.get('path') == 'kernel.amdgcn' and value is not record['assembly']:
                section = value.get('section', value.get('stage', value.get('symbol')))
                assert section is not None, 'assembly reference needs a compilation-unit section'
                unit = units[section] if isinstance(section, int) else next(
                    (u for u in units if section in (u['stage'], u['symbol'])), None)
                assert unit is not None, section
                value.update(sha256=unit['sha256'], offset_bytes=unit['offset_bytes'],
                             size_bytes=unit['size_bytes'], hsaco_sha256=unit['hsaco_sha256'],
                             executable_sections=unit['executable_sections'])
            for child in value.values():
                refresh_unit_references(child)
        elif isinstance(value, list):
            for child in value:
                refresh_unit_references(child)
    refresh_unit_references(record['assembly'])
    for key in ('packing_assembly', 'auxiliary_assembly', 'auxiliary'):
        refresh_unit_references(record.get(key))
    return bundle


def _compact_raw_stats(work, sample, stage, index):
    name = stage.get('stats', stage.get('stats_path'))
    path = work/name if name else work/f'stats-{index}-{stage["_stage"]}.txt.gz'
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = _compact_gzip.decompress(raw)
    assert b'End Simulation Statistics' in raw
    expected = stage.get('stats_sha256', stage.get('sha256'))
    assert expected is None or _compact_sha(raw) == expected
    return raw


def _compact_samples(work, samples, filename):
    blocks = []
    converted = []
    offset = 0
    for index, original in enumerate(samples):
        sample = _compact_copy.deepcopy(original)
        sample.pop('perf', None)
        sample.pop('perf_path', None)
        stage_value = sample.get('stages', sample.get('dispatches'))
        assert stage_value, sample
        items = list(stage_value.items()) if isinstance(stage_value, dict) else [(None, item) for item in stage_value]
        total = 0
        for label, stage in items:
            if label is not None:
                stage['_stage'] = label
            raw = _compact_raw_stats(work, sample, stage, index)
            stage.pop('_stage', None)
            cycles = re.findall(rb'^system.gcSimCycles\s+(\d+)', raw, re.M)
            assert len(cycles) == 1
            total += int(cycles[0])
            stage.update(stats_path=filename, stats_section=len(blocks), stats_sha256=_compact_sha(raw),
                         stats_offset_bytes=offset, stats_size_bytes=len(raw))
            if 'stats' in stage:
                stage['stats'] = filename
            stage.pop('raw_stats_path', None)
            blocks.append(raw)
            offset += len(raw)
        sample.update(index=index, status='PASS', kernel_cycles=total, kernel_us=total*455/1e6)
        sample.setdefault('phase', 'cold' if len(samples) == 1 else 'warmup' if index == 0 else 'timed')
        if len(items) == 1:
            stage = items[0][1]
            sample.update(stats_path=filename, stats_section=stage['stats_section'], stats_sha256=stage['stats_sha256'],
                          stats_offset_bytes=stage['stats_offset_bytes'], stats_size_bytes=stage['stats_size_bytes'])
        converted.append(sample)
    return converted, _compact_gzip.compress(b''.join(blocks), mtime=0)


def _compact_refresh_measurement(work, record, args):
    path = work/'result.json'
    assert path.is_file(), 'measurement did not produce a validated result.json'
    measured = json.loads(path.read_text())
    assert measured['status'] == 'PASS', measured.get('status')
    compiled = _compact_compile_outputs(work, record)
    replacements = {'kernel.amdgcn': _compact_assembly(record, compiled)}
    is_perf = bool(getattr(args, 'perf', False))
    samples, packed = _compact_samples(work, measured['samples'], 'perf_stats.txt.gz' if is_perf else 'stats.txt.gz')
    if is_perf:
        assert len(samples) >= 2
        selected = min(samples[1:], key=lambda item: item['kernel_cycles'])
        perf_name = measured['samples'][selected['index']]['perf']
        trace = (work/perf_name).read_bytes()
        assert trace[1:8] == b'HYGPERF'
        replacements.update({'single_tg.perf': trace, 'perf_stats.txt.gz': packed})
        capture = record.setdefault('single_tg_perf', {})
        capture.update(path='single_tg.perf', sha256=_compact_sha(trace), grid=[1], samples=samples,
                       samples_validated=len(samples), repeat=selected['index'],
                       kernel_cycles=selected['kernel_cycles'], kernel_us=selected['kernel_us'],
                       selection='minimum cycles of repeats 1 and 2; repeat 0 excluded',
                       full_and_single_text_rodata_equal=True,
                       validation=dict(elements=selected.get('validated_elements'),
                           outside_tolerance=selected.get('mismatches'), max_abs=selected.get('max_abs'),
                           canary_ok=True, inputs_byte_identical=True, unselected_output_nan=True),
                       hsaco_sha256=_compact_sha(compiled[0][2]),
                       stats=dict(path='perf_stats.txt.gz', sha256=_compact_sha(packed), dispatches=len(samples)))
        selected.update(perf_path='single_tg.perf', perf_sha256=_compact_sha(trace))
    else:
        record['samples'] = samples
        replacements['stats.txt.gz'] = packed
        selected = samples if len(samples) == 1 else samples[1:]
        us = statistics.median(item['kernel_us'] for item in selected)
        cycles = statistics.median(item['kernel_cycles'] for item in selected)
        performance = record['performance']
        performance.update(kernel_us=us, kernel_cycles=cycles, samples=len(selected),
                           kernel_tops=2*__import__('math').prod(record['shape'])/(us*1e6))
        performance['aggregation'] = 'one cold full-grid sample' if len(samples) == 1 else 'median of timed repeats after repeat 0 warmup'
        record['evidence']['stats']['dispatches'] = sum(len(s.get('stages', s.get('dispatches'))) for s in samples)
        if 'mmac_util_percent' in performance:
            performance['mmac_util_percent'] = performance['kernel_tops']/(96*4096*(1000/455)/1000)*100
        baseline = record['comparison'].get('baseline_kernel_us', record.get('baseline_record', {}).get('kernel_us'))
        if baseline:
            record['comparison'].update(speedup=baseline/us, latency_change_percent=(us/baseline-1)*100)
        for candidate in (work/'model/config.json', work/'g/config.json'):
            if candidate.is_file():
                config = candidate.read_bytes()
                replacements['pmd-config.json.gz'] = _compact_gzip.compress(config, mtime=0)
                model_config = record.get('evidence', {}).get('model_config', {})
                if 'raw_sha256' in model_config:
                    model_config['raw_sha256'] = _compact_sha(config)
                break
        else:
            raise AssertionError('measurement model config is missing')
        # All guards/oracles run in the original runner before status PASS.
        record['validation'] = [dict(index=s['index'], status='PASS',
                                     accuracy=s.get('validation', s.get('accuracy', dict(
                                         elements=s.get('validated_elements'), mismatches=s.get('mismatches'),
                                         max_abs=s.get('max_abs'), inputs_byte_identical=True, canary_ok=True))),
                                     baseline_comparison=s.get('baseline_comparison')) for s in samples]
    record['benchmark']['last_run'] = dict(status='PASS', source='current local kernel.py',
        timing='single-TG capture' if is_perf else 'full-grid measured with original runner',
        compiled_dispatches_verified=True, archived_code_objects=getattr(args, 'archived', False))
    return replacements


def _compact_main():
    global _COMPACT_PARSE_ONLY
    argv = sys.argv[1:]
    if '--regenerate' in argv:
        p = argparse.ArgumentParser(description='Republish the exact artifact manifest without executing PMD.')
        p.add_argument('--regenerate', type=Path, required=True)
        p.add_argument('--record', type=Path, default=Path(__file__).with_name('record.json'))
        args = p.parse_args()
        record_path = args.record.resolve()
        _compact_write(args.regenerate.resolve(), _compact_read_record(record_path), record_path.parent)
        return
    _COMPACT_PARSE_ONLY = True
    args = _measurement_main()
    _COMPACT_PARSE_ONLY = False
    if _COMPACT_FAMILY == 'small':
        record_path = args.input.resolve()
        output = args.output.resolve()
    elif _COMPACT_FAMILY == 'compute':
        record_path = args.job.resolve()
        output = (args.out or Path(_compact_tempfile.gettempdir()) /
                  ('boltops-gemm-replay-'+_compact_uuid.uuid4().hex)).resolve()
    else:
        record_path = args.record.resolve()
        output = args.out.resolve()
    record = _compact_read_record(record_path)
    if getattr(args, 'dry_run', False):
        print(json.dumps(dict(requirement=record['requirement'], shape=record['shape'],
            launch=record['launch'], artifacts=_compact_manifest(record), output=str(output)), indent=2))
        return
    for flag in ('auto', 'probe_tail', 'pack_only', 'stream', 'compile_only'):
        assert not getattr(args, flag, False), flag+' is diagnostic and cannot publish full measured artifacts'
    if _COMPACT_FAMILY != 'small':
        assert REPO is not None, 'run from the BoltOPs repository or include its path in PYTHONPATH'
    assert not output.exists(), 'output directory must be new'
    source = Path(__file__).resolve().parent
    if getattr(args, 'archived', False):
        assert _compact_sha((source/'kernel.py').read_bytes()) == record['source']['sha256'], (
            'archived replay requires the recorded local kernel source')
    if _COMPACT_FAMILY == 'axes':
        assert args.source.read_bytes() == (source/'kernel.py').read_bytes(), (
            'compact publication requires the local kernel source; capture a new record for other sources')
    with _compact_tempfile.TemporaryDirectory(prefix='boltops-measure-') as temporary:
        temporary = Path(temporary)
        raw = temporary/'measurement'
        runtime_record = temporary/'record.json'
        current = _compact_copy.deepcopy(record)
        if _COMPACT_FAMILY == 'small':
            for variant in current['variants'].values():
                if variant.get('module'):
                    variant['module'] = 'kernel'
                variant['_compact_archived'] = args.archived
            if args.seed is not None:
                current['seed'] = args.seed
            runtime_record.write_text(json.dumps(current))
            repeats = args.repeats
            if args.perf:
                repeats = 3 if '--repeats' not in argv else repeats
                assert repeats == 3, 'single-TG PERF requires exactly three launches'
            command = ['--input', str(runtime_record), '--output', str(raw), '--variant', args.variant,
                       '--repeats', str(repeats), '--repo', str(args.repo.resolve())]
            if args.seed is not None:
                command += ['--seed', str(args.seed)]
            if args.perf:
                command += ['--perf']
            if args.archived:
                command += ['--archived']
        elif _COMPACT_FAMILY == 'compute':
            job = _compact_copy.deepcopy(current['benchmark']['job'])
            job['source'] = str(source/'kernel.py')
            job['_compact_record'] = current
            job['_compact_source_dir'] = str(source)
            job['_compact_archived'] = args.archived
            runtime_record.write_text(json.dumps(job))
            command = [str(runtime_record), '--out', str(raw)]
            if args.baseline_dir:
                command += ['--baseline-dir', str(args.baseline_dir.resolve())]
        else:
            runtime_record.write_text(json.dumps(current))
            command = [str(raw), '--record', str(runtime_record), '--source', str(args.source),
                       '--layout', args.layout, '--dispatches', str(args.dispatches),
                       '--pack-k', str(args.pack_k), '--pack-n', str(args.pack_n), '--options', args.options]
            if args.baseline_out:
                command += ['--baseline-out', str(args.baseline_out)]
            if args.archived:
                command += ['--archived']
        env = dict(os.environ, BOLTOPS_COMPACT_RAW='1', PYTHONDONTWRITEBYTECODE='1')
        process = _compact_subprocess.run([sys.executable, str(Path(__file__).resolve()), *command], env=env)
        if process.returncode:
            log = raw/'worker.log'
            if log.is_file():
                print(log.read_text()[-20000:], file=sys.stderr)
            raise SystemExit(process.returncode)
        replacements = _compact_refresh_measurement(raw, current, args)
        _compact_write(output, current, source, replacements)


if __name__ == '__main__':
    if os.environ.get('BOLTOPS_COMPACT_RAW') == '1':
        _measurement_main()
    else:
        _compact_main()
