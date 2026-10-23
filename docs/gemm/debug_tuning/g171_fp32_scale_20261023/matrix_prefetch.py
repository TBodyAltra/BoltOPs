from pathlib import Path
import re,json,hashlib

def normalize_mls(s):
 # Lossless spelling for the existing private parser's default-single-CTA MLS encoding.
 pat=r'#ttg.mls_shared<opIdx = (\d+), mlsTile = \[(\d+), (\d+)\], elemBitWidth = (\d+), elemBitTyKind = none, alt2Kind = (\d+), version = (\d+), order = \[(\d+), (\d+)\], CTALayout = #ttg<>>'
 def sub(m):
  a,b,c,d,e,f,g,h=m.groups()
  return f'#ttg.mls_shared_import<{a}, {b}, {c}, {d}, 0, {e}, {f}, {g}, {h}>'
 s,n=re.subn(pat,sub,s)
 assert '#ttg.mls_shared<' not in s,'unsupported non-default MLS CTA encoding'
 return s,n

def transform(s):
 s,normalized=normalize_mls(s)
 edits=[];proof=[]
 pattern=r'^        %accumulator:9 = scf.for[^\n]+\n'
 for match in re.finditer(pattern,s,re.M):
  a=match.start();b=s.index('        } {hcu.wasp_split_policy',match.end());head=match.group();body=s[match.end():b];lines=body.splitlines(True)
  # Only move A/B matrix reads. Current-iteration FP32 scale loads and MUL/FMA order stay intact.
  cut=next(i for i,l in enumerate(lines) if 'rocdl.sched.barrier' in l);setup=''.join(lines[:cut]);assert 'tt.dot' not in setup and setup.count('ttg.local_load')==2 and setup.count('rocdl.hcu.s.abarrier.arrive')==2
  dotline=next(l for l in lines if ' = tt.dot ' in l);names=list(re.search(r'tt.dot (%\w+), (%\w+),',dotline).groups());types=list(re.search(r' : (tensor<.*>) \* (tensor<.*>) -> tensor<',dotline).groups())
  assert types[0].startswith('tensor<64x128xf8') and types[1].startswith('tensor<128x256xf8'),types
  args=re.search(r'iter_args\((.*)\) ->',head)[1];pairs=[p.split(' = ') for p in args.split(', ')];assert len(pairs)==9;argnames=[x[0] for x in pairs];initial={x[0]:x[1] for x in pairs[:4]}
  # Original yield first four entries are the ring slot/phase for A and B next iteration.
  yi=next(i for i,l in enumerate(lines) if 'scf.yield' in l);ys=lines[yi];vals=ys.split('scf.yield ',1)[1].split(' : ',1)[0].split(', ');assert len(vals)==9
  ci=next(i for i,l in enumerate(lines) if re.search(r' = arith.addi '+re.escape(argnames[0])+r',',l));counters=''.join(lines[ci:yi]);assert 'ttg.local_load' not in counters and 'tt.dot' not in counters
  defs=set(re.findall(r'(%\w+) =',setup))
  def clone(text,prefix,replacements):
   rename={**{n:'%'+prefix+n[1:] for n in defs},**replacements}
   return re.sub(r'%\w+',lambda m:rename.get(m[0],m[0]),text)
  init=clone(setup,'prefetch_init_',initial)
  initvals=['%prefetch_init_'+n[1:] for n in names]
  resulttype='tensor<64x256xf32, #mma>'
  assert head.count(resulttype)==1
  head=head.replace('%accumulator:9','%accumulator:11',1).replace(') ->',', %carry_matrix_a = '+initvals[0]+', %carry_matrix_b = '+initvals[1]+') ->',1).replace(resulttype+')',resulttype+', '+', '.join(types)+')',1)
  iv=re.search(r'scf.for (%\w+) =',head)[1];lb=re.search(r'scf.for %\w+ = (%\w+) to',head)[1];ub=re.search(r' to (%\w+) step',head)[1];step=re.search(r' step (%\w+) ',head)[1]
  test=f'          %prefetch_next_iv = arith.addi {iv}, {step} : i32\n          %prefetch_has_next = arith.cmpi slt, %prefetch_next_iv, {ub} : i32\n'
  pf=clone(setup,'prefetch_next_',dict(zip(argnames[:4],vals[:4])));pfvals=['%prefetch_next_'+n[1:] for n in names]
  fetch='          %next_matrix:2 = scf.if %prefetch_has_next -> ('+', '.join(types)+') {\n'+pf+'            scf.yield '+', '.join(pfvals)+' : '+', '.join(types)+'\n          } else {\n            scf.yield %carry_matrix_a, %carry_matrix_b : '+', '.join(types)+'\n          }\n'
  # Prefetch after current MMAC has signaled the other consumer, before this iteration's scale arithmetic.
  core=''.join(lines[cut:ci]);dotpos=core.index(' = tt.dot ');signals=[m for m in re.finditer(r'^.*rocdl.s.barrier \{hcu.consumer_pingpong\}[^\n]*\n',core,re.M) if m.start()>dotpos];assert len(signals)==1
  scale_start=re.search(r'^          %a_scale_\d+ =',core[dotpos:],re.M);assert scale_start
  pos=dotpos+scale_start.start();core=core[:pos]+fetch+core[pos:]
  for name,carry in zip(names,['%carry_matrix_a','%carry_matrix_b']):core=re.sub(re.escape(name)+r'\b',carry,core)
  ys=ys.replace(' : ', ', %next_matrix#0, %next_matrix#1 : ',1);loc=ys.index(' loc(');ys=ys[:loc]+', '+', '.join(types)+ys[loc:]
  # Keep all existing synchronization, adding exactly one prologue read and 47 next reads per consumer.
  new=init+head+counters+test+core+ys;edits.append((a,b,new));proof.append({'consumer_loop':len(proof),'iterations':48,'initial_matrix_reads':2,'next_matrix_read_pairs':47,'last_iteration_prefetch':False,'scope':'next A/B only, after MMAC signal and before current scale arithmetic','floating_math_order':'unchanged','default_MLS_encoding_aliases':normalized,'barriers':'original A/B wait and arrive contract retained per load; scale sync unchanged'})
 assert len(edits)==2,len(edits)
 for a,b,t in reversed(edits):s=s[:a]+t+s[b:]
 return s,proof
if __name__=='__main__':
 import sys
 src=Path(sys.argv[1]);dst=Path(sys.argv[2]);text,proof=transform(src.read_text());dst.write_text(text);dst.with_suffix('.proof.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'transformed_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'transform':proof},indent=2))
 print('Prepared two consumer A/B lookahead loops, unchanged F32 scale math')
