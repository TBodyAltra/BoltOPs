"""Bind actual shader novelty, exact defaults and address/bounds proof before PMD."""
import pathlib,json,hashlib,re
R=pathlib.Path(__file__).resolve().parent;B=R.parent;P=B/'single-tg-pipeline-20261027-round175-group1-index-cpu';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def code(p):
 lines=[]
 for line in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',line):break
  line=line.split(';',1)[0].strip()
  if line and not line.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(line)
 return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
assert not(R/'rebase-review.json').exists();e=load(R/'job-exit.json');assert e['returncode']==0 and not pathlib.Path('/proc',str(e['pid'])).exists()
q=load(R/'cpu-screen.json');assert q['passed'] and q['default_controls_exact'] and len(q['results'])==12 and all(sha(pathlib.Path(p))==h for p,h in q['source_hashes'].items());assert load(R/'address-bounds-proof.json')['passed']
known={};old=load(P/'cpu-screen.json');assert old['passed'] and all(sha(pathlib.Path(p))==h for p,h in old['source_hashes'].items())
for x in old['results']+load(P/'control-qualification.json')['controls']:
 if x['status']=='COMPILED':known.setdefault(code(pathlib.Path(x['directory'])/'kernel.amdgcn'),[]).append(x['directory'])
seen={};selected=[];duplicates=[];excluded=[]
for x in q['results']:
 if x['status']!='COMPILED':excluded.append(dict(variant=x['variant'],reason=x['status'],exception=x.get('exception')));continue
 c=x['config'];assert c['REBASE_MLS_TILE'] in [1,2,3] and not c['WARP_SPECIALIZE'] and not c['wasp_wdra'] and c['MMA_K_SPLIT']==4 and c['LOAD_B_FIRST'] and c['num_stages']==1
 p=pathlib.Path(x['directory']);isa=(p/'kernel.amdgcn').read_text();assert len(re.findall(r'^\.Lfunc_end\d+:',isa,re.M))==1
 h=code(p/'kernel.amdgcn');assert h==x['normalized_ISA_sha256']
 if x['fields']['private_segment_fixed_size'] or x['fields']['vgpr_spill_count'] or x['metadata']['shared']>65536:
  excluded.append(dict(variant=x['variant'],reason='scratch/VGPR spill or LDS budget',fields=x['fields'],shared=x['metadata']['shared']));continue
 if h in known or h in seen:
  duplicates.append(dict(variant=x['variant'],normalized_ISA=h,equal_to=known.get(h,[seen.get(h)]),fields=x['fields']));continue
 selected.append(x);seen[h]=x['directory']
out=dict(passed=True,CPU_only=True,CPU_screen_sha256=sha(R/'cpu-screen.json'),source_hashes_unchanged=True,parent_source_hashes_unchanged=True,default_controls_exact=2,features=12,selected=selected,duplicates=duplicates,excluded=excluded,strict_physical_zero=sum(x['all_physical_spills_zero'] for x in selected),native_sha256=sha(R/'native/libtriton.so'),address_proof_sha256=sha(R/'address-bounds-proof.json'),native_admission='Needs single-TG full-gold pilot, off-origin edge correctness before any long calibration; unchanged K contract; no automatic performance promotion',PMD_models_launched=0)
(R/'rebase-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,selected=len(selected),duplicates=len(duplicates),excluded=len(excluded),strictzero=out['strict_physical_zero'],variants=[dict(variant=x['variant'],fields=x['fields']) for x in selected])))
