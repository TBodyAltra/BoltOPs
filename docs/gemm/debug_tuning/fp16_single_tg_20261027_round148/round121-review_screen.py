import pathlib,json,re,hashlib,collections
R=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def code(p):
 lines=[]
 for l in p.read_text().splitlines():
  if re.match(r'\s*\.Lfunc_end\d+:',l):break
  l=l.split(';',1)[0].strip()
  if l and not l.startswith(('.file','.loc','.Ltmp','.cfi')):lines.append(l)
 return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
assert not(R/'screen-review.json').exists()
assert json.loads((R/'job-exit.json').read_text())['returncode']==0
c=json.loads((R/'cpu-complete.json').read_text());assert c['passed'] and sha(R/'cpu-screen.json')==c['screen_sha256']
s=json.loads((R/'cpu-screen.json').read_text());assert all(sha(pathlib.Path(p))==h for p,h in s['source_hashes'].items())
q=json.loads((R/'reference-map.json').read_text());prior=collections.defaultdict(list);missing=[]
for x in q['candidates']:
 if x['tile'] not in [[256,128,128],[256,256,128]] or x['layout']!='TT' or x.get('mode')!='nowasp':continue
 root=pathlib.Path(x['source_root_container']);ev=pathlib.Path(x.get('evidence_root_absolute',str(root/'runs')));p=ev/x['source_ids'][0]/'kernel.amdgcn'
 if not p.is_file():missing.append(str(p));continue
 prior[(tuple(x['tile']),code(p))].append(dict(source_root=str(root),source_ids=x['source_ids'],config=x['config'],fraction=x['mmac_theoretical_fraction'],valid_metric=x.get('valid_metric'),asm_sha256=sha(p)))
groups=collections.defaultdict(list)
for x in s['eligible']:
 p=pathlib.Path(x['directory']);h=code(p/'kernel.amdgcn');x=dict(x,normalized_ISA_sha256=h)
 groups[(tuple(x['tile']),h)].append(x)
novel=[];equivalent=[]
for (tile,h),xs in groups.items():
 row=dict(tile=list(tile),normalized_ISA_sha256=h,representative=xs[0],equivalent_CPU_candidates=xs[1:],previous_measurement_matches=prior[(tile,h)])
 (equivalent if row['previous_measurement_matches'] else novel).append(row)
out=dict(passed=True,scope='CPU_RESOURCE_AND_EMITTED_ISA_SCREEN_ONLY_NO_PERFORMANCE_CLAIM',screen_sha256=sha(R/'cpu-screen.json'),default_controls_exact=True,features=len(s['results']),all_zero_physical_spill_candidates=len(s['eligible']),unique_eligible_ISA_groups=len(groups),novel_eligible_groups=novel,already_measured_ISA_groups=equivalent,previous_assembly_files_missing=missing,source_hashes_unchanged=True,PMD_performance_pending=True)
(R/'screen-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(features=out['features'],eligible=out['all_zero_physical_spill_candidates'],unique_groups=len(groups),novel_groups=len(novel),previously_measured_groups=len(equivalent),missing_previous_ASM=len(missing),eligible_by_tile=dict(collections.Counter('x'.join(map(str,x['tile'])) for x in s['eligible'])))))
