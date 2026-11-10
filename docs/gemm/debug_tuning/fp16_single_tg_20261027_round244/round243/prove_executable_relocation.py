from pathlib import Path
import json,struct,hashlib,re
R=Path(__file__).resolve().parent;P=R.parent/'single-tg-pipeline-20261027-round241-64square-oneconsumer-native'
load=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
def sections(raw):
 h=struct.unpack_from('<16sHHIQQQIHHHHHH',raw);assert h[0][:6]==b'\x7fELF\x02\x01'
 ss=[struct.unpack_from('<IIQQQQIIQQ',raw,h[6]+i*h[11]) for i in range(h[12])];st=ss[h[13]];names=raw[st[4]:st[4]+st[5]];out={}
 for s in ss:
  name=names[s[0]:].split(b'\0',1)[0].decode();out[name]=dict(type=s[1],flags=s[2],address=s[3],size=s[5],align=s[8],entry_size=s[9],sha256=sha(raw[s[4]:s[4]+s[5]] if s[1]!=8 else b''))
 return out
def bound(root,sid):
 d=root/'runs'/sid;q=load(d/'result.json');matches=[p for p in (root/'cache').rglob('*.hsaco') if p.with_suffix('.json').exists() and load(p.with_suffix('.json')).get('hash')==q['metadata']['hash']];assert len(matches)==1
 p=matches[0];a=(d/'kernel.amdgcn').read_bytes();assert p.with_suffix('.amdgcn').read_bytes()==a and sha(a)==q['asm_sha256']
 return p,a,sections(p.read_bytes())
def canon(a,root):
 text=a.decode().replace(str(root),'<MEASUREMENT_ROOT>')
 # Only backend debug string offset comments change with source path lengths.
 return re.sub(r'(?m)(^\s*\.asciz\s+.*?; string offset=)\d+(\s*)$',r'\1<DEBUG_OFFSET>\2',text)
rows=[]
for n in load(R/'initial-selection.json')['nominees']:
 old=n['candidate'];fresh=next(c for c in load(R/'pipeline-candidates.json')['candidates'] if c['config']==old['config'] and c['tile']==old['tile'] and c['layout']==old['layout'])
 for a,b in zip(old['source_ids'],fresh['source_ids']):
  pa,aa,sa=bound(P,a);pb,ab,sb=bound(R,b);assert canon(aa,P)==canon(ab,R),(old['tile'],old['layout'],'ASM beyond known path relocation')
  assert sa.keys()==sb.keys();diff=[k for k in sa if sa[k]!=sb[k]];assert all(k.startswith('.debug_') for k in diff),diff
  allocated={k:v for k,v in sa.items() if v['flags']&2};assert allocated=={k:v for k,v in sb.items() if v['flags']&2} and '.note' in allocated and '.text' in allocated and '.rodata' in allocated
  rows.append(dict(tile=old['tile'],layout=old['layout'],initial_id=a,fresh_id=b,initial_hsaco=str(pa),fresh_hsaco=str(pb),initial_hsaco_sha256=sha(pa.read_bytes()),fresh_hsaco_sha256=sha(pb.read_bytes()),initial_asm_sha256=sha(aa),fresh_asm_sha256=sha(ab),canonical_asm_sha256=sha(canon(aa,P).encode()),differing_sections=diff,all_allocated_sections=allocated,all_non_debug_sections_identical=True))
out=dict(passed=True,pairs=len(rows),raw_full_ASM_identical=False,canonicalization='Exact measurement root relocation and .asciz string-offset comments only; all other assembly retained',cached_hsaco_bound_to_measured_asm=True,all_allocated_sections_and_AMDGPU_notes_identical=True,all_non_debug_ELF_sections_identical=True,rows=rows,script_sha256=sha(Path(__file__).read_bytes()))
(R/'executable-relocation-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,pairs=len(rows),only_debug_differences=True)))
