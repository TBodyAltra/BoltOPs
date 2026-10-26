import re
def ops(text):return [l.split(';',1)[0].strip() for l in text.splitlines() if re.match(r'^\s+(?:s_|v_|ds_|matrix_|buffer_)',l)]
def phi_phase(ll,value):
 m=re.search(r'^  '+re.escape(value)+r' = phi i32 ([^\n]+)',ll,re.M);assert m,value
 arms=re.findall(r'\[\s*([^,]+), [^\]]+\]',m[1]);assert len(arms)==2 and '0' in arms,arms;back=next(x.strip() for x in arms if x.strip()!='0')
 assert re.search(r'^  '+re.escape(back)+r' = xor i32 '+re.escape(value)+r', 1(?:,|$)',ll,re.M),(value,back)
 return back

def actual_phase_xor(producer,first_index,last_index,phase_first,phase_last,bphases):
 env={phase_first:('g',0)}
 for p in bphases:env[p]=('g',0)
 def val(s):
  if s in env:return env[s]
  try:return int(s,0)
  except ValueError:return None
 for line in producer[first_index+1:last_index]:
  m=re.fullmatch(r's_mov_b32\s+(s\d+), (.+)',line)
  if m:env[m[1]]=val(m[2]);continue
  m=re.fullmatch(r's_xor_b32\s+(s\d+), ([^,]+), ([^,]+)',line)
  if m:
   a,b=val(m[2]),val(m[3]);x=None
   if isinstance(a,tuple) and isinstance(b,int):x=('g',a[1]^b)
   elif isinstance(b,tuple) and isinstance(a,int):x=('g',b[1]^a)
   elif isinstance(a,int) and isinstance(b,int):x=a^b
   env[m[1]]=x;continue
  m=re.match(r's_\w+\s+(s\d+),',line)
  if m:env[m[1]]=None
 assert env.get(phase_last)==('g',1),(phase_first,phase_last,env.get(phase_last))

