"""Native-only G171 register-bank experiments; no production dispatch changes.

Validated for the pinned natural single-CTA M128/N256/K6144 WDRA48 IR.
The two-iteration loop must restore BOTH loop-carried and raw A-scale registers:
the epilogue consumes the raw bank. Keep all hardware waits and barriers.
Use this module through prepare.py; output artifacts must live outside git.
"""
import re

def transform(asm,mode):
 assert mode in ['rotate-A','rotate-AB','unroll-only'];proof=[]
 # Fixed external experiment for the verified K6144 natural single-CTA specialization.
 for label in ['.LBB0_48','.LBB0_53']:
  start=asm.index(label+':');br=re.search(r'^\s*s_cbranch_scc1 '+re.escape(label)+r'\s*$',asm[start:],re.M);assert br
  end=start+br.end();original=asm[start:end];body=original.split('\n',1)[1];body=body[:body.rfind('\ts_cbranch_scc1')]
  # Drop only duplicate debug directives; retain all executable waits/barriers and arithmetic.
  body=''.join(l for l in body.splitlines(True) if not re.match(r'\s*\.loc\b',l))
  if mode=='unroll-only':
   second=''.join(l for l in body.splitlines(True) if not re.match(r'\s*\.Ltmp\d+:',l))
   new=label+':\n'+body+f'\ts_cbranch_scc0 {label}_done\n'+second+f'\ts_cbranch_scc1 {label}\n{label}_done:\n'
   asm=asm[:start]+new+asm[end:];proof.append(dict(label=label,mode=mode));continue
  moves=list(re.finditer(r'^\s*v_mov_b32_e32 v(\d+), v(\d+)\s*$',body,re.M));pairs=[(int(m[1]),int(m[2])) for m in moves if 84<=int(m[1])<100 and 68<=int(m[2])<84];assert len(pairs)==16
  raw_to_dest={s:d for d,s in pairs};fix=''.join(f'\tv_mov_b32_e32 v{src}, v{210+d-84}\n' for d,src in pairs)+''.join(f'\tv_mov_b32_e32 v{d}, v{210+d-84}\n' for d,src in pairs)
  even_fix=''.join(f'\tv_mov_b32_e32 v{src}, v{d}\n' for d,src in pairs)
  # Future A scale gets its own bank. Target the final register order directly in LDS reads.
  for raw in range(68,84,2):
   dest=raw_to_dest[raw];assert raw_to_dest[raw+1]==dest+1
   pattern=r'(ds_read2_b32 )v\['+str(raw)+':'+str(raw+1)+r'\](, v70\b)'
   body,n=re.subn(pattern,lambda m:m[1]+f'v[{210+dest-84}:{211+dest-84}]'+m[2],body);assert n==1,(label,raw,n)
  for d,s in pairs:body,n=re.subn(r'^\s*v_mov_b32_e32 v'+str(d)+', v'+str(s)+r'\s*$', '',body,flags=re.M);assert n==1
  removed_b=0
  if mode=='rotate-AB':
   # Consume raw current B before future B overwrites it. Use a separate address register.
   for dst,src in zip(range(206,210),range(64,68)):
    body,n=re.subn(r'^\s*v_mov_b32_e32 v'+str(dst)+', v'+str(src)+r'\s*$', '',body,flags=re.M);assert n==1;removed_b+=1
    body=re.sub(r'\bv'+str(dst)+r'\b','v'+str(src),body)
   body,n=re.subn(r'(v_lshl_add_u32 )v66(, s10, 10, v103)',r'\g<1>v226\2',body);assert n==1
   reads=re.findall(r'^\s*ds_read2_b32 v\[(?:64:65|66:67)\], v66[^\n]*\n',body,re.M);assert len(reads)==2
   for l in reads:body=body.replace(l,'',1)
   pos=list(re.finditer(r'^\s*v_lshrrev_b32_e32 v\d+, s2, v(?:64|65|66|67)\s*$',body,re.M));assert len(pos)==4
   at=pos[-1].end();body=body[:at]+'\n'+''.join(l.replace(', v66',', v226') for l in reads)+body[at:]
  def swap(t):
   t="".join(l for l in t.splitlines(True) if not re.match(r"\s*\.Ltmp\d+:",l))
   def n(x):return x+126 if 84<=x<=99 else x-126 if 210<=x<=225 else x
   # Every renamed interval is contiguous. Validate this before spelling the range.
   def ran(m):
    a,b=int(m[1]),int(m[2]);vals=[n(v) for v in range(a,b+1)];assert vals==list(range(vals[0],vals[-1]+1));return f'v[{vals[0]}:{vals[-1]}]'
   t=re.sub(r'v\[(\d+):(\d+)\]',ran,t)
   return re.sub(r'\bv(\d+)\b',lambda m:'v'+str(n(int(m[1]))),t)
  # Restore raw A live-outs and carried A at loop exits; steady-state has no copies.
  new=label+':\n'+body+f'\ts_cbranch_scc0 {label}_odd_exit\n'+swap(body)+f'\ts_cbranch_scc1 {label}\n'+even_fix+f'\ts_branch {label}_done\n{label}_odd_exit:\n'+fix+label+'_done:\n'
  asm=asm[:start]+new+asm[end:];proof.append(dict(label=label,A_copies_removed_per_iteration=16,B_copies_removed_per_iteration=removed_b,odd_exit_A_copies=32,even_exit_A_copies=16,exit_note="restore raw A registers as well as loop-carried A: epilogue consumes raw bank 68:83",A_banks=[[84,99],[210,225]],B_prefetch='after last raw-B consumer' if removed_b else 'unchanged',unroll=2))
 needed=227 if mode=='rotate-AB' else 226
 # Preserve original launch descriptor: dynamic WDRA quota stays 248 in both consumers.
 return asm,dict(mode=mode,consumer_vgpr_high_water=needed,wdra_consumer_quota=248,loops=proof,math='unchanged packed mul/FMA order; register renaming and LDS destination/scheduling only')
