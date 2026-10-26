"""Evaluate only integer/pointer SSA needed by actual MLS intrinsic operands.
This is an address equivalence checker, not an LLVM or GPU execution simulator.
Unrecognized required operations fail closed; no shader or model is initialized.
"""
import re
class Unknown(Exception):pass

def split(s):
 out=[];start=0;stack=[]
 for i,c in enumerate(s):
  if c in '<[({':stack.append(c)
  elif c in '>])}':stack.pop()
  elif c==',' and not stack:out.append(s[start:i].strip());start=i+1
 out.append(s[start:].strip());return out

def typed(s):
 s=s.strip();m=re.match(r'(i\d+|ptr(?: addrspace\(\d+\))?|<\d+ x i\d+>|\[\d+ x i\d+\])\s+(.*)',s)
 if not m:raise Unknown('typed '+s)
 return m[1],m[2]

def bits(t):
 if t.startswith('ptr'):return 64
 m=re.fullmatch(r'i(\d+)',t)
 if m:return int(m[1])
 raise Unknown('bits '+t)

def signed(v,n):return v-(1<<n) if v&(1<<(n-1)) else v

class Addresses:
 A=0x10000000;B=0x40000000
 def __init__(self,text,tile,layout,loops,warp,iteration):
  self.tile=tile;self.layout=layout;self.K=tile[2]*loops;self.warp=warp;self.iteration=iteration
  self.defs={m[1]:re.sub(r', !\w+ !\d+.*$','',m[2]) for l in text.splitlines() if (m:=re.match(r'\s+(%[\w.]+) = (.*)',l))}
  self.cache={f'%{i}':v for i,v in enumerate([self.A,self.B,0x80000000,0,tile[0],tile[1],self.K,self.K,tile[1] if layout=='TT' else self.K,tile[1],0,0])};self.busy=set()
  self.calls=[]
  for ln,line in enumerate(text.splitlines(),1):
   m=re.search(r'call void @(llvm.hcu.matrix.load.[\w.]+)\((.*)\)',line)
   if m:self.calls.append((ln,m[1],split(m[2])))
 def tv(self,s):t,v=typed(s);return t,self.val(v)
 def val(self,s):
  s=s.strip()
  if s in self.cache:return self.cache[s]
  if re.fullmatch(r'-?\d+',s):return int(s)
  if s in ('true','false'):return int(s=='true')
  if s in ('poison','undef'):return None
  if s=='null':return 0
  if s=='@global_smem':return 0
  if s.startswith('<') and s.endswith('>'):return [self.tv(v)[1] for v in split(s[1:-1])]
  if s.startswith('%'):
   if s in self.busy:raise Unknown('cycle '+s)
   if s not in self.defs:raise Unknown('undefined '+s)
   self.busy.add(s)
   try:v=self.expr(self.defs[s],s)
   finally:self.busy.remove(s)
   self.cache[s]=v;return v
  if s.startswith('getelementptr'):return self.expr(s)
  raise Unknown('value '+s)
 def expr(self,r,name=None):
  op=r.split(' ',1)[0];rest=r[len(op):].strip()
  if op in ('tail','musttail','notail'):return self.expr(rest,name)
  if op in ('add','sub','mul','udiv','sdiv','urem','srem','and','or','xor','shl','lshr','ashr'):
   rest=re.sub(r'^(?:(?:nuw|nsw|exact|disjoint)\s+)+','',rest);args=split(rest);t,a=self.tv(args[0]);b=self.val(args[1]);n=bits(t)
   a%=1<<n;b%=1<<n
   if op=='add':v=a+b
   elif op=='sub':v=a-b
   elif op=='mul':v=a*b
   elif op=='udiv':v=a//b
   elif op=='urem':v=a%b
   elif op in ('sdiv','srem'):
    a,b=signed(a,n),signed(b,n);div=(abs(a)//abs(b))*(-1 if (a<0)!=(b<0) else 1);v=div if op=='sdiv' else a-div*b
   elif op=='and':v=a&b
   elif op=='or':v=a|b
   elif op=='xor':v=a^b
   elif op=='shl':v=a<<b
   elif op=='lshr':v=a>>b
   else:v=signed(a,n)>>b
   return v% (1<<n)
  if op in ('zext','sext','trunc','ptrtoint','inttoptr','addrspacecast'):
   a,t=rest.rsplit(' to ',1);src,v=self.tv(a)
   if op=='sext':v=signed(v% (1<<bits(src)),bits(src))
   return v% (1<<bits(t))
  if op=='freeze':return self.tv(rest)[1]
  if op=='icmp':
   pred,rest=rest.split(' ',1);a,b=split(rest);t,a=self.tv(a);b=self.val(b);n=bits(t)
   if pred.startswith('s'):a,b=signed(a%(1<<n),n),signed(b%(1<<n),n)
   f={'eq':a==b,'ne':a!=b,'ugt':a>b,'uge':a>=b,'ult':a<b,'ule':a<=b,'sgt':a>b,'sge':a>=b,'slt':a<b,'sle':a<=b};return int(f[pred])
  if op=='select':
   cond,a,b=split(rest);return self.tv(a if self.tv(cond)[1] else b)[1]
  if op=='phi':
   t,arms=rest.split(' ',1);arms=re.findall(r'\[\s*([^,]+),\s*%[^\]]+\]',arms)
   for arm in arms:
    if arm.startswith('%') and re.match(r'add (?:nuw |nsw )*'+re.escape(t)+r' '+re.escape(name)+r', 1$',self.defs.get(arm,'')):return self.iteration
   for arm in arms:
    try:return self.val(arm)
    except Unknown:pass
   raise Unknown('phi '+r)
  if op=='getelementptr':
   rest=re.sub(r'^(?:(?:inbounds|nuw|nsw|nusw)\s+)+','',rest)
   if rest.startswith('(') and rest.endswith(')'):rest=rest[1:-1]
   fields=split(rest);base=self.tv(fields[1])[1];ty=fields[0]
   if re.fullmatch(r'i\d+',ty):assert len(fields)==3;return base+self.tv(fields[2])[1]*(bits(ty)//8)
   a=re.fullmatch(r'\[(\d+) x (i\d+)\]',ty)
   if a:
    step=bits(a[2])//8;return base+self.tv(fields[2])[1]*int(a[1])*step+(self.tv(fields[3])[1]*step if len(fields)>3 else 0)
   raise Unknown('gep '+r)
  if op=='insertelement':
   a,b,idx=split(rest);t,vec=self.tv(a);v=self.tv(b)[1];idx=self.tv(idx)[1];m=re.fullmatch(r'<(\d+) x i\d+>',t);assert m
   vec=[None]*int(m[1]) if vec is None else list(vec);vec[idx]=v;return vec
  if op=='extractelement':
   a,i=split(rest);return self.tv(a)[1][self.tv(i)[1]]
  if op=='bitcast':
   a,dst=rest.rsplit(' to ',1);src,v=self.tv(a)
   def shape(t):
    m=re.fullmatch(r'<(\d+) x i(\d+)>',t);return (int(m[1]),int(m[2])) if m else (1,bits(t))
   sn,sb=shape(src);dn,db=shape(dst);assert sn*sb==dn*db
   if sn==dn==1:return v
   vals=v if isinstance(v,list) else [v];packed=sum(x<<(i*sb) for i,x in enumerate(vals));ret=[(packed>>(i*db))&((1<<db)-1) for i in range(dn)];return ret if dn>1 else ret[0]
  if op=='call':
   m=re.search(r'@([\w.]+)\((.*)\)',rest)
   if not m:raise Unknown('call '+r)
   fn,args=m.groups();args=[self.tv(a)[1] for a in split(args)] if args else []
   if 'get.wave.id' in fn:return self.warp
   if 'workitem.id.x' in fn:return self.warp*64
   if 'workgroup.id.' in fn:return 0
   if 'readfirstlane' in fn:return args[0]
   if fn.startswith(('llvm.smin.','llvm.smax.','llvm.umin.','llvm.umax.')):
    signed_op=fn.startswith(('llvm.smin.','llvm.smax.'));n=int(fn.rsplit('i',1)[1]);args=[signed(a%(1<<n),n) for a in args] if signed_op else args;v=min(args) if 'min' in fn else max(args);return v%(1<<n)
   raise Unknown('intrinsic '+fn)
  raise Unknown('operation '+r)
 def records(self):
  out=[]
  for ln,op,args in self.calls:
   desc=self.tv(args[0])[1];assert len(desc)==4 and all(v is not None for v in desc)
   base=desc[0]|((desc[1]&65535)<<32);dest=self.tv(args[1])[1];off=self.tv(args[2])[1];flags=tuple(self.tv(x)[1] for x in args[3:]);t,r=flags[:2]
   if self.A<=base<self.A+0x10000000:operand='A';strides=(self.K,1)
   elif self.B<=base<self.B+0x10000000:operand='B';strides=(self.tile[1],1) if self.layout=='TT' else (1,self.K)
   else:raise Unknown('global base '+str(base))
   effective=base+off*strides[int(r)]*2
   out.append((operand,effective,dest,op,t,desc[2],desc[3],*flags[2:]))
  return out
