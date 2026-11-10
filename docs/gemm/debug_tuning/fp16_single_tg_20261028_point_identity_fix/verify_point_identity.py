from pathlib import Path
import ast,json,hashlib
R=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def keyfn(p):
 tree=ast.parse(p.read_text());assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SPEC_KEYS' for t in n.targets));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='point_key');ns=dict(SPEC_KEYS=ast.literal_eval(assign.value),json=json,hashlib=hashlib);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns);return ns['point_key']
old,new=keyfn(R/'old_point_key.py'),keyfn(R/'scan.py');specs=load(R/'recorded-contracts.json');assert len(specs)==194
assert len({old(s,'regression-context') for s in specs})==130
assert len({new(s,'regression-context') for s in specs})==194
z=dict(specs[0],wasp_global_k_fragments=1);omitted=dict(z);omitted.pop('wasp_global_k_fragments');assert new(z,'x')==new(omitted,'x')
for group in load(R/'published-model-contracts.json'):
 ids=[]
 for x in group['models']:
  i=old(x['spec'],x['context']);assert i==x['id'];assert x['measured_global_fragments']==x['spec'].get('wasp_global_k_fragments',1);ids.append(i)
 assert len(ids)==len(set(ids))==(34 if group['round']==298 else 23)
assert load(R/'verification.json')['passed'];print('PASS:194 distinct mixed-fragment contracts;old130;published34+23 no collisions;default1 normalized;no GPU execution')
