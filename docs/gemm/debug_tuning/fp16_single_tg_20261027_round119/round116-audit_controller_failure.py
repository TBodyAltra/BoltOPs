"""Audit six completed controls after controller-only schema failure; no alias run."""
import pathlib,json,hashlib,re,copy,sys
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));import scan
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();e=json.loads((R/'job-exit.json').read_text());assert e['returncode']==1 and not pathlib.Path('/proc/'+str(e['pid'])).exists() and json.loads((R/'state.json').read_text())['phase']=='default_controls'
assert 'accepted_features' in (R/'logs/tune.py.log').read_text() and not (R/'alias_liveness_full_gold_pilots-plan.json').exists()
a=json.loads((R/'active-manifest.json').read_text());m=json.loads((R/('manifest-'+a['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
rows=[]
for p in (R/'runs').glob('*/result.json'):
 x=json.loads(p.read_text());sp=json.loads((p.parent/'spec.json').read_text());assert x['status']=='PASS' and not sp['compiler_options'].get('hcu_wasp_alias_ab_slots',False) and not x['compiler_options'].get('hcu_wasp_alias_ab_slots',False) and x['pipeline_loops']==16 and x['accuracy']['mismatches']==x['accuracy']['nonfinite']==x['n_spills']==0 and x['accuracy']['elements']==x['tile'][0]*x['tile'][1]
 assert x['measurement_context_id']==a['measurement_context_id'] and x['asm_sha256']==sha(p.parent/'kernel.amdgcn');y=copy.deepcopy(x);scan.stats_for(p.parent,y);assert y['dispatches']==x['dispatches'] and len(y['dispatches'])==4
 expected=2*x['tile'][0]*x['tile'][1]*128/8192*16;assert all(z['instructions']['MMOP']==expected and z['active_mmac_cus']==[0] for z in y['dispatches'])
 for name in ['private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count']:
  z=re.search(r'\.'+name+r'\s*[:=]?\s*(\d+)',(p.parent/'kernel.amdgcn').read_text());assert z and int(z[1])==0
 rows.append(dict(id=x['id'],tile=x['tile'],layout=x['layout'],result_sha256=sha(p),asm_sha256=x['asm_sha256'],raw_stats_hashes={z['stats']:sha(pathlib.Path(z['stats'])) for z in y['dispatches']},full_output_gold=True,MMOP_ratio=1,all_physical_spills_zero=True))
assert len(rows)==6 and json.loads((R/'default_controls-qualification.json').read_text())['passed'];out=dict(passed=True,controller_failed=True,control_points=6,alias_points=0,raw_dispatches=24,rows=rows,source_hashes_unchanged=True,failure='Obsolete accepted_features18/late-wait gate copied into alias pilot cohort; controller failure after six controls, before any alias spec or model.',measurement_context_id=a['measurement_context_id'],script_sha256=sha(pathlib.Path(__file__)));(R/'controller-failure-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,control_points=6,alias_points=0,raw_dispatches=24)))
