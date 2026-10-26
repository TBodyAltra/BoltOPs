import pathlib,json,hashlib,sys,copy,os
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R));import scan
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((R/'native-qualification.json').read_text());assert q['passed'] and json.loads((R/'job-exit.json').read_text())['returncode']==0
active=json.loads((R/'active-manifest.json').read_text());m=json.loads((R/('manifest-'+active['manifest_id']+'.json')).read_text());assert all(sha(pathlib.Path(p))==h for p,h in m['hashes'].items())
rows=[]
for p in sorted((R/'runs').glob('*/result.json')):
 x=json.loads(p.read_text());assert x['measurement_context_id']==active['measurement_context_id'] and x['status']=='PASS' and x['grid']==[1] and x['input_seed']==29 and x['input_scale']==1 and x['accuracy']['elements']==x['tile'][0]*x['tile'][1] and x['accuracy']['mismatches']==x['accuracy']['nonfinite']==x['n_spills']==0
 y=copy.deepcopy(x);scan.stats_for(p.parent,y);assert x['dispatches']==y['dispatches'] and len(x['dispatches'])==4 and x['asm_sha256']==sha(p.parent/'kernel.amdgcn')
 expected=2*x['tile'][0]*x['tile'][1]*x['tile'][2]/8192*x['pipeline_loops'];assert all(d['instructions']['MMOP']==expected and d['active_mmac_cus']==[0] for d in y['dispatches'])
 rows.append(dict(id=x['id'],tile=x['tile'],layout=x['layout'],feature=x['compiler_options'].get('hcu_wasp_split_b_global_probe',False),status=x['status'],median_cycles=x['median_cycles'],MMOP_per_dispatch=expected,executed_to_logical_mmac_ratio=1,raw_stats_hashes={d['stats']:sha(pathlib.Path(d['stats'])) for d in y['dispatches']},result_sha256=sha(p),asm_sha256=x['asm_sha256']))
assert len(rows)==8 and sum(x['feature'] for x in rows)==4
out=dict(passed=True,terminal_job_exit_sha256=sha(R/'job-exit.json'),rows=rows,source_hashes_unchanged=True,no_map_promotion=True,short_precision_qualification_only=True,script_sha256=sha(pathlib.Path(__file__)))
(R/'native-raw-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,points=8,raw_dispatches=32,all_MMOP_ratios=1,all_full_output_gold=True)))
