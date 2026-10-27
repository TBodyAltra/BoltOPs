from pathlib import Path
import json,hashlib,fcntl,os
R=Path(__file__).resolve().parent;K=R.parent/'single-tg-continuation-20261023';P=K/'published/natural-stride-map.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='96346c10712e04043e3b52f311c63210f52b2fd6d289073cae4c9b236b4603ed'
with (K/'map-integration.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert sha(P)==parent and not (R/'fragment-metadata-correction-round237.json').exists();old=P.read_bytes();m=json.loads(old);keys=['best','best_no_spill','best_64k','best_64k_no_spill'];before={k:json.dumps(m[k],sort_keys=True) for k in keys};rows=[x for x in m['candidates'] if x.get('measurement_context_id')=='844baaadee3c4433'];assert len(rows)==24;changes=[]
 for x in rows:
  expected=x['tile'][2]//x['config']['MMA_K_SPLIT']
  if x['dot_fragment_K']!=expected:changes.append(dict(source_ids=x['source_ids'],old=x['dot_fragment_K'],new=expected));x['dot_fragment_K']=expected
 assert changes
 for k in keys:assert json.dumps(m[k],sort_keys=True)==before[k]
 proof=dict(passed=True,parent_map_sha256=parent,corrected_fields=len(changes),changes=changes,reason='Round236 integrator inherited fixed dot_fragment_K64 from WDRA-only cohort;64square includes MMA_K_SPLIT1 and2,so derive128/split exactly',no_performance_or_model_data_changes=True,all503_choices_unchanged=True,script_sha256=sha(Path(__file__)));m['continuation_round237_metadata_correction']=proof;backup=K/'published/natural-stride-map-round236-before-round237.json';assert not backup.exists();backup.write_bytes(old);tmp=P.with_name(P.name+'.round237.tmp');tmp.write_text(json.dumps(m,indent=2,allow_nan=False)+'\n');os.replace(tmp,P);proof['new_map_sha256']=sha(P);(R/'fragment-metadata-correction-round237.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(passed=True,corrected_fields=len(changes),new_map_sha256=sha(P))))
