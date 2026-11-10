"""Private parser-only Dialect.o overlay on the qualified E4 compiler.
All original link inputs remain read-only and hash-stable.
"""
import pathlib,json,subprocess,hashlib,shlex,time,difflib
R=pathlib.Path(__file__).resolve().parent;B=R.parent;P=B/'single-tg-pipeline-20261023-round42-layout-buffer-ownership';F=B/'fp8-G171-wdra88-ab-20261021'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((P/'build-plan.json').read_text());cwd=pathlib.Path(old['link_cwd']);native=R/'native';native.mkdir(exist_ok=True)
if (cwd/'compile_commands.json').exists():
 commands=json.loads((cwd/'compile_commands.json').read_text());entry=next(x for x in commands if x['file'].endswith('/lib/Dialect/TritonGPU/IR/Dialect.cpp'));command=entry.get('arguments') or shlex.split(entry['command'])
else:
 target='lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/Dialect.cpp.o'
 lines=subprocess.check_output(['ninja','-t','commands',target],cwd=cwd,text=True).splitlines();matches=[line for line in lines if ' -c ' in line and '/lib/Dialect/TritonGPU/IR/Dialect.cpp' in line];assert len(matches)==1;command=shlex.split(matches[0]);entry=dict(directory=str(cwd),file=command[command.index('-c')+1])
src=pathlib.Path(entry['file']);src=src if src.is_absolute() else pathlib.Path(entry['directory'])/src;src=src.resolve();source_root=src.parents[4];ours=R/'triton/lib/Dialect/TritonGPU/IR/Dialect.cpp';assert sha(src)==sha(ours)
base=cwd/'include/triton/Dialect/TritonGPU/IR/AttrDefs.cpp.inc';text=base.read_text();snippet=(F/'parser-overlay/added-parser.cpp.txt').read_text();assert '.Case("mls_shared_import"' in snippet and 'CTAEncodingAttr::getDefault(parser.getContext(), 2)' in snippet
anchor='return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)\n';assert text.count(anchor)==1 and 'mls_shared_import' not in text;modified=text.replace(anchor,anchor+snippet,1)
overlay=native/'parser-overlay/include/triton/Dialect/TritonGPU/IR/AttrDefs.cpp.inc';overlay.parent.mkdir(parents=True,exist_ok=True);overlay.write_text(modified)
tracked=R/'triton/native-overlays';tracked.mkdir(exist_ok=True);(tracked/'added-parser.cpp.inc').write_text(snippet);(tracked/'mls-shared-import.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True),modified.splitlines(True),fromfile='AttrDefs.cpp.inc',tofile='AttrDefs.cpp.inc')))
subprocess.run(['git','-C',str(R/'triton'),'add','-N','native-overlays/added-parser.cpp.inc','native-overlays/mls-shared-import.patch'],check=True)
command=[str(x).replace(str(source_root),str(R/'triton')) for x in command];command.insert(1,'-I'+str(native/'parser-overlay/include'));command[command.index('-o')+1]=str(native/'Dialect.cpp.o');command[command.index('-c')+1]=str(ours)
link=list(old['link']);link[link.index('-o')+1]=str(native/'libtriton.so');target='lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/Dialect.cpp.o';assert link.count(target)==1;link[link.index(target)]=str(native/'Dialect.cpp.o')
inputs={}
for arg in link:
 p=pathlib.Path(arg) if pathlib.Path(arg).is_absolute() else cwd/arg
 if p.is_file() and p.suffix in ['.o','.a','.so'] and str(native) not in str(p):inputs[str(p)]=sha(p)
assert len(inputs)>=500
previous=json.loads((P/'native/link-input-hashes.json').read_text());assert all(v==previous[p] for p,v in inputs.items() if p in previous)
plan=dict(compile=command,compile_cwd=entry['directory'],link=link,link_cwd=str(cwd),overlay=str(overlay),base_generated_sha256=sha(base),overlay_sha256=sha(overlay),reference_snippet=str(F/'parser-overlay/added-parser.cpp.txt'),reference_snippet_sha256=sha(F/'parser-overlay/added-parser.cpp.txt'),CPP_source_sha256=sha(ours),stable_link_inputs=inputs)
(R/'build-plan.json').write_text(json.dumps(plan,indent=2)+'\n');(native/'link-input-hashes.json').write_text(json.dumps(inputs,indent=2)+'\n');start=time.time()
for name,args,where in [('compile',command,entry['directory']),('link',link,str(cwd))]:
 with (R/('parser-'+name+'.log')).open('w') as f:subprocess.run(args,cwd=where,stdout=f,stderr=subprocess.STDOUT,check=True)
assert all(sha(pathlib.Path(p))==v for p,v in inputs.items());assert sha(base)==plan['base_generated_sha256']
result=dict(status='BUILT_PRIVATE_PARSER_OVERLAY',sha256=sha(native/'libtriton.so'),stable_link_inputs=len(inputs),seconds=time.time()-start,parser_alias='mls_shared_import, 9 integers, constructs existingMlsShared/defaultCTA(rank2)',codegen_and_WASP_objects='ExistingE4 objects unchanged',overlay_sha256=sha(overlay));(R/'build-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
