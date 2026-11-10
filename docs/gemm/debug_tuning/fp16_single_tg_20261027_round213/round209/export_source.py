"""Export tracked and untracked source without altering the frozen209 index/tree."""
from pathlib import Path
import subprocess,json,hashlib,os,stat
R=Path(__file__).resolve().parent;B=R.parent;C=B/'single-tg-pipeline-20261027-round209-deterministic-mls-cpu';T=C/'triton';env=dict(os.environ,GIT_OPTIONAL_LOCKS='0');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(T),*args],env=env)
base=git('rev-parse','HEAD').decode().strip();assert base=='46f0bed3561364b63ba4046d9c8c0cb70380ea30';index=Path(git('rev-parse','--git-path','index').decode().strip());index=index if index.is_absolute() else T/index;index_before=sha(index);tracked=git('diff','HEAD','--binary');assert tracked==(C/'triton-source.patch').read_bytes();changed=[x.decode() for x in git('diff','--no-renames','--name-only','-z','HEAD').split(b'\0') if x];untracked=[x.decode() for x in git('ls-files','--others','--exclude-standard','-z').split(b'\0') if x];included=[];excluded=[];parts=[tracked]
for name in untracked:
 p=T/name
 if name.endswith(('.so','.pyc','.pyd')):
  excluded.append(dict(path=name,reason='Qualified generated binary dependency; not source',symlink_target=str(p.resolve()),sha256=sha(p)));continue
 assert p.is_file() and p.suffix in ['.py','.inc','.patch','.h','.cpp','.td','.txt','.md','.json'],name
 diff=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],cwd=T,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);assert diff.returncode==1 and diff.stdout,name;parts.append(diff.stdout);included.append(name)
assert len(included)==13 and len(excluded)==1
patch=R/'triton-parent209-complete-source.patch';patch.write_bytes(b''.join(parts));assert not(R/'triton').exists();subprocess.run(['git','-C',str(T),'worktree','add','-b','perf/complete-source-export-round226-triton',str(R/'triton'),base],env=env,check=True,stdout=subprocess.DEVNULL);subprocess.run(['git','-C',str(R/'triton'),'apply',str(patch)],env=env,check=True,stdout=subprocess.DEVNULL)
files={}
for name in sorted(set(changed+included)):
 src=T/name;dst=R/'triton'/name
 if src.is_symlink():assert dst.is_symlink() and os.readlink(src)==os.readlink(dst);files[name]=dict(kind='symlink',target=os.readlink(src))
 elif src.exists():assert src.read_bytes()==dst.read_bytes();files[name]=dict(kind='regular',sha256=sha(src),bytes=src.stat().st_size)
 else:assert not dst.exists();files[name]=dict(kind='deleted')
assert sha(index)==index_before and git('diff','HEAD','--binary')==tracked
out=dict(passed=True,source_only=True,no_PMD_execution=True,no_native_rebuild=True,no_measured_source_or_index_change=True,source_worktree=str(T),base_commit=base,original_tracked_only_patch=str(C/'triton-source.patch'),original_tracked_only_patch_sha256=sha(C/'triton-source.patch'),complete_source_patch=str(patch),complete_source_patch_sha256=sha(patch),new_source_files_missing_from_original_patch=included,excluded_generated_binary_dependencies=excluded,changed_source_files_verified=files,applied_to_fresh_isolated_worktree=str(R/'triton'),fresh_source_bytes_equal=True,original_index_sha256_before=index_before,original_index_sha256_after=sha(index),qualified_native_sha256=sha(C/'native/libtriton.so'),native_provenance='Retains the separately recorded503 qualified link inputs and one MLS replacement object; this source export does not assert that a full-tree rebuild produces the measured native.',script_sha256=sha(Path(__file__)))
(R/'complete-source-export-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,included_untracked_source=len(included),excluded_binary=len(excluded),verified_source_files=len(files),complete_patch_sha256=sha(patch),bytes=patch.stat().st_size,original_source_and_index_unchanged=True)))
