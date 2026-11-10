"""Read or restore losslessly archived PMD statistics using their logical paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def read_stats(path):
    path=Path(path)
    if path.is_file():return path.read_bytes()
    for parent in path.parents:
        index=parent/'stats-archive.json'
        if not index.is_file():continue
        record=json.loads(index.read_text());relative=str(path.relative_to(parent))
        member=next((m for m in record['members'] if m['name']==relative),None)
        if member is None:raise FileNotFoundError(path)
        with tarfile.open(parent/record['archive'],'r:gz') as archive:
            stream=archive.extractfile(relative)
            if stream is None:raise FileNotFoundError(relative)
            data=stream.read()
        assert len(data)==member['size'] and hashlib.sha256(data).hexdigest()==member['sha256'],path
        return data
    raise FileNotFoundError(path)


def restore(case):
    case=Path(case);record=json.loads((case/'stats-archive.json').read_text())
    for member in record['members']:
        target=case/member['name'];assert target.resolve().is_relative_to(case.resolve())
        data=read_stats(target);target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file():continue
        temporary=target.with_name(target.name+'.restore-'+str(os.getpid()));temporary.write_bytes(data);temporary.replace(target)
        os.utime(target,(member['mtime'],member['mtime']))
    print(str(case)+': restored '+str(len(record['members']))+' raw statistics files; archive retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['read','restore']);parser.add_argument('path',type=Path);args=parser.parse_args()
    if args.action=='restore':restore(args.path)
    else:print(read_stats(args.path).decode(),end='')
