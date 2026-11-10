"""Verify and restore the case archives into the repository layout."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def join_parts(directory):
    record=ROOT/'provenance/DATA_PACKAGES.json'
    expected={Path(p['archive']).name:p['sha256'] for p in json.loads(record.read_text())}
    joined=[]
    for first in sorted(directory.rglob('*.zip.part-001')):
        name=first.name.removesuffix('.part-001')
        parts=sorted(first.parent.glob(name+'.part-*'))
        for i,part in enumerate(parts,1):
            if part.name!=name+f'.part-{i:03d}':raise ValueError('Missing or unexpected archive part: '+name)
        destination=first.with_name(name)
        if destination.exists():
            with destination.open('rb') as f:
                if digest(f)!=expected.get(name):raise FileExistsError(destination)
            joined.append(destination);continue
        temporary=destination.with_suffix('.joining')
        if temporary.exists():raise FileExistsError(temporary)
        h=hashlib.sha256()
        try:
            with temporary.open('xb') as out:
                for part in parts:
                    with part.open('rb') as src:
                        for block in iter(lambda:src.read(2**20),b''):h.update(block);out.write(block)
            if h.hexdigest()!=expected.get(name):raise ValueError('Archive checksum mismatch after joining: '+name)
            temporary.rename(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f'{name}: joined and verified {len(parts)} parts')
        joined.append(destination)
    return joined

def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def restore(path):
    with zipfile.ZipFile(path) as z:
        rows=list(csv.DictReader(io.StringIO(z.read('_release/FILES.csv').decode('utf-8-sig'))))
        for row in rows:
            part=PurePosixPath(row['path'])
            if part.is_absolute() or '..' in part.parts:raise ValueError('Unsafe archive path')
            destination=ROOT.joinpath(*part.parts)
            if not destination.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Data destination leaves repository')
            with z.open(row['path']) as f:
                if digest(f)!=row['sha256']:raise ValueError('Archive member hash mismatch: '+row['path'])
            if destination.is_symlink():raise ValueError('Existing symlink at data destination')
            if destination.exists():
                with destination.open('rb') as f:
                    if digest(f)!=row['sha256']:raise FileExistsError(destination)
                continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            with z.open(row['path']) as src,destination.open('xb') as out:shutil.copyfileobj(src,out)
        print(f'{path.name}: {len(rows)} verified files')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('download_directory',type=Path);args=p.parse_args()
    join_parts(args.download_directory)
    archives=sorted(args.download_directory.rglob('*.zip'))
    if not archives:raise FileNotFoundError('No data archives found')
    expected={'shared_provenance.zip','simulation.zip','pdac.zip','mouse_inputs.zip','mouse_results.zip','mouse_fitted_models.zip'}
    for archive in archives:
        if archive.name in expected:restore(archive)
    alias=ROOT/'04_数值实验/overleaf'
    if not alias.exists() and not alias.is_symlink():alias.symlink_to('../05_论文草稿/latex',target_is_directory=True)
    missing=expected-{p.name for p in archives}
    if missing:print('Not supplied: '+', '.join(sorted(missing)))

if __name__=='__main__':main()
