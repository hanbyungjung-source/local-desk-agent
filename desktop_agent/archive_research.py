import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

import psutil


ROOT=Path(__file__).resolve().parent.parent
ARCHIVE=Path('G:/LocalDesk-research-archive/notebook')
FOLDERS=('workspace/vk-gcn','desktop_agent/data/benchmarks')
SHARE_SUFFIXES={'.dll','.exe','.a','.lib','.obj','.o','.pdb','.bin','.ptx','.spv','.zip','.7z',
                '.png','.jpg','.jpeg','.webp','.csv','.json','.jsonl','.log','.gguf','.pt'}


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''): result.update(chunk)
    return result.hexdigest()


def inventory():
    entries=[]
    for folder in FOLDERS:
        base=ROOT/folder
        if base.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise RuntimeError('Source is already a link: '+str(base))
        for current,directories,files in os.walk(base):
            for name in directories+files:
                path=Path(current)/name
                info=path.lstat()
                if info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    raise RuntimeError('Nested reparse point requires review: '+str(path))
            for name in files:
                path=Path(current)/name
                info=path.stat()
                entries.append(dict(path=str(path.relative_to(ROOT)),bytes=info.st_size,mtime_ns=info.st_mtime_ns))
    return entries


def ensure_idle():
    for process in psutil.process_iter(['name','cmdline']):
        if process.pid==os.getpid(): continue
        name=(process.info['name'] or '').lower()
        if name in ('llama-server.exe','clang.exe','clang++.exe','ninja.exe','cmake.exe'):
            raise RuntimeError('Active model/build left untouched: '+str(process.pid))
        if name.startswith('python') and '-m' in (process.info['cmdline'] or []):
            command=process.info['cmdline']
            module=command[command.index('-m')+1]
            if module.startswith(('desktop_agent.app','desktop_agent.data.benchmarks')):
                raise RuntimeError('Active application/research left untouched: '+str(process.pid))


def plan():
    entries=inventory()
    totals={folder:dict(files=0,bytes=0) for folder in FOLDERS}
    for entry in entries:
        for folder in FOLDERS:
            if (ROOT/entry['path']).is_relative_to(ROOT/folder):
                totals[folder]['files']+=1;totals[folder]['bytes']+=entry['bytes']
    print(json.dumps(dict(totals=totals,free_g=shutil.disk_usage('G:/').free,archive=str(ARCHIVE)),indent=2))


def move():
    ensure_idle()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    entries=inventory()
    needed=sum(entry['bytes'] for entry in entries)
    if shutil.disk_usage('G:/').free<needed+2*1024**3:
        raise RuntimeError('G reserve insufficient for lossless copy')
    for folder in FOLDERS:
        if (ARCHIVE/folder).exists(): raise RuntimeError('Archive exists; inspect state before resume')
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    record_dir=ROOT/'desktop_agent/data/maintenance'
    record_dir.mkdir(exist_ok=True)
    manifest=record_dir/'archive-20260922-files.jsonl'
    if manifest.exists(): raise RuntimeError('Manifest exists; inspect before resume')
    before={drive:shutil.disk_usage(drive+':/').free for drive in ('F','G')}
    canonical={};linked=[];hashes={};saved=0
    with manifest.open('x',encoding='utf-8') as output:
        for index,entry in enumerate(entries):
            source=ROOT/entry['path'];target=ARCHIVE/entry['path']
            info=source.stat()
            if (info.st_size,info.st_mtime_ns)!=(entry['bytes'],entry['mtime_ns']): raise RuntimeError('Source changed before copy')
            checksum=digest(source)
            key=(entry['bytes'],checksum)
            target.parent.mkdir(parents=True,exist_ok=True)
            shared=canonical.get(key) if source.suffix.lower() in SHARE_SUFFIXES and entry['bytes']>=4096 else None
            if shared is not None:
                os.link(shared,target)
                linked.extend((shared,target))
                saved+=entry['bytes']
            else:
                shutil.copy2(source,target)
                if digest(target)!=checksum: raise RuntimeError('Copy hash mismatch: '+str(source))
                if source.suffix.lower() in SHARE_SUFFIXES and entry['bytes']>=4096: canonical[key]=target
            if (source.stat().st_size,source.stat().st_mtime_ns)!=(entry['bytes'],entry['mtime_ns']): raise RuntimeError('Source changed during copy')
            row=dict(entry,sha256=checksum,target=str(target),canonical=str(shared) if shared else None)
            output.write(json.dumps(row)+'\n')
            hashes[entry['path']]=checksum
            if index and index%10000==0:
                output.flush();print('ARCHIVE_COPIED',index,'/',len(entries),flush=True)
        output.flush();os.fsync(output.fileno())
    for path in set(linked): path.chmod(stat.S_IREAD)
    for entry in entries:
        source=ROOT/entry['path'];target=ARCHIVE/entry['path']
        info=source.stat()
        if (info.st_size,info.st_mtime_ns)!=(entry['bytes'],entry['mtime_ns']) or digest(source)!=hashes[entry['path']]:
            raise RuntimeError('Source changed before switch: '+str(source))
        if target.stat().st_size!=entry['bytes']: raise RuntimeError('Target size changed')
    ensure_idle()
    for folder in FOLDERS:
        source=ROOT/folder;target=ARCHIVE/folder
        backup=source.with_name(source.name+'.archive-transfer-backup')
        if backup.exists(): raise RuntimeError('Backup already exists')
        source.rename(backup)
        result=subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(source),str(target)],capture_output=True,text=True)
        if result.returncode:
            backup.rename(source)
            raise RuntimeError('Junction failed: '+result.stderr)
        if source.resolve()!=target.resolve(): raise RuntimeError('Junction target differs; backup preserved')
        def remove_readonly(function,path,error):
            os.chmod(path,stat.S_IWRITE)
            function(path)
        shutil.rmtree(backup,onerror=remove_readonly)
    report=dict(status='ARCHIVED_IDENTICAL_CONTENT_DEDUPLICATED',archive=str(ARCHIVE),folders=FOLDERS,
        files=len(entries),logical_bytes=needed,duplicate_logical_bytes=saved,hardlinked_paths=len(set(linked)),
        manifest=str(manifest),manifest_sha256=digest(manifest),free_before=before,
        free_after={drive:shutil.disk_usage(drive+':/').free for drive in ('F','G')},
        unique_evidence_deleted=False,source_paths_preserved_by_junction=True,
        shared_files_readonly=True,source_code_not_hardlinked=True,
        note='Copy a read-only linked artifact to a new work file before editing. Frozen research scripts using Path.resolve may need explicit F workspace root.')
    with (record_dir/'archive-20260922.json').open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


def resume_verify():
    ensure_idle()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    record_dir=ROOT/'desktop_agent/data/maintenance'
    manifest=record_dir/'archive-20260922-files.jsonl'
    entries=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    expected={entry['path'] for entry in entries}
    assert len(expected)==len(entries)
    actual=inventory()
    assert {entry['path'] for entry in actual}==expected,'Source file list changed; preserve both copies'
    verified_targets={}
    linked=[]
    for index,entry in enumerate(entries,1):
        source=ROOT/entry['path'];target=ARCHIVE/entry['path']
        info=source.stat()
        assert (info.st_size,info.st_mtime_ns)==(entry['bytes'],entry['mtime_ns']),str(source)
        assert digest(source)==entry['sha256'],str(source)
        target_info=target.stat()
        identity=(target_info.st_dev,target_info.st_ino,target_info.st_size,target_info.st_mtime_ns)
        if identity not in verified_targets:
            verified_targets[identity]=digest(target)
        assert target_info.st_size==entry['bytes'] and verified_targets[identity]==entry['sha256'],str(target)
        if entry['canonical']:
            canonical=Path(entry['canonical'])
            assert canonical.is_relative_to(ARCHIVE) and os.path.samefile(canonical,target)
            linked.extend((canonical,target))
        if index%2000==0: print('ARCHIVE_VERIFIED',index,'/',len(entries),flush=True)
    for path in set(linked): path.chmod(stat.S_IREAD)
    for folder in FOLDERS:
        for current,directories,_ in os.walk(ROOT/folder):
            for name in directories:
                (ARCHIVE/(Path(current)/name).relative_to(ROOT)).mkdir(parents=True,exist_ok=True)
    report=dict(status='VERIFIED_READY_FOR_SWITCH',files=len(entries),manifest_sha256=digest(manifest),
        logical_bytes=sum(entry['bytes'] for entry in entries),
        duplicate_logical_bytes=sum(entry['bytes'] for entry in entries if entry['canonical']),
        unique_target_files_hashed=len(verified_targets),hardlinked_paths=len(set(linked)),
        free_before_switch={drive:shutil.disk_usage(drive+':/').free for drive in ('F','G')})
    with (record_dir/'archive-20260922-resume-verification.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print('ARCHIVE_RESUME_VERIFIED files='+str(len(entries)),flush=True)


def switch():
    ensure_idle()
    record_dir=ROOT/'desktop_agent/data/maintenance'
    manifest=record_dir/'archive-20260922-files.jsonl'
    report=json.loads((record_dir/'archive-20260922-resume-verification.json').read_text())
    assert digest(manifest)==report['manifest_sha256']
    entries=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    actual=inventory()
    assert {entry['path'] for entry in actual}=={entry['path'] for entry in entries}
    for entry in entries:
        info=(ROOT/entry['path']).stat()
        assert (info.st_size,info.st_mtime_ns)==(entry['bytes'],entry['mtime_ns']),entry['path']
    for folder in FOLDERS:
        source=ROOT/folder;target=ARCHIVE/folder
        backup=source.with_name(source.name+'.archive-transfer-backup')
        assert not backup.exists()
        source.rename(backup)
        result=subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(source),str(target)],capture_output=True,text=True)
        if result.returncode:
            backup.rename(source)
            raise RuntimeError('Junction failed: '+result.stderr)
        assert source.resolve()==target.resolve(),'Junction mismatch; backup preserved'
        def remove_readonly(function,path,error):
            os.chmod(path,stat.S_IWRITE)
            function(path)
        shutil.rmtree(backup,onerror=remove_readonly)
        print('ARCHIVE_SWITCHED',folder,flush=True)
    report.update(status='ARCHIVED_IDENTICAL_CONTENT_DEDUPLICATED',archive=str(ARCHIVE),folders=FOLDERS,
        free_after={drive:shutil.disk_usage(drive+':/').free for drive in ('F','G')},
        source_paths_preserved_by_junction=True,unique_evidence_deleted=False,
        source_code_not_hardlinked=True,shared_files_readonly=True,
        note='Frozen scripts resolving __file__ now see G; do not rerun archived experiments unchanged')
    with (record_dir/'archive-20260922.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2),flush=True)


def self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        first=Path(directory)/'first.bin';second=Path(directory)/'second.bin'
        first.write_bytes(b'fixture'*1024)
        os.link(first,second)
        assert first.stat().st_ino==second.stat().st_ino
        assert digest(first)==digest(second)
        second.unlink()
        assert first.read_bytes()==b'fixture'*1024
    print('ARCHIVE_HASH_HARDLINK_PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','move','resume_verify','switch','self_check'))
    globals()[parser.parse_args().action]()