import difflib
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import uuid


MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_DIFF_CHARS = 8 * 1024 * 1024
MAX_SEARCH_SECONDS = 60
MAX_SEARCH_FILES = 100000
MAX_SEARCH_FOLDERS = 20000
MAX_SEARCH_BYTES = 512 * 1024 * 1024
PRIVATE_PARTS = {'.git','.ssh','.aws','.azure','.config','credentials','secrets','node_modules','__pycache__'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Workspace:
    def __init__(self, root, directory, check=None):
        if not root or not Path(root).is_absolute():
            raise ValueError('Select a workspace folder first')
        self.root = Path(os.path.abspath(root))
        self.directory = Path(directory).resolve()
        self.check = check or (lambda:None)
        self.protected = (self.directory,Path(__file__).parent.resolve()/'data')
        self._links(self.root)
        if not self.root.is_dir():
            raise ValueError('Workspace folder is unavailable')

    def _links(self, path):
        for part in (*reversed(path.parents),path):
            if os.path.lexists(part):
                info = part.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0) & 0x400:
                    raise ValueError('Links and junctions are not supported')

    def path(self, relative, *, directory=False):
        value = Path(relative)
        if value.is_absolute() or value.drive or not relative or any(part == '..' for part in value.parts):
            raise ValueError('Use a relative path inside the selected workspace')
        for part in value.parts:
            lowered = part.lower()
            if (':' in part or part.endswith((' ','.')) or Path(part).is_reserved()
                    or lowered in PRIVATE_PARTS or lowered.startswith('.env')
                    or lowered.endswith(('.pem','.key','.pfx','.p12','.dpapi'))
                    or 'credential' in lowered or 'secret' in lowered):
                raise ValueError('Protected or unsupported path')
        candidate = self.root/value
        self._links(candidate)
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root.resolve()) or any(resolved.is_relative_to(path) for path in self.protected):
            raise ValueError('Path is outside the workspace or in protected app data')
        if candidate.exists():
            if directory != candidate.is_dir():
                raise ValueError('Unexpected path type')
            if not directory and candidate.stat().st_nlink != 1:
                raise ValueError('Shared hardlinked files are protected')
        return candidate

    def contents(self, path):
        with path.open('rb') as stream:
            data = stream.read(MAX_FILE_BYTES+1)
        if len(data) > MAX_FILE_BYTES or b'\0' in data:
            raise ValueError('Only UTF-8 text files up to 16 MiB are supported')
        return data,data.decode('utf-8')

    def read(self, path, start_line=1, end_line=200):
        if not 1 <= start_line <= end_line or end_line-start_line >= 2000:
            raise ValueError('Read between 1 and 2000 lines per call')
        data,text = self.contents(self.path(path))
        lines = text.splitlines()
        rendered = '\n'.join(f'{index+1}: {line}' for index,line in enumerate(lines) if start_line <= index+1 <= end_line)
        return dict(path=path,sha256=digest(data),total_lines=len(lines),text=rendered[:64000],truncated=len(rendered)>64000,
                    line_ending='\r\n' if '\r\n' in text else '\n')

    def search(self, query, path='.', limit=100):
        if not query or not 1 <= limit <= 1000:
            raise ValueError('A nonempty literal query and limit 1..1000 are required')
        folder = self.path(path,directory=True)
        matches,scanned,total = [],0,0
        deadline = time.monotonic()+MAX_SEARCH_SECONDS
        for folder_count,(parent,folders,files) in enumerate(os.walk(folder,followlinks=False),1):
            self.check()
            if folder_count > MAX_SEARCH_FOLDERS or time.monotonic() >= deadline:
                return dict(matches=matches,scanned=scanned,truncated=True)
            permitted = []
            for name in sorted(folders):
                try:
                    self.path(str((Path(parent)/name).relative_to(self.root)),directory=True)
                    permitted.append(name)
                except (ValueError,OSError):
                    pass
            folders[:] = permitted
            for name in sorted(files):
                self.check()
                if scanned >= MAX_SEARCH_FILES or total >= MAX_SEARCH_BYTES or time.monotonic() >= deadline:
                    return dict(matches=matches,scanned=scanned,truncated=True)
                scanned += 1
                relative = str((Path(parent)/name).relative_to(self.root))
                try:
                    data,text = self.contents(self.path(relative))
                except (ValueError,OSError,UnicodeError):
                    continue
                total += len(data)
                for line_number,line in enumerate(text.splitlines(),1):
                    if query.casefold() in line.casefold():
                        matches.append(dict(path=relative,line=line_number,text=line[:300]))
                        if len(matches) >= limit:
                            return dict(matches=matches,scanned=scanned,truncated=True)
        return dict(matches=matches,scanned=scanned,truncated=False)

    def prepare(self, path, expected_sha256, old_text, new_text):
        target = self.path(path)
        if target.exists():
            if not target.stat().st_mode & stat.S_IWRITE or getattr(target.stat(),'st_file_attributes',0) & 1:
                raise ValueError('Read-only files are protected')
            original,text = self.contents(target)
            if digest(original) != expected_sha256:
                raise ValueError('File changed; read it again before editing')
            if not old_text or text.count(old_text) != 1:
                raise ValueError('old_text must match exactly once, including line endings')
            updated = text.replace(old_text,new_text,1).encode('utf-8')
        else:
            if expected_sha256 or old_text or not target.parent.is_dir():
                raise ValueError('New files require empty hash/old_text and an existing parent folder')
            original,updated = None,new_text.encode('utf-8')
        if len(updated) > MAX_FILE_BYTES or b'\0' in updated or original == updated:
            raise ValueError('Empty change, binary content or oversized file')
        difference = ''.join(difflib.unified_diff((original or b'').decode('utf-8').splitlines(True),
                            updated.decode('utf-8').splitlines(True),fromfile=path,tofile=path))
        if len(difference) > MAX_DIFF_CHARS:
            raise ValueError('Diff exceeds 8 MiB; make a smaller change')
        return dict(path=path,original=original,updated=updated,diff=difference)

    def commit(self, prepared):
        path = self.path(prepared['path'])
        original,updated = prepared['original'],prepared['updated']
        if original is not None:
            if not path.stat().st_mode & stat.S_IWRITE or getattr(path.stat(),'st_file_attributes',0) & 1:
                raise ValueError('Read-only files are protected')
            if self.contents(path)[0] != original:
                raise ValueError('File changed during approval; no change applied')
        elif path.exists():
            raise ValueError('File was created during approval; no change applied')
        identifier = uuid.uuid4().hex
        history = self.directory/'file-changes'/identifier
        history.mkdir(parents=True)
        if original is not None:
            (history/'before.txt').write_bytes(original)
        (history/'after.txt').write_bytes(updated)
        (history/'change.diff').write_text(prepared['diff'],encoding='utf-8')
        record = dict(change_id=identifier,path=str(path),workspace_root=str(self.root),status='prepared',
                      before_sha256=digest(original) if original is not None else '',after_sha256=digest(updated),
                      diff=prepared['diff'],time=time.time())
        manifest = history/'change.json'
        manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary = None
        try:
            self.path(prepared['path'])
            if original is None:
                with path.open('xb') as stream:
                    stream.write(updated)
            else:
                descriptor,temporary = tempfile.mkstemp(dir=path.parent,prefix='.local-desk-')
                with os.fdopen(descriptor,'wb') as stream:
                    stream.write(updated)
                    stream.flush()
                    os.fsync(stream.fileno())
                if self.contents(self.path(prepared['path']))[0] != original:
                    raise ValueError('File changed before replacement; no change applied')
                os.replace(temporary,path)
            record['status'] = 'applied'
        except Exception as error:
            record.update(status='failed',error=str(error))
            raise
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
            manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        return record