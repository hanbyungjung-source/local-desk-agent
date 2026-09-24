import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

import psutil


OUTPUT_LIMIT = 64 * 1024 * 1024
MAX_EXECUTIONS = 8
kernel = ctypes.WinDLL('kernel32',use_last_error=True)
kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p,wintypes.LPCWSTR]
kernel.CreateJobObjectW.restype = wintypes.HANDLE
kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
kernel.SetInformationJobObject.restype = wintypes.BOOL
kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,wintypes.HANDLE]
kernel.AssignProcessToJobObject.restype = wintypes.BOOL
kernel.TerminateJobObject.argtypes = [wintypes.HANDLE,wintypes.UINT]
kernel.TerminateJobObject.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.OpenThread.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenThread.restype = wintypes.HANDLE
kernel.ResumeThread.argtypes = [wintypes.HANDLE]
kernel.ResumeThread.restype = wintypes.DWORD


class JobLimits(ctypes.Structure):
    _fields_ = [('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',wintypes.DWORD),
                ('min_working_set',ctypes.c_size_t),('max_working_set',ctypes.c_size_t),
                ('active_limit',wintypes.DWORD),('affinity',ctypes.c_size_t),
                ('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic',JobLimits),('io',ctypes.c_uint64*6),('process_memory',ctypes.c_size_t),
                ('job_memory',ctypes.c_size_t),('peak_process_memory',ctypes.c_size_t),
                ('peak_job_memory',ctypes.c_size_t)]


class ProcessJob:
    def __init__(self):
        self.handle = kernel.CreateJobObjectW(None,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000
        if not kernel.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach_and_resume(self, process):
        if not kernel.AssignProcessToJobObject(self.handle,wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
        threads = psutil.Process(process.pid).threads()
        if len(threads) != 1:
            raise ValueError('Unexpected suspended process thread count')
        thread = kernel.OpenThread(0x0002,False,threads[0].id)
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if kernel.ResumeThread(thread) != 1:
                raise ValueError('Cannot resume owned suspended PowerShell process')
        finally:
            kernel.CloseHandle(thread)

    def terminate(self):
        if self.handle and not kernel.TerminateJobObject(self.handle,1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            kernel.CloseHandle(self.handle)
            self.handle = None


class Terminals:
    def __init__(self, directory, stopped, notify):
        self.directory = Path(directory)/'terminals'
        self.stopped,self.notify = stopped,notify
        self.records = {}
        self.lock = threading.RLock()
        self.closed = False

    @staticmethod
    def validate(command, timeout_seconds):
        if not command.strip() or len(command) > 12000 or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
            raise ValueError('Command and timeout 1..3600 seconds are required')
        if re.search(r'(?i)\brunas(?:\.exe)?\b|\bget-credential\b|\bread-host\b|\b(?:-encodedcommand|-enc)\b',command):
            raise ValueError('Elevation, credential prompts and nested encoded commands are unsupported')
        if ctypes.windll.shell32.IsUserAnAdmin():
            raise ValueError('Restart Local Desk without administrator privileges before running shell tools')

    def persist(self, record):
        path = self.directory/(record['info']['execution_id']+'.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record['info'],ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(temporary,path)

    def start(self, command, cwd, timeout_seconds, *, call_id=''):
        self.validate(command,timeout_seconds)
        with self.lock:
            if self.closed or self.stopped.is_set():
                raise ValueError('Terminal session is stopped')
            if sum(not record['done'].is_set() for record in self.records.values()) >= MAX_EXECUTIONS:
                raise ValueError('At most eight terminal executions may run at once')
            self.directory.mkdir(parents=True,exist_ok=True)
            identifier = uuid.uuid4().hex
            output = self.directory/(identifier+'.log')
            output.touch(exist_ok=False)
            info = dict(execution_id=identifier,command=command,cwd=str(cwd),timeout_seconds=timeout_seconds,
                        started_at=time.time(),status='starting',pid=None,exit_code=None,output_bytes=0,call_id=call_id)
            record = dict(info=info,done=threading.Event(),job=None,process=None)
            self.records[identifier] = record
            self.persist(record)
            job,process = None,None
            try:
                job = ProcessJob()
                wrapper = ("$ErrorActionPreference='Stop'; [Console]::OutputEncoding=New-Object System.Text.UTF8Encoding $false; "
                           "$OutputEncoding=[Console]::OutputEncoding; $global:LASTEXITCODE=$null; try { & {\n"+command+
                           "\n}; if ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE } } catch { "
                           "[Console]::Error.WriteLine($_.ToString()); exit 1 }")
                encoded = base64.b64encode(wrapper.encode('utf-16le')).decode('ascii')
                executable = Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
                environment = {name:value for name,value in os.environ.items()
                               if not re.search(r'(?i)api.?key|token|password|secret',name)}
                process = subprocess.Popen([str(executable),'-NoLogo','-NoProfile','-NonInteractive','-EncodedCommand',encoded],
                    cwd=cwd,env=environment,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW|0x00000004)
                info.update(pid=process.pid,process_created_at=psutil.Process(process.pid).create_time())
                record.update(job=job,process=process)
                job.attach_and_resume(process)
                info['status'] = 'running'
                self.persist(record)
                reader = threading.Thread(target=self.capture,args=(record,output),daemon=True,name='desk-terminal-output')
                watcher = threading.Thread(target=self.watch,args=(record,reader),daemon=True,name='desk-terminal-watch')
                record.update(reader=reader,watcher=watcher)
                reader.start()
                watcher.start()
            except Exception as error:
                if job:
                    job.close()
                if process:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                    process.stdout.close()
                info.update(status='failed',error=str(error),ended_at=time.time())
                self.persist(record)
                record['done'].set()
                raise
            self.notify('terminal',dict(info))
            return dict(info)

    def capture(self, record, path):
        try:
            with path.open('ab',buffering=0) as output:
                while True:
                    data = record['process'].stdout.read1(4096)
                    if not data:
                        break
                    with self.lock:
                        remaining = OUTPUT_LIMIT-record['info']['output_bytes']
                        output.write(data[:remaining])
                        record['info']['output_bytes'] += min(len(data),remaining)
                        if len(data) > remaining:
                            self.stop(record['info']['execution_id'],'output_limit')
                            break
        except Exception as error:
            with self.lock:
                record['info']['error'] = str(error)
                self.stop(record['info']['execution_id'],'output_error')
        finally:
            record['process'].stdout.close()

    def watch(self, record, reader):
        process,info = record['process'],record['info']
        deadline = time.monotonic()+info['timeout_seconds']
        try:
            while process.poll() is None:
                if self.stopped.is_set():
                    self.stop(info['execution_id'],'cancelled')
                elif time.monotonic() >= deadline:
                    self.stop(info['execution_id'],'timed_out')
                try:
                    process.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
        except Exception as error:
            with self.lock:
                info.update(status='failed',error=str(error))
        finally:
            try:
                with self.lock:
                    record['job'].close()
                process.wait(timeout=5)
                reader.join(timeout=5)
                with self.lock:
                    info.update(exit_code=process.returncode,ended_at=time.time())
                    if info['status'] == 'running':
                        info['status'] = 'completed' if process.returncode == 0 else 'failed'
                    if reader.is_alive():
                        info.update(status='output_error',error='Output stream did not close')
                    try:
                        self.persist(record)
                    except OSError as error:
                        info.update(status='storage_error',error=str(error))
                    self.notify('terminal',dict(info))
            except Exception as error:
                with self.lock:
                    info.update(status='failed',error=str(error))
            finally:
                record['done'].set()

    def output(self, execution_id, offset=0, limit=12000):
        if not re.fullmatch('[0-9a-f]{32}',execution_id) or not 0 <= offset <= OUTPUT_LIMIT or not 1 <= limit <= 64000:
            raise ValueError('Invalid terminal output page')
        with self.lock:
            record = self.records.get(execution_id)
            if record:
                info = dict(record['info'])
            else:
                info = json.loads((self.directory/(execution_id+'.json')).read_text(encoding='utf-8'))
                if info['status'] in ('starting','running'):
                    info['status'] = 'unknown_after_restart'
            with (self.directory/(execution_id+'.log')).open('rb') as stream:
                stream.seek(offset)
                data = stream.read(limit)
            return dict(info,text=data.decode('utf-8',errors='replace'),offset=offset,next_offset=offset+len(data))

    def stop(self, execution_id, reason='cancelled'):
        with self.lock:
            record = self.records.get(execution_id)
            if record is None:
                raise ValueError('Only executions owned by this live session can be stopped')
            if not record['done'].is_set() and record['job'] and record['job'].handle:
                record['job'].terminate()
                if record['info']['status'] in ('starting','running'):
                    record['info']['status'] = reason
                self.persist(record)
            return dict(record['info'])

    def cancel_all(self):
        with self.lock:
            records = list(self.records.values())
            for record in records:
                if not record['done'].is_set():
                    try:
                        self.stop(record['info']['execution_id'])
                    except OSError as error:
                        record['info'].update(status='storage_error',error=str(error))
                        if record['job']:
                            record['job'].close()
        for record in records:
            if not record['done'].wait(6):
                raise RuntimeError('Owned terminal cleanup did not complete')

    def summary(self):
        with self.lock:
            records = list(self.records.values())
            visible = [record for record in records if not record['done'].is_set()]
            visible += [record for record in records if record['done'].is_set()][-4:]
            return [{key:record['info'].get(key) for key in ('execution_id','status','exit_code','output_bytes','call_id')} for record in visible]

    def close(self):
        self.closed = True
        self.cancel_all()