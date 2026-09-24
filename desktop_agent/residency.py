import hashlib
import json
import os
from pathlib import Path
import threading
import time

from desktop_agent import models


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def deployment(model,projector):
    runtime=models.Q2_FRONT_RUNTIME
    record=json.loads((runtime/'deployment.json').read_text(encoding='utf-8'))
    if record.get('schema')!=1 or record.get('baseline_id')!='front_native_postencode':
        raise ValueError('Invalid front residency deployment')
    for kind,filename in (('model',model),('projector',projector)):
        expected=record[kind]
        path=Path(filename).resolve()
        if path!=Path(expected['path']).resolve():
            raise ValueError('Front residency requires the validated '+kind+' path')
        stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(expected['size'],expected['mtime_ns']):
            raise ValueError('Validated '+kind+' changed; front residency disabled')
    for name,digest in record['files'].items():
        if Path(name).name!=name or sha256(runtime/name)!=digest:
            raise ValueError('Front runtime file mismatch: '+name)
    table=runtime/'workspace-table.json'
    if sha256(table)!=record['table_sha256'] or json.loads(table.read_text())['identity']!=record['identity']:
        raise ValueError('Front workspace table mismatch')
    child={key:value for key,value in os.environ.items() if not key.startswith(('GGML_','LOCAL_DESK_'))}
    for key in ('CUDA_LAUNCH_BLOCKING','CUDA_LOG_FILE','CUDA_VISIBLE_DEVICES','LLAMA_GRAPH_REUSE_DISABLE','LLAMA_MTP_IMAGE_PACKED'):
        child.pop(key,None)
    child.update(record['environment'])
    child.update(GGML_BACKEND_PATH=str(runtime/'ggml-vulkan.dll'),LOCAL_DESK_FRONT_TABLE=str(table),
        PATH=str(runtime)+os.pathsep+child.get('PATH',''))
    return record,child


def memory_reason(row,available_ram,pid,record,require_process=False):
    caps=record['caps']
    rtx,rx=record['rtx_luid'],record['rx_luid']
    if available_ram<caps['available_ram']:
        return 'Front residency: available RAM below 4 GiB'
    for adapter,limit in ((rtx,caps['rtx_dedicated']),(rx,(8192-128)*1024**2)):
        used=row['adapter_dedicated'].get(adapter)
        if used is None or used<0:
            return 'Front residency: required adapter telemetry missing'
        if used>limit:
            return 'Front residency: dedicated GPU memory limit exceeded'
    if pid is not None:
        value=row['process_shared'].get(f'pid_{pid}_{rtx}')
        if require_process and value is None:
            return 'Front residency: process GPU telemetry missing'
        if value is not None and (value<0 or value>caps['rtx_shared']):
            return 'Front residency: RTX shared GPU memory limit exceeded'
        if any(value>384*1024**2 for key,value in row['process_shared'].items() if key.startswith(f'pid_{pid}_')):
            return 'Front residency: shared GPU memory limit exceeded'
    return ''


class ResidencyGuard:
    def __init__(self,server,stopped,record):
        from desktop_agent.benchmark_resources import GPUCounters,system_memory
        self.server,self.stopped,self.record=server,stopped,record
        self.counters=GPUCounters()
        self.system_memory=system_memory
        self.finished=threading.Event()
        self.thread=None
        self.sample_lock=threading.Lock()
        self.reason=''
        self.process_seen=False
        self.engine_missing_since=None
        self.request_started=None

    def sample(self):
        with self.sample_lock:
            self._sample()

    def _sample(self):
        began=time.monotonic()
        row=self.counters.read()
        available=self.system_memory()['physical_available']
        process=self.server.process
        if process is not None and process.poll() is not None:
            return
        pid=process.pid if process is not None else None
        key=f'pid_{pid}_{self.record["rtx_luid"]}'
        self.process_seen=self.process_seen or key in row['process_shared']
        reason=memory_reason(row,available,pid,self.record,self.process_seen)
        if time.monotonic()-began>1:
            reason=reason or 'Front residency: stale memory sample'
        if self.process_seen and not row.get('engine_utilization'):
            self.engine_missing_since=self.engine_missing_since or began
            if began-self.engine_missing_since>5:
                reason=reason or 'Front residency: GPU utilization telemetry missing'
        else:
            self.engine_missing_since=None
        if reason: raise RuntimeError(reason)

    def start(self):
        try:
            self.sample()
        except Exception:
            self.counters.close()
            raise
        def monitor():
            previous=time.monotonic()
            while not self.finished.wait(0.1):
                try:
                    now=time.monotonic()
                    if now-previous>1:
                        raise RuntimeError('Front residency: memory monitor stalled')
                    self.sample()
                    previous=now
                except Exception as error:
                    self.reason=str(error)
                    self.stopped.set()
                    process=self.server.process
                    if process is not None and process.poll() is None:
                        try: process.terminate()
                        except OSError: pass
                    return
        self.thread=threading.Thread(target=monitor,daemon=True)
        self.thread.start()

    def begin_request(self):
        if self.reason:
            raise RuntimeError(self.reason)
        self.sample()
        if not self.process_seen: raise RuntimeError('Front residency: GPU process not observed')
        self.request_started=time.monotonic()

    def end_request(self):
        self.request_started=None

    def close(self):
        self.finished.set()
        if self.thread: self.thread.join(timeout=5)
        if not self.thread or not self.thread.is_alive(): self.counters.close()