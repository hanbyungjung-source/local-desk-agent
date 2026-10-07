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


def front_profile(context_tokens,kv_cache_type,cache_ram_mib):
    if type(context_tokens) is not int or not 4096 <= context_tokens <= 65536 or context_tokens%1024:
        raise ValueError('Front context must be 4096..65536 in multiples of 1024')
    if kv_cache_type not in ('default','q4_0','q8_0','f16'):
        raise ValueError('Unsupported front KV type')
    if type(cache_ram_mib) is not int or cache_ram_mib not in (0,2048):
        raise ValueError('Unsupported front RAM cache size')
    profile=dict(context_tokens=context_tokens,kv_cache_type='q8_0' if kv_cache_type=='default' else kv_cache_type,
                 cache_ram_mib=cache_ram_mib,n_batch=512,n_ubatch=128,n_seq_max=1)
    profile['identity']=hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode('ascii')).hexdigest()
    return profile


def gpu_adapters():
    import ctypes
    from ctypes import wintypes
    import uuid
    class Luid(ctypes.Structure):
        _fields_=[('low',wintypes.DWORD),('high',wintypes.LONG)]
    class Description(ctypes.Structure):
        _fields_=[('name',wintypes.WCHAR*128),('vendor',wintypes.UINT),('device',wintypes.UINT),
                  ('subsystem',wintypes.UINT),('revision',wintypes.UINT),('dedicated',ctypes.c_size_t),
                  ('system',ctypes.c_size_t),('shared',ctypes.c_size_t),('luid',Luid),('flags',wintypes.UINT)]
    def method(pointer,index,result,*arguments):
        table=ctypes.cast(pointer,ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result,ctypes.c_void_p,*arguments)(table[index])
    library=ctypes.WinDLL('dxgi',use_last_error=True)
    library.CreateDXGIFactory1.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
    library.CreateDXGIFactory1.restype=ctypes.c_long
    identity=(ctypes.c_ubyte*16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    factory=ctypes.c_void_p()
    if library.CreateDXGIFactory1(identity,ctypes.byref(factory))!=0:
        raise RuntimeError('DXGI factory unavailable')
    rows=[]
    try:
        for index in range(16):
            adapter=ctypes.c_void_p()
            status=method(factory,12,ctypes.c_long,wintypes.UINT,ctypes.POINTER(ctypes.c_void_p))(factory,index,ctypes.byref(adapter))
            if status & 0xffffffff==0x887a0002:
                break
            if status!=0:
                raise RuntimeError('DXGI adapter enumeration failed')
            try:
                description=Description()
                if method(adapter,10,ctypes.c_long,ctypes.POINTER(Description))(adapter,ctypes.byref(description))!=0:
                    raise RuntimeError('DXGI adapter description failed')
                rows.append(dict(name=description.name,vendor=description.vendor,device=description.device,
                                 dedicated_bytes=description.dedicated,software=bool(description.flags & 2),
                                 luid=f'luid_0x{description.luid.high & 0xffffffff:08X}_0x{description.luid.low:08X}_phys_0'))
            finally:
                method(adapter,2,wintypes.ULONG)(adapter)
    finally:
        method(factory,2,wintypes.ULONG)(factory)
    return rows


def profile_adapters(rows):
    result={}
    for name,vendor,device in (('rtx',0x10de,0x2d83),('rx',0x1002,0x6fdf)):
        matched=[row for row in rows if not row['software'] and row['vendor']==vendor]
        if len(matched)!=1 or matched[0]['device']!=device:
            raise ValueError('Front profile requires one RTX5050 and one RX580 2048SP')
        result[name+'_luid']=matched[0]['luid']
    return result


def deployment(model,projector,*,profile=None):
    runtime=models.Q2_PROFILE_RUNTIME if profile is not None else models.Q2_FRONT_RUNTIME
    record=json.loads((runtime/'deployment.json').read_text(encoding='utf-8'))
    expected_schema,expected_id=(2,'front_native_profiles_v1') if profile is not None else (1,'front_native_postencode')
    if record.get('schema')!=expected_schema or record.get('baseline_id')!=expected_id:
        raise ValueError('Invalid front residency deployment')
    if profile is not None and profile!=front_profile(profile['context_tokens'],profile['kv_cache_type'],profile['cache_ram_mib']):
        raise ValueError('Invalid front profile identity')
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
    if profile is None and (sha256(table)!=record['table_sha256'] or json.loads(table.read_text())['identity']!=record['identity']):
        raise ValueError('Front workspace table mismatch')
    child={key:value for key,value in os.environ.items() if not key.startswith(('GGML_','LOCAL_DESK_'))}
    for key in ('CUDA_LAUNCH_BLOCKING','CUDA_LOG_FILE','CUDA_VISIBLE_DEVICES','LLAMA_GRAPH_REUSE_DISABLE','LLAMA_MTP_IMAGE_PACKED'):
        child.pop(key,None)
    child.update(record['environment'])
    child.update(GGML_BACKEND_PATH=str(runtime/'ggml-vulkan.dll'),LOCAL_DESK_FRONT_TABLE=str(table),
        PATH=str(runtime)+os.pathsep+child.get('PATH',''))
    if profile is not None:
        if record.get('profile_planning')!='runtime_signature':
            raise ValueError('Front profile planning contract mismatch')
        expected_caps=dict(rtx_dedicated=7850*1024**2,rtx_shared=190*1024**2,available_ram=4*1024**3,headroom=32*1024**2)
        if record.get('caps')!=expected_caps:
            raise ValueError('Front profile memory caps differ from the native runtime')
        record=dict(record,profile=profile,context_tokens=profile['context_tokens'],cache_ram_mib=profile['cache_ram_mib'],
                    **profile_adapters(gpu_adapters()))
        child.pop('LOCAL_DESK_FRONT_TABLE',None)
        child.update(LOCAL_DESK_FRONT_DYNAMIC_PROFILE='1',LOCAL_DESK_FRONT_ID=profile['identity'],
                     LOCAL_DESK_RTX_LUID=record['rtx_luid'])
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