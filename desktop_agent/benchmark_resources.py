import ctypes
from ctypes import wintypes
from collections import defaultdict
import os
import re
import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import numpy as np
import psutil


class MemoryRegion(ctypes.Structure):
    _fields_ = [('base',ctypes.c_void_p),('allocation_base',ctypes.c_void_p),
                ('allocation_protect',wintypes.DWORD),('partition',wintypes.WORD),
                ('size',ctypes.c_size_t),('state',wintypes.DWORD),
                ('protect',wintypes.DWORD),('kind',wintypes.DWORD)]


class PerformanceInfo(ctypes.Structure):
    _fields_ = [('size',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in (
        'commit_total','commit_limit','commit_peak','physical_total','physical_available',
        'system_cache','kernel_total','kernel_paged','kernel_nonpaged','page_size')]+[
        ('handles',wintypes.DWORD),('processes',wintypes.DWORD),('threads',wintypes.DWORD)]


kernel = ctypes.WinDLL('kernel32',use_last_error=True)
psapi = ctypes.WinDLL('psapi',use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.VirtualQueryEx.argtypes = [wintypes.HANDLE,ctypes.c_void_p,ctypes.POINTER(MemoryRegion),ctypes.c_size_t]
kernel.VirtualQueryEx.restype = ctypes.c_size_t
psapi.QueryWorkingSet.argtypes = [wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
psapi.QueryWorkingSet.restype = wintypes.BOOL
psapi.GetMappedFileNameW.argtypes = [wintypes.HANDLE,ctypes.c_void_p,wintypes.LPWSTR,wintypes.DWORD]
psapi.GetMappedFileNameW.restype = wintypes.DWORD
psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PerformanceInfo),wintypes.DWORD]
psapi.GetPerformanceInfo.restype = wintypes.BOOL
kernel.OpenThread.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenThread.restype = wintypes.HANDLE
kernel.SuspendThread.argtypes = [wintypes.HANDLE]
kernel.SuspendThread.restype = wintypes.DWORD
kernel.ResumeThread.argtypes = [wintypes.HANDLE]
kernel.ResumeThread.restype = wintypes.DWORD
kernel.GetThreadContext.argtypes = [wintypes.HANDLE,ctypes.c_void_p]
kernel.GetThreadContext.restype = wintypes.BOOL


def instruction_location(pid,thread_id):
    if thread_id == threading.get_native_id():
        raise ValueError('Cannot suspend the calling thread')
    target = kernel.OpenProcess(0x410,False,pid)
    thread = kernel.OpenThread(0x4A,False,thread_id)
    if not target or not thread:
        if target:
            kernel.CloseHandle(target)
        if thread:
            kernel.CloseHandle(thread)
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        storage = ctypes.create_string_buffer(1232+15)
        aligned = (ctypes.addressof(storage)+15) & ~15
        ctypes.c_uint32.from_address(aligned+48).value = 0x100001
        if kernel.SuspendThread(thread) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel.GetThreadContext(thread,aligned):
                raise ctypes.WinError(ctypes.get_last_error())
            address = ctypes.c_uint64.from_address(aligned+248).value
        finally:
            if kernel.ResumeThread(thread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        region = MemoryRegion()
        if not kernel.VirtualQueryEx(target,address,ctypes.byref(region),ctypes.sizeof(region)):
            raise ctypes.WinError(ctypes.get_last_error())
        path = ctypes.create_unicode_buffer(32768)
        psapi.GetMappedFileNameW(target,address,path,len(path))
        return dict(thread_id=thread_id,address=address,module=path.value,
                    module_offset=address-(region.allocation_base or 0),region_kind=region.kind)
    finally:
        kernel.CloseHandle(thread)
        kernel.CloseHandle(target)


def system_memory():
    info = PerformanceInfo()
    info.size = ctypes.sizeof(info)
    if not psapi.GetPerformanceInfo(ctypes.byref(info),info.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return {name:getattr(info,name)*info.page_size for name in (
        'commit_total','commit_limit','physical_total','physical_available','system_cache','kernel_paged','kernel_nonpaged')}


def memory_snapshot(pid):
    started = time.perf_counter()
    handle = kernel.OpenProcess(0x410,False,pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        regions = []
        paths = {}
        address = 0
        while address < 0x7FFFFFFFFFFF:
            region = MemoryRegion()
            if not kernel.VirtualQueryEx(handle,address,ctypes.byref(region),ctypes.sizeof(region)):
                break
            base = region.base or 0
            if region.state == 0x1000:
                path = ''
                if region.kind in (0x40000,0x1000000):
                    allocation = region.allocation_base or base
                    if allocation not in paths:
                        text = ctypes.create_unicode_buffer(32768)
                        length = psapi.GetMappedFileNameW(handle,base,text,len(text))
                        paths[allocation] = text.value if length else ''
                    path = paths[allocation]
                regions.append((base,region.size,region.kind,path))
            following = base+region.size
            if following <= address:
                raise RuntimeError('Non-advancing memory region')
            address = following
        capacity = 1024*1024
        for attempt in range(4):
            buffer = np.empty(capacity,dtype=np.uint64)
            if psapi.QueryWorkingSet(handle,buffer.ctypes.data,buffer.nbytes):
                break
            error = ctypes.get_last_error()
            if error != 24:
                raise ctypes.WinError(error)
            capacity *= 2
        else:
            raise RuntimeError('Working set exceeded query capacity')
        count = int(buffer[0])
        if count >= capacity:
            raise RuntimeError('Invalid working set length')
        pages = buffer[1:count+1]
        addresses = pages & np.uint64(0xFFFFFFFFFFFFF000)
        bases = np.array([region[0] for region in regions],dtype=np.uint64)
        sizes = np.array([region[1] for region in regions],dtype=np.uint64)
        indices = np.searchsorted(bases,addresses,side='right')-1
        safe_indices = np.maximum(indices,0)
        valid = (indices >= 0) & (addresses < bases[safe_indices]+sizes[safe_indices])
        counts = np.bincount(indices[valid],minlength=len(regions))
        private_counts = np.bincount(indices[valid & ((pages & 0x100)==0)],minlength=len(regions))
        totals = defaultdict(lambda:dict(virtual_committed=0,resident=0,nonshareable_resident=0))
        mapped_files = defaultdict(lambda:dict(virtual_committed=0,resident=0,nonshareable_resident=0))
        for index,(_,size,kind,path) in enumerate(regions):
            values = dict(virtual_committed=size,resident=int(counts[index])*4096,
                          nonshareable_resident=int(private_counts[index])*4096)
            label = {0x20000:'private',0x40000:'mapped',0x1000000:'image'}.get(kind,'other')
            for key,value in values.items():
                totals[label][key] += value
                if kind in (0x40000,0x1000000):
                    mapped_files[path or '<unnamed>'][key] += value
        return dict(pid=pid,at_ms=round(time.time()*1000),seconds=time.perf_counter()-started,
            page_bytes=4096,queried_resident=count*4096,unclassified_resident=int((~valid).sum())*4096,
            regions=len(regions),totals=dict(totals),mapped_files=dict(mapped_files),
            process=psutil.Process(pid).memory_info()._asdict(),system=system_memory())
    finally:
        kernel.CloseHandle(handle)


def self_check():
    assert os.name == 'nt' and ctypes.sizeof(ctypes.c_void_p)==8
    assert ctypes.sizeof(MemoryRegion)==48
    snapshot = memory_snapshot(os.getpid())
    assert snapshot['queried_resident'] > 0
    assert sum(item['resident'] for item in snapshot['totals'].values())+snapshot['unclassified_resident']==snapshot['queried_resident']
    assert snapshot['system']['physical_total'] > snapshot['system']['physical_available']
    print('RESOURCE_PROBE_SELF_CHECK_PASS',snapshot['regions'],round(snapshot['seconds'],3),flush=True)


class CounterValue(ctypes.Union):
    _fields_ = [('integer',ctypes.c_longlong),('real',ctypes.c_double)]


class FormattedCounter(ctypes.Structure):
    _fields_ = [('status',wintypes.DWORD),('value',CounterValue)]


class CounterItem(ctypes.Structure):
    _fields_ = [('name',wintypes.LPWSTR),('formatted',FormattedCounter)]


class GPUCounters:
    def __init__(self):
        self.pdh = ctypes.WinDLL('pdh')
        self.pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR,ctypes.c_size_t,ctypes.POINTER(wintypes.HANDLE)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE,wintypes.LPCWSTR,ctypes.c_size_t,ctypes.POINTER(wintypes.HANDLE)]
        self.pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        self.pdh.PdhGetFormattedCounterArrayW.argtypes = [wintypes.HANDLE,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p]
        self.pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        self.query = wintypes.HANDLE()
        self.counters = {}
        self.check(self.pdh.PdhOpenQueryW(None,0,ctypes.byref(self.query)))
        try:
            for kind,path in (
                ('adapter_dedicated',r'\GPU Adapter Memory(*)\Dedicated Usage'),
                ('process_dedicated',r'\GPU Process Memory(*)\Dedicated Usage'),
                ('process_shared',r'\GPU Process Memory(*)\Shared Usage'),
                ('engine_utilization',r'\GPU Engine(*)\Utilization Percentage')):
                handle = wintypes.HANDLE()
                self.check(self.pdh.PdhAddEnglishCounterW(self.query,path,0,ctypes.byref(handle)))
                self.counters[kind] = handle
            self.check(self.pdh.PdhCollectQueryData(self.query))
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise OSError('GPU counter error '+hex(status & 0xFFFFFFFF))

    def read(self):
        self.check(self.pdh.PdhCollectQueryData(self.query))
        result = {}
        for kind,handle in self.counters.items():
            format_code = 0x200 if kind=='engine_utilization' else 0x400
            size,count = wintypes.DWORD(),wintypes.DWORD()
            status = self.pdh.PdhGetFormattedCounterArrayW(handle,format_code,ctypes.byref(size),ctypes.byref(count),None)
            if kind=='engine_utilization' and (status & 0xFFFFFFFF) in (0xC0000BBA,0x800007D5):
                result[kind] = {}
                result['engine_counter_status'] = hex(status & 0xFFFFFFFF)
                continue
            if status & 0xFFFFFFFF != 0x800007D2:
                self.check(status)
            buffer = ctypes.create_string_buffer(size.value)
            status = self.pdh.PdhGetFormattedCounterArrayW(handle,format_code,ctypes.byref(size),ctypes.byref(count),buffer)
            if kind=='engine_utilization' and (status & 0xFFFFFFFF) in (0xC0000BBA,0x800007D5):
                result[kind] = {}
                result['engine_counter_status'] = hex(status & 0xFFFFFFFF)
                continue
            self.check(status)
            items = ctypes.cast(buffer,ctypes.POINTER(CounterItem))
            result[kind] = {items[index].name:(float(items[index].formatted.value.real) if kind=='engine_utilization' else int(items[index].formatted.value.integer))
                            for index in range(count.value) if items[index].formatted.status in (0,1)}
        return result

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


def gpu_engine_usage(counters,pid):
    total,server = {},{}
    for instance,value in counters.items():
        match = re.match(r'^pid_(\d+)_(luid_0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+_phys_\d+)_(eng_\d+_engtype_.+)$',instance)
        if not match:
            continue
        process_id,adapter,engine = match.groups()
        engines = total.setdefault(adapter,{})
        engines[engine] = engines.get(engine,0)+value
        if int(process_id)==pid:
            engines = server.setdefault(adapter,{})
            engines[engine] = engines.get(engine,0)+value
    return {adapter:dict(total_engines=engines,server_engines=server.get(adapter,{}),
                         total_busiest=min(100,max(engines.values(),default=0)),
                         server_busiest=min(100,max(server.get(adapter,{}).values(),default=0)))
            for adapter,engines in total.items()}


def usage_statistics(intervals,threshold=80):
    if not intervals:
        return None
    duration = sum(end-start for start,end,value in intervals)
    if duration<=0:
        return None
    average = sum(value*(end-start) for start,end,value in intervals)/duration
    ordered = sorted((value,end-start) for start,end,value in intervals)
    cumulative = 0
    percentile = ordered[-1][0]
    for value,span in ordered:
        cumulative += span
        if cumulative >= duration*0.95:
            percentile = value
            break
    episodes = 0
    previous_high = False
    previous_end = None
    high_seconds = 0
    starts = []
    for start,end,value in intervals:
        high = value >= threshold
        if high:
            high_seconds += end-start
            if not previous_high or previous_end is None or start-previous_end>1.5:
                episodes += 1
                starts.append(start)
        previous_high,previous_end = high,end
    periods = [later-earlier for earlier,later in zip(starts,starts[1:])]
    return dict(observed_seconds=duration,mean=average,p95=percentile,maximum=max(value for _,_,value in intervals),
                threshold=threshold,high_seconds=high_seconds,high_fraction=high_seconds/duration,
                high_episodes=episodes,episodes_per_minute=episodes*60/duration,
                mean_episode_spacing_seconds=sum(periods)/len(periods) if periods else None,
                percent_seconds=average*duration)


def sampled_intervals(rows,start,end,value_for,*,rate=False,max_gap=1.5):
    intervals = []
    for previous,current in zip(rows,rows[1:]):
        left,right = previous['at_ms']/1000,current['at_ms']/1000
        if right<=left or right-left>max_gap:
            continue
        clipped_left,clipped_right = max(start,left),min(end,right)
        if clipped_right<=clipped_left:
            continue
        value = value_for(current if rate else previous)
        if value is not None:
            intervals.append((clipped_left,clipped_right,value))
    return intervals


def cpu_intervals(rows,start,end,logical_processors,field='cpu'):
    intervals = []
    first = None
    for row in rows:
        moment = row['at_ms']/1000
        if not start<=moment<=end or field not in row:
            first = None
            continue
        if first is None or row.get('pid')!=first.get('pid'):
            first = row
            continue
        elapsed = row['monotonic']-first['monotonic']
        if elapsed<1:
            continue
        delta = sum(row[field][key]-first[field][key] for key in ('user','system'))
        if delta>=0 and elapsed<=1.5:
            intervals.append((first['at_ms']/1000,moment,delta/elapsed/logical_processors*100))
        first = row
    return intervals


def summarize_efficiency(directory):
    report = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    variants = []
    for variant in report['variants']:
        name = variant['candidate']['name']
        gpu = json.loads((directory/(name+'-vram.json')).read_text(encoding='utf-8'))
        resource_path = directory/(name+'-resources.json')
        if not resource_path.is_file():
            variants.append(dict(name=name,error=variant.get('error'),guard_reason=gpu['reason']))
            continue
        resources = json.loads(resource_path.read_text(encoding='utf-8'))
        logical = resources['logical_processors']
        pid = variant.get('pid')
        windows = [dict(kind=sample['kind'],index=index,start=sample['started_ms']/1000,
                        end=sample['finished_ms']/1000,sample=sample)
                   for index,sample in enumerate(variant['samples'])]
        phase_rows = defaultdict(list)
        for row in resources['rows']:
            if row['phase'] in ('loading','idle_loaded','idle_final'):
                phase_rows[row['phase']].append(row)
        windows.extend(dict(kind=phase,start=rows[0]['at_ms']/1000,end=rows[-1]['at_ms']/1000)
                       for phase,rows in phase_rows.items() if len(rows)>1)
        summaries = []
        for window in windows:
            start,end = window['start'],window['end']
            duration = end-start
            def stats(intervals,memory=False):
                result = usage_statistics(intervals)
                if result is None:
                    return None
                if memory:
                    result = {key:result[key] for key in ('observed_seconds','mean','p95','maximum')}
                result['coverage'] = result['observed_seconds']/duration
                return result
            summary = {key:value for key,value in window.items() if key!='sample'}
            summary['duration_seconds'] = duration
            sample = window.get('sample')
            if sample:
                summary.update(output_tokens=sample['usage'].get('completion_tokens'),
                    decode_tps=sample['timings'].get('predicted_per_second'),
                    answer_sha256=sample.get('answer_sha256'),breakdown=sample.get('breakdown'))
            summary['gpu'] = {}
            for adapter in gpu['limits_mib']:
                values = {}
                for scope in ('server','total'):
                    def usage_value(row):
                        value = row.get('gpu_usage',{}).get(adapter)
                        if not value or not value.get(scope+'_engines'):
                            return None
                        if scope=='server' and row.get('pid')!=pid:
                            return None
                        return value[scope+'_busiest']
                    values[scope+'_usage'] = stats(sampled_intervals(gpu['rows'],start,end,usage_value,rate=True))
                for source,key in (('adapter_dedicated','adapter_dedicated_mib'),
                                   ('process_dedicated','server_dedicated_mib'),
                                   ('process_shared','server_shared_mib')):
                    def memory_value(row):
                        instance = adapter if source=='adapter_dedicated' else f'pid_{pid}_{adapter}'
                        value = row[source].get(instance)
                        return value/1024**2 if value is not None else None
                    values[key] = stats(sampled_intervals(gpu['rows'],start,end,memory_value),memory=True)
                summary['gpu'][adapter] = values
            summary['cpu'] = stats(cpu_intervals(resources['rows'],start,end,logical))
            if summary['cpu']:
                summary['cpu']['cpu_seconds_observed'] = summary['cpu']['percent_seconds']*logical/100
            summary['observer_cpu'] = stats(cpu_intervals(resources['rows'],start,end,logical,'observer_cpu'))
            summary['memory'] = {}
            for source,key,label in (('memory','rss','rss_gib'),('memory','private','private_commit_gib'),
                                     ('system','commit_total','system_commit_gib'),
                                     ('system','physical_available','system_available_gib')):
                def memory_value(row):
                    value = row.get(source,{}).get(key)
                    return value/1024**3 if value is not None else None
                summary['memory'][label] = stats(sampled_intervals(resources['rows'],start,end,memory_value),memory=True)
            summaries.append(summary)
        variants.append(dict(name=name,pid=pid,error=variant.get('error'),guard_reason=gpu['reason'],
            logical_processors=logical,requests_and_phases=summaries,
            missing_engine_rows=sum(not row.get('engine_utilization') for row in gpu['rows']),
            gpu_rows=len(gpu['rows']),source_files=[name+'-resources.json',name+'-vram.json']))
    return dict(source='results.json',method=dict(gpu_interval_seconds=0.5,cpu_min_interval_seconds=1,
        high_threshold_percent=80,gpu_usage='busiest physical engine; server PID and whole adapter separated',
        rate_alignment='value at right endpoint describes preceding interval',
        memory_alignment='hold previous gauge; clips at request boundaries',
        missing='excluded, never zero-filled; coverage reported',
        cpu='process user+kernel time, normalized to all logical processors; observed integral only',
        peak='sampled maximum, not instantaneous; high episodes are consecutive >=80% intervals',
        phases='requests exclude snapshots, loading and idle; no per-kernel energy measurement'),variants=variants)


def gpu_limit_reason(sample,pid,limits,*,reserve_mib=128,shared_limit_mib=384):
    for adapter,capacity in limits.items():
        if adapter not in sample['adapter_dedicated']:
            return 'Required GPU counter disappeared: '+adapter
        used = sample['adapter_dedicated'][adapter]/1024**2
        if used > capacity-reserve_mib:
            return f'Adapter VRAM safety margin: {adapter} {used:.1f}/{capacity} MiB'
    if pid is not None:
        for name,value in sample['process_shared'].items():
            if name.startswith(f'pid_{pid}_') and value/1024**2 > shared_limit_mib:
                return f'Process shared GPU memory limit: {name} {value/1024**2:.1f} MiB'
    return ''


class GPUWatchdog:
    def __init__(self,server,stopped,directory,name):
        self.server,self.stopped = server,stopped
        self.path = directory/(name+'-vram.json')
        self.rows = []
        self.reason = ''
        self.finished = threading.Event()
        self.counters = GPUCounters()
        self.thread = None
        self.engine_missing_since = None
        self.limits = {'luid_0x00000000_0x0000DF31_phys_0':8151,
                       'luid_0x00000000_0x0000C59F_phys_0':8192}

    def sample(self):
        row = self.counters.read()
        process = self.server.process
        row.update(at_ms=round(time.time()*1000),pid=process.pid if process else None)
        row['gpu_usage'] = gpu_engine_usage(row['engine_utilization'],row['pid'])
        self.rows.append(row)
        return row

    def preflight(self,candidate):
        row = self.sample()
        reason = gpu_limit_reason(row,None,self.limits)
        for adapter,key,overhead in (('luid_0x00000000_0x0000DF31_phys_0','primary_weight_mib',2000),
                                     ('luid_0x00000000_0x0000C59F_phys_0','secondary_weight_mib',1600)):
            estimated = row['adapter_dedicated'].get(adapter,0)/1024**2+candidate[key]+overhead
            if estimated > self.limits[adapter]-128:
                reason = f'Preflight VRAM estimate: {key} plus conservative buffers/background = {estimated:.1f} MiB'
        if reason:
            self.reason = reason
            raise ValueError(reason)

    def start(self):
        def monitor():
            while not self.finished.wait(0.5):
                try:
                    row = self.sample()
                    reason = gpu_limit_reason(row,row['pid'],self.limits)
                    if not row['engine_utilization'] and row['pid'] is not None:
                        if self.engine_missing_since is None:
                            self.engine_missing_since = time.monotonic()
                        elif time.monotonic()-self.engine_missing_since>5:
                            reason = 'GPU utilization telemetry missing for more than 5 seconds'
                    else:
                        self.engine_missing_since = None
                except Exception as error:
                    reason = 'GPU telemetry failed: '+str(error)
                if reason:
                    self.reason = reason
                    self.stopped.set()
                    process = self.server.process
                    if process is not None and process.poll() is None:
                        try:
                            process.terminate()
                        except OSError:
                            pass
                    return
        self.thread = threading.Thread(target=monitor,daemon=True)
        self.thread.start()

    def close(self):
        self.finished.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.counters.close()
        self.path.write_text(json.dumps(dict(limits_mib=self.limits,reserve_mib=128,shared_limit_mib=384,
            nominal_interval_seconds=0.5,reason=self.reason,rows=self.rows),indent=2),encoding='utf-8')


class ResourceProbe:
    def __init__(self,directory,name,sample_instructions=False):
        self.directory = directory
        self.name = name
        self.phase = 'before_load'
        self.rows = []
        self.snapshots = []
        self.stopped = threading.Event()
        self.condition = threading.Condition()
        self.server = None
        self.thread = None
        self.sample_instructions = sample_instructions

    def mark(self,phase):
        self.phase = phase

    def start(self,server):
        self.server = server
        self.thread = threading.Thread(target=self.collect,daemon=True)
        self.thread.start()

    def collect(self):
        observer = psutil.Process()
        previous_threads = {}
        while not self.stopped.is_set():
            row = dict(at_ms=round(time.time()*1000),monotonic=time.perf_counter(),phase=self.phase,
                       system=system_memory(),observer_cpu=observer.cpu_times()._asdict())
            process = self.server.process
            if process is not None:
                try:
                    target = psutil.Process(process.pid)
                    with target.oneshot():
                        row.update(pid=target.pid,cpu=target.cpu_times()._asdict(),
                            memory=target.memory_info()._asdict(),io=target.io_counters()._asdict(),
                            threads=[item._asdict() for item in target.threads()])
                    current = {item['id']:item['user_time']+item['system_time'] for item in row['threads']}
                    deltas = {identifier:seconds-previous_threads.get(identifier,seconds) for identifier,seconds in current.items()}
                    previous_threads = current
                    if self.sample_instructions and self.phase.startswith('request_') and deltas:
                        hottest = max(deltas,key=deltas.get)
                        if deltas[hottest] > 0:
                            try:
                                row['instruction'] = instruction_location(target.pid,hottest)
                                row['instruction']['cpu_delta_seconds'] = deltas[hottest]
                            except OSError as error:
                                row['instruction_error'] = str(error)
                except (psutil.NoSuchProcess,psutil.AccessDenied) as error:
                    row['error'] = type(error).__name__
            with self.condition:
                self.rows.append(row)
                self.condition.notify_all()
            self.stopped.wait(0.2)

    def idle(self,phase):
        self.mark(phase)
        with self.condition:
            target_count = len(self.rows)+11
            if not self.condition.wait_for(lambda:len(self.rows)>=target_count,timeout=5):
                raise RuntimeError('Resource sampler stopped producing observations')

    def snapshot(self,label):
        self.mark('snapshot_'+label)
        snapshot = memory_snapshot(self.server.process.pid)
        snapshot['label'] = label
        self.snapshots.append(snapshot)
        print('MEMORY '+json.dumps(dict(name=self.name,label=label,
            resident_gib=round(snapshot['queried_resident']/1024**3,3),
            categories_mib={key:round(value['resident']/1024**2,1) for key,value in snapshot['totals'].items()},
            seconds=round(snapshot['seconds'],3))),flush=True)

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.thread and self.thread.is_alive():
            raise RuntimeError('Resource sampler did not stop')
        report = dict(logical_processors=psutil.cpu_count(),interval_seconds=0.2,
                  instruction_sampling=self.sample_instructions,
                      rows=self.rows,snapshots=self.snapshots)
        (self.directory/(self.name+'-resources.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Synthetic dual-GPU CPU/RAM attribution research')
    parser.add_argument('--summarize',type=Path)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--variants',default='baseline')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--context',type=int,default=8192)
    parser.add_argument('--padding-repeats',type=int,default=0)
    parser.add_argument('--sample-instructions',action='store_true')
    arguments = parser.parse_args()
    if arguments.summarize is not None:
        destination = arguments.summarize/'efficiency-summary.json'
        if destination.exists():
            parser.error('Summary already exists; preserve prior results')
        summary = summarize_efficiency(arguments.summarize)
        destination.write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('EFFICIENCY_SUMMARY '+str(destination.resolve()),flush=True)
        return
    if arguments.self_check:
        self_check()
        return
    if arguments.output is None or arguments.output.exists():
        parser.error('Provide a new output directory')
    if not 1 <= arguments.runs <= 8 or not 0 <= arguments.padding_repeats <= 300:
        parser.error('runs must be 1..8; padding-repeats must be 0..300')
    if any((process.info['name'] or '').lower()=='llama-server.exe' for process in psutil.process_iter(['name'])):
        parser.error('Another model server is running')
    from desktop_agent.agent import Settings
    from desktop_agent.benchmark_placement import placement_candidates,secondary_candidates,run_candidate
    settings = replace(Settings(),context_tokens=arguments.context)
    _,weights = placement_candidates(settings.model)
    combined = next(item for item in secondary_candidates(weights,'CUDA0','Vulkan0') if item['name']=='secondary_ffn_output')
    candidates = {
        'baseline':dict(combined,name='baseline'),
        'no_poll':dict(combined,name='no_poll',poll=0,poll_batch=0),
        'no_checkpoints':dict(combined,name='no_checkpoints',checkpoints=0),
        'no_mmap':dict(combined,name='no_mmap',load_mode='none'),
        'baseline_end':dict(combined,name='baseline_end')}
    names = arguments.variants.split(',')
    if len(set(names)) != len(names) or any(name not in candidates for name in names):
        parser.error('Choose unique variants from '+','.join(candidates))
    backend = Path('desktop_agent/data/runtimes/llama-b11000-vulkan/ggml-vulkan.dll').resolve()
    if not backend.is_file():
        parser.error('Research Vulkan backend is missing')
    arguments.output.mkdir(parents=True)
    report = dict(created=datetime.now().isoformat(),synthetic_only=True,context=settings.context_tokens,
                  padding_repeats=arguments.padding_repeats,runs=arguments.runs,backend=str(backend),variants=[])
    previous_backend = os.environ.get('GGML_BACKEND_PATH')
    os.environ['GGML_BACKEND_PATH'] = str(backend)
    try:
        for name in names:
            probe = ResourceProbe(arguments.output,name,arguments.sample_instructions)
            result = run_candidate(candidates[name],settings,arguments.output,arguments.runs,
                                   arguments.padding_repeats,resource_probe=probe)
            report['variants'].append(result)
            (arguments.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        if previous_backend is None:
            os.environ.pop('GGML_BACKEND_PATH',None)
        else:
            os.environ['GGML_BACKEND_PATH'] = previous_backend
    print('RESOURCE_RESEARCH_COMPLETE '+str(arguments.output.resolve()),flush=True)


if __name__ == '__main__':
    main()