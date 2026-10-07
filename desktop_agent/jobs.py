from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import uuid

from game_agent.core import Halted
from desktop_agent.protocol import validate_action
from desktop_agent.tools import Tools, ToolResult


class StopSignal:
    def __init__(self, parent, local):
        self.parent, self.local = parent, local

    def is_set(self):
        return self.parent.is_set() or self.local.is_set()

    def wait(self, seconds):
        deadline = time.monotonic()+seconds
        while not self.is_set():
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                return False
            self.local.wait(min(remaining,0.05))
        return True


class ToolRunner(Tools):
    def __init__(self, *args, approve_cancellable=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.approve_cancellable = approve_cancellable
        self.pool = ThreadPoolExecutor(max_workers=4,thread_name_prefix='desk-tools')
        self.browser_lane = ThreadPoolExecutor(max_workers=1,thread_name_prefix='desk-browser')
        self.input_lane = ThreadPoolExecutor(max_workers=1,thread_name_prefix='desk-input')
        self.browser_tools = None
        self.background = {}
        from desktop_agent.terminals import Terminals
        self.terminals = Terminals(self.directory,self.stopped,self.notify)

    def snapshot(self, stop):
        worker = Tools(self.directory,stop,self.approve,self.notify,headless=self.headless)
        if self.approve_cancellable is not None:
            worker.approve = lambda action,reason,context:self.approve_cancellable(action,reason,context,stop)
        for field in ('allow_screen','allow_input','allow_browser','mode','request_intent','capture_margin','workspace_root'):
            setattr(worker,field,getattr(self,field))
        worker.tool_policies=dict(self.tool_policies)
        worker.terminals=self.terminals
        worker.call_id=self.call_id
        worker.window = dict(self.window) if self.window else None
        worker.coordinate_frame = dict(self.coordinate_frame) if self.coordinate_frame else None
        return worker

    def browser_execute(self, action, snapshot):
        if self.browser_tools is None:
            self.browser_tools = snapshot
        worker = self.browser_tools
        for field in ('allow_browser','mode','stopped','approve','notify','request_intent','tool_policies','call_id'):
            setattr(worker,field,getattr(snapshot,field))
        return worker.execute(action)

    def recording_connection(self, snapshot):
        if not snapshot.allow_browser:
            raise ValueError('Browser access disabled')
        if self.browser_tools is None:
            self.browser_tools = snapshot
        worker = self.browser_tools
        worker.stopped = snapshot.stopped
        worker.ensure_browser()
        session = worker.context.new_cdp_session(worker.page)
        try:
            target = session.send('Target.getTargetInfo')['targetInfo']['targetId']
        finally:
            session.detach()
        port = int((self.directory/'browser-profile'/'DevToolsActivePort').read_text().splitlines()[0])
        return f'http://127.0.0.1:{port}',target

    def record_browser(self, action, snapshot, connection):
        from playwright.sync_api import sync_playwright
        endpoint,target = connection
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(endpoint,timeout=10000)
            page = None
            for context in browser.contexts:
                for candidate in context.pages:
                    session = context.new_cdp_session(candidate)
                    try:
                        if session.send('Target.getTargetInfo')['targetInfo']['targetId'] == target:
                            page = candidate
                    finally:
                        session.detach()
            if page is None:
                raise ValueError('Recording tab closed')
            snapshot.page, snapshot.context = page,page.context
            snapshot.ensure_browser = snapshot.check
            return snapshot.execute(action)

    def submit(self, action, ready=None, call_id=None):
        self.check_permission(action)
        local_stop = threading.Event()
        snapshot = self.snapshot(StopSignal(self.stopped,local_stop))
        if call_id is not None:
            snapshot.call_id = call_id
        snapshot.recording_ready = ready
        name = action['tool']
        if name == 'browser_record':
            connection = self.browser_lane.submit(self.recording_connection,snapshot).result()
            future = self.pool.submit(self.record_browser,action,snapshot,connection)
        elif name.startswith('browser_'):
            future = self.browser_lane.submit(self.browser_execute,action,snapshot)
        elif name.startswith('desktop_') and name not in ('desktop_record','desktop_capture','desktop_screen_capture'):
            future = self.input_lane.submit(snapshot.execute,action)
        else:
            future = self.pool.submit(snapshot.execute,action)
        return future,local_stop

    def run_parallel(self, action):
        self.authorize(action)
        pending,results = [],[]
        try:
            for index,child in enumerate(action['arguments']['actions']):
                child = dict(child)
                if action['risk']=='sensitive':
                    child['risk']='sensitive'
                identity = f'{self.call_id or uuid.uuid4().hex}:{index+1}'
                pending.append((child,identity,*self.submit(child,call_id=identity)))
            for child,identity,future,stop in pending:
                try:
                    result = future.result()
                except Exception as failure:
                    result = ToolResult(str(failure),error=str(failure),interrupted=str(failure) if isinstance(failure,Halted) else '')
                results.append(dict(tool=child['tool'],call_id=identity,text=result.text,error=result.error,interrupted=result.interrupted))
                if result.interrupted:
                    for _,_,_,signal in pending:
                        signal.set()
        finally:
            for _,_,future,signal in pending:
                signal.set()
            for _,_,future,_ in pending:
                try:
                    future.result()
                except Exception:
                    pass
        error = next((result['error'] for result in results if result['error']),'')
        interrupted = next((result['interrupted'] for result in results if result['interrupted']),'')
        return ToolResult(json.dumps({'results':results,'call_id':self.call_id},ensure_ascii=False),
                          error=error,interrupted=interrupted,call_id=self.call_id)

    def execute(self, action):
        validate_action(action)
        self.check()
        self.check_permission(action)
        name, arguments = action['tool'],action['arguments']
        if name == 'tool_parallel':
            return self.run_parallel(action)
        if name.startswith('job_'):
            self.authorize(action)
        if name == 'job_start':
            if len(self.background) >= 4:
                raise ValueError('Collect job_result before starting more than four jobs')
            nested = dict(arguments['action'])
            if action['risk'] == 'sensitive':
                nested['risk'] = 'sensitive'
            identifier = uuid.uuid4().hex
            ready = threading.Event()
            future, stop = self.submit(nested,ready)
            self.background[identifier] = dict(action=nested,future=future,stop=stop,ready=ready,call_id=self.call_id)
            if nested['tool'] in ('desktop_record','browser_record'):
                deadline = time.monotonic()+10
                while not ready.is_set() and not future.done() and time.monotonic() < deadline:
                    if self.stopped.wait(0.05):
                        break
            return ToolResult(json.dumps({'job_id':identifier,'tool':nested['tool'],'ready':ready.is_set(),
                'status':'completed' if future.done() else 'running','next':'Use job_result to collect; other tools may run now.'}))
        if name == 'job_status':
            return ToolResult(json.dumps([{'job_id':identifier,'tool':job['action']['tool'],
                'ready':job['ready'].is_set(),'done':job['future'].done()} for identifier,job in self.background.items()]))
        if name in ('job_result','job_cancel'):
            identifier = arguments['job_id']
            if identifier not in self.background:
                raise ValueError('Unknown or already collected job')
            if name == 'job_cancel':
                self.background[identifier]['stop'].set()
                return ToolResult(json.dumps({'job_id':identifier,'status':'cancel_requested','next':'Call job_result'}))
            return self.collect(identifier)
        if name in ('window_list','window_select'):
            return super().execute(action)
        future, stop = self.submit(action)
        return future.result()

    def collect(self, identifier):
        job = self.background.pop(identifier)
        try:
            result = job['future'].result()
        except Exception as error:
            result = ToolResult(str(error),interrupted=str(error) if isinstance(error,Halted) else '',error=str(error))
        result.tool, result.job_id = job['action']['tool'],identifier
        result.call_id = job.get('call_id','')
        return result

    def drain(self, cancel=False):
        if cancel:
            for job in self.background.values():
                job['stop'].set()
        return [self.collect(identifier) for identifier in list(self.background)]

    def job_summary(self):
        return [{'job_id':identifier,'tool':job['action']['tool'],'ready':job['ready'].is_set(),
                 'done':job['future'].done(),'call_id':job.get('call_id','')} for identifier,job in self.background.items()]

    def collect_ready(self):
        return [self.collect(identifier) for identifier,job in list(self.background.items()) if job['future'].done()]

    def close(self):
        self.terminals.close()
        self.drain(cancel=True)
        if self.browser_tools is not None:
            self.browser_lane.submit(self.browser_tools.close).result()
        for lane in (self.pool,self.input_lane,self.browser_lane):
            lane.shutdown(wait=True)