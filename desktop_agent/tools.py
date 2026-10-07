import ctypes
from ctypes import wintypes
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote, urlsplit
import uuid

from PIL import Image, ImageDraw
from game_agent import windows
from game_agent.core import Halted, INPUT_TAG, Region
from desktop_agent.protocol import WORKSPACE_TOOLS, approval_reason, target_approval_reason, tool_policy, validate_action, web_url


windows.user32.IsIconic.argtypes = [wintypes.HWND]
windows.user32.ShowWindow.argtypes = [wintypes.HWND,ctypes.c_int]
windows.user32.BringWindowToTop.argtypes = [wintypes.HWND]
windows.user32.AttachThreadInput.argtypes = [wintypes.DWORD,wintypes.DWORD,wintypes.BOOL]
windows.kernel32.GetCurrentThreadId.restype = wintypes.DWORD


_dpi_user32 = ctypes.WinDLL('user32',use_last_error=True)
_dpi_user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
_dpi_user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
_dwm = ctypes.WinDLL('dwmapi',use_last_error=True)
_dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD]
_dwm.DwmGetWindowAttribute.restype = ctypes.c_long
_capture_user32 = ctypes.WinDLL('user32',use_last_error=True)
_capture_gdi32 = ctypes.WinDLL('gdi32',use_last_error=True)
_capture_user32.GetWindowDC.argtypes = [wintypes.HWND]
_capture_user32.GetWindowDC.restype = wintypes.HDC
_capture_user32.ReleaseDC.argtypes = [wintypes.HWND,wintypes.HDC]
_capture_user32.PrintWindow.argtypes = [wintypes.HWND,wintypes.HDC,wintypes.UINT]
_capture_user32.PrintWindow.restype = wintypes.BOOL
_capture_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_capture_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_capture_gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC,ctypes.c_int,ctypes.c_int]
_capture_gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
_capture_gdi32.SelectObject.argtypes = [wintypes.HDC,wintypes.HANDLE]
_capture_gdi32.SelectObject.restype = wintypes.HANDLE
_capture_gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
_capture_gdi32.DeleteDC.argtypes = [wintypes.HDC]
_capture_gdi32.GetDIBits.argtypes = [wintypes.HDC,wintypes.HBITMAP,wintypes.UINT,wintypes.UINT,
                                   ctypes.c_void_p,ctypes.c_void_p,wintypes.UINT]


@contextmanager
def physical_pixels():
    previous = _dpi_user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
    if not previous:
        raise ValueError('Cannot establish physical-pixel coordinates for desktop capture/input')
    try:
        yield
    finally:
        _dpi_user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))


def list_windows(own_pid):
    found = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(handle, parameter):
        process = wintypes.DWORD()
        windows.user32.GetWindowThreadProcessId(handle, ctypes.byref(process))
        title = windows.window_title(handle)
        if title and windows.user32.IsWindowVisible(handle) and process.value != own_pid:
            found.append({'handle': int(handle), 'pid': process.value, 'title': title})
        return True
    windows.user32.EnumWindows(callback_type(visit), 0)
    return found


@dataclass
class ToolResult:
    text: str
    image: object = None
    video: str = ''
    interrupted: str = ''
    tool: str = ''
    job_id: str = ''
    error: str = ''
    call_id: str = ''


@physical_pixels()
def window_frame_rect(handle):
    rectangle = wintypes.RECT()
    status = _dwm.DwmGetWindowAttribute(handle,9,ctypes.byref(rectangle),ctypes.sizeof(rectangle))
    if status == 0 and rectangle.right > rectangle.left and rectangle.bottom > rectangle.top:
        return rectangle.left,rectangle.top,rectangle.right,rectangle.bottom
    return windows.window_rect(handle)


@physical_pixels()
def render_window_image(handle, rectangle):
    left,top,right,bottom = rectangle
    width,height = right-left,bottom-top
    if width <= 0 or height <= 0 or width*height > 64000000:
        raise ValueError('Unsupported window capture dimensions')
    source = _capture_user32.GetWindowDC(handle)
    memory,bitmap,previous = None,None,None
    try:
        if not source:
            raise OSError('Cannot acquire window DC')
        memory = _capture_gdi32.CreateCompatibleDC(source)
        bitmap = _capture_gdi32.CreateCompatibleBitmap(source,width,height)
        if not memory or not bitmap:
            raise OSError('Cannot allocate window capture')
        previous = _capture_gdi32.SelectObject(memory,bitmap)
        if not previous or previous == ctypes.c_void_p(-1).value:
            previous = None
            raise OSError('Cannot select capture bitmap')
        if not _capture_user32.PrintWindow(handle,memory,2):
            raise OSError('Window refused capture')
        _capture_gdi32.SelectObject(memory,previous)
        previous = None
        header = windows.BitmapHeader()
        header.biSize = ctypes.sizeof(header)
        header.biWidth,header.biHeight = width,-height
        header.biPlanes,header.biBitCount = 1,32
        buffer = ctypes.create_string_buffer(width*height*4)
        if _capture_gdi32.GetDIBits(memory,bitmap,0,height,buffer,ctypes.byref(header),0) != height:
            raise OSError('Cannot read captured window pixels')
        return Image.frombytes('RGB',(width,height),buffer.raw,'raw','BGRX')
    finally:
        if previous and memory:
            _capture_gdi32.SelectObject(memory,previous)
        if bitmap:
            _capture_gdi32.DeleteObject(bitmap)
        if memory:
            _capture_gdi32.DeleteDC(memory)
        if source:
            _capture_user32.ReleaseDC(handle,source)


def capture_window_image(handle, rectangle, stopped, *, timeout=3.0):
    left,top,right,bottom = rectangle
    width,height = right-left,bottom-top
    if width <= 0 or height <= 0 or width*height > 64000000:
        raise ValueError('Unsupported window capture dimensions')
    if stopped.is_set():
        raise Halted('Stopped by user')
    deadline = time.monotonic()+timeout
    command = [sys.executable,'-m','desktop_agent.tools','--capture-window',str(handle),
               *map(str,rectangle)]
    with subprocess.Popen(command,cwd=Path(__file__).resolve().parent.parent,
                          stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                          creationflags=subprocess.CREATE_NO_WINDOW) as process:
        try:
            while True:
                if stopped.is_set():
                    raise Halted('Stopped by user')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise ValueError('Selected window capture timed out; use desktop_screen_capture for the visible desktop')
                try:
                    pixels,_ = process.communicate(timeout=min(0.05,remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if stopped.is_set():
                raise Halted('Stopped by user')
            if process.returncode or len(pixels) != width*height*3:
                raise OSError('Cannot render selected window pixels')
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    return Image.frombytes('RGB',(width,height),pixels)


@physical_pixels()
def visible_window_region(handle):
    screen = Region(*windows.virtual_screen())
    left,top,right,bottom = window_frame_rect(handle)
    screen_left,screen_top,screen_right,screen_bottom = screen.bbox
    left,top = max(left,screen_left),max(top,screen_top)
    right,bottom = min(right,screen_right),min(bottom,screen_bottom)
    if right <= left or bottom <= top:
        raise ValueError('Selected window is outside the visible desktop; use desktop_screen_capture')
    return Region(left,top,right-left,bottom-top)


class InputTarget:
    @physical_pixels()
    def __init__(self, window, region):
        self.handle,self.pid = window['handle'],window['pid']
        self.region = region
        self.rect = windows.window_rect(self.handle)
        self.title = windows.window_title(self.handle)
        self.require_geometry = True
        self.failure_reason = ''

    @physical_pixels()
    def valid(self):
        self.failure_reason = ''
        process = wintypes.DWORD()
        windows.user32.GetWindowThreadProcessId(self.handle,ctypes.byref(process))
        if not windows.user32.IsWindow(self.handle) or process.value != self.pid:
            self.failure_reason = 'Selected window closed or changed process'
        elif windows.user32.IsIconic(self.handle):
            self.failure_reason = 'Selected window was minimized'
        elif windows.user32.GetForegroundWindow() != self.handle:
            self.failure_reason = 'Selected window lost focus; a different window is active'
        elif self.require_geometry and (windows.window_rect(self.handle) != self.rect or visible_window_region(self.handle).bbox != self.region.bbox):
            self.failure_reason = 'Selected window moved or resized; mouse coordinates must be refreshed'
        return not self.failure_reason


class Tools:
    def __init__(self, directory, stopped, approve, notify, *, headless=False):
        self.directory = Path(directory)
        self.stopped, self.approve, self.notify = stopped, approve, notify
        self.headless = headless
        self.mode = 'routine'
        self.request_intent = ''
        self.allow_screen = True
        self.allow_input = True
        self.allow_browser = True
        self.workspace_root = ''
        self.tool_policies = {}
        self.terminals = None
        self.call_id = ''
        self.window = None
        self.coordinate_frame = None
        self.capture_margin = 0
        self.playwright = self.context = self.page = None
        self.recording_ready = None

    def check(self):
        if self.stopped.is_set():
            raise Halted('Stopped by user')

    def check_permission(self, action):
        from desktop_agent.protocol import COMPOSITE_TOOLS
        if tool_policy(self.tool_policies,action['tool']) == 'disabled':
            raise ValueError('Tool disabled by user: '+action['tool'])
        if action['tool'] == 'job_start':
            self.check_permission(action['arguments']['action'])
        if action['tool'] in COMPOSITE_TOOLS:
            for child in action['arguments']['actions']:
                self.check_permission(child)
        name = action['tool']
        if name.startswith('desktop_') and not (self.allow_screen if name in ('desktop_capture','desktop_screen_capture','desktop_record') else self.allow_input):
            raise ValueError('Screen capture disabled' if name in ('desktop_capture','desktop_screen_capture','desktop_record') else 'Desktop input disabled')
        if name.startswith('browser_') and not self.allow_browser:
            raise ValueError('Browser access disabled')

    def authorize(self, action, context=None):
        self.check()
        self.check_permission(action)
        context = dict(context or {},request_intent=self.request_intent,tool_policies=dict(self.tool_policies))
        reason = approval_reason(action,self.mode,context)
        if reason and not self.approve(action,reason,context):
            self.notify('approval_log',{'tool':action['tool'],'mode':'denied','reason':reason})
            raise Halted('User denied the tool; task stopped')
        self.check()
        self.check_permission(action)
        self.notify('approval_log',{'tool':action['tool'],'mode':'confirmed' if reason else 'automatic','reason':reason})

    def run_macro(self, action):
        import uuid
        self.authorize(action)
        arguments = action['arguments']
        parent_call = self.call_id
        self.directory.mkdir(parents=True,exist_ok=True)
        journal = self.directory/('macro-'+uuid.uuid4().hex+'.jsonl')
        completed = completed_actions = 0
        last = None
        error = interrupted = ''
        with journal.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps({'action':action,'call_id':parent_call},ensure_ascii=False)+'\n')
            try:
                for repetition in range(arguments['repeat']):
                    if repetition and self.stopped.wait(arguments['interval_ms']/1000):
                        raise Halted('Macro cancelled between repetitions')
                    for index,child in enumerate(arguments['actions']):
                        self.check()
                        child = dict(child)
                        if action['risk']=='sensitive':
                            child['risk']='sensitive'
                        self.call_id = f'{parent_call or journal.stem}:{repetition+1}:{index+1}'
                        try:
                            result = self.execute(child)
                        except Exception as failure:
                            result = ToolResult(str(failure),error=str(failure),interrupted=str(failure) if isinstance(failure,Halted) else '')
                        last = dict(iteration=repetition+1,step=index+1,tool=child['tool'],call_id=self.call_id,
                                    text=result.text,error=result.error,interrupted=result.interrupted)
                        stream.write(json.dumps(last,ensure_ascii=False)+'\n')
                        stream.flush()
                        if result.error or result.interrupted:
                            error,interrupted = result.error,result.interrupted
                            break
                        completed_actions += 1
                    else:
                        completed += 1
                        self.notify('tool_progress',dict(tool=action['tool'],completed=completed,total=arguments['repeat'],call_id=parent_call))
                        continue
                    break
            except Exception as failure:
                error = str(failure)
                interrupted = str(failure) if isinstance(failure,Halted) else ''
            finally:
                self.call_id = parent_call
            summary = dict(status='stopped' if interrupted else 'error' if error else 'completed',
                           completed_iterations=completed,completed_actions=completed_actions,
                           requested_iterations=arguments['repeat'],steps_per_iteration=len(arguments['actions']),
                           last_action={key:value[:2000] if isinstance(value,str) else value for key,value in last.items()} if last else None,
                           log_path=str(journal.resolve()),call_id=parent_call,error=error,interrupted=interrupted,
                           replay_safe=False)
            stream.write(json.dumps({'summary':summary},ensure_ascii=False)+'\n')
        return ToolResult(json.dumps(summary,ensure_ascii=False),error=error,interrupted=interrupted,call_id=parent_call)

    def execute(self, action):
        validate_action(action)
        self.check()
        self.check_permission(action)
        name, arguments = action['tool'], action['arguments']
        if name in ('desktop_macro','browser_macro'):
            return self.run_macro(action)
        if name.startswith('terminal_') and name in WORKSPACE_TOOLS:
            if self.terminals is None:
                raise ValueError('Terminal session is unavailable')
            if name == 'terminal_start':
                from desktop_agent.workspace_tools import Workspace
                workspace = Workspace(self.workspace_root,self.directory)
                cwd = workspace.path(arguments['cwd'],directory=True)
                self.terminals.validate(arguments['command'],arguments['timeout_seconds'])
                self.authorize(action,{'cwd':str(cwd),'warning':'PowerShell is not sandboxed. It can access files and networks outside this folder. No interactive or secret input.'})
                cwd = workspace.path(arguments['cwd'],directory=True)
                record = self.terminals.start(arguments['command'],cwd,arguments['timeout_seconds'],call_id=self.call_id)
            else:
                self.authorize(action)
                record = self.terminals.output(**arguments) if name == 'terminal_output' else self.terminals.stop(**arguments)
            return ToolResult(json.dumps(record,ensure_ascii=False))
        if name.startswith('workspace_') and name in WORKSPACE_TOOLS:
            from desktop_agent.workspace_tools import Workspace
            workspace = Workspace(self.workspace_root,self.directory,self.check)
            if name == 'workspace_apply_patch':
                prepared = workspace.prepare(**arguments)
                self.authorize(action,{'path':str(workspace.path(arguments['path'])),'diff':prepared['diff']})
                record = workspace.commit(prepared)
                self.notify('file_change',dict({key:value for key,value in record.items() if key!='diff'},call_id=self.call_id))
                record = dict(record,diff=record['diff'][:12000],diff_truncated=len(record['diff'])>12000)
            else:
                self.authorize(action,{'workspace_root':self.workspace_root})
                if name in ('workspace_code_search','workspace_symbols'):
                    from desktop_agent.retrieval import CodeIndex
                    index=CodeIndex(workspace)
                    record=index.search(**arguments) if name=='workspace_code_search' else index.find_symbols(**arguments)
                else:
                    record = workspace.read(**arguments) if name == 'workspace_read' else workspace.search(**arguments)
            return ToolResult(json.dumps(record,ensure_ascii=False))
        context = {'request_intent':self.request_intent}
        context['tool_policies']=dict(self.tool_policies)
        candidate = None
        if name in ('window_list','window_select'):
            if not (self.allow_screen or self.allow_input):
                raise ValueError('Desktop access disabled')
            available = list_windows(os.getpid())
            if name == 'window_list':
                self.authorize(action,context)
                return ToolResult(json.dumps(available,ensure_ascii=False))
            candidate = next((row for row in available if row['handle'] == arguments['handle'] and row['pid'] == arguments['pid']),None)
            if candidate is None:
                raise ValueError('Window no longer available; list windows again')
            context['label'] = candidate['title']
        elif name.startswith('desktop_'):
            if self.window is None and name not in ('desktop_screen_capture','desktop_capture'):
                raise ValueError('No desktop window selected')
            if name in ('desktop_capture','desktop_record','desktop_screen_capture') and not self.allow_screen:
                raise ValueError('Screen capture disabled')
            if name not in ('desktop_capture','desktop_record','desktop_screen_capture') and not self.allow_input:
                raise ValueError('Desktop input disabled')
            context['label'] = 'Currently visible desktop' if name == 'desktop_screen_capture' or self.window is None else windows.window_title(self.window['handle'])
        elif name.startswith('browser_'):
            if not self.allow_browser:
                raise ValueError('Browser access disabled')
            if name == 'browser_open':
                web_url(arguments['url'])
            if name in ('browser_click', 'browser_type', 'browser_key','browser_hold','browser_drag'):
                self.ensure_browser()
                context.update(self.browser_target(name, arguments))
                if context.get('password'):
                    raise ValueError('Use password/file inputs manually; do not send secrets to the model')
        else:
            raise ValueError('Tool is not executable')
        self.authorize(action,context)
        if name in ('browser_click', 'browser_type', 'browser_key','browser_hold','browser_drag') and context != dict(self.browser_target(name, arguments),request_intent=self.request_intent,tool_policies=dict(self.tool_policies)):
            raise ValueError('Browser target changed during approval; request the action again')
        if candidate is not None:
            if candidate not in list_windows(os.getpid()):
                raise ValueError('Window changed during approval; list windows again')
            if self.window is None or (self.window['handle'],self.window['pid'])!=(candidate['handle'],candidate['pid']):
                self.coordinate_frame = None
            self.window = candidate
            self.notify('target',dict(candidate))
            return ToolResult(json.dumps({'selected_window':candidate},ensure_ascii=False))
        return self.desktop(name, arguments) if name.startswith('desktop_') else self.browse(name, arguments)

    def coordinate_reference(self,image=None,max_edge=1280):
        if image is not None:
            from game_agent.vision import prepare_image
            frame = image.info.get('desktop_frame')
            size = prepare_image(image,max_edge,'rgb').size
            self.coordinate_frame = dict(frame,image_size=list(size)) if frame else dict(source='unbound_image')
        elif self.coordinate_frame is None and self.window is not None:
            region = self.capture_region()
            self.coordinate_frame = dict(source='selected_window',handle=self.window['handle'],pid=self.window['pid'],
                bounds=region.bbox,image_size=[region.width,region.height])
        frame = self.coordinate_frame
        if frame and frame.get('source') in ('selected_window','visible_desktop'):
            return dict(coordinate_space='image_pixels',width=frame['image_size'][0],height=frame['image_size'][1],
                        source=frame['source'],origin='top-left; software handles screen offsets')
        return dict(coordinate_space='image_pixels',source='no_desktop_reference')

    def pointer_region(self,region,arguments):
        from desktop_agent.coordinates import input_region
        return input_region(region,arguments,self.coordinate_frame,self.window)

    @physical_pixels()
    def capture_region(self):
        self.check()
        if self.window is None:
            return Region(*windows.virtual_screen())
        handle = self.window['handle']
        process = wintypes.DWORD()
        windows.user32.GetWindowThreadProcessId(handle,ctypes.byref(process))
        if not windows.user32.IsWindow(handle) or process.value != self.window['pid']:
            raise ValueError('Selected window closed or changed process')
        if windows.user32.IsIconic(handle):
            raise ValueError('Selected window is minimized; use desktop_screen_capture for the current screen')
        return visible_window_region(handle)

    @physical_pixels()
    def capture_window(self):
        region = self.capture_region()
        handle = self.window['handle']
        rectangle = window_frame_rect(handle)
        raw_rectangle = windows.window_rect(handle)
        try:
            image = capture_window_image(handle,raw_rectangle,self.stopped)
        except OSError:
            raise ValueError('Selected window capture is unsupported; use desktop_screen_capture for the visible desktop') from None
        if windows.window_rect(handle) != raw_rectangle or window_frame_rect(handle) != rectangle or self.capture_region().bbox != region.bbox:
            raise ValueError('Selected window moved during capture; request a new capture')
        left,top,right,bottom = raw_rectangle
        if image.size != (right-left,bottom-top):
            raise ValueError('Window capture pixel size differs from its physical bounds; use desktop_screen_capture')
        crop = (region.left-left,region.top-top,region.left-left+region.width,region.top-top+region.height)
        image = image.crop(crop)
        return region,image

    @physical_pixels()
    def focus_window(self):
        self.check()
        handle = self.window['handle']
        if not windows.user32.IsWindow(handle):
            raise ValueError('Selected window closed')
        process = wintypes.DWORD()
        windows.user32.GetWindowThreadProcessId(handle, ctypes.byref(process))
        if process.value != self.window['pid']:
            raise ValueError('Selected window process changed')
        if windows.user32.IsIconic(handle):
            windows.user32.ShowWindow(handle, 9)
        windows.user32.SetForegroundWindow(handle)
        foreground = windows.user32.GetForegroundWindow()
        if foreground != handle:
            current_thread = windows.kernel32.GetCurrentThreadId()
            foreground_thread = windows.user32.GetWindowThreadProcessId(foreground,None)
            if foreground_thread and foreground_thread != current_thread:
                attached = windows.user32.AttachThreadInput(current_thread,foreground_thread,True)
                try:
                    if attached:
                        windows.user32.BringWindowToTop(handle)
                        windows.user32.SetForegroundWindow(handle)
                finally:
                    if attached:
                        windows.user32.AttachThreadInput(current_thread,foreground_thread,False)
        if self.stopped.wait(0.2):
            raise Halted('Stopped')
        region = visible_window_region(handle)
        target = InputTarget(self.window,region)
        if not target.valid():
            raise ValueError('Selected input window is not focused or changed')
        return region, target

    @physical_pixels()
    def desktop(self, name, arguments):
        if name == 'desktop_screen_capture':
            region = Region(*windows.virtual_screen())
            image = windows.capture(region,max_edge=max(region.width,region.height))
            if self.window is not None:
                image.info['desktop_frame'] = dict(source='visible_desktop',bounds=region.bbox,
                    handle=self.window['handle'],pid=self.window['pid'],target_bounds=self.capture_region().bbox)
            return ToolResult(json.dumps({'source':'visible_desktop','bounds':region.bbox,'width':image.width,'height':image.height,
                'notice':'Captured without changing any window. Whole-desktop coordinates differ from selected-window input coordinates.'}),image)
        if name == 'desktop_capture':
            if self.window is None:
                return self.desktop('desktop_screen_capture',{})
            if self.capture_margin:
                target = self.capture_region()
                left,top,width,height = windows.virtual_screen()
                margin = self.capture_margin
                bounds = (max(left,target.left-margin),max(top,target.top-margin),
                          min(left+width,target.left+target.width+margin),min(top+height,target.top+target.height+margin))
                region = Region(bounds[0],bounds[1],bounds[2]-bounds[0],bounds[3]-bounds[1])
                image = windows.capture(region,max_edge=max(region.width,region.height))
                if self.capture_region().bbox != target.bbox:
                    raise ValueError('Selected window moved during capture; request a new capture')
                image.info['desktop_frame'] = dict(source='visible_desktop',bounds=region.bbox,
                    handle=self.window['handle'],pid=self.window['pid'],target_bounds=target.bbox)
                return ToolResult(json.dumps(dict(source='visible_desktop',capture_method='window_surroundings',
                    bounds=region.bbox,target_bounds=target.bbox,margin=margin,width=image.width,height=image.height,
                    notice='Visible screen around selected window, including other windows and overlays. Input remains limited to the selected window.')),image)
            region,image = self.capture_window()
            image.info['desktop_frame'] = dict(source='selected_window',bounds=region.bbox,
                handle=self.window['handle'],pid=self.window['pid'])
            return ToolResult(json.dumps({'source':'selected_window','capture_method':'hwnd','handle':self.window['handle'],
                'window':windows.window_title(self.window['handle']), 'coordinate_space':'physical_pixels',
                'bounds':region.bbox,'width':image.width,'height':image.height,
                'notice':'New input calls use pixels of the transmitted image; software handles scaling and screen offsets. '+
                         'Captured the selected HWND, not the desktop rectangle or overlapping windows. No focus change or fixed border inset. '+
                         'This is not the managed browser page; browser_read cannot read this window.'},ensure_ascii=False),image)
        if name == 'desktop_record':
            def capture_frame():
                try:
                    region,image = self.capture_window()
                except ValueError as error:
                    raise Halted(str(error)) from None
                return image
            return self.record_video(capture_frame,arguments,
                'Selected HWND recording (physical-pixel bounds, no focus change): '+windows.window_title(self.window['handle']))
        if name in ('desktop_hold','desktop_drag_timed'):
            from desktop_agent.input_modes import message_input, scan_key, move_pointer, timed_drag
            if arguments['mode'] == 'message':
                kind = 'drag' if name == 'desktop_drag_timed' else 'click' if arguments['kind'] == 'mouse' else 'key'
                return ToolResult(message_input(self.window,dict(arguments,kind=kind),self.stopped,coordinate_frame=self.coordinate_frame))
            region,target = self.focus_window()
            if name=='desktop_drag_timed' or arguments['kind']=='mouse':
                region = self.pointer_region(region,arguments)
            target.require_geometry = name == 'desktop_drag_timed' or arguments['kind'] == 'mouse'
            backend = windows.DesktopInput(region,target,self.stopped)
            try:
                if name == 'desktop_drag_timed':
                    timed_drag(backend,region.point(arguments['x'],arguments['y']),region.point(arguments['end_x'],arguments['end_y']),
                               arguments['button'],arguments['duration_ms'],self.stopped)
                elif arguments['kind'] == 'mouse':
                    move_pointer(backend,region.point(arguments['x'],arguments['y']))
                    backend.down('click',arguments['button'])
                    try:
                        if self.stopped.wait(arguments['hold_ms']/1000):
                            raise Halted('Mouse hold cancelled')
                    finally:
                        backend.up('click',arguments['button'])
                elif arguments['mode'] == 'device':
                    scan_key(backend,arguments['key'],arguments['hold_ms'],self.stopped)
                else:
                    self.press_desktop_key(backend,arguments['key'],arguments['hold_ms'])
            finally:
                backend.release_all()
            return ToolResult('Timed input delivered and released in '+target.title+'. Verify application effect separately.')
        if name == 'desktop_input':
            from desktop_agent.input_modes import message_input, scan_key
            mode,kind = arguments['mode'],arguments['kind']
            if mode == 'message':
                return ToolResult(message_input(self.window,arguments,self.stopped,coordinate_frame=self.coordinate_frame))
            if mode == 'device' and kind == 'text':
                raise ValueError('Scancode mode does not type arbitrary Unicode; use general or message text mode')
            if mode == 'device' and kind == 'key':
                region,target = self.focus_window()
                target.require_geometry = False
                backend = windows.DesktopInput(region,target,self.stopped)
                try:
                    scan_key(backend,arguments['key'],arguments['hold_ms'],self.stopped)
                finally:
                    backend.release_all()
                return ToolResult('Software scancode input delivered to '+target.title+'; not physical hardware input.')
            mapped = {'click':('desktop_click',dict(x=arguments['x'],y=arguments['y'],button=arguments['button'],clicks=1)),
                      'key':('desktop_key',dict(key=arguments['key'])),
                      'text':('desktop_type',dict(text=arguments['text'])),
                      'scroll':('desktop_scroll',dict(x=arguments['x'],y=arguments['y'],amount=arguments['amount']))}
            tool,values = mapped[kind]
            if 'coordinate_space' in arguments and kind in ('click','scroll'):
                values['coordinate_space'] = arguments['coordinate_space']
            return self.desktop(tool,values)
        region, target = self.focus_window()
        region = self.pointer_region(region,arguments)
        target.require_geometry = name not in ('desktop_type','desktop_key','desktop_key_queue')
        backend = windows.DesktopInput(region, target, self.stopped)
        press_positions = []
        try:
            if name in ('desktop_click', 'desktop_drag', 'desktop_scroll'):
                point = region.point(arguments['x'], arguments['y'])
                backend._check(point)
                left, top, width, height = windows.virtual_screen()
                backend._send(windows.Input(0, windows.InputUnion(mouse=windows.MouseInput(
                    int((point[0]-left+0.5)*65536/width), int((point[1]-top+0.5)*65536/height), 0, 0xC001, 0, INPUT_TAG))))
                backend._check(point)
            if name == 'desktop_click':
                for index in range(arguments['clicks']):
                    press_positions.append(windows.cursor_position())
                    backend.down('click', arguments['button'])
                    if self.stopped.wait(0.05):
                        raise Halted('Stopped')
                    backend.up('click', arguments['button'])
            elif name == 'desktop_drag':
                backend.down('click', 'left')
                if self.stopped.wait(0.08):
                    raise Halted('Stopped')
                backend.move(region.point(arguments['end_x'], arguments['end_y']))
                backend.up('click', 'left')
            elif name == 'desktop_type':
                units = arguments['text'].encode('utf-16-le')
                for offset in range(0, len(units), 2):
                    backend._check()
                    code = int.from_bytes(units[offset:offset+2], 'little')
                    backend._send(windows.Input(1, windows.InputUnion(keyboard=windows.KeyboardInput(0, code, 4, 0, INPUT_TAG))))
                    backend._send(windows.Input(1, windows.InputUnion(keyboard=windows.KeyboardInput(0, code, 6, 0, INPUT_TAG))))
            elif name in ('desktop_key','desktop_key_queue'):
                steps = arguments['steps'] if name == 'desktop_key_queue' else [dict(key=arguments['key'],delay_ms=0)]
                for index, step in enumerate(steps):
                    self.wait_key(step,index,len(steps))
                    try:
                        self.press_desktop_key(backend,step['key'])
                    except windows.TargetUnavailable:
                        raise Halted(f'Desktop key sequence stopped: {target.failure_reason or "selected window unavailable"}; {index}/{len(steps)} key steps completed. Current step may be partial; remaining keys were not sent. Inspect the target before continuing; do not replay the whole queue.') from None
            elif name == 'desktop_scroll':
                backend._check(point)
                backend._send(windows.Input(0, windows.InputUnion(mouse=windows.MouseInput(0, 0, arguments['amount']*120, 0x0800, 0, INPUT_TAG))))
        finally:
            backend.release_all()
        if name == 'desktop_click':
            return ToolResult(json.dumps({'source':'selected_window','handle':self.window['handle'] if self.window else None,
                'coordinate_space':arguments.get('coordinate_space','normalized_0_1000'),
                ('pixels' if arguments.get('coordinate_space')=='image_pixels' else 'normalized'):[arguments['x'],arguments['y']],
                'bounds':region.bbox,'requested_desktop':point,'pointer_before_press':press_positions,
                'pointer_after_action':windows.cursor_position(),'button':arguments['button'],'clicks':arguments['clicks'],
                'notice':'Input delivered. Pointer samples are not application hit-test receipts; verify the intended control separately.'}))
        return ToolResult('Input delivered to '+target.title+'. Delivery is not proof of task success.')

    def record_video(self, capture_frame, arguments, label):
        import cv2
        import numpy as np
        seconds, fps = arguments['seconds'], arguments['fps']
        path = self.directory/(uuid.uuid4().hex+'.mp4')
        writer, previous = None, None
        written, samples, last_sample = 0, [], None
        interrupted = ''
        started = time.monotonic()
        try:
            while written < seconds*fps:
                self.check()
                delay = max(0,started+written/fps-time.monotonic())
                if self.stopped.wait(delay):
                    raise Halted('Recording stopped by user')
                self.check()
                if written and time.monotonic()-started >= seconds:
                    while written < seconds*fps:
                        self.check()
                        writer.write(previous)
                        written += 1
                    break
                image = capture_frame().convert('RGB')
                self.check()
                image.thumbnail((1280,1280),Image.Resampling.LANCZOS)
                size = (max(2,image.width//2*2),max(2,image.height//2*2))
                image = image.resize(size,Image.Resampling.LANCZOS)
                if writer is None:
                    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
                    if not writer.isOpened():
                        raise ValueError('MP4 encoder could not be opened')
                    frame_size = size
                elif size != frame_size:
                    raise Halted('Recording stopped: frame size changed')
                elapsed = time.monotonic()-started
                frame = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                due = min(seconds*fps,max(written+1,int(elapsed*fps)+1))
                while written < due:
                    self.check()
                    writer.write(previous if previous is not None and written < due-1 else frame)
                    written += 1
                previous = frame
                if self.recording_ready is not None:
                    self.recording_ready.set()
                thumbnail = image.copy()
                thumbnail.thumbnail((640,360),Image.Resampling.LANCZOS)
                last_sample = (elapsed,thumbnail)
                if len(samples) < 3 and elapsed >= len(samples)*seconds/3:
                    samples.append(last_sample)
                self.notify('status',f'Recording {min(elapsed,seconds):.1f}/{seconds}s | {fps} fps')
        except Halted as error:
            interrupted = str(error)
        except Exception:
            if writer is not None:
                writer.release()
            path.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.release()
        if not written:
            path.unlink(missing_ok=True)
            raise Halted(interrupted or 'Recording produced no frames')
        if last_sample is not None and (not samples or samples[-1][0] != last_sample[0]):
            samples.append(last_sample)
        sheet = Image.new('RGB',(1280,768),'#eef1f4')
        draw = ImageDraw.Draw(sheet)
        for index, (timestamp, thumbnail) in enumerate(samples):
            left, top = (index%2)*640,(index//2)*384
            draw.text((left+10,top+6),f'{timestamp:.2f}s',fill='#253342')
            sheet.paste(thumbnail,(left+(640-thumbnail.width)//2,top+24))
        text = json.dumps({'source':label,'video':str(path),'fps':fps,'frames':written,
            'duration_seconds':written/fps,'elapsed_seconds':time.monotonic()-started,
            'audio':False,'status':'interrupted' if interrupted else 'recorded',
            'reason':interrupted,'sample_times':[round(sample[0],2) for sample in samples],
            'image':'Timestamped sample frames only; not every video frame. Use only for this next response.'},ensure_ascii=False)
        return ToolResult(text,sheet,str(path),interrupted)

    def wait_key(self, step, index, total):
        self.check()
        self.notify('status',f'Key queue {index+1}/{total}: {step["key"]} ({step["delay_ms"]}ms)')
        if self.stopped.wait(step['delay_ms']/1000):
            raise Halted(f'Key queue stopped before step {index+1}/{total}; earlier inputs may already be delivered')
        self.check()

    def press_desktop_key(self, backend, chord, hold_ms=0):
        from desktop_agent.input_modes import key_code
        pressed = []
        try:
            for key in chord.split('+'):
                backend._check()
                code = key_code(key)
                backend._send(windows.Input(1, windows.InputUnion(keyboard=windows.KeyboardInput(code,0,0,0,INPUT_TAG))))
                pressed.append(code)
            if hold_ms and self.stopped.wait(hold_ms/1000):
                raise Halted('Key hold cancelled')
        finally:
            for code in reversed(pressed):
                backend._send(windows.Input(1, windows.InputUnion(keyboard=windows.KeyboardInput(code,0,2,0,INPUT_TAG))))

    def browser_key_queue(self, arguments):
        steps = arguments['steps']
        for index, step in enumerate(steps):
            self.wait_key(step,index,len(steps))
            action = dict(message='Queued key',tool='browser_key',arguments={'key':step['key']},risk='routine')
            context = self.browser_target('browser_key',action['arguments'])
            if context.get('password'):
                raise Halted('Key queue stopped at password/file input; use it manually')
            reason = target_approval_reason(context)
            if reason:
                allowed = self.approve(action,reason,context)
                self.notify('approval_log',dict(tool='browser_key',mode='confirmed' if allowed else 'denied',reason=reason))
                if not allowed:
                    raise Halted('User denied queued key; remaining inputs cancelled')
            self.check()
            if context != self.browser_target('browser_key',action['arguments']):
                raise Halted('Key queue target changed during approval; remaining inputs cancelled')
            self.page.keyboard.press(step['key'])

    def ensure_browser(self):
        self.check()
        if self.context is None:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.context = self.playwright.chromium.launch_persistent_context(
                str(self.directory/'browser-profile'), headless=self.headless, accept_downloads=False,
                args=['--remote-debugging-port=0','--remote-debugging-address=127.0.0.1'],
                viewport={'width':1280, 'height':800})
            self.context.set_default_timeout(5000)
            self.context.set_default_navigation_timeout(15000)
            self.context.on('page', self.register_page)
            for page in self.context.pages:
                self.register_page(page)
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        if self.page is None or self.page.is_closed():
            self.page = self.context.new_page()

    def register_page(self, page):
        page.on('dialog', lambda dialog: dialog.dismiss())
        page.on('download', lambda download: download.cancel())

    def browser_target(self, name, arguments):
        locator = self.page.locator(arguments['selector']) if 'selector' in arguments else self.page.locator(':focus')
        if locator.count() != 1:
            if name == 'browser_key':
                return {'url':self.page.url, 'label':''}
            raise ValueError('Selector must match one element')
        if not locator.is_visible():
            raise ValueError('Browser target is not visible')
        data = locator.evaluate('''element => {
            const form=element.form || element.closest('form');
            const search=element.type==='search' || (form && (form.getAttribute('role')==='search' || /search/i.test(form.getAttribute('action') || '')));
            return {label:(element.getAttribute('aria-label') || element.innerText || element.getAttribute('placeholder') || '').slice(0,300),
                    password:['password','file'].includes(element.type),
                    submit:!!form && (element.type==='submit' || element.type==='image'),search:!!search,
                    form_action:form ? (form.getAttribute('action') || '') : '',
                    form_button:form ? Array.from(form.querySelectorAll('button,input[type="submit"]')).map(button=>button.innerText || button.value || '').join(' ').slice(0,300) : '',
                    has_form:!!form,type:element.type || '',tag:element.tagName};
        }''')
        if name in ('browser_key','browser_hold') and arguments.get('key') in ('Enter','Control+Enter') and data['has_form'] and data['tag'] != 'TEXTAREA':
            data['submit'] = True
            data['label'] += ' '+data['form_button']+' '+data['form_action']
        if name == 'browser_type':
            data['submit'] = False
        data['url'] = self.page.url
        if name == 'browser_drag':
            destination = self.browser_target('browser_click',{'selector':arguments['target_selector']})
            data['destination'] = destination
            data['label'] += ' '+destination['label']
            data['password'] = data['password'] or destination['password']
        return data

    def browser_snapshot(self):
        controls = self.page.locator('a,button,input,textarea,select,[role="button"],[contenteditable="true"]').evaluate_all('''elements => elements.filter(element => element.getClientRects().length).slice(0,60).map(element => {
            let node=element, path=[];
            while(node && node.nodeType===1) {
                if(node.id) {path.unshift('#'+CSS.escape(node.id)); break;}
                let tag=node.tagName.toLowerCase(), position=1, sibling=node.previousElementSibling;
                while(sibling) {if(sibling.tagName===node.tagName) position++; sibling=sibling.previousElementSibling;}
                path.unshift(tag+':nth-of-type('+position+')'); node=node.parentElement;
            }
            return {selector:path.join(' > '),type:element.type || '',label:(element.getAttribute('aria-label')||element.innerText||element.getAttribute('placeholder')||element.getAttribute('name')||'').slice(0,100)};
        })''')
        text = self.page.locator('body').inner_text(timeout=5000)[:6000]
        notice = ''
        if self.page.url == 'about:blank' and not text.strip() and not controls:
            notice = ('The managed browser is empty, not the selected desktop Chrome window. '+
                      'For the selected window use desktop_capture and read its image; otherwise open a requested URL. '+
                      'Repeating browser_read here will not reveal the selected desktop window.')
        return ToolResult(json.dumps({'scope':'managed_browser_only','url':self.page.url, 'title':self.page.title(),
            'text':text, 'controls':controls, 'notice':notice}, ensure_ascii=False))

    def browser_search(self, query):
        from playwright.sync_api import Error as BrowserError
        engines = [('google','https://www.google.com/search?q='),('bing','https://www.bing.com/search?q=')]
        if getattr(self,'search_provider','google') == 'bing':
            engines = engines[1:]
        attempts = []
        for provider,base in engines:
            self.check()
            status = None
            reason = ''
            try:
                response = self.page.goto(base+quote(query,safe=''),wait_until='domcontentloaded')
                status = response.status if response is not None else None
                self.check()
                snapshot = self.browser_snapshot()
                data = json.loads(snapshot.text)
                location = urlsplit(data['url'])
                body = data.get('text','').casefold()
                if status in (403,429):
                    reason = 'http_'+str(status)
                elif status is not None and status >= 400:
                    reason = 'http_error_'+str(status)
                elif provider == 'google' and (location.path.startswith('/sorry') or
                        any(marker in body for marker in ('our systems have detected unusual traffic',
                            'unusual traffic from your computer network','\ube44\uc815\uc0c1\uc801\uc778 \ud2b8\ub798\ud53d'))):
                    reason = 'unusual_traffic'
                elif location.hostname == 'consent.google.com':
                    reason = 'consent_required'
                elif self.page.locator('form[action*="/sorry"], #captcha, #b_captcha, iframe[src*="recaptcha/api2/bframe"], iframe[src*="hcaptcha.com/captcha"]').count():
                    reason = 'captcha'
            except BrowserError:
                self.check()
                reason = 'navigation_failed'
                data = dict(scope='managed_browser_only',url=self.page.url,title='',text='',controls=[])
            attempts.append(dict(provider=provider,url=data['url'],http_status=status,reason=reason))
            if reason and provider == 'google':
                self.search_provider = 'bing'
                self.notify('status','Google \uac80\uc0c9 \ucc28\ub2e8 \ub610\ub294 \uc811\uc18d \uc2e4\ud328: Bing\uc73c\ub85c \ud55c \ubc88 \uc804\ud658\ud569\ub2c8\ub2e4.')
                continue
            data['search'] = dict(provider=provider,status='blocked' if reason else 'available',attempts=attempts)
            if reason:
                data.update(text='',controls=[],notice='Search is blocked or unavailable; no search results were retrieved. Do not retry or solve CAPTCHA automatically.')
            elif len(attempts)>1:
                data['notice'] = 'Google was unavailable; these are Bing results. Further searches in this browser session use Bing.'
            self.check()
            return ToolResult(json.dumps(data,ensure_ascii=False),error='Web search unavailable: '+reason if reason else '')

    def browse(self, name, arguments):
        self.ensure_browser()
        if name == 'browser_record':
            page = self.page
            def capture_frame():
                return Image.open(BytesIO(page.screenshot(timeout=8000))).convert('RGB')
            return self.record_video(capture_frame,arguments,'Managed browser: '+page.url)
        if name == 'browser_capture':
            image = Image.open(BytesIO(self.page.screenshot(timeout=8000))).convert('RGB')
            return ToolResult('Current browser capture: '+self.page.url, image)
        if name == 'browser_tabs':
            return ToolResult(json.dumps([{'index':index, 'title':page.title(), 'url':page.url}
                for index, page in enumerate(self.context.pages) if not page.is_closed()], ensure_ascii=False))
        if name in ('browser_hold','browser_drag'):
            source = self.page.locator(arguments['selector'])
            source.scroll_into_view_if_needed()
            if name == 'browser_hold' and arguments['kind'] == 'key':
                source.focus()
                pressed = []
                try:
                    for key in arguments['key'].split('+'):
                        self.check()
                        self.page.keyboard.down(key)
                        pressed.append(key)
                    if self.stopped.wait(arguments['hold_ms']/1000):
                        raise Halted('Browser key hold cancelled')
                finally:
                    for key in reversed(pressed):
                        self.page.keyboard.up(key)
            else:
                box = source.bounding_box()
                if not box:
                    raise ValueError('Mouse target has no visible bounding box')
                start = (box['x']+box['width']/2,box['y']+box['height']/2)
                if name == 'browser_drag':
                    destination = self.page.locator(arguments['target_selector'])
                    box = destination.bounding_box()
                    if not box:
                        raise ValueError('Drag destination is not visible')
                    end = (box['x']+box['width']/2,box['y']+box['height']/2)
                button = arguments.get('button','left')
                self.page.mouse.move(*start)
                self.check()
                self.page.mouse.down(button=button)
                try:
                    if name == 'browser_drag':
                        steps = max(1,arguments['duration_ms']//20)
                        for step in range(1,steps+1):
                            if self.stopped.wait(arguments['duration_ms']/steps/1000):
                                raise Halted('Browser drag cancelled')
                            self.page.mouse.move(*(start[axis]+(end[axis]-start[axis])*step/steps for axis in (0,1)))
                    elif self.stopped.wait(arguments['hold_ms']/1000):
                        raise Halted('Browser mouse hold cancelled')
                finally:
                    self.page.mouse.up(button=button)
        elif name == 'browser_click':
            self.page.locator(arguments['selector']).click()
        elif name == 'browser_type':
            self.page.locator(arguments['selector']).fill(arguments['text'])
        elif name == 'browser_key':
            self.page.keyboard.press(arguments['key'])
        elif name == 'browser_key_queue':
            self.browser_key_queue(arguments)
        elif name == 'browser_open':
            self.page.goto(web_url(arguments['url']), wait_until='domcontentloaded')
        elif name == 'browser_search':
            return self.browser_search(arguments['query'])
        elif name == 'browser_scroll':
            self.page.mouse.wheel(0, arguments['amount']*600)
        elif name == 'browser_back':
            self.page.go_back(wait_until='domcontentloaded')
        elif name == 'browser_tab':
            if arguments['index'] >= len(self.context.pages):
                raise ValueError('Unknown tab')
            self.page = self.context.pages[arguments['index']]
            self.page.bring_to_front()
        self.check()
        return self.browser_snapshot()

    def close(self):
        try:
            if self.context is not None:
                self.context.close()
        finally:
            if self.playwright is not None:
                self.playwright.stop()
            self.context = self.playwright = self.page = None


if __name__ == '__main__':
    if len(sys.argv) != 7 or sys.argv[1] != '--capture-window':
        raise SystemExit(2)
    try:
        handle,left,top,right,bottom = map(int,sys.argv[2:])
        image = render_window_image(handle,(left,top,right,bottom))
        sys.stdout.buffer.write(image.tobytes())
    except (ValueError,OSError):
        raise SystemExit(1)