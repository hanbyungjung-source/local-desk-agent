import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import tempfile
import textwrap
import threading
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from markdown_it import MarkdownIt
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import ImageGrab, ImageTk
from desktop_agent.agent import Agent, DesktopServer, HOME, Model, Settings, StopEvent, validate_context
from desktop_agent.models import MODEL_PRESETS, model_preset
from desktop_agent.store import Store
from desktop_agent.tools import list_windows
from desktop_agent.jobs import ToolRunner
from desktop_agent.protocol import TOOLS, TOOL_GROUPS, WORKSPACE_TOOLS, tool_policy
from desktop_agent.history import execution_status,linked_events,result_record
from desktop_agent.api import APISettings, FORMATS, TOKENIZERS, endpoint, uses_google_thinking
from desktop_agent.credentials import credential_scope, read_vertex_file
from game_agent.windows import InputMonitor, enable_dpi


COLORS = dict(panel='#252526',editor='#1e1e1e',input='#313131',text='#d4d4d4',muted='#a9a9a9',
              border='#454545',selection='#264f78',accent='#007acc',link='#75beff',warning='#e5c07b')


def dialog_work_area(parent):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class MonitorInfo(ctypes.Structure):
            _fields_ = [('size',wintypes.DWORD),('monitor',wintypes.RECT),
                        ('work',wintypes.RECT),('flags',wintypes.DWORD)]
        native = ctypes.WinDLL('user32',use_last_error=True)
        native.MonitorFromRect.argtypes = [ctypes.POINTER(wintypes.RECT),wintypes.DWORD]
        native.MonitorFromRect.restype = wintypes.HANDLE
        native.GetMonitorInfoW.argtypes = [wintypes.HANDLE,ctypes.POINTER(MonitorInfo)]
        native.GetMonitorInfoW.restype = wintypes.BOOL
        left,top = parent.winfo_rootx(),parent.winfo_rooty()
        bounds = wintypes.RECT(left,top,left+parent.winfo_width(),top+parent.winfo_height())
        monitor = native.MonitorFromRect(ctypes.byref(bounds),2)
        info = MonitorInfo(size=ctypes.sizeof(MonitorInfo))
        if monitor and native.GetMonitorInfoW(monitor,ctypes.byref(info)):
            return info.work.left,info.work.top,info.work.right,info.work.bottom
    return 0,0,parent.winfo_screenwidth(),parent.winfo_screenheight()


def place_dialog(dialog,parent,width,height):
    dialog.withdraw()
    dialog.transient(parent)
    parent.update_idletasks()
    left,top,right,bottom = dialog_work_area(parent)
    width,height = min(width,right-left-24),min(height,bottom-top-56)
    minimum_width,minimum_height = dialog.minsize()
    dialog.minsize(min(minimum_width,width),min(minimum_height,height))
    position_x = parent.winfo_rootx()+(parent.winfo_width()-width)//2
    position_y = parent.winfo_rooty()+(parent.winfo_height()-height)//2
    position_x = max(left+12,min(position_x,right-width-12))
    position_y = max(top+32,min(position_y,bottom-height-24))
    dialog.geometry(f'{width}x{height}+{position_x}+{position_y}')
    dialog.deiconify()


def dark_theme(root):
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.',font=('Malgun Gothic',10),background=COLORS['panel'],foreground=COLORS['text'],
                    bordercolor=COLORS['border'],lightcolor=COLORS['border'],darkcolor=COLORS['border'])
    style.configure('TButton',padding=(7,5),background=COLORS['input'])
    style.map('TButton',background=[('active','#404040'),('pressed',COLORS['selection'])])
    style.configure('Accent.TButton',background=COLORS['accent'],foreground='white')
    style.map('Accent.TButton',background=[('active','#168bd2'),('pressed','#005a9e')])
    style.map('.',foreground=[('disabled','#777777')])
    for name in ('TEntry','TCombobox','TSpinbox'):
        style.configure(name,fieldbackground=COLORS['input'],insertcolor=COLORS['text'],padding=5)
        style.map(name,fieldbackground=[('readonly',COLORS['input']),('disabled',COLORS['panel'])],
                  foreground=[('readonly',COLORS['text'])],selectbackground=[('!disabled',COLORS['selection'])])
    for name in ('TCheckbutton','TRadiobutton'):
        style.configure(name,indicatorbackground=COLORS['input'],indicatormargin=5)
        style.map(name,background=[('active',COLORS['input'])],indicatorbackground=[('selected',COLORS['accent'])])
    style.configure('Treeview',background=COLORS['editor'],fieldbackground=COLORS['editor'],rowheight=29)
    style.map('Treeview',background=[('selected',COLORS['selection'])],foreground=[('selected','white')])
    style.configure('Treeview.Heading',background=COLORS['panel'],padding=6)
    style.map('Treeview.Heading',background=[('active',COLORS['input'])])
    style.configure('TNotebook',background=COLORS['panel'],borderwidth=0)
    style.configure('TNotebook.Tab',padding=(12,6),background=COLORS['panel'])
    style.map('TNotebook.Tab',background=[('selected',COLORS['editor'])],foreground=[('selected','white')])
    style.configure('TScrollbar',background=COLORS['border'],troughcolor=COLORS['editor'],arrowsize=12)
    for pattern,value in (('*background',COLORS['panel']),('*foreground',COLORS['text']),
                          ('*Text.background',COLORS['editor']),('*Text.insertBackground',COLORS['text']),
                          ('*Text.selectBackground',COLORS['selection']),('*Text.selectForeground','white'),
                          ('*Listbox.background',COLORS['editor']),('*Listbox.selectBackground',COLORS['selection']),
                          ('*Listbox.selectForeground','white'),('*Entry.background',COLORS['input']),
                          ('*Entry.insertBackground',COLORS['text']),('*Menu.activeBackground',COLORS['selection']),
                          ('*Menu.activeForeground','white'),('*highlightBackground',COLORS['border']),
                          ('*highlightColor',COLORS['accent']),('*TCombobox*Listbox.background',COLORS['input'])):
        root.option_add(pattern,value)
    root.configure(background=COLORS['panel'])


def display_excerpt(text, limit=12000):
    text = str(text)
    shortened=len(text)>limit
    visible=text[-limit:] if shortened else text
    visible='\n'.join(textwrap.fill(line,width=200,replace_whitespace=False,drop_whitespace=False)
                      if len(line)>500 else line for line in visible.split('\n'))
    return ('[\ud45c\uc2dc \uc904\uc784: \uc6d0\ubb38\uc740 \ub300\ud654 \uae30\ub85d\uc5d0 \ubcf4\uc874]\n' if shortened else '')+visible


class PromptText(tk.Text):
    def __init__(self,parent,variable,**kwargs):
        super().__init__(parent,undo=True,autoseparators=True,maxundo=200,**kwargs)
        self.variable=variable
        self.syncing=False
        self.bind('<<Modified>>',self.changed)
        variable.trace_add('write',self.external_change)
        for sequence,action in (('<Control-z>',self.edit_undo),('<Control-y>',self.edit_redo),
                                ('<Control-Shift-Z>',self.edit_redo)):
            self.bind(sequence,lambda event,action=action:self.history(action))

    def history(self,action):
        try: action()
        except tk.TclError: pass
        return 'break'

    def changed(self,event=None):
        if self.edit_modified():
            if not self.syncing:
                self.syncing=True
                self.variable.set(self.get('1.0','end-1c'))
                self.syncing=False
            self.edit_modified(False)

    def external_change(self,*args):
        if not self.syncing and self.get('1.0','end-1c')!=self.variable.get():
            self.syncing=True
            self.delete('1.0','end')
            self.insert('1.0',self.variable.get())
            self.edit_reset()
            self.edit_modified(False)
            self.syncing=False


def performance_text(metrics):
    if not isinstance(metrics,dict) or not metrics:
        return ''
    def measured(value):
        return type(value) in (int,float) and math.isfinite(value) and value >= 0
    timings = metrics.get('timings') or {}
    def speed(prefix):
        rate = timings.get(prefix+'_per_second')
        if not measured(rate):
            count, milliseconds = timings.get(prefix+'_n'), timings.get(prefix+'_ms')
            rate = count*1000/milliseconds if measured(count) and measured(milliseconds) and milliseconds > 0 else None
        return f'{rate:,.1f} tok/s' if measured(rate) else '\ubbf8\uce21\uc815'
    lines = ['\uc785\ub825 '+speed('prompt')+' \u00b7 \ucd9c\ub825 '+speed('predicted')]
    elapsed = []
    for key, label in (('seconds','\uc774\ubc88 \ucd94\ub860'),('task_seconds','\ub2f5\ubcc0\uae4c\uc9c0')):
        if measured(metrics.get(key)):
            elapsed.append(f'{label} {metrics[key]:.1f}\ucd08')
    if elapsed:
        lines.append(' \u00b7 '.join(elapsed))
    effort = metrics.get('reasoning_effort')
    if effort in ('none','low','medium','xhigh'):
        lines.append('reasoning_effort: '+effort)
    context,limit = metrics.get('context_tokens'),metrics.get('context_limit')
    if measured(context) and measured(limit) and limit > 0:
        source = '\uc11c\ubc84 \ubcf4\uace0' if metrics.get('context_source') == 'server_usage' else '\ucd94\uc815'
        lines.append(f'\ucee8\ud14d\uc2a4\ud2b8(\uc785\ub825+\ucd9c\ub825): {context:,.0f} / {limit:,.0f} tok ({source})')
    if metrics.get('backend') == 'api':
        usage = metrics.get('usage') or {}
        lines.append(f'API tokens: {usage.get("prompt_tokens","?")} in / {usage.get("completion_tokens","?")} out')
        if measured(metrics.get('observed_output_tps')):
            lines.append(f'\uc694\uccad \ud3c9\uade0 \ucd9c\ub825 {metrics["observed_output_tps"]:.1f} tok/s (\ucd94\uc815)')
        if metrics.get('request_attempts',1) > 1:
            lines.append(f'API requests: {metrics["request_attempts"]}')
    cache = metrics.get('prompt_cache')
    if isinstance(cache,dict):
        cached = cache.get('reported_cached_tokens')
        lines.append('\uc785\ub825 KV \uc7ac\uc0ac\uc6a9: '+(f'{cached:,} tok (\uc11c\ubc84 \ubcf4\uace0)' if type(cached) is int else '\uc11c\ubc84 \ubcf4\uace0 \uc5c6\uc74c'))
    return '\n'.join(lines)


def tool_rows(events):
    rows=[]
    by_call={}
    for event in linked_events(events):
        metadata=event.get('metadata',{})
        if event['role']=='assistant':
            try:
                call=json.loads(event.get('raw_content',event['content']))
                if call.get('tool','finish')!='finish':
                    narration=call.get('message','')
                    if narration in (call['tool'],'Load tool group') or narration.startswith('Start desktop_') or narration.startswith('Start browser_'):
                        narration=''
                    row=dict(id=event['id'],tool=call['tool'],arguments=call.get('arguments',{}),commentary=narration,call_id=metadata.get('call_id',''),
                             status='running',execution_status='running',result=None,results=[])
                    rows.append(row)
                    if row['call_id']:
                        by_call[row['call_id']]=row
            except (ValueError,TypeError):
                pass
        elif event['role']=='tool' or (event['role']=='system' and metadata.get('status') in ('terminal','file_change')):
            name=metadata.get('tool','tool')
            pending=by_call.get(metadata.get('call_id'))
            if pending is None:
                pending=dict(id=event['id'],tool=name,arguments=metadata.get('call_arguments',{}),call_id=metadata.get('call_id',''),
                             status='unknown',execution_status='unknown',result=None,results=[])
                rows.append(pending)
                if pending['call_id']:
                    by_call[pending['call_id']]=pending
            targets=[pending]
            origin=by_call.get(metadata.get('origin_call_id'))
            if origin is not None and origin is not pending:
                targets.append(origin)
            for row in targets:
                outcome=execution_status(event)
                row['results'].append(event)
                if row['execution_status'] in ('completed','failed','cancelled','timed_out','output_limit','applied') and outcome in ('starting','running','ready_to_collect'):
                    continue
                row.update(status=metadata.get('status','delivered'),execution_status=outcome,result=event)
    return rows


def conversation_items(events):
    items, activity = [], []
    for event in events:
        role, content = event['role'], event['content']
        metadata = event.get('metadata',{})
        if role == 'assistant' and not metadata.get('partial'):
            try:
                action = json.loads(content)
                content = action['message']
                if action.get('tool','finish') != 'finish':
                    activity.append(dict(event,content=content,raw_content=event['content']))
                    continue
            except (ValueError,KeyError,TypeError):
                pass
        if role == 'tool' or (role == 'system' and metadata.get('status') in ('approval','terminal','file_change')):
            activity.append(event)
            continue
        if activity:
            items.append({'role':'activity','events':activity,'id':activity[0].get('id',len(items))})
            activity = []
        items.append(dict(event,content=content))
    if activity:
        items.append({'role':'activity','events':activity,'id':activity[0].get('id',len(items))})
    return items


class Console(TkinterDnD.Tk):
    def __init__(self, directory=None):
        super().__init__()
        self.title('Local Desk')
        self.geometry('1180x800+30+30')
        self.minsize(960,640)
        self.data = Path(directory or HOME/'data')
        self.store = Store(self.data)
        self.settings = Settings.load(self.data/'settings.json')
        rows = self.store.sessions()
        self.identifier = rows[0]['id'] if rows else self.store.create()
        self.events, self.jobs = queue.Queue(), queue.Queue()
        self.stopped = StopEvent()
        self.pending_jobs = 0
        self.model = Model(self.settings,self.data/'api-keys.dpapi')
        self.api_consents = set()
        self.busy = self.compact = self.closing = False
        self.approval = None
        self.approval_lock = threading.Lock()
        self.attachments = []
        self.photo = None
        self.window_choices = []
        self.session_rows = []
        self.capture_paths = []
        self.active_tools = None
        self.active_tools_session = None
        self.execution_windows = {}
        self.tooltips = []
        self.expanded_activity = set()
        self.expanded_reasoning = set()
        self.live_reasoning = None
        self.live_display = None
        self.rendered_session = None
        self.transcript_dragging = False
        self.transcript_refresh_pending = False
        self.markdown = MarkdownIt('commonmark',{'html':False})
        self.status = tk.StringVar(value='Ready')
        self.context_status = tk.StringVar(value=f'Local | {self.settings.context_tokens//1024}K')
        self.prompt = tk.StringVar()
        self.reasoning = tk.BooleanVar(value=self.settings.reasoning_enabled)
        self.automatic = tk.BooleanVar(value=True)
        self.auto_compact = tk.BooleanVar(value=self.settings.auto_compact)
        self.screen = tk.BooleanVar(value=True)
        self.inputs = tk.BooleanVar(value=True)
        self.browser = tk.BooleanVar(value=True)
        self.topmost = tk.BooleanVar(value=False)
        self.build()
        for widget in (self,self.entry,self.transcript):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind('<<Drop>>',self.drop_files)
        self.refresh_sessions()
        self.render()
        self.refresh_windows()
        self.monitor = InputMonitor(lambda reason:self.events.put(('interrupt',reason)))
        self.monitor.stop_on_input = False
        self.update_backend_display()
        self.worker = threading.Thread(target=self.work,daemon=True)
        self.worker.start()
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.bind('<F8>',lambda event:self.stop('F8'))
        self.after(60,self.poll)

    def icon(self, parent, symbol, label, command):
        button = ttk.Button(parent,text=symbol,width=3,command=command)
        popup = []
        def leave(event=None):
            if popup:
                popup.pop().destroy()
        def enter(event):
            leave()
            tip = tk.Toplevel(self)
            tip.overrideredirect(True)
            tip.geometry(f'+{button.winfo_rootx()}+{button.winfo_rooty()+32}')
            tk.Label(tip,text=label,bg=COLORS['input'],fg=COLORS['text'],padx=7,pady=4).pack()
            popup.append(tip)
        button.bind('<Enter>',enter)
        button.bind('<Leave>',leave)
        button.bind('<ButtonPress>',leave,add='+')
        return button

    def build(self):
        dark_theme(self)
        self.header = ttk.Frame(self,padding=(12,10))
        self.header.pack(fill='x')
        ttk.Label(self.header,text='Local Desk',font=('Bahnschrift',18)).pack(side='left')
        ttk.Label(self.header,textvariable=self.context_status).pack(side='left',padx=10)
        self.icon(self.header,'\u2699','Model settings',self.settings_dialog).pack(side='right')
        self.api_button = self.icon(self.header,'\u2601','API settings / credentials',self.api_settings_dialog)
        self.api_button.pack(side='right',padx=3)
        self.load_button = self.icon(self.header,'\u25b6','Load model',lambda:self.queue_model('load'))
        self.load_button.pack(side='right',padx=4)
        self.unload_button = self.icon(self.header,'\u23cf','Unload model and browser',lambda:self.queue_model('unload'))
        self.unload_button.pack(side='right')
        self.think_button = ttk.Checkbutton(self.header,text='Reasoning',variable=self.reasoning,command=self.reasoning_changed)
        self.think_button.pack(side='right',padx=6)
        self.reason_level = tk.StringVar(value=self.settings.reasoning_level)
        self.reason_selector = ttk.Combobox(self.header,textvariable=self.reason_level,values=self.settings.reasoning_levels,width=9,state='readonly')
        self.reason_selector.pack(side='right',padx=4)
        self.reason_selector.bind('<<ComboboxSelected>>',self.reasoning_level_changed)
        self.body = ttk.Panedwindow(self,orient='horizontal')
        self.body.pack(fill='both',expand=True,padx=10)
        sidebar = ttk.Frame(self.body,padding=8,width=195)
        self.body.add(sidebar,weight=0)
        row = ttk.Frame(sidebar)
        row.pack(fill='x')
        ttk.Label(row,text='Sessions',font=('Bahnschrift',12)).pack(side='left')
        self.icon(row,'+','New session',self.new_session).pack(side='right')
        self.sessions = tk.Listbox(sidebar,width=22,relief='flat',exportselection=False,activestyle='none',
            bg=COLORS['panel'],fg=COLORS['text'],selectbackground=COLORS['selection'],selectforeground='white',font=('Malgun Gothic',10))
        self.sessions.pack(fill='both',expand=True,pady=8)
        self.sessions.bind('<<ListboxSelect>>',self.select_session)
        row = ttk.Frame(sidebar)
        row.pack(fill='x')
        self.icon(row,'\u270e','Rename session',self.rename_session).pack(side='left')
        self.icon(row,'\u2193','Export conversation',self.export_session).pack(side='left',padx=6)
        self.delete_session_button = self.icon(row,'\u00d7','Delete session and files',self.delete_session)
        self.delete_session_button.pack(side='right')
        center = ttk.Frame(self.body,padding=8)
        self.body.add(center,weight=4)
        row = ttk.Frame(center)
        row.pack(fill='x',pady=(0,8))
        ttk.Label(row,text='\ub300\ud654',font=('Malgun Gothic',11,'bold')).pack(side='left')
        self.icon(row,'\u2398','Copy last reply',self.copy_answer).pack(side='right')
        self.transcript = ScrolledText(center,wrap='word',state='disabled',relief='flat',bg=COLORS['editor'],
            fg=COLORS['text'],padx=22,pady=18,font=('Malgun Gothic',12),width=40,spacing2=6,
            selectbackground=COLORS['selection'],selectforeground='white',borderwidth=0,highlightthickness=0)
        self.transcript.pack(fill='both',expand=True)
        self.transcript.configure(exportselection=False)
        self.transcript.bind('<ButtonPress-1>',self.begin_transcript_selection)
        self.transcript.bind('<ButtonRelease-1>',self.end_transcript_selection)
        self.transcript.bind('<<Copy>>',self.copy_selection)
        self.transcript.bind('<Control-c>',self.copy_selection)
        self.transcript.bind('<Control-C>',self.copy_selection)
        self.transcript.bind('<Button-3>',self.show_selection_menu)
        self.message_context = tk.Menu(self,tearoff=False)
        for tag, options in {
            'user_label':dict(justify='right',foreground=COLORS['muted'],font=('Malgun Gothic',9),spacing1=14,spacing3=7),
            'user':dict(justify='right',background='#2b3036',lmargin1=70,lmargin2=70,rmargin=12,spacing1=10,spacing3=10),
            'assistant_label':dict(foreground='#4ec9b0',font=('Malgun Gothic',9,'bold'),spacing1=18,spacing3=9),
            'assistant':dict(lmargin1=4,lmargin2=4,rmargin=12,spacing3=9),
            'system':dict(foreground=COLORS['warning'],background='#302c23',font=('Malgun Gothic',10),spacing1=10,spacing3=10),
            'activity':dict(foreground=COLORS['muted'],font=('Malgun Gothic',10),spacing1=12,spacing3=12),
            'detail':dict(foreground='#b8c3ce',background=COLORS['panel'],font=('Malgun Gothic',10),lmargin1=12,lmargin2=12,spacing3=7),
            'metrics':dict(foreground=COLORS['muted'],font=('Malgun Gothic',9),spacing1=6,spacing2=3,spacing3=8),
            'strong':dict(font=('Malgun Gothic',12,'bold')),
            'em':dict(font=('Malgun Gothic',12,'italic')),
            'heading':dict(font=('Malgun Gothic',14,'bold'),spacing1=10,spacing3=10),
            'code':dict(font=('Consolas',11),background=COLORS['panel'],foreground='#ce9178'),
            'codeblock':dict(font=('Consolas',11),background=COLORS['panel'],foreground='#ce9178',lmargin1=14,lmargin2=14,spacing1=8,spacing3=8),
            'quote':dict(lmargin1=20,lmargin2=20,foreground=COLORS['muted']),
            'link':dict(foreground=COLORS['link'],underline=True),
            'gap':dict(font=('Malgun Gothic',5),spacing1=0,spacing2=0,spacing3=0),
        }.items():
            self.transcript.tag_configure(tag,**options)
        self.stream = tk.Text(center,height=3,wrap='word',state='disabled',relief='flat',bg=COLORS['editor'],
            fg=COLORS['text'],padx=18,pady=10,spacing2=5,font=('Malgun Gothic',11))
        right = ttk.Frame(self.body,padding=8,width=290)
        self.body.add(right,weight=0)
        ttk.Label(right,text='Tools',font=('Bahnschrift',12)).pack(anchor='w')
        self.tool_picker_button = ttk.Button(right,text='\u2637  \ub3c4\uad6c \uc120\ud0dd',command=self.tools_dialog)
        self.tool_picker_button.pack(fill='x',pady=(6,8))
        self.execution_button = ttk.Button(right,text='>_  \ud130\ubbf8\ub110 / \ubcc0\uacbd \ud30c\uc77c',command=self.execution_dialog)
        self.execution_button.pack(fill='x',pady=(0,8))
        self.permissions = []
        for label, variable in (('Window capture / video',self.screen),('Mouse / keyboard',self.inputs),
                                ('Managed browser',self.browser),('Auto-approve routine tasks',self.automatic)):
            control = ttk.Checkbutton(right,text=label,variable=variable)
            control.pack(anchor='w',pady=2)
            self.permissions.append(control)
        compact_row=ttk.Frame(right)
        compact_row.pack(fill='x')
        compact_control=ttk.Checkbutton(compact_row,text='\ub300\ud654 \uc790\ub3d9 \uc555\ucd95',variable=self.auto_compact,command=self.compaction_changed)
        compact_control.pack(side='left')
        self.permissions.append(compact_control)
        self.icon(compact_row,'\u2261','\ubcf4\uad00\ub41c \ub300\ud654 \uc694\uc57d',self.show_context_summary).pack(side='right')
        row = ttk.Frame(right)
        row.pack(fill='x',pady=(8,2))
        ttk.Label(row,text='Target window').pack(side='left')
        self.icon(row,'\u21bb','Refresh target windows',self.refresh_windows).pack(side='right')
        self.window_selector = ttk.Combobox(right,state='readonly',width=28)
        self.window_selector.pack(fill='x')
        image_row = ttk.Frame(right)
        image_row.pack(fill='x',pady=(8,0))
        ttk.Label(image_row,text='Model image').pack(side='left')
        self.image_size = tk.StringVar(value='Original' if self.settings.image_max_edge==0 else 'Max 1280px')
        self.image_selector = ttk.Combobox(image_row,textvariable=self.image_size,
            values=('Max 1280px','Original'),state='readonly',width=13)
        self.image_selector.pack(side='right')
        self.image_selector.bind('<<ComboboxSelected>>',self.image_size_changed)
        margin_row = ttk.Frame(right)
        margin_row.pack(fill='x',pady=(6,0))
        ttk.Label(margin_row,text='Capture margin').pack(side='left')
        self.capture_margin = tk.StringVar(value=str(self.settings.capture_margin)+' px')
        self.margin_selector = ttk.Combobox(margin_row,textvariable=self.capture_margin,
            values=('0 px','128 px','256 px','512 px'),state='readonly',width=13)
        self.margin_selector.pack(side='right')
        self.margin_selector.bind('<<ComboboxSelected>>',self.capture_margin_changed)
        self.preview = tk.Label(right,text='No capture',bg=COLORS['editor'],fg=COLORS['muted'],width=30,height=7)
        self.preview.pack(fill='x',pady=8)
        row = ttk.Frame(right)
        row.pack(fill='x')
        ttk.Label(row,text='Captures / videos').pack(side='left')
        self.icon(row,'\u2197','Open image or video',self.open_capture).pack(side='right')
        self.captures = tk.Listbox(right,height=3,exportselection=False,font=('Consolas',9))
        self.captures.pack(fill='x',pady=4)
        self.captures.bind('<Double-Button-1>',lambda event:self.open_capture())
        self.tool_log = ScrolledText(right,width=28,height=7,wrap='word',state='disabled',font=('Consolas',9))
        self.tool_log.pack(fill='both',expand=True)
        self.footer = ttk.Frame(self,padding=(12,4))
        self.footer.pack(fill='x')
        ttk.Label(self.footer,textvariable=self.status).pack(side='left')
        ttk.Checkbutton(self.footer,text='On top',variable=self.topmost,
                        command=lambda:self.attributes('-topmost',self.topmost.get())).pack(side='right')
        self.composer = ttk.Frame(self,padding=10)
        self.composer.pack(fill='x',side='bottom')
        self.toggle = self.icon(self.composer,'\u2199','Compact / expand',self.toggle_compact)
        self.toggle.pack(side='left',padx=(0,6))
        self.attach_button = ttk.Menubutton(self.composer,text='+ 0',width=4)
        self.attach_menu = tk.Menu(self.attach_button,tearoff=False)
        self.attach_button.configure(menu=self.attach_menu)
        self.attach_button.pack(side='left',padx=(0,6))
        self.refresh_attachments()
        self.entry = PromptText(self.composer,self.prompt,font=('Malgun Gothic',12),height=2,wrap='word',
                       padx=8,pady=6,relief='solid',borderwidth=1)
        self.entry.pack(side='left',fill='x',expand=True)
        self.entry.bind('<Return>',self.composer_return)
        self.send_button = self.icon(self.composer,'\u2191','Send task',self.send)
        self.send_button.configure(style='Accent.TButton')
        self.send_button.pack(side='left',padx=6)
        self.icon(self.composer,'\u25a0','Stop (F8)',self.stop).pack(side='left')
        self.approval_frame = ttk.Frame(self,padding=10)
        approval_actions = ttk.Frame(self.approval_frame)
        approval_actions.pack(side='right',fill='y',padx=(8,0))
        self.approval_text = ScrolledText(self.approval_frame,height=7,wrap='word',state='disabled',bg='#302c23',fg=COLORS['warning'],font=('Consolas',10))
        self.approval_text.pack(side='left',fill='both',expand=True)
        ttk.Button(approval_actions,text='Approve once',command=lambda:self.answer(True)).pack(fill='x',pady=2)
        ttk.Button(approval_actions,text='Deny / stop',command=lambda:self.answer(False)).pack(fill='x',pady=2)
        self.icon(approval_actions,'\u2197','\uc2b9\uc778 \ub0b4\uc6a9 \uc804\uccb4 \ubcf4\uae30',self.show_approval).pack(anchor='e',pady=2)
        self.composer.pack_configure(before=self.header)
        self.footer.pack_configure(side='bottom',before=self.header)

    def composer_return(self,event):
        if event.state & 1:
            return None
        self.entry.changed()
        self.send()
        return 'break'

    def toggle_compact(self):
        if not self.compact:
            if self.approval is not None:
                return
            self.full_geometry = self.geometry()
            for frame in (self.header,self.body,self.footer):
                frame.pack_forget()
            self.minsize(440,64)
            self.geometry(f'760x64+{self.winfo_x()}+{self.winfo_y()}')
            self.toggle.configure(text='\u2197')
        else:
            self.minsize(960,640)
            self.geometry(self.full_geometry)
            self.footer.pack(fill='x',side='bottom')
            self.header.pack(fill='x')
            self.body.pack(fill='both',expand=True,padx=10)
            self.toggle.configure(text='\u2199')
        self.compact = not self.compact
        self.entry.focus_set()

    def set_text(self, widget, content):
        widget.configure(state='normal')
        widget.delete('1.0','end')
        widget.insert('end',content)
        widget.configure(state='disabled')

    def refresh_sessions(self):
        self.session_rows = self.store.sessions()
        self.sessions.configure(state='normal')
        self.sessions.delete(0,'end')
        for index,row in enumerate(self.session_rows):
            self.sessions.insert('end',row['title'])
            if row['id'] == self.identifier:
                self.sessions.selection_set(index)
        self.sessions.configure(state='disabled' if self.busy else 'normal')

    def select_session(self, event=None):
        if self.busy or not self.sessions.curselection():
            return
        self.identifier = self.session_rows[self.sessions.curselection()[0]]['id']
        self.attachments.clear()
        self.refresh_attachments()
        self.render()
        self.preview.configure(image='',text='No capture',width=30,height=7)
        self.photo = None
        self.set_text(self.stream,'')

    def new_session(self):
        if not self.busy:
            self.attachments.clear()
            self.refresh_attachments()
            self.identifier = self.store.create()
            self.refresh_sessions()
            self.render()
            self.preview.configure(image='',text='No capture',width=30,height=7)
            self.photo = None

    def rename_session(self):
        if not self.busy:
            title = simpledialog.askstring('Session','Title',parent=self)
            if title and title.strip():
                self.store.rename(self.identifier,title)
                self.refresh_sessions()

    def export_session(self):
        path = filedialog.asksaveasfilename(parent=self,defaultextension='.json',filetypes=[('JSON','*.json')])
        if path:
            self.store.export(self.identifier,path)

    def delete_session(self):
        if self.busy:
            return
        title = next(row['title'] for row in self.session_rows if row['id'] == self.identifier)
        if messagebox.askyesno('\uc138\uc158 \uc0ad\uc81c',
                title+'\n\n\ub300\ud654, \ucea1\ucc98, \uc601\uc0c1, \ucca8\ubd80 \ubcf5\uc0ac\ubcf8, \ube0c\ub77c\uc6b0\uc800 \ud504\ub85c\ud544\uc744 \uc0ad\uc81c\ud569\ub2c8\ub2e4. \ubcf5\uad6c\ud560 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4. \uacc4\uc18d\ud560\uae4c\uc694?',parent=self):
            self.enqueue('delete_session',self.identifier)

    def show_message_menu(self, event_id, event):
        self.message_context.delete(0,'end')
        self.message_context.add_command(label='\uba54\uc2dc\uc9c0 \uc804\uccb4 \ubcf5\uc0ac',command=lambda:self.copy_message(event_id))
        selected = bool(self.transcript.tag_ranges('sel'))
        self.message_context.add_command(label='\uc120\ud0dd \uc601\uc5ed \ubcf5\uc0ac',command=self.copy_selection,
                                         state='normal' if selected else 'disabled')
        if not self.busy:
            self.message_context.add_separator()
            self.message_context.add_command(label='\ud3b8\uc9d1...',command=lambda:self.edit_message(event_id))
            self.message_context.add_command(label='\uc0ad\uc81c...',command=lambda:self.delete_message(event_id))
            self.message_context.add_separator()
            self.message_context.add_command(label='\uc774 \uc694\uccad \uc7ac\uc2e4\ud589...',command=lambda:self.replay_message(event_id))
        try:
            self.message_context.tk_popup(event.x_root,event.y_root)
        finally:
            self.message_context.grab_release()
        return 'break'

    def show_selection_menu(self, event):
        self.message_context.delete(0,'end')
        self.message_context.add_command(label='\uc120\ud0dd \uc601\uc5ed \ubcf5\uc0ac',command=self.copy_selection,
            state='normal' if self.transcript.tag_ranges('sel') else 'disabled')
        try:
            self.message_context.tk_popup(event.x_root,event.y_root)
        finally:
            self.message_context.grab_release()
        return 'break'

    def edit_message(self, event_id):
        if self.busy:
            return
        identifier = self.identifier
        try:
            stored,content = self.store.message(identifier,event_id)
        except ValueError as error:
            messagebox.showerror('Message',str(error),parent=self)
            return
        dialog = tk.Toplevel(self)
        dialog.title('\uba54\uc2dc\uc9c0 \ud3b8\uc9d1')
        dialog.minsize(420,240)
        place_dialog(dialog,self,680,400)
        row = ttk.Frame(dialog,padding=10)
        row.pack(side='bottom',fill='x')
        editor = ScrolledText(dialog,wrap='word',font=('Malgun Gothic',11),padx=12,pady=12)
        editor.pack(fill='both',expand=True,padx=10,pady=10)
        editor.insert('1.0',content)
        def save(replay=False):
            if self.busy or self.identifier != identifier:
                return
            try:
                self.store.edit_message(identifier,event_id,editor.get('1.0','end-1c'))
                dialog.destroy()
                self.render(preserve_scroll=True)
                self.refresh_sessions()
                self.status.set('\uba54\uc2dc\uc9c0 \ud3b8\uc9d1\ub428; \uae30\uc874 \ub3c4\uad6c \uc2e4\ud589 \uacb0\uacfc\ub294 \uc720\uc9c0\ub429\ub2c8\ub2e4')
                if replay:
                    self.replay_message(event_id)
            except Exception as error:
                messagebox.showerror('Message',str(error),parent=self)
        ttk.Button(row,text='\ucde8\uc18c',command=dialog.destroy).pack(side='left')
        ttk.Button(row,text='\ud655\uc778',command=save).pack(side='right')
        if stored['role'] == 'user':
            ttk.Button(row,text='\uc800\uc7a5 \ud6c4 \uc7ac\uc2e4\ud589',command=lambda:save(True)).pack(side='right',padx=8)
        dialog.transient(self)
        dialog.grab_set()
        editor.focus_set()

    def delete_message(self, event_id):
        if self.busy:
            return
        try:
            stored,content = self.store.message(self.identifier,event_id)
            description = ('\uc120\ud0dd\ud55c \uc694\uccad\uacfc \uadf8 \ub2f5\ubcc0\u00b7\ub3c4\uad6c \uae30\ub85d' if stored['role'] == 'user' else '\uc120\ud0dd\ud55c \ub2f5\ubcc0')
            if not messagebox.askyesno('\uba54\uc2dc\uc9c0 \uc0ad\uc81c',description+'\uc744 \uc0ad\uc81c\ud560\uae4c\uc694?\n\uc774\ubbf8 \uc2e4\ud589\ud55c \uc791\uc5c5\uc740 \ub418\ub3cc\ub9ac\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4. \uc800\uc7a5 \ud30c\uc77c\uc740 \uc138\uc158 \uc0ad\uc81c \uc2dc \uc815\ub9ac\ub429\ub2c8\ub2e4.',parent=self):
                return
            self.store.delete_message(self.identifier,event_id)
            self.render(preserve_scroll=True)
            self.refresh_sessions()
        except Exception as error:
            messagebox.showerror('Message',str(error),parent=self)

    def replay_message(self, event_id):
        if self.busy:
            return
        try:
            self.store.replay_request(self.identifier,event_id)
            if not messagebox.askyesno('\uc694\uccad \uc7ac\uc2e4\ud589',
                    '\ud604\uc7ac \uc138\uc158\uc5d0\uc11c \ud574\ub2f9 \uc694\uccad\uacfc \uc774\ud6c4 \ub300\ud654\u00b7\ub3c4\uad6c \uae30\ub85d\uc744 \uc0ad\uc81c\ud55c \ub4a4 \uac19\uc740 \uc694\uccad\uc744 \ub2e4\uc2dc \uc2e4\ud589\ud569\ub2c8\ub2e4. \uc774\uc804 \ub300\ud654\ub294 \uc720\uc9c0\ub418\uba70 \uc0c8 \uc138\uc158\uc744 \ub9cc\ub4e4\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.\n\uc0ad\uc81c\ub41c \ub300\ud654\ub294 \ubcf5\uad6c\ud560 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4. \ud604\uc7ac \uad8c\ud55c\uc73c\ub85c \ud074\ub9ad\u00b7\uc785\ub825 \ub4f1\uc774 \ub2e4\uc2dc \uc2e4\ud589\ub420 \uc218 \uc788\uc73c\uba70 \uc678\ubd80 \uc571 \uc0c1\ud0dc\ub294 \ub418\ub3cc\ub9ac\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4. \uacc4\uc18d\ud560\uae4c\uc694?',parent=self):
                return
            if not self.confirm_api_transfer():
                return
            self.stopped.clear()
            self.enqueue('replay',(self.identifier,event_id,self.permission_snapshot()))
        except Exception as error:
            messagebox.showerror('Replay',str(error),parent=self)

    def open_capture(self):
        if self.captures.curselection():
            path = self.capture_paths[self.captures.curselection()[0]]
            if path.is_file() and path.suffix.lower() in ('.png','.mp4') and self.data.resolve() in path.resolve().parents:
                os.startfile(str(path))

    def refresh_windows(self):
        if not self.busy:
            self.window_choices = list_windows(os.getpid())
            self.window_selector.configure(values=['No target']+[f'{row["title"][:60]} [{row["handle"]}]' for row in self.window_choices])
            self.window_selector.current(0)

    def compaction_changed(self):
        if self.busy:
            return
        self.settings=replace(self.settings,auto_compact=self.auto_compact.get())
        self.settings.save(self.data/'settings.json')
        self.model.settings=self.settings

    def show_context_summary(self):
        from desktop_agent.compaction import restore_summary
        saved=restore_summary(self.store,self.identifier,self.store.events(self.identifier))
        if saved:
            self.show_record({'content':saved['text'],'metadata':{'tool':'\ub300\ud654 \uc694\uc57d'}})
        else:
            self.status.set('\ud604\uc7ac \uc720\ud6a8\ud55c \uc694\uc57d \uc5c6\uc74c')

    def image_size_changed(self,event=None):
        if self.busy:
            return
        self.settings = replace(self.settings,image_max_edge=0 if self.image_size.get()=='Original' else 1280)
        self.settings.save(self.data/'settings.json')
        self.enqueue('image_settings',self.settings)

    def capture_margin_changed(self,event=None):
        if self.busy:
            return
        margin = int(self.capture_margin.get().split()[0])
        if margin and not self.settings.capture_margin:
            if not messagebox.askyesno('Capture surroundings',
                    'This includes nearby windows and other visible screen content. Continue?',parent=self):
                self.capture_margin.set('0 px')
                return
        self.settings = replace(self.settings,capture_margin=margin)
        self.settings.save(self.data/'settings.json')

    def reasoning_changed(self):
        if self.settings.backend == 'api':
            return
        self.settings = replace(self.settings,reasoning_enabled=self.reasoning.get())
        self.settings.save(self.data/'settings.json')
        self.enqueue('reasoning' if self.settings.uses_reasoning_effort else 'settings',self.settings)
        self.status.set('Reasoning changed; applies to next request' if self.settings.uses_reasoning_effort else 'Reasoning changed; model will reload')

    def reasoning_level_changed(self, event=None):
        if self.busy or self.settings.backend == 'api':
            return
        self.settings = self.settings.with_reasoning_level(self.reason_level.get())
        self.reasoning.set(True)
        self.settings.save(self.data/'settings.json')
        self.enqueue('reasoning' if self.settings.uses_reasoning_effort else 'settings',self.settings)
        suffix = 'reasoning_effort' if self.settings.uses_reasoning_effort else 'unlimited'
        self.status.set(f'Reasoning: {self.reason_level.get()} ({suffix})')

    def refresh_attachments(self):
        self.attach_button.configure(text='+ '+str(len(self.attachments)))
        self.attach_menu.delete(0,'end')
        self.attach_menu.add_command(label='Attach files...',command=self.choose_files)
        if self.attachments:
            self.attach_menu.add_separator()
            for path in self.attachments:
                self.attach_menu.add_command(label='Remove: '+Path(path).name,command=lambda path=path:self.remove_attachment(path))

    def remove_attachment(self, path):
        if not self.busy:
            self.attachments.remove(path)
            self.refresh_attachments()

    def choose_files(self):
        if not self.busy:
            self.add_files(filedialog.askopenfilenames(parent=self,title='Attach files'))

    def add_files(self, paths):
        if self.busy:
            self.status.set('Wait for the current task before attaching files')
            return
        from desktop_agent.attachments import MAX_FILE_BYTES
        try:
            combined = list(dict.fromkeys(self.attachments+[str(Path(path).resolve()) for path in paths]))
            if len(combined) > 8 or sum(Path(path).stat().st_size for path in combined) > 200*1024*1024:
                raise ValueError('Maximum 8 files / 200MB per message')
            if any(not Path(path).is_file() or Path(path).stat().st_size > MAX_FILE_BYTES for path in combined):
                raise ValueError('Only files up to 100MB each can be attached')
            self.attachments = combined
            self.refresh_attachments()
            self.status.set(f'{len(combined)} file(s) attached')
        except (ValueError,OSError) as error:
            messagebox.showerror('Attachments',str(error),parent=self)

    def drop_files(self, event):
        self.add_files(self.tk.splitlist(event.data))
        return 'copy'

    def update_backend_display(self):
        remote = self.settings.backend == 'api'
        self.reason_level.set(self.settings.reasoning_level)
        self.reason_selector.configure(values=self.settings.reasoning_levels)
        self.reasoning.set(self.settings.reasoning_enabled)
        label = 'API (estimated)' if remote else 'Local'
        self.context_status.set(f'{label} | {self.settings.context_limit//1024}K')
        self.title('Local Desk | '+(self.settings.api.model+' [API]' if remote else self.settings.local_label))
        self.think_button.configure(state='disabled' if remote or self.busy else 'normal')
        self.reason_selector.configure(state='disabled' if remote or self.busy else 'readonly')
        self.api_button.configure(style='Accent.TButton' if remote else 'TButton')

    def confirm_api_transfer(self):
        if self.settings.backend != 'api':
            return True
        url = endpoint(self.settings.api)
        scope = credential_scope(url)
        consent = (self.identifier,scope,self.settings.api.model)
        if consent in self.api_consents:
            return True
        if not messagebox.askyesno('External API',scope+'\n'+self.settings.api.model+
            '\n\n\uc774 \uc138\uc158\uc758 \ub300\ud654\u00b7\ub3c4\uad6c \ub85c\uadf8\uc640 \ub3c4\uad6c\uac00 \uc77d\uc740 \ucea1\ucc98\u00b7\ucca8\ubd80\u00b7\ucf54\ub4dc \uac80\uc0c9 \ub0b4\uc6a9\uc744 \uc678\ubd80 API\ub85c \uc804\uc1a1\ud569\ub2c8\ub2e4. \uc790\ub3d9 \uc555\ucd95\uc744 \ucf1c\uba74 \ucd94\uac00 \uc694\uc57d \uc694\uccad\uc774 \ubc1c\uc0dd\ud558\uba70 \uc694\uae08\uc774 \uc99d\uac00\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4. \uacc4\uc18d\ud560\uae4c\uc694?',parent=self):
            return False
        self.api_consents.add(consent)
        return True

    def api_keys_dialog(self, config, parent):
        config.validate()
        if config.vertex:
            messagebox.showinfo('Vertex','Select a service-account JSON file in the Vertex connection settings.',parent=parent)
            return
        scope = credential_scope(endpoint(config),config.key_profile_id)
        keys = self.model.vault.load(scope)
        dialog = tk.Toplevel(parent)
        dialog.title('API keys | '+scope)
        place_dialog(dialog,parent,540,310)
        ttk.Label(dialog,text=scope,padding=10).pack(anchor='w')
        listing = tk.Listbox(dialog,height=7,exportselection=False)
        listing.pack(fill='both',expand=True,padx=10)
        def refresh():
            listing.delete(0,'end')
            for index in range(len(keys)):
                listing.insert('end',f'Key {index+1:02d}  ********')
        refresh()
        row = ttk.Frame(dialog,padding=10)
        row.pack(fill='x')
        value = tk.StringVar()
        entry = ttk.Entry(row,textvariable=value,show='*')
        entry.pack(side='left',fill='x',expand=True)
        def add():
            import re
            incoming = [key for key in re.split(r'[\s,;]+',value.get().strip()) if key]
            combined = list(dict.fromkeys(keys+incoming))
            if len(combined) > 20 or any(len(key) > 4096 for key in combined):
                messagebox.showerror('Keys','Maximum 20 keys, 4096 characters each',parent=dialog)
                return
            keys[:] = combined
            value.set('')
            refresh()
        def remove():
            if listing.curselection():
                del keys[listing.curselection()[0]]
                refresh()
        self.icon(row,'+','Add pasted keys',add).pack(side='left',padx=4)
        self.icon(row,'\u2212','Remove selected key',remove).pack(side='left')
        def save():
            try:
                if value.get().strip():
                    add()
                    if value.get().strip():
                        return
                self.model.vault.save(scope,keys)
                value.set('')
                dialog.destroy()
                parent.grab_set()
            except Exception:
                messagebox.showerror('Keys','Could not save encrypted keys for this Windows account',parent=dialog)
        ttk.Button(dialog,text='Save encrypted keys',command=save).pack(anchor='e',padx=10,pady=(0,10))
        def close():
            value.set('')
            dialog.destroy()
            parent.grab_set()
        dialog.protocol('WM_DELETE_WINDOW',close)
        dialog.transient(parent)
        dialog.grab_set()
        entry.focus_set()

    def api_settings_dialog(self):
        if self.busy:
            return
        dialog = tk.Toplevel(self)
        dialog.title('Model backend / API settings')
        dialog.minsize(640,640)
        place_dialog(dialog,self,760,700)
        profile_row = ttk.Frame(dialog,padding=(12,10,12,0))
        profile_row.pack(fill='x')
        ttk.Label(profile_row,text='API profile').pack(side='left',padx=(0,10))
        profile_name = tk.StringVar(value=self.settings.active_api_profile)
        profile_selector = ttk.Combobox(profile_row,textvariable=profile_name,state='readonly',values=list(self.settings.api_profiles))
        profile_selector.pack(side='left',fill='x',expand=True)
        loaded_name = self.settings.active_api_profile
        backend = tk.StringVar(value=self.settings.backend)
        row = ttk.Frame(dialog,padding=12)
        row.pack(fill='x')
        ttk.Radiobutton(row,text='Local llama.cpp',variable=backend,value='local').pack(side='left')
        ttk.Radiobutton(row,text='External API',variable=backend,value='api').pack(side='left',padx=20)
        tabs = ttk.Notebook(dialog)
        tabs.pack(fill='both',expand=True,padx=12)
        connection,generation = ttk.Frame(tabs,padding=12),ttk.Frame(tabs,padding=12)
        tabs.add(connection,text='Connection')
        tabs.add(generation,text='Generation')
        variables = {}
        controls, labels = {}, {}
        pending_vertex = {'info':None,'removed':False}
        credential_id = self.settings.api.vertex_credential_id
        key_profile_id = self.settings.api.key_profile_id
        choices = {'format':FORMATS,'tokenizer':TOKENIZERS,'rotation':('on_failure','round_robin'),
                   'token_limit_field':('auto','max_tokens','max_completion_tokens'),
                   'reasoning_effort':('default','none','minimal','low','medium','high','xhigh'),
                   'thinking_level':('default','minimal','low','medium','high')}
        sections = ((connection,('format','url','model','vertex','project','location','rotation')),
                    (generation,('context_tokens','max_output_tokens','token_limit_field','tokenizer','stream','structured_output',
                     'native_tools','reasoning_effort','thinking_level','thinking_budget','include_thoughts','timeout_seconds','max_retries')))
        for frame,names in sections:
            frame.columnconfigure(1,weight=1)
            for index,name in enumerate(names):
                current = getattr(self.settings.api,name)
                variable = tk.BooleanVar(value=current) if type(current) is bool else tk.StringVar(value=str(current))
                variables[name] = variable
                label = ttk.Label(frame,text=name)
                padding = 3 if frame is generation else 6
                label.grid(row=index,column=0,sticky='w',padx=(0,12),pady=padding)
                labels[name] = label
                if type(current) is bool:
                    widget = ttk.Checkbutton(frame,variable=variable)
                elif name in choices:
                    widget = ttk.Combobox(frame,textvariable=variable,values=choices[name],state='readonly')
                elif name == 'location':
                    widget = ttk.Combobox(frame,textvariable=variable,values=('global','us-central1','us-east5','europe-west1','asia-northeast3','us','eu'))
                elif name == 'project':
                    widget = ttk.Entry(frame,textvariable=variable,state='readonly')
                elif name == 'max_retries':
                    widget = ttk.Spinbox(frame,textvariable=variable,from_=0,to=10)
                else:
                    widget = ttk.Entry(frame,textvariable=variable)
                widget.grid(row=index,column=1,sticky='ew',pady=padding)
                controls[name] = widget
        def config():
            values = {name:variable.get() for name,variable in variables.items()}
            if values['vertex'] and pending_vertex['info'] is None and (not credential_id or pending_vertex['removed']):
                raise ValueError('Select a Vertex service-account JSON key file first')
            for name in ('context_tokens','max_output_tokens','thinking_budget','timeout_seconds','max_retries'):
                values[name] = int(values[name])
            values['vertex_credential_id'] = '' if pending_vertex['removed'] else credential_id
            values['key_profile_id'] = key_profile_id
            values['location'] = values['location'].strip() or 'global'
            if values['format'] == 'openai' and not uses_google_thinking(APISettings(**values)):
                values.update(thinking_level='default',thinking_budget=-1,include_thoughts=False)
            if values['format'] != 'openai':
                values['native_tools'] = False
            return APISettings(**values).validate()
        def keys():
            try:
                self.api_keys_dialog(config(),dialog)
            except Exception as error:
                messagebox.showerror('API keys',str(error),parent=dialog)
        key_button = ttk.Button(connection,text='Manage API keys...',command=keys)
        key_button.grid(row=7,column=1,sticky='w',pady=15)
        vertex_frame = ttk.Frame(connection)
        vertex_frame.grid(row=7,column=0,columnspan=2,sticky='ew',pady=10)
        vertex_frame.columnconfigure(0,weight=1)
        account_status = tk.StringVar(value='No service-account JSON selected')
        if credential_id:
            try:
                saved_account = self.model.vault.load_vertex(credential_id)
                account_status.set('Saved: '+saved_account['client_email'])
                variables['project'].set(saved_account['project_id'])
            except ValueError:
                account_status.set('Saved key unavailable; select the JSON file again')
        def choose_vertex_json():
            path = filedialog.askopenfilename(parent=dialog,title='Select Vertex service-account JSON',
                                             filetypes=[('Service account JSON','*.json')])
            if not path:
                return
            try:
                account = read_vertex_file(path)
            except ValueError as error:
                messagebox.showerror('Vertex JSON',str(error),parent=dialog)
                return
            pending_vertex['info'],pending_vertex['removed'] = account,False
            variables['project'].set(account['project_id'])
            variables['vertex'].set(True)
            backend.set('api')
            account_status.set('Selected: '+account['client_email'])
        def remove_vertex_json():
            pending_vertex['info'],pending_vertex['removed'] = None,True
            variables['project'].set('')
            variables['vertex'].set(False)
            account_status.set('Key will be removed on Save')
        file_row = ttk.Frame(vertex_frame)
        file_row.grid(row=0,column=0,sticky='w')
        ttk.Button(file_row,text='Select JSON key file...',command=choose_vertex_json).pack(side='left')
        self.icon(file_row,'\u2212','Remove saved Vertex key',remove_vertex_json).pack(side='left',padx=6)
        ttk.Label(vertex_frame,textvariable=account_status,wraplength=560).grid(row=1,column=0,sticky='w',pady=(8,0))
        presets = ttk.Frame(connection)
        presets.grid(row=8,column=0,columnspan=2,sticky='ew')
        preset = tk.StringVar(value='Gemini (OpenAI)')
        chooser = ttk.Combobox(presets,textvariable=preset,state='readonly',values=['Gemini (OpenAI)','Gemini native','OpenAI Responses','Anthropic Claude'],width=22)
        chooser.pack(side='left')
        def apply_preset():
            format,url,model = {'Gemini (OpenAI)':('openai',APISettings().url,'gemini-3.8-flash'),
                'Gemini native':('gemini','https://generativelanguage.googleapis.com/v1beta','gemini-3.8-flash'),
                'OpenAI Responses':('responses','https://api.openai.com/v1/responses','gpt-4.1'),
                'Anthropic Claude':('anthropic','https://api.anthropic.com/v1/messages','claude-sonnet-4-6')}[preset.get()]
            for name,value in (('format',format),('url',url),('model',model),('reasoning_effort','default'),('thinking_level','default'),('thinking_budget','-1')):
                variables[name].set(value)
            variables['vertex'].set(False)
        ttk.Button(presets,text='Apply preset',command=apply_preset).pack(side='left',padx=8)
        def vertex_changed(*args):
            active = variables['vertex'].get()
            for name in ('url','rotation'):
                for widget in (labels[name],controls[name]):
                    widget.grid_remove() if active else widget.grid()
            for name in ('project','location'):
                for widget in (labels[name],controls[name]):
                    widget.grid() if active else widget.grid_remove()
            labels['project'].configure(text='Project (from JSON)')
            labels['location'].configure(text='Region (optional)')
            if active:
                key_button.grid_remove()
                presets.grid_remove()
                vertex_frame.grid()
            else:
                vertex_frame.grid_remove()
                key_button.grid()
                presets.grid()
        variables['vertex'].trace_add('write',vertex_changed)
        vertex_changed()
        def thinking_controls_changed(*args):
            draft = APISettings(format=variables['format'].get(),url=variables['url'].get(),vertex=variables['vertex'].get())
            google_only = draft.format == 'openai' and not uses_google_thinking(draft)
            controls['native_tools'].configure(state='normal' if draft.format == 'openai' else 'disabled')
            for name in ('thinking_level','thinking_budget','include_thoughts'):
                controls[name].configure(state='disabled' if google_only else 'readonly' if name in choices else 'normal')
        for name in ('format','url','vertex'):
            variables[name].trace_add('write',thinking_controls_changed)
        thinking_controls_changed()
        def draft_state():
            return (backend.get(),tuple(variable.get() for variable in variables.values()),credential_id,key_profile_id,
                    pending_vertex['info'] is not None,pending_vertex['removed'])
        baseline = draft_state()
        def show_config(api):
            nonlocal credential_id,key_profile_id,baseline
            credential_id,key_profile_id = api.vertex_credential_id,api.key_profile_id
            pending_vertex.update(info=None,removed=False)
            for name,variable in variables.items():
                variable.set(getattr(api,name))
            backend.set('api')
            account_status.set('No service-account JSON selected')
            if credential_id:
                try:
                    account = self.model.vault.load_vertex(credential_id)
                    variables['project'].set(account['project_id'])
                    account_status.set('Saved: '+account['client_email'])
                except ValueError:
                    account_status.set('Saved key unavailable; select the JSON file again')
            vertex_changed()
            baseline = draft_state()
        def load_profile(event=None):
            nonlocal loaded_name
            name = profile_name.get()
            if name not in self.settings.api_profiles:
                return
            if draft_state() != baseline and not messagebox.askyesno('Load profile',
                    '\uc800\uc7a5\ud558\uc9c0 \uc54a\uc740 \uc124\uc815\uc744 \ubc84\ub9ac\uace0 \ud504\ub85c\ud544\uc744 \ubd88\ub7ec\uc62c\uae4c\uc694?',parent=dialog):
                profile_name.set(loaded_name)
                return
            loaded_name = name
            show_config(self.settings.api_profiles[name])
        profile_selector.bind('<<ComboboxSelected>>',load_profile)
        def persist(settings):
            previous = self.settings
            settings.save(self.data/'settings.json')
            self.settings = settings
            try:
                for reference in previous.vertex_credential_references()-settings.vertex_credential_references():
                    self.model.vault.remove_vertex(reference)
                for scope in previous.api_key_references()-settings.api_key_references():
                    self.model.vault.save(scope,[])
            except Exception:
                messagebox.showwarning('Credentials','Settings saved; some unused encrypted credentials could not be removed.',parent=dialog)
        def materialize(copy_keys=False):
            new_reference,new_scope = '', ''
            try:
                api = config()
                if pending_vertex['info'] is not None:
                    new_reference = self.model.vault.save_vertex(pending_vertex['info'])
                    api = replace(api,vertex_credential_id=new_reference)
                if api.vertex:
                    account = self.model.vault.load_vertex(api.vertex_credential_id)
                    if account['project_id'] != api.project:
                        raise ValueError('Project does not match the selected JSON key')
                if copy_keys:
                    identifier = uuid.uuid4().hex
                    if not api.vertex:
                        source_scope = credential_scope(endpoint(api),api.key_profile_id)
                        new_scope = credential_scope(endpoint(api),identifier)
                        self.model.vault.save(new_scope,self.model.vault.load(source_scope))
                    api = replace(api,key_profile_id=identifier)
                return api,new_reference,new_scope
            except Exception:
                rollback(new_reference,new_scope)
                raise
        def rollback(reference, scope):
            try:
                if reference:
                    self.model.vault.remove_vertex(reference)
                if scope:
                    self.model.vault.save(scope,[])
            except Exception:
                messagebox.showwarning('Credentials','An unused encrypted credential could not be removed.',parent=dialog)
        def save_profile(as_new=False):
            nonlocal loaded_name,baseline,credential_id,key_profile_id
            name = loaded_name
            if as_new or not name:
                name = simpledialog.askstring('Save API profile','Profile name',parent=dialog)
                if name is None:
                    return
                name = name.strip()
            if name in self.settings.api_profiles and not messagebox.askyesno('Save API profile',
                    name+'\n\uc774 \ud504\ub85c\ud544\uc744 \ub36e\uc5b4\uc4f8\uae4c\uc694?',parent=dialog):
                return
            reference,scope = '', ''
            try:
                self.settings.with_api_profile(name,config())
                api,reference,scope = materialize(as_new or name != loaded_name or not key_profile_id)
                settings = self.settings.with_api_profile(name,api)
                if settings.active_api_profile == name and settings.api != api:
                    settings = replace(settings,active_api_profile='')
                persist(settings)
            except Exception as error:
                rollback(reference,scope)
                messagebox.showerror('Save API profile',str(error),parent=dialog)
                return
            loaded_name = name
            profile_name.set(name)
            profile_selector.configure(values=list(self.settings.api_profiles))
            credential_id,key_profile_id = api.vertex_credential_id,api.key_profile_id
            pending_vertex.update(info=None,removed=False)
            if credential_id:
                account_status.set('Saved: '+self.model.vault.load_vertex(credential_id)['client_email'])
            baseline = draft_state()
            self.status.set('API profile saved: '+name)
        def delete_profile():
            nonlocal loaded_name,baseline
            name = profile_name.get()
            if name not in self.settings.api_profiles:
                return
            if not messagebox.askyesno('Delete API profile',name+'\n\uc800\uc7a5\ub41c \ud504\ub85c\ud544\uc744 \uc0ad\uc81c\ud560\uae4c\uc694? \ub300\ud654\ub294 \uc0ad\uc81c\ub418\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.',parent=dialog):
                return
            try:
                persist(self.settings.without_api_profile(name))
            except Exception as error:
                messagebox.showerror('Delete API profile',str(error),parent=dialog)
                return
            loaded_name = ''
            profile_name.set('')
            profile_selector.configure(values=list(self.settings.api_profiles))
            show_config(self.settings.api)
            self.status.set('API profile deleted: '+name)
        self.icon(profile_row,'\u2193','Save API profile',save_profile).pack(side='left',padx=(6,0))
        self.icon(profile_row,'+','Save as new API profile',lambda:save_profile(True)).pack(side='left',padx=3)
        self.icon(profile_row,'\u00d7','Delete API profile',delete_profile).pack(side='left')
        row = ttk.Frame(dialog,padding=12)
        row.pack(fill='x')
        ttk.Label(row,text='API: conversation / tool data may leave this computer.',foreground=COLORS['warning']).pack(side='left')
        def save():
            reference,scope = '', ''
            try:
                api,reference,scope = materialize()
                settings = replace(self.settings,backend=backend.get(),api=api,active_api_profile=loaded_name)
                if loaded_name:
                    settings = settings.with_api_profile(loaded_name,api)
                persist(settings)
            except Exception as error:
                rollback(reference,scope)
                messagebox.showerror('API settings',str(error),parent=dialog)
                return
            self.api_consents.clear()
            self.update_backend_display()
            self.enqueue('settings',settings)
            pending_vertex['info'] = None
            dialog.destroy()
        ttk.Button(row,text='Save',command=save).pack(side='right')
        dialog.bind('<Destroy>',lambda event:pending_vertex.update(info=None) if event.widget is dialog else None)
        dialog.transient(self)
        dialog.grab_set()

    def settings_dialog(self):
        if self.busy:
            return
        dialog = tk.Toplevel(self)
        dialog.title('Local model settings')
        dialog.minsize(740,500)
        place_dialog(dialog,self,940,550)
        dialog.columnconfigure(1,weight=1)
        dialog.rowconfigure(9,weight=1)
        preset_names = {preset['label']:name for name,preset in MODEL_PRESETS.items()}
        selection = tk.StringVar(value='Custom paths')
        ttk.Label(dialog,text='Model preset').grid(row=0,column=0,padx=8,pady=7,sticky='w')
        selector = ttk.Combobox(dialog,textvariable=selection,values=['Custom paths',*preset_names],state='readonly')
        selector.grid(row=0,column=1,padx=8,pady=7,sticky='ew')
        fields = {}
        effort = tk.StringVar(value=self.settings.reasoning_effort)
        effort_control = ttk.Combobox(dialog,textvariable=effort,values=('low','medium','xhigh'),state='readonly')
        unlimited_control = ttk.Label(dialog,text='Unlimited')
        budget_label = budget_control = None
        for index,name in enumerate(('executable','model','projector','reasoning_tokens','max_steps','context_tokens'),1):
            label = ttk.Label(dialog,text=name)
            label.grid(row=index,column=0,padx=8,pady=7,sticky='w')
            variable = tk.StringVar(value=str(getattr(self.settings,name)))
            if name == 'context_tokens':
                ttk.Combobox(dialog,textvariable=variable,values=[4096,8192,12288,16384,24576,32768,65536]).grid(row=index,column=1,padx=8,sticky='ew')
            else:
                control = ttk.Entry(dialog,textvariable=variable)
                control.grid(row=index,column=1,padx=8,sticky='ew')
                if name == 'reasoning_tokens':
                    budget_label,budget_control = label,control
            fields[name] = variable
        cache_enabled = tk.BooleanVar(value=self.settings.cache_ram_mib == 2048)
        ttk.Checkbutton(dialog,text='RAM prompt cache (2 GiB)',variable=cache_enabled).grid(
            row=7,column=1,padx=8,pady=7,sticky='w')
        kv_choices={'Preset default':'default','4-bit (Q4_0)':'q4_0','8-bit (Q8_0)':'q8_0','16-bit (F16)':'f16'}
        kv_value=tk.StringVar(value=next(label for label,value in kv_choices.items() if value==self.settings.kv_cache_type))
        ttk.Label(dialog,text='KV cache (K / V)').grid(row=8,column=0,padx=8,pady=5,sticky='w')
        ttk.Combobox(dialog,textvariable=kv_value,values=list(kv_choices),state='readonly').grid(row=8,column=1,padx=8,sticky='ew')
        ttk.Label(dialog,text='Model options').grid(row=9,column=0,padx=8,pady=7,sticky='nw')
        preview = ScrolledText(dialog,height=5,wrap='word',font=('Consolas',10),state='disabled')
        preview.grid(row=9,column=1,padx=8,pady=7,sticky='nsew')
        def draft():
            values = {key:variable.get().strip() for key,variable in fields.items()}
            for key in ('reasoning_tokens','max_steps','context_tokens'):
                values[key] = int(values[key])
            validate_context(values['context_tokens'])
            if not 64 <= values['reasoning_tokens'] <= 1024 or not 1 <= values['max_steps'] <= 100:
                raise ValueError('Reasoning budget 64..1024; steps 1..100')
            if effort.get() not in ('low','medium','xhigh'):
                raise ValueError('Reasoning effort must be low, medium or xhigh')
            return replace(self.settings,backend='local',reasoning_effort=effort.get(),
                           cache_ram_mib=2048 if cache_enabled.get() else 0,kv_cache_type=kv_choices[kv_value.get()],**values)
        def options(settings):
            server = DesktopServer()
            server.context_tokens = settings.context_tokens
            server.cache_ram_mib = settings.cache_ram_mib
            server.kv_cache_type = settings.kv_cache_type
            minimum,maximum = settings.local_image_tokens
            result = list(server.settings_for(settings.executable,settings.model,settings.projector,minimum,maximum,
                                             settings.reasoning_enabled,settings.reasoning_tokens)[-1])
            if '-ngl' not in result:
                result = ['-ngl','all',*result]
            if not settings.reasoning_enabled and not settings.uses_reasoning_effort:
                result += ['--reasoning','off','--reasoning-budget','0']
            return result+['--image-min-tokens',str(minimum),'--image-max-tokens',str(maximum)]
        def refresh(*args):
            preset = model_preset(fields['model'].get())
            selection.set(preset['label'] if preset else 'Custom paths')
            use_effort = replace(self.settings,model=fields['model'].get()).uses_reasoning_effort
            budget_label.configure(text='reasoning_effort' if use_effort else 'Reasoning')
            budget_control.grid_remove()
            if use_effort:
                unlimited_control.grid_remove()
                effort_control.grid(row=4,column=1,padx=8,sticky='ew')
            else:
                effort_control.grid_remove()
                unlimited_control.grid(row=4,column=1,padx=8,sticky='ew')
            try:
                current = draft()
                text = ' '.join(options(current))
                if use_effort:
                    text += '\nRequest: reasoning_effort='+('"'+effort.get()+'"' if current.reasoning_enabled else '"none"')
            except (ValueError,OSError) as error:
                text = str(error)
            self.set_text(preview,text)
        def select(event=None):
            name = preset_names.get(selection.get())
            if name is None:
                return
            current = replace(self.settings,model=fields['model'].get(),executable=fields['executable'].get())
            try:
                try:
                    chosen = current.with_local_preset(name)
                except FileNotFoundError:
                    directory = filedialog.askdirectory(parent=dialog,title='Select model and projector folder',initialdir=str(Path(current.model).parent))
                    if not directory:
                        refresh()
                        return
                    chosen = current.with_local_preset(name,directory)
                for key in ('model','projector','context_tokens'):
                    fields[key].set(str(getattr(chosen,key)))
            except (ValueError,OSError) as error:
                messagebox.showerror('Model preset',str(error),parent=dialog)
            refresh()
        selector.bind('<<ComboboxSelected>>',select)
        for variable in fields.values():
            variable.trace_add('write',refresh)
        effort.trace_add('write',refresh)
        cache_enabled.trace_add('write',refresh)
        kv_value.trace_add('write',refresh)
        refresh()
        def save():
            try:
                settings = draft()
                options(settings)
                for key in ('executable','model','projector'):
                    if not Path(getattr(settings,key)).is_file():
                        raise ValueError('File not found: '+getattr(settings,key))
                if settings.context_tokens > max(8192,self.settings.context_tokens):
                    if not messagebox.askyesno('Context memory',
                            '\ucee8\ud14d\uc2a4\ud2b8\ub97c \ub298\ub9ac\uba74 VRAM/RAM \uc0ac\uc6a9\ub7c9\uc774 \uc99d\uac00\ud569\ub2c8\ub2e4. 8GB GPU\uc5d0\uc11c \ub85c\ub4dc \uc2e4\ud328\ub098 \uc18d\ub3c4 \uc800\ud558\uac00 \uc0dd\uae38 \uc218 \uc788\uc2b5\ub2c8\ub2e4. \uc801\uc6a9\ud560\uae4c\uc694?',parent=dialog):
                        return
                if settings.cache_ram_mib and not self.settings.cache_ram_mib:
                    if not messagebox.askyesno('RAM prompt cache',
                            '\uc7ac\ubc29\ubb38\ud560 \ubb38\ub9e5 \uc0c1\ud0dc\ub97c \ubaa8\ub378\uc744 \ub0b4\ub9b4 \ub54c\uae4c\uc9c0 RAM\uc5d0 \ubcf4\uad00\ud569\ub2c8\ub2e4.\n'
                            '\uc2e4\ud5d8\uc5d0\uc11c \ucd94\uac00 RAM\uc740 \ud3c9\uade0 \uc57d 1.2 GiB, \ud53c\ud06c \uc57d 1.7 GiB\uc600\uc73c\uba70 \uc791\uc5c5\uc5d0 \ub530\ub77c \ub2ec\ub77c\uc9d1\ub2c8\ub2e4.\n'
                            '2 GiB\ub294 \uce90\uc2dc \uc6a9\ub7c9\uc774\uba70 \uc804\uccb4 RAM \uc0ac\uc6a9\ub7c9 \uc0c1\ud55c\uc740 \uc544\ub2d9\ub2c8\ub2e4. \ud65c\uc131\ud654\ud560\uae4c\uc694?',parent=dialog):
                        return
                if settings.kv_cache_type!=self.settings.kv_cache_type and settings.kv_cache_type in ('q4_0','f16'):
                    if not messagebox.askyesno('KV cache',
                            'Q4 may reduce quality; F16 increases VRAM use. Q2 dynamic residency is available only with Q8. Apply and unload?',parent=dialog):
                        return
                settings.save(self.data/'settings.json')
                self.settings = settings
                self.update_backend_display()
                self.enqueue('settings',self.settings)
                dialog.destroy()
            except Exception as error:
                messagebox.showerror('Settings',str(error),parent=dialog)
        buttons = ttk.Frame(dialog)
        buttons.grid(row=10,column=0,columnspan=2,sticky='ew',padx=8,pady=8)
        ttk.Button(buttons,text='Cancel',command=dialog.destroy).pack(side='left')
        ttk.Button(buttons,text='Save and unload',command=save).pack(side='right')
        dialog.transient(self)
        dialog.grab_set()

    def tools_dialog(self):
        if self.busy:
            return
        from desktop_agent.workspace_tools import Workspace
        dialog = tk.Toplevel(self)
        dialog.title('\ub3c4\uad6c \uc120\ud0dd')
        dialog.minsize(620,480)
        place_dialog(dialog,self,800,650)
        dialog.transient(self)
        policies = {name:tool_policy(self.settings.tool_policies,name) for name in TOOLS if name != 'finish'}
        labels = {'disabled':'\uc0ac\uc6a9 \uc548 \ud568','ask':'\ub9e4\ubc88 \ud655\uc778','allow':'\ud5c8\uc6a9'}
        root_value = tk.StringVar(value=self.settings.workspace_root)
        search = tk.StringVar()
        selected_policy = tk.StringVar(value=labels['disabled'])
        folder_row = ttk.Frame(dialog,padding=12)
        folder_row.pack(fill='x')
        ttk.Label(folder_row,text='\uc791\uc5c5 \ud3f4\ub354').pack(side='left',padx=(0,8))
        ttk.Entry(folder_row,textvariable=root_value).pack(side='left',fill='x',expand=True)
        def choose_folder():
            value = filedialog.askdirectory(parent=dialog,initialdir=root_value.get() or str(Path.home()))
            if value:
                root_value.set(value)
        self.icon(folder_row,'\u2026','\uc791\uc5c5 \ud3f4\ub354 \uc120\ud0dd',choose_folder).pack(side='left',padx=(6,0))
        search_row = ttk.Frame(dialog,padding=(12,0,12,10))
        search_row.pack(fill='x')
        ttk.Label(search_row,text='\u2315').pack(side='left',padx=(0,8))
        search_entry = ttk.Entry(search_row,textvariable=search)
        search_entry.pack(fill='x',expand=True)
        bottom = ttk.Frame(dialog,padding=12)
        bottom.pack(side='bottom',fill='x')
        ttk.Button(bottom,text='\ucde8\uc18c',command=dialog.destroy).pack(side='left')
        editor = ttk.Frame(dialog,padding=(12,8))
        editor.pack(side='bottom',fill='x')
        ttk.Label(editor,text='\uc120\ud0dd\ud55c \ub3c4\uad6c').pack(side='left')
        selector = ttk.Combobox(editor,textvariable=selected_policy,values=tuple(labels.values()),state='readonly',width=16)
        selector.pack(side='right')
        warning = ttk.Label(dialog,text='\ud30c\uc77c \ubcc0\uacbd\u00b7\uc170 \uc2e4\ud589\uc740 \ud56d\uc0c1 \ud655\uc778\ud569\ub2c8\ub2e4. PowerShell\uc740 \uc791\uc5c5 \ud3f4\ub354 \ubc16\uc5d0\ub3c4 \uc601\ud5a5\uc744 \uc904 \uc218 \uc788\uc2b5\ub2c8\ub2e4.',
                            foreground=COLORS['warning'],wraplength=570,padding=(12,4))
        warning.pack(side='bottom',fill='x')
        holder = ttk.Frame(dialog,padding=(12,0))
        holder.pack(fill='both',expand=True)
        tree = ttk.Treeview(holder,columns=('policy',),selectmode='extended')
        tree.heading('#0',text='\ub3c4\uad6c')
        tree.heading('policy',text='\uc2b9\uc778')
        tree.column('#0',width=390,minwidth=280)
        tree.column('policy',width=160,minwidth=130,stretch=False)
        scroll = ttk.Scrollbar(holder,orient='vertical',command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y')
        tree.pack(fill='both',expand=True)
        membership = {name:group for group,(_,names) in TOOL_GROUPS.items() for name in names}
        group_labels = {'base':'\uae30\ubcf8','browser':'\ube0c\ub77c\uc6b0\uc800','recording':'\ub179\ud654\u00b7\uc791\uc5c5',
                        'advanced_input':'\uace0\uae09 \uc785\ub825','files':'\ucca8\ubd80\u00b7\uae30\ub85d','workspace':'\uc791\uc5c5 \ud3f4\ub354','terminal':'PowerShell'}
        def render():
            selection = tree.selection()
            tree.delete(*tree.get_children())
            needle = search.get().strip().casefold()
            for group in ('base',*TOOL_GROUPS):
                names = [name for name in policies if membership.get(name,'base') == group
                         and (not needle or needle in (name+' '+group_labels.get(group,group)+' '+TOOLS[name][0]).casefold())]
                if not names:
                    continue
                parent = 'group:'+group
                tree.insert('','end',iid=parent,text=group_labels.get(group,group),open=True)
                for name in names:
                    policy = policies[name]
                    mandatory = name in ('workspace_apply_patch','terminal_start') and policy != 'disabled'
                    tree.insert(parent,'end',iid=name,text=('\u2610 ' if policy=='disabled' else '\u2611 ')+name,
                                values=('\ub9e4\ubc88 \ud655\uc778 (\ud544\uc218)' if mandatory else labels[policy],))
            tree.selection_set([name for name in selection if tree.exists(name)])
        def selected_names():
            names = set()
            for name in tree.selection():
                names.update(tree.get_children(name) if name.startswith('group:') else (name,))
            return names
        def change(event=None):
            policy = next(key for key,label in labels.items() if label == selected_policy.get())
            for name in selected_names():
                policies[name] = policy
            render()
        def toggle(event=None):
            names = selected_names()
            enabled = any(policies[name]=='disabled' for name in names)
            for name in names:
                policies[name] = 'ask' if enabled else 'disabled'
            render()
            return 'break'
        def selection(event=None):
            names = selected_names()
            if names:
                selected_policy.set(labels[policies[sorted(names)[0]]])
        def click(event):
            name = tree.identify_row(event.y)
            if name and not name.startswith('group:') and tree.identify_column(event.x)=='#0':
                bounds = tree.bbox(name,'#0')
                if bounds and bounds[0]+20 <= event.x <= bounds[0]+48:
                    tree.selection_set(name)
                    return toggle()
        def save():
            try:
                root = root_value.get().strip()
                if root:
                    root = str(Workspace(root,self.store.artifact_directory(self.identifier)).root)
                if any(policies[name]!='disabled' for name in WORKSPACE_TOOLS) and not root:
                    raise ValueError('Select a workspace folder before enabling file or shell tools')
                if policies['terminal_start']!='disabled' and any(policies[name]=='disabled' for name in ('terminal_output','terminal_stop')):
                    raise ValueError('Enable terminal_output and terminal_stop together with terminal_start')
                settings = replace(self.settings,workspace_root=root,tool_policies=dict(policies))
                settings.save(self.data/'settings.json')
                self.settings = settings
                self.model.settings = settings
                self.status.set('\ub3c4\uad6c \uad8c\ud55c\uc744 \uc800\uc7a5\ud588\uc2b5\ub2c8\ub2e4')
                dialog.destroy()
            except (ValueError,OSError) as error:
                messagebox.showerror('\ub3c4\uad6c \uc124\uc815',str(error),parent=dialog)
        ttk.Button(bottom,text='\uc800\uc7a5',style='Accent.TButton',command=save).pack(side='right')
        selector.bind('<<ComboboxSelected>>',change)
        tree.bind('<<TreeviewSelect>>',selection)
        tree.bind('<space>',toggle)
        tree.bind('<Button-1>',click)
        search.trace_add('write',lambda *args:render())
        render()
        search_entry.focus_set()
        dialog.grab_set()
        return dialog

    def execution_dialog(self):
        from desktop_agent.terminals import OUTPUT_LIMIT,Terminals
        from desktop_agent.workspace_tools import Workspace
        identifier = self.identifier
        previous = self.execution_windows.get(identifier)
        if previous and previous.winfo_exists():
            previous.lift()
            return previous
        directory = self.store.artifact_directory(identifier)
        archived = Terminals(directory,threading.Event(),lambda *args:None)
        dialog = tk.Toplevel(self)
        self.execution_windows[identifier] = dialog
        dialog.title('Local Desk | \ud130\ubbf8\ub110 / \ubcc0\uacbd \ud30c\uc77c')
        dialog.minsize(620,420)
        place_dialog(dialog,self,940,620)
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill='both',expand=True,padx=8,pady=8)
        terminal_tab,changes_tab = ttk.Frame(notebook),ttk.Frame(notebook)
        notebook.add(terminal_tab,text='PowerShell')
        notebook.add(changes_tab,text='\ubcc0\uacbd \ud30c\uc77c')
        terminal_row = ttk.Frame(terminal_tab,padding=6)
        terminal_row.pack(fill='x')
        selected = tk.StringVar()
        picker = ttk.Combobox(terminal_row,textvariable=selected,state='readonly')
        picker.pack(side='left',fill='x',expand=True)
        offset = [0]
        follow = tk.BooleanVar(value=True)
        status = tk.StringVar(value='\uc2e4\ud589 \uae30\ub85d \uc5c6\uc74c')
        paging = ttk.Frame(terminal_tab,padding=6)
        paging.pack(side='bottom',fill='x')
        metadata = tk.StringVar()
        metadata_label = ttk.Label(terminal_tab,textvariable=metadata,wraplength=820,padding=8)
        metadata_label.pack(fill='x')
        metadata_label.bind('<Configure>',lambda event:metadata_label.configure(wraplength=max(200,event.width-16)))
        output = ScrolledText(terminal_tab,wrap='word',font=('Consolas',10),state='disabled',padx=10,pady=8)
        output.pack(fill='both',expand=True)
        identifiers = {}
        def manager():
            if self.active_tools is not None and self.active_tools_session == identifier:
                return self.active_tools.terminals
            return archived
        def stop_selected():
            execution = identifiers.get(selected.get())
            if execution:
                try:
                    manager().stop(execution)
                    refresh_output()
                except (ValueError,OSError) as error:
                    messagebox.showerror('PowerShell',str(error),parent=dialog)
        stop_button = self.icon(terminal_row,'\u25a0','\uc120\ud0dd\ud55c \uc2e4\ud589 \uc911\ub2e8',stop_selected)
        stop_button.pack(side='left',padx=6)
        def refresh_output():
            execution = identifiers.get(selected.get())
            if not execution:
                stop_button.configure(state='disabled')
                return
            try:
                current = manager().output(execution,offset=offset[0],limit=12000)
                if follow.get():
                    offset[0] = max(0,current['output_bytes']-12000)
                    current = manager().output(execution,offset=offset[0],limit=12000)
                text = display_excerpt(current['text'])
                if output.get('1.0','end-1c') != text:
                    self.set_text(output,text)
                    if follow.get():
                        output.see('end')
                metadata.set(f"PID {current['pid']} | cwd: {current['cwd']}\n{display_excerpt(current['command'],1800)}")
                status.set(f"{current['status']} | exit {current['exit_code']} | bytes {offset[0]}..{current['next_offset']}")
                live = self.active_tools_session == identifier and current['status']=='running'
                stop_button.configure(state='normal' if live else 'disabled')
            except (ValueError,OSError,KeyError) as error:
                status.set(str(error))
        def page(amount):
            follow.set(False)
            offset[0] = max(0,min(OUTPUT_LIMIT-1,offset[0]+amount*12000))
            refresh_output()
        self.icon(paging,'\u2190','\uc774\uc804 \ucd9c\ub825',lambda:page(-1)).pack(side='left')
        self.icon(paging,'\u2192','\ub2e4\uc74c \ucd9c\ub825',lambda:page(1)).pack(side='left',padx=4)
        ttk.Label(paging,textvariable=status).pack(side='left',padx=8)
        ttk.Checkbutton(paging,text='\ucd5c\uc2e0 \ucd9c\ub825',variable=follow,command=refresh_output).pack(side='right')
        def changed_selection(event=None):
            offset[0] = 0
            refresh_output()
        picker.bind('<<ComboboxSelected>>',changed_selection)
        change_records = {}
        change_versions = {}
        changes = ttk.Treeview(changes_tab,columns=('status','path'),show='headings',height=5,selectmode='browse')
        changes.heading('status',text='\uc0c1\ud0dc')
        changes.heading('path',text='\ud30c\uc77c')
        changes.column('status',width=100,stretch=False)
        changes.column('path',width=600)
        changes.pack(fill='x',padx=6,pady=6)
        actions = ttk.Frame(changes_tab,padding=6)
        actions.pack(side='bottom',fill='x')
        diff_view = ScrolledText(changes_tab,wrap='none',font=('Consolas',10),state='disabled',padx=10,pady=8)
        diff_view.pack(fill='both',expand=True)
        def selected_change():
            selection = changes.selection()
            return change_records.get(selection[0]) if selection else None
        def show_diff(event=None):
            record = selected_change()
            if record:
                self.set_text(diff_view,display_excerpt(record.get('diff',''),12000))
        def open_change(which):
            record = selected_change()
            if record is None:
                return
            try:
                if which == 'current':
                    workspace = Workspace(record['workspace_root'],directory)
                    relative = str(Path(record['path']).relative_to(workspace.root))
                    _,text = workspace.contents(workspace.path(relative))
                elif which == 'diff':
                    text = record.get('diff','')
                else:
                    path = directory/'file-changes'/record['change_id']/which
                    text = path.read_text(encoding='utf-8') if path.exists() else '(new file: no original)'
                self.show_record({'content':text,'metadata':{'tool':which+' | '+record['path']}})
            except (ValueError,OSError) as error:
                messagebox.showerror('\ubcc0\uacbd \ud30c\uc77c',str(error),parent=dialog)
        for label,which in (('\u2197 \ud604\uc7ac \ud30c\uc77c','current'),('\ubcc0\uacbd \uc804','before.txt'),
                            ('\ubcc0\uacbd \ud6c4','after.txt'),('\uc804\uccb4 \ucc28\uc774','diff')):
            ttk.Button(actions,text=label,command=lambda which=which:open_change(which)).pack(side='left',padx=3)
        changes.bind('<<TreeviewSelect>>',show_diff)
        def tick():
            if not dialog.winfo_exists():
                return
            for path in sorted((directory/'terminals').glob('*.json')):
                if path.stem in identifiers.values():
                    continue
                try:
                    info = json.loads(path.read_text(encoding='utf-8'))
                    label = datetime.fromtimestamp(info['started_at']).strftime('%H:%M:%S')+' | '+info['command'].replace('\n',' ')[:65]+' | '+path.stem[:8]
                    identifiers[label] = path.stem
                except (ValueError,OSError,KeyError):
                    continue
            picker.configure(values=tuple(identifiers))
            if not selected.get() and identifiers:
                selected.set(next(reversed(identifiers)))
            refresh_output()
            for path in sorted((directory/'file-changes').glob('*/change.json')):
                try:
                    version = (path.stat().st_mtime_ns,path.stat().st_size)
                    if change_versions.get(path) == version:
                        continue
                    record = json.loads(path.read_text(encoding='utf-8'))
                    key = path.parent.name
                    if record.get('change_id') != key:
                        continue
                    change_records[key] = record
                    change_versions[path] = version
                    values = (record['status'],record['path'])
                    if changes.exists(key):
                        changes.item(key,values=values)
                    else:
                        changes.insert('','end',iid=key,values=values)
                except (ValueError,OSError,KeyError):
                    continue
            if not changes.selection() and change_records:
                changes.selection_set(next(reversed(change_records)))
            dialog.after(750,tick)
        tick()
        return dialog

    def set_busy(self, value):
        self.busy = value
        self.monitor.armed = value
        for widget in (self.entry,self.send_button,self.load_button,self.unload_button,self.api_button,self.think_button,self.attach_button,self.delete_session_button,self.tool_picker_button,*self.permissions):
            widget.configure(state='disabled' if value else 'normal')
        self.reason_selector.configure(state='disabled' if value else 'readonly')
        self.image_selector.configure(state='disabled' if value else 'readonly')
        self.margin_selector.configure(state='disabled' if value else 'readonly')
        self.window_selector.configure(state='disabled' if value else 'readonly')
        self.sessions.configure(state='disabled' if value else 'normal')
        if self.settings.backend == 'api':
            self.think_button.configure(state='disabled')
            self.reason_selector.configure(state='disabled')

    def enqueue(self, kind, value):
        self.pending_jobs += 1
        self.set_busy(True)
        self.jobs.put((kind,value))

    def permission_snapshot(self):
        index = self.window_selector.current()-1
        window = self.window_choices[index] if 0 <= index < len(self.window_choices) else None
        return dict(allow_screen=self.screen.get(),allow_input=self.inputs.get(),allow_browser=self.browser.get(),
                    mode='routine' if self.automatic.get() else 'manual',window=window,capture_margin=self.settings.capture_margin,
                    workspace_root=self.settings.workspace_root,tool_policies=dict(self.settings.tool_policies))

    def send(self):
        prompt = self.prompt.get().strip()
        if self.busy or (not prompt and not self.attachments):
            return
        if len(prompt) > 12000:
            messagebox.showerror('Message','Maximum 12000 characters per task.',parent=self)
            return
        if not self.confirm_api_transfer():
            return
        permissions = self.permission_snapshot()
        if not self.store.events(self.identifier):
            self.store.rename(self.identifier,prompt[:60] or Path(self.attachments[0]).name)
        self.stopped.clear()
        self.set_busy(True)
        self.live_reasoning = None
        self.prompt.set('')
        attachments,self.attachments = self.attachments,[]
        self.refresh_attachments()
        self.enqueue('task',(self.identifier,prompt,permissions,attachments))

    def queue_model(self, operation):
        if not self.busy:
            self.stopped.clear()
            self.set_busy(True)
            self.enqueue(operation,None)

    def stop(self, reason='Stop button'):
        self.stopped.set(reason)
        self.answer(False)
        self.model.cancel()
        self.status.set('Stopped: '+self.stopped.reason)

    def approve(self, action, reason, context, stopped=None):
        stopped = stopped or self.stopped
        while not stopped.is_set():
            if self.approval_lock.acquire(timeout=0.1):
                break
        else:
            return False
        try:
            if stopped.is_set():
                return False
            return self.approve_serial(action,reason,context,stopped)
        finally:
            self.approval_lock.release()

    def approve_serial(self, action, reason, context, stopped):
        request = dict(action=action,reason=reason,context=context,event=threading.Event(),allowed=False)
        self.events.put(('approval',request))
        while not request['event'].wait(0.1):
            if stopped.is_set():
                request['event'].set()
                return False
        return request['allowed'] and not stopped.is_set()

    def show_approval(self):
        if self.approval is not None:
            context = self.approval['context']
            text = context.get('diff') or json.dumps(dict(action=self.approval['action'],context=context),ensure_ascii=False,indent=2)
            self.show_record({'content':text,'metadata':{'tool':'Approval | '+self.approval['action']['tool']}})

    def answer(self, allowed):
        if self.approval is not None:
            self.approval['allowed'] = allowed
            self.approval['event'].set()
            self.approval = None
            self.approval_frame.pack_forget()

    def work(self):
        tools, tools_session = None, None
        try:
            while True:
                kind,value = self.jobs.get()
                if kind == 'close':
                    break
                try:
                    if kind in ('task','replay'):
                        if kind == 'replay':
                            source,event_id,permissions = value
                            if self.stopped.is_set():
                                continue
                            identifier,prompt,attachments = self.store.prepare_replay(source,event_id)
                            self.events.put(('session_ready',identifier))
                        else:
                            identifier,prompt,permissions,attachments = value
                        def notify(event,payload,session_id=identifier):
                            if event == 'approval_log':
                                self.store.append(session_id,'system',json.dumps(payload),status='approval')
                            elif event in ('terminal','file_change'):
                                self.store.append(session_id,'system',json.dumps(payload,ensure_ascii=False),status=event,
                                                  call_id=payload.get('call_id',''),tool='terminal_start' if event=='terminal' else 'workspace_apply_patch')
                                payload = dict(payload,session_id=session_id)
                            self.events.put((event,payload))
                        if tools_session != identifier:
                            if tools is not None:
                                tools.close()
                            tools = ToolRunner(self.store.artifact_directory(identifier),self.stopped,self.approve,notify,
                                               approve_cancellable=self.approve)
                            tools_session = identifier
                            self.active_tools,self.active_tools_session = tools,identifier
                        tools.notify = notify
                        for key,setting in permissions.items():
                            setattr(tools,key,setting)
                        Agent(self.store,self.model,tools,self.stopped,notify).run(identifier,prompt,attachments)
                    elif kind == 'delete_session':
                        if tools is not None and tools_session == value:
                            tools.close()
                            tools,tools_session = None,None
                            self.active_tools,self.active_tools_session = None,None
                        self.model.close()
                        self.store.delete_session(value)
                        rows = self.store.sessions()
                        self.events.put(('session_ready',rows[0]['id'] if rows else self.store.create()))
                        self.events.put(('status','Session and its files deleted'))
                    elif kind == 'load':
                        self.events.put(('status','Loading model'))
                        endpoint = self.model.ensure(self.stopped)
                        self.events.put(('status',('API configured (connection not tested): ' if self.settings.backend == 'api' else 'Model ready: ')+endpoint))
                    elif kind in ('unload','settings','reasoning','image_settings'):
                        if kind not in ('reasoning','image_settings'):
                            self.model.close()
                        if kind in ('settings','reasoning','image_settings'):
                            self.model.settings = value
                        if tools is not None and kind == 'unload':
                            tools.close()
                            tools,tools_session = None,None
                            self.active_tools,self.active_tools_session = None,None
                        self.events.put(('status','Image size updated' if kind=='image_settings' else 'Reasoning updated for next request' if kind == 'reasoning' else 'Model unloaded'))
                except Exception as error:
                    self.events.put(('status',str(error)))
                    if kind in ('delete_session','replay'):
                        self.events.put(('operation_error',str(error)))
                finally:
                    self.events.put(('worker_idle',None))
        finally:
            try:
                if tools is not None:
                    tools.close()
            finally:
                self.active_tools,self.active_tools_session = None,None
                self.model.close()

    def copy_answer(self):
        for item in reversed(conversation_items(self.store.events(self.identifier))):
            if item['role'] == 'assistant':
                self.clipboard_clear()
                self.clipboard_append(item['content'])
                self.status.set('\ub2f5\ubcc0\uc744 \ubcf5\uc0ac\ud588\uc2b5\ub2c8\ub2e4')
                return

    def toggle_activity(self, key):
        if key in self.expanded_activity:
            self.expanded_activity.remove(key)
        else:
            self.expanded_activity.add(key)
        self.render(preserve_scroll=True)

    def insert_markdown(self, content):
        blocks, lists, styles = [], [], []
        for token in self.markdown.parse(content):
            if token.type in ('heading_open','blockquote_open'):
                blocks.append('heading' if token.type == 'heading_open' else 'quote')
            elif token.type in ('heading_close','blockquote_close'):
                if blocks:
                    blocks.pop()
            elif token.type in ('bullet_list_open','ordered_list_open'):
                lists.append(int(token.attrGet('start') or 1) if token.type == 'ordered_list_open' else None)
            elif token.type in ('bullet_list_close','ordered_list_close'):
                lists.pop()
            elif token.type == 'list_item_open':
                marker = '\u2022 ' if lists[-1] is None else str(lists[-1])+'. '
                self.transcript.insert('end','  '*(len(lists)-1)+marker,('assistant',))
                if lists[-1] is not None:
                    lists[-1] += 1
            elif token.type == 'inline':
                for child in token.children or []:
                    if child.type in ('strong_open','em_open','link_open'):
                        styles.append(child.type.removesuffix('_open'))
                    elif child.type in ('strong_close','em_close','link_close'):
                        if styles:
                            styles.pop()
                    elif child.type in ('softbreak','hardbreak'):
                        self.transcript.insert('end','\n',('assistant',*blocks,*styles))
                    else:
                        tags = ('assistant',*blocks,*styles,*(('code',) if child.type == 'code_inline' else ()))
                        self.transcript.insert('end',child.content,tags)
                self.transcript.insert('end','\n',('assistant',*blocks))
            elif token.type in ('fence','code_block'):
                self.transcript.insert('end',token.content.rstrip('\n')+'\n',('assistant','codeblock'))
            elif token.type == 'hr':
                self.transcript.insert('end','\u2500'*12+'\n','activity')

    def begin_transcript_selection(self, event=None):
        self.transcript_dragging = True
        self.transcript_press = (event.x,event.y) if event else None

    def end_transcript_selection(self, event=None):
        self.transcript_dragging = False
        if self.transcript_refresh_pending:
            self.transcript_refresh_pending = False
            self.after_idle(self.render)

    def bind_chat_command(self, tag, command):
        def released(event):
            press = getattr(self,'transcript_press',None)
            if press and abs(event.x-press[0])+abs(event.y-press[1]) <= 5:
                self.after_idle(command)
        self.transcript.tag_bind(tag,'<ButtonRelease-1>',released)

    def copy_selection(self, event=None):
        ranges = self.transcript.tag_ranges('sel')
        if ranges:
            self.clipboard_clear()
            self.clipboard_append(self.transcript.get(*ranges))
        return 'break'

    def transcript_view(self):
        top = self.transcript.index('@0,0')
        ranges = tuple(str(index) for index in self.transcript.tag_ranges('sel'))
        selected = self.transcript.get(*ranges) if ranges else ''
        return top,ranges,selected,self.transcript.yview()[1]>=0.999 and not ranges

    def restore_transcript_view(self, view):
        top,ranges,selected,at_bottom = view
        if at_bottom:
            self.transcript.update_idletasks()
            self.transcript.yview_moveto(1)
        else:
            self.transcript.yview(top)
        if ranges and self.transcript.get(*ranges)==selected:
            self.transcript.tag_add('sel',*ranges)

    def render(self, preserve_scroll=False):
        if self.transcript_dragging and self.rendered_session==self.identifier:
            self.transcript_refresh_pending = True
            return
        previous = self.transcript_view()
        same_session = self.rendered_session == self.identifier
        self.rendered_session = self.identifier
        self.live_display = None
        self.transcript.configure(state='normal')
        for mark in ('live_reasoning_start','live_reasoning_end','tail_padding'):
            if mark in self.transcript.mark_names():
                self.transcript.mark_unset(mark)
        self.transcript.delete('1.0','end')
        for tag in self.transcript.tag_names():
            if tag.startswith(('activity_','message_','menu_','reasoning_')):
                self.transcript.tag_delete(tag)
        self.capture_paths = []
        self.captures.delete(0,'end')
        events = self.store.events(self.identifier)
        excluded_ids = set(self.store.context_selection(self.identifier).get('excluded_ids',[]))
        summarized_ids = set(self.store.context_selection(self.identifier).get('summarized_ids',[]))
        log = []
        for event in events:
            for key, label in (('video','MP4'),('image','PNG')):
                if event['metadata'].get(key):
                    self.capture_paths.append(Path(event['metadata'][key]))
                    partial = ' (partial)' if event['metadata'].get('status') == 'stopped' else ''
                    self.captures.insert('end',label+' '+event['created']+partial)
            if event['role'] in ('tool','system'):
                log.append(event['created']+'  '+event['metadata'].get('tool',event['role'])+'\n'+event['content'])
        self.set_text(self.tool_log,display_excerpt('\n\n'.join(log),30000))
        for item in conversation_items(events):
            start = self.transcript.index('end-1c')
            role = item['role']
            if role == 'activity':
                key = (self.identifier,item['id'])
                expanded = key in self.expanded_activity
                tag = 'activity_'+str(item['id'])
                count = sum(event['role'] == 'tool' for event in item['events'])
                excluded = sum(event['id'] in excluded_ids for event in item['events'])
                marker = f' \u00b7 \ubb38\ub9e5 \uc81c\uc678 {excluded}' if excluded else ''
                summarized=sum(event['id'] in summarized_ids for event in item['events'])
                if summarized:
                    marker+=f' \u00b7 \uc694\uc57d \ubcf4\uc874 {summarized}'
                label = ('\u25be' if expanded else '\u25b8')+f'  \uc791\uc5c5 \uae30\ub85d \u00b7 \ub3c4\uad6c {count}\ud68c'+marker+'\n'
                self.transcript.insert('end',label,('activity',tag))
                self.bind_chat_command(tag,lambda key=key:self.toggle_activity(key))
                self.transcript.tag_bind(tag,'<Enter>',lambda event:self.transcript.configure(cursor='hand2'))
                self.transcript.tag_bind(tag,'<Leave>',lambda event:self.transcript.configure(cursor='xterm'))
                for row in tool_rows(item['events']):
                    outcome=row['execution_status']
                    status={'running':'\uc2e4\ud589 \uc911','starting':'\uc2dc\uc791 \uc911','returned':'\uacb0\uacfc \uc218\uc2e0',
                        'completed':'\uc2e4\ud589 \uc644\ub8cc','applied':'\ubcc0\uacbd \uc801\uc6a9','failed':'\uc2e4\ud328',
                        'cancelled':'\uc911\ub2e8','blocked':'\ucc28\ub2e8','ready_to_collect':'\uacb0\uacfc \uc218\uc9d1 \ub300\uae30',
                        'cancel_requested':'\uc911\ub2e8 \uc694\uccad','timed_out':'\uc2dc\uac04 \ucd08\uacfc',
                        'output_limit':'\ucd9c\ub825 \ud55c\ub3c4','unknown_after_restart':'\uc7ac\uc2dc\uc791 \ud6c4 \uc0c1\ud0dc \ubd88\uba85',
                        'unknown':'\uc5f0\uacb0 \ubbf8\ud655\uc778'}.get(outcome,outcome)
                    row_key=(self.identifier,'tool',row['id'])
                    row_tag='activity_tool_'+str(row['id'])
                    opened=row_key in self.expanded_activity
                    if row.get('commentary'):
                        self.transcript.insert('end',display_excerpt(row['commentary'],1200)+'\n','assistant')
                    trace=' #'+row['call_id'][:8] if row['call_id'] else ' [\uc5f0\uacb0 \ubbf8\ud655\uc778]'
                    self.transcript.insert('end',('\u25be' if opened else '\u25b8')+' '+row['tool']+trace+'  ['+status+']\n',('activity',row_tag))
                    self.bind_chat_command(row_tag,lambda key=row_key:self.toggle_activity(key))
                    if opened:
                        self.transcript.insert('end',display_excerpt(json.dumps(row['arguments'],ensure_ascii=False),2000)+'\n','detail')
                        for record in row['results'][-4:]:
                            structured=result_record(record)
                            self.transcript.insert('end','#'+str(record['id'])+' '+structured['execution_status']+' '+
                                                   display_excerpt(json.dumps(structured['summary'],ensure_ascii=False),700)+'\n','metrics')
                        if row['result']:
                            self.transcript.insert('end',display_excerpt(row['result']['content'],6000)+'\n','detail')
                            full_tag='activity_full_'+str(row['id'])
                            self.transcript.insert('end','[\uc6d0\ubb38 \ubcf4\uae30]\n',('link',full_tag))
                            self.bind_chat_command(full_tag,lambda event=row['result']:self.show_record(event))
                if expanded:
                    for event in item['events']:
                        metadata = event.get('metadata',{})
                        if event['id'] in excluded_ids:
                            self.transcript.insert('end','[\ubb38\ub9e5 \uc81c\uc678]\n','metrics')
                        if metadata.get('status') == 'approval':
                            try:
                                audit = json.loads(event['content'])
                                text = audit['tool']+' \u00b7 '+audit['mode']
                            except (ValueError,KeyError,TypeError):
                                text = event['content']
                        else:
                            text = (metadata.get('tool','')+'\n' if event['role'] == 'tool' else '')+event['content']
                        self.transcript.insert('end',display_excerpt(text,6000)+'\n','detail')
                        statistics = performance_text(metadata.get('metrics'))
                        if statistics:
                            self.transcript.insert('end',statistics+'\n','metrics')
                        self.insert_reasoning_link(event)
                continue
            content = item['content']
            if item['id'] in excluded_ids:
                self.transcript.insert('end','[\ubb38\ub9e5 \uc81c\uc678: \ub9c8\uc9c0\ub9c9 \ubaa8\ub378 \uc694\uccad\uc5d0 \ubbf8\ud3ec\ud568]\n','metrics')
            attached = item.get('metadata',{}).get('attachments',[])
            if attached:
                content += '\n'+'\n'.join('\u2295 '+attachment['name'] for attachment in attached)
            if item.get('metadata',{}).get('partial'):
                content = '[\uc911\ub2e8\ub41c \ub2f5\ubcc0]\n'+content
            try:
                stamp = datetime.fromisoformat(item.get('created','')).astimezone().strftime('%H:%M')
            except ValueError:
                stamp = ''
            if role == 'user':
                self.insert_message_header(item,'\ub098',stamp,'user_label')
                self.transcript.insert('end',content+'\n','user')
            elif role == 'assistant':
                self.insert_message_header(item,'Local Desk',stamp,'assistant_label')
                self.insert_reasoning_link(item)
                self.insert_markdown(content)
                statistics = performance_text(item.get('metadata',{}).get('metrics'))
                if statistics:
                    self.transcript.insert('end',statistics+'\n','metrics')
            else:
                self.transcript.insert('end',content+'\n','system')
                self.insert_reasoning_link(item)
            self.transcript.insert('end','\n','gap')
            if role in ('user','assistant'):
                tag = 'message_'+str(item['id'])
                self.transcript.tag_add(tag,start,'end-1c')
                self.transcript.tag_bind(tag,'<Button-3>',lambda event,event_id=item['id']:self.show_message_menu(event_id,event))
        self.transcript.configure(state='disabled')
        self.update_live_reasoning(preserve_scroll=True)
        if not same_session:
            self.transcript.see('end')
        else:
            self.restore_transcript_view(previous)

    def insert_reasoning_link(self, item):
        reasoning = item.get('metadata',{}).get('metrics',{}).get('reasoning')
        if not isinstance(reasoning,str) or not reasoning:
            return
        tag = 'reasoning_'+str(item['id'])
        expanded = (self.identifier,item['id']) in self.expanded_reasoning
        partial = ' (\ubbf8\uc644\ub8cc)' if item.get('metadata',{}).get('metrics',{}).get('reasoning_partial') else ''
        self.transcript.insert('end',('\u25be' if expanded else '\u25b8')+' [\uc0ac\uace0 \ub0b4\uc6a9]'+partial+'\n',('activity',tag))
        self.bind_chat_command(tag,lambda:self.show_reasoning(item['id']))
        self.transcript.tag_bind(tag,'<Enter>',lambda event:self.transcript.configure(cursor='hand2'))
        self.transcript.tag_bind(tag,'<Leave>',lambda event:self.transcript.configure(cursor='xterm'))
        if expanded:
            self.transcript.insert('end',display_excerpt(reasoning)+'\n','detail')
            if len(reasoning)>12000:
                full_tag=tag+'_full'
                self.transcript.insert('end','[\uc0ac\uace0 \uc6d0\ubb38 \ubcf4\uae30]\n',('link',full_tag))
                self.bind_chat_command(full_tag,lambda:self.show_record(item,reasoning))

    def show_reasoning(self, event_id):
        event = next((event for event in self.store.events(self.identifier) if event['id'] == event_id),None)
        reasoning = event.get('metadata',{}).get('metrics',{}).get('reasoning') if event else None
        if not isinstance(reasoning,str) or not reasoning:
            return
        key = (self.identifier,event_id)
        if key in self.expanded_reasoning:
            self.expanded_reasoning.remove(key)
        else:
            self.expanded_reasoning.add(key)
        self.render(preserve_scroll=True)

    def update_live_reasoning(self, preserve_scroll=False):
        if self.transcript_dragging:
            self.transcript_refresh_pending = True
            return
        previous = self.transcript_view()
        self.transcript.configure(state='normal')
        if 'tail_padding' in self.transcript.mark_names():
            self.transcript.delete('tail_padding','end-1c')
            self.transcript.mark_unset('tail_padding')
        live = self.live_reasoning
        key=(self.identifier,live['key']) if live and live['session_id']==self.identifier else None
        expanded=key in self.expanded_reasoning if key else False
        display=display_excerpt(live['text']) if key and expanded else ''
        signature=(key,expanded)
        incremental=(self.live_display is not None and self.live_display[0]==signature
                     and display.startswith(self.live_display[1]) and 'live_reasoning_end' in self.transcript.mark_names())
        if incremental:
            self.transcript.insert('live_reasoning_end',display[len(self.live_display[1]):],'detail')
            self.transcript.mark_set('live_reasoning_end','end-1c')
        elif 'live_reasoning_start' in self.transcript.mark_names():
            self.transcript.delete('live_reasoning_start','live_reasoning_end')
            self.transcript.mark_unset('live_reasoning_start','live_reasoning_end')
        if not incremental and live and live['session_id']==self.identifier and (live.get('text') or live.get('enabled')):
            key = (self.identifier,live['key'])
            expanded = key in self.expanded_reasoning
            self.transcript.mark_set('live_reasoning_start','end-1c')
            self.transcript.mark_gravity('live_reasoning_start','left')
            self.transcript.insert('end','Local Desk\n','assistant_label')
            self.transcript.insert('end',('\u25be' if expanded else '\u25b8')+' [\uc0ac\uace0\uc911]\n',('activity','reasoning_live'))
            self.bind_chat_command('reasoning_live',self.toggle_live_reasoning)
            self.transcript.tag_bind('reasoning_live','<Enter>',lambda event:self.transcript.configure(cursor='hand2'))
            self.transcript.tag_bind('reasoning_live','<Leave>',lambda event:self.transcript.configure(cursor='xterm'))
            if expanded:
                self.transcript.insert('end',display,'detail')
            self.transcript.mark_set('live_reasoning_end','end-1c')
            self.transcript.mark_gravity('live_reasoning_end','left')
        self.live_display=(signature,display) if key else None
        self.transcript.mark_set('tail_padding','end-1c')
        self.transcript.mark_gravity('tail_padding','left')
        self.transcript.insert('end','\n\n\n')
        self.transcript.configure(state='disabled')
        self.restore_transcript_view(previous)

    def toggle_live_reasoning(self):
        if not self.live_reasoning or self.live_reasoning['session_id']!=self.identifier:
            return
        key = (self.identifier,self.live_reasoning['key'])
        if key in self.expanded_reasoning:
            self.expanded_reasoning.remove(key)
        else:
            self.expanded_reasoning.add(key)
        self.update_live_reasoning(preserve_scroll=True)

    def complete_live_reasoning(self, value=None):
        if self.live_reasoning:
            key = (self.live_reasoning['session_id'],self.live_reasoning['key'])
            if key in self.expanded_reasoning:
                self.expanded_reasoning.remove(key)
                if value and value['session_id']==self.live_reasoning['session_id']:
                    self.expanded_reasoning.add((value['session_id'],value['event_id']))
        self.live_reasoning = None

    def show_record(self,item,content=None):
        dialog=tk.Toplevel(self)
        dialog.title(item.get('metadata',{}).get('tool','Record'))
        dialog.minsize(480,320)
        place_dialog(dialog,self,820,580)
        footer=ttk.Frame(dialog,padding=8)
        footer.pack(side='bottom',fill='x')
        raw=item['content'] if content is None else content
        position=[0]
        viewer=ScrolledText(dialog,wrap='word',font=('Malgun Gothic',11),padx=12,pady=12)
        viewer.pack(fill='both',expand=True)
        status=tk.StringVar()
        def page(change):
            position[0]=max(0,min(max(0,len(raw)-1)//12000,position[0]+change))
            start=position[0]*12000
            self.set_text(viewer,display_excerpt(raw[start:start+12000]))
            status.set(f'{position[0]+1} / {max(1,(len(raw)+11999)//12000)}')
        ttk.Button(footer,text='<',width=3,command=lambda:page(-1)).pack(side='left')
        ttk.Label(footer,textvariable=status).pack(side='left',padx=8)
        ttk.Button(footer,text='>',width=3,command=lambda:page(1)).pack(side='left')
        def copy():
            self.clipboard_clear();self.clipboard_append(raw)
        ttk.Button(footer,text='Copy all',command=copy).pack(side='right')
        page(0)

    def copy_message(self,event_id):
        _,text=self.store.message(self.identifier,event_id)
        self.clipboard_clear()
        self.clipboard_append(text)

    def insert_message_header(self, item, name, stamp, style):
        edited = ' \u00b7 \ud3b8\uc9d1\ub428' if item.get('metadata',{}).get('edited') else ''
        self.transcript.insert('end',name+'  '+stamp+edited,style)
        copy_tag='menu_copy_'+str(item['id'])
        self.transcript.insert('end','   \u2398',(style,copy_tag))
        self.bind_chat_command(copy_tag,lambda:self.copy_message(item['id']))
        tag = 'menu_'+str(item['id'])
        self.transcript.insert('end','   \u22ef',(style,tag))
        self.transcript.insert('end','\n',style)
        self.transcript.tag_bind(tag,'<ButtonRelease-1>',lambda event,event_id=item['id']:self.show_message_menu(event_id,event)
            if getattr(self,'transcript_press',None) and abs(event.x-self.transcript_press[0])+abs(event.y-self.transcript_press[1])<=5 else None)
        self.transcript.tag_bind(tag,'<Enter>',lambda event:self.transcript.configure(cursor='hand2'))
        self.transcript.tag_bind(tag,'<Leave>',lambda event:self.transcript.configure(cursor='xterm'))

    def update_stream(self, content):
        top = self.stream.index('@0,0')
        ranges = tuple(str(index) for index in self.stream.tag_ranges('sel'))
        selected = self.stream.get(*ranges) if ranges else ''
        self.set_text(self.stream,display_excerpt(content,6000))
        if ranges and self.stream.get(*ranges)==selected:
            self.stream.tag_add('sel',*ranges)
        if content:
            if not self.stream.winfo_manager():
                self.stream.pack(fill='x',pady=(6,0))
            self.stream.yview(top)
        else:
            self.stream.pack_forget()

    def poll(self):
        reasoning_dirty = False
        stream_pending = None
        for count in range(300):
            try:
                kind,value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'interrupt':
                self.stop(value)
            elif kind == 'status':
                self.status.set(value)
            elif kind == 'session_ready':
                stream_pending = ''
                self.identifier = value
                self.expanded_activity.clear()
                self.preview.configure(image='',text='No capture',width=30,height=7)
                self.photo = None
                self.update_stream('')
                self.complete_live_reasoning()
                self.render()
                self.refresh_sessions()
            elif kind == 'operation_error':
                messagebox.showerror('Session operation',value,parent=self)
            elif kind == 'target':
                index = next((index for index,row in enumerate(self.window_choices) if row['handle'] == value['handle']),None)
                if index is None:
                    self.window_choices.append(value)
                    index = len(self.window_choices)-1
                else:
                    self.window_choices[index] = value
                self.window_selector.configure(values=['No target']+[f'{row["title"][:60]} [{row["handle"]}]' for row in self.window_choices])
                self.window_selector.current(index+1)
            elif kind == 'reasoning_start':
                self.complete_live_reasoning()
                self.live_reasoning = dict(session_id=value['session_id'],key=('live',value['request_id'],value['step']),
                                           enabled=value.get('enabled',False),text='')
                reasoning_dirty = True
            elif kind == 'reasoning_saved':
                stream_pending = ''
                self.complete_live_reasoning(value)
                self.render()
            elif kind in ('stream','reasoning'):
                if kind == 'stream':
                    stream_pending = value
                else:
                    if self.live_reasoning and isinstance(value,str):
                        self.live_reasoning['text'] = value
                        reasoning_dirty = True
            elif kind == 'context':
                prefix = 'API est. ' if self.settings.backend == 'api' else ''
                self.context_status.set(f'{prefix}~{value["tokens"]}/{self.settings.context_limit//1024}K | omitted {value["omitted"]}')
                if value.get('session_id') == self.identifier:
                    self.render(preserve_scroll=True)
            elif kind == 'refresh':
                self.render()
                self.refresh_sessions()
            elif kind in ('terminal','file_change'):
                if value.get('session_id') == self.identifier:
                    self.status.set(kind+': '+value.get('status',''))
                    self.render(preserve_scroll=True)
            elif kind in ('tool','approval_log'):
                if kind == 'tool':
                    stream_pending = ''
            elif kind == 'image':
                image = value.copy()
                image.thumbnail((280,190))
                self.photo = ImageTk.PhotoImage(image)
                self.preview.configure(image=self.photo,text='',width=280,height=190)
            elif kind == 'approval':
                if self.stopped.is_set() or value['event'].is_set():
                    value['event'].set()
                    continue
                if self.compact:
                    self.toggle_compact()
                self.approval = value
                context = dict(value['context'])
                difference = context.pop('diff','')
                context.pop('tool_policies',None)
                action = dict(value['action'])
                if action['tool'] == 'workspace_apply_patch':
                    action = dict(tool=action['tool'],path=action['arguments']['path'])
                preview = value['reason']+'\n'+json.dumps(action,ensure_ascii=False,indent=2)+'\n'+json.dumps(context,ensure_ascii=False)+'\n'+difference
                self.set_text(self.approval_text,display_excerpt(preview[:12000])+('\n[\uc804\uccb4 \ub0b4\uc6a9\uc740 \u2197 \ubcf4\uae30]' if len(preview)>12000 else ''))
                self.approval_frame.pack(fill='x',before=self.footer)
                self.lift()
            elif kind == 'worker_idle':
                self.pending_jobs = max(0,self.pending_jobs-1)
                if not self.pending_jobs:
                    self.set_busy(False)
                    self.complete_live_reasoning()
                    reasoning_dirty = True
            elif kind == 'done':
                stream_pending = ''
                self.complete_live_reasoning()
                reasoning_dirty = True
        if stream_pending is not None:
            self.update_stream(stream_pending)
        if reasoning_dirty:
            self.update_live_reasoning()
        if self.approval is not None and self.approval['event'].is_set():
            self.answer(False)
        if self.closing and not self.worker.is_alive():
            self.monitor.close()
            self.destroy()
            return
        self.after(60,self.poll)

    def close(self):
        if not self.closing:
            self.closing = True
            self.stop('Application closing')
            self.jobs.put(('close',None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    arguments = parser.parse_args()
    enable_dpi()
    temporary = tempfile.TemporaryDirectory() if arguments.smoke else None
    app = Console(temporary.name if temporary else None)
    passed = []
    if arguments.smoke:
        from types import SimpleNamespace
        from unittest.mock import patch
        import time
        attached_path = Path(temporary.name)/'attachment with spaces.txt'
        attached_path.write_text('Attachment test',encoding='utf-8')
        app.drop_files(SimpleNamespace(data=app.tk.call('list',str(attached_path))))
        assert app.attachments == [str(attached_path.resolve())]
        app.remove_attachment(str(attached_path.resolve()))
        assert app.screen.get() and app.inputs.get() and app.browser.get() and app.automatic.get()
        assert app.reason_selector.get() == 'medium'
        app.store.append(app.identifier,'user','\uc774 \ud654\uba74\uc5d0 \ubcf4\uc774\ub294 \ud14d\uc2a4\ud2b8\ub97c \ud655\uc778\ud574\uc918.')
        app.store.append(app.identifier,'assistant',json.dumps(dict(message='\ud654\uba74\uc744 \ud655\uc778\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4.',tool='desktop_capture')))
        app.store.append(app.identifier,'system',json.dumps(dict(tool='desktop_capture',mode='automatic')),status='approval')
        app.store.append(app.identifier,'tool','Capture test detail',tool='desktop_capture',status='delivered')
        sample_reply = ('## \ud654\uba74\uc5d0\uc11c \ud655\uc778\ud55c \ub0b4\uc6a9\n\n'
                        '**Local Desk**\n\n'
                        '- \ucea1\ucc98 \uc6d0\ubcf8: 1920 x 1080\n'
                        '- \uc120\ud0dd\ud55c \ucc3d: Chrome\n\n'
                        '\uc778\uc2dd\ud55c \ubb38\uad6c\ub294 `ORBIT READY`\uc785\ub2c8\ub2e4.\n\n'
                        '```text\nORBIT READY\n```')
        app.store.append(app.identifier,'assistant',json.dumps(dict(message=sample_reply,tool='finish')),
                 metrics={'timings':{'prompt_per_second':428.6,'predicted_per_second':8.5},'seconds':9.3,'task_seconds':22.8,
                          'context_tokens':4100,'context_limit':8192,'context_source':'server_usage'})
        app.render()
        app.update_idletasks()
        assert app.transcript.tag_ranges('strong') and app.transcript.tag_ranges('codeblock')
        assert app.transcript.tag_ranges('metrics') and '428.6 tok/s' in app.transcript.get('1.0','end')
        assert '4,100 / 8,192 tok' in app.transcript.get('1.0','end')
        assert 'Capture test detail' not in app.transcript.get('1.0','end')
        activity = next(item for item in conversation_items(app.store.events(app.identifier)) if item['role'] == 'activity')
        key = (app.identifier,activity['id'])
        app.toggle_activity(key)
        assert 'Capture test detail' in app.transcript.get('1.0','end')
        app.toggle_activity(key)
        sample_session = app.identifier
        app.identifier = app.store.create('Scroll test')
        app.store.append(app.identifier,'user','Long conversation')
        app.store.append(app.identifier,'assistant',json.dumps(dict(message='Paragraph\n\n'*80,tool='finish')))
        app.render()
        app.update_idletasks()
        app.transcript.yview_moveto(0.3)
        previous = app.transcript.yview()[0]
        app.store.append(app.identifier,'system','Status update',status='stopped')
        app.render()
        app.update_idletasks()
        assert abs(app.transcript.yview()[0]-previous) < 0.03
        app.identifier = sample_session
        app.render()
        app.refresh_sessions()
        sample_events = app.store.events(sample_session)
        request_id = sample_events[0]['id']
        reply_id = sample_events[-1]['id']
        assert app.transcript.tag_ranges('menu_'+str(request_id))
        app.edit_message(request_id)
        dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        editor = next(widget for widget in descendants(dialog) if isinstance(widget,tk.Text))
        editor.delete('1.0','end')
        editor.insert('1.0','\uc774 \ud654\uba74\uc5d0 \ubcf4\uc774\ub294 \ud14d\uc2a4\ud2b8\ub97c \ud655\uc778\ud574\uc918.')
        save_button = next(widget for widget in descendants(dialog) if isinstance(widget,ttk.Button) and widget.cget('text') == '\ud655\uc778')
        for geometry in ('680x400','420x240'):
            dialog.geometry(geometry)
            app.update()
            assert save_button.winfo_ismapped()
            assert save_button.winfo_rooty()+save_button.winfo_height() <= dialog.winfo_rooty()+dialog.winfo_height()
        save_button.invoke()
        assert app.store.message(sample_session,request_id)[0]['metadata']['edited']
        with patch('desktop_agent.app.messagebox.askyesno',return_value=False):
            app.delete_message(reply_id)
        assert app.store.message(sample_session,reply_id)
        removable = app.store.append(sample_session,'assistant',json.dumps(dict(message='Temporary reply',tool='finish')))
        with patch('desktop_agent.app.messagebox.askyesno',return_value=True):
            app.delete_message(removable)
        assert all(event['id'] != removable for event in app.store.events(sample_session))
        app.model.ensure = lambda stopped:'test'
        app.model.count = lambda text:len(text)//4
        app.model.generate = lambda *args:(dict(message='Replay test complete',tool='finish',arguments={},risk='routine'),{})
        with patch('desktop_agent.app.messagebox.askyesno',return_value=True):
            app.replay_message(reply_id)
        app.attributes('-topmost',True)
        app.lift()
        stages = [('1180x800','desktop-full.png'),('960x640','desktop-small.png'),('compact','desktop-compact.png')]
        def prepare(index=0):
            if index == len(stages):
                app.toggle_compact()
                app.update_idletasks()
                passed.append(True)
                print('PASS: message editor/delete/replay, session deletion, rich chat, activity, scroll, full/small/compact; stub model, no input',flush=True)
                app.close()
                return
            geometry,filename = stages[index]
            if geometry == 'compact':
                app.toggle_compact()
            else:
                app.geometry(geometry)
            app.after(250,lambda:capture(index,filename))
        def capture(index,filename):
            try:
                assert app.entry.winfo_ismapped() and app.entry.winfo_width() > 200
                assert app.entry.winfo_rooty()+app.entry.winfo_height() <= app.winfo_rooty()+app.winfo_height()
                if not app.compact:
                    assert app.transcript.winfo_width() > 200
                else:
                    assert not app.body.winfo_ismapped() and app.winfo_height() <= 90
                ImageGrab.grab(bbox=(app.winfo_rootx(),app.winfo_rooty(),app.winfo_rootx()+app.winfo_width(),app.winfo_rooty()+app.winfo_height())).save(HOME/filename)
                app.after(100,lambda:prepare(index+1))
            except Exception:
                app.close()
                raise
        deadline = time.monotonic()+15
        def after_replay():
            if app.busy:
                if time.monotonic() > deadline:
                    app.close()
                    raise RuntimeError('Replay UI test timed out')
                app.after(50,after_replay)
                return
            assert app.identifier == sample_session
            assert len(app.store.events(sample_session)) == 2
            assert all(event['id'] not in (request_id,reply_id) for event in app.store.events(sample_session))
            assert len(app.store.sessions()) == 2
            app.store.append(sample_session,'assistant',json.dumps(dict(message=sample_reply,tool='finish')))
            removable_session = app.store.create('Delete probe')
            app.identifier = removable_session
            app.refresh_sessions()
            with patch('desktop_agent.app.messagebox.askyesno',return_value=True):
                app.delete_session()
            app.after(50,lambda:after_delete(removable_session))
        def after_delete(branch):
            if app.busy:
                if time.monotonic() > deadline:
                    app.close()
                    raise RuntimeError('Session delete UI test timed out')
                app.after(50,lambda:after_delete(branch))
                return
            assert branch not in [row['id'] for row in app.store.sessions()]
            assert not (app.data/'artifacts'/branch).exists()
            assert app.store.events(sample_session)
            app.identifier = sample_session
            app.refresh_sessions()
            app.render()
            app.status.set('Ready')
            app.after(250,prepare)
        app.after(100,after_replay)
    app.mainloop()
    if temporary:
        temporary.cleanup()
        if not passed:
            raise RuntimeError('UI smoke failed')


if __name__ == '__main__':
    main()