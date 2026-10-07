import json
import re
from urllib.parse import urlsplit


def text(limit=4000):
    return {'type': 'string', 'maxLength': limit}


POINT = {'type': 'integer', 'minimum': 0, 'maximum': 1000}
PIXEL_POINT = {'type':'integer','minimum':0,'maximum':65535}
KEYS = ['Enter', 'Tab', 'Escape', 'Backspace', 'Delete', 'Space', 'ArrowUp', 'ArrowDown',
        'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown', 'Control+a',
        'Control+c', 'Control+v', 'Control+z', 'Control+f', 'Control+s', 'Control+l',
        'Control+w', 'Control+Enter', 'Shift+Tab', 'Shift+Delete', 'Alt+F4']
KEYS += list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
KEYS += ['F'+str(number) for number in range(1,13)]
KEYS += ['Shift+ArrowUp','Shift+ArrowDown','Shift+ArrowLeft','Shift+ArrowRight']
KEY_QUEUE = {'type':'array', 'minItems':1, 'maxItems':32, 'items':{
    'type':'object', 'additionalProperties':False, 'required':['key','delay_ms'], 'properties':{
        'key':{'type':'string','enum':KEYS},
        'delay_ms':{'type':'integer','minimum':0,'maximum':5000}}}}
RECORDING = {'seconds':{'type':'integer','minimum':1,'maximum':30},
             'fps':{'type':'integer','minimum':1,'maximum':15}}
CAPTURE_TOOLS = {'desktop_capture','desktop_screen_capture','browser_capture','desktop_record','browser_record'}
TOOLS = {
    'finish': ('Reply to the user; no tool execution.', {}),
    'window_list': ('List visible desktop windows with handle, pid and title. Use when no target or another window is needed.', {}),
    'window_select': ('Choose a listed window as desktop target. Does not capture or type. Requires desktop permission.',
        {'handle':{'type':'integer','minimum':1,'maximum':9223372036854775807},
         'pid':{'type':'integer','minimum':1,'maximum':4294967295}}),
    'desktop_capture': ('Capture selected window without focus changes. Default: HWND only, excluding overlaps. User-enabled capture margin instead shows visible surrounding screen, including other windows. Returned bounds define image pixels; input remains restricted to selected window. No target: entire desktop.', {}),
    'desktop_screen_capture': ('Capture the currently visible full desktop across monitors AS IS. No target required; never focus/restore/move any window. Includes taskbar and visible overlapping windows. Not selected-window coordinates.', {}),
    'desktop_record': ('Record selected HWND directly without focusing; no desktop rectangle substitution. Silent MP4 max1280px; only when asked. Returns a timestamped contact sheet retained until the next capture.', RECORDING),
    'desktop_click': ('Click selected window; x/y normalized 0..1000 over full visible window bounds, with no fixed border inset.',
        {'x': POINT, 'y': POINT, 'button': {'type': 'string', 'enum': ['left', 'right']},
         'clicks': {'type': 'integer', 'minimum': 1, 'maximum': 2}}),
    'desktop_type': ('Type Unicode text into selected window focused field.', {'text': text()}),
    'desktop_key': ('Press key/chord in selected window.', {'key': {'type': 'string', 'enum': KEYS}}),
    'desktop_key_queue': ('Execute ordered keys/chords in selected window. delay_ms waits BEFORE each key; max 32 steps and 60000ms total. Cancellable.',
        {'steps':KEY_QUEUE}),
    'desktop_input': ('Input modes: general=foreground virtual keys/Unicode; device=SendInput scancodes (software, NOT physical hardware); message=Win32 child control at x/y without moving cursor/focus, unsupported by some apps. device text unsupported; message modifier chords unsupported. Unused fields empty/0. general/device keyboard uses focused field, x/y for mouse.',
        {'mode':{'type':'string','enum':['general','device','message']},
         'kind':{'type':'string','enum':['click','key','text','scroll']},'x':POINT,'y':POINT,
         'button':{'type':'string','enum':['left','right']},'key':{'type':'string','enum':['']+KEYS},
         'text':text(),'amount':{'type':'integer','minimum':-8,'maximum':8},
         'hold_ms':{'type':'integer','minimum':0,'maximum':2000}}),
    'desktop_scroll': ('Vertical scroll at point; positive up.',
        {'x': POINT, 'y': POINT, 'amount': {'type': 'integer', 'minimum': -8, 'maximum': 8}}),
    'desktop_drag': ('Left-button drag in selected window.', {'x': POINT, 'y': POINT, 'end_x': POINT, 'end_y': POINT}),
    'desktop_hold': ('Hold mouse button or key/chord for hold_ms then release, even if cancelled. mode general/device/message; message keys cannot use modifiers.',
        {'kind':{'type':'string','enum':['mouse','key']},'mode':{'type':'string','enum':['general','device','message']},
         'x':POINT,'y':POINT,'button':{'type':'string','enum':['left','right']},'key':{'type':'string','enum':['']+KEYS},
         'hold_ms':{'type':'integer','minimum':1,'maximum':10000}}),
    'desktop_drag_timed': ('Drag from x/y to end_x/end_y over duration_ms then release. message mode does not move physical pointer; general/device use SendInput.',
        {'mode':{'type':'string','enum':['general','device','message']},'x':POINT,'y':POINT,'end_x':POINT,'end_y':POINT,
         'button':{'type':'string','enum':['left','right']},'duration_ms':{'type':'integer','minimum':50,'maximum':10000}}),
    'browser_open': ('Open HTTP(S) URL in managed browser.', {'url': text(2048)}),
    'browser_search': ('Search web in managed browser.', {'query': text(400)}),
    'browser_read': ('Read MANAGED browser page text and CSS selectors, NOT the selected external Chrome window. No image attached.', {}),
    'browser_capture': ('Capture managed browser screenshot for the next model request only.', {}),
    'browser_record': ('Record managed browser page as silent MP4, max 1280px, only when asked for video/motion. Returns a timestamped contact sheet retained until the next capture.', RECORDING),
    'browser_click': ('Click a unique visible CSS selector on current page.', {'selector': text(500)}),
    'browser_type': ('Fill a CSS-selected text field. Passwords/files require manual user input.',
        {'selector': text(500), 'text': text()}),
    'browser_key': ('Press key/chord on managed page.', {'key': {'type': 'string', 'enum': KEYS}}),
    'browser_key_queue': ('Execute ordered keys/chords on managed page. delay_ms waits BEFORE each key; max 32 steps and 60000ms total. Cancellable.',
        {'steps':KEY_QUEUE}),
    'browser_hold':('Hold a mouse button on selector or key/chord in its focused element, then always release.',
        {'selector':text(500),'kind':{'type':'string','enum':['mouse','key']},'key':{'type':'string','enum':['']+KEYS},
         'button':{'type':'string','enum':['left','right']},'hold_ms':{'type':'integer','minimum':1,'maximum':10000}}),
    'browser_drag':('Drag visible source selector onto target_selector over duration_ms; release on stop.',
        {'selector':text(500),'target_selector':text(500),'duration_ms':{'type':'integer','minimum':50,'maximum':10000}}),
    'browser_scroll': ('Scroll page; positive down.', {'amount': {'type': 'integer', 'minimum': -3, 'maximum': 3}}),
    'browser_back': ('Go back in managed browser history.', {}),
    'browser_tabs': ('List managed browser tabs.', {}),
    'browser_tab': ('Select managed tab by index.', {'index': {'type': 'integer', 'minimum': 0, 'maximum': 30}}),
}


def tool_schemas(names):
    return [{'type': 'object', 'additionalProperties': False,
        'required': ['message', 'tool', 'arguments', 'risk'], 'properties': {
            'message': text(5000), 'tool': {'type': 'string', 'enum': [name]},
            'risk': {'type': 'string', 'enum': ['routine', 'sensitive']},
            'arguments': {'type': 'object', 'properties': parameters, 'required': list(parameters),
                          'additionalProperties': False}}} for name in names for _, parameters in [TOOLS[name]]]


MACRO_TOOLS = {
    'desktop_macro': ('desktop_type','desktop_key','desktop_click','desktop_scroll'),
    'browser_macro': ('browser_type','browser_key','browser_click','browser_scroll'),
}
for macro, children in MACRO_TOOLS.items():
    TOOLS[macro] = ('Repeat ordered actions on one target; stop on failure. interval_ms waits AFTER each complete repetition except the last, not a real-time start interval. Cancellable; never replay partial work.', {
        'actions':{'type':'array','minItems':1,'maxItems':32,'items':{'oneOf':tool_schemas(children)}},
        'repeat':{'type':'integer','minimum':1,'maximum':1000},
        'interval_ms':{'type':'integer','minimum':0,'maximum':60000}})
BACKGROUND_TOOLS = tuple(name for name in TOOLS if name.startswith(('desktop_','browser_')))
TOOLS.update({
    'job_start':('Start a tool independently; other tools may run during recording/waits. Returns job_id. Collect with job_result.',
        {'action':{'oneOf':tool_schemas(BACKGROUND_TOOLS)}}),
    'job_status':('List background jobs and readiness.',{}),
    'job_result':('Wait for and collect one background result/image.',{'job_id':text(40)}),
    'job_cancel':('Cancel a background job; collect partial results with job_result.',{'job_id':text(40)}),
})
TOOLS.update({
    'file_read':('Read text/PDF/DOCX from a registered session attachment or artifact; never execute files.',
        {'path':text(2048),'offset':{'type':'integer','minimum':0,'maximum':100000000},'limit':{'type':'integer','minimum':1,'maximum':6000}}),
    'file_view':('View a registered image/video attachment or artifact. Replaces the latest attached tool image, which persists until replaced, subject to permissions.',{'path':text(2048)}),
    'session_history':('Read complete session log including tool calls, failures and file paths as paginated text. No image reload.',
        {'offset':{'type':'integer','minimum':0,'maximum':100000000},'limit':{'type':'integer','minimum':1,'maximum':6000}}),
    'session_record':('Read one exact stored event by raw_ref event_id within this session. Returns original content pages, not a summary. offset/next_offset count characters; no tool re-execution.',
        {'event_id':{'type':'integer','minimum':1,'maximum':9007199254740991},'offset':{'type':'integer','minimum':0,'maximum':100000000},'limit':{'type':'integer','minimum':1,'maximum':64000}}),
})
CAPTURE_TOOLS.add('file_view')

WORKSPACE_TOOLS = {
    'workspace_code_search':('Retrieve ranked source snippets from a local BM25 code index with identifier terms. Relative folder, .gitignore honored, no embeddings or network. Returns paths, lines, hashes and read_ref; incomplete index is reported.',
        {'query':text(2000),'path':text(2048),'limit':{'type':'integer','minimum':1,'maximum':30}}),
    'workspace_symbols':('Find parser-extracted definitions or exact-name syntactic references in supported source languages. Not language-server binding resolution. Returns path/line/hash/read_ref. Relative folder.',
        {'query':text(300),'path':text(2048),'kind':{'type':'string','enum':['definitions','references']},'limit':{'type':'integer','minimum':1,'maximum':100}}),
    'workspace_read':('Read UTF-8 text at a path relative to the selected workspace. Returns numbered lines, sha256 and line_ending. Up to 2000 lines per call; not for registered attachments.',
        {'path':text(2048),'start_line':{'type':'integer','minimum':1,'maximum':1000000},'end_line':{'type':'integer','minimum':1,'maximum':1000000}}),
    'workspace_search':('Search literal text, case-insensitively, in UTF-8 files under a relative workspace folder. Not regex. Links and protected paths excluded; truncated marks limits.',
        {'query':text(4000),'path':text(2048),'limit':{'type':'integer','minimum':1,'maximum':1000}}),
    'workspace_apply_patch':('Replace one exact unique old_text block with new_text after expected_sha256 matches. Not unified diff. Relative path; empty hash/old_text creates a file in an existing folder. Always confirm diff; no file deletion.',
        {'path':text(2048),'expected_sha256':text(64),'old_text':text(256000),'new_text':text(256000)}),
    'terminal_start':('Start a fresh non-interactive Windows PowerShell 5.1 process after confirmation. Relative cwd in selected workspace; NOT sandboxed. Closed stdin, no secret input. Returns execution_id, not completed output.',
        {'command':text(12000),'cwd':text(2048),'timeout_seconds':{'type':'integer','minimum':1,'maximum':3600}}),
    'terminal_output':('Read this session execution output, status and exit_code. offset/next_offset count bytes. running/starting is pending; unknown_after_restart is not success.',
        {'execution_id':text(40),'offset':{'type':'integer','minimum':0,'maximum':67108864},'limit':{'type':'integer','minimum':1,'maximum':64000}}),
    'terminal_stop':('Stop only this session owned execution and its job process tree; cannot stop an arbitrary PID.',{'execution_id':text(40)}),
}
TOOLS.update(WORKSPACE_TOOLS)
PARALLEL_TOOLS = ('workspace_read','workspace_search','browser_read','browser_tabs','terminal_output')
TOOLS['tool_parallel'] = ('Run up to four independent read-only calls; results preserve input order. Same browser remains serialized. No dependencies between calls.', {
    'actions':{'type':'array','minItems':1,'maxItems':4,'items':{'oneOf':tool_schemas(PARALLEL_TOOLS)}}})
COMPOSITE_TOOLS = dict(MACRO_TOOLS,tool_parallel=PARALLEL_TOOLS)
TOOL_POLICIES = ('disabled','ask','allow')


def tool_policy(policies,name):
    value=policies.get(name,'disabled' if name in WORKSPACE_TOOLS else 'allow')
    if value not in TOOL_POLICIES:
        raise ValueError('Invalid tool permission')
    return value


def action_schema():
    return {'oneOf':tool_schemas(TOOLS)}


TOOL_GROUPS = {
    'browser': ('Managed browser navigation, page text, clicks, typing and tabs.',
                tuple(name for name in TOOLS if name.startswith('browser_') and name not in ('browser_record','browser_hold','browser_drag','browser_key_queue','browser_macro'))),
    'recording': ('Silent window/browser video; optional background recording while interacting.',('desktop_record','browser_record','job_status','job_result','job_cancel')),
    'advanced_input': ('Key queues, held keys/buttons, timed drags, scancode and window-message input.',
                       ('desktop_input','desktop_hold','desktop_drag_timed','desktop_key_queue','browser_key_queue','browser_hold','browser_drag')),
    'files': ('Read registered attachments, view images/video, retrieve full history or an exact stored event.',('file_read','file_view','session_history','session_record')),
    'workspace': ('Read/search and patch files in the selected workspace.',tuple(name for name in WORKSPACE_TOOLS if name.startswith('workspace_'))),
    'terminal': ('Visible, non-interactive PowerShell executions, output and cancellation.',tuple(name for name in WORKSPACE_TOOLS if name.startswith('terminal_'))),
    'automation': ('Ordered repeated input and independent parallel reads.',
                   ('desktop_macro','browser_macro','tool_parallel','job_status','job_result','job_cancel')),
}
BASE_TOOLS = ('finish','window_list','window_select','desktop_capture','desktop_screen_capture',
              'desktop_click','desktop_type','desktop_key','desktop_scroll','desktop_drag')
ARGUMENT_DEFAULTS = {
    'desktop_click': {'button':'left','clicks':1},
    'desktop_record': {'fps':5}, 'browser_record': {'fps':5},
    'desktop_input': {'mode':'general','x':0,'y':0,'button':'left','key':'','text':'','amount':0,'hold_ms':0},
    'desktop_hold': {'mode':'general','x':0,'y':0,'button':'left','key':''},
    'desktop_drag_timed': {'mode':'general','button':'left','duration_ms':500},
    'browser_hold': {'key':'','button':'left'}, 'browser_drag': {'duration_ms':500},
    'file_read': {'offset':0,'limit':3000}, 'session_history': {'offset':0,'limit':3000},
    'session_record': {'offset':0,'limit':6000},
    'workspace_read': {'start_line':1,'end_line':200},
    'workspace_search': {'path':'.','limit':100},
    'workspace_code_search': {'path':'.','limit':8},
    'workspace_symbols': {'path':'.','kind':'definitions','limit':30},
    'terminal_start': {'cwd':'.','timeout_seconds':600},
    'terminal_output': {'offset':0,'limit':12000},
    'desktop_macro': {'repeat':1,'interval_ms':0},
    'browser_macro': {'repeat':1,'interval_ms':0},
}


class ToolCatalog:
    def __init__(self, capabilities, prompt='', attachments=False):
        self.capabilities = capabilities
        self.loaded = set()
        hints = {'browser':r'browser|web|page|search|\ube0c\ub77c\uc6b0\uc800|\uc6f9|\ud398\uc774\uc9c0|\uac80\uc0c9',
                 'recording':r'record|video|\ub179\ud654|\ub3d9\uc601\uc0c1',
                 'advanced_input':r'hold|drag|queue|scancode|\uae38\uac8c|\ub204\ub974\uace0|\ub4dc\ub798\uadf8|\ube44\ud65c\uc131|\ud0a4 \ud050|\uc7a5\uce58 \ubc29\uc2dd',
                 'files':r'file|attach|history|\ucca8\ubd80|\ud30c\uc77c|\ubb38\uc11c|\uae30\ub85d',
                 'workspace':r'file|code|workspace|edit|patch|\ud30c\uc77c|\ud3b8\uc9d1|\ucf54\ub4dc',
                 'terminal':r'terminal|powershell|shell|command|\ud130\ubbf8\ub110|\uba85\ub839|\uc2e4\ud589',
                 'automation':r'macro|repeat|parallel|queue|\ub9e4\ud06c\ub85c|\ubc18\ubcf5|\ubcd1\ub82c|\ud050|\uac04\uaca9'}
        for group,pattern in hints.items():
            if (re.search(pattern,prompt,re.I) or (group == 'files' and attachments)) and self.available(group):
                self.loaded.add(group)

    def permitted(self, name):
        if tool_policy(self.capabilities.get('tool_policies',{}),name)=='disabled':
            return False
        if name in WORKSPACE_TOOLS:
            return bool(self.capabilities.get('workspace_root'))
        if name in ('window_list','window_select'):
            return bool(self.capabilities.get('screen') or self.capabilities.get('input'))
        if name.startswith('desktop_'):
            return bool(self.capabilities.get('screen' if name in CAPTURE_TOOLS else 'input'))
        if name.startswith('browser_'):
            return bool(self.capabilities.get('browser'))
        return True

    def available(self, group):
        return any(self.permitted(name) for name in TOOL_GROUPS[group][1] if not name.startswith('job_'))

    def names(self):
        selected = list(BASE_TOOLS)
        for group in TOOL_GROUPS:
            if group in self.loaded:
                selected.extend(TOOL_GROUPS[group][1])
        return tuple(dict.fromkeys(['load_tool_group']+[name for name in selected if self.permitted(name)]))

    def load(self, groups):
        normalize_call({'tool':'load_tool_group','arguments':{'groups':groups}})
        if any(not self.available(group) for group in groups):
            raise ValueError('Tool group disabled by current permissions')
        self.loaded.update(groups)
        return {'loaded':sorted(self.loaded),'tools':list(self.names())}

    def normalize(self, call):
        allowed = self.names()
        if isinstance(call,dict) and call.get('tool') == 'job_start':
            nested = call.get('arguments',{}).get('action',{})
            if nested.get('tool') not in allowed:
                raise ValueError('Background tool not loaded or not permitted')
            allowed = (*allowed,'job_start')
        action = normalize_call(call,allowed)
        def check_children(parent):
            children = [parent['arguments']['action']] if parent['tool']=='job_start' else parent['arguments'].get('actions',[])
            for child in children:
                if child['tool'] not in allowed:
                    raise ValueError('Child tool not loaded or not permitted')
                check_children(child)
        check_children(action)
        return action


def compact_call(action):
    name,arguments = action['tool'],dict(action['arguments'])
    if name == 'finish':
        return {'tool':'finish','arguments':{'text':action['message']}}
    if name == 'job_start':
        nested = compact_call(arguments['action'])
        nested['arguments']['background'] = True
        if action['message'] not in ('Start '+arguments['action']['tool'],'job_start'):
            nested['arguments']['commentary']=action['message']
        return nested
    if name in COMPOSITE_TOOLS:
        arguments['actions'] = [compact_call(child) for child in arguments['actions']]
    keep = {'x','y'} if arguments.get('kind') in ('click','mouse','scroll') or arguments.get('mode') == 'message' else set()
    for key,value in ARGUMENT_DEFAULTS.get(name,{}).items():
        if key not in keep and arguments.get(key) == value:
            arguments.pop(key,None)
    if name in ('desktop_scroll','browser_scroll'):
        amount = arguments['amount']
        arguments['direction'] = ('up' if amount >= 0 else 'down') if name == 'desktop_scroll' else ('down' if amount >= 0 else 'up')
        arguments['amount'] = abs(amount)
    if action['message'] not in (name,'Load tool group'):
        arguments['commentary']=action['message']
    return {'tool':name,'arguments':arguments}


def tool_description(name):
    overrides = {'finish':'Answer user; no execution.',
                 'load_tool_group':'Load group specifications; does not execute them.',
                 'desktop_input':'Advanced input: kind=click/key/text/scroll. mode defaults general; device uses scancodes, message targets a child without focus/cursor movement. key/text required for its kind; x/y required for mouse or message mode.',
                 'desktop_scroll':'Scroll selected window at x/y; direction up/down, positive amount.',
                 'browser_scroll':'Scroll managed page; direction up/down, positive amount.'}
    description = overrides.get(name,TOOLS.get(name,('',))[0])
    if name.startswith('desktop_') and 'x' in TOOLS.get(name,('',{}))[1]:
        description = description.replace('x/y normalized 0..1000 over full visible window bounds','x/y in the reference image pixels')
        description += ' Use image_pixels, origin top-left; do not normalize or add desktop offsets.'
    return description


def compact_parameters(name):
    if name == 'finish':
        return {'type':'object','properties':{'text':text(5000)},'required':['text'],'additionalProperties':False}
    if name == 'load_tool_group':
        return {'type':'object','properties':{'groups':{'type':'array','items':{'type':'string','enum':list(TOOL_GROUPS)},
                'minItems':1,'maxItems':len(TOOL_GROUPS)},'commentary':text(1200)},'required':['groups'],'additionalProperties':False}
    from copy import deepcopy
    parameters = deepcopy(TOOLS[name][1])
    if name in COMPOSITE_TOOLS:
        parameters['actions']['items'] = compact_schema(COMPOSITE_TOOLS[name])
        for child in parameters['actions']['items']['oneOf']:
            child['properties']['arguments']['properties'].pop('background',None)
    pixel_tool = name.startswith('desktop_') and 'x' in parameters
    if pixel_tool:
        for key in ('x','y','end_x','end_y'):
            if key in parameters:
                parameters[key] = dict(PIXEL_POINT)
        parameters['coordinate_space'] = {'type':'string','enum':['image_pixels']}
    defaults = ARGUMENT_DEFAULTS.get(name,{})
    for key,spec in parameters.items():
        if key == 'key':
            parameters[key] = text(40)
        if key == 'steps':
            spec['items']['properties']['key'] = text(40)
            spec['items']['required'] = ['key']
        if key in defaults:
            parameters[key]['default'] = defaults[key]
    if name in ('desktop_scroll','browser_scroll'):
        parameters['amount']['minimum'] = 1
        parameters['direction'] = {'type':'string','enum':['up','down']}
    if name in BACKGROUND_TOOLS:
        parameters['background'] = {'type':'boolean','default':False}
    parameters['commentary']=dict(text(1200),description='Brief user-facing finding from the preceding result and next action. Not hidden reasoning. Optional; use the user language.')
    required = [key for key in TOOLS[name][1] if key not in defaults]
    if pixel_tool:
        required.append('coordinate_space')
    if name in ('desktop_scroll','browser_scroll'):
        required.append('direction')
    return {'type':'object','properties':parameters,'required':required,'additionalProperties':False}


def compact_schema(names):
    return {'oneOf':[{'type':'object','properties':{'tool':{'type':'string','enum':[name]},
                     'arguments':compact_parameters(name)},'required':['tool','arguments'],'additionalProperties':False}
                    for name in names]}


class InvalidToolCall(ValueError):
    pass


def normalize_call(call, allowed=None):
    if not isinstance(call,dict):
        raise ValueError('Expected tool and arguments')
    legacy = set(call) == {'message','tool','arguments','risk'}
    name = call.get('tool')
    if not isinstance(name,str) or (name not in TOOLS and name != 'load_tool_group'):
        raise ValueError('Unknown tool')
    if allowed is not None and name not in allowed:
        raise ValueError('Tool not loaded or not permitted; use load_tool_group for an available group')
    if legacy and name == 'load_tool_group':
        return normalize_call({'tool':name,'arguments':dict(call['arguments'],commentary=call['message'])},allowed)
    if legacy:
        return validate_action(call)
    if set(call)-{'tool','arguments'}:
        raise ValueError('Only tool and arguments are accepted')
    arguments = call.get('arguments',{})
    spec = compact_parameters(name)
    required = set(spec['required'])-{'coordinate_space'}
    if not isinstance(arguments,dict) or set(arguments)-set(spec['properties']) or required-set(arguments):
        raise ValueError('Unexpected or missing arguments for '+name+'; required: '+', '.join(spec['required']))
    arguments=dict(arguments)
    commentary=arguments.pop('commentary','')
    if not isinstance(commentary,str) or len(commentary)>1200:
        raise ValueError('Invalid public commentary')
    if name == 'load_tool_group':
        validate_argument(arguments['groups'],spec['properties']['groups'],'groups')
        return dict(message=commentary or 'Load tool group',tool=name,arguments=arguments,risk='routine')
    if name == 'finish':
        return validate_action(dict(message=arguments['text'],tool=name,arguments={},risk='routine'))
    values = dict(ARGUMENT_DEFAULTS.get(name,{}),**arguments)
    if name in COMPOSITE_TOOLS:
        if not isinstance(values['actions'],list):
            raise ValueError('Invalid actions')
        values['actions'] = [normalize_call(child,COMPOSITE_TOOLS[name]) for child in values['actions']]
    if 'steps' in values:
        if not isinstance(values['steps'],list) or any(not isinstance(step,dict) for step in values['steps']):
            raise ValueError('Invalid key steps')
        values['steps'] = [dict(delay_ms=0,**step) if 'delay_ms' not in step else dict(step) for step in values['steps']]
    if name in ('desktop_input','desktop_hold','browser_hold'):
        kind = values.get('kind')
        required = ['key'] if kind == 'key' else ['text'] if kind == 'text' else []
        if name != 'browser_hold' and (kind in ('click','scroll','mouse') or values['mode'] == 'message'):
            required += ['x','y']
        if kind == 'scroll':
            required += ['amount']
        if any(key not in arguments for key in required):
            raise ValueError('Missing input fields: '+', '.join(required))
    if name in ('desktop_scroll','browser_scroll'):
        direction = values.pop('direction')
        if direction not in ('up','down') or type(values['amount']) is not int or values['amount'] <= 0:
            raise ValueError('Scroll requires up/down and a positive amount')
        positive = 'up' if name == 'desktop_scroll' else 'down'
        values['amount'] *= 1 if direction == positive else -1
    background = values.pop('background',False) if name in BACKGROUND_TOOLS else False
    if type(background) is not bool:
        raise ValueError('background must be boolean')
    action = validate_action(dict(message=commentary or name,tool=name,arguments=values,risk='routine'))
    if background:
        return validate_action(dict(message=commentary or 'Start '+name,tool='job_start',arguments={'action':action},risk='routine'))
    return action


def require_pixel_coordinates(action):
    if action['tool']=='job_start':
        return require_pixel_coordinates(action['arguments']['action'])
    name,arguments = action['tool'],action['arguments']
    if name in COMPOSITE_TOOLS:
        for child in arguments['actions']:
            require_pixel_coordinates(child)
        return
    if not name.startswith('desktop_') or 'x' not in TOOLS.get(name,('',{}))[1]:
        return
    if name in ('desktop_input','desktop_hold') and arguments.get('kind') in ('key','text') and arguments.get('mode')!='message':
        return
    if arguments.get('coordinate_space')!='image_pixels':
        raise ValueError('New desktop input must specify coordinate_space=image_pixels; no input executed')


def validate_argument(value, spec, key):
    if 'oneOf' in spec:
        validate_action(value)
        allowed = {choice['properties']['tool']['enum'][0] for choice in spec['oneOf']}
        if value['tool'] not in allowed:
            raise ValueError('Tool not allowed in this action group')
    elif spec['type'] == 'integer':
        if type(value) is not int or not spec['minimum'] <= value <= spec['maximum']:
            raise ValueError('Invalid numeric argument: '+key)
    elif spec['type'] == 'array':
        if not isinstance(value,list) or not spec['minItems'] <= len(value) <= spec['maxItems']:
            raise ValueError('Invalid queue length: '+key)
        for item in value:
            validate_argument(item,spec['items'],key)
    elif spec['type'] == 'object':
        if not isinstance(value,dict) or set(value) != set(spec['properties']):
            raise ValueError('Invalid queue step: '+key)
        for name, parameter in spec['properties'].items():
            validate_argument(value[name],parameter,name)
    elif not isinstance(value,str) or len(value) > spec.get('maxLength',10000):
        raise ValueError('Invalid text argument: '+key)
    if 'enum' in spec and value not in spec['enum']:
        raise ValueError('Unsupported value: '+key)


def validate_action(action):
    if not isinstance(action, dict) or set(action) != {'message', 'tool', 'arguments', 'risk'}:
        raise ValueError('Expected message, tool, arguments and risk')
    if not isinstance(action['message'], str) or len(action['message']) > 5000:
        raise ValueError('Invalid message')
    name, arguments = action['tool'], action['arguments']
    if not isinstance(name, str) or name not in TOOLS or not isinstance(arguments, dict):
        raise ValueError('Unknown tool or invalid arguments')
    if action['risk'] not in ('routine', 'sensitive'):
        raise ValueError('Invalid action risk: expected routine or sensitive; no tool executed from this response')
    pixel_tool = name.startswith('desktop_') and 'x' in TOOLS[name][1]
    pixel = pixel_tool and arguments.get('coordinate_space')=='image_pixels'
    expected_keys = set(TOOLS[name][1])|({'coordinate_space'} if pixel else set())
    if set(arguments) != expected_keys:
        expected = ', '.join(TOOLS[name][1]) or '(empty object {})'
        raise ValueError('Unexpected arguments for '+name+'; expected exactly '+expected+'; no tool executed from this response')
    for key, spec in TOOLS[name][1].items():
        if pixel and key in ('x','y','end_x','end_y'):
            spec = PIXEL_POINT
        validate_argument(arguments[key],spec,key)
    if 'steps' in arguments and sum(step['delay_ms'] for step in arguments['steps']) > 60000:
        raise ValueError('Queue delay exceeds 60000ms')
    if name in MACRO_TOOLS:
        if arguments['repeat']*len(arguments['actions']) > 10000 or (arguments['repeat']-1)*arguments['interval_ms'] > 3600000:
            raise ValueError('Macro exceeds 10000 actions or one hour of interval waits')
    if name == 'finish' and not action['message'].strip():
        raise ValueError('Empty final reply')
    if 'selector' in arguments and not arguments['selector'].strip():
        raise ValueError('Empty selector')
    if name in ('desktop_input','desktop_hold','browser_hold') and arguments['kind'] == 'key' and not arguments['key']:
        raise ValueError('Key is required for keyboard input')
    return action


def web_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Only HTTP(S) URLs without embedded credentials are supported')
    return value


SENSITIVE = re.compile(r'\b(delete|erase|remove|purchase|checkout|pay|buy|send|submit|publish|upload|transfer|install|permission|administrator|password|authorize|terminal|powershell|registry)\b|sign.?in|log.?in|command prompt|\uc0ad\uc81c|\uacb0\uc81c|\uad6c\ub9e4|\uc804\uc1a1|\uc81c\ucd9c|\uac8c\uc2dc|\uacc4\uc815|\ube44\ubc00\ubc88\ud638|\uad8c\ud55c|\uc124\uce58|\uc1a1\uae08|\ub85c\uadf8\uc778', re.I)
RISKY_KEYS = {'Delete', 'Shift+Delete', 'Alt+F4', 'Control+w', 'Control+Enter'}


def target_approval_reason(context):
    if context.get('password'):
        return 'Sensitive input field'
    if SENSITIVE.search(context.get('label','')):
        return 'Potential deletion, payment, external submission or system/account change'
    return ''


def approval_reason(action, mode, context=None):
    validate_action(action)
    context = context or {}
    name, arguments = action['tool'], action['arguments']
    policy=tool_policy(context.get('tool_policies',{}),name)
    if policy=='disabled':
        raise ValueError('Tool disabled by user: '+name)
    if policy=='ask':
        return 'User requires approval for this tool'
    if name in ('workspace_apply_patch','terminal_start'):
        return 'Confirm file changes or PowerShell execution; the shell is not sandboxed'
    if name in ('finish', 'browser_read', 'browser_tabs', 'window_list'):
        return ''
    if mode != 'routine':
        return 'Manual approval mode'
    if action['risk'] == 'sensitive':
        return 'Model marked the action as sensitive'
    if SENSITIVE.search(context.get('request_intent','')):
        return 'User request includes a potentially sensitive operation'
    if arguments.get('key') in RISKY_KEYS or any(step['key'] in RISKY_KEYS for step in arguments.get('steps',[])):
        return 'Destructive or window-closing key combination'
    if SENSITIVE.search(action['message']+' '+json.dumps(arguments, ensure_ascii=False)):
        return 'Potential deletion, payment, external submission or system/account change'
    target_reason = target_approval_reason(context)
    if target_reason:
        return target_reason
    if name in ('browser_click','browser_hold') and context.get('submit') and not context.get('search'):
        return 'Non-search form submission'
    if name == 'browser_open':
        web_url(arguments['url'])
    return ''