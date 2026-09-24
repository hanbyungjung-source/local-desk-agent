import ctypes
from ctypes import wintypes

from game_agent import windows
from game_agent.core import Halted, INPUT_TAG
from desktop_agent.tools import physical_pixels


KEY_CODES = {'Enter':13,'Tab':9,'Escape':27,'Backspace':8,'Delete':46,'Space':32,
             'ArrowUp':38,'ArrowDown':40,'ArrowLeft':37,'ArrowRight':39,'Home':36,'End':35,
             'PageUp':33,'PageDown':34,'Control':17,'Shift':16,'Alt':18,'F4':115}
KEY_CODES.update({'F'+str(number):111+number for number in range(1,13)})
EXTENDED = {33,34,35,36,37,38,39,40,46}
windows.user32.MapVirtualKeyW.argtypes = [wintypes.UINT,wintypes.UINT]
windows.user32.ScreenToClient.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.POINT)]
windows.user32.ClientToScreen.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.POINT)]
windows.user32.ChildWindowFromPointEx.argtypes = [wintypes.HWND,wintypes.POINT,wintypes.UINT]
windows.user32.ChildWindowFromPointEx.restype = wintypes.HWND
windows.user32.SendMessageTimeoutW.argtypes = [wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM,
                                              wintypes.UINT,wintypes.UINT,ctypes.POINTER(ctypes.c_size_t)]
windows.user32.SendMessageTimeoutW.restype = ctypes.c_ssize_t


def key_code(key):
    return KEY_CODES[key] if key in KEY_CODES else ord(key.upper())


def scan_key(backend, chord, hold_ms, stopped):
    pressed = []
    try:
        for key in chord.split('+'):
            backend._check()
            code = key_code(key)
            scan = windows.user32.MapVirtualKeyW(code,0)
            if not scan:
                raise ValueError('No scancode for key: '+key)
            flags = 8 | (1 if code in EXTENDED else 0)
            backend._send(windows.Input(1,windows.InputUnion(keyboard=windows.KeyboardInput(0,scan,flags,0,INPUT_TAG))))
            pressed.append((scan,flags))
        if stopped.wait(hold_ms/1000):
            raise Halted('Device-style key hold cancelled')
    finally:
        for scan,flags in reversed(pressed):
            backend._send(windows.Input(1,windows.InputUnion(keyboard=windows.KeyboardInput(0,scan,flags|2,0,INPUT_TAG))))


def move_pointer(backend, point):
    backend._check(point)
    left,top,width,height = windows.virtual_screen()
    backend._send(windows.Input(0,windows.InputUnion(mouse=windows.MouseInput(
        int((point[0]-left+0.5)*65536/width),int((point[1]-top+0.5)*65536/height),0,0xc001,0,INPUT_TAG))))


def timed_drag(backend, start, end, button, duration_ms, stopped):
    move_pointer(backend,start)
    pressed = False
    try:
        backend.down('click',button)
        pressed = True
        steps = max(1,duration_ms//20)
        for step in range(1,steps+1):
            if stopped.wait(duration_ms/steps/1000):
                raise Halted('Drag cancelled')
            point = tuple(round(start[axis]+(end[axis]-start[axis])*step/steps) for axis in (0,1))
            move_pointer(backend,point)
    finally:
        if pressed:
            backend.up('click',button)


@physical_pixels()
def message_input(window, arguments, stopped, *, coordinate_frame=None):
    from desktop_agent.tools import visible_window_region
    from desktop_agent.coordinates import input_region
    root = window['handle']
    def check(handle):
        process = wintypes.DWORD()
        windows.user32.GetWindowThreadProcessId(handle,ctypes.byref(process))
        if not windows.user32.IsWindow(handle) or process.value != window['pid']:
            raise Halted('Window message target closed or changed process')
    check(root)
    if windows.user32.IsIconic(root):
        raise ValueError('Window messages need a non-minimized target for coordinate mapping')
    rect = windows.window_rect(root)
    region = visible_window_region(root)
    region = input_region(region,arguments,coordinate_frame,window)
    screen = region.point(arguments['x'],arguments['y'])
    handle = root
    for depth in range(32):
        point = wintypes.POINT(*screen)
        if not windows.user32.ScreenToClient(handle,ctypes.byref(point)):
            raise ValueError('Cannot map window-message coordinates')
        child = windows.user32.ChildWindowFromPointEx(handle,point,7)
        if not child or child == handle:
            break
        handle = child
    check(handle)
    point = wintypes.POINT(*screen)
    if not windows.user32.ScreenToClient(handle,ctypes.byref(point)):
        raise ValueError('Cannot map child control coordinates')
    if not -32768 <= point.x <= 32767 or not -32768 <= point.y <= 32767:
        raise ValueError('Window-message coordinates out of range')
    packed = (point.y & 0xffff)<<16 | (point.x & 0xffff)
    def send(message, parameter=0, location=0, release=False):
        if stopped.is_set() and not release:
            raise Halted('Window message cancelled')
        check(handle)
        if windows.window_rect(root) != rect:
            raise Halted('Window geometry changed during message input')
        result = ctypes.c_size_t()
        if not windows.user32.SendMessageTimeoutW(handle,message,parameter,location,2,300,ctypes.byref(result)):
            raise ValueError('Target rejected or timed out receiving window message')
    kind = arguments['kind']
    if kind in ('click','drag'):
        down,up,mask = (0x201,0x202,1) if arguments['button'] == 'left' else (0x204,0x205,2)
        send(down,mask,packed)
        try:
            if kind == 'drag':
                endpoint = wintypes.POINT(*region.point(arguments['end_x'],arguments['end_y']))
                if not windows.user32.ScreenToClient(handle,ctypes.byref(endpoint)):
                    raise ValueError('Cannot map drag endpoint')
                steps = max(1,arguments['duration_ms']//20)
                for step in range(1,steps+1):
                    if stopped.wait(arguments['duration_ms']/steps/1000):
                        raise Halted('Window drag cancelled')
                    horizontal = round(point.x+(endpoint.x-point.x)*step/steps)
                    vertical = round(point.y+(endpoint.y-point.y)*step/steps)
                    packed = ((vertical&0xffff)<<16)|(horizontal&0xffff)
                    send(0x200,mask,packed)
            elif stopped.wait(arguments['hold_ms']/1000):
                raise Halted('Window click cancelled')
        finally:
            send(up,0,packed,release=True)
    elif kind == 'scroll':
        position = (screen[1] & 0xffff)<<16 | (screen[0] & 0xffff)
        send(0x20a,((arguments['amount']*120)&0xffff)<<16,position)
    elif kind == 'text':
        units = arguments['text'].encode('utf-16-le')
        for offset in range(0,len(units),2):
            send(0x102,int.from_bytes(units[offset:offset+2],'little'),1)
    else:
        key = arguments['key']
        if '+' in key:
            raise ValueError('Window-message modifier chords cannot reproduce keyboard state; use general/device')
        code = key_code(key)
        scan = windows.user32.MapVirtualKeyW(code,0)
        flags = 1 | (scan<<16) | ((1<<24) if code in EXTENDED else 0)
        send(0x100,code,flags)
        try:
            if code in (8,9,13,32):
                send(0x102,code,flags)
            if stopped.wait(arguments['hold_ms']/1000):
                raise Halted('Window key cancelled')
        finally:
            send(0x101,code,flags|(3<<30),release=True)
    return f'Message delivered to child HWND {handle}; cursor/focus not moved. Application acceptance is NOT guaranteed.'