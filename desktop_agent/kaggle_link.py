import base64
from pathlib import Path
import shutil
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from desktop_agent.credentials import KeyVault, credential_scope, protect


STATE = Path(__file__).resolve().parent / 'data'
PRIVATE_KEY = STATE / 'kaggle-link-private.dpapi'
PROFILE_NAME = 'Kaggle T4x2 Qwen3.8 Q4 MTP'


class ConnectionCheckError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__('Kaggle connection check failed: '+str(code))


def connection_error_message(error):
    code = error.code if isinstance(error, ConnectionCheckError) else None
    if code in (401, 403):
        return 'HTTP '+str(code)+': API \ud0a4 \uc778\uc99d\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4. Kaggle Secrets\uc640 \uac19\uc740 \ud0a4\uc778\uc9c0 \ud655\uc778\ud558\uc138\uc694.'
    if code in (502, 503, 504):
        return 'HTTP '+str(code)+': \ud130\ub110 \ub610\ub294 \uc11c\ubc84\uc5d0 \uc5f0\uacb0\ud560 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4. Kaggle\uc5d0\uc11c \uc11c\ubc84\uc640 \ud130\ub110 \uc0c1\ud0dc, \ucd5c\uc2e0 API_URL\uc744 \ud655\uc778\ud558\uc138\uc694.'
    if code == 'network':
        return '\ub124\ud2b8\uc6cc\ud06c \uc5f0\uacb0 \uc2e4\ud328: \uc8fc\uc18c\u00b7DNS\u00b7TLS \ub610\ub294 \uc5f0\uacb0 \uc2dc\uac04 \ucd08\uacfc\ub97c \ud655\uc778\ud558\uc138\uc694.'
    if isinstance(code, int):
        return 'HTTP '+str(code)+': Kaggle \uc5f0\uacb0 \ud655\uc778\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4.'
    return '\uc5f0\uacb0\ud558\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4. \uc8fc\uc18c\u00b7\ud0a4\u00b7\ubaa8\ub378 \ub610\ub294 \uc124\uc815 \uc800\uc7a5 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uc138\uc694. \ud0a4\ub97c \ube44\uc6b0\uba74 \uae30\uc874 \uc800\uc7a5 \ud0a4\ub97c \uc0ac\uc6a9\ud569\ub2c8\ub2e4.'


def normalize_tunnel_url(url):
    import re
    from urllib.parse import urlsplit
    from desktop_agent.kaggle_server import is_ngrok_host
    parsed = urlsplit(url.strip())
    if (parsed.scheme != 'https' or parsed.username or parsed.password or
            parsed.query or parsed.fragment or
            not (re.fullmatch(r'[a-z0-9][a-z0-9-]*\.(?:lhr\.life|localhost\.run)', parsed.netloc) or
                 is_ngrok_host(parsed.netloc)) or
            parsed.path.rstrip('/') not in ('', '/v1', '/v1/chat/completions')):
        raise ValueError('Use the HTTPS API_URL issued by the Kaggle tunnel')
    return 'https://'+parsed.netloc+'/v1'


def verify_connection(url, key):
    import httpx
    import json
    import time
    from desktop_agent.kaggle_server import ALIAS, is_ngrok_host
    from urllib.parse import urlsplit
    url = normalize_tunnel_url(url)
    if (not isinstance(key, str) or not key or len(key) > 4096 or
            not key.isascii() or any(ord(char) < 33 or ord(char) > 126 for char in key)):
        raise ValueError('A single ASCII API key is required')
    try:
        deadline = time.monotonic()+15
        headers = {'Authorization':'Bearer '+key}
        if is_ngrok_host(urlsplit(url).hostname):
            headers['ngrok-skip-browser-warning'] = '1'
        with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
            with client.stream('GET', url+'/models', headers=headers) as response:
                if response.status_code != 200:
                    raise ConnectionCheckError(response.status_code)
                body = bytearray()
                for chunk in response.iter_bytes():
                    if time.monotonic() > deadline:
                        raise ValueError('Kaggle connection check took too long')
                    body.extend(chunk)
                    if len(body) > 65536:
                        raise ValueError('Unexpected model response size')
        data = json.loads(body)
        if (not isinstance(data, dict) or not isinstance(data.get('data'), list) or
                not any(isinstance(item, dict) and item.get('id') == ALIAS for item in data['data'])):
            raise ValueError('Unexpected Kaggle model identity')
    except httpx.HTTPError:
        raise ConnectionCheckError('network') from None
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError('Invalid Kaggle model response') from None


def connect_profile(url, key='', *, state=None, cancel=None):
    from dataclasses import replace
    from desktop_agent.agent import Settings
    from desktop_agent.api import APISettings, endpoint
    from desktop_agent.kaggle_server import ALIAS
    state = Path(state) if state is not None else STATE
    settings_path = state / 'settings.json'
    settings = Settings.load(settings_path)
    previous = settings.api_profiles.get(PROFILE_NAME)
    if previous and (previous.model != ALIAS or previous.vertex or previous.format != 'openai'):
        raise ValueError('The Kaggle profile name is already used by another provider')
    config = replace(previous, url=normalize_tunnel_url(url), context_tokens=196608) if previous else APISettings(
        url=normalize_tunnel_url(url), model=ALIAS, context_tokens=196608,
        max_output_tokens=-1, tokenizer='estimate', reasoning_effort='xhigh',
        timeout_seconds=10800, max_retries=0)
    config = replace(config, key_profile_id=config.key_profile_id or uuid.uuid4().hex)
    vault = KeyVault(state / 'api-keys.dpapi')
    keys = [key] if key else vault.load(credential_scope(endpoint(previous), previous.key_profile_id)) if previous else []
    if not keys:
        raise ValueError('Enter LOCAL_DESK_API_KEY in the masked input')
    config.validate()
    if cancel is not None and cancel.is_set():
        raise InterruptedError('Connection cancelled')
    verify_connection(config.url, keys[0])
    if cancel is not None and cancel.is_set():
        raise InterruptedError('Connection cancelled')
    updated = settings.with_api_profile(PROFILE_NAME, config).use_api_profile(PROFILE_NAME)
    state.mkdir(parents=True, exist_ok=True)
    backup = state / 'settings.before-kaggle-connect.json'
    if settings_path.exists() and not backup.exists():
        shutil.copy2(settings_path, backup)
    scope = credential_scope(endpoint(config), config.key_profile_id)
    original_keys = vault.load(scope)
    vault.save(scope, keys)
    try:
        updated.save(settings_path)
    except BaseException:
        vault.save(scope, original_keys)
        raise
    return config


def desktop_is_running():
    import os
    import psutil
    excluded = {os.getpid(), *(process.pid for process in psutil.Process().parents())}
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.info['pid'] in excluded:
            continue
        command = process.info.get('cmdline') or []
        name = (process.info.get('name') or '').lower()
        if name not in ('python.exe', 'pythonw.exe'):
            continue
        modules = [command[index+1] for index, argument in enumerate(command[:-1]) if argument == '-m']
        if ('desktop_agent.app' in modules or
                ('desktop_agent.kaggle_link' in modules and '--connect' in command)):
            return True
    return False


class ConnectionDialog:
    def __init__(self, root, *, state=None):
        import queue
        import threading
        import tkinter as tk
        from tkinter import ttk
        from desktop_agent.agent import Settings
        self.root = root
        self.state = Path(state) if state is not None else STATE
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.connected = False
        self.failure_message = ''
        settings = Settings.load(self.state / 'settings.json')
        previous = settings.api_profiles.get(PROFILE_NAME)
        self.url = tk.StringVar(root, value=previous.url if previous else '')
        self.key = tk.StringVar(root)
        self.status = tk.StringVar(root, value='\uc11c\ubc84 \uc8fc\uc18c\uc640 API \ud0a4\uac00 \ud544\uc694\ud569\ub2c8\ub2e4.')
        root.title('Local Desk - Kaggle')
        root.geometry('640x250')
        root.minsize(480, 250)
        root.columnconfigure(0, weight=1)
        body = ttk.Frame(root, padding=16)
        body.grid(sticky='nsew')
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text='API URL').grid(row=0, column=0, sticky='w', padx=(0, 12), pady=6)
        self.url_entry = ttk.Entry(body, textvariable=self.url)
        self.url_entry.grid(row=0, column=1, sticky='ew', pady=6)
        ttk.Label(body, text='API \ud0a4 (\uc120\ud0dd)').grid(row=1, column=0, sticky='w', padx=(0, 12), pady=6)
        self.key_entry = ttk.Entry(body, textvariable=self.key, show='*')
        self.key_entry.grid(row=1, column=1, sticky='ew', pady=6)
        ttk.Label(body, textvariable=self.status, wraplength=420).grid(row=2, column=0, columnspan=2, sticky='w', pady=12)
        actions = ttk.Frame(body)
        actions.grid(row=3, column=0, columnspan=2, sticky='e')
        self.submit_button = ttk.Button(actions, text='\uc5f0\uacb0 \ud6c4 \uc571 \uc2dc\uc791', command=self.submit)
        self.submit_button.pack(side='right', padx=(8, 0))
        ttk.Button(actions, text='\ucde8\uc18c', command=self.close).pack(side='right')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Return>', lambda event: self.submit())
        root.bind('<Escape>', lambda event: self.close())
        root.after(50, self.poll)
        if previous:
            root.after(100, self.submit)
        else:
            self.url_entry.focus_set()

    def submit(self):
        import threading
        if self.busy:
            return
        if desktop_is_running():
            self.status.set('\uae30\uc874 Local Desk\ub97c \uc644\uc804\ud788 \uc885\ub8cc\ud55c \ub4a4 \ub2e4\uc2dc \uc5f0\uacb0\ud558\uc138\uc694.')
            return
        url, key = self.url.get(), self.key.get()
        self.key.set('')
        self.busy = True
        self.failure_message = ''
        for widget in (self.url_entry, self.key_entry, self.submit_button):
            widget.configure(state='disabled')
        self.status.set('\uc5f0\uacb0 \ud655\uc778 \uc911...')
        def work():
            try:
                connect_profile(url, key, state=self.state, cancel=self.cancel)
            except Exception as error:
                self.failure_message = connection_error_message(error)
                self.events.put(False)
            else:
                self.events.put(True)
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        import queue
        try:
            result = self.events.get_nowait()
        except queue.Empty:
            self.root.after(50, self.poll)
            return
        self.busy = False
        if result or self.cancel.is_set():
            self.connected = result and not self.cancel.is_set()
            self.key.set('')
            self.root.destroy()
            return
        for widget in (self.url_entry, self.key_entry, self.submit_button):
            widget.configure(state='normal')
        self.status.set(self.failure_message or connection_error_message(None))
        self.url_entry.focus_set()
        self.root.after(50, self.poll)

    def close(self):
        if self.busy:
            self.cancel.set()
            self.status.set('\ucde8\uc18c \ucc98\ub9ac \uc911...')
        else:
            self.key.set('')
            self.root.destroy()


def connect_desktop():
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    try:
        if desktop_is_running():
            messagebox.showwarning('Local Desk', '\uae30\uc874 Local Desk\ub97c \uc644\uc804\ud788 \uc885\ub8cc\ud55c \ub4a4 \uc774 \ubc30\uce58\ud30c\uc77c\uc744 \uc2e4\ud589\ud558\uc138\uc694.', parent=root)
            root.destroy()
            return False
        dialog = ConnectionDialog(root)
    except Exception:
        messagebox.showerror('Local Desk', '\uc5f0\uacb0 \uc124\uc815\uc744 \uc5f4 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4. Python \ud658\uacbd\uacfc \uc800\uc7a5 \uc124\uc815\uc744 \ud655\uc778\ud558\uc138\uc694.', parent=root)
        root.destroy()
        return False
    root.deiconify()
    root.mainloop()
    return dialog.connected


def public_key():
    if PRIVATE_KEY.is_file():
        key = serialization.load_pem_private_key(protect(PRIVATE_KEY.read_bytes(), decrypt=True), None)
    else:
        key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())
        STATE.mkdir(parents=True, exist_ok=True)
        PRIVATE_KEY.write_bytes(protect(private))
    return key.public_key().public_bytes(serialization.Encoding.PEM,
                                         serialization.PublicFormat.SubjectPublicKeyInfo).decode('ascii')


def import_key(ciphertext, url):
    import requests
    from desktop_agent.api import APISettings, endpoint
    config = APISettings(url=url, model='qwen3.8-27b-ud-q4-k-xl',
                         context_tokens=196608, max_output_tokens=-1,
                         tokenizer='estimate', reasoning_effort='xhigh',
                         timeout_seconds=10800, max_retries=0)
    config.validate()
    private = serialization.load_pem_private_key(protect(PRIVATE_KEY.read_bytes(), decrypt=True), None)
    key = private.decrypt(base64.b64decode(ciphertext, validate=True),
                          padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                       algorithm=hashes.SHA256(), label=None)).decode('ascii')
    if not key or any(char.isspace() for char in key):
        raise ValueError('Invalid API key')
    scope = credential_scope(endpoint(config))
    response = requests.get(scope+'/v1/models', headers={'Authorization':'Bearer '+key},
                            timeout=30, allow_redirects=False)
    response.raise_for_status()
    if config.model not in [item['id'] for item in response.json()['data']]:
        raise ValueError('Unexpected remote model')
    KeyVault(STATE / 'api-keys.dpapi').save(scope, [key])
    return config


def save_profile(config, name='Kaggle T4x2 Qwen3.8 Q4 MTP'):
    from dataclasses import replace
    from desktop_agent.agent import Settings
    from desktop_agent.api import endpoint
    settings_path = STATE / 'settings.json'
    settings = Settings.load(settings_path)
    if name in settings.api_profiles:
        raise ValueError('Profile already exists; update it in API settings')
    config = replace(config, key_profile_id=uuid.uuid4().hex)
    updated = settings.with_api_profile(name, config)
    vault = KeyVault(STATE / 'api-keys.dpapi')
    keys = vault.load(credential_scope(endpoint(config)))
    if not keys:
        raise ValueError('Import the key before saving a profile')
    backup = STATE / 'settings.before-kaggle.json'
    if settings_path.exists() and not backup.exists():
        shutil.copy2(settings_path, backup)
    scope = credential_scope(endpoint(config), config.key_profile_id)
    vault.save(scope, keys)
    try:
        updated.save(settings_path)
    except BaseException:
        vault.save(scope, [])
        raise
    return name


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--connect', action='store_true')
    arguments = parser.parse_args()
    if arguments.connect:
        if connect_desktop():
            from desktop_agent.app import main as app_main
            sys.argv = [sys.argv[0]]
            app_main()
    else:
        print(public_key())


if __name__ == '__main__':
    main()