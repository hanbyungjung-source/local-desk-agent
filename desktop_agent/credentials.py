import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
import uuid


class Blob(ctypes.Structure):
    _fields_ = [('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]


def protect(data, decrypt=False):
    crypt = ctypes.WinDLL('crypt32',use_last_error=True)
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.POINTER(Blob),ctypes.c_void_p,
                         ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    buffer = (ctypes.c_ubyte*len(data)).from_buffer_copy(data)
    source, destination = Blob(len(data),buffer),Blob()
    if not function(ctypes.byref(source),None,None,None,None,1,ctypes.byref(destination)):
        raise ValueError('Windows credential protection failed; use the same Windows account')
    try:
        return ctypes.string_at(destination.data,destination.size)
    finally:
        kernel.LocalFree(destination.data)


def credential_scope(url, profile_id=''):
    parsed = urlsplit(url)
    origin = parsed.scheme.lower()+'://'+parsed.netloc.lower()
    if profile_id and not re.fullmatch(r'[a-f0-9]{32}',profile_id):
        raise ValueError('Invalid API key profile reference')
    return origin+('#profile:'+profile_id if profile_id else '')


def validate_vertex_account(info):
    from google.oauth2 import service_account
    fields = ('type','project_id','private_key_id','private_key','client_email','token_uri')
    if not isinstance(info,dict) or any(not isinstance(info.get(field),str) or not info[field] for field in fields):
        raise ValueError('Select a Google service-account JSON key file with project_id and private_key')
    if (info['type'] != 'service_account' or
            not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]',info['project_id']) or
            not re.fullmatch(r'[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.iam\.gserviceaccount\.com',info['client_email']) or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',info['private_key_id']) or
            info['token_uri'] != 'https://oauth2.googleapis.com/token' or
            info.get('universe_domain','googleapis.com') != 'googleapis.com'):
        raise ValueError('Invalid Google service-account identity or token endpoint')
    account = {field:info[field] for field in fields}
    try:
        service_account.Credentials.from_service_account_info(account,
            scopes=['https://www.googleapis.com/auth/cloud-platform'])
    except Exception:
        raise ValueError('Service-account JSON has an invalid private key') from None
    return account


def read_vertex_file(path):
    try:
        with Path(path).open('rb') as source:
            raw = source.read(65537)
        if len(raw) > 65536:
            raise ValueError()
        info = json.loads(raw.decode('utf-8-sig'))
    except (OSError,ValueError,UnicodeError):
        raise ValueError('Cannot read service-account JSON (UTF-8, maximum 64KB)') from None
    return validate_vertex_account(info)


class KeyVault:
    def __init__(self, path):
        self.path = Path(path)

    def read_all(self):
        if not self.path.exists():
            return {}
        data = self.path.read_bytes()
        if not data.startswith(b'LDKEY1'):
            raise ValueError('Invalid encrypted credential file')
        try:
            return json.loads(protect(data[6:],decrypt=True))
        except (ValueError,UnicodeError):
            raise ValueError('Cannot unlock API keys with this Windows account') from None

    def load(self, scope):
        return self.read_all().get(scope,[])

    def save(self, scope, keys):
        keys = list(dict.fromkeys(key.strip() for key in keys if key.strip()))
        if len(keys) > 20 or any(len(key) > 4096 or any(character.isspace() for character in key) for key in keys):
            raise ValueError('Maximum 20 keys; each key must be a single non-whitespace value')
        data = self.read_all()
        if keys:
            data[scope] = keys
        else:
            data.pop(scope,None)
        self.write_all(data)

    def save_vertex(self, info):
        account = validate_vertex_account(info)
        identifier = uuid.uuid4().hex
        data = self.read_all()
        data['vertex:'+identifier] = account
        self.write_all(data)
        return identifier

    def load_vertex(self, identifier):
        if not isinstance(identifier,str) or not re.fullmatch(r'[a-f0-9]{32}',identifier):
            raise ValueError('Import a Vertex service-account JSON key in API settings')
        account = self.read_all().get('vertex:'+identifier)
        if account is None:
            raise ValueError('Saved Vertex key is missing; import the service-account JSON again')
        return validate_vertex_account(account)

    def remove_vertex(self, identifier):
        data = self.read_all()
        data.pop('vertex:'+identifier,None)
        self.write_all(data)

    def write_all(self, data):
        encrypted = b'LDKEY1'+protect(json.dumps(data).encode('utf-8'))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_bytes(encrypted)
        temporary.replace(self.path)