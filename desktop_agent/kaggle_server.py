import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request


MODEL_REPO = 'unsloth/Qwen3.8-27B-GGUF'
MODEL_REVISION = '4ca720788d1e01f1bff70c033e0d0028fd02e502'
MODEL_FILE = 'Qwen3.8-27B-UD-Q4_K_XL.gguf'
PROJECTOR_FILE = 'mmproj-F16.gguf'
RUNTIME_TAG = 'v0.5.0'
RUNTIME_COMMIT = '7fe450e19305b828c199d602c23a8337aaa1f03b'
MMVQ_MAX_BATCH = '1'
ROOT = Path('/kaggle/temp/local-desk-qwen38')
# Kaggle "Files only" persistence keeps /kaggle/working between sessions.
CACHE = Path('/kaggle/working/local-desk-cache')
BUILD_FLAGS = ('-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=75',
               '-DBUILD_SHARED_LIBS=OFF',
               '-DCUDA_cuda_driver_LIBRARY=/usr/local/nvidia/lib64/libcuda.so')
ALIAS = 'qwen3.8-27b-ud-q4-k-xl'

# Same source patch as the T4 study (KAGGLE-T4-ENGINE.md): GGML_CUDA_MMVQ_MAX_BATCH overrides
# the MMVQ/MMQ batch threshold; unset keeps upstream behaviour.
MMVQ_INCLUDE_ANCHOR = '#include <type_traits>\n'
MMVQ_INCLUDES = '#include <type_traits>\n#include <array>\n#include <cstdlib>\n#include <string>\n'
MMVQ_ANCHOR = ('bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {\n'
               '    if (!ggml_is_quantized(type)) {\n'
               '        return false;\n'
               '    }\n')
MMVQ_OVERRIDE = (
    '    // LOCAL-DESK-T4: optional env override of the MMVQ/MMQ batch threshold; unset = upstream behaviour\n'
    '    static const std::array<int, GGML_TYPE_COUNT> t4_override = [] {\n'
    '        std::array<int, GGML_TYPE_COUNT> table;\n'
    '        const char * fallback = getenv("GGML_CUDA_MMVQ_MAX_BATCH");\n'
    '        const int base = fallback ? atoi(fallback) : -1;\n'
    '        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {\n'
    '            const char * name = ggml_type_name((ggml_type) i);\n'
    '            const std::string key = std::string("GGML_CUDA_MMVQ_MAX_BATCH_") + (name ? name : "");\n'
    '            const char * value = getenv(key.c_str());\n'
    '            table[i] = value ? atoi(value) : base;\n'
    '        }\n'
    '        return table;\n'
    '    }();\n'
    '    if (t4_override[type] >= 0) {\n'
    '        return ne11 <= t4_override[type];\n'
    '    }\n')


def patch_mmvq(source):
    path = Path(source) / 'ggml' / 'src' / 'ggml-cuda' / 'mmvq.cu'
    original = path.read_text(encoding='utf-8')
    text = original
    for anchor, replacement in ((MMVQ_INCLUDE_ANCHOR, MMVQ_INCLUDES),
                                (MMVQ_ANCHOR, MMVQ_ANCHOR + MMVQ_OVERRIDE)):
        if replacement in text:
            continue
        if text.count(anchor) != 1 or 'GGML_CUDA_MMVQ_MAX_BATCH' in text:
            raise RuntimeError('Unexpected mmvq.cu layout; refusing to patch')
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text, encoding='utf-8')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_key():
    compiler = subprocess.run(['nvcc', '--version'], check=True, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    recipe = {'tag': RUNTIME_TAG, 'commit': RUNTIME_COMMIT, 'flags': list(BUILD_FLAGS),
              'patch': [MMVQ_INCLUDES, MMVQ_ANCHOR, MMVQ_OVERRIDE], 'nvcc': compiler}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def restore_build(key):
    cached = CACHE / 'llama-server'
    try:
        record = json.loads((CACHE / 'build.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if (not isinstance(record, dict) or record.get('key') != key or not cached.is_file()
            or file_sha256(cached) != record.get('sha256')):
        return False
    binary = ROOT / 'llama.cpp/build/bin/llama-server'
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, binary)
    binary.chmod(0o755)
    return True


def store_build(key):
    CACHE.mkdir(parents=True, exist_ok=True)
    partial = CACHE / 'llama-server.partial'
    shutil.copy2(ROOT / 'llama.cpp/build/bin/llama-server', partial)
    record = {'key': key, 'sha256': file_sha256(partial)}
    os.replace(partial, CACHE / 'llama-server')
    (CACHE / 'build.json').write_text(json.dumps(record), encoding='utf-8')


def is_ngrok_host(host):
    import re
    return isinstance(host, str) and re.fullmatch(
        r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.(?:ngrok-free\.app|ngrok-free\.dev)', host) is not None


def inspect_environment():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=20,
    )
    devices = [
        {'index': int(row[0]), 'name': row[1].strip(),
         'total_mib': int(row[2]), 'free_mib': int(row[3])}
        for row in csv.reader(io.StringIO(result.stdout)) if row
    ]
    if len(devices) != 2 or any('T4' not in device['name'] for device in devices):
        raise RuntimeError('Exactly two NVIDIA T4 GPUs are required')
    if any(device['free_mib'] < 14000 for device in devices):
        raise RuntimeError('At least 14000 MiB free per GPU is required')
    scratch = Path('/kaggle/temp')
    scratch.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(scratch).free
    if free_bytes < 25 * 1024 ** 3:
        raise RuntimeError('At least 25 GiB scratch disk space is required')
    return {'python': platform.python_version(), 'devices': devices,
            'scratch_free_bytes': free_bytes, 'nvcc': shutil.which('nvcc')}


def prepare():
    environment = inspect_environment()
    if not environment['nvcc']:
        raise RuntimeError('CUDA compiler is required')
    ROOT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(environment), flush=True)
    key = build_key()
    cached = restore_build(key)
    source = ROOT / 'llama.cpp'
    if cached:
        print('BUILD_CACHE_HIT', flush=True)
    else:
        if not (source / '.git').is_dir():
            shutil.rmtree(source, ignore_errors=True)
            subprocess.run(['git', 'clone', '--depth', '1', '--branch', RUNTIME_TAG,
                            'https://github.com/ggml-org/llama.cpp.git', str(source)],
                           check=True, timeout=180)
        tag = subprocess.check_output(
            ['git', '-C', str(source), 'describe', '--tags', '--exact-match'],
            text=True, timeout=10,
        ).strip()
        if tag != RUNTIME_TAG:
            raise RuntimeError('Unexpected llama.cpp checkout')
        revision = subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True, timeout=10,
        ).strip()
        if revision != RUNTIME_COMMIT:
            raise RuntimeError('Unexpected llama.cpp revision')
        patch_mmvq(source)
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'huggingface_hub==0.34.4'], check=True, timeout=180)
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
    files = {item.rfilename: item for item in info.siblings}
    manifest = {'model_revision': info.sha, 'runtime_revision': RUNTIME_COMMIT, 'files': {}}
    for name in (MODEL_FILE, PROJECTOR_FILE):
        expected = files[name]
        if not expected.lfs:
            raise RuntimeError('Model file has no SHA256 metadata: '+name)
        print('DOWNLOAD '+name, flush=True)
        path = Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION,
                                   local_dir=ROOT / 'models'))
        digest = file_sha256(path)
        if path.stat().st_size != expected.size or digest != expected.lfs.sha256:
            raise RuntimeError('Model integrity check failed: '+name)
        manifest['files'][name] = {'bytes': expected.size, 'sha256': digest}
        print('VERIFIED '+name, flush=True)
    if not cached:
        with (ROOT / 'build.log').open('a', encoding='utf-8') as log:
            subprocess.run(['cmake', '-S', str(source), '-B', str(source / 'build'), *BUILD_FLAGS],
                           check=True, stdout=log,
                           stderr=subprocess.STDOUT, timeout=180)
            subprocess.run(['cmake', '--build', str(source / 'build'), '--target',
                            'llama-server', '--parallel', '4'], check=True,
                           stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        store_build(key)
        print('BUILD_CACHE_STORED', flush=True)
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('PREPARED '+str(ROOT), flush=True)


def server_command(context_tokens=196608):
    if type(context_tokens) is not int or not 4096 <= context_tokens <= 196608:
        raise ValueError('Context must be an integer between 4096 and 196608')
    return [str(ROOT / 'llama.cpp/build/bin/llama-server'),
            '--model', str(ROOT / 'models' / MODEL_FILE),
            '--mmproj', str(ROOT / 'models' / PROJECTOR_FILE),
            '--alias', ALIAS, '--host', '127.0.0.1', '--port', '8080',
            '--device', 'CUDA0,CUDA1', '--split-mode', 'tensor',
            '--n-gpu-layers', 'all', '--fit', 'off',
            '--ctx-size', str(context_tokens), '--parallel', '1',
            '--batch-size', '512', '--ubatch-size', '128',
            '--threads', '4', '--threads-batch', '4',
            '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
            '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
            '--jinja', '--chat-template-kwargs', '{"reasoning_effort":"xhigh"}',
            '--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0',
            '--n-predict', '-1', '--metrics', '--verbosity', '4']


def read_secret(label, *, allow_short=False):
    value = os.environ.get(label, '')
    if not value:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret(label)
    minimum = 1 if allow_short else 32
    if not isinstance(value, str) or len(value) < minimum or not value.isascii() or any(char.isspace() for char in value):
        raise ValueError(label+' has an invalid value')
    return value


def start_server(context_tokens=196608, *, allow_short_key=False):
    import time
    if not (ROOT / 'manifest.json').is_file():
        raise RuntimeError('Run prepare() first')
    key = read_secret('LOCAL_DESK_API_KEY', allow_short=allow_short_key)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', 8080))
    environment = dict(os.environ)
    environment['LLAMA_API_KEY'] = key
    environment['CUDA_VISIBLE_DEVICES'] = '0,1'
    environment['GGML_CUDA_MMVQ_MAX_BATCH'] = MMVQ_MAX_BATCH
    log = (ROOT / 'server.log').open('a', encoding='utf-8')
    process = subprocess.Popen(server_command(context_tokens), env=environment,
                               stdout=log, stderr=subprocess.STDOUT)
    log.close()
    deadline = time.monotonic() + 300
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Server exited; inspect server.log')
            request = urllib.request.Request('http://127.0.0.1:8080/health',
                                             headers={'Authorization': 'Bearer '+key})
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    if json.load(response).get('status') == 'ok':
                        print('SERVER_READY pid='+str(process.pid), flush=True)
                        return process
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise TimeoutError('Server startup exceeded 300 seconds')
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        raise


def ngrok_domain():
    from urllib.parse import urlsplit
    value = read_secret('NGROK_DOMAIN', allow_short=True)
    parsed = urlsplit(value if '://' in value else 'https://'+value)
    if (parsed.scheme != 'https' or not is_ngrok_host(parsed.netloc) or
            parsed.path.rstrip('/') not in ('', '/v1') or parsed.query or parsed.fragment):
        raise ValueError('NGROK_DOMAIN must be your assigned ngrok-free.app or ngrok-free.dev hostname')
    return parsed.netloc


def prepare_ngrok():
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version('pyngrok')
    except PackageNotFoundError:
        installed = None
    if installed != '8.1.2':
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyngrok==8.1.2'],
                       check=True, timeout=180)
    from pyngrok import installer
    executable = ROOT / 'ngrok' / 'ngrok'
    executable.parent.mkdir(parents=True, exist_ok=True)
    if not executable.is_file():
        installer.install_ngrok(str(executable), ngrok_version='3')
    return executable


def start_tunnel(server, *, allow_short_key=False):
    import time
    if server.poll() is not None:
        raise RuntimeError('Server is not running')
    key = read_secret('LOCAL_DESK_API_KEY', allow_short=allow_short_key)
    url = 'http://127.0.0.1:8080/v1/models'
    for invalid_key in ('', 'invalid-'+key):
        request = urllib.request.Request(url, headers={'Authorization':'Bearer '+invalid_key})
        try:
            with urllib.request.urlopen(request, timeout=10):
                raise RuntimeError('Server accepted an invalid API key')
        except urllib.error.HTTPError as error:
            if error.code not in (401, 403):
                raise
    request = urllib.request.Request(url, headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(request, timeout=10) as response:
        if ALIAS not in [item['id'] for item in json.load(response)['data']]:
            raise RuntimeError('Unexpected model identity')
    domain = ngrok_domain()
    token = read_secret('NGROK_AUTHTOKEN')
    executable = prepare_ngrok()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', 4040))
    config = ROOT / 'ngrok-config.json'
    config.write_text(json.dumps({'version':'2', 'web_addr':'127.0.0.1:4040'}), encoding='utf-8')
    environment = {name:value for name,value in os.environ.items()
                   if not name.startswith(('KAGGLE_', 'NGROK_')) and
                   name not in ('LOCAL_DESK_API_KEY', 'LLAMA_API_KEY')}
    environment['NGROK_AUTHTOKEN'] = token
    process = subprocess.Popen(
        [str(executable), 'http', 'http://127.0.0.1:8080', '--url', 'https://'+domain,
         '--inspect=false', '--config', str(config)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=environment,
    )
    try:
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('ngrok exited; check NGROK_AUTHTOKEN, NGROK_DOMAIN and account limits')
            try:
                with urllib.request.urlopen('http://127.0.0.1:4040/api/tunnels', timeout=2) as response:
                    raw = response.read(65537)
                if len(raw) > 65536:
                    raise ValueError('Unexpected ngrok response size')
                tunnels = json.loads(raw).get('tunnels', [])
                ready = any(item.get('public_url') == 'https://'+domain and
                            item.get('config',{}).get('addr') == 'http://127.0.0.1:8080'
                            for item in tunnels if isinstance(item, dict))
                if ready:
                    state = ROOT / 'fixed-tunnel.json'
                    temporary = state.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'provider':'ngrok', 'pid':process.pid,
                                                     'url':'https://'+domain+'/v1'}), encoding='utf-8')
                    temporary.replace(state)
                    print('FIXED_TUNNEL_READY https://'+domain+'/v1', flush=True)
                    return process
            except (OSError, ValueError):
                pass
            time.sleep(0.25)
        raise TimeoutError('ngrok startup exceeded 30 seconds')
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise


def tunnel_url(log_path=None, offset=0):
    from urllib.parse import urlsplit
    state = ROOT / 'fixed-tunnel.json'
    if log_path is None and state.is_file():
        record = json.loads(state.read_text(encoding='utf-8'))
        value = record.get('url', '')
        parsed = urlsplit(value)
        if (record.get('provider') != 'ngrok' or parsed.scheme != 'https' or
                not is_ngrok_host(parsed.netloc) or parsed.path != '/v1' or parsed.query or parsed.fragment):
            raise ValueError('Invalid fixed tunnel state')
        return value
    path = Path(log_path) if log_path else ROOT / 'tunnel.log'
    if not path.exists():
        return None
    latest = None
    with path.open(encoding='utf-8') as source:
        source.seek(offset)
        for line in source:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'opened' and event.get('status') == 'success':
                address = event.get('address', '')
                parsed = urlsplit('https://'+address)
                if parsed.hostname == address and address.endswith(('.lhr.life', '.localhost.run')):
                    latest = 'https://'+address+'/v1'
    return latest


def launch_service(*, allow_short_key=False):
    import time
    ngrok_domain()
    read_secret('NGROK_AUTHTOKEN')
    if not (ROOT / 'manifest.json').is_file():
        prepare()
    server = start_server(allow_short_key=allow_short_key)
    tunnel = None
    try:
        log = ROOT / 'tunnel.log'
        offset = log.stat().st_size if log.exists() else 0
        tunnel = start_tunnel(server, allow_short_key=allow_short_key)
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError('Tunnel exited; inspect tunnel.log')
            url = tunnel_url(offset=offset)
            if url:
                print('API_URL '+url, flush=True)
                return server, tunnel, url
            time.sleep(0.25)
        raise TimeoutError('Tunnel startup exceeded 45 seconds')
    except BaseException:
        for process in (tunnel, server):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        raise


if __name__ == '__main__':
    prepare()