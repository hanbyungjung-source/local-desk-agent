import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from desktop_agent import kaggle_link, kaggle_server


class KaggleServerTests(unittest.TestCase):
    def test_ngrok_failure_keeps_existing_model_and_does_not_publish_wrong_domain(self):
        import io
        import json
        import urllib.error
        for exit_code in (1, None):
            server, tunnel = Mock(), Mock(pid=4321)
            server.poll.return_value = None
            tunnel.poll.return_value = exit_code
            responses = [urllib.error.HTTPError('http://localhost',401,'',{},None),
                         urllib.error.HTTPError('http://localhost',403,'',{},None),
                         io.BytesIO(json.dumps({'data':[{'id':kaggle_server.ALIAS}]}).encode()),
                         io.BytesIO(json.dumps({'tunnels':[{'public_url':'https://wrong.ngrok-free.dev',
                                                           'config':{'addr':'http://127.0.0.1:8080'}}]}).encode())]
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as folder, \
                    patch.object(kaggle_server, 'ROOT', Path(folder)), \
                    patch.object(kaggle_server, 'prepare_ngrok', return_value=Path(folder)/'ngrok'), \
                    patch.object(kaggle_server, 'ngrok_domain', return_value='desk.ngrok-free.dev'), \
                    patch.object(kaggle_server, 'read_secret', return_value='test-secret'), \
                    patch.object(kaggle_server.socket, 'socket'), \
                    patch.object(kaggle_server.urllib.request, 'urlopen', side_effect=responses), \
                    patch.object(kaggle_server.subprocess, 'Popen', return_value=tunnel), \
                    patch('time.monotonic', side_effect=[0,0,31]), patch('time.sleep'):
                with self.assertRaises((RuntimeError, TimeoutError)) as caught:
                    kaggle_server.start_tunnel(server, allow_short_key=True)
                self.assertNotIn('test-secret', str(caught.exception))
                self.assertFalse((Path(folder)/'fixed-tunnel.json').exists())
                server.terminate.assert_not_called()
                self.assertEqual(tunnel.terminate.call_count, int(exit_code is None))

    def test_ngrok_tunnel_is_fixed_and_keeps_tokens_out_of_files_and_arguments(self):
        import io
        import json
        import urllib.error
        server, tunnel = Mock(), Mock(pid=4321)
        server.poll.return_value = tunnel.poll.return_value = None
        values = {'LOCAL_DESK_API_KEY':'test-api-key','NGROK_DOMAIN':'desk.ngrok-free.dev',
                  'NGROK_AUTHTOKEN':'test-ngrok-token-value-123456789012345'}
        responses = [urllib.error.HTTPError('http://localhost',401,'',{},None),
                     urllib.error.HTTPError('http://localhost',403,'',{},None),
                     io.BytesIO(json.dumps({'data':[{'id':kaggle_server.ALIAS}]}).encode()),
                     io.BytesIO(json.dumps({'tunnels':[{'public_url':'https://desk.ngrok-free.dev',
                                                       'config':{'addr':'http://127.0.0.1:8080'}}]}).encode())]
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(kaggle_server, 'ROOT', Path(folder)), \
                patch.object(kaggle_server, 'prepare_ngrok', return_value=Path(folder)/'ngrok'), \
                patch.object(kaggle_server, 'read_secret', side_effect=lambda label,**kwargs:values[label]), \
                patch.object(kaggle_server.socket, 'socket'), \
                patch.object(kaggle_server.urllib.request, 'urlopen', side_effect=responses), \
                patch.object(kaggle_server.subprocess, 'Popen', return_value=tunnel) as launch, \
                patch.dict(os.environ, {'KAGGLE_TEST':'private','LLAMA_API_KEY':'private','LOCAL_DESK_API_KEY':'private'}):
            self.assertIs(kaggle_server.start_tunnel(server, allow_short_key=True), tunnel)
            self.assertEqual(kaggle_server.tunnel_url(), 'https://desk.ngrok-free.dev/v1')
            command = launch.call_args.args[0]
            environment = launch.call_args.kwargs['env']
            self.assertEqual(command[command.index('--url')+1], 'https://desk.ngrok-free.dev')
            self.assertIn('--inspect=false', command)
            self.assertNotIn(values['NGROK_AUTHTOKEN'], repr(command))
            self.assertEqual(environment['NGROK_AUTHTOKEN'], values['NGROK_AUTHTOKEN'])
            self.assertNotIn('KAGGLE_TEST', environment)
            self.assertNotIn('LLAMA_API_KEY', environment)
            self.assertNotIn('LOCAL_DESK_API_KEY', environment)
            for path in Path(folder).glob('*.json'):
                self.assertNotIn(values['NGROK_AUTHTOKEN'], path.read_text())
                self.assertNotIn(values['LOCAL_DESK_API_KEY'], path.read_text())
            server.terminate.assert_not_called()

    def test_ngrok_missing_setup_prevents_model_download_or_start(self):
        with patch.object(kaggle_server, 'ngrok_domain', side_effect=ValueError('Missing setup')), \
                patch.object(kaggle_server, 'prepare') as prepare, \
                patch.object(kaggle_server, 'start_server') as start:
            with self.assertRaises(ValueError):
                kaggle_server.launch_service(allow_short_key=True)
            prepare.assert_not_called()
            start.assert_not_called()

    def test_ngrok_fixed_urls_keep_authentication_and_strict_schema(self):
        import httpx
        from desktop_agent.api import APIClient, APISettings, build_payload
        for host in ('my-desk.ngrok-free.app', 'my-desk.ngrok-free.dev'):
            url = 'https://'+host+'/v1'
            self.assertTrue(kaggle_server.is_ngrok_host(host))
            self.assertEqual(kaggle_link.normalize_tunnel_url(url+'/chat/completions'), url)
            config = APISettings(url=url, model=kaggle_server.ALIAS)
            client = APIClient(config, keys=['test-key'])
            self.assertEqual(client.headers('test-key')['Authorization'], 'Bearer test-key')
            self.assertEqual(client.headers('test-key')['ngrok-skip-browser-warning'], '1')
            payload = build_payload(config, [{'role':'user','content':'Ready'}], tool_names=['finish'])
            self.assertEqual(payload['response_format']['type'], 'json_schema')
            requests = []
            def respond(request):
                requests.append(request)
                return httpx.Response(200, json={'data':[{'id':kaggle_server.ALIAS}]})
            transport = httpx.Client(transport=httpx.MockTransport(respond))
            with patch.object(httpx, 'Client', return_value=transport):
                kaggle_link.verify_connection(url, 'test-key')
            self.assertEqual(requests[0].headers['ngrok-skip-browser-warning'], '1')
        for host in ('ngrok-free.app', 'desk.ngrok-free.app.evil.test', 'desk.ngrok-free.app:443', 'desk/path.ngrok-free.app'):
            self.assertFalse(kaggle_server.is_ngrok_host(host))
            with self.assertRaises(ValueError):
                kaggle_link.normalize_tunnel_url('https://'+host)
        self.assertNotIn('ngrok-skip-browser-warning', APIClient(APISettings(url='https://example.test'), keys=['test-key']).headers('test-key'))

    def test_fixed_mtp_comparison_changes_only_length_and_requires_matching_inputs(self):
        from desktop_agent.benchmark_kaggle_mtp import build_cases, input_hash, summarize, with_draft_length
        command = kaggle_server.server_command()
        other = with_draft_length(command, 2)
        index = command.index('--spec-draft-n-max')+1
        self.assertEqual(other[:index]+other[index+1:], command[:index]+command[index+1:])
        self.assertEqual(command[index], '3')
        self.assertEqual(other[index], '2')
        for invalid in (True, 1, 4, '2'):
            with self.assertRaises(ValueError):
                with_draft_length(command, invalid)
        cases = build_cases()
        self.assertEqual([case['name'] for case in cases], ['warmup','copy','arithmetic','image'])
        self.assertEqual(len({case['input_sha256'] for case in cases}), 4)
        for case in cases:
            self.assertEqual(input_hash(case['payload']), case['input_sha256'])
            self.assertEqual(case['payload']['reasoning_effort'], 'xhigh')
            self.assertEqual(case['payload']['response_format']['type'], 'json_schema')
        arms = [{'length':length, 'passed':True, 'samples':[
            {'name':case['name'], 'input_sha256':case['input_sha256'], 'passed':True,
             'seconds':100 if case['name']=='warmup' else length, 'usage':{'completion_tokens':50}}
            for case in cases]} for length in (3,2)]
        self.assertEqual(summarize(arms)['winner'], 2)
        for arm in arms:
            for sample in arm['samples']:
                sample['timings'] = {'predicted_n':3, 'predicted_ms':1000}
        self.assertEqual(summarize(arms)['arms'][0]['decode_tps'], 2)
        arms[0]['samples'][1]['timings'] = {}
        self.assertIsNone(summarize(arms)['arms'][0]['decode_tps'])
        arms[1]['samples'][0]['input_sha256'] = 'different'
        with self.assertRaises(ValueError):
            summarize(arms)

    def test_pc_connection_failure_messages_do_not_echo_private_details(self):
        for code in (401, 403, 502, 503, 504, 429, 'network'):
            message = kaggle_link.connection_error_message(kaggle_link.ConnectionCheckError(code))
            self.assertTrue(message)
            if isinstance(code, int):
                self.assertIn('HTTP '+str(code), message)
        self.assertNotIn('private-key-and-response', kaggle_link.connection_error_message(ValueError('private-key-and-response')))

    def test_pc_entry_starts_app_only_after_connection(self):
        import sys
        from desktop_agent import app
        for connected in (False, True):
            with self.subTest(connected=connected), \
                    patch.object(sys, 'argv', ['kaggle_link', '--connect']), \
                    patch.object(kaggle_link, 'connect_desktop', return_value=connected), \
                    patch.object(app, 'main') as start:
                kaggle_link.main()
                self.assertEqual(start.call_count, int(connected))
                if connected:
                    self.assertEqual(sys.argv, ['kaggle_link'])
        with patch.object(sys, 'argv', ['kaggle_link']), \
                patch.object(kaggle_link, 'public_key', return_value='public-key') as public_key, \
                patch('builtins.print') as output:
            kaggle_link.main()
            public_key.assert_called_once_with()
            output.assert_called_once_with('public-key')

    def test_pc_connection_cancel_does_not_save(self):
        import threading
        cancel = threading.Event()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(kaggle_link, 'KeyVault') as vault, \
                patch.object(kaggle_link, 'verify_connection', side_effect=lambda *args: cancel.set()):
            with self.assertRaises(InterruptedError):
                kaggle_link.connect_profile('https://new.lhr.life', 'test-key', state=folder, cancel=cancel)
            vault.return_value.save.assert_not_called()
            self.assertFalse((Path(folder) / 'settings.json').exists())

    def test_pc_process_guard_excludes_launcher_ancestry(self):
        import psutil
        rows = [Mock(info={'pid':os.getpid(), 'name':'python.exe', 'cmdline':['python', '-m', 'desktop_agent.kaggle_link', '--connect']}),
                Mock(info={'pid':123, 'name':'python.exe', 'cmdline':['python', '-m', 'desktop_agent.kaggle_link', '--connect']}),
                Mock(info={'pid':456, 'name':'python.exe', 'cmdline':['python', 'unrelated.py', 'desktop_agent.app']})]
        with patch.object(psutil, 'Process') as current, patch.object(psutil, 'process_iter', return_value=rows):
            current.return_value.parents.return_value = [Mock(pid=123)]
            self.assertFalse(kaggle_link.desktop_is_running())
            rows.append(Mock(info={'pid':789, 'name':'pythonw.exe', 'cmdline':['pythonw', '-m', 'desktop_agent.app']}))
            self.assertTrue(kaggle_link.desktop_is_running())

    def test_pc_batch_launches_kaggle_module_without_secrets(self):
        import json
        import shutil
        import subprocess
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder) / 'Desk With Spaces'
            home.mkdir()
            launcher = home / 'start.ps1'
            shutil.copy2(root / 'desktop_agent/start.ps1', launcher)
            python = home / 'python.exe'
            python.touch()
            (home / 'pythonw.exe').touch()
            script = """
$ErrorActionPreference = 'Stop'
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $RedirectStandardOutput, $RedirectStandardError)
    $global:captured = @{file=$FilePath; arguments=$ArgumentList; root=$WorkingDirectory; log=$RedirectStandardError}
}
& 'LAUNCHER' -Detached -Kaggle -PythonPath 'PYTHON'
$global:captured | ConvertTo-Json -Compress
""".replace('LAUNCHER', str(launcher).replace("'", "''")).replace('PYTHON', str(python).replace("'", "''"))
            result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                                     '-Command', script], capture_output=True, text=True, timeout=20, check=True)
            command = json.loads(result.stdout)
            self.assertEqual(command['arguments'], ['-m', 'desktop_agent.kaggle_link', '--connect'])
            self.assertEqual(Path(command['file']), home / 'pythonw.exe')
            self.assertEqual(Path(command['root']), Path(folder))
            self.assertEqual(Path(command['log']).parent, home / 'data/launcher')
        batch = (root / 'desktop_agent/start-kaggle.bat').read_text()
        self.assertIn('call "%~dp0start.bat" -Kaggle %*', batch)
        self.assertNotIn('API_KEY', batch)

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1', 'Requires Tk desktop')
    def test_pc_dialog_masks_key_and_handles_failure_success_and_cancel(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(kaggle_link, 'desktop_is_running', return_value=False), \
                patch.object(kaggle_link, 'connect_profile', side_effect=[kaggle_link.ConnectionCheckError(503), None]) as connect:
            root = tk.Tk()
            dialog = kaggle_link.ConnectionDialog(root, state=folder)
            root.geometry('480x250')
            root.update()
            self.assertEqual(dialog.key_entry.cget('show'), '*')
            self.assertLessEqual(dialog.submit_button.winfo_rooty()+dialog.submit_button.winfo_height(), root.winfo_rooty()+root.winfo_height())
            dialog.url.set('https://new.lhr.life/v1')
            dialog.key.set('test-key')
            dialog.submit()
            self.assertEqual(dialog.key.get(), '')
            result = dialog.events.get(timeout=3)
            self.assertFalse(result)
            dialog.events.put(result)
            dialog.poll()
            self.assertFalse(dialog.busy)
            self.assertNotIn('private error', dialog.status.get())
            self.assertIn('HTTP 503', dialog.status.get())
            root.update()
            self.assertLessEqual(dialog.submit_button.winfo_rooty()+dialog.submit_button.winfo_height(), root.winfo_rooty()+root.winfo_height())
            dialog.key.set('new-key')
            dialog.submit()
            result = dialog.events.get(timeout=3)
            self.assertTrue(result)
            dialog.events.put(result)
            dialog.poll()
            self.assertTrue(dialog.connected)
            self.assertEqual(connect.call_args.args, ('https://new.lhr.life/v1', 'new-key'))
        for result in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = tk.Tk()
                dialog = kaggle_link.ConnectionDialog(root, state=folder)
                dialog.busy = True
                dialog.close()
                self.assertTrue(dialog.cancel.is_set())
                dialog.events.put(result)
                dialog.poll()
                self.assertFalse(dialog.connected)

    def test_pc_connection_defaults_to_unlimited_output_and_three_hour_timeout(self):
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            vault = Mock()
            vault.load.return_value = []
            with patch.object(kaggle_link, 'KeyVault', return_value=vault), \
                    patch.object(kaggle_link, 'verify_connection'):
                config = kaggle_link.connect_profile('https://new.lhr.life', 'test-key', state=folder)
            self.assertEqual(config.timeout_seconds, 10800)
            self.assertEqual(config.max_output_tokens, -1)
            current = Settings.load(Path(folder) / 'settings.json')
            self.assertEqual(current.api.max_output_tokens, -1)
            self.assertEqual(current.api_profiles[kaggle_link.PROFILE_NAME].max_output_tokens, -1)
            self.assertEqual(current.output_budget, 4096)
            self.assertEqual(current.api.timeout_seconds, 10800)
            self.assertEqual(current.api_profiles[kaggle_link.PROFILE_NAME].timeout_seconds, 10800)
            with patch.object(kaggle_link, 'KeyVault', return_value=vault), \
                    patch.object(kaggle_link, 'verify_connection'):
                reconnected = kaggle_link.connect_profile('https://new.lhr.life', 'test-key', state=folder)
            self.assertEqual(reconnected.max_output_tokens, -1)

    def test_pc_connection_preserves_settings_and_reuses_scoped_key(self):
        from dataclasses import replace
        from desktop_agent.agent import Settings
        from desktop_agent.api import APISettings
        from desktop_agent.credentials import credential_scope
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            profile = APISettings(url='https://old.lhr.life/v1', model=kaggle_server.ALIAS,
                                  context_tokens=196608, key_profile_id='a'*32, timeout_seconds=21600)
            original = replace(Settings(), compaction_target_percent=30).with_api_profile(
                kaggle_link.PROFILE_NAME, profile).with_api_profile('Other', APISettings())
            original.save(state / 'settings.json')
            before = (state / 'settings.json').read_bytes()
            old_scope = credential_scope(profile.url, profile.key_profile_id)
            saved_keys = {old_scope:['test-short-key']}
            vault = Mock()
            vault.load.side_effect = lambda scope: list(saved_keys.get(scope, []))
            vault.save.side_effect = lambda scope, keys: saved_keys.__setitem__(scope, list(keys))
            with patch.object(kaggle_link, 'KeyVault', return_value=vault), \
                    patch.object(kaggle_link, 'verify_connection') as verify:
                config = kaggle_link.connect_profile('https://new.lhr.life/v1/chat/completions', state=state)
            verify.assert_called_once_with('https://new.lhr.life/v1', 'test-short-key')
            current = Settings.load(state / 'settings.json')
            self.assertEqual(config.timeout_seconds, 21600)
            self.assertEqual(config.max_output_tokens, profile.max_output_tokens)
            self.assertEqual(current.api.timeout_seconds, 21600)
            self.assertEqual(current.api_profiles[kaggle_link.PROFILE_NAME].timeout_seconds, 21600)
            self.assertEqual(current, original.with_api_profile(kaggle_link.PROFILE_NAME, config).use_api_profile(kaggle_link.PROFILE_NAME))
            self.assertEqual((state / 'settings.before-kaggle-connect.json').read_bytes(), before)
            self.assertEqual(saved_keys[old_scope], ['test-short-key'])
            self.assertEqual(saved_keys[credential_scope(config.url, config.key_profile_id)], ['test-short-key'])
            self.assertNotIn(b'test-short-key', (state / 'settings.json').read_bytes())

    def test_pc_connection_failure_does_not_save_and_save_error_restores_keys(self):
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            Settings().save(state / 'settings.json')
            before = (state / 'settings.json').read_bytes()
            vault = Mock()
            vault.load.return_value = ['previous-key']
            with patch.object(kaggle_link, 'KeyVault', return_value=vault), \
                    patch.object(kaggle_link, 'verify_connection', side_effect=ValueError('HTTP 401')):
                with self.assertRaises(ValueError):
                    kaggle_link.connect_profile('https://new.lhr.life', 'test-key', state=state)
            vault.save.assert_not_called()
            self.assertEqual((state / 'settings.json').read_bytes(), before)
            with patch.object(kaggle_link, 'KeyVault', return_value=vault), \
                    patch.object(kaggle_link, 'verify_connection'), \
                    patch.object(Settings, 'save', side_effect=OSError('Cannot save')):
                with self.assertRaises(OSError):
                    kaggle_link.connect_profile('https://new.lhr.life', 'test-key', state=state)
            self.assertEqual(vault.save.call_args_list[0].args[1], ['test-key'])
            self.assertEqual(vault.save.call_args_list[1].args[1], ['previous-key'])
            self.assertEqual((state / 'settings.json').read_bytes(), before)

    def test_pc_connection_checks_models_without_inference_or_redirects(self):
        import httpx
        for status, data in ((200, {'data':[{'id':kaggle_server.ALIAS}]}),
                             (401, {}), (503, {}), (302, {}), (200, {'data':[{'id':'other'}]}), (200, [])):
            requests = []
            def respond(request):
                requests.append(request)
                return httpx.Response(status, json=data, headers={'Location':'https://other.example'})
            client = httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=False)
            with self.subTest(status=status, data=data), \
                    patch.object(httpx, 'Client', return_value=client) as factory:
                if status == 200 and isinstance(data, dict) and data.get('data', [{}])[0].get('id') == kaggle_server.ALIAS:
                    kaggle_link.verify_connection('https://new.lhr.life/v1', 'test-key')
                else:
                    with self.assertRaises(ValueError) as caught:
                        kaggle_link.verify_connection('https://new.lhr.life/v1', 'test-key')
                    if status != 200:
                        self.assertEqual(caught.exception.code, status)
                factory.assert_called_once_with(timeout=10, follow_redirects=False, trust_env=False)
                self.assertEqual(len(requests), 1)
                self.assertEqual((requests[0].method, str(requests[0].url)), ('GET', 'https://new.lhr.life/v1/models'))
        for url in ('http://new.lhr.life', 'https://evil.example/v1',
                    'https://new.lhr.life@evil.example', 'https://new.lhr.life/v1?key=secret',
                    'https://new.lhr.life:443/v1', 'https://new.lhr.life/other'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                kaggle_link.normalize_tunnel_url(url)

    def test_latest_tunnel_address_ignores_old_or_invalid_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tunnel.log'
            path.write_text('banner\n{"type":"opened","status":"success","address":"old.lhr.life"}\n', encoding='utf-8')
            offset = path.stat().st_size
            with path.open('a', encoding='utf-8') as output:
                output.write('{"type":"opened","status":"success","address":"new.lhr.life"}\n')
                output.write('{"type":"opened","status":"success","address":"evil.example/path.lhr.life"}\n')
            self.assertEqual(kaggle_server.tunnel_url(path, offset), 'https://new.lhr.life/v1')

    def test_launcher_prepares_then_starts_without_restarting_ready_assets(self):
        server, tunnel = Mock(), Mock()
        tunnel.poll.return_value = None
        with patch.object(kaggle_server.Path, 'is_file', return_value=True), \
                patch.object(kaggle_server.Path, 'exists', return_value=False), \
            patch.object(kaggle_server, 'ngrok_domain', return_value='desk.ngrok-free.dev'), \
            patch.object(kaggle_server, 'read_secret', return_value='test-token'), \
                patch.object(kaggle_server, 'prepare') as prepare, \
                patch.object(kaggle_server, 'start_server', return_value=server), \
                patch.object(kaggle_server, 'start_tunnel', return_value=tunnel), \
                patch.object(kaggle_server, 'tunnel_url', return_value='https://new.lhr.life/v1'):
            self.assertEqual(kaggle_server.launch_service(allow_short_key=True),
                             (server, tunnel, 'https://new.lhr.life/v1'))
            prepare.assert_not_called()

    def test_preflight_requires_two_free_t4_devices(self):
        result = Mock(stdout='0, Tesla T4, 15360, 14912\n1, Tesla T4, 15360, 14912\n')
        with patch.object(kaggle_server.subprocess, 'run', return_value=result), \
                patch.object(kaggle_server.Path, 'mkdir'), \
                patch.object(kaggle_server.shutil, 'disk_usage', return_value=Mock(free=30*1024**3)):
            self.assertEqual(len(kaggle_server.inspect_environment()['devices']), 2)
            for output in ('0, Tesla P100, 16384, 16000\n',
                           '0, Tesla T4, 15360, 13999\n1, Tesla T4, 15360, 14912\n'):
                result.stdout = output
                with self.assertRaises(RuntimeError):
                    kaggle_server.inspect_environment()

    def test_command_keeps_split_mtp_and_ub128(self):
        command = kaggle_server.server_command()
        for flag, value in {'--device':'CUDA0,CUDA1', '--split-mode':'tensor', '--fit':'off',
                            '--spec-type':'draft-mtp', '--spec-draft-n-max':'3',
                            '--ubatch-size':'128', '--host':'127.0.0.1',
                            '--ctx-size':'196608', '--verbosity':'4'}.items():
            self.assertEqual(command[command.index(flag)+1], value)
        for flag in ('--api-key', '--tensor-split', '--spec-draft-model', '--spec-draft-device'):
            self.assertNotIn(flag, command)
        for context in (True, 1024, 196609, '196608'):
            with self.assertRaises(ValueError):
                kaggle_server.server_command(context)

    def test_mmvq_patch_is_idempotent_and_refuses_unknown_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ggml'/'src'/'ggml-cuda'/'mmvq.cu'
            path.parent.mkdir(parents=True)
            original = ('#include "mmvq.cuh"\n#include <type_traits>\n\n' + kaggle_server.MMVQ_ANCHOR +
                        '    return ne11 <= 8;\n}\n')
            path.write_text(original, encoding='utf-8')
            kaggle_server.patch_mmvq(folder)
            patched = path.read_text(encoding='utf-8')
            self.assertEqual(patched.count('GGML_CUDA_MMVQ_MAX_BATCH_'), 1)
            self.assertEqual(patched.count('#include <array>'), 1)
            self.assertLess(patched.index('t4_override'), patched.index('return ne11 <= 8;'))
            kaggle_server.patch_mmvq(folder)
            self.assertEqual(path.read_text(encoding='utf-8'), patched)
            path.write_text(original.replace('int64_t ne11', 'int64_t n'), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                kaggle_server.patch_mmvq(folder)
            path.write_text(patched.replace('LOCAL-DESK-T4', 'OTHER'), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                kaggle_server.patch_mmvq(folder)

    def test_build_cache_restores_only_matching_intact_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root, cache = Path(folder)/'root', Path(folder)/'cache'
            binary = root/'llama.cpp/build/bin/llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'server-v1')
            with patch.object(kaggle_server, 'ROOT', root), patch.object(kaggle_server, 'CACHE', cache):
                kaggle_server.store_build('key-a')
                binary.unlink()
                self.assertFalse(kaggle_server.restore_build('key-b'))
                self.assertFalse(binary.exists())
                self.assertTrue(kaggle_server.restore_build('key-a'))
                self.assertEqual(binary.read_bytes(), b'server-v1')
                (cache/'llama-server').write_bytes(b'server-v2')
                self.assertFalse(kaggle_server.restore_build('key-a'))
                (cache/'build.json').write_text('[]', encoding='utf-8')
                self.assertFalse(kaggle_server.restore_build('key-a'))
        with patch.object(kaggle_server.subprocess, 'run', return_value=Mock(stdout='release 12.8')):
            first = kaggle_server.build_key()
            self.assertEqual(first, kaggle_server.build_key())
        with patch.object(kaggle_server.subprocess, 'run', return_value=Mock(stdout='release 12.9')):
            self.assertNotEqual(first, kaggle_server.build_key())

    def test_prepare_uses_build_cache_or_builds_and_stores(self):
        import hashlib
        import sys
        import types
        names = (kaggle_server.MODEL_FILE, kaggle_server.PROJECTOR_FILE)
        siblings = [Mock(rfilename=name, size=len(name.encode()),
                         lfs=Mock(sha256=hashlib.sha256(name.encode()).hexdigest())) for name in names]
        def download(repo, name, revision, local_dir):
            path = Path(local_dir)/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            return str(path)
        hub = types.SimpleNamespace(hf_hub_download=download, HfApi=Mock(return_value=Mock(
            model_info=Mock(return_value=Mock(sha='revision', siblings=siblings)))))
        for cached in (True, False):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as folder, \
                    patch.object(kaggle_server, 'ROOT', Path(folder)), \
                    patch.dict(sys.modules, {'huggingface_hub': hub}), \
                    patch.object(kaggle_server, 'inspect_environment', return_value={'nvcc': '/usr/bin/nvcc'}), \
                    patch.object(kaggle_server, 'build_key', return_value='key'), \
                    patch.object(kaggle_server, 'restore_build', return_value=cached) as restore, \
                    patch.object(kaggle_server, 'store_build') as store, \
                    patch.object(kaggle_server, 'patch_mmvq') as patch_source, \
                    patch.object(kaggle_server.subprocess, 'check_output',
                                 side_effect=[kaggle_server.RUNTIME_TAG, kaggle_server.RUNTIME_COMMIT]), \
                    patch.object(kaggle_server.subprocess, 'run') as run:
                kaggle_server.prepare()
                tools = [call.args[0][0] for call in run.call_args_list]
                restore.assert_called_once_with('key')
                self.assertTrue((Path(folder)/'manifest.json').is_file())
                if cached:
                    self.assertNotIn('git', tools)
                    self.assertNotIn('cmake', tools)
                    store.assert_not_called()
                    patch_source.assert_not_called()
                else:
                    self.assertIn('git', tools)
                    self.assertEqual(tools.count('cmake'), 2)
                    store.assert_called_once_with('key')
                    patch_source.assert_called_once()

    def test_server_environment_sets_mmvq_threshold(self):
        process = Mock(pid=77)
        process.poll.return_value = None
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(kaggle_server, 'ROOT', Path(folder)), \
                patch.object(kaggle_server, 'read_secret', return_value='test-key'), \
                patch.object(kaggle_server.socket, 'socket'), \
                patch.object(kaggle_server.subprocess, 'Popen', return_value=process) as launch, \
                patch.object(kaggle_server.urllib.request, 'urlopen', return_value=response), \
                patch.object(kaggle_server.json, 'load', return_value={'status':'ok'}):
            (Path(folder)/'manifest.json').write_text('{}', encoding='utf-8')
            self.assertIs(kaggle_server.start_server(allow_short_key=True), process)
        environment = launch.call_args.kwargs['env']
        self.assertEqual(environment['GGML_CUDA_MMVQ_MAX_BATCH'], '1')
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0,1')

    def test_short_key_requires_explicit_approval(self):
        with patch.dict(os.environ, {'LOCAL_DESK_API_KEY':'test-short-key'}):
            with self.assertRaises(ValueError):
                kaggle_server.read_secret('LOCAL_DESK_API_KEY')
            self.assertEqual(kaggle_server.read_secret('LOCAL_DESK_API_KEY', allow_short=True),
                             'test-short-key')
        with patch.dict(os.environ, {'LOCAL_DESK_API_KEY':'invalid key'}):
            with self.assertRaises(ValueError):
                kaggle_server.read_secret('LOCAL_DESK_API_KEY', allow_short=True)

    def test_existing_server_prevents_launch(self):
        with patch.object(kaggle_server.Path, 'is_file', return_value=True), \
                patch.object(kaggle_server, 'read_secret', return_value='test-key'), \
                patch.object(kaggle_server.socket, 'socket') as socket_factory, \
                patch.object(kaggle_server.subprocess, 'Popen') as launch:
            socket_factory.return_value.__enter__.return_value.bind.side_effect = OSError('Port in use')
            with self.assertRaises(OSError):
                kaggle_server.start_server(allow_short_key=True)
            socket_factory.return_value.__enter__.return_value.setsockopt.assert_called_once_with(
                kaggle_server.socket.SOL_SOCKET, kaggle_server.socket.SO_REUSEADDR, 1)
            launch.assert_not_called()

    def test_tunnel_refuses_server_without_authentication(self):
        server = Mock()
        server.poll.return_value = None
        with patch.object(kaggle_server, 'read_secret', return_value='test-key'), \
                patch.object(kaggle_server.urllib.request, 'urlopen') as urlopen, \
                patch.object(kaggle_server.subprocess, 'Popen') as launch:
            urlopen.return_value.__enter__.return_value = Mock()
            with self.assertRaisesRegex(RuntimeError, 'invalid API key'):
                kaggle_server.start_tunnel(server, allow_short_key=True)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()