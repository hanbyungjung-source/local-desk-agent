from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from desktop_agent.api import APIClient, APISettings, StreamResult, build_payload, endpoint


@lru_cache(maxsize=1)
def synthetic_vertex_account():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537,key_size=2048)
    return {'type':'service_account','project_id':'test-project','private_key_id':'synthetic-key-id',
            'private_key':key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()).decode('ascii'),
            'client_email':'test@test-project.iam.gserviceaccount.com','token_uri':'https://oauth2.googleapis.com/token'}


class APIConfigTests(unittest.TestCase):
    def test_kaggle_unlimited_output_keeps_context_reserve_and_request_options(self):
        from desktop_agent.agent import Settings
        from desktop_agent.kaggle_server import ALIAS
        config = APISettings(url='https://example.ngrok-free.dev/v1',model=ALIAS,
                             context_tokens=196608,max_output_tokens=-1,reasoning_effort='xhigh')
        messages = [{'role':'system','content':'Synthetic test'},{'role':'user','content':'Continue'}]
        for field in ('auto','max_tokens','max_completion_tokens'):
            for native in (False,True):
                with self.subTest(field=field,native=native):
                    candidate = replace(config,token_limit_field=field,native_tools=native)
                    payload = build_payload(candidate,messages,tool_names=['finish'])
                    expected = build_payload(replace(candidate,max_output_tokens=4096),messages,tool_names=['finish'])
                    limit_field = 'max_tokens' if field=='auto' else field
                    expected[limit_field] = -1
                    self.assertEqual(payload,expected)
        self.assertEqual(config.output_reserve,4096)
        self.assertEqual(Settings(backend='api',api=config).output_budget,4096)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            Settings().with_api_profile('Kaggle',config).use_api_profile('Kaggle').save(path)
            restored = Settings.load(path)
            self.assertEqual(restored.api.max_output_tokens,-1)
            self.assertEqual(restored.api_profiles['Kaggle'].max_output_tokens,-1)
            self.assertEqual(restored.output_budget,4096)
        for invalid in (replace(config,format='responses'),replace(config,format='gemini'),
                        replace(config,format='anthropic'),replace(config,model='other'),
                        replace(config,url='https://other.example/v1'),replace(config,context_tokens=4096),
                        replace(config,max_output_tokens=0),replace(config,max_output_tokens=-2),
                        replace(config,max_output_tokens=True),replace(config,max_output_tokens=-1.0)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                invalid.validate()

    def test_incomplete_response_is_reported_and_same_task_continues(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Model, Settings
        from desktop_agent.store import Store
        for finish, reason in (('length','output_limit'),(None,'missing_completion'),
                               ('content_filter','rejected_or_unsupported_finish')):
            requests = []
            async def handler(request):
                requests.append(json.loads(request.content))
                answer = {'tool':'finish','arguments':{'text':'Discarded partial' if len(requests)==1 else 'Recovered'}}
                return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(answer)},
                                                            'finish_reason':finish if len(requests)==1 else 'stop'}]})
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as folder:
                store = Store(folder)
                identifier = store.create('Incomplete response recovery')
                settings = Settings(backend='api',max_steps=3,auto_compact=False,
                                    api=APISettings(url='https://example.test/v1',model='test',stream=False,
                                                    tokenizer='estimate',max_retries=0))
                model = Model(settings,Path(folder)/'keys.dpapi')
                model.remote = APIClient(settings.api,keys=['test-key'],transport=httpx.MockTransport(handler))
                tools = Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='manual')
                Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Reply briefly')
                events = store.events(identifier)
                errors = [event for event in events if event['metadata'].get('tool')=='response_validation']
                self.assertEqual(len(requests),2)
                self.assertEqual(len(errors),1)
                self.assertEqual(errors[0]['metadata']['status'],'error')
                self.assertEqual(errors[0]['metadata']['reason'],reason)
                self.assertEqual(errors[0]['metadata']['api_request']['http_status'],200)
                self.assertIn('Incomplete or rejected API response',json.dumps(requests[1]))
                self.assertIn('do not bypass',json.dumps(requests[1]))
                self.assertNotIn('Discarded partial',json.dumps(requests[1]))
                self.assertEqual(json.loads(events[-1]['content'])['message'],'Recovered')
                self.assertEqual(len(store.sessions()),1)
                self.assertEqual(sum(event['role']=='user' for event in events),1)
                tools.execute.assert_not_called()
                model.close()

    def test_incomplete_response_recovery_respects_step_limit_and_stop(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Model, Settings
        from desktop_agent.store import Store
        for cancel, expected_requests in ((False,2),(True,1)):
            requests = []
            stopped = threading.Event()
            async def handler(request):
                requests.append(json.loads(request.content))
                event = {'choices':[{'delta':{'content':'{"tool":"finish","arguments":'}}]}
                return httpx.Response(200,headers={'content-type':'text/event-stream'},
                                      text='data: '+json.dumps(event)+'\n\n')
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as folder:
                store = Store(folder)
                identifier = store.create('Bounded incomplete response recovery')
                settings = Settings(backend='api',max_steps=2,auto_compact=False,
                                    api=APISettings(url='https://example.test/v1',model='test',
                                                    tokenizer='estimate',max_retries=0))
                model = Model(settings,Path(folder)/'keys.dpapi')
                model.remote = APIClient(settings.api,keys=['test-key'],transport=httpx.MockTransport(handler))
                tools = Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='manual')
                def notify(kind, value):
                    if cancel and kind=='refresh' and any(
                            event['metadata'].get('tool')=='response_validation' for event in store.events(identifier)):
                        stopped.set()
                Agent(store,model,tools,stopped,notify).run(identifier,'Reply briefly')
                events = store.events(identifier)
                self.assertEqual(len(requests),expected_requests)
                self.assertEqual(sum(event['metadata'].get('tool')=='response_validation' for event in events),expected_requests)
                self.assertIn('Stopped' if cancel else 'Step limit',events[-1]['content'])
                self.assertFalse(any(event['role']=='assistant' for event in events))
                tools.execute.assert_not_called()
                model.close()

    def test_timeout_has_no_upper_limit(self):
        for seconds in (10,120,600,601,10800,86400):
            with self.subTest(seconds=seconds):
                config = APISettings(timeout_seconds=seconds)
                self.assertIs(config.validate(),config)
                self.assertEqual(config.timeout_seconds,seconds)
        for invalid in (-1,0,9,True,10800.0,'10800',None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError,'timeout_seconds'):
                APISettings(timeout_seconds=invalid).validate()

    def test_long_timeout_reaches_request_and_transport(self):
        import asyncio
        import httpx
        from unittest.mock import patch
        requests = []
        async def handler(request):
            requests.append(request)
            return httpx.Response(200,json={'choices':[{'message':{'content':'{"tool":"finish","arguments":{"text":"Ready"}}'},'finish_reason':'stop'}]})
        config = APISettings(url='https://example.test/v1',model='test',stream=False,
                             timeout_seconds=10800,max_retries=0)
        client = APIClient(config,keys=['synthetic-key'],transport=httpx.MockTransport(handler))
        with patch('desktop_agent.api.asyncio.wait',wraps=asyncio.wait) as wait:
            action, _ = client.generate([{'role':'user','content':'Ready'}],None,threading.Event(),lambda text:None,lambda *args:None)
        self.assertEqual(action['tool'],'finish')
        self.assertEqual(wait.call_args.kwargs['timeout'],10800)
        self.assertEqual(len(requests),1)
        timeout = requests[0].extensions['timeout']
        self.assertEqual(timeout,dict(connect=10,read=10800,write=10800,pool=10800))

    def test_kaggle_expired_tunnel_reports_url_change_without_retry_or_private_body(self):
        import httpx
        from desktop_agent.kaggle_server import ALIAS
        for body, expired in ((b'<h1>no tunnel here :(</h1> private-response-value',True),
                              (b'<h1>Server busy</h1> private-response-value',False)):
            requests = []
            async def handler(request):
                requests.append(request)
                return httpx.Response(503, content=body, headers={'Content-Type':'text/html'})
            config = APISettings(url='https://expired.lhr.life/v1', model=ALIAS, max_retries=0)
            client = APIClient(config, keys=['synthetic-key'], transport=httpx.MockTransport(handler))
            with self.subTest(expired=expired), self.assertRaises(ValueError) as caught:
                client.generate([{'role':'user','content':'Ready'}],None,threading.Event(),lambda text:None,lambda *args:None)
            message = str(caught.exception)
            self.assertIn('API HTTP 503', message)
            self.assertEqual('latest API_URL' in message, expired)
            self.assertNotIn('private-response-value', message)
            self.assertNotIn('synthetic-key', message)
            self.assertEqual(len(requests), 1)
            self.assertEqual(caught.exception.api_diagnostics['attempts'], 1)

    def test_invalid_tool_call_is_logged_and_corrected_in_same_task(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Model, Settings
        from desktop_agent.store import Store
        requests = []
        async def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            answer = ({'tool':'finish','arguments':{'text':'Discarded'},'unexpected':True}
                      if len(requests)==1 else {'tool':'finish','arguments':{'text':'Corrected'}})
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(answer)},'finish_reason':'stop'}]})
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create('Recovery')
            settings = Settings(backend='api', max_steps=3, auto_compact=False,
                                api=APISettings(url='https://example.test/v1',model='test',stream=False,tokenizer='estimate',max_retries=0))
            model = Model(settings, Path(folder)/'keys.dpapi')
            model.remote = APIClient(settings.api,keys=['test-key'],transport=httpx.MockTransport(handler))
            tools = Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='manual')
            Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Reply briefly')
            events = store.events(identifier)
            errors = [event for event in events if event['metadata'].get('tool')=='call_validation']
            self.assertEqual(len(requests),2)
            self.assertEqual(len(errors),1)
            self.assertEqual(errors[0]['metadata']['status'],'error')
            self.assertEqual(errors[0]['metadata']['api_request']['http_status'],200)
            self.assertIn('Invalid tool call',json.dumps(requests[1]))
            self.assertEqual(json.loads(events[-1]['content'])['message'],'Corrected')
            self.assertEqual(len(store.sessions()),1)
            self.assertEqual(sum(event['role']=='user' for event in events),1)
            tools.execute.assert_not_called()
            model.close()

    def test_invalid_tool_call_recovery_keeps_step_limit_and_transport_errors_terminal(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Model, Settings
        from desktop_agent.store import Store
        for status, expected_requests, expected_validation in ((200,2,2),(503,1,0)):
            requests = []
            async def handler(request):
                requests.append(request)
                return httpx.Response(status,json={'choices':[{'message':{'content':'{"tool":"finish","arguments":{},"extra":true}'},'finish_reason':'stop'}]})
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                store=Store(folder)
                identifier=store.create('Bounded recovery')
                settings=Settings(backend='api',max_steps=2,auto_compact=False,
                                  api=APISettings(url='https://example.test/v1',model='test',stream=False,tokenizer='estimate',max_retries=0))
                model=Model(settings,Path(folder)/'keys.dpapi')
                model.remote=APIClient(settings.api,keys=['test-key'],transport=httpx.MockTransport(handler))
                tools=Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='manual')
                Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Reply briefly')
                events=store.events(identifier)
                self.assertEqual(len(requests),expected_requests)
                self.assertEqual(sum(event['metadata'].get('tool')=='call_validation' for event in events),expected_validation)
                self.assertIn('Step limit' if status==200 else 'HTTP 503',events[-1]['content'])
                tools.execute.assert_not_called()
                model.close()

    def test_kaggle_schema_enforces_only_loaded_tools_without_changing_other_providers(self):
        from desktop_agent.kaggle_server import ALIAS
        from desktop_agent.protocol import compact_schema
        config = APISettings(url='https://example.lhr.life/v1', model=ALIAS, tokenizer='estimate')
        messages = [{'role':'system','content':'Synthetic test'}, {'role':'user','content':'Ready'}]
        names = ['finish','browser_read']
        payload = build_payload(config, messages, tool_names=names)
        schema = payload['response_format']['json_schema']
        self.assertEqual(payload['response_format']['type'], 'json_schema')
        self.assertTrue(schema['strict'])
        self.assertEqual(schema['schema'], compact_schema(names))
        self.assertEqual(payload['messages'], messages)
        for branch in schema['schema']['oneOf']:
            self.assertEqual(set(branch['properties']), {'tool','arguments'})
            self.assertFalse(branch['additionalProperties'])
        for other in (replace(config, url='https://other.example/v1'),
                      replace(config, url='https://example.lhr.life.evil.test/v1'),
                      replace(config, model='other-model')):
            with self.subTest(url=other.url, model=other.model):
                self.assertEqual(build_payload(other, messages, tool_names=names)['response_format'], {'type':'json_object'})
        self.assertNotIn('response_format', build_payload(replace(config, structured_output=False), messages, tool_names=names))
        native = build_payload(replace(config, native_tools=True), messages, tool_names=names)
        self.assertNotIn('response_format', native)
        self.assertEqual([item['function']['name'] for item in native['tools']], names)
        client = APIClient(config, keys=['test-key'])
        client.tool_names = names
        plain = APIClient(replace(config, structured_output=False), keys=['test-key'])
        self.assertGreater(client.count('Synthetic test'), plain.count('Synthetic test'))

    def test_kaggle_schema_roundtrip_still_rejects_extra_fields_without_retry(self):
        import httpx
        from desktop_agent.kaggle_server import ALIAS
        from desktop_agent.protocol import normalize_call
        for extra in (False, True):
            requests = []
            answer = {'tool':'finish','arguments':{'text':'Ready'}}
            if extra:
                answer['unexpected'] = 'private-extra-value'
            async def handler(request):
                payload = json.loads(request.content)
                requests.append(payload)
                self.assertEqual(payload['response_format']['type'], 'json_schema')
                return httpx.Response(200, json={'choices':[{'message':{'content':json.dumps(answer)},'finish_reason':'stop'}]})
            config = APISettings(url='https://example.lhr.life/v1', model=ALIAS, stream=False, max_retries=2)
            client = APIClient(config, keys=['test-key'], transport=httpx.MockTransport(handler))
            client.tool_names = ['finish']
            with self.subTest(extra=extra):
                if extra:
                    with self.assertRaisesRegex(ValueError, 'Only tool and arguments') as caught:
                        client.generate([{'role':'user','content':'Ready'}], None, threading.Event(), lambda text:None, lambda *args:None)
                    self.assertNotIn('private-extra-value', str(caught.exception))
                else:
                    action, _ = client.generate([{'role':'user','content':'Ready'}], None, threading.Event(), lambda text:None, lambda *args:None)
                    self.assertEqual(action, normalize_call(answer))
                self.assertEqual(len(requests), 1)

    def test_image_size_option_and_pixel_dimensions_in_all_transports(self):
        import base64
        import io
        from PIL import Image
        image = Image.new('RGB',(1920,1080),'white')
        messages = [{'role':'system','content':'Synthetic test'},{'role':'user','content':'Read image'}]
        for format,native in (('openai',False),('openai',True),('responses',False),('gemini',False),('anthropic',False)):
            for edge,size in ((1280,(1280,720)),(None,(1920,1080))):
                with self.subTest(format=format,native=native,edge=edge):
                    payload = build_payload(APISettings(format=format,native_tools=native,url='https://example.test'),messages,image,image_max_edge=edge)
                    raw = json.dumps(payload)
                    self.assertIn(f'{size[0]} x {size[1]} pixels',raw)
                    if format=='gemini':
                        data = payload['contents'][-1]['parts'][-1]['inlineData']['data']
                    elif format=='anthropic':
                        data = payload['messages'][-1]['content'][-1]['source']['data']
                    else:
                        content = payload['input' if format=='responses' else 'messages'][-1]['content'][-1]
                        url = content['image_url'] if format=='responses' else content['image_url']['url']
                        data = url.split(',',1)[1]
                    self.assertEqual(Image.open(io.BytesIO(base64.b64decode(data))).size,size)

    def test_runtime_state_remains_after_history_in_every_api_format(self):
        from desktop_agent.agent import pack_messages, system_prompt
        from desktop_agent.protocol import ToolCatalog
        capabilities = dict(screen=False,input=False,browser=True,approval='manual')
        catalog = ToolCatalog(capabilities,'Read the browser')
        system = system_prompt(capabilities,catalog)
        events = [dict(id=1,role='user',content='Inspect'),
                  dict(id=2,role='assistant',content=json.dumps({'tool':'browser_read','arguments':{}})),
                  dict(id=3,role='system',content=json.dumps({'tool':'browser_read','mode':'automatic','reason':''}),metadata={'status':'approval'}),
                  dict(id=4,role='tool',content='Page data',metadata={'tool':'browser_read','status':'delivered'})]
        first,_,_ = pack_messages(events,system,len,20000,state={'steps_remaining':19})
        second,_,_ = pack_messages(events,system,len,20000,state={'steps_remaining':18})
        for format,native in (('openai',False),('openai',True),('responses',False),('gemini',False),('anthropic',False)):
            with self.subTest(format=format,native=native):
                config = APISettings(format=format,native_tools=native,url='https://example.test/v1')
                before = build_payload(config,first,tool_names=catalog.names())
                after = build_payload(config,second,tool_names=catalog.names())
                key = 'input' if format == 'responses' else 'contents' if format == 'gemini' else 'messages'
                self.assertEqual(before[key][:-1],after[key][:-1])
                self.assertNotEqual(before[key][-1],after[key][-1])
                history = before[key][1:] if format == 'openai' else before[key]
                self.assertNotIn('automatic',json.dumps(history))
                self.assertNotIn('_result_tool',json.dumps(before))
                if native:
                    self.assertEqual(before['messages'][-2]['role'],'tool')
                    self.assertEqual(before['messages'][-1]['role'],'user')
                    self.assertEqual(before['tools'],after['tools'])
                for field in ('system','systemInstruction','instructions'):
                    if field in before:
                        self.assertEqual(before[field],after[field])

    def test_native_discovery_roundtrip_preserves_tool_roles_and_permissions(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Model, Settings
        from desktop_agent.store import Store
        from desktop_agent.tools import ToolResult
        requests = []
        sequence = [('load_tool_group',{'groups':['browser']}),('browser_read',{}),('finish',{'text':'Done'})]
        async def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            names = [item['function']['name'] for item in payload['tools']]
            self.assertNotIn('desktop_click',names)
            if len(requests) == 1:
                self.assertNotIn('browser_read',names)
            else:
                self.assertIn('browser_read',names)
                self.assertTrue(any(message['role'] == 'tool' for message in payload['messages']))
            name,arguments = sequence[len(requests)-1]
            return httpx.Response(200,json={'choices':[{'message':{'tool_calls':[{'id':'test','type':'function',
                'function':{'name':name,'arguments':json.dumps(arguments)}}]},'finish_reason':'tool_calls'}]})
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            config = APISettings(native_tools=True,stream=False)
            model = Model(Settings(backend='api',api=config))
            model.remote = APIClient(config,keys=['synthetic'],transport=httpx.MockTransport(handler))
            tools = Mock(allow_screen=False,allow_input=False,allow_browser=True,window=None,mode='routine')
            tools.execute.return_value = ToolResult('Page read')
            Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Proceed')
            self.assertEqual(len(requests),3)
            tools.execute.assert_called_once()
            self.assertEqual(json.loads(store.events(identifier)[-1]['content'])['message'],'Done')
            model.close()

    def test_native_incomplete_multiple_and_malformed_calls_never_validate(self):
        for finish,arguments in (('length','{"text":"Hi"}'),('stop','{"text":"Hi"}'),('tool_calls','{bad json')):
            result = StreamResult('openai',True,('finish',))
            result.consume({'choices':[{'delta':{'tool_calls':[{'function':{'name':'finish','arguments':arguments}}]},'finish_reason':finish}]})
            with self.assertRaises(ValueError):
                result.action()
        with self.assertRaises(ValueError):
            StreamResult('openai',True).consume({'choices':[{'delta':{'tool_calls':[{'index':0},{'index':1}]}}]})

    def test_native_tool_stream_and_nonstream_use_compact_validated_arguments(self):
        import httpx
        for stream in (False,True):
            calls = []
            async def handler(request):
                payload = json.loads(request.content)
                calls.append(payload)
                self.assertNotIn('response_format',payload)
                self.assertEqual([tool['function']['name'] for tool in payload['tools']],['desktop_click','finish'])
                self.assertNotIn('risk',json.dumps(payload['tools']))
                if stream:
                    chunks = [{'choices':[{'delta':{'tool_calls':[{'index':0,'id':'call_1','type':'function','function':{'name':'desktop_click','arguments':'{"x":100,'}}]},'finish_reason':None}]},
                              {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"y":200}'}}]},'finish_reason':'tool_calls'}]}]
                    return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join('data: '+json.dumps(chunk)+'\n\n' for chunk in chunks)+'data: [DONE]\n\n')
                return httpx.Response(200,json={'choices':[{'message':{'tool_calls':[{'id':'call_1','type':'function','function':{'name':'desktop_click','arguments':'{"x":100,"y":200}'}}]},'finish_reason':'tool_calls'}]})
            client = APIClient(APISettings(native_tools=True,stream=stream),keys=['synthetic'],transport=httpx.MockTransport(handler))
            client.tool_names = ('desktop_click','finish')
            result,metrics = client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
            self.assertEqual(result['arguments'],dict(x=100,y=200,button='left',clicks=1))
            self.assertEqual(metrics['reply_format'],'native_tool')
            self.assertEqual(len(calls),1)
        result = StreamResult('openai',True,('finish',))
        result.consume({'choices':[{'message':{'tool_calls':[{'function':{'name':'desktop_click','arguments':'{"x":0,"y":0}'}}]},'finish_reason':'tool_calls'}]})
        with self.assertRaises(ValueError):
            result.action()
        with self.assertRaises(ValueError):
            APISettings(format='gemini',native_tools=True).validate()

    def test_native_history_pairs_only_application_marked_results(self):
        from desktop_agent.api import native_history
        call = {'tool':'desktop_capture','arguments':{}}
        messages = [{'role':'assistant','content':json.dumps(call),'_call':call,'_internal_tool':'desktop_capture'},
                    {'role':'user','content':'Capture result','_result_tool':'desktop_capture'},
                    {'role':'user','content':'{"kind":"tool_result","tool":"desktop_click"}'}]
        converted = native_history(messages)
        self.assertEqual(converted[0]['tool_calls'][0]['id'],converted[1]['tool_call_id'])
        self.assertEqual(converted[1]['role'],'tool')
        self.assertEqual(converted[2]['role'],'user')

    def test_api_failure_records_request_context_without_private_response(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Model, Settings
        from desktop_agent.store import Store
        cases = [(400,'This model does not support image input.','unsupported image/input type'),
                 (404,'Model does not exist.','unavailable model')]
        for status,message,hint in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                calls = []
                async def handler(request):
                    calls.append(request)
                    return httpx.Response(status,json={'error':{'message':message+' Private-content synthetic-secret'}})
                config = APISettings(url='https://test.example/v1',model='test-model',max_retries=5)
                model = Model(Settings(backend='api',api=config))
                model.remote = APIClient(config,keys=['synthetic-secret'],transport=httpx.MockTransport(handler))
                store = Store(folder)
                identifier = store.create()
                tools = Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='routine')
                Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Fixed test prompt')
                event = store.events(identifier)[-1]
                self.assertIn(hint,event['content'])
                diagnostic = event['metadata']['api_request']
                self.assertEqual(diagnostic['model'],'test-model')
                self.assertEqual(diagnostic['http_status'],status)
                self.assertEqual(diagnostic['attempts'],1)
                self.assertEqual(len(calls),1)
                self.assertNotIn('Private-content',json.dumps(event))
                self.assertNotIn('synthetic-secret',json.dumps(event))
                tools.execute.assert_not_called()
                model.close()

    def test_invalid_risk_and_finish_arguments_are_distinct_nonexecuting_errors(self):
        for action,expected in ((dict(message='Reply',tool='finish',arguments={},risk='private-invalid'),'expected routine or sensitive'),
                                (dict(message='Reply',tool='finish',arguments={'private-field':'private-value'},risk='routine'),'empty object {}')):
            result = StreamResult('openai')
            result.terminal,result.finish = True,'stop'
            result.text = json.dumps(action)
            with self.assertRaises(ValueError) as error:
                result.action()
            self.assertIn(expected,str(error.exception))
            self.assertNotIn('private-',str(error.exception))

    def test_transport_json_and_model_json_failures_are_distinguished_without_echo(self):
        import httpx
        for data, expected in (
            ('not-json private-content','API SSE event is not valid JSON'),
            (json.dumps({'choices':[{'delta':{'content':''},'finish_reason':'stop'}]}),'API completed without answer text'),
            (json.dumps({'choices':[{'delta':{'content':'```json\n{"message":"private-content"\n```'},'finish_reason':'stop'}]}),'Model action JSON is malformed')):
            async def handler(request):
                return httpx.Response(200,headers={'content-type':'text/event-stream'},text='data: '+data+'\n\ndata: [DONE]\n\n')
            client = APIClient(APISettings(),keys=['synthetic'],transport=httpx.MockTransport(handler))
            with self.assertRaises(ValueError) as error:
                client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
            self.assertIn(expected,str(error.exception))
            self.assertNotIn('private-content',str(error.exception))

    def test_long_reasoning_does_not_abort_answer(self):
        thinking = 'x' * 250001
        events = {
            'openai': {'choices':[{'delta':{'reasoning_content':thinking}}]},
            'responses': {'type':'response.reasoning_summary_text.delta','delta':thinking},
            'gemini': {'candidates':[{'content':{'parts':[{'thought':True,'text':thinking}]}}]},
            'anthropic': {'type':'content_block_delta','delta':{'type':'thinking_delta','thinking':thinking}},
        }
        for format, event in events.items():
            with self.subTest(format=format):
                result = StreamResult(format)
                result.consume(event)
                self.assertEqual(result.thinking, thinking)
                result.text = 'Completed answer'
                result.terminal = True
                result.finish = {'openai':'stop','responses':'completed','gemini':'STOP','anthropic':'end_turn'}[format]
                self.assertEqual(result.action()['message'], 'Completed answer')
                result.text = 'x' * 100001
                with self.assertRaisesRegex(ValueError, 'client size limit'):
                    result.consume({})

    def test_plain_nvidia_greeting_is_a_reply_and_never_executes_tools(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Model, Settings
        from desktop_agent.store import Store
        greeting = '\uc548\ub155\ud558\uc138\uc694! \ubb34\uc5c7\uc744 \ub3c4\uc640\ub4dc\ub9b4\uae4c\uc694?'
        async def handler(request):
            chunks = [{'choices':[{'delta':{'content':greeting[:5]},'finish_reason':None}]},
                      {'choices':[{'delta':{'content':greeting[5:]},'finish_reason':'stop'}]}]
            return httpx.Response(200,headers={'content-type':'text/event-stream'},
                text=''.join('data: '+json.dumps(chunk)+'\n\n' for chunk in chunks)+'data: [DONE]\n\n')
        config = APISettings(url='https://integrate.api.nvidia.com/v1/chat/completions',model='meta/muse-glimmer-30b')
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            model = Model(Settings(backend='api',api=config))
            model.remote = APIClient(config,keys=['synthetic'],transport=httpx.MockTransport(handler))
            tools = Mock(allow_screen=True,allow_input=True,allow_browser=True,window=None,mode='routine')
            Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Hello')
            events = store.events(identifier)
            self.assertEqual([event['role'] for event in events],['user','assistant'])
            self.assertEqual(json.loads(events[-1]['content']),dict(message=greeting,tool='finish',arguments={},risk='routine'))
            self.assertEqual(events[-1]['metadata']['metrics']['reply_format'],'plain_text')
            tools.execute.assert_not_called()
            model.close()

    def test_complete_fenced_actions_validate_but_ambiguous_outputs_never_execute(self):
        action = dict(message='Inspect',tool='desktop_capture',arguments={},risk='routine')
        for format in ('openai','responses','gemini','anthropic'):
            result = StreamResult(format)
            result.terminal = True
            result.finish = {'openai':'stop','responses':'completed','gemini':'STOP','anthropic':'end_turn'}[format]
            result.text = '\ufeff ```json\n'+json.dumps(action)+'\n``` '
            self.assertEqual(result.action(),action)
            self.assertEqual(result.reply_format,'fenced_json')
            for malformed in ('{"message":"unfinished"',json.dumps(action)+'\n'+json.dumps(action),
                              '```json\n'+json.dumps(action)+'\n``` trailing text','<think>internal</think>'+json.dumps(action),
                              '[]','{"tool":"unknown"}'):
                result.text = malformed
                with self.assertRaises(ValueError):
                    result.action()
            result.text = 'Example only: '+json.dumps(action)
            self.assertEqual(result.action()['tool'],'finish')
            self.assertEqual(result.action()['message'],result.text)
            result.finish = 'length'
            result.text = 'Partial greeting'
            with self.assertRaises(ValueError):
                result.action()

    def test_bad_request_reports_only_known_option_names_without_echoed_secrets(self):
        import httpx
        private_text = 'Private prompt not for logging'
        bodies = [
            {'error':{'param':'response_format','message':'Unsupported response_format; synthetic-secret '+private_text}},
            {'detail':[{'loc':['body','extra_body'],'msg':'Extra inputs are not permitted','input':{'sensitive':private_text}}]},
            {'detail':[{'loc':['body','max_tokens'],'msg':'max_tokens out of range; '+private_text}]},
            {'error':{'message':private_text+' synthetic-secret'}}]
        for body,field in zip(bodies,('response_format','extra_body','max_tokens',None)):
            calls = []
            async def handler(request):
                calls.append(request)
                return httpx.Response(400,json=body)
            client = APIClient(APISettings(max_retries=4),keys=['synthetic-secret'],transport=httpx.MockTransport(handler))
            with self.assertRaises(ValueError) as error:
                client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
            self.assertEqual(len(calls),1)
            self.assertNotIn(private_text,str(error.exception))
            self.assertNotIn('synthetic-secret',str(error.exception))
            if field:
                self.assertIn('server referenced: '+field,str(error.exception))
            else:
                self.assertNotIn('server referenced',str(error.exception))

    def test_nvidia_request_does_not_inherit_gemini_thinking_extensions(self):
        import httpx
        config = APISettings(url='https://integrate.api.nvidia.com/v1/chat/completions',
                             model='meta/muse-glimmer-30b',thinking_budget=0,include_thoughts=True)
        messages = [{'role':'system','content':'Return an action JSON object'}, {'role':'user','content':'Hello'}]
        payload = build_payload(config,messages)
        self.assertNotIn('extra_body',payload)
        self.assertNotIn('thinking',json.dumps(payload))
        self.assertEqual(payload['model'],config.model)
        self.assertEqual(payload['max_tokens'],4096)
        self.assertEqual(payload['response_format'],{'type':'json_object'})
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        calls = []
        async def handler(request):
            calls.append(request)
            body = json.loads(request.content)
            if 'extra_body' in body:
                return httpx.Response(400)
            return httpx.Response(200,headers={'content-type':'text/event-stream'},
                text='data: '+json.dumps({'choices':[{'delta':{'content':json.dumps(action)},'finish_reason':'stop'}]})+'\n\ndata: [DONE]\n\n')
        client = APIClient(config,keys=['synthetic-nvidia-key'],transport=httpx.MockTransport(handler))
        result,metrics = client.generate(messages,None,threading.Event(),lambda text:None,lambda *args:None)
        self.assertEqual(result,action)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].url.host,'integrate.api.nvidia.com')
        for google in (APISettings(thinking_budget=500,include_thoughts=True),
                       APISettings(vertex=True,project='test-project',thinking_budget=500,include_thoughts=True)):
            self.assertEqual(build_payload(google,messages)['extra_body']['google']['thinking_config'],
                             {'thinking_budget':500,'include_thoughts':True})

    def test_vertex_agent_capture_to_image_reply_closes_completed_open_stream(self):
        import asyncio
        import httpx
        from unittest.mock import Mock
        from PIL import Image
        from desktop_agent.agent import Agent, Model, Settings
        from desktop_agent.credentials import KeyVault
        from desktop_agent.store import Store
        from desktop_agent.tools import ToolResult
        replies = [dict(message='Inspecting',tool='desktop_capture',arguments={},risk='routine'),
                   dict(message='Image inspected',tool='finish',arguments={},risk='routine')]
        requests = []
        class CompletedStream(httpx.AsyncByteStream):
            def __init__(self, reply):
                self.reply = reply
            async def __aiter__(self):
                yield ('data: '+json.dumps({'choices':[{'delta':{'content':json.dumps(self.reply)},'finish_reason':'stop'}]})+'\n\n').encode()
                await asyncio.Event().wait()
        async def handler(request):
            requests.append(json.loads(request.content))
            self.assertIn('aiplatform.googleapis.com',request.url.host)
            self.assertEqual(request.headers['authorization'],'Bearer synthetic-vertex-token')
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=CompletedStream(replies[len(requests)-1]))
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            vault = KeyVault(Path(folder)/'keys.dpapi')
            reference = vault.save_vertex(synthetic_vertex_account())
            api = APISettings(format='openai',vertex=True,project='test-project',vertex_credential_id=reference,
                              tokenizer='estimate',context_tokens=65536)
            client = APIClient(api,vault,transport=httpx.MockTransport(handler))
            client.prepare()
            client.credentials = Mock(valid=True,token='synthetic-vertex-token',quota_project_id=None)
            model = Model(Settings(backend='api',api=api),vault.path)
            model.remote = client
            screenshot = Image.new('RGB',(320,240),'white')
            tools = Mock(allow_screen=True,allow_input=False,allow_browser=False,window={'handle':1,'pid':2},mode='routine')
            tools.execute.return_value = ToolResult('Current capture',screenshot)
            stopped = threading.Event()
            watchdog = threading.Timer(3,stopped.set)
            watchdog.start()
            try:
                Agent(store,model,tools,stopped,Mock()).run(identifier,'Inspect selected window')
            finally:
                watchdog.cancel()
                model.close()
            events = store.events(identifier)
            self.assertFalse(stopped.is_set())
            self.assertEqual(len(requests),2)
            self.assertNotIn('image_url',json.dumps(requests[0]))
            self.assertEqual(requests[1]['messages'][-1]['content'][1]['type'],'image_url')
            self.assertTrue(requests[1]['messages'][-1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,'))
            tools.execute.assert_called_once()
            self.assertEqual(json.loads(events[-1]['content'])['message'],'Image inspected')
            self.assertEqual(events[-1]['metadata']['metrics']['image_count'],1)
            self.assertTrue(any(event['metadata'].get('image') for event in events))

    def test_usage_after_completion_is_collected_without_requiring_disconnect(self):
        import asyncio
        import httpx
        from PIL import Image
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        for format in ('openai','gemini'):
            terminal = ({'choices':[{'delta':{'content':json.dumps(action)},'finish_reason':'stop'}]} if format == 'openai'
                        else {'candidates':[{'content':{'parts':[{'text':json.dumps(action)}]},'finishReason':'STOP'}]})
            usage = {'usage':{'prompt_tokens':100,'completion_tokens':20}} if format == 'openai' else {'usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':20}}
            class UsageTail(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield ('data: '+json.dumps(terminal)+'\n\n').encode()
                    yield ('data: '+json.dumps(usage)+'\n\n').encode()
                    await asyncio.Event().wait()
            async def handler(request):
                return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=UsageTail())
            client = APIClient(APISettings(format=format,url='https://test.example/v1'),keys=['synthetic'],transport=httpx.MockTransport(handler))
            result,metrics = client.generate([],Image.new('RGB',(32,32),'white'),threading.Event(),lambda text:None,lambda *args:None)
            self.assertEqual(result,action)
            self.assertEqual(metrics['usage']['completion_tokens'],20)
            self.assertEqual(metrics['usage']['prompt_tokens'],100)

    def test_waiting_image_request_reports_stage_and_remains_cancellable(self):
        import asyncio
        import httpx
        from PIL import Image
        from game_agent.core import Halted
        from desktop_agent.agent import StopEvent
        stopped = StopEvent()
        updates = []
        async def handler(request):
            await asyncio.Event().wait()
        def notify(kind,value):
            if kind == 'status':
                updates.append(value)
                if 'waiting for response headers' in value:
                    stopped.set('Emergency stop (F8 / Esc)')
        client = APIClient(APISettings(),keys=['synthetic'],transport=httpx.MockTransport(handler))
        with self.assertRaises(Halted) as error:
            client.generate([],Image.new('RGB',(32,32),'white'),stopped,lambda text:None,notify)
        self.assertTrue(any('image + text' in update and 'waiting for response headers' in update for update in updates))
        self.assertIn('Emergency stop (F8 / Esc)',str(error.exception))
        self.assertIn('Earlier actions may already have run',str(error.exception))
        diagnostic = error.exception.api_diagnostics
        self.assertEqual(diagnostic['image_count'],1)
        self.assertEqual(diagnostic['attempts'],1)
        self.assertIsNone(diagnostic['http_status'])
        self.assertNotIn('synthetic',json.dumps(diagnostic))
        stopped.set('Another reason')
        self.assertEqual(stopped.reason,'Emergency stop (F8 / Esc)')
        stopped.clear()
        self.assertFalse(stopped.is_set())
        self.assertEqual(stopped.reason,'')

    def test_vertex_image_response_finishes_without_waiting_for_stream_disconnect(self):
        import asyncio
        import httpx
        from PIL import Image
        action = dict(message='Image inspected',tool='finish',arguments={},risk='routine')
        raw = json.dumps(action)
        terminal_events = {
            'openai':[{'choices':[{'delta':{'content':raw},'finish_reason':'stop'}]}],
            'gemini':[{'candidates':[{'content':{'parts':[{'text':raw}]},'finishReason':'STOP'}]}],
            'anthropic':[{'type':'content_block_delta','delta':{'type':'text_delta','text':raw}},
                         {'type':'message_delta','delta':{'stop_reason':'end_turn'}},{'type':'message_stop'}],
            'responses':[{'type':'response.output_text.delta','delta':raw},
                         {'type':'response.completed','response':{'status':'completed'}}]}
        for format,chunks in terminal_events.items():
            stopped = threading.Event()
            closed = []
            class OpenConnection(httpx.AsyncByteStream):
                async def __aiter__(self):
                    for chunk in chunks:
                        yield ('data: '+json.dumps(chunk)+'\n\n').encode()
                    await asyncio.Event().wait()
                async def aclose(self):
                    closed.append(True)
            async def handler(request):
                self.assertIn('image',request.content.decode().lower())
                return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=OpenConnection())
            client = APIClient(APISettings(format=format,url='https://test.example/v1'),
                               keys=['synthetic'],transport=httpx.MockTransport(handler))
            watchdog = threading.Timer(1,stopped.set)
            watchdog.start()
            try:
                result,metrics = client.generate([{'role':'user','content':'Inspect image'}],Image.new('RGB',(32,32),'white'),
                                                 stopped,lambda text:None,lambda *args:None)
                self.assertEqual(result,action)
                self.assertEqual(metrics['image_count'],1)
                self.assertTrue(closed)
            finally:
                watchdog.cancel()

    def test_retry_backoff_can_be_cancelled_and_key_switch_uses_budget(self):
        import httpx
        import time
        from game_agent.core import Halted
        from desktop_agent.api import retry_delay
        stopped = threading.Event()
        received = []
        async def limited(request):
            received.append(request)
            return httpx.Response(429,headers={'retry-after':'5'})
        def notify(kind,value):
            if kind == 'status' and 'cooldown:' in value:
                stopped.set()
        client = APIClient(APISettings(max_retries=3),keys=['synthetic'],transport=httpx.MockTransport(limited))
        started = time.monotonic()
        with self.assertRaises(Halted):
            client.generate([],None,stopped,lambda text:None,notify)
        self.assertLess(time.monotonic()-started,2)
        self.assertEqual(len(received),1)
        received.clear()
        async def rejected(request):
            received.append(request)
            return httpx.Response(401)
        for retries,count in ((0,1),(1,2),(5,3)):
            received.clear()
            client = APIClient(APISettings(max_retries=retries),keys=['synthetic-one','synthetic-two','synthetic-three'],
                               transport=httpx.MockTransport(rejected))
            with self.assertRaises(ValueError):
                client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
            self.assertEqual(len(received),count)
        self.assertEqual(retry_delay('nan',1),2)
        self.assertEqual(retry_delay(None,8),8)

    def test_vertex_retries_temporary_server_failure_with_same_credentials(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.credentials import KeyVault
        calls = []
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        async def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503,headers={'retry-after':'0'})
            return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'text':json.dumps(action)}]},'finishReason':'STOP'}]})
        with tempfile.TemporaryDirectory() as folder:
            vault = KeyVault(Path(folder)/'keys.dpapi')
            identifier = vault.save_vertex(synthetic_vertex_account())
            config = APISettings(format='gemini',vertex=True,project='test-project',vertex_credential_id=identifier,stream=False,max_retries=1)
            client = APIClient(config,vault,transport=httpx.MockTransport(handler))
            client.prepare()
            client.credentials = Mock(valid=True,token='synthetic-token',quota_project_id=None)
            result,metrics = client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
            self.assertEqual(result,action)
            self.assertEqual(metrics['request_attempts'],2)
            self.assertEqual([request.headers['Authorization'] for request in calls],['Bearer synthetic-token']*2)

    def test_named_profiles_roundtrip_independence_and_shared_vertex_references(self):
        from desktop_agent.agent import Settings
        original = APISettings(max_retries=4)
        settings = Settings().with_api_profile('Gemini work',original).with_api_profile('Claude',APISettings(format='anthropic'))
        original.model = 'changed-after-saving'
        self.assertEqual(settings.api_profiles['Gemini work'].model,'gemini-3.8-flash')
        settings = settings.use_api_profile('Gemini work')
        settings.api.max_retries = 1
        self.assertEqual(settings.api_profiles['Gemini work'].max_retries,4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            settings.save(path)
            reopened = Settings.load(path)
            self.assertEqual(reopened.active_api_profile,'Gemini work')
            self.assertEqual(reopened.use_api_profile('Gemini work').api.max_retries,4)
            self.assertEqual(reopened.api.max_retries,1)
            removed = reopened.without_api_profile('Gemini work')
            self.assertEqual(removed.active_api_profile,'')
            self.assertEqual(removed.api,reopened.api)
            self.assertIn('Claude',removed.api_profiles)
        reference = 'a'*32
        shared = settings.with_api_profile('Vertex1',APISettings(vertex=True,project='test-project',vertex_credential_id=reference))
        shared = shared.with_api_profile('Vertex2',shared.api_profiles['Vertex1'])
        self.assertIn(reference,shared.without_api_profile('Vertex1').vertex_credential_references())
        self.assertNotIn(reference,shared.without_api_profile('Vertex1').without_api_profile('Vertex2').vertex_credential_references())
        for name in ('',' '*3,'x'*81,'invalid\nname'):
            with self.assertRaises(ValueError):
                settings.with_api_profile(name,APISettings())

    def test_key_pools_are_isolated_between_profiles_on_same_server(self):
        from desktop_agent.credentials import KeyVault, credential_scope
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            vault = KeyVault(Path(folder)/'keys.dpapi')
            first,second = APISettings(key_profile_id='a'*32),APISettings(key_profile_id='b'*32)
            vault.save(credential_scope(endpoint(first),first.key_profile_id),['synthetic-first'])
            vault.save(credential_scope(endpoint(second),second.key_profile_id),['synthetic-second'])
            one,two = APIClient(first,vault),APIClient(second,vault)
            one.prepare()
            two.prepare()
            self.assertEqual(one.keys,['synthetic-first'])
            self.assertEqual(two.keys,['synthetic-second'])
            references = Settings(api=first).with_api_profile('Second',second).api_key_references()
            self.assertEqual(len(references),2)

    def test_retry_budget_counts_all_requests_and_zero_disables_retries(self):
        import httpx
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        for limit in (0,1,2):
            requests = []
            async def handler(request):
                requests.append(request)
                if len(requests) < 3:
                    return httpx.Response(503,headers={'retry-after':'0'})
                return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(action)},'finish_reason':'stop'}]})
            client = APIClient(APISettings(max_retries=limit,stream=False),keys=['synthetic'],transport=httpx.MockTransport(handler))
            if limit < 2:
                with self.assertRaisesRegex(ValueError,'retry limit'):
                    client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
            else:
                result,metrics = client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
                self.assertEqual(result,action)
                self.assertEqual(metrics['request_attempts'],3)
            self.assertEqual(len(requests),limit+1)
        for invalid in (-1,11,True):
            with self.assertRaises(ValueError):
                APISettings(max_retries=invalid).validate()

    def test_connect_error_retries_but_stream_failure_does_not(self):
        import httpx
        from unittest.mock import patch
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        received = []
        async def handler(request):
            received.append(request)
            if len(received) == 1:
                raise httpx.ConnectError('synthetic sensitive detail')
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(action)},'finish_reason':'stop'}]})
        with patch('desktop_agent.api.retry_delay',return_value=0):
            client = APIClient(APISettings(max_retries=1,stream=False),keys=['synthetic'],transport=httpx.MockTransport(handler))
            self.assertEqual(client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)[0],action)
        self.assertEqual(len(received),2)
        class BrokenStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                raise httpx.ReadError('synthetic sensitive detail')
        received.clear()
        async def broken(request):
            received.append(request)
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=BrokenStream())
        client = APIClient(APISettings(max_retries=5),keys=['synthetic'],transport=httpx.MockTransport(broken))
        with self.assertRaises(ValueError) as error:
            client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
        self.assertEqual(len(received),1)
        self.assertNotIn('sensitive',str(error.exception))

    def test_vertex_json_refresh_uses_google_token_endpoint_without_adc(self):
        from desktop_agent.credentials import KeyVault
        from unittest.mock import Mock, patch
        with tempfile.TemporaryDirectory() as folder:
            vault = KeyVault(Path(folder)/'keys.dpapi')
            identifier = vault.save_vertex(synthetic_vertex_account())
            client = APIClient(APISettings(vertex=True,project='test-project',vertex_credential_id=identifier),vault)
            client.prepare()
            response = Mock(status=200,data=json.dumps({'access_token':'synthetic-refreshed-token','token_type':'Bearer','expires_in':3600}).encode())
            request = Mock(return_value=response)
            with patch('google.auth.transport.requests.Request',return_value=request), \
                 patch('google.auth.default',side_effect=AssertionError('No ADC fallback')):
                headers = client.vertex_headers()
            self.assertEqual(headers['Authorization'],'Bearer synthetic-refreshed-token')
            self.assertEqual(request.call_args.kwargs['url'],'https://oauth2.googleapis.com/token')
            self.assertEqual(request.call_args.kwargs['timeout'],10)
            self.assertNotIn('PRIVATE KEY',str(request.call_args))
            self.assertEqual(client.redact('synthetic-refreshed-token'),'[REDACTED]')

    def test_vertex_json_import_encryption_validation_and_project_binding(self):
        from desktop_agent.credentials import KeyVault, read_vertex_file
        from desktop_agent.agent import Settings
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'service account.json'
            account = synthetic_vertex_account()
            source.write_text(json.dumps(account),encoding='utf-8-sig')
            vault = KeyVault(Path(folder)/'keys.dpapi')
            vault.save('https://other.example',['synthetic-other-key'])
            identifier = vault.save_vertex(read_vertex_file(source))
            source.unlink()
            restored = KeyVault(vault.path).load_vertex(identifier)
            self.assertEqual(restored,account)
            self.assertNotIn(b'PRIVATE KEY',vault.path.read_bytes())
            self.assertNotIn(account['private_key'].encode(),vault.path.read_bytes())
            settings = Settings(backend='api',api=APISettings(vertex=True,project=account['project_id'],vertex_credential_id=identifier))
            settings.save(Path(folder)/'settings.json')
            self.assertNotIn('PRIVATE KEY',(Path(folder)/'settings.json').read_text())
            client = APIClient(settings.api,vault)
            with patch('google.auth.default',side_effect=AssertionError('Must not use ADC')):
                client.prepare()
            self.assertEqual(client.credentials.service_account_email,account['client_email'])
            with self.assertRaisesRegex(ValueError,'does not match'):
                APIClient(replace(settings.api,project='other-project'),vault).prepare()
            for invalid in ({'type':'authorized_user'},dict(account,private_key='SENSITIVE-MALFORMED-KEY'),
                            dict(account,token_uri='https://untrusted.example/token'),dict(account,project_id='')):
                source.write_text(json.dumps(invalid),encoding='utf-8')
                with self.assertRaises(ValueError) as error:
                    read_vertex_file(source)
                self.assertNotIn('SENSITIVE-MALFORMED-KEY',str(error.exception))
            source.write_text('x'*65537,encoding='utf-8')
            with self.assertRaises(ValueError):
                read_vertex_file(source)
            vault.remove_vertex(identifier)
            self.assertEqual(vault.load('https://other.example'),['synthetic-other-key'])
            with self.assertRaises(ValueError):
                vault.load_vertex(identifier)

    def test_usage_whitelists_numbers_and_missing_usage_is_not_zero(self):
        from desktop_agent.api import normalize_usage
        self.assertEqual(normalize_usage({'prompt_tokens':12,'completion_tokens':20,'secret':'synthetic-secret'},'openai'),
                         {'prompt_tokens':12,'completion_tokens':20})
        self.assertEqual(normalize_usage({},'gemini'),{})
        self.assertEqual(normalize_usage({'prompt_tokens':'synthetic-secret'},'openai'),{})
        claude = normalize_usage({'input_tokens':100,'output_tokens':50,'cache_read_input_tokens':900,
                      'cache_creation_input_tokens':200},'anthropic')
        self.assertEqual(claude['prompt_tokens'],1200)
        self.assertEqual(claude['completion_tokens'],50)
        responses = normalize_usage({'input_tokens':1200,'output_tokens':50,
                          'input_tokens_details':{'cached_tokens':900}},'responses')
        self.assertEqual(responses['prompt_tokens'],1200)
        gemini = normalize_usage({'promptTokenCount':1200,'cachedContentTokenCount':900,
                      'candidatesTokenCount':50,'thoughtsTokenCount':20},'gemini')
        self.assertEqual(gemini['prompt_tokens'],1200)
        self.assertEqual(gemini['completion_tokens'],70)

    def test_non_streaming_formats_and_secret_redaction(self):
        import httpx
        secret = 'synthetic-provider-key'
        action = dict(message='Done '+secret,tool='finish',arguments={},risk='routine')
        raw = json.dumps(action)
        responses = {
            'openai':{'choices':[{'message':{'content':raw},'finish_reason':'stop'}]},
            'responses':{'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':raw}]}]},
            'gemini':{'candidates':[{'content':{'parts':[{'text':raw}]},'finishReason':'STOP'}]},
            'anthropic':{'type':'message','content':[{'type':'text','text':raw}],'stop_reason':'end_turn'}}
        for format, body in responses.items():
            captured = []
            async def handler(request):
                captured.append(request)
                return httpx.Response(200,json=body)
            client = APIClient(APISettings(format=format,url='https://test.example/v1',stream=False),keys=[secret],transport=httpx.MockTransport(handler))
            partials = []
            result, metrics = client.generate([{'role':'user','content':'Return JSON'}],None,threading.Event(),partials.append,lambda *args:None)
            self.assertEqual(result['message'],'Done [REDACTED]')
            self.assertNotIn(secret,json.dumps(partials))
            self.assertNotIn(secret,captured[0].url.__str__())
            self.assertNotIn(secret,captured[0].content.decode())

    def test_redirects_and_long_rate_limits_do_not_leak_or_retry(self):
        import httpx
        for status, headers in ((302,{'location':'https://other.example/steal'}),(429,{'retry-after':'120'}),(400,{})):
            received = []
            async def handler(request):
                received.append(request)
                return httpx.Response(status,headers=headers,text='synthetic-first secret error body')
            client = APIClient(APISettings(timeout_seconds=10),keys=['synthetic-first','synthetic-second'],transport=httpx.MockTransport(handler))
            with self.assertRaises(ValueError) as error:
                client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
            self.assertNotIn('synthetic-first',str(error.exception))
            self.assertEqual(len(received),1)

    def test_vertex_json_headers_and_missing_credentials_are_separate_from_keys(self):
        import httpx
        from unittest.mock import Mock, patch
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        received = []
        async def handler(request):
            received.append(request)
            return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'text':json.dumps(action)}]},'finishReason':'STOP'}]})
        from desktop_agent.credentials import KeyVault
        with tempfile.TemporaryDirectory() as folder:
            vault = KeyVault(Path(folder)/'keys.dpapi')
            identifier = vault.save_vertex(synthetic_vertex_account())
            config = APISettings(format='gemini',vertex=True,project='test-project',stream=False,vertex_credential_id=identifier)
            client = APIClient(config,vault,transport=httpx.MockTransport(handler))
            client.prepare()
            client.credentials = Mock(valid=True,token='synthetic-service-token',quota_project_id=None)
            with patch('google.auth.default',side_effect=AssertionError('Must not use ADC')):
                result,metrics = client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
            with self.assertRaisesRegex(ValueError,'Import a Vertex'):
                APIClient(replace(config,vertex_credential_id=''),vault).prepare()
        self.assertEqual(result,action)
        self.assertEqual(received[0].headers['authorization'],'Bearer synthetic-service-token')
        self.assertNotIn('x-goog-api-key',received[0].headers)
        self.assertNotIn('x-goog-user-project',received[0].headers)
        self.assertNotIn('PRIVATE KEY',received[0].content.decode())

    def test_tokenizer_choice_and_remote_budget(self):
        from desktop_agent.agent import Settings
        client = APIClient(APISettings(tokenizer='o200k_base'),keys=['synthetic'])
        self.assertGreater(client.count('Hello \ud55c\uae00 <|endoftext|>'),0)
        self.assertEqual(client.encoder.name,'o200k_base')
        settings = Settings(backend='api',api=APISettings(context_tokens=65536,max_output_tokens=8192))
        self.assertEqual(settings.context_limit,65536)
        self.assertEqual(settings.output_budget,8192)

    def test_http_key_rotation_and_round_robin_are_bounded(self):
        import httpx
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        received = []
        async def handler(request):
            received.append(request.headers['authorization'])
            if len(received) == 1:
                return httpx.Response(429,headers={'retry-after':'0'})
            chunk = {'choices':[{'delta':{'content':json.dumps(action)},'finish_reason':'stop'}],
                     'usage':{'prompt_tokens':100,'completion_tokens':20}}
            return httpx.Response(200,headers={'content-type':'text/event-stream'},text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
        config = APISettings(url='https://test.example/v1',rotation='round_robin')
        client = APIClient(config,keys=['synthetic-one','synthetic-two'],transport=httpx.MockTransport(handler))
        result,metrics = client.generate([{'role':'user','content':'JSON please'}],None,threading.Event(),lambda text:None,lambda *args:None)
        self.assertEqual(result,action)
        self.assertEqual(received,['Bearer synthetic-one','Bearer synthetic-two'])
        self.assertEqual(client.cursor,0)
        self.assertEqual(metrics['usage']['completion_tokens'],20)

    def test_partial_response_is_not_retried_and_api_never_loads_local_server(self):
        import httpx
        from unittest.mock import Mock
        from desktop_agent.agent import Model, Settings
        received = []
        async def handler(request):
            received.append(request)
            return httpx.Response(200,headers={'content-type':'text/event-stream'},
                text='data: '+json.dumps({'choices':[{'delta':{'content':'{"message":"partial"'}}]})+'\n\n')
        client = APIClient(APISettings(),keys=['synthetic-one','synthetic-two'],transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            client.generate([],None,threading.Event(),lambda text:None,lambda *args:None)
        self.assertEqual(len(received),1)
        model = Model(Settings(backend='api'))
        model.remote = client
        model.server = Mock()
        model.ensure(threading.Event())
        model.server.start.assert_not_called()

    def test_active_stream_cancels_without_waiting_for_next_chunk(self):
        import asyncio
        import httpx
        import time
        from game_agent.core import Halted
        stopped = threading.Event()
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                stopped.set()
                await asyncio.sleep(30)
                yield b'data: [DONE]\n\n'
        async def handler(request):
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=SlowStream())
        client = APIClient(APISettings(),keys=['synthetic'],transport=httpx.MockTransport(handler))
        started = time.monotonic()
        with self.assertRaises(Halted):
            client.generate([],None,stopped,lambda text:None,lambda *args:None)
        self.assertLess(time.monotonic()-started,2)

    def test_payload_images_and_reasoning_for_each_format(self):
        from PIL import Image
        messages = [{'role':'system','content':'Return JSON'},{'role':'user','content':'Inspect'}]
        image = Image.new('RGB',(32,32),'white')
        for format in ('openai','responses','gemini','anthropic'):
            config = APISettings(format=format,url='https://example.com/v1',reasoning_effort='low')
            payload = build_payload(config,messages,image)
            serialized = json.dumps(payload)
            self.assertNotIn('api_key',serialized)
            self.assertEqual(messages[-1]['content'],'Inspect')
            if format == 'gemini':
                self.assertEqual(payload['generationConfig']['thinkingConfig']['thinkingLevel'],'LOW')
                self.assertIn('inlineData',serialized)
            elif format == 'anthropic':
                self.assertEqual(payload['thinking']['type'],'adaptive')
                self.assertIn('base64',serialized)
            elif format == 'responses':
                self.assertFalse(payload['store'])
                self.assertIn('input_image',serialized)
            else:
                self.assertEqual(payload['reasoning_effort'],'low')
                self.assertIn('image_url',serialized)
        google = build_payload(APISettings(thinking_level='low'),messages)
        self.assertNotIn('reasoning_effort',google)
        self.assertEqual(google['extra_body']['google']['thinking_config']['thinking_level'],'low')
        vertex = build_payload(APISettings(format='anthropic',vertex=True,project='test'),messages)
        self.assertNotIn('model',vertex)
        self.assertEqual(vertex['anthropic_version'],'vertex-2023-10-16')

    def test_all_stream_formats_validate_completion_and_actions(self):
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        raw = json.dumps(action)
        events = {
            'openai':[{'choices':[{'delta':{'content':raw},'finish_reason':'stop'}]},{'usage':{'completion_tokens':20}}],
            'responses':[{'type':'response.output_text.delta','delta':raw},{'type':'response.completed','response':{'status':'completed','usage':{'output_tokens':20}}}],
            'gemini':[{'candidates':[{'content':{'parts':[{'text':'Summary','thought':True},{'text':raw}]},'finishReason':'STOP'}],'usageMetadata':{'candidatesTokenCount':20}}],
            'anthropic':[{'type':'content_block_delta','delta':{'type':'text_delta','text':raw}},
                         {'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':20}},{'type':'message_stop'}]}
        for format, chunks in events.items():
            result = StreamResult(format)
            with self.assertRaises(ValueError):
                result.action()
            for event in chunks:
                result.consume(event)
            self.assertEqual(result.action(),action)
            self.assertTrue(result.usage)
            result.finish = 'length'
            with self.assertRaises(ValueError):
                result.action()
        with self.assertRaises(ValueError):
            StreamResult('responses').consume({'type':'response.incomplete'})

    def test_keys_are_encrypted_scoped_and_not_in_settings(self):
        from desktop_agent.credentials import KeyVault
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'keys.dpapi'
            vault = KeyVault(path)
            vault.save('https://one.example',['synthetic-secret-one','synthetic-secret-two'])
            self.assertNotIn(b'synthetic-secret',path.read_bytes())
            self.assertEqual(KeyVault(path).load('https://one.example'),['synthetic-secret-one','synthetic-secret-two'])
            self.assertEqual(vault.load('https://two.example'),[])
            vault.save('https://one.example',[])
            self.assertEqual(vault.load('https://one.example'),[])

    def test_endpoint_routes_and_vertex(self):
        config = APISettings(url='https://generativelanguage.googleapis.com/v1beta/chat/completions')
        self.assertEqual(endpoint(config),'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions')
        self.assertEqual(endpoint(replace(config,format='responses',url='https://api.openai.com/v1')),'https://api.openai.com/v1/responses')
        gemini = replace(config,format='gemini',url='https://generativelanguage.googleapis.com/v1beta')
        self.assertTrue(endpoint(gemini).endswith('/models/gemini-3.8-flash:streamGenerateContent?alt=sse'))
        self.assertTrue(endpoint(replace(gemini,stream=False)).endswith(':generateContent'))
        vertex = replace(gemini,vertex=True,project='test-project',location='global')
        self.assertIn('https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/',endpoint(vertex))
        self.assertIn('streamRawPredict',endpoint(replace(vertex,format='anthropic')))
        self.assertIn('/endpoints/openapi/chat/completions',endpoint(replace(vertex,format='openai')))

    def test_invalid_configs_and_backward_compatible_settings(self):
        from desktop_agent.agent import Settings
        for config in (APISettings(reasoning_effort='low',thinking_level='high'),APISettings(url='http://example.com/v1'),
                       APISettings(url='https://example.com/v1?key=not-a-real-key'),APISettings(vertex=True),
                       APISettings(max_output_tokens=32768),APISettings(thinking_budget=True)):
            with self.assertRaises(ValueError):
                config.validate()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            path.write_text('{}',encoding='utf-8')
            self.assertEqual(Settings.load(path).backend,'local')
            Settings(backend='api').save(path)
            self.assertEqual(Settings.load(path).api.model,'gemini-3.8-flash')
            self.assertNotIn('api_key',json.loads(path.read_text(encoding='utf-8')))


class APIUITests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_switch_from_vertex_to_nvidia_disables_and_clears_only_inapplicable_options(self):
        import tkinter as tk
        from tkinter import ttk
        import time
        from desktop_agent.app import Console
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            app.withdraw()
            vertex = APISettings(vertex=True,project='test-project',thinking_budget=500,include_thoughts=True)
            app.settings = replace(app.settings,api=vertex).with_api_profile('Vertex saved',vertex)
            app.api_settings_dialog()
            dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
            tabs = next(widget for widget in dialog.winfo_children() if isinstance(widget,ttk.Notebook))
            connection,generation = [dialog.nametowidget(tab) for tab in tabs.tabs()]
            def control(frame,name):
                label = next(widget for widget in frame.winfo_children() if isinstance(widget,ttk.Label) and widget.cget('text') == name)
                return frame.grid_slaves(row=int(label.grid_info()['row']),column=1)[0]
            control(connection,'vertex').invoke()
            app.setvar(control(connection,'url').cget('textvariable'),'https://integrate.api.nvidia.com/v1/chat/completions')
            app.setvar(control(connection,'model').cget('textvariable'),'meta/muse-glimmer-30b')
            for name in ('thinking_budget','thinking_level','include_thoughts'):
                self.assertEqual(str(control(generation,name).cget('state')),'disabled')
            self.assertEqual(str(control(generation,'reasoning_effort').cget('state')),'readonly')
            self.assertEqual(str(control(generation,'native_tools').cget('state')),'normal')
            control(generation,'native_tools').invoke()
            app.setvar(control(connection,'format').cget('textvariable'),'gemini')
            self.assertEqual(str(control(generation,'native_tools').cget('state')),'disabled')
            app.setvar(control(connection,'format').cget('textvariable'),'openai')
            footer = dialog.winfo_children()[-1]
            next(widget for widget in footer.winfo_children() if isinstance(widget,ttk.Button) and widget.cget('text') == 'Save').invoke()
            deadline = time.monotonic()+10
            failures = []
            def finish():
                if app.busy and time.monotonic() < deadline:
                    app.after(50,finish)
                    return
                try:
                    self.assertFalse(app.busy)
                    self.assertEqual(app.settings.api.thinking_budget,-1)
                    self.assertFalse(app.settings.api.include_thoughts)
                    self.assertFalse(app.settings.api.vertex)
                    self.assertTrue(app.settings.api.native_tools)
                    self.assertEqual(app.settings.api_profiles['Vertex saved'].thinking_budget,500)
                    self.assertTrue(app.settings.api_profiles['Vertex saved'].include_thoughts)
                    self.assertTrue(app.settings.api_profiles['Vertex saved'].vertex)
                    self.assertEqual(app.store.events(app.identifier),[])
                except Exception as error:
                    failures.append(error)
                finally:
                    app.close()
            app.after(100,finish)
            app.mainloop()
            if failures:
                raise failures[0]

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_profile_save_load_delete_retry_settings_and_independent_keys(self):
        import tkinter as tk
        from tkinter import ttk
        from unittest.mock import patch
        import time
        from desktop_agent.app import Console
        from desktop_agent.agent import Settings
        from desktop_agent.credentials import credential_scope
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            app.withdraw()
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            app.model.vault.save(credential_scope(endpoint(app.settings.api)),['synthetic-shared-original'])
            app.api_settings_dialog()
            dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
            bar = next(widget for widget in dialog.winfo_children() if any(isinstance(child,ttk.Label) and child.cget('text') == 'API profile' for child in widget.winfo_children()))
            selector = next(widget for widget in bar.winfo_children() if isinstance(widget,ttk.Combobox))
            add = next(widget for widget in bar.winfo_children() if isinstance(widget,ttk.Button) and widget.cget('text') == '+')
            delete = next(widget for widget in bar.winfo_children() if isinstance(widget,ttk.Button) and widget.cget('text') == '\u00d7')
            def control(name):
                label = next(widget for widget in descendants(dialog) if isinstance(widget,ttk.Label) and widget.cget('text') == name)
                return label.master.grid_slaves(row=int(label.grid_info()['row']),column=1)[0]
            def set_value(name,value):
                app.setvar(control(name).cget('textvariable'),str(value))
            def select(name):
                selector.set(name)
                selector.event_generate('<<ComboboxSelected>>')
            set_value('max_retries',4)
            with patch('desktop_agent.app.simpledialog.askstring',return_value='Primary'):
                add.invoke()
            self.assertEqual(app.settings.api_profiles['Primary'].max_retries,4)
            self.assertEqual(app.settings.api.max_retries,2)
            first = app.settings.api_profiles['Primary']
            first_scope = credential_scope(endpoint(first),first.key_profile_id)
            self.assertEqual(app.model.vault.load(first_scope),['synthetic-shared-original'])
            set_value('model','backup-model')
            set_value('max_retries',0)
            with patch('desktop_agent.app.simpledialog.askstring',return_value='Backup'):
                add.invoke()
            second = app.settings.api_profiles['Backup']
            second_scope = credential_scope(endpoint(second),second.key_profile_id)
            self.assertNotEqual(first_scope,second_scope)
            app.model.vault.save(second_scope,['synthetic-backup'])
            self.assertEqual(app.model.vault.load(first_scope),['synthetic-shared-original'])
            select('Primary')
            self.assertEqual(control('max_retries').get(),'4')
            self.assertEqual(control('model').get(),'gemini-3.8-flash')
            set_value('model','unsaved-edit')
            with patch('desktop_agent.app.messagebox.askyesno',return_value=False):
                select('Backup')
            self.assertEqual(selector.get(),'Primary')
            self.assertEqual(control('model').get(),'unsaved-edit')
            with patch('desktop_agent.app.messagebox.askyesno',return_value=True):
                select('Backup')
                delete.invoke()
            self.assertNotIn('Backup',app.settings.api_profiles)
            self.assertEqual(app.model.vault.load(second_scope),[])
            self.assertIn('Primary',Settings.load(Path(folder)/'settings.json').api_profiles)
            select('Primary')
            next(widget for widget in descendants(dialog) if isinstance(widget,ttk.Button) and widget.cget('text') == 'Save').invoke()
            failures = []
            deadline = time.monotonic()+10
            def finish():
                if app.busy and time.monotonic() < deadline:
                    app.after(50,finish)
                    return
                try:
                    self.assertFalse(app.busy)
                    self.assertEqual(app.settings.active_api_profile,'Primary')
                    self.assertEqual(app.model.settings.api.max_retries,4)
                    self.assertEqual(app.settings.backend,'api')
                    self.assertEqual(app.store.events(app.identifier),[])
                    self.assertNotIn('synthetic-shared-original',(Path(folder)/'settings.json').read_text())
                except Exception as error:
                    failures.append(error)
                finally:
                    app.close()
            app.after(100,finish)
            app.mainloop()
            if failures:
                raise failures[0]

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_deleting_vertex_profile_keeps_other_profile_credentials(self):
        import tkinter as tk
        from tkinter import ttk
        from unittest.mock import patch
        from desktop_agent.app import Console
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            app.withdraw()
            reference = app.model.vault.save_vertex(synthetic_vertex_account())
            config = APISettings(vertex=True,project='test-project',vertex_credential_id=reference)
            app.settings = app.settings.with_api_profile('Vertex One',config).with_api_profile('Vertex Two',config)
            app.settings.save(Path(folder)/'settings.json')
            app.api_settings_dialog()
            dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
            bar = next(widget for widget in dialog.winfo_children() if any(isinstance(child,ttk.Label) and child.cget('text') == 'API profile' for child in widget.winfo_children()))
            selector = next(widget for widget in bar.winfo_children() if isinstance(widget,ttk.Combobox))
            delete = next(widget for widget in bar.winfo_children() if isinstance(widget,ttk.Button) and widget.cget('text') == '\u00d7')
            for name in ('Vertex One','Vertex Two'):
                selector.set(name)
                selector.event_generate('<<ComboboxSelected>>')
                with patch('desktop_agent.app.messagebox.askyesno',return_value=True):
                    delete.invoke()
                if name == 'Vertex One':
                    self.assertEqual(app.model.vault.load_vertex(reference)['project_id'],'test-project')
            with self.assertRaises(ValueError):
                app.model.vault.load_vertex(reference)
            self.assertEqual(app.settings.api_profiles,{})
            app.after(100,app.close)
            app.mainloop()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_vertex_json_picker_autofills_project_and_keeps_key_out_of_conversation(self):
        import tkinter as tk
        from tkinter import ttk
        from unittest.mock import patch
        import time
        from desktop_agent.app import Console
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'project key.json'
            source.write_text(json.dumps(synthetic_vertex_account()),encoding='utf-8')
            invalid = Path(folder)/'invalid.json'
            invalid.write_text('{malformed',encoding='utf-8')
            app = Console(Path(folder)/'appdata')
            app.withdraw()
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            def open_dialog():
                app.api_settings_dialog()
                dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
                tabs = next(widget for widget in dialog.winfo_children() if isinstance(widget,ttk.Notebook))
                connection = dialog.nametowidget(tabs.tabs()[0])
                return dialog,connection
            def import_file(dialog, path):
                button = next(widget for widget in descendants(dialog) if isinstance(widget,ttk.Button) and widget.cget('text') == 'Select JSON key file...')
                with patch('desktop_agent.app.filedialog.askopenfilename',return_value=str(path)):
                    button.invoke()
            dialog,connection = open_dialog()
            import_file(dialog,source)
            project = connection.grid_slaves(row=4,column=1)[0]
            self.assertEqual(project.get(),'test-project')
            self.assertEqual(str(project.cget('state')),'readonly')
            self.assertFalse(connection.grid_slaves(row=1,column=1))
            self.assertFalse(connection.grid_slaves(row=6,column=1))
            self.assertFalse(app.model.vault.path.exists())
            dialog.destroy()
            self.assertEqual(app.settings.api.vertex_credential_id,'')
            dialog,connection = open_dialog()
            with patch('desktop_agent.app.messagebox.showerror') as error:
                import_file(dialog,invalid)
                error.assert_called_once()
            self.assertFalse(app.model.vault.path.exists())
            import_file(dialog,source)
            next(widget for widget in descendants(dialog) if isinstance(widget,ttk.Button) and widget.cget('text') == 'Save').invoke()
            deadline = time.monotonic()+10
            failures = []
            saved = []
            def finish():
                if app.busy and time.monotonic() < deadline:
                    app.after(50,finish)
                    return
                try:
                    self.assertFalse(app.busy)
                    self.assertTrue(app.settings.api.vertex)
                    self.assertEqual(app.settings.api.project,'test-project')
                    saved.append(app.settings.api.vertex_credential_id)
                    self.assertEqual(app.model.vault.load_vertex(saved[0])['project_id'],'test-project')
                    self.assertEqual(app.attachments,[])
                    self.assertEqual(app.store.events(app.identifier),[])
                    self.assertIsNone(app.model.server.process)
                    persisted = (app.data/'settings.json').read_text(encoding='utf-8')
                    self.assertNotIn('PRIVATE KEY',persisted)
                    self.assertNotIn('project key.json',persisted)
                    source.unlink()
                    reopened,connection = open_dialog()
                    self.assertEqual(connection.grid_slaves(row=4,column=1)[0].get(),'test-project')
                    self.assertTrue(any(isinstance(widget,ttk.Label) and widget.cget('textvariable') and
                                        'Saved: test@' in str(app.getvar(widget.cget('textvariable'))) for widget in descendants(reopened)))
                    reopened.destroy()
                except Exception as error:
                    failures.append(error)
                finally:
                    app.close()
            app.after(100,finish)
            app.mainloop()
            if failures:
                raise failures[0]

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_api_settings_masked_keys_and_transfer_denial(self):
        import tkinter as tk
        from tkinter import ttk
        from unittest.mock import patch
        import time
        from desktop_agent.app import Console
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            app.withdraw()
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            app.api_settings_dialog()
            dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
            app.api_keys_dialog(APISettings(url='https://test.example/v1'),dialog)
            keys = next(widget for widget in dialog.winfo_children() if isinstance(widget,tk.Toplevel))
            entry = next(widget for widget in descendants(keys) if isinstance(widget,ttk.Entry))
            self.assertEqual(entry.cget('show'),'*')
            entry.insert(0,'synthetic-ui-key')
            next(widget for widget in descendants(keys) if isinstance(widget,ttk.Button) and widget.cget('text') == 'Save encrypted keys').invoke()
            self.assertEqual(app.model.vault.load('https://test.example'),['synthetic-ui-key'])
            next(widget for widget in descendants(dialog) if isinstance(widget,ttk.Radiobutton) and widget.cget('text') == 'External API').invoke()
            next(widget for widget in descendants(dialog) if isinstance(widget,ttk.Button) and widget.cget('text') == 'Save').invoke()
            deadline = time.monotonic()+10
            failures = []
            def finish():
                if app.busy and time.monotonic() < deadline:
                    app.after(50,finish)
                    return
                try:
                    self.assertFalse(app.busy)
                    self.assertEqual(app.settings.backend,'api')
                    self.assertEqual(str(app.think_button.cget('state')),'disabled')
                    app.prompt.set('Do not send')
                    with patch('desktop_agent.app.messagebox.askyesno',return_value=False):
                        app.send()
                    self.assertEqual(app.store.events(app.identifier),[])
                    self.assertIsNone(app.model.server.process)
                    self.assertNotIn('synthetic-ui-key',(Path(folder)/'settings.json').read_text(encoding='utf-8'))
                except Exception as error:
                    failures.append(error)
                finally:
                    app.close()
            app.after(100,finish)
            app.mainloop()
            if failures:
                raise failures[0]


if __name__ == '__main__':
    unittest.main()