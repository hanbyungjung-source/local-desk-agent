from dataclasses import dataclass
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import re
import threading
import time
from urllib.parse import quote, urlsplit, urlunsplit


FORMATS = ('openai','responses','gemini','anthropic')
TOKENIZERS = ('o200k_base','cl100k_base','p50k_base','r50k_base','estimate')


@dataclass
class APISettings:
    format: str = 'openai'
    url: str = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
    model: str = 'gemini-3.8-flash'
    tokenizer: str = 'o200k_base'
    context_tokens: int = 32768
    max_output_tokens: int = 4096
    token_limit_field: str = 'auto'
    stream: bool = True
    rotation: str = 'on_failure'
    key_profile_id: str = ''
    reasoning_effort: str = 'default'
    thinking_level: str = 'default'
    thinking_budget: int = -1
    include_thoughts: bool = False
    structured_output: bool = True
    native_tools: bool = False
    vertex: bool = False
    vertex_credential_id: str = ''
    project: str = ''
    location: str = 'us-central1'
    timeout_seconds: int = 120
    max_retries: int = 2

    def validate(self):
        if self.format not in FORMATS or self.tokenizer not in TOKENIZERS:
            raise ValueError('Unknown API format/tokenizer')
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError('API model name is required')
        if self.rotation not in ('on_failure','round_robin'):
            raise ValueError('Unknown key rotation mode')
        if self.token_limit_field not in ('auto','max_tokens','max_completion_tokens'):
            raise ValueError('Invalid OpenAI token limit field')
        if self.reasoning_effort not in ('default','none','minimal','low','medium','high','xhigh'):
            raise ValueError('Invalid reasoning_effort')
        if self.thinking_level not in ('default','minimal','low','medium','high'):
            raise ValueError('Invalid thinking_level')
        if sum((self.reasoning_effort != 'default',self.thinking_level != 'default',self.thinking_budget != -1)) > 1:
            raise ValueError('Choose only one of reasoning_effort, thinking_level or thinking_budget')
        if self.format == 'responses' and (self.thinking_level != 'default' or self.thinking_budget != -1):
            raise ValueError('Responses API uses reasoning_effort')
        if self.format == 'anthropic' and self.thinking_level != 'default':
            raise ValueError('Claude uses reasoning_effort (adaptive) or thinking_budget')
        if self.format == 'anthropic' and self.thinking_budget not in (-1,0) and self.thinking_budget < 1024:
            raise ValueError('Claude thinking_budget must be at least 1024')
        if self.thinking_budget >= self.max_output_tokens:
            raise ValueError('Thinking budget must be smaller than max output tokens')
        for value, minimum, maximum in ((self.context_tokens,4096,2097152),(self.max_output_tokens,256,131072),
                                        (self.thinking_budget,-1,65536),(self.timeout_seconds,10,600),(self.max_retries,0,10)):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError('API numeric setting out of range')
        if self.max_output_tokens+2048 >= self.context_tokens:
            raise ValueError('Context must exceed output budget by more than 2048 tokens')
        if any(type(value) is not bool for value in (self.stream,self.vertex,self.include_thoughts,self.structured_output,self.native_tools)):
            raise ValueError('Invalid API toggle')
        if self.native_tools and self.format != 'openai':
            raise ValueError('Native tools currently require OpenAI chat format; use compact JSON for other formats')
        if self.vertex:
            if self.format == 'responses':
                raise ValueError('Vertex does not expose the OpenAI Responses endpoint here')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*',self.project) or not re.fullmatch(r'[a-z0-9-]+',self.location):
                raise ValueError('Vertex project and location are required')
        if not isinstance(self.vertex_credential_id,str) or (self.vertex_credential_id and not re.fullmatch(r'[a-f0-9]{32}',self.vertex_credential_id)):
            raise ValueError('Invalid Vertex credential reference')
        if not isinstance(self.key_profile_id,str) or (self.key_profile_id and not re.fullmatch(r'[a-f0-9]{32}',self.key_profile_id)):
            raise ValueError('Invalid API key profile reference')
        endpoint(self)
        return self


def endpoint(config):
    if config.vertex:
        host = 'aiplatform.googleapis.com' if config.location == 'global' else config.location+'-aiplatform.googleapis.com'
        if config.location in ('us','eu'):
            host = 'aiplatform.'+config.location+'.rep.googleapis.com'
        base = f'https://{host}/v1/projects/{quote(config.project,safe="")}/locations/{config.location}'
        if config.format == 'openai':
            return base.replace('/v1/','/v1beta1/')+'/endpoints/openapi/chat/completions'
        publisher = 'anthropic' if config.format == 'anthropic' else 'google'
        method = ('streamRawPredict' if config.stream else 'rawPredict') if publisher == 'anthropic' else ('streamGenerateContent' if config.stream else 'generateContent')
        return base+'/publishers/'+publisher+'/models/'+quote(config.model,safe='-_.@')+':'+method+('?alt=sse' if config.stream and publisher == 'google' else '')
    parsed = urlsplit(config.url.strip())
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise ValueError('Use an endpoint URL without credentials, query strings or fragments')
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('localhost','127.0.0.1','::1')):
        raise ValueError('Remote API requires HTTPS (HTTP allowed only on loopback)')
    path = parsed.path.rstrip('/')
    if config.format == 'openai':
        if parsed.hostname == 'generativelanguage.googleapis.com' and path in ('/v1beta','/v1beta/chat/completions'):
            path = '/v1beta/openai/chat/completions'
        elif not path.endswith('/chat/completions'):
            path += '/chat/completions'
    elif config.format == 'responses':
        if not path.endswith('/responses'):
            path += '/responses'
    elif config.format == 'anthropic':
        if not path.endswith('/messages'):
            path += '/messages'
    else:
        method = ':streamGenerateContent' if config.stream else ':generateContent'
        if path.endswith((':streamGenerateContent',':generateContent')):
            path = path.rsplit(':',1)[0]+method
        else:
            path += '/models/'+quote(config.model,safe='-_.')+method
    return urlunsplit((parsed.scheme,parsed.netloc,path,'alt=sse' if config.format == 'gemini' and config.stream else '', ''))


def uses_google_thinking(config):
    return config.format == 'openai' and (config.vertex or
        urlsplit(config.url).hostname == 'generativelanguage.googleapis.com')


def default_tool_names():
    from desktop_agent.protocol import ToolCatalog
    return ToolCatalog(dict(screen=True,input=True,browser=True)).names()


def native_definitions(names):
    from desktop_agent.protocol import compact_parameters, tool_description
    return [{'type':'function','function':{'name':name,
             'description':tool_description(name),
             'parameters':compact_parameters(name)}} for name in names]


def native_history(messages):
    history = []
    pending = None
    for message in messages:
        row = {'role':message['role'],'content':message['content']}
        call = message.get('_call')
        if row['role'] == 'assistant' and isinstance(call,dict) and call['tool'] != 'finish':
            pending = (len(history),call,message.get('_internal_tool',call['tool']),message.get('_call_id',''))
        elif message.get('_result_tool') and pending:
            index,call,internal_tool,call_id = pending
            result_id = message.get('_call_id','')
            matches = (call_id==result_id) if call_id or result_id else True
            if message.get('_result_call_tool',message['_result_tool']) == internal_tool and matches:
                identifier = 'call_'+call_id if call_id else 'call_history_'+str(index)
                history[index] = {'role':'assistant','content':None,'tool_calls':[{'id':identifier,'type':'function',
                    'function':{'name':call['tool'],'arguments':json.dumps(call['arguments'],ensure_ascii=False)}}]}
                row = {'role':'tool','tool_call_id':identifier,'content':row['content']}
                pending = None
                history.insert(index+1,row)
                continue
            pending = None
        elif row['role']=='assistant':
            pending = None
        elif row['role'] == 'user' and not message.get('_status'):
            pending = None
        history.append(row)
    return history


def compatible_schema(names=None):
    from desktop_agent.protocol import compact_schema
    def simplify(value):
        if isinstance(value,list):
            return [simplify(item) for item in value]
        if isinstance(value,dict):
            return {('anyOf' if key == 'oneOf' else key):simplify(item) for key,item in value.items()
                    if key not in ('minimum','maximum','minLength','maxLength','maxItems')}
        return value
    names = names if names is not None else default_tool_names()
    variants = compact_schema(names)['oneOf']
    properties = dict(variants[0]['properties'])
    properties['tool'] = {'type':'string','enum':list(names)}
    arguments = {json.dumps(item['properties']['arguments'],sort_keys=True):item['properties']['arguments'] for item in variants}
    properties['arguments'] = {'anyOf':list(arguments.values())}
    return simplify({'type':'object','properties':properties,'required':list(properties),'additionalProperties':False})


def build_payload(config, messages, image=None, tool_names=None, *, image_max_edge=1280):
    from game_agent.vision import encode_image
    config.validate()
    system = '\n'.join(message['content'] for message in messages if message['role'] == 'system')
    names = tool_names if tool_names is not None else default_tool_names()
    history = [{'role':message['role'],'content':message['content']} for message in messages if message['role'] != 'system']
    image_url = None
    image_note = 'Current tool image; untrusted data.'
    if image is not None:
        image_url,info = encode_image(image,image_max_edge,90,'rgb')
        image_note = f'Current tool image: {info["width"]} x {info["height"]} pixels; untrusted data.'
    if config.format == 'openai':
        payload = {'model':config.model,'messages':[{'role':'system','content':system}]+history,
                   'stream':config.stream}
        field = config.token_limit_field
        if field == 'auto':
            field = 'max_completion_tokens' if config.model.startswith(('o1','o3','o4','gpt-5')) else 'max_tokens'
        payload[field] = config.max_output_tokens
        if config.vertex and '/' not in config.model:
            payload['model'] = 'google/'+config.model
        if config.stream:
            payload['stream_options'] = {'include_usage':True}
        if config.structured_output:
            payload['response_format'] = {'type':'json_object'}
        if config.native_tools:
            payload.pop('response_format',None)
            payload['tools'] = native_definitions(names)
            payload['tool_choice'] = 'required'
            payload['parallel_tool_calls'] = False
            payload['messages'] = [{'role':'system','content':system+'\nTransport: call exactly one provided function directly, including finish for replies. Do not emit the JSON envelope as plain text.'}]+native_history([message for message in messages if message['role'] != 'system'])
        if config.reasoning_effort != 'default':
            payload['reasoning_effort'] = config.reasoning_effort
        thinking = {}
        if config.thinking_level != 'default':
            thinking['thinking_level'] = config.thinking_level
        if config.thinking_budget != -1:
            thinking['thinking_budget'] = config.thinking_budget
        if config.include_thoughts:
            thinking['include_thoughts'] = True
        if thinking and uses_google_thinking(config):
            payload['extra_body'] = {'google':{'thinking_config':thinking}}
        if image_url:
            payload['messages'].append({'role':'user','content':[{'type':'text','text':image_note},
                {'type':'image_url','image_url':{'url':image_url}}]})
    elif config.format == 'responses':
        payload = {'model':config.model,'instructions':system,'input':history,'stream':config.stream,
                   'max_output_tokens':config.max_output_tokens,'store':False}
        if config.structured_output:
            payload['text'] = {'format':{'type':'json_object'}}
        if config.reasoning_effort != 'default':
            payload['reasoning'] = {'effort':config.reasoning_effort}
        if config.include_thoughts:
            payload.setdefault('reasoning',{})['summary'] = 'auto'
        if image_url:
            payload['input'].append({'role':'user','content':[{'type':'input_text','text':image_note},
                {'type':'input_image','image_url':image_url}]})
    elif config.format == 'gemini':
        payload = {'systemInstruction':{'parts':[{'text':system}]},'contents':[
            {'role':'model' if item['role'] == 'assistant' else 'user','parts':[{'text':item['content']}]} for item in history],
            'generationConfig':{'maxOutputTokens':config.max_output_tokens}}
        generation = payload['generationConfig']
        if config.structured_output:
            generation['responseMimeType'] = 'application/json'
        thinking = {}
        level = config.thinking_level if config.thinking_level != 'default' else config.reasoning_effort
        if level != 'default':
            if level == 'none':
                thinking['thinkingBudget'] = 0
            else:
                thinking['thinkingLevel'] = level.upper()
        if config.thinking_budget != -1:
            thinking['thinkingBudget'] = config.thinking_budget
        if config.include_thoughts:
            thinking['includeThoughts'] = True
        if thinking:
            generation['thinkingConfig'] = thinking
        if image_url:
            payload['contents'].append({'role':'user','parts':[{'text':image_note},
                {'inlineData':{'mimeType':'image/jpeg','data':image_url.split(',',1)[1]}}]})
    else:
        payload = {'model':config.model,'system':system,'messages':history,'max_tokens':config.max_output_tokens,'stream':config.stream}
        if config.structured_output:
            payload['output_config'] = {'format':{'type':'json_schema','schema':compatible_schema(names)}}
        if config.reasoning_effort not in ('default','none'):
            payload['thinking'] = {'type':'adaptive'}
            payload.setdefault('output_config',{})['effort'] = config.reasoning_effort
        elif config.thinking_budget > 0:
            payload['thinking'] = {'type':'enabled','budget_tokens':config.thinking_budget}
        elif config.reasoning_effort == 'none' or config.thinking_budget == 0:
            payload['thinking'] = {'type':'disabled'}
        if config.vertex:
            payload.pop('model')
            payload['anthropic_version'] = 'vertex-2023-10-16'
        if image_url:
            payload['messages'].append({'role':'user','content':[{'type':'text','text':image_note},
                {'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':image_url.split(',',1)[1]}}]})
    return payload


def decode_api_event(raw, source):
    try:
        event = json.loads(raw)
    except (ValueError,UnicodeError):
        raise ValueError(f'API {source} is not valid JSON; no tool executed') from None
    if not isinstance(event,dict):
        raise ValueError(f'API {source} must be a JSON object; no tool executed')
    return event


class StreamResult:
    def __init__(self, format, native_tools=False, tool_names=None):
        self.format = format
        self.native_tools,self.tool_names = native_tools,tool_names
        self.function_name = self.function_arguments = self.function_id = ''
        self.has_function = False
        self.text = self.thinking = ''
        self.usage = {}
        self.finish = None
        self.terminal = False
        self.timings = {}
        self.reply_format = 'action_json'

    def consume(self, event):
        if 'error' in event or event.get('type') in ('error','response.failed','response.incomplete'):
            raise ValueError('API reported an error or incomplete response; no tool executed')
        if self.format == 'openai':
            self.usage.update(event.get('usage') or {})
            self.timings.update(event.get('timings') or {})
            for choice in event.get('choices',[]):
                if choice.get('index',0) != 0:
                    continue
                delta = choice.get('delta') or choice.get('message') or {}
                if delta.get('refusal') or (delta.get('tool_calls') and not self.native_tools):
                    raise ValueError('API returned a refusal or unsupported native tool call')
                calls = delta.get('tool_calls') or []
                if len(calls) > 1:
                    raise ValueError('Only one native tool call per response is allowed')
                for call in calls:
                    if call.get('index',0) != 0 or call.get('type','function') != 'function':
                        raise ValueError('Unsupported native tool call')
                    if call.get('id'):
                        if self.function_id and self.function_id != call['id']:
                            raise ValueError('Multiple native tool calls were rejected')
                        self.function_id = call['id']
                    function = call.get('function',{})
                    self.has_function = True
                    self.function_name += function.get('name') or ''
                    self.function_arguments += function.get('arguments') or ''
                self.text += delta.get('content') or ''
                self.thinking += delta.get('reasoning_content') or ''
                if choice.get('finish_reason'):
                    self.finish = choice['finish_reason']
                    self.terminal = True
        elif self.format == 'responses':
            kind = event.get('type','')
            if kind == 'response.output_text.delta':
                self.text += event.get('delta','')
            elif kind == 'response.reasoning_summary_text.delta':
                self.thinking += event.get('delta','')
            elif kind == 'response.refusal.delta':
                raise ValueError('API declined the request')
            elif kind == 'response.completed' or 'output' in event:
                response = event.get('response',event)
                self.finish = response.get('status')
                self.terminal = True
                self.usage.update(response.get('usage') or {})
                if not self.text:
                    self.text = ''.join(part.get('text','') for item in response.get('output',[]) if item.get('type') == 'message'
                                        for part in item.get('content',[]) if part.get('type') == 'output_text')
        elif self.format == 'gemini':
            if event.get('promptFeedback',{}).get('blockReason'):
                raise ValueError('Gemini blocked the request')
            self.usage.update(event.get('usageMetadata') or {})
            for candidate in event.get('candidates',[])[:1]:
                for part in candidate.get('content',{}).get('parts',[]):
                    if part.get('thought'):
                        self.thinking += part.get('text','')
                    else:
                        self.text += part.get('text','')
                if candidate.get('finishReason'):
                    self.finish = candidate['finishReason']
                    self.terminal = True
        else:
            kind = event.get('type')
            if kind == 'message_start':
                self.usage.update(event.get('message',{}).get('usage') or {})
            elif kind == 'content_block_delta':
                delta = event.get('delta',{})
                if delta.get('type') == 'text_delta':
                    self.text += delta.get('text','')
                elif delta.get('type') == 'thinking_delta':
                    self.thinking += delta.get('thinking','')
            elif kind == 'content_block_start':
                block = event.get('content_block',{})
                if block.get('type') == 'text':
                    self.text += block.get('text','')
                elif block.get('type') not in ('thinking','redacted_thinking'):
                    raise ValueError('Unsupported Claude response block')
            elif kind == 'message_delta':
                self.finish = event.get('delta',{}).get('stop_reason',self.finish)
                self.usage.update(event.get('usage') or {})
            elif kind == 'message_stop':
                self.terminal = True
            elif kind == 'message':
                self.text = ''.join(part.get('text','') for part in event.get('content',[]) if part.get('type') == 'text')
                self.thinking = ''.join(part.get('thinking','') for part in event.get('content',[]) if part.get('type') == 'thinking')
                self.finish,self.terminal = event.get('stop_reason'),True
                self.usage.update(event.get('usage') or {})
        if len(self.text) > 100000 or len(self.thinking) > 200000 or len(self.function_arguments) > 30000 or len(self.function_name) > 100:
            raise ValueError('API response exceeded client size limit')

    def action(self):
        from desktop_agent.protocol import normalize_call
        if self.has_function:
            if not self.terminal or self.finish != 'tool_calls':
                raise ValueError('Incomplete native tool call; no tool executed')
            arguments = decode_api_event(self.function_arguments,'tool arguments')
            self.reply_format = 'native_tool'
            return normalize_call({'tool':self.function_name,'arguments':arguments},self.tool_names)
        valid = {'openai':('stop',),'responses':('completed',),'gemini':('STOP','FINISH_REASON_STOP'),'anthropic':('end_turn',)}
        if not self.terminal or self.finish not in valid[self.format]:
            raise ValueError('Incomplete or rejected API response; no tool executed')
        if not self.text.strip():
            raise ValueError('API completed without answer text; no tool executed')
        text = self.text.strip().lstrip('\ufeff').strip()
        fenced = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```',text,re.DOTALL|re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
            self.reply_format = 'fenced_json'
        try:
            action = json.loads(text)
        except ValueError:
            if fenced or text.startswith(('{','[','"','```','<think>','</think>')):
                raise ValueError(f'Model action JSON is malformed (characters={len(self.text)}); no tool executed') from None
            self.reply_format = 'plain_text'
            action = {'message':self.text,'tool':'finish','arguments':{},'risk':'routine'}
        return normalize_call(action,self.tool_names)


class APIClient:
    def __init__(self, config, vault=None, *, keys=None, transport=None):
        self.config,self.vault = config,vault
        self.keys = list(keys) if keys is not None else []
        self.supplied_keys = keys is not None
        self.transport = transport
        self.cursor = 0
        self.cancelled = threading.Event()
        self.credentials = None
        self.encoder = None
        self.tool_names = None
        self.cooldown_until = 0
        self.secrets = []

    def prepare(self):
        from desktop_agent.credentials import credential_scope
        self.config.validate()
        self.secrets = []
        if self.config.vertex:
            if self.vault is None:
                raise ValueError('Import a Vertex service-account JSON key in API settings')
            account = self.vault.load_vertex(self.config.vertex_credential_id)
            if account['project_id'] != self.config.project:
                raise ValueError('Vertex project does not match the imported JSON key; reimport it')
            self.secrets.extend((account['private_key'],account['private_key_id']))
            if self.credentials is None:
                from google.oauth2 import service_account
                self.credentials = service_account.Credentials.from_service_account_info(account,
                    scopes=['https://www.googleapis.com/auth/cloud-platform'])
        else:
            if not self.supplied_keys:
                self.keys = self.vault.load(credential_scope(endpoint(self.config),self.config.key_profile_id)) if self.vault else []
            if not self.keys:
                raise ValueError('No API key saved for this server; enter a key in API settings')
            if len(self.keys) > 20 or any(not isinstance(key,str) or not key or any(character.isspace() for character in key) for key in self.keys):
                raise ValueError('Invalid API key list')
            self.secrets = list(self.keys)
        return endpoint(self.config)

    def count(self, text):
        if self.config.native_tools:
            text += json.dumps(native_definitions(self.tool_names or default_tool_names()))
        elif self.config.format == 'anthropic' and self.config.structured_output:
            text += json.dumps(compatible_schema(self.tool_names))
        if self.config.tokenizer == 'estimate':
            return math.ceil(len(text.encode('utf-8'))/3)+32
        import tiktoken
        if self.encoder is None:
            try:
                self.encoder = tiktoken.get_encoding(self.config.tokenizer)
            except Exception:
                raise ValueError('Tokenizer could not be loaded; choose estimate or check tokenizer download access') from None
        return math.ceil(len(self.encoder.encode(text,disallowed_special=()))*1.15)+32

    def redact(self, text):
        for secret in self.secrets:
            if secret:
                text = text.replace(secret,'[REDACTED]')
        return text

    def vertex_headers(self):
        from google.auth.transport.requests import Request
        try:
            if self.credentials is None:
                self.prepare()
            if not self.credentials.valid:
                request = Request()
                self.credentials.refresh(lambda *args,**kwargs:request(*args,**dict(kwargs,timeout=10)))
            self.secrets.append(self.credentials.token)
            headers = {'Authorization':'Bearer '+self.credentials.token}
            if self.credentials.quota_project_id:
                headers['x-goog-user-project'] = self.credentials.quota_project_id
            return headers
        except Exception:
            raise ValueError('Vertex service-account authentication failed; check the imported JSON key and project access') from None

    def headers(self, key):
        headers = {'Content-Type':'application/json'}
        if self.config.format == 'gemini':
            headers['x-goog-api-key'] = key
        elif self.config.format == 'anthropic':
            headers.update({'x-api-key':key,'anthropic-version':'2023-06-01'})
        else:
            headers['Authorization'] = 'Bearer '+key
        return headers

    def cancel(self):
        self.cancelled.set()

    def generate(self, messages, image, stopped, on_text, notify):
        from game_agent.core import Halted
        self.prepare()
        if stopped.is_set():
            raise Halted('API request cancelled')
        self.cancelled.clear()
        payload = build_payload(self.config,messages,image,self.tool_names,image_max_edge=getattr(self,'image_max_edge',1280))
        return asyncio.run(self.run_request(payload,image is not None,stopped,on_text,notify))

    async def run_request(self, payload, has_image, stopped, on_text, notify):
        from game_agent.core import Halted
        started = time.monotonic()
        self.request_stage = 'Preparing request'
        self.request_attempt = 0
        self.request_http_status = None
        self.response_characters = self.reasoning_characters = 0
        async def watch():
            last_update = -1
            while not stopped.is_set() and not self.cancelled.is_set():
                elapsed = int(time.monotonic()-started)
                if elapsed != last_update:
                    last_update = elapsed
                    kind = 'image + text' if has_image else 'text'
                    notify('status',f'API {kind} | {self.request_stage} | attempt {self.request_attempt}/{self.config.max_retries+1} | {elapsed}s/{self.config.timeout_seconds}s')
                await asyncio.sleep(0.05)
        request = asyncio.create_task(self.request(payload,has_image,on_text,notify))
        watcher = asyncio.create_task(watch())
        try:
            done, pending = await asyncio.wait((request,watcher),timeout=self.config.timeout_seconds,return_when=asyncio.FIRST_COMPLETED)
            if watcher in done or stopped.is_set() or self.cancelled.is_set():
                reason = getattr(stopped,'reason','') or ('stop signal' if stopped.is_set() else 'client cancellation')
                raise Halted(f'API request cancelled by {reason} after {time.monotonic()-started:.1f}s while {self.request_stage}; no tool executed from this response. Earlier actions may already have run.')
            if request not in done:
                kind = 'image + text' if has_image else 'text'
                raise TimeoutError(f'API {kind} request exceeded {self.config.timeout_seconds}s while {self.request_stage}; no tool executed')
            return request.result()
        except Exception as error:
            error.api_diagnostics = {
                'model':self.redact(self.config.model),'format':self.config.format,'vertex':self.config.vertex,
                'host':urlsplit(endpoint(self.config)).hostname,'image_count':int(has_image),
                'seconds':round(time.monotonic()-started,2),'stage':self.request_stage,
                'attempts':self.request_attempt,'http_status':self.request_http_status,
                'answer_characters':self.response_characters,'reasoning_characters':self.reasoning_characters,
                'max_output_tokens':self.config.max_output_tokens,'structured_output':self.config.structured_output}
            raise
        finally:
            for task in (request,watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(request,watcher,return_exceptions=True)

    async def request(self, payload, has_image, on_text, notify):
        import httpx
        started = time.monotonic()
        attempts = self.config.max_retries+1
        key_count = 1 if self.config.vertex else len(self.keys)
        index = self.cursor % key_count
        rejected_keys = set()
        if self.config.vertex:
            self.request_stage = 'Authenticating Vertex'
            authorization = await asyncio.to_thread(self.vertex_headers)
        async with httpx.AsyncClient(transport=self.transport,follow_redirects=False,trust_env=False,
                timeout=httpx.Timeout(self.config.timeout_seconds,connect=10)) as client:
            for attempt in range(attempts):
                self.request_attempt = attempt+1
                delay = self.cooldown_until-time.monotonic()
                if delay > 0:
                    if delay >= self.config.timeout_seconds-(time.monotonic()-started):
                        raise ValueError('API retry cooldown exceeds remaining request timeout; retry later')
                    notify('status',f'API retry cooldown: {delay:.1f}s')
                    self.request_stage = 'Retry cooldown'
                    await asyncio.sleep(delay)
                headers = dict(authorization,**{'Content-Type':'application/json'}) if self.config.vertex else self.headers(self.keys[index])
                receiving = False
                self.request_stage = 'Sending request / waiting for response headers'
                try:
                    async with client.stream('POST',endpoint(self.config),headers=headers,json=payload) as response:
                        status = response.status_code
                        self.request_http_status = status
                        if status == 200:
                            receiving = True
                            self.request_stage = 'Waiting for response content'
                            action,metrics = await self.read_response(response,has_image,started,on_text,notify)
                            self.cursor = (index+1 if self.config.rotation == 'round_robin' else index)%key_count
                            metrics['request_attempts'] = attempt+1
                            return action,metrics
                        if status in (401,403):
                            rejected_keys.add(index)
                            available = [candidate%key_count for candidate in range(index+1,index+key_count+1)
                                         if candidate%key_count not in rejected_keys]
                            if self.config.vertex or not available or attempt+1 >= attempts:
                                raise ValueError(f'API HTTP {status}; credentials rejected or retry limit reached')
                            index = available[0]
                        elif status in (408,429,500,502,503,504,529):
                            self.cooldown_until = time.monotonic()+retry_delay(response.headers.get('retry-after'),attempt)
                            if attempt+1 >= attempts:
                                raise ValueError(f'API HTTP {status}; retry limit reached ({attempts} request(s))')
                            if status == 429 and not self.config.vertex:
                                index = next((candidate%key_count for candidate in range(index+1,index+key_count+1)
                                              if candidate%key_count not in rejected_keys),index)
                        else:
                            hint = await request_error_hint(response) if status in (400,404,415,422) else ''
                            explanation = 'route or model not found; verify the provider model ID, endpoint, region and access' if status == 404 else 'request rejected; check supported options, token limits and input types'
                            raise ValueError(f'API HTTP {status}; {explanation}; model={self.redact(self.config.model)}'+hint)
                        notify('status',f'API HTTP {status}; retry {attempt+1}/{self.config.max_retries}')
                        self.request_stage = f'HTTP {status}; retry pending'
                except httpx.HTTPError as error:
                    transient = isinstance(error,(httpx.ConnectError,httpx.ConnectTimeout,httpx.ReadTimeout,httpx.ReadError,httpx.RemoteProtocolError))
                    if receiving or not transient or attempt+1 >= attempts:
                        raise ValueError('API network/stream failure ('+type(error).__name__+'); no further retry') from None
                    self.cooldown_until = time.monotonic()+retry_delay(None,attempt)
                    notify('status',f'API connection failure; retry {attempt+1}/{self.config.max_retries}')
                    self.request_stage = 'Connection failure; retry pending'
                except (json.JSONDecodeError,UnicodeError):
                    raise ValueError('API returned invalid JSON; no tool executed') from None

    async def read_response(self, response, has_image, started, on_text, notify):
        from httpx_sse import EventSource
        result = StreamResult(self.config.format,self.config.native_tools,self.tool_names)
        first_token = None
        if self.config.stream:
            source = EventSource(response)
            events = source.aiter_sse()
            async for event in events:
                if not event.data:
                    continue
                if event.data.strip() == '[DONE]':
                    break
                result.consume(decode_api_event(event.data,'SSE event'))
                self.response_characters,self.reasoning_characters = len(result.text)+len(result.function_arguments),len(result.thinking)
                if result.has_function:
                    self.request_stage = 'Receiving tool call'
                elif result.text:
                    self.request_stage = 'Receiving answer'
                elif result.thinking:
                    self.request_stage = 'Receiving reasoning summary'
                if first_token is None and (result.text or result.thinking or result.has_function):
                    first_token = time.monotonic()-started
                on_text(self.redact(result.text))
                if result.thinking:
                    notify('reasoning',self.redact(result.thinking[-20000:]))
                if result.terminal:
                    self.request_stage = 'Finalizing response'
                    if self.config.format in ('openai','gemini'):
                        await self.collect_trailing_usage(events,result)
                    break
        else:
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > 2*1024*1024:
                    raise ValueError('API response too large')
            result.consume(decode_api_event(data,'HTTP response'))
            self.response_characters,self.reasoning_characters = len(result.text)+len(result.function_arguments),len(result.thinking)
            on_text(self.redact(result.text))
            if result.thinking:
                notify('reasoning',self.redact(result.thinking[-20000:]))
        result.text = self.redact(result.text)
        result.function_arguments = self.redact(result.function_arguments)
        action = result.action()
        seconds = time.monotonic()-started
        usage = normalize_usage(result.usage,self.config.format)
        timings = {key:value for key,value in result.timings.items() if key in (
            'cache_n','prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_ms','predicted_per_second')
            and type(value) in (int,float) and 0 <= value <= 1000000000 and math.isfinite(value)}
        return action,{'seconds':seconds,'first_token_seconds':first_token,'usage':usage,'timings':timings,
            'reasoning':self.redact(result.thinking[-20000:]),'image_count':int(has_image),'backend':'api',
            'reply_format':result.reply_format,
            'api_format':self.config.format,'model':self.config.model,'tokenizer_estimate':self.config.tokenizer,
            'observed_output_tps':usage['completion_tokens']/seconds if seconds and 'completion_tokens' in usage else None}

    async def collect_trailing_usage(self, events, result):
        import httpx
        from httpx_sse import SSEError
        async def collect():
            async for event in events:
                if event.data.strip() == '[DONE]':
                    return
                if not event.data:
                    continue
                data = json.loads(event.data)
                usage = data.get('usage' if self.config.format == 'openai' else 'usageMetadata')
                if isinstance(usage,dict):
                    result.usage.update(usage)
                    return
        try:
            await asyncio.wait_for(collect(),timeout=0.2)
        except (asyncio.TimeoutError,httpx.HTTPError,SSEError,ValueError):
            pass


async def request_error_hint(response):
    import httpx
    async def read_hint():
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > 16384:
                return ''
        data = json.loads(body)
        if not isinstance(data,dict):
            return ''
        error = data.get('error',data.get('detail',data))
        entries = error if isinstance(error,list) else [error]
        descriptions = []
        for entry in entries[:20]:
            if isinstance(entry,str):
                descriptions.append(entry)
            elif isinstance(entry,dict):
                descriptions.extend(entry[field] for field in ('param','message','msg') if isinstance(entry.get(field),str))
                descriptions.extend(item for item in entry.get('loc',[]) if isinstance(item,str))
        text = ' '.join(descriptions)
        fields = [name for name in ('extra_body','thinking_config','thinking_budget','thinking_level','include_thoughts',
                                    'reasoning_effort','response_format','stream_options','max_tokens','max_completion_tokens',
                                    'model','messages','image_url') if re.search(r'(?<![\w])'+name+r'(?![\w])',text,re.I)]
        hint = '; server referenced: '+', '.join(fields) if fields else ''
        if 'response_format' in fields:
            hint += '; this provider/model may require structured_output=False'
        if any(field in fields for field in ('max_tokens','max_completion_tokens')):
            hint += '; check max_output_tokens and token_limit_field'
        lower = text.lower()
        unsupported = any(word in lower for word in ('not support','unsupported','not allowed','only supports','invalid'))
        if unsupported and any(word in lower for word in ('image','vision','multimodal')):
            hint += '; server indicates an unsupported image/input type; use a vision-capable model for captures'
        if any(word in lower for word in ('context length','context window','too many tokens')):
            hint += '; server indicates a context/token limit; reduce conversation or output budget'
        if 'model' in lower and any(word in lower for word in ('not found','does not exist','not available','unknown model')):
            hint += '; server indicates an unavailable model; verify its exact provider ID and deployment access'
        return hint
    try:
        return await asyncio.wait_for(read_hint(),timeout=0.5)
    except (ValueError,TypeError,RecursionError,asyncio.TimeoutError,httpx.HTTPError):
        return ''


def retry_delay(value, attempt):
    fallback = min(2**attempt,8)
    if value is None:
        return fallback
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds()
        except (ValueError,TypeError,OverflowError):
            return fallback
    return max(0,delay) if math.isfinite(delay) else fallback


def normalize_usage(usage, format):
    numeric = lambda value:type(value) in (int,float) and 0 <= value <= 1000000000
    if format == 'gemini':
        result = {'prompt_tokens':usage.get('promptTokenCount'),'thinking_tokens':usage.get('thoughtsTokenCount'),
                  'cached_tokens':usage.get('cachedContentTokenCount')}
        if numeric(usage.get('candidatesTokenCount')):
            result['completion_tokens'] = usage['candidatesTokenCount']+(usage.get('thoughtsTokenCount',0) if numeric(usage.get('thoughtsTokenCount',0)) else 0)
    elif format in ('responses','anthropic'):
        result = {'prompt_tokens':usage.get('input_tokens'),'completion_tokens':usage.get('output_tokens'),
                  'cached_tokens':usage.get('cache_read_input_tokens',usage.get('input_tokens_details',{}).get('cached_tokens'))}
        if format == 'anthropic' and numeric(result['prompt_tokens']):
            for key in ('cache_read_input_tokens','cache_creation_input_tokens'):
                if numeric(usage.get(key)):
                    result['prompt_tokens'] += usage[key]
    else:
        result = {key:usage.get(key) for key in ('prompt_tokens','completion_tokens','total_tokens')}
        details=usage.get('prompt_tokens_details') or {}
        result['cached_tokens']=details.get('cached_tokens') if isinstance(details,dict) else None
    return {key:int(value) for key,value in result.items() if numeric(value)}