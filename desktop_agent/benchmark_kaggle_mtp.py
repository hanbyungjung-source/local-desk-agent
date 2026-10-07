import base64
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import urllib.request


def with_draft_length(command, length):
    if type(length) is not int or length not in (2, 3):
        raise ValueError('Only fixed MTP lengths 2 and 3 are supported')
    if command.count('--spec-draft-n-max') != 1:
        raise ValueError('Expected one explicit MTP length')
    result = list(command)
    result[result.index('--spec-draft-n-max')+1] = str(length)
    return result


def input_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_cases():
    from PIL import Image, ImageDraw
    from desktop_agent.agent import SYSTEM
    from desktop_agent.api import APISettings, build_payload
    from desktop_agent.kaggle_server import ALIAS
    config = APISettings(url='https://benchmark.lhr.life/v1', model=ALIAS,
                         context_tokens=196608, max_output_tokens=4096,
                         tokenizer='estimate', reasoning_effort='xhigh', max_retries=0)
    records = '\n'.join('ROW'+str(index).zfill(2)+':READY' for index in range(1, 25))
    picture = Image.new('RGB', (1280, 720), 'white')
    draw = ImageDraw.Draw(picture)
    draw.rectangle((100, 140, 560, 580), fill=(220, 35, 45))
    draw.rectangle((720, 140, 1180, 580), fill=(25, 80, 220))
    specifications = [
        ('warmup', 'Return the number of letters in ORBIT as text, with no explanation.', '5', None),
        ('copy', 'Return exactly the following records, preserving every line and adding no other text:\n'+records, records, None),
        ('arithmetic', 'There are 7 boxes with 18 units each. Then 29 units are removed and 14 units are added. Return only the final total as text.', '111', None),
        ('image', 'Name the color of the large rectangle on the right. Return exactly BLUE or RED as text.', 'BLUE', picture),
    ]
    cases = []
    for name, prompt, expected, image in specifications:
        payload = build_payload(config, [{'role':'system','content':SYSTEM},
                                        {'role':'user','content':prompt}], image, ['finish'])
        payload.update(temperature=0, seed=42, timings=True, cache_prompt=True)
        cases.append({'name':name, 'expected':expected, 'payload':payload,
                      'input_sha256':input_hash(payload)})
    return cases


def summarize(arms):
    if {arm['length'] for arm in arms} != {2, 3} or len(arms) != 2:
        raise ValueError('Both fixed lengths are required')
    reference = None
    summary = []
    for arm in arms:
        samples = arm['samples']
        if (not arm['passed'] or len(samples) != 4 or
                [row['name'] for row in samples] != ['warmup','copy','arithmetic','image'] or
                not all(row['passed'] for row in samples)):
            raise ValueError('Incomplete or incorrect comparison; keep the original default')
        inputs = [row['input_sha256'] for row in samples]
        if reference is not None and inputs != reference:
            raise ValueError('Comparison prompts differ')
        reference = inputs
        scored = samples[1:]
        milliseconds = sum(row.get('timings',{}).get('predicted_ms',0) for row in scored)
        tokens = sum(max(0,row.get('timings',{}).get('predicted_n',0)-1) for row in scored)
        complete_timings = all(row.get('timings',{}).get('predicted_ms',0)>0 and
                       row.get('timings',{}).get('predicted_n',0)>1 for row in scored)
        summary.append({'length':arm['length'], 'scored_seconds':sum(row['seconds'] for row in scored),
                        'output_tokens':[row.get('usage',{}).get('completion_tokens') for row in scored],
                'decode_tps':1000*tokens/milliseconds if complete_timings else None,
                        'case_seconds':{row['name']:row['seconds'] for row in scored}})
    chosen = min(summary, key=lambda row:(row['scored_seconds'], -row['length']))
    return {'winner':chosen['length'], 'criterion':'sum of non-warmup response seconds', 'arms':summary,
            'same_output_token_counts':summary[0]['output_tokens']==summary[1]['output_tokens'],
            'limitations':'One pass per length; fixed synthetic prompts; no randomized repeats or full-context stress.'}


def run_arm(server, length, directory, cases, key):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    command = list(server.args)
    if server.poll() is not None or command != with_draft_length(command, length):
        raise ValueError('Server is not running the requested fixed MTP length')
    result = {'length':length, 'pid':server.pid, 'command':command, 'samples':[], 'passed':False}
    try:
        for case in cases:
            if input_hash(case['payload']) != case['input_sha256']:
                raise ValueError('Frozen request changed')
            row = {'name':case['name'], 'input_sha256':case['input_sha256'], 'passed':False}
            result['samples'].append(row)
            timed_out = threading.Event()
            def stop_owned_server():
                timed_out.set()
                if server.poll() is None:
                    server.terminate()
            timer = threading.Timer(120, stop_owned_server)
            text, finish, usage, timings = '', None, {}, {}
            started = time.monotonic()
            timer.start()
            try:
                request = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',
                    data=json.dumps(case['payload']).encode(),
                    headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    for raw in response:
                        if not raw.startswith(b'data:'):
                            continue
                        data = raw[5:].strip()
                        if data == b'[DONE]':
                            break
                        event = json.loads(data)
                        for choice in event.get('choices', []):
                            text += choice.get('delta', {}).get('content') or ''
                            finish = choice.get('finish_reason') or finish
                        usage.update(event.get('usage') or {})
                        timings.update(event.get('timings') or {})
                        if len(text) > 100000:
                            raise ValueError('Unexpected response size')
                row.update(seconds=time.monotonic()-started, usage=usage, timings=timings)
                action = json.loads(text)
                row['passed'] = (not timed_out.is_set() and finish == 'stop' and
                    isinstance(action, dict) and set(action)=={'tool','arguments'} and action['tool']=='finish' and
                    isinstance(action['arguments'], dict) and
                    not set(action['arguments'])-{'text','commentary'} and
                    action['arguments'].get('text')==case['expected'])
                row['answer_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                if not row['passed']:
                    raise ValueError('Incorrect or incomplete answer')
            finally:
                timer.cancel()
                timer.join()
            print('MTP_CASE', length, case['name'], round(row['seconds'],3), row['passed'], flush=True)
        result['passed'] = True
    except Exception as error:
        result['error_type'] = type(error).__name__
    finally:
        with (directory / 'results.json').open('x', encoding='utf-8') as output:
            json.dump(result, output, indent=2)
    return result