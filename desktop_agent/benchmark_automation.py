import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from unittest.mock import patch

from desktop_agent.agent import pack_messages
from desktop_agent.jobs import StopSignal, ToolRunner
from desktop_agent.protocol import normalize_call
from desktop_agent.tools import Tools, ToolResult


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()


def check_prefix():
    events=[dict(id=1,role='user',content='fixture',metadata={})]
    action=normalize_call(dict(tool='desktop_key',arguments={'key':'Enter'}))
    events.extend([dict(id=2,role='assistant',content=json.dumps(action),metadata={'call_id':'fixture-call'}),
                   dict(id=3,role='tool',content='ok',metadata={'tool':'desktop_key','status':'delivered','call_id':'fixture-call'})])
    before,_,_=pack_messages(events,'fixed guide',len,100000,state={'steps_remaining':20})
    after,_,_=pack_messages(events+[dict(id=4,role='user',content='next',metadata={})],
                            'fixed guide',len,100000,state={'steps_remaining':19})
    before_history=[message for message in before if message.get('_event_id')]
    after_history=[message for message in after if message.get('_event_id')][:len(before_history)]
    assert before_history==after_history
    assert before[-1]['_status'] and before[-1]!=after[-1]
    return {'historical_messages_stable':True,'state_suffix_changes':True,'tokenizer_used':False}


def measured(function):
    started=time.perf_counter()
    result=function()
    return result,time.perf_counter()-started


def tool_measurements(output):
    report={}
    with tempfile.TemporaryDirectory(prefix='local-desk-speed-') as temporary:
        root=Path(temporary)
        workspace=root/'workspace'
        workspace.mkdir()
        for index in range(4):
            (workspace/f'fixture-{index}.txt').write_text((f'fixture {index} payload '+('a'*110)+'\n')*60000,encoding='utf-8')
        children=[normalize_call(dict(tool='desktop_type',arguments={'text':'fixture'})),
                  normalize_call(dict(tool='desktop_key',arguments={'key':'Enter'})),
                  normalize_call(dict(tool='desktop_key',arguments={'key':'Enter'}))]
        macro=normalize_call(dict(tool='desktop_macro',arguments=dict(actions=children,repeat=100,interval_ms=500)))
        deliveries=[]
        waits=[]
        active=threading.local()
        intervals=[]
        interval_lock=threading.Lock()
        original_execute=Tools.execute

        def execute(worker,action):
            started=time.perf_counter()
            previous=getattr(active,'depth',0)
            active.depth=previous+1
            try:
                return original_execute(worker,action)
            finally:
                active.depth=previous
                if action['tool']=='workspace_read':
                    with interval_lock:
                        intervals.append((started,time.perf_counter(),threading.get_ident()))

        def desktop(worker,name,arguments):
            deliveries.append((name,arguments))
            return ToolResult('delivered to fixture')

        def virtual_wait(signal,seconds):
            waits.append(seconds)
            return signal.is_set()

        def runner_at(name):
            runner=ToolRunner(output/name,threading.Event(),lambda *args:True,lambda *args:None)
            runner.window={'handle':1,'pid':2}
            runner.workspace_root=str(workspace)
            runner.tool_policies={'workspace_read':'allow'}
            runner.call_id=name
            return runner

        with ExitStack() as stack:
            stack.enter_context(patch('desktop_agent.tools.windows.window_title',return_value='Fixture'))
            stack.enter_context(patch.object(Tools,'desktop',desktop))
            stack.enter_context(patch.object(StopSignal,'wait',virtual_wait))
            stack.enter_context(patch.object(Tools,'execute',execute))
            sequential=runner_at('sequential')
            grouped=runner_at('macro')
            stack.callback(sequential.close)
            stack.callback(grouped.close)

            def run_sequential():
                for repetition in range(100):
                    if repetition:
                        waits.append(0.5)
                    for child in children:
                        result=sequential.execute(child)
                        assert not result.error and not result.interrupted

            _,sequential_seconds=measured(run_sequential)
            expected=list(deliveries)
            assert len(expected)==300 and len(waits)==99 and sum(waits)==49.5
            deliveries.clear();waits.clear()
            result,macro_seconds=measured(lambda:grouped.execute(macro))
            assert not result.error and not result.interrupted
            assert deliveries==expected and len(waits)==99 and sum(waits)==49.5
            summary=json.loads(result.text)
            assert summary['completed_iterations']==100 and summary['completed_actions']==300
            report['macro']=dict(sequential_seconds=sequential_seconds,macro_seconds=macro_seconds,
                                 dispatch_speedup=sequential_seconds/macro_seconds,deliveries=300,
                                 planned_wait_seconds=49.5,waits_actually_elapsed=False,
                                 model_requests_executed=0,desktop_input_mocked=True,
                                 macro_summary=summary)

            reads=[normalize_call(dict(tool='workspace_read',arguments={'path':f'fixture-{index}.txt','start_line':1,'end_line':20})) for index in range(4)]
            batch=normalize_call(dict(tool='tool_parallel',arguments={'actions':reads}))
            intervals.clear()
            serial_results,serial_seconds=measured(lambda:[sequential.execute(action) for action in reads])
            assert all(not result.error for result in serial_results)
            intervals.clear()
            batch_result,batch_seconds=measured(lambda:grouped.execute(batch))
            assert not batch_result.error,batch_result.text
            records=json.loads(batch_result.text)['results']
            assert [record['text'] for record in records]==[result.text for result in serial_results]
            marks=sorted([(start,1) for start,_,_ in intervals]+[(end,-1) for _,end,_ in intervals])
            concurrency=peak=0
            for _,change in marks:
                concurrency+=change
                peak=max(peak,concurrency)
            report['parallel_reads']=dict(sequential_seconds=serial_seconds,parallel_seconds=batch_seconds,
                                          speedup=serial_seconds/batch_seconds,peak_in_flight=peak,
                                          files=4,total_bytes=sum(path.stat().st_size for path in workspace.iterdir()),
                                          result_hashes=[digest(json.loads(record['text'])) for record in records],
                                          same_inputs_and_outputs=True,order='sequential_then_parallel',
                                          filesystem_cache_not_controlled=True)
    return report


def cache_review(database):
    connection=sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)
    rows=connection.execute("SELECT id,session_id,role,metadata FROM events WHERE role IN ('assistant','system') ORDER BY id").fetchall()
    connection.close()
    sessions=defaultdict(list)
    for identifier,session,role,raw in rows:
        metadata=json.loads(raw)
        metrics=metadata.get('metrics',{})
        timings=metrics.get('timings',{})
        if not timings:
            continue
        cache=metrics.get('prompt_cache',{})
        sessions[session].append(dict(event_id=identifier,role=role,status=metadata.get('status'),
                                     prompt_tokens=metrics.get('usage',{}).get('prompt_tokens'),
                                     cache_n=timings.get('cache_n'),prompt_n=timings.get('prompt_n'),
                                     prompt_seconds=timings.get('prompt_ms',0)/1000,
                                     decode_seconds=timings.get('predicted_ms',0)/1000,
                                     image_count=metrics.get('image_count'),
                                     system_hash=cache.get('system_text_sha256')))
    result=[]
    for identifier,records in sessions.items():
        if not identifier.startswith(('e4070416','2952e54e')):
            continue
        previous=None
        for record in records:
            if previous:
                record['previous_event_id']=previous['event_id']
                record['same_system_hash']=record['system_hash']==previous['system_hash'] if record['system_hash'] and previous['system_hash'] else None
                record['after_compaction']=previous['status']=='compaction'
            previous=record
        result.append(dict(session=identifier[:8],requests=records))
    return dict(sessions=result,privacy='No conversation content copied',historical_payloads_reconstructed=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('check','run'))
    parser.add_argument('--output',type=Path,default=Path('desktop_agent/data/maintenance/automation-cache-20260925-v1'))
    args=parser.parse_args()
    if args.mode=='check':
        print(json.dumps(check_prefix()))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',runs=1,model_runs=0,settings_changed=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination=args.output/'report.json'
    try:
        report['tools']=tool_measurements(args.output)
        report['prefix_check']=check_prefix()
        report['cache']=cache_review('desktop_agent/data/sessions.sqlite3')
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],tools=report['tools'],report=str(destination)),ensure_ascii=True))


if __name__=='__main__':
    main()