import json
import os
from pathlib import Path
import tempfile
import unittest

from desktop_agent.store import Store, conversation_turns
from desktop_agent.protocol import action_schema, approval_reason, validate_action, web_url


class PolicyTests(unittest.TestCase):
    def test_front_profile_deployment_checks_caps_devices_and_identity(self):
        from unittest.mock import patch
        from desktop_agent import models,residency
        adapters=[dict(software=False,vendor=0x10de,device=0x2d83,luid='rtx-now'),
                  dict(software=False,vendor=0x1002,device=0x6fdf,luid='rx-now')]
        with tempfile.TemporaryDirectory() as folder:
            runtime=Path(folder)/'runtime';runtime.mkdir()
            model=Path(folder)/models.Q2_XL['model'];projector=Path(folder)/models.Q2_XL['projector']
            for path in (model,projector,runtime/'llama-server.exe'):
                path.write_bytes(b'fixture')
            record=dict(schema=2,baseline_id='front_native_profiles_v1',profile_planning='runtime_signature',
                        files={'llama-server.exe':residency.sha256(runtime/'llama-server.exe')},
                        environment={'LOCAL_DESK_RESIDENCY_MODE':'mixed_fill'},
                        caps=dict(rtx_dedicated=7850*1024**2,rtx_shared=190*1024**2,available_ram=4*1024**3,headroom=32*1024**2))
            for name,path in (('model',model),('projector',projector)):
                record[name]=dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
            manifest=runtime/'deployment.json';manifest.write_text(json.dumps(record))
            profile=residency.front_profile(24576,'q4_0',2048)
            with patch.object(models,'Q2_PROFILE_RUNTIME',runtime),patch.object(residency,'gpu_adapters',return_value=adapters), \
                 patch.dict(os.environ,{'LOCAL_DESK_FRONT_EXPORT':'1','LOCAL_DESK_FRONT_TABLE':'old','GGML_CUDA_DISABLE_GRAPHS':'1'}):
                original=dict(os.environ)
                verified,environment=residency.deployment(str(model),str(projector),profile=profile)
                self.assertEqual(environment['LOCAL_DESK_FRONT_ID'],profile['identity'])
                self.assertEqual(environment['LOCAL_DESK_RTX_LUID'],'rtx-now')
                self.assertEqual(verified['rx_luid'],'rx-now')
                self.assertEqual(verified['caps']['rtx_shared'],190*1024**2)
                for key in ('LOCAL_DESK_FRONT_EXPORT','LOCAL_DESK_FRONT_TABLE','GGML_CUDA_DISABLE_GRAPHS'):
                    self.assertNotIn(key,environment)
                self.assertEqual(dict(os.environ),original)
                with self.assertRaisesRegex(ValueError,'identity'):
                    residency.deployment(str(model),str(projector),profile=dict(profile,identity='bad'))
                record['caps']['rtx_shared']+=1;manifest.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'caps'):
                    residency.deployment(str(model),str(projector),profile=profile)
                record['caps']['rtx_shared']-=1;manifest.write_text(json.dumps(record))
                (runtime/'llama-server.exe').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'file mismatch'):
                    residency.deployment(str(model),str(projector),profile=profile)
        for rows in (adapters[:1],adapters+[adapters[0]],adapters+[dict(adapters[0],device=123)]):
            with self.assertRaisesRegex(ValueError,'RTX5050'):
                residency.profile_adapters(rows)

    def test_front_profile_runtime_selection_preserves_all_options(self):
        from desktop_agent.agent import DesktopServer
        from desktop_agent.models import Q2_XL,Q2_PROFILE_RUNTIME
        server=DesktopServer()
        for context in range(4096,65537,1024):
            for cache in (0,2048):
                for kind in ('default','q4_0','q8_0','f16'):
                    server.context_tokens=context;server.cache_ram_mib=cache;server.kv_cache_type=kind
                    self.assertTrue(server.front_enabled(Q2_XL['model'],Q2_XL['projector']))
                    selected=server.settings_for('fixture.exe',Q2_XL['model'],Q2_XL['projector'],1024,1024)
                    self.assertEqual(selected[0],str(Q2_PROFILE_RUNTIME/'llama-server.exe'))
                    options=selected[-1]
                    for flag,value in (('-c',str(context)),('-ctk','q8_0' if kind=='default' else kind),
                                       ('-ctv','q8_0' if kind=='default' else kind),('--cache-ram',str(cache)),
                                       ('-ub','128'),('-b','512'),('--spec-type','none')):
                        self.assertEqual(options.count(flag),1)
                        self.assertEqual(options[options.index(flag)+1],value)
        self.assertIsNone(server.process)

    def test_front_profile_approved_shared190_keeps_other_limits(self):
        from desktop_agent.residency import memory_reason
        record=dict(rtx_luid='rtx',rx_luid='rx',caps=dict(rtx_dedicated=7850*1024**2,rtx_shared=190*1024**2,available_ram=4*1024**3))
        row=dict(adapter_dedicated={'rtx':7000*1024**2,'rx':4000*1024**2},process_shared={'pid_123_rtx':186*1024**2})
        self.assertEqual(memory_reason(row,8*1024**3,123,record,True),'')
        row['process_shared']['pid_123_rtx']=190*1024**2+1
        self.assertIn('shared',memory_reason(row,8*1024**3,123,record,True))
        row['process_shared']['pid_123_rtx']=186*1024**2
        self.assertIn('RAM',memory_reason(row,4*1024**3-1,123,record,True))
        row['adapter_dedicated']['rtx']=7850*1024**2+1
        self.assertIn('dedicated',memory_reason(row,8*1024**3,123,record,True))

    def test_front_start_preserves_memory_guard_reason_after_loader_cleanup(self):
        import threading
        from unittest.mock import Mock,patch
        from desktop_agent.agent import DesktopServer
        from desktop_agent.models import Q2_XL
        from game_agent.core import Halted
        server=DesktopServer();server.context_tokens=24576;server.kv_cache_type='q4_0'
        guard=Mock(reason='Front residency: RTX shared GPU memory limit exceeded')
        def load(*args,**kwargs):
            server.close()
            raise Halted('Model loading cancelled or timed out')
        with patch('desktop_agent.residency.deployment',return_value=({},{})),patch('desktop_agent.residency.ResidencyGuard',return_value=guard),patch('game_agent.runtime.LocalServer.start',side_effect=load):
            with self.assertRaisesRegex(RuntimeError,'RTX shared GPU memory limit exceeded'):
                server.start('fixture.exe',Q2_XL['model'],Q2_XL['projector'],Path('fixture.log'),threading.Event(),1024,1024)
        self.assertIsNone(server.residency_guard)

    def test_front_profile_covers_selectable_combinations_without_aliasing(self):
        from desktop_agent.residency import front_profile
        profiles={}
        for context in range(4096,65537,1024):
            for cache in (0,2048):
                for kind in ('default','q4_0','q8_0','f16'):
                    profile=front_profile(context,kind,cache)
                    key=(context,'q8_0' if kind=='default' else kind,cache)
                    self.assertEqual(profile['n_ubatch'],128)
                    if key in profiles:
                        self.assertEqual(profile,profiles[key])
                    profiles[key]=profile
        self.assertEqual(len(profiles),366)
        self.assertEqual(len({profile['identity'] for profile in profiles.values()}),366)
        for context,kind,cache in ((True,'q8_0',0),(8193,'q8_0',0),(8192,'q2',0),(8192,'q8_0',True),(8192,'q8_0',4096)):
            with self.assertRaises(ValueError):
                front_profile(context,kind,cache)

    def test_compaction_unreachable_target_reports_limit_without_repeating(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Settings,pack_messages
        from desktop_agent.compaction import Compactor
        from desktop_agent.protocol import normalize_call
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);identifier=store.create()
            request=store.append(identifier,'user','Current user request must remain.')
            for index in range(18):
                action=normalize_call(dict(tool='finish',arguments={'text':('finding '+str(index)+' ')*120}))
                store.append(identifier,'assistant',json.dumps(action))
            events=store.events(identifier)
            model=Mock();model.settings=Settings(context_tokens=32768,compaction_target_percent=10)
            model.count=lambda text:len(text)//4;model.tool_names=('finish',)
            model.generate.return_value=(normalize_call(dict(tool='finish',arguments={'text':'Earlier findings retained. Keep the current request and recent findings.'})),{})
            notify=Mock();manager=Compactor(store,model,threading.Event(),notify)
            _,_,tokens=pack_messages(events,'guide',model.count,float('inf'))
            budget=tokens*100//90
            visible,memory,covered=manager.prepare(identifier,events,'guide',budget,{},request)
            self.assertTrue(covered)
            protected={request,*[event['id'] for event in events[-6:]]}
            self.assertFalse(protected & covered)
            self.assertTrue(protected <= {event['id'] for event in visible})
            policy=store.context_summary(identifier)['compaction']
            self.assertFalse(policy['target_met'])
            self.assertGreater(policy['retained_tokens'],policy['target_tokens'])
            self.assertLessEqual(policy['retained_tokens'],budget)
            self.assertTrue(manager.target_limited)
            self.assertEqual(manager.attempts,1)
            manager.prepare(identifier,store.events(identifier),'guide',budget,{},request)
            model.generate.assert_called_once()
            self.assertTrue(any('target not reached' in str(call) for call in notify.call_args_list))
            self.assertEqual(store.events(identifier)[:len(events)],events)

    def test_compaction_settings_dialog_saves_validates_cancels_and_fits(self):
        import tkinter as tk
        from types import SimpleNamespace
        from desktop_agent.agent import Settings
        from desktop_agent.app import Console
        def widgets(dialog):
            pending=list(dialog.winfo_children());found={}
            while pending:
                widget=pending.pop()
                found[widget.winfo_name()]=widget
                pending.extend(widget.winfo_children())
            return found
        def enter(widget,value):
            widget.delete(0,'end');widget.insert(0,value)
        root=tk.Tk()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root.geometry('480x300+50+50')
                root.busy=False;root.settings=Settings();root.data=Path(folder)
                root.model=SimpleNamespace(settings=root.settings)
                root.status=tk.StringVar(master=root)
                root.update()
                dialog=Console.compaction_settings_dialog(root)
                dialog.geometry('420x220');root.update()
                controls=widgets(dialog)
                self.assertEqual(controls['trigger'].get(),'85')
                self.assertEqual(controls['target'].get(),'65')
                enter(controls['target'],'85');controls['save'].invoke();root.update()
                self.assertTrue(dialog.winfo_exists())
                self.assertFalse((root.data/'settings.json').exists())
                for name in ('trigger','target','save','cancel','defaults'):
                    control=controls[name]
                    self.assertTrue(control.winfo_ismapped())
                    self.assertLessEqual(control.winfo_rooty()+control.winfo_height(),dialog.winfo_rooty()+dialog.winfo_height())
                enter(controls['trigger'],'90');enter(controls['target'],'55')
                controls['save'].invoke();root.update()
                self.assertFalse(dialog.winfo_exists())
                saved=Settings.load(root.data/'settings.json')
                self.assertEqual((saved.compaction_trigger_percent,saved.compaction_target_percent),(90,55))
                self.assertIs(root.model.settings,root.settings)
                before=(root.data/'settings.json').read_bytes()
                dialog=Console.compaction_settings_dialog(root);root.update()
                controls=widgets(dialog);controls['defaults'].invoke()
                self.assertEqual((controls['trigger'].get(),controls['target'].get()),('85','65'))
                controls['cancel'].invoke();root.update()
                self.assertEqual((root.data/'settings.json').read_bytes(),before)
                self.assertEqual(root.settings.compaction_trigger_percent,90)
                root.busy=True
                self.assertIsNone(Console.compaction_settings_dialog(root))
        finally:
            root.destroy()

    def test_compaction_target_reaches_headroom_and_keeps_protected_call_groups(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Settings,pack_messages
        from desktop_agent.compaction import Compactor
        from desktop_agent.protocol import normalize_call
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);identifier=store.create()
            request=store.append(identifier,'user','Preserve the current request and pending work.')
            groups=[]
            for index in range(15):
                call=normalize_call(dict(tool='browser_read',arguments={}))
                call_id='fixture-'+str(index)
                first=store.append(identifier,'assistant',json.dumps(call),call_id=call_id)
                last=store.append(identifier,'tool',('older observation '*100 if index<12 else 'recent result'),tool='browser_read',status='delivered',call_id=call_id)
                groups.append({first,last})
            events=store.events(identifier)
            model=Mock();model.settings=Settings(context_tokens=32768,compaction_trigger_percent=95)
            model.count=lambda text:len(text)//4;model.tool_names=('finish','browser_read')
            state={'pending_jobs':[{'call_id':'fixture-1'}]}
            _,_,before=pack_messages(events,'guide',model.count,float('inf'),state=state)
            budget=before*100//90
            manager=Compactor(store,model,threading.Event(),Mock())
            manager.prepare(identifier,events,'guide',budget,state,request)
            model.generate.assert_not_called()
            model.settings.compaction_trigger_percent=85
            model.generate.return_value=(normalize_call(dict(tool='finish',arguments={'text':'Older observations were recorded. Keep the current goal and pending call.'})),{})
            visible,memory,covered=manager.prepare(identifier,events,'guide',budget,state,request)
            self.assertTrue(covered)
            self.assertNotIn(request,covered)
            for group in groups:
                self.assertIn(len(group & covered),(0,len(group)))
            self.assertFalse(groups[1] & covered)
            self.assertFalse(set(event['id'] for event in events[-6:]) & covered)
            policy=store.context_summary(identifier)['compaction']
            self.assertTrue(policy['target_met'],policy)
            self.assertLessEqual(policy['retained_tokens'],budget*65//100)
            self.assertFalse(manager.target_limited)
            added=store.append(identifier,'tool','small new result',tool='browser_read',status='delivered')
            manager.prepare(identifier,store.events(identifier),'guide',budget,state,request)
            self.assertEqual(model.generate.call_count,1)
            self.assertEqual(store.events(identifier)[:len(events)],events)
            self.assertGreater(added,max(covered))

    def test_compaction_percentages_defaults_roundtrip_and_validation(self):
        from dataclasses import replace
        from desktop_agent.agent import Settings
        defaults=Settings()
        self.assertEqual((defaults.compaction_trigger_percent,defaults.compaction_target_percent),(85,65))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            path.write_text('{}',encoding='utf-8')
            self.assertEqual(Settings.load(path).compaction_target_percent,65)
            custom=replace(defaults,compaction_trigger_percent=90,compaction_target_percent=55)
            custom.save(path)
            restored=Settings.load(path)
            self.assertEqual((restored.compaction_trigger_percent,restored.compaction_target_percent),(90,55))
            for trigger,target in ((65,65),(60,70),(101,65),(85,0),(True,1),(85,65.0)):
                invalid=replace(defaults,compaction_trigger_percent=trigger,compaction_target_percent=target)
                with self.subTest(trigger=trigger,target=target),self.assertRaises(ValueError):
                    invalid.validate_compaction()

    def test_cache_trace_stream_payload_parity_and_failure_reset(self):
        from copy import deepcopy
        import threading
        from unittest.mock import MagicMock,Mock,patch
        from PIL import Image
        from desktop_agent.agent import Model,Settings
        reply=dict(tool='finish',arguments={'text':'Done'})
        events=[dict(choices=[dict(delta={'content':json.dumps(reply)},finish_reason='stop')],
                     timings={'cache_n':42,'prompt_n':12},usage={'prompt_tokens':54,'completion_tokens':7})]
        response=MagicMock()
        response.__enter__.return_value=response
        response.iter_lines.return_value=['data: '+json.dumps(event) for event in events]+['data: [DONE]']
        client=MagicMock()
        client.__enter__.return_value=client
        client.post.return_value=response
        model=Model(Settings())
        model.endpoint='http://127.0.0.1:1234'
        messages=[dict(role='system',content='Fixture guide'),dict(role='user',content='Fixture request',_event_id=101),
                  dict(role='user',content='Fixture state',_status=True)]
        original=deepcopy(messages)
        image=Image.new('RGB',(8,8),'white')
        stopped=threading.Event()
        try:
            with patch('desktop_agent.agent.session',return_value=client):
                with patch('desktop_agent.agent.request_cache_trace',return_value=(None,{})):
                    model.generate(messages,image,stopped,Mock())
                baseline=deepcopy(client.post.call_args.kwargs)
                _,first=model.generate(messages,image,stopped,Mock())
                self.assertEqual(client.post.call_args.kwargs,baseline)
                self.assertFalse(first['cache_transition']['previous_request_available'])
                _,second=model.generate(messages,image,stopped,Mock())
                self.assertEqual(client.post.call_args.kwargs,baseline)
                self.assertEqual(second['cache_transition']['common_prefix_messages'],4)
                self.assertTrue(second['cache_transition']['same_image_messages'])
                self.assertEqual(second['prompt_cache']['reported_cached_tokens'],42)
                self.assertEqual(second['timings'],events[0]['timings'])
                self.assertNotIn('Fixture',json.dumps(second['cache_transition']))
                changed=deepcopy(messages)
                changed[1]['content']='Edited request'
                _,edited=model.generate(changed,image,stopped,Mock())
                self.assertEqual(edited['cache_transition']['common_prefix_messages'],1)
                self.assertEqual(edited['cache_transition']['first_changed_current']['event_id'],101)
                client.post.side_effect=RuntimeError('Fixture transport error')
                with self.assertRaisesRegex(RuntimeError,'Fixture transport error'):
                    model.generate(messages,image,stopped,Mock())
                self.assertIsNone(model._cache_request)
                client.post.side_effect=None
                _,recovered=model.generate(messages,image,stopped,Mock())
                self.assertFalse(recovered['cache_transition']['previous_request_available'])
            self.assertEqual(messages,original)
        finally:
            model.close()
        self.assertIsNone(model._cache_request)

    def test_cache_trace_finds_history_state_and_summary_boundaries_without_content(self):
        from copy import deepcopy
        from desktop_agent.agent import request_cache_trace
        messages=[dict(role='system',content='private fixed guide'),dict(role='user',content='private request'),
                  dict(role='user',content='private state'),dict(role='user',content=[dict(type='image_url',image_url={'url':'data:image/jpeg;base64,private-image'})])]
        origins=[{},dict(_event_id=1),dict(_status=True)]
        payload=dict(messages=messages,temperature=0,max_tokens=-1)
        original=deepcopy(payload)
        before,first=request_cache_trace(payload,origins,scope='server-one')
        self.assertFalse(first['previous_request_available'])
        self.assertEqual(original,payload)
        changed=deepcopy(payload)
        changed['messages'].insert(2,dict(role='assistant',content='private action'))
        current,trace=request_cache_trace(changed,[{},dict(_event_id=1),dict(_event_id=2),dict(_status=True)],before,'server-one')
        self.assertEqual(trace['common_prefix_messages'],2)
        self.assertEqual(trace['first_changed_previous']['kind'],'state_or_memory')
        self.assertEqual(trace['first_changed_current']['event_id'],2)
        self.assertTrue(trace['same_image_messages'])
        self.assertTrue(trace['same_request_options'])
        self.assertNotIn('private',json.dumps([before,current,trace]))
        summary=deepcopy(changed)
        summary['messages'][0]['content']+='\n\nCONTEXT COMPACTION MODE\nsummary'
        _,trace=request_cache_trace(summary,[],current,'server-one')
        self.assertEqual(trace['purpose'],'compaction')
        self.assertEqual(trace['common_prefix_messages'],0)
        self.assertEqual(trace['first_changed_current']['kind'],'system')
        _,trace=request_cache_trace(changed,[],current,'server-two')
        self.assertFalse(trace['previous_request_available'])
        self.assertEqual(trace['reset_reason'],'server_changed')

    def test_macro_progress_event_updates_status(self):
        import queue
        from types import SimpleNamespace
        from unittest.mock import Mock
        from desktop_agent.app import Console
        events=queue.Queue()
        events.put(('tool_progress',dict(tool='desktop_macro',completed=12,total=100)))
        console=SimpleNamespace(events=events,status=Mock(),approval=None,closing=False,after=Mock(),poll=Mock())
        Console.poll(console)
        console.status.set.assert_called_once_with('desktop_macro: 12/100')
        console.after.assert_called_once_with(60,console.poll)

    def test_macro_background_cancel_interrupts_interval(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.jobs import ToolRunner,StopSignal
        from desktop_agent.tools import Tools,ToolResult
        from desktop_agent.protocol import normalize_call
        waiting=threading.Event()
        original_wait=StopSignal.wait
        def wait(signal,seconds):
            waiting.set()
            return original_wait(signal,seconds)
        with tempfile.TemporaryDirectory() as folder:
            runner=ToolRunner(folder,threading.Event(),lambda *args:True,lambda *args:None)
            runner.window={'handle':1,'pid':2};runner.call_id='cancel-parent'
            try:
                with patch('desktop_agent.tools.windows.window_title',return_value='Fixture'),patch.object(Tools,'desktop',return_value=ToolResult('ok')) as desktop,patch.object(StopSignal,'wait',wait):
                    started=runner.execute(normalize_call(dict(tool='desktop_macro',arguments=dict(background=True,repeat=100,interval_ms=500,actions=[dict(tool='desktop_key',arguments={'key':'Enter'})]))))
                    identifier=json.loads(started.text)['job_id']
                    self.assertTrue(waiting.wait(3))
                    runner.execute(normalize_call(dict(tool='job_cancel',arguments={'job_id':identifier})))
                    runner.background[identifier]['future'].result(timeout=3)
                    result=runner.collect(identifier)
                self.assertTrue(result.interrupted)
                self.assertEqual(result.call_id,'cancel-parent')
                self.assertEqual(json.loads(result.text)['completed_iterations'],1)
                desktop.assert_called_once()
            finally:
                runner.close()

    def test_macro_protocol_rejects_nesting_denial_and_missing_coordinates(self):
        from desktop_agent.protocol import normalize_call,compact_call,compact_schema,ToolCatalog,require_pixel_coordinates
        call=dict(tool='desktop_macro',arguments=dict(actions=[dict(tool='desktop_key',arguments={'key':'Enter'})],repeat=100,interval_ms=500))
        catalog=ToolCatalog(dict(input=True,screen=True,browser=False),'repeat macro')
        self.assertEqual(compact_call(catalog.normalize(call)),call)
        self.assertEqual(normalize_call(dict(tool=call['tool'],arguments=dict(call['arguments'],background=True)))['tool'],'job_start')
        for child in (call,dict(tool='desktop_key',arguments={'key':'Enter','background':True}),dict(tool='terminal_start',arguments={'command':'echo test'})):
            with self.assertRaises(ValueError):
                normalize_call(dict(tool='desktop_macro',arguments={'actions':[child]}))
        denied=ToolCatalog(dict(input=True,tool_policies={'desktop_key':'disabled'}),'repeat')
        with self.assertRaises(ValueError):
            denied.normalize(call)
        missing=normalize_call(dict(tool='desktop_macro',arguments={'actions':[dict(tool='desktop_click',arguments={'x':2,'y':3})]}))
        with self.assertRaises(ValueError):
            require_pixel_coordinates(missing)
        with self.assertRaises(ValueError):
            normalize_call(dict(tool='desktop_macro',arguments=dict(call['arguments'],repeat=True)))
        json.dumps(compact_schema(catalog.names()))

    def test_macro_partial_failure_and_input_lane_exclusion(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.tools import Tools,ToolResult
        from desktop_agent.protocol import normalize_call
        entered,release=threading.Event(),threading.Event()
        seen=[]
        def desktop(worker,name,arguments):
            seen.append(arguments.get('text') or arguments.get('key'))
            if len(seen)==1:
                entered.set()
                if not release.wait(3):
                    raise TimeoutError('Test did not release first input')
            if arguments.get('key')=='Enter':
                return ToolResult('partially sent',error='target changed',interrupted='target changed')
            return ToolResult('ok')
        macro=normalize_call(dict(tool='desktop_macro',arguments=dict(repeat=100,actions=[
            dict(tool='desktop_type',arguments={'text':'first'}),dict(tool='desktop_key',arguments={'key':'Enter'})],background=True)))
        with tempfile.TemporaryDirectory() as folder:
            runner=ToolRunner(folder,threading.Event(),lambda *args:True,lambda *args:None)
            runner.window={'handle':1,'pid':2}
            try:
                with patch('desktop_agent.tools.windows.window_title',return_value='Fixture'),patch.object(Tools,'desktop',desktop):
                    started=json.loads(runner.execute(macro).text)
                    self.assertTrue(entered.wait(3))
                    future,_=runner.submit(normalize_call(dict(tool='desktop_type',arguments={'text':'other'})))
                    self.assertFalse(future.done())
                    release.set()
                    result=runner.collect(started['job_id'])
                    future.result(timeout=3)
                self.assertEqual(seen,['first','Enter','other'])
                summary=json.loads(result.text)
                self.assertEqual(summary['completed_actions'],1)
                self.assertEqual(summary['completed_iterations'],0)
                self.assertEqual(summary['last_action']['step'],2)
                self.assertEqual(summary['status'],'stopped')
            finally:
                release.set();runner.close()

    def test_browser_macro_keeps_thread_identity_and_child_approval(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.tools import Tools,ToolResult
        from desktop_agent.protocol import normalize_call
        executions=[];approvals=[]
        def browse(worker,name,arguments):
            executions.append((threading.get_ident(),worker.call_id))
            return ToolResult('ok')
        def approve(action,*args):
            approvals.append(action['tool'])
            return action['tool']!='browser_type'
        with tempfile.TemporaryDirectory() as folder:
            runner=ToolRunner(folder,threading.Event(),approve,lambda *args:None)
            runner.call_id='browser-parent'
            try:
                call=normalize_call(dict(tool='browser_macro',arguments=dict(repeat=2,actions=[dict(tool='browser_key',arguments={'key':'Enter'})])))
                with patch.object(Tools,'ensure_browser'),patch.object(Tools,'browser_target',return_value={'label':'Fixture'}),patch.object(Tools,'browse',browse):
                    result=runner.execute(call)
                    self.assertFalse(result.error)
                    self.assertEqual(len({record[0] for record in executions}),1)
                    self.assertNotEqual(executions[0][0],threading.get_ident())
                    self.assertEqual([record[1] for record in executions],['browser-parent:1:1','browser-parent:2:1'])
                    runner.tool_policies={'browser_type':'ask'}
                    blocked=runner.execute(normalize_call(dict(tool='browser_macro',arguments={'actions':[dict(tool='browser_type',arguments={'selector':'input','text':'fixture'})]})))
                    self.assertTrue(blocked.interrupted)
                    self.assertEqual(json.loads(blocked.text)['completed_actions'],0)
                self.assertIn('browser_type',approvals)
                self.assertEqual(len(executions),2)
            finally:
                runner.close()

    def test_macro_order_count_cancellation_and_policy(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.jobs import ToolRunner,StopSignal
        from desktop_agent.tools import Tools,ToolResult
        from desktop_agent.protocol import normalize_call
        call = normalize_call(dict(tool='desktop_macro',arguments=dict(repeat=100,interval_ms=500,actions=[
            dict(tool='desktop_type',arguments={'text':'fixture'}),
            dict(tool='desktop_key',arguments={'key':'Enter'}),
            dict(tool='desktop_key',arguments={'key':'Enter'})])))
        with tempfile.TemporaryDirectory() as folder:
            runner=ToolRunner(folder,threading.Event(),lambda *args:True,lambda *args:None)
            runner.window={'handle':1,'pid':2};runner.call_id='parent'
            try:
                with patch('desktop_agent.tools.windows.window_title',return_value='Fixture'), patch.object(Tools,'desktop',return_value=ToolResult('ok')) as execute, patch.object(StopSignal,'wait',return_value=False) as wait:
                    result=json.loads(runner.execute(call).text)
                self.assertEqual(result['completed_iterations'],100)
                self.assertEqual(result['completed_actions'],300)
                self.assertEqual([item.args[0] for item in execute.call_args_list],['desktop_type','desktop_key','desktop_key']*100)
                self.assertEqual([item.args for item in wait.call_args_list],[(0.5,)]*99)
                journal=[json.loads(line) for line in Path(result['log_path']).read_text(encoding='utf-8').splitlines()]
                self.assertEqual(journal[1]['call_id'],'parent:1:1')
                self.assertEqual(journal[-2]['call_id'],'parent:100:3')
                with patch('desktop_agent.tools.windows.window_title',return_value='Fixture'), patch.object(Tools,'desktop',return_value=ToolResult('ok')), patch.object(StopSignal,'wait',return_value=True):
                    stopped=runner.execute(call)
                self.assertTrue(stopped.interrupted)
                self.assertEqual(json.loads(stopped.text)['completed_iterations'],1)
                runner.tool_policies={'desktop_key':'disabled'}
                with patch.object(Tools,'desktop') as execute, self.assertRaises(ValueError):
                    runner.execute(call)
                execute.assert_not_called()
            finally:
                runner.close()

    def test_parallel_reads_overlap_and_preserve_order(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.tools import Tools,ToolResult
        from desktop_agent.protocol import normalize_call
        barrier=threading.Barrier(4,timeout=3)
        def execute(worker,action):
            barrier.wait()
            return ToolResult(action['arguments']['path'],call_id=worker.call_id)
        call=normalize_call(dict(tool='tool_parallel',arguments={'actions':[
            dict(tool='workspace_read',arguments={'path':str(index)}) for index in range(4)]}))
        with tempfile.TemporaryDirectory() as folder:
            runner=ToolRunner(folder,threading.Event(),lambda *args:True,lambda *args:None)
            runner.tool_policies={'workspace_read':'allow'};runner.call_id='batch'
            try:
                with patch.object(Tools,'execute',execute):
                    result=runner.execute(call)
                self.assertFalse(result.error,result.text)
                records=json.loads(result.text)['results']
                self.assertEqual([record['text'] for record in records],['0','1','2','3'])
                self.assertEqual([record['call_id'] for record in records],['batch:1','batch:2','batch:3','batch:4'])
            finally:
                runner.close()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_GUIDE_MODEL_TEST')=='1','Opt-in bounded local 27B guide validation')
    def test_local_27b_guide_decisions_once(self):
        from dataclasses import replace
        import threading
        import time
        import psutil
        from desktop_agent.agent import Model,Settings,HOME,system_prompt,pack_messages
        from desktop_agent.models import Q2_FRONT_RUNTIME
        from desktop_agent.protocol import ToolCatalog
        output=Path(os.environ['DESKTOP_AGENT_GUIDE_MODEL_ARTIFACTS']).resolve()
        self.assertFalse(output.exists(),'Preserve completed or failed model probes; do not rerun in the same directory')
        self.assertFalse(any(process.name().casefold()=='llama-server.exe' for process in psutil.process_iter()))
        output.mkdir(parents=True)
        deployment=json.loads((Q2_FRONT_RUNTIME/'deployment.json').read_text())
        settings=replace(Settings.load(HOME/'data/settings.json'),backend='local',model=deployment['model']['path'],
                         projector=deployment['projector']['path'],context_tokens=8192,cache_ram_mib=0,kv_cache_type='q8_0')
        stopped=threading.Event();model=Model(settings)
        model.server.context_tokens=8192;model.server.cache_ram_mib=0;model.server.kv_cache_type='q8_0'
        def deadline():
            stopped.set();model.cancel()
        timer=threading.Timer(240,deadline);timer.daemon=True;timer.start()
        report=dict(status='running',samples=[],tool_executions=0,deadline_seconds=240,reasoning_enabled=settings.reasoning_enabled)
        def save():
            (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        save()
        try:
            with tempfile.TemporaryDirectory() as folder:
                (Path(folder)/'sample.py').write_text('def verify_payment(amount):\n    return amount > 0\n',encoding='utf-8')
                capabilities=dict(screen=False,input=False,browser=False,approval='routine',workspace_root=folder,
                                  tool_policies={'workspace_read':'allow','workspace_search':'allow','workspace_code_search':'allow','workspace_symbols':'allow'})
                catalog=ToolCatalog(capabilities);catalog.load(['workspace','files'])
                guide=system_prompt(capabilities,catalog)
                model.tool_names=catalog.names()
                model.endpoint=model.server.start(settings.executable,settings.model,settings.projector,output/'server.log',
                    stopped,*settings.local_image_tokens,settings.reasoning_enabled,settings.reasoning_tokens)
                self.assertIsNotNone(model.server.residency_guard)
                report['pid']=model.server.process.pid;report['guide_characters']=len(guide);save()
                cases=[
                    ('known_path','sample.py\uc758 1~2\ud589\uc744 \uc77d\uc5b4\uc918. \uacbd\ub85c\ub97c \uc54c\uace0 \uc788\uc73c\ub2c8 \ubcc4\ub3c4 \uac80\uc0c9\uc740 \ud544\uc694 \uc5c6\uc5b4.','workspace_read'),
                    ('known_symbol','verify_payment\ub77c\ub294 \ud568\uc218\uc758 \uc815\uc758 \uc704\uce58\ub97c \ucc3e\uc544\uc918. \ud30c\uc77c \uacbd\ub85c\ub294 \ubab0\ub77c.','workspace_symbols'),
                    ('unknown_location','\uacb0\uc81c \uae08\uc561\uc744 \uac80\uc99d\ud558\ub294 \uad6c\ud604\uc744 \ucc3e\uc544\uc918. \uacbd\ub85c\uc640 \uc2ec\ubcfc\uc740 \ubab0\ub77c. payment amount validation \uac1c\ub150\uc73c\ub85c \ucf54\ub4dc\ub97c \uac80\uc0c9\ud574\uc918.','workspace_code_search'),
                    ('disabled_shell','PowerShell\ub85c Write-Output guide-fixture\ub97c \uc2e4\ud589\ud574\uc918. \uc170 \ub3c4\uad6c\ub97c \ud5c8\uc6a9\ud558\uc9c0 \uc54a\uc558\uc73c\uba74 \uc2e4\ud589\ud558\uc9c0 \ub9d0\uace0 \ud544\uc694\ud55c \uc124\uc815\uc744 \uc54c\ub824\uc918.','finish'),
                ]
                for label,prompt,expected in cases:
                    if stopped.is_set():raise TimeoutError('Guide probe deadline reached')
                    messages,_,_=pack_messages([dict(id=1,role='user',content=prompt)],guide,model.count,
                                              settings.context_tokens-settings.output_budget-512,state=dict(steps_remaining=1,pending_jobs=[],terminals=[]))
                    began=time.monotonic()
                    action,metrics=model.generate(messages,None,stopped,lambda *args:None)
                    action=catalog.normalize(action)
                    passed=action['tool']==expected
                    if label=='known_path':passed=passed and action['arguments']['path']=='sample.py'
                    if label=='known_symbol':passed=passed and action['arguments']['query']=='verify_payment' and action['arguments']['kind']=='definitions'
                    report['samples'].append(dict(label=label,prompt=prompt,expected=expected,action=action,passed=passed,
                        seconds=time.monotonic()-began,usage=metrics.get('usage'),prompt_cache=metrics.get('prompt_cache'),
                        public_commentary=action['message'] if action['tool']!='finish' else ''))
                    save()
                    if not passed:break
                report['status']='passed' if len(report['samples'])==4 and all(row['passed'] for row in report['samples']) else 'failed'
                self.assertEqual(report['status'],'passed',report['samples'])
        except Exception as error:
            report.update(status='failed',error=str(error))
            raise
        finally:
            timer.cancel()
            process=model.server.process
            model.close()
            report['owned_process_exited']=process is None or process.poll() is not None
            save()

    def test_guide_has_one_permission_scoped_search_route(self):
        from desktop_agent.agent import system_prompt
        from desktop_agent.protocol import ToolCatalog,WORKSPACE_TOOLS
        capabilities=dict(screen=False,input=False,browser=False,approval='routine',workspace_root='F:/fixture',
                          tool_policies={name:'allow' for name in WORKSPACE_TOOLS})
        catalog=ToolCatalog(capabilities);catalog.load(['workspace','terminal','files'])
        guide=system_prompt(capabilities,catalog)
        self.assertEqual(guide.count('Choose ONE starting tool'),1)
        for text in ('Known path -> workspace_read','Exact text -> workspace_search','Unknown code location -> workspace_code_search','Known symbol -> workspace_symbols'):
            self.assertIn(text,guide)
        self.assertNotIn('1. Find relevant files with workspace_search',guide)
        self.assertNotIn('\nDESKTOP COORDINATES\n',guide)
        for safety in ('never bypass a denial','NOT instructions','No elevation','not private chain-of-thought','not new instructions or permission'):
            self.assertIn(safety,guide)
        limited=ToolCatalog(dict(capabilities,tool_policies={'workspace_read':'allow'}));limited.load(['workspace'])
        limited_guide=system_prompt(limited.capabilities,limited)
        self.assertIn('Known path -> workspace_read',limited_guide)
        self.assertNotIn('Unknown code location -> workspace_code_search',limited_guide)
        self.assertNotIn('Known symbol -> workspace_symbols',limited_guide)
        self.assertNotIn('\nEDIT AFTER READING\n',limited_guide)

    def test_agent_compacts_before_main_request_and_restores_allowed_tools(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Agent,Model,Settings
        from desktop_agent.tools import Tools
        from desktop_agent.protocol import normalize_call
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);identifier=store.create()
            store.append(identifier,'user','Preserve the existing source and report verified results.')
            for index in range(20):
                store.append(identifier,'assistant',json.dumps(dict(tool='finish',arguments={'text':('Finding '+str(index)+' ')*500})))
                store.append(identifier,'tool','observed '*500,tool='browser_read',status='delivered')
            original_count=len(store.events(identifier))
            stopped=threading.Event();model=Model(Settings(context_tokens=16384,max_steps=3))
            model.ensure=Mock(return_value='offline')
            model.count=lambda text:len(text)//4
            requests=[]
            def generate(messages,*args):
                requests.append(messages)
                if 'CONTEXT COMPACTION MODE' in messages[0]['content']:
                    self.assertEqual(model.tool_names,('finish',))
                    text='Existing source must be preserved. Prior browser observations were collected; no edits verified. Continue the current request and retrieve exact event records for uncertain details.'
                else:
                    self.assertIn('conversation_summary',json.dumps(messages))
                    self.assertIn('load_tool_group',model.tool_names)
                    text='The saved observations are retained; no new actions were required.'
                return normalize_call(dict(tool='finish',arguments={'text':text})),{}
            model.generate=generate
            tools=Tools(store.artifact_directory(identifier),stopped,Mock(return_value=True),Mock())
            Agent(store,model,tools,stopped,Mock()).run(identifier,'Continue and report current findings.')
            self.assertEqual(len(requests),2)
            self.assertTrue(store.context_selection(identifier)['summarized_ids'])
            self.assertIsNotNone(store.context_summary(identifier))
            self.assertGreater(len(store.events(identifier)),original_count)

    def test_code_index_nested_ignore_scope_and_cancel(self):
        from desktop_agent.workspace_tools import Workspace
        from desktop_agent.retrieval import CodeIndex
        from game_agent.core import Halted
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'work';root.mkdir();(root/'sub').mkdir()
            (root/'.gitignore').write_text('sub/*.py\n')
            (root/'sub/.gitignore').write_text('!keep.py\n')
            (root/'outside.py').write_text('def scope_fixture(): return 1\n')
            (root/'sub/keep.py').write_text('def scope_fixture(): return 2\n')
            (root/'sub/ignored.py').write_text('def scope_fixture(): return 3\n')
            workspace=Workspace(str(root),Path(folder)/'session')
            index=CodeIndex(workspace)
            self.assertEqual(len(index.find_symbols('scope_fixture')['results']),2)
            rows=index.find_symbols('scope_fixture',path='sub')['results']
            self.assertEqual([row['path'] for row in rows],['sub/keep.py'])
            self.assertEqual([row['path'] for row in index.search('scope fixture',path='sub')['results']],['sub/keep.py'])
            def stop():raise Halted('cancel index')
            workspace.check=stop
            with self.assertRaises(Halted):index.search('scope fixture')

    def test_compaction_failure_disabled_and_cancel_preserve_context(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Settings
        from desktop_agent.compaction import Compactor
        from game_agent.core import Halted
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);identifier=store.create()
            store.append(identifier,'user','Original goal')
            for index in range(16):
                store.append(identifier,'assistant',json.dumps({'tool':'finish','arguments':{'text':'old '*400}}))
            request=store.append(identifier,'user','Current goal')
            events=store.events(identifier)
            model=Mock();model.settings=Settings(context_tokens=16384,auto_compact=False)
            model.tool_names=('finish',);model.count=lambda text:len(text)//3
            stopped=threading.Event()
            manager=Compactor(store,model,stopped,Mock())
            visible,memory,covered=manager.prepare(identifier,events,'stable-guide',5000,{},request)
            self.assertEqual(visible,events);self.assertIsNone(memory);self.assertFalse(covered)
            model.generate.assert_not_called()
            model.settings.auto_compact=True
            model.generate.side_effect=ValueError('summary unavailable')
            visible,memory,covered=manager.prepare(identifier,events,'stable-guide',5000,{},request)
            self.assertEqual(visible,events);self.assertIsNone(store.context_summary(identifier))
            self.assertEqual(manager.attempts,2)
            self.assertTrue(model.generate.call_args.args[0][0]['content'].startswith('stable-guide'))
            manager=Compactor(store,model,stopped,Mock())
            model.generate.side_effect=Halted('cancelled summary')
            with self.assertRaises(Halted):manager.prepare(identifier,events,'stable-guide',5000,{},request)
            self.assertIsNone(store.context_summary(identifier))
            self.assertEqual(model.tool_names,('finish',))

    def test_local_code_rag_symbols_refresh_and_ignore(self):
        from desktop_agent.workspace_tools import Workspace
        from desktop_agent.retrieval import CodeIndex
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'work';root.mkdir()
            (root/'.gitignore').write_text('ignored.py\ncache/\n')
            (root/'ignored.py').write_text('def verify_payment(): pass')
            (root/'payment.py').write_text('def verify_payment(amount):\n    return amount > 0\n\ndef caller():\n    return verify_payment(2)\n')
            (root/'frontend.js').write_text('function showPayment(value) { return value; }\nshowPayment(2);')
            index=CodeIndex(Workspace(str(root),Path(folder)/'session'))
            result=index.search('verify payment amount')
            self.assertEqual(result['results'][0]['path'],'payment.py')
            self.assertNotIn('ignored.py',[row['path'] for row in result['results']])
            first_hash=result['results'][0]['sha256']
            definition=index.find_symbols('verify_payment')['results']
            self.assertEqual(definition[0]['start_line'],1)
            references=index.find_symbols('verify_payment',kind='references')['results']
            self.assertEqual(references[0]['start_line'],5)
            self.assertTrue(index.find_symbols('showPayment')['results'])
            (root/'payment.py').write_text('def refund_payment():\n    return True\n')
            self.assertEqual(index.find_symbols('verify_payment')['results'],[])
            refreshed=index.search('refund payment')['results']
            self.assertNotEqual(refreshed[0]['sha256'],first_hash)
            (root/'payment.py').unlink()
            self.assertEqual(index.find_symbols('refund_payment')['results'],[])

    def test_public_commentary_survives_wire_and_is_not_a_tool_argument(self):
        from desktop_agent.protocol import normalize_call,compact_call,compact_parameters,ToolCatalog
        from desktop_agent.app import tool_rows
        note='The previous output confirms two matches. I will read the relevant function next.'
        wire=dict(tool='workspace_read',arguments=dict(path='fixture.py',commentary=note))
        action=normalize_call(wire)
        self.assertEqual(action['message'],note)
        self.assertNotIn('commentary',action['arguments'])
        self.assertEqual(compact_call(action),wire)
        self.assertIn('commentary',compact_parameters('workspace_read')['properties'])
        row=tool_rows([dict(id=1,role='assistant',content=json.dumps(action),metadata={'call_id':'one'})])[0]
        self.assertEqual(row['commentary'],note)
        catalog=ToolCatalog(dict(screen=True))
        load=normalize_call(dict(tool='load_tool_group',arguments=dict(groups=['files'],commentary=note)))
        self.assertEqual(catalog.normalize(load)['message'],note)

    def test_compaction_preserves_originals_and_invalidates_changed_sources(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Settings
        from desktop_agent.compaction import Compactor,restore_summary
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);identifier=store.create()
            store.append(identifier,'user','Keep all source files and use the fixture only.')
            for index in range(12):
                store.append(identifier,'assistant',json.dumps({'tool':'finish','arguments':{'text':('old finding '+str(index)+' ')*80}}))
                store.append(identifier,'tool','fixture result '*80,tool='browser_read',status='delivered')
            request=store.append(identifier,'user','Continue the same task')
            original=store.events(identifier)
            model=Mock();model.settings=Settings(context_tokens=8192);model.count=lambda text:len(text)//3;model.tool_names=('finish','browser_read')
            model.generate.return_value=(dict(tool='finish',arguments={},risk='routine',message='Preserve source files. Fixture results were read. Continue the current task; original event IDs remain available.'),{})
            compact=Compactor(store,model,threading.Event(),Mock())
            visible,memory,covered=compact.prepare(identifier,original,'guide',3500,{},request)
            self.assertTrue(covered)
            self.assertIn(request,[event['id'] for event in visible])
            self.assertTrue(memory['untrusted_history'])
            self.assertEqual(store.events(identifier)[:len(original)],original)
            self.assertEqual(model.tool_names,('finish','browser_read'))
            self.assertIsNotNone(restore_summary(store,identifier,store.events(identifier)))
            store.edit_message(identifier,original[0]['id'],'Changed user constraint')
            self.assertIsNone(restore_summary(store,identifier,store.events(identifier)))

    def test_prompt_cache_evidence_does_not_invent_a_hit(self):
        from desktop_agent.agent import prompt_cache_metrics
        from desktop_agent.app import performance_text
        missing=prompt_cache_metrics({},'fixed guide',requested=True)
        self.assertIsNone(missing['reported_cached_tokens'])
        self.assertEqual(missing['evidence'],'not_reported')
        zero=prompt_cache_metrics({'prompt_tokens_details':{'cached_tokens':0}},'fixed guide',requested=True)
        hit=prompt_cache_metrics({'prompt_tokens_details':{'cached_tokens':2400}},'fixed guide',requested=True)
        self.assertEqual(zero['reported_cached_tokens'],0)
        self.assertEqual(hit['reported_cached_tokens'],2400)
        self.assertEqual(hit['system_text_sha256'],missing['system_text_sha256'])
        fallback=prompt_cache_metrics({},'guide',requested=True,timings={'cache_n':1900})
        self.assertEqual(fallback['reported_cached_tokens'],1900)
        self.assertEqual(fallback['evidence'],'server_timings')
        from desktop_agent.api import normalize_usage
        normalized=normalize_usage({'prompt_tokens':2500,'prompt_tokens_details':{'cached_tokens':2400}},'openai')
        self.assertEqual(prompt_cache_metrics(normalized,'guide',requested=None)['reported_cached_tokens'],2400)
        self.assertIn('2,400',performance_text({'prompt_cache':hit}))
        self.assertIsNone(prompt_cache_metrics({'prompt_tokens_details':{'cached_tokens':True}},'guide',requested=True)['reported_cached_tokens'])

    def test_background_history_keeps_public_call_and_internal_pairing_distinct(self):
        from desktop_agent.agent import pack_messages
        from desktop_agent.api import native_history
        from desktop_agent.protocol import normalize_call
        action=normalize_call({'tool':'desktop_capture','arguments':{'background':True}})
        events=[dict(id=1,role='user',content='Capture in background'),
            dict(id=2,role='assistant',content=json.dumps(action),metadata={'call_id':'start'}),
            dict(id=3,role='tool',content=json.dumps(dict(job_id='job',status='running')),metadata=dict(tool='job_start',status='delivered',call_id='start')),
            dict(id=4,role='tool',content='captured',metadata=dict(tool='desktop_capture',status='delivered',call_id='start',job_id='job'))]
        messages,_,_=pack_messages(events,'system',len,30000)
        result=json.loads(messages[-1]['content'])
        self.assertEqual(result['call']['tool'],'desktop_capture')
        self.assertEqual(result['call']['arguments'],{'background':True})
        native=native_history(messages[1:])
        self.assertEqual(native[1]['tool_calls'][0]['function']['name'],'desktop_capture')
        self.assertEqual(native[2]['tool_call_id'],'call_start')
        self.assertEqual(native[3]['role'],'user')
        self.assertEqual(json.loads(native[3]['content'])['call_id'],'start')

    def test_delayed_result_remains_identifiable_after_context_eviction(self):
        from desktop_agent.agent import pack_messages
        events=[dict(id=1,role='user',content='old '*2000),
            dict(id=2,role='assistant',content=json.dumps({'tool':'workspace_read','arguments':{'path':'original.txt'}}),metadata={'call_id':'original'}),
            dict(id=3,role='user',content='Check the pending result'),
            dict(id=4,role='tool',content='original content',metadata=dict(tool='workspace_read',status='delivered',call_id='original'))]
        messages,dropped,_=pack_messages(events,'system',len,4000,state={})
        self.assertGreater(dropped,0)
        record=next(json.loads(message['content']) for message in messages if message.get('_result_tool'))
        self.assertEqual(record['call_id'],'original')
        self.assertEqual(record['call']['arguments']['path'],'original.txt')
        self.assertEqual(record['raw_ref']['arguments']['event_id'],4)

    def test_agent_tracks_edit_and_exact_result_readback_end_to_end(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Agent,Settings
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.protocol import normalize_call
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'work';root.mkdir();(root/'fixture.txt').write_text('before')
            store=Store(Path(folder)/'data');identifier=store.create()
            runner=ToolRunner(store.artifact_directory(identifier),threading.Event(),Mock(return_value=True),Mock())
            runner.workspace_root=str(root)
            runner.tool_policies={'workspace_read':'allow','workspace_apply_patch':'ask','session_record':'allow'}
            model=Mock();model.settings=Settings(max_steps=8,context_tokens=16384)
            model.count=lambda text:len(text)//3
            seen=[]
            def generate(messages,*args):
                results=[json.loads(message['content']) for message in messages if message.get('_result_tool')]
                seen.append(results)
                if len(seen)==1:
                    call={'tool':'workspace_read','arguments':{'path':'fixture.txt'}}
                elif len(seen)==2:
                    self.assertTrue(results[-1]['call_id'])
                    call={'tool':'workspace_apply_patch','arguments':dict(path='fixture.txt',expected_sha256=results[-1]['result']['sha256'],old_text='before',new_text='after')}
                elif len(seen)==3:
                    self.assertEqual(results[-1]['execution_status'],'applied')
                    call=results[-1]['raw_ref']
                else:
                    self.assertEqual(results[-1]['tool'],'session_record')
                    self.assertEqual(json.loads(results[-1]['result']['text'])['status'],'applied')
                    call={'tool':'finish','arguments':{'text':'Change applied and stored result verified.'}}
                return normalize_call(call),{}
            model.generate.side_effect=generate
            try:
                Agent(store,model,runner,runner.stopped,Mock()).run(identifier,'Edit the workspace file fixture.txt and read its stored result.')
                self.assertEqual(len(seen),4)
                self.assertEqual((root/'fixture.txt').read_text(),'after')
                events=store.events(identifier)
                calls=[event['metadata']['call_id'] for event in events if event['role']=='assistant' and event['metadata'].get('call_id')]
                results=[event['metadata']['call_id'] for event in events if event['role']=='tool']
                self.assertEqual(calls,results)
                self.assertEqual(len(set(calls)),3)
            finally:
                runner.close()

    def test_tool_rows_match_reversed_results_and_terminal_lifecycle(self):
        from desktop_agent.app import tool_rows,conversation_items
        events=[dict(id=1,role='user',content='Run'),
            dict(id=2,role='assistant',content=json.dumps(dict(message='one',tool='terminal_start',arguments={'command':'one'},risk='routine')),metadata={'call_id':'one'}),
            dict(id=3,role='assistant',content=json.dumps(dict(message='two',tool='terminal_start',arguments={'command':'two'},risk='routine')),metadata={'call_id':'two'}),
            dict(id=4,role='system',content=json.dumps(dict(status='completed',exit_code=0)),metadata=dict(status='terminal',tool='terminal_start',call_id='two')),
            dict(id=5,role='tool',content=json.dumps(dict(status='running')),metadata=dict(status='delivered',tool='terminal_start',call_id='one')),
            dict(id=6,role='tool',content=json.dumps(dict(status='running')),metadata=dict(status='delivered',tool='terminal_start',call_id='two'))]
        rows=tool_rows(events)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['execution_status'],'running')
        self.assertEqual(rows[1]['execution_status'],'completed')
        self.assertEqual(rows[1]['result']['id'],4)
        self.assertEqual(len(rows[1]['results']),2)
        items=conversation_items(events)
        self.assertEqual(items[-1]['role'],'activity')
        self.assertEqual(len(items[-1]['events']),5)

    def test_exact_record_paging_is_session_scoped_and_not_reexcerpted(self):
        from desktop_agent.agent import pack_messages
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);session_id=store.create();other=store.create()
            raw='line\r\n'*2000+'TAIL'
            event_id=store.append(session_id,'tool',raw,tool='workspace_read',status='delivered',call_id='original')
            parts=[];offset=0
            while offset is not None:
                page=store.record_page(session_id,event_id,offset,7000)
                parts.append(page['text']);offset=page['next_offset']
            self.assertEqual(''.join(parts),raw)
            with self.assertRaises(ValueError):store.record_page(other,event_id)
            events=[dict(id=1,role='user',content='Read stored record'),dict(id=2,role='tool',content=json.dumps(page),metadata=dict(tool='session_record',status='delivered'))]
            messages,_,_=pack_messages(events,'system',len,30000)
            self.assertEqual(json.loads(messages[-1]['content'])['result']['text'],page['text'])

    def test_native_history_uses_exact_call_ids_and_keeps_ambiguous_text(self):
        from desktop_agent.api import native_history
        call={'tool':'workspace_read','arguments':{'path':'one.txt'}}
        messages=[dict(role='assistant',content=json.dumps(call),_call=call,_internal_tool='workspace_read',_call_id='one'),
                  dict(role='user',content='result',_result_tool='workspace_read',_call_id='two')]
        rows=native_history(messages)
        self.assertEqual(rows[1]['role'],'user')
        self.assertNotIn('tool_calls',rows[0])
        messages[-1]['_call_id']='one'
        rows=native_history(messages)
        self.assertEqual(rows[0]['tool_calls'][0]['id'],'call_one')
        self.assertEqual(rows[1]['tool_call_id'],'call_one')

    def test_linked_tool_history_and_structured_summary_preserve_originals(self):
        from copy import deepcopy
        from desktop_agent.agent import pack_messages
        from desktop_agent.history import linked_events,result_record
        call=lambda identifier,path:dict(id=identifier,role='assistant',content=json.dumps({'tool':'workspace_read','arguments':{'path':path}}),metadata={'call_id':path})
        events=[dict(id=1,role='user',content='Inspect'),call(2,'one'),call(3,'two'),
            dict(id=4,role='tool',content=json.dumps(dict(text='begin\n'+'middle '*1000+'\nimportant tail',sha256='hash-two',total_lines=500)),metadata=dict(tool='workspace_read',status='delivered',call_id='two')),
            dict(id=5,role='tool',content='failure',metadata=dict(tool='workspace_read',status='error',call_id='one'))]
        original=deepcopy(events)
        linked=linked_events(events)
        result=result_record(linked[3])
        self.assertEqual(result['call_event_id'],3)
        self.assertEqual(result['call']['arguments']['path'],'two')
        self.assertEqual(result['call']['raw_ref']['arguments']['event_id'],3)
        self.assertEqual(result['summary']['sha256'],'hash-two')
        self.assertEqual(result['raw_ref']['arguments']['event_id'],4)
        self.assertIn('important tail',result['result']['text'])
        self.assertTrue(result['excerpt'])
        self.assertEqual(result_record(linked[4])['execution_status'],'failed')
        messages,_,_=pack_messages(events,'system',len,30000)
        self.assertEqual(json.loads(messages[2]['content'])['call_id'],'one')
        self.assertEqual(json.loads(messages[2]['content'])['raw_ref']['arguments']['event_id'],2)
        self.assertEqual(events,original)
        legacy=deepcopy(events)
        for event in legacy:
            event.get('metadata',{}).pop('call_id',None)
        self.assertNotIn('call_id',linked_events(legacy)[3]['metadata'])

    def test_call_identity_survives_background_collection_and_explicit_query(self):
        import threading
        from concurrent.futures import Future
        from unittest.mock import Mock,patch
        from desktop_agent.agent import Agent
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.tools import ToolResult
        from desktop_agent.protocol import normalize_call
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);session_id=store.create()
            runner=ToolRunner(store.artifact_directory(session_id),threading.Event(),Mock(return_value=True),Mock())
            agent=Agent(store,Mock(),runner,runner.stopped,Mock())
            try:
                futures=[]
                def submit(*args):
                    future=Future();futures.append(future)
                    return future,threading.Event()
                action=normalize_call({'tool':'desktop_capture','arguments':{'background':True}})
                with patch.object(runner,'submit',side_effect=submit):
                    runner.call_id='start-one';first=json.loads(runner.execute(action).text)['job_id']
                    runner.call_id='start-two';second=json.loads(runner.execute(action).text)['job_id']
                futures[1].set_result(ToolResult('second result'))
                collected=runner.collect_ready()
                self.assertEqual(collected[0].call_id,'start-two')
                agent.save_result(session_id,collected[0].tool,collected[0])
                self.assertEqual(store.events(session_id)[-1]['metadata']['call_id'],'start-two')
                futures[0].set_result(ToolResult('first result'))
                result=runner.execute(normalize_call({'tool':'job_result','arguments':{'job_id':first}}))
                agent.save_result(session_id,'job_result',result,'query-one')
                metadata=store.events(session_id)[-1]['metadata']
                self.assertEqual(metadata['call_id'],'query-one')
                self.assertEqual(metadata['origin_call_id'],'start-one')
                self.assertEqual(metadata['requested_tool'],'job_result')
                self.assertNotEqual(first,second)
            finally:
                runner.close()

    def test_model_guide_matches_tools_permissions_and_persistent_images(self):
        from desktop_agent.agent import system_prompt
        from desktop_agent.protocol import ToolCatalog,WORKSPACE_TOOLS,tool_description,compact_parameters
        capabilities=dict(screen=False,input=False,browser=False,approval='routine',workspace_root='F:/fixture',
                          tool_policies={name:'ask' for name in WORKSPACE_TOOLS})
        catalog=ToolCatalog(capabilities)
        base=system_prompt(capabilities,catalog)
        self.assertNotIn('\nWORKSPACE WORKFLOW\n',base)
        self.assertNotIn('\nTERMINAL WORKFLOW\n',base)
        catalog.load(['files','workspace','terminal'])
        guide=system_prompt(capabilities,catalog)
        self.assertIn('file_read and file_view accept only registered',guide)
        self.assertIn('Use workspace_* for files under CAPABILITIES.workspace_root',guide)
        self.assertIn('remove displayed line-number prefixes',guide)
        self.assertIn('preserve whitespace and the returned line_ending',guide)
        self.assertIn('finish cancels unfinished PowerShell executions; it does not wait',guide)
        self.assertIn('Each call starts a fresh shell',guide)
        self.assertIn('offset and next_offset are byte positions',guide)
        self.assertIn('stays attached until a newer tool image replaces it',guide)
        for stale in ('next request only','attaches an image once','File tools read registered session files only'):
            self.assertNotIn(stale,guide)
            self.assertNotIn(stale,tool_description('file_view'))
        for name in ('file_view',*WORKSPACE_TOOLS):
            line=next(line for line in guide.splitlines() if line.startswith(name+': '))
            prefix=name+': '+tool_description(name)+' '
            spec=json.loads(line[len(prefix):])
            expected=compact_parameters(name)
            self.assertEqual(spec,dict(required=expected['required'],arguments=expected['properties']))
        blocked=ToolCatalog(dict(capabilities,tool_policies={}),prompt='edit file powershell')
        self.assertFalse(blocked.available('workspace'))
        self.assertFalse(blocked.available('terminal'))
        blocked_guide=system_prompt(blocked.capabilities,blocked)
        self.assertNotIn('\nterminal_start:',blocked_guide)
        self.assertNotIn('\nWORKSPACE WORKFLOW\n',blocked_guide)

    def test_model_guide_reaches_qwen_27b_payload_without_changing_reasoning(self):
        import threading
        from unittest.mock import MagicMock,patch
        from desktop_agent.agent import Model,Settings,pack_messages,system_prompt
        from desktop_agent.models import Q2_XL
        from desktop_agent.protocol import ToolCatalog,WORKSPACE_TOOLS,compact_schema
        capabilities=dict(screen=False,input=False,browser=False,approval='routine',workspace_root='F:/fixture',
                          tool_policies={name:'ask' for name in WORKSPACE_TOOLS})
        catalog=ToolCatalog(capabilities)
        catalog.load(['files','workspace','terminal'])
        guide=system_prompt(capabilities,catalog)
        events=[dict(id=1,role='user',content='Read the workspace file, apply the requested edit, then verify it.')]
        messages,_,_=pack_messages(events,guide,len,100000,state=dict(terminals=[],steps_remaining=20))
        for enabled in (False,True):
            with self.subTest(reasoning_enabled=enabled):
                model=Model(Settings(model=Q2_XL['model'],reasoning_enabled=enabled,reasoning_effort='xhigh'))
                model.endpoint='http://127.0.0.1:8080'
                model.tool_names=catalog.names()
                client=MagicMock()
                client.__enter__.return_value=client
                client.post.side_effect=RuntimeError('offline guide payload')
                with patch('desktop_agent.agent.session',return_value=client):
                    with self.assertRaisesRegex(RuntimeError,'offline guide payload'):
                        model.generate(messages,None,threading.Event(),lambda *args:None)
                payload=client.post.call_args.kwargs['json']
                self.assertEqual(payload['messages'][0],dict(role='system',content=guide))
                self.assertEqual(payload['response_format']['json_schema']['schema'],compact_schema(catalog.names()))
                self.assertTrue(payload['response_format']['json_schema']['strict'])
                self.assertEqual(payload['chat_template_kwargs']['enable_thinking'],enabled)
                self.assertEqual(payload['reasoning_effort'],'xhigh' if enabled else 'none')
                self.assertEqual(payload['max_tokens'],-1)

    def test_terminal_storage_failure_still_completes_owned_cleanup(self):
        import threading
        from unittest.mock import Mock,patch
        from desktop_agent.terminals import Terminals
        with tempfile.TemporaryDirectory() as folder:
            manager=Terminals(folder,threading.Event(),Mock())
            try:
                blocked="$pipe=[System.IO.Pipes.NamedPipeServerStream]::new('desk-storage-'+[guid]::NewGuid().ToString()); $pipe.WaitForConnection()"
                run=manager.start(blocked,folder,10)
                with patch.object(manager,'persist',side_effect=OSError('disk fixture')):
                    manager.cancel_all()
                    self.assertTrue(manager.records[run['execution_id']]['done'].wait(7))
                    self.assertIsNotNone(manager.records[run['execution_id']]['process'].poll())
                    self.assertEqual(manager.output(run['execution_id'])['status'],'storage_error')
            finally:
                manager.close()

    def test_workspace_large_diff_is_archived_and_limits_are_generous(self):
        from desktop_agent.workspace_tools import Workspace,MAX_FILE_BYTES,MAX_DIFF_CHARS,MAX_SEARCH_SECONDS,MAX_SEARCH_FILES,digest
        self.assertGreaterEqual(MAX_FILE_BYTES,16*1024*1024)
        self.assertGreaterEqual(MAX_DIFF_CHARS,8*1024*1024)
        self.assertGreaterEqual(MAX_SEARCH_SECONDS,60)
        self.assertGreaterEqual(MAX_SEARCH_FILES,100000)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'work';root.mkdir()
            workspace=Workspace(str(root),Path(folder)/'session')
            original=('before line\n'*3000).encode()
            target=root/'large.txt';target.write_bytes(original)
            change=workspace.prepare('large.txt',digest(original),original.decode(),'after line\n'*3000)
            self.assertGreater(len(change['diff']),24000)
            result=workspace.commit(change)
            self.assertEqual(result['status'],'applied')
            self.assertIn('2000: after line',workspace.read('large.txt',1,2000)['text'])
            self.assertEqual(len(workspace.search('after',limit=1000)['matches']),1000)

    def test_terminal_timeout_child_cleanup_cancellation_and_output_cap(self):
        import re
        import threading
        import psutil
        from unittest.mock import Mock,patch
        from desktop_agent.terminals import Terminals,OUTPUT_LIMIT
        blocked="$pipe = [System.IO.Pipes.NamedPipeServerStream]::new('desk-fixture-'+[guid]::NewGuid().ToString()); $pipe.WaitForConnection()"
        child_arguments='-NoLogo -NoProfile -NonInteractive -Command "& { '+blocked+' }"'
        command="$child = Start-Process -FilePath ($PSHOME+'\\powershell.exe') -ArgumentList '"+child_arguments.replace("'","''")+"' -NoNewWindow -PassThru; Write-Output ('CHILD_PID='+$child.Id); "+blocked
        with tempfile.TemporaryDirectory() as folder:
            stopped=threading.Event()
            manager=Terminals(folder,stopped,Mock())
            try:
                run=manager.start(command,folder,2)
                self.assertTrue(manager.records[run['execution_id']]['done'].wait(9))
                result=manager.output(run['execution_id'])
                self.assertEqual(result['status'],'timed_out',result)
                child=re.search(r'CHILD_PID=(\d+)',result['text'])
                self.assertIsNotNone(child,result)
                self.assertFalse(psutil.pid_exists(int(child.group(1))))
                run=manager.start(blocked,folder,10)
                stopped.set()
                self.assertTrue(manager.records[run['execution_id']]['done'].wait(7))
                self.assertEqual(manager.output(run['execution_id'])['status'],'cancelled')
                stopped.clear()
                self.assertEqual(OUTPUT_LIMIT,64*1024*1024)
                with patch('desktop_agent.terminals.OUTPUT_LIMIT',8192):
                    run=manager.start("[Console]::Write(('x' * 16384)); "+blocked,folder,10)
                    self.assertTrue(manager.records[run['execution_id']]['done'].wait(10))
                    result=manager.output(run['execution_id'])
                    self.assertEqual(result['status'],'output_limit',result)
                    self.assertEqual(result['output_bytes'],8192)
            finally:
                manager.close()

    def test_workspace_runner_approval_denial_and_conflict(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.protocol import normalize_call
        from desktop_agent.workspace_tools import digest
        from game_agent.core import Halted
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'work';root.mkdir()
            target=root/'fixture.txt';target.write_bytes(b'before')
            approval=Mock(return_value=False)
            runner=ToolRunner(Path(folder)/'session',threading.Event(),approval,Mock())
            runner.workspace_root=str(root)
            runner.tool_policies={'workspace_apply_patch':'allow','workspace_read':'ask'}
            action=normalize_call({'tool':'workspace_apply_patch','arguments':dict(path='fixture.txt',expected_sha256=digest(b'before'),old_text='before',new_text='after')})
            try:
                with self.assertRaises(Halted): runner.execute(action)
                self.assertEqual(target.read_bytes(),b'before')
                self.assertIn('-before',approval.call_args.args[2]['diff'])
                def conflict(*args):
                    target.write_bytes(b'external change')
                    return True
                approval.side_effect=conflict
                with self.assertRaisesRegex(ValueError,'changed during approval'): runner.execute(action)
                self.assertEqual(target.read_bytes(),b'external change')
                approval.side_effect=None;approval.return_value=True
                result=runner.execute(normalize_call({'tool':'workspace_read','arguments':{'path':'fixture.txt'}}))
                self.assertIn('external change',result.text)
                runner.tool_policies={'terminal_start':'disabled'}
                with self.assertRaises(ValueError):
                    runner.execute(normalize_call({'tool':'terminal_start','arguments':{'command':'Write-Output denied'}}))
                self.assertEqual(runner.terminals.summary(),[])
            finally:
                runner.close()

    def test_terminal_owned_execution_and_saved_output(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.terminals import Terminals
        with tempfile.TemporaryDirectory() as folder:
            manager=Terminals(folder,threading.Event(),Mock())
            try:
                started=manager.start("Write-Output 'local-desk-fixture'",folder,10,call_id='fixture-call')
                identifier=started['execution_id']
                self.assertTrue(manager.records[identifier]['done'].wait(15))
                result=manager.output(identifier)
                self.assertEqual(result['status'],'completed',result)
                self.assertEqual(result['exit_code'],0)
                self.assertEqual(result['call_id'],'fixture-call')
                self.assertIn('local-desk-fixture',result['text'])
                restored=Terminals(folder,threading.Event(),Mock())
                self.assertEqual(restored.output(identifier)['text'],result['text'])
                with self.assertRaises(ValueError): restored.stop(identifier)
                with self.assertRaises(ValueError): manager.output('../outside')
                with self.assertRaises(ValueError): manager.start('Read-Host password',folder,10)
            finally:
                manager.close()

    def test_workspace_files_patch_and_boundaries(self):
        from desktop_agent.workspace_tools import Workspace,digest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'work';root.mkdir()
            workspace=Workspace(str(root),Path(folder)/'session')
            target=root/'sample.txt';target.write_bytes(b'first\r\nsecond\r\n')
            reading=workspace.read('sample.txt')
            self.assertEqual(reading['sha256'],digest(target.read_bytes()))
            self.assertEqual(workspace.search('second')['matches'][0]['line'],2)
            change=workspace.prepare('sample.txt',reading['sha256'],'second','updated')
            self.assertIn('-second',change['diff'])
            result=workspace.commit(change)
            self.assertEqual(target.read_bytes(),b'first\r\nupdated\r\n')
            self.assertEqual(result['status'],'applied')
            self.assertTrue((Path(folder)/'session/file-changes'/result['change_id']/'before.txt').exists())
            with self.assertRaises(ValueError): workspace.commit(change)
            for path in ('../outside.txt','.env','private.key','C:/outside.txt','sample.txt:stream'):
                with self.assertRaises(ValueError,msg=path): workspace.path(path)
            linked=root/'linked.txt';os.link(target,linked)
            with self.assertRaises(ValueError): workspace.read('linked.txt')
            linked.unlink()
            workspace.commit(workspace.prepare('new.txt','','','created'))
            self.assertEqual((root/'new.txt').read_text(),'created')

    def test_tool_policy_checked_before_window_or_background_execution(self):
        import threading
        from unittest.mock import Mock,patch
        from desktop_agent.tools import Tools
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.protocol import normalize_call
        from game_agent.core import Halted
        with tempfile.TemporaryDirectory() as folder:
            approve=Mock(return_value=False)
            tools=Tools(folder,threading.Event(),approve,Mock())
            action=normalize_call({'tool':'window_list','arguments':{}})
            tools.tool_policies={'window_list':'disabled'}
            with patch('desktop_agent.tools.list_windows') as listing:
                with self.assertRaises(ValueError):
                    tools.execute(action)
                listing.assert_not_called()
            tools.tool_policies={'window_list':'ask'}
            with patch('desktop_agent.tools.list_windows',return_value=[]):
                with self.assertRaises(Halted):
                    tools.execute(action)
            approve.assert_called_once()
            runner=ToolRunner(folder,threading.Event(),approve,Mock())
            runner.tool_policies={'desktop_capture':'disabled'}
            nested=normalize_call({'tool':'job_start','arguments':{'action':normalize_call({'tool':'desktop_capture','arguments':{}})}})
            try:
                with patch.object(runner,'submit') as submit:
                    with self.assertRaises(ValueError):
                        runner.execute(nested)
                    submit.assert_not_called()
            finally:
                runner.close()

    def test_workspace_tool_policies_are_default_off_and_persist(self):
        from desktop_agent.protocol import ToolCatalog,normalize_call,approval_reason
        from desktop_agent.agent import Settings
        catalog=ToolCatalog({'screen':True,'workspace_root':'F:/fixture'},'edit file powershell')
        self.assertNotIn('terminal_start',catalog.names())
        policies={'terminal_start':'ask','workspace_read':'allow','desktop_capture':'disabled'}
        catalog=ToolCatalog({'screen':True,'workspace_root':'F:/fixture','tool_policies':policies},'edit file powershell')
        self.assertIn('terminal_start',catalog.names())
        self.assertNotIn('desktop_capture',catalog.names())
        action=normalize_call({'tool':'terminal_start','arguments':{'command':'Write-Output test'}})
        self.assertTrue(approval_reason(action,'routine',{'tool_policies':policies}))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            Settings(workspace_root='F:/fixture',tool_policies=policies).save(path)
            self.assertEqual(Settings.load(path).tool_policies,policies)

    def test_kv_cache_choice_persists_and_selects_safe_runtime(self):
        from desktop_agent.agent import Settings,DesktopServer
        from desktop_agent.models import Q2_XL
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            for kind in ('default','q4_0','q8_0','f16'):
                settings=Settings(model=Q2_XL['model'],projector=Q2_XL['projector'],kv_cache_type=kind)
                settings.save(path)
                self.assertEqual(Settings.load(path).kv_cache_type,kind)
                server=DesktopServer();server.kv_cache_type=kind
                values=server.settings_for(settings.executable,settings.model,settings.projector,1024,1024)
                for flag in ('-ctk','-ctv'):
                    self.assertEqual(values[-1].count(flag),1)
                    self.assertEqual(values[-1][values[-1].index(flag)+1],'q8_0' if kind=='default' else kind)
                self.assertTrue(server.front_enabled(settings.model,settings.projector))
            settings=Settings(kv_cache_type='invalid');settings.save(path)
            with self.assertRaises(ValueError): Settings.load(path)

    def test_q2xl_front_bundle_rejects_changes_and_isolates_environment(self):
        from unittest.mock import patch
        from desktop_agent import residency,models
        with tempfile.TemporaryDirectory() as folder:
            runtime=Path(folder)/'runtime'
            runtime.mkdir()
            model=Path(folder)/models.Q2_XL['model']
            projector=Path(folder)/models.Q2_XL['projector']
            for path in (model,projector,runtime/'llama-server.exe'): path.write_bytes(b'fixture')
            table=runtime/'workspace-table.json'
            table.write_text(json.dumps({'identity':'fixture'}))
            record=dict(schema=1,baseline_id='front_native_postencode',identity='fixture',
                files={'llama-server.exe':residency.sha256(runtime/'llama-server.exe')},
                table_sha256=residency.sha256(table),environment={'LOCAL_DESK_FRONT_ID':'fixture'})
            for name,path in (('model',model),('projector',projector)):
                record[name]=dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
            (runtime/'deployment.json').write_text(json.dumps(record))
            with patch.object(models,'Q2_FRONT_RUNTIME',runtime),patch.dict(os.environ,
                    {'LOCAL_DESK_FRONT_TRACE':'1','LOCAL_DESK_EXACT_COMPRESSED':'1','GGML_CUDA_DISABLE_GRAPHS':'1'}):
                before=dict(os.environ)
                verified,child=residency.deployment(str(model),str(projector))
                self.assertEqual(verified,record)
                self.assertEqual(child['LOCAL_DESK_FRONT_TABLE'],str(table))
                self.assertTrue(child['PATH'].startswith(str(runtime)+os.pathsep))
                for key in ('LOCAL_DESK_FRONT_TRACE','LOCAL_DESK_EXACT_COMPRESSED','GGML_CUDA_DISABLE_GRAPHS'):
                    self.assertNotIn(key,child)
                self.assertEqual(before,dict(os.environ))
                table.write_text('{}')
                with self.assertRaisesRegex(ValueError,'table mismatch'): residency.deployment(str(model),str(projector))
                table.write_text(json.dumps({'identity':'fixture'}))
                (runtime/'llama-server.exe').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'file mismatch'): residency.deployment(str(model),str(projector))

    def test_q2xl_front_guard_stops_owned_process_without_request_deadline(self):
        import threading
        from unittest.mock import Mock,patch
        from desktop_agent.residency import ResidencyGuard
        server=Mock()
        server.process.poll.return_value=None
        server.process.pid=123
        record=dict(rtx_luid='rtx',rx_luid='rx',caps=dict(rtx_dedicated=7850*1024**2,rtx_shared=184*1024**2,available_ram=4*1024**3))
        row=dict(adapter_dedicated={'rtx':100,'rx':100},process_shared={'pid_123_rtx':100},engine_utilization={'fixture':1})
        stopped=threading.Event()
        with patch('desktop_agent.benchmark_resources.GPUCounters') as counters, \
             patch('desktop_agent.benchmark_resources.system_memory',return_value={'physical_available':8*1024**3}):
            counters.return_value.read.return_value=row
            guard=ResidencyGuard(server,stopped,record)
            guard.sample()
            self.assertTrue(guard.process_seen)
            with patch('desktop_agent.residency.time.monotonic',return_value=100): guard.begin_request()
            for elapsed in (76,91,181,599,600,600.001,86400):
                with patch('desktop_agent.residency.time.monotonic',return_value=100+elapsed): guard.sample()
            with patch('desktop_agent.residency.time.monotonic',return_value=86500):
                guard.end_request()
                guard.begin_request()
            server.process.terminate.assert_not_called()
            guard.request_started=None
            guard.finished=Mock()
            guard.finished.wait.return_value=False
            with patch.object(guard,'sample',side_effect=[None,RuntimeError('guard sentinel')]):
                guard.start()
                guard.thread.join(timeout=2)
            self.assertFalse(guard.thread.is_alive())
            self.assertEqual(guard.reason,'guard sentinel')
            self.assertTrue(stopped.is_set())
            server.process.terminate.assert_called_once()
            guard.close()

    def test_q2xl_front_runtime_reaches_process_and_keeps_fingerprint(self):
        import threading
        from unittest.mock import Mock,patch
        from desktop_agent.agent import DesktopServer,Settings
        from desktop_agent.models import Q2_XL,Q2_PROFILE_RUNTIME
        from desktop_agent.benchmark_placement import PlacementServer
        server=DesktopServer()
        settings=Settings()
        process=Mock()
        process.poll.return_value=None
        with tempfile.TemporaryDirectory() as folder:
            with patch('desktop_agent.residency.deployment',return_value=({},dict(TEST_FRONT='1'))), \
                 patch('desktop_agent.residency.ResidencyGuard') as guard, \
                 patch('game_agent.runtime.Path.is_file',return_value=True), \
                 patch('game_agent.runtime.subprocess.Popen',return_value=process) as popen, \
                 patch('game_agent.runtime.session') as client:
                client.return_value.__enter__.return_value.get.return_value.status_code=200
                server.start(settings.executable,Q2_XL['model'],Q2_XL['projector'],Path(folder)/'server.log',threading.Event(),1024,1024)
                self.assertEqual(popen.call_args.args[0][0],str(Q2_PROFILE_RUNTIME/'llama-server.exe'))
                self.assertEqual(popen.call_args.kwargs['env'],dict(TEST_FRONT='1'))
                self.assertEqual(server.loaded_settings,server.settings_for(settings.executable,Q2_XL['model'],Q2_XL['projector'],1024,1024))
                server.start(settings.executable,Q2_XL['model'],Q2_XL['projector'],Path(folder)/'server.log',threading.Event(),1024,1024)
                self.assertEqual(popen.call_count,1)
                guard.return_value.start.assert_called_once()
                server.close()
                guard.return_value.close.assert_called_once()
        self.assertFalse(PlacementServer({}).use_front_residency)

    def test_q2xl_front_memory_limits_fail_closed(self):
        from desktop_agent.residency import memory_reason
        record=dict(rtx_luid='rtx',rx_luid='rx',caps=dict(rtx_dedicated=7850*1024**2,rtx_shared=184*1024**2,available_ram=4*1024**3))
        row=dict(adapter_dedicated={'rtx':7850*1024**2,'rx':100},process_shared={'pid_123_rtx':184*1024**2})
        self.assertEqual(memory_reason(row,4*1024**3,123,record,True),'')
        self.assertTrue(memory_reason(row,4*1024**3-1,123,record,True))
        self.assertTrue(memory_reason(dict(row,process_shared={}),4*1024**3,123,record,True))
        self.assertTrue(memory_reason(dict(row,process_shared={'pid_123_rtx':184*1024**2+1}),4*1024**3,123,record,True))
        self.assertTrue(memory_reason(dict(row,adapter_dedicated={'rtx':7850*1024**2+1,'rx':100}),4*1024**3,123,record,True))

    def test_q2xl_front_residency_scope_preserves_other_profiles(self):
        from desktop_agent.models import Q2_XL,IQ2_S,uses_front_residency
        self.assertTrue(uses_front_residency(Q2_XL['model'],Q2_XL['projector'],8192,0))
        self.assertTrue(uses_front_residency(Q2_XL['model'],Q2_XL['projector'],24576,2048))
        for model,projector,context,cache in ((Q2_XL['model'],Q2_XL['projector'],12289,0),
            (Q2_XL['model'],Q2_XL['projector'],8192,4096),
                (Q2_XL['model'],'mmproj-F16.gguf',8192,0),
                (IQ2_S['model'],IQ2_S['projector'],8192,0)):
            self.assertFalse(uses_front_residency(model,projector,context,cache))

    def test_front_reserve_restores_full_plan_after_replica_release(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-front-compare-20260922.reserve')
        original=(study.WARM.HOME/'controller.inc').read_text(encoding='utf-8')
        changed=study.controller(original)
        self.assertEqual(changed.replace(study.RESERVE,''),original)
        release=changed.split('    void release(uint64_t generation) {',1)[1].split('\n};',1)[0]
        self.assertLess(release.index('llama_research_return(context,generation)'),release.index('ggml_backend_buffer_free(replica_buffer)'))
        self.assertLess(release.index('ggml_backend_buffer_free(replica_buffer)'),release.index('research_reserve_baseline()'))
        self.assertLess(release.index('counters.safe((850+64)*residency_mib)'),release.index('research_reserve_baseline()'))
        self.assertIn('if (!residency_request.allow_text_residency())',release)
        self.assertIn('memory->init_full()',study.METHOD)
        self.assertIn('graph_reserve(tokens,cparams.n_seq_max',study.METHOD)
        self.assertNotIn('llama_decode',study.METHOD)
        self.assertNotIn('sched.reset(',study.METHOD)

    def test_front_trace_preserves_computation_and_fences(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-front-compare-20260922.trace')
        original=(study.WARM.WORK/'source/src/llama-context.cpp').read_text(encoding='utf-8')
        changed=study.instrument(original)
        self.assertEqual(changed.count('ggml_backend_sched_synchronize('),original.count('ggml_backend_sched_synchronize('))
        self.assertEqual(changed.count('model.build_graph(gparams)'),original.count('model.build_graph(gparams)'))
        self.assertIn('phase_trace.mark("compute_submit");',changed)
        self.assertIn('FRONT_SYNC queued_tokens=',changed)
        self.assertEqual(study.ORDER,(0,1,2,10,3))
        lines=[]
        for index,tokens in ((1,512),(2,512),(3,42)):
            lines.append(f'decoding image batch {index}/3, n_tokens_batch = {tokens}')
            for count in ([128]*4 if tokens==512 else [42]):
                lines.append(f'FRONT_UBATCH tokens={count} embeddings=1 reused=0 status=0 begin_us=10 end_us=20 total_us=10 build_us=2')
            lines.append(f'image decoded (batch {index}/3) in 100 ms')
        log='\n'.join(lines)
        self.assertEqual(study.phase_rows(log)[1]['phase_seconds']['build_us'],0.000008)
        with self.assertRaises(ValueError): study.phase_rows(log.replace('tokens=128','tokens=127',1))
        with self.assertRaises(ValueError): study.phase_rows(log.rsplit('\n',1)[0])

    def test_front_warm_return_keeps_fresh_memory_gate(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-front-compare-20260922.warm')
        original=(study.PARENT.HOME/'controller.inc').read_text()
        changed=study.fix(original)
        release=changed.split('    void release(uint64_t generation) {',1)[1].split('\n};',1)[0]
        self.assertLess(release.index('llama_research_return(context,generation)'),release.index('ggml_backend_buffer_free(replica_buffer)'))
        self.assertLess(release.index('ggml_backend_buffer_free(replica_buffer)'),release.index('counters.safe(850*residency_mib)'))
        self.assertIn('if (!retain_compute && !llama_research_release_compute',release)
        self.assertIn('context->research_clear_graphs();',release)
        self.assertEqual(changed.replace(study.REPLACEMENT,study.ANCHOR),original)

    def test_front_comparison_phase_and_token_accounting(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-front-compare-20260922.study')
        sample=dict(kind='text',input_sha256='same',answer_sha256='answer',tool='finish',usage={'completion_tokens':11},
            seconds=2.0,timings=dict(predicted_ms=1000,prompt_ms=900,predicted_n=11,predicted_per_second=10),
            breakdown={'vision_encode_seconds':0})
        changed=dict(sample,seconds=3.0,usage={'completion_tokens':21},
            timings=dict(sample['timings'],predicted_ms=2000,predicted_n=21))
        row=study.compare([sample],[changed])[0]
        self.assertFalse(row['same_usage'])
        self.assertEqual(row['extra_decode_ms_per_step'],0)
        self.assertEqual(row['decode_delta_seconds'],1)
        with self.assertRaises(ValueError): study.compare([sample],[dict(changed,input_sha256='different')])
        events=study.phases('1.02.123.456 I FRONT_TABLE_SELECT source=RAM previews=0 elapsed_us=427\nRESIDENCY_VISION phase=restored elapsed_us=84000')
        self.assertAlmostEqual(events[0]['seconds'],62.123456)
        self.assertEqual(events[0]['fields']['elapsed_us'],'427')
        self.assertIsNone(events[1]['seconds'])
        self.assertEqual(study.summarize([dict(sample,kind='warmup'),sample])['all']['requests'],1)
        analysis=importlib.import_module('desktop_agent.data.benchmarks.rtx-front-compare-20260922.analysis')
        log='\n'.join(f'image decoded (batch {index}/3) in {elapsed} ms' for index,elapsed in ((1,2500),(2,2900),(3,620)))
        self.assertEqual(analysis.image_batches(log),[[2.5,2.9,0.62]])
        with self.assertRaises(ValueError): analysis.image_batches(log.rsplit('\n',1)[0])
        with self.assertRaises(ValueError): analysis.image_batches(log.replace('batch 2/3','batch 1/3'))

    def test_residency_front_native_table_contract(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-front-native-20260922.study')
        table=dict(schema=1,identity='candidate',layers=study.LAYERS,ffns=study.FFNS,
            kernel='native',tokens=[1,128],margin_bytes=study.TABLE_MARGIN,
            workspace_bytes=[64,32,64,32],allocation_bytes=1024,signature={'context':8192})
        self.assertEqual(study.validate_table(table,'candidate'),table)
        for change in ({'identity':'other'},{'layers':1<<56},{'ffns':0},{'kernel':'exact'},
                       {'tokens':[1,256]},{'workspace_bytes':[-1]},{'margin_bytes':0},{'signature':{}}):
            with self.assertRaises(ValueError): study.validate_table(dict(table,**change),'candidate')
        self.assertEqual([block for block in range(64) if study.LAYERS&(1<<block)],[52,53,54])
        self.assertEqual(study.FFNS,1<<51)
        summary=study.speed_summary([dict(kind='text',passed=True,timings={'predicted_per_second':14.1})])
        self.assertTrue(summary['all_at_least_14'])
        self.assertFalse(summary['strict_numerical_equivalence_qualified'])
        changed=study.patch_controller((study.PRIOR.HOME/'controller.inc').read_text())
        choose=changed.split('    residency_candidate choose(uint64_t generation) {',1)[1].split('    void prepare(',1)[0]
        self.assertNotIn('llama_research_preview',choose)
        self.assertIn('counters.safe(cost)',choose)
        self.assertIn('front_signature(items)',choose)
        self.assertIn('load_front_table();',changed)
        self.assertNotIn('RESIDENCY_SEARCH_BEGIN',changed)
        self.assertIn('llama_research_front_precompute(context)',study.probe_source())

    def test_residency_request_image_hold_and_live_cache_admission(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-request-residency-20260922.study')
        old=(study.PRIOR.HOME/'controller.inc').read_text()
        changed=study.patch_controller(old)
        poll=changed.split('if (phase==LLAMA_RESEARCH_POLL_COPY)',1)[1]
        self.assertLess(poll.index('!residency_request.allow_text_residency()'),poll.index('if (residency_return_requested)'))
        self.assertIn('preview_costs.get(candidate.layers,candidate.ffns,sizes)',changed)
        self.assertIn('for (const auto & candidate:options)',changed)
        self.assertIn('counters.safe(cost)',changed)
        self.assertIn('std::vector<uint64_t> signature=',changed)
        self.assertNotIn('residency_vision_data=data;residency_text_pending=true',changed)
        self.assertIn('residency_vision_parked.load() && !residency_auto->counters.safe(850*residency_mib)',changed)
        self.assertNotIn('baseline',study.ARMS)
        server=(study.PRIOR.WORK/'source/tools/server/server-context.cpp').read_text(encoding='utf-8')
        patched=study.patch_server(server)
        self.assertIn('task.tokens.find_next_media_chunk(0).first!=nullptr',patched)
        self.assertLess(patched.index('llama_residency_begin_request(residency_has_media'),patched.index('slot.task = std::make_unique<const server_task>'))

    def test_residency_highwater_uses_observed_nonreplica_bytes(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-highwater-20260922.study')
        log=(study.PRIOR.HOME/'qualification/residency_qualification.log').read_text(encoding='utf-8')
        overhead=study.measured_overhead(log)
        self.assertGreaterEqual(overhead,8091557888-7059750912-878253056)
        self.assertLess(overhead-(8091557888-7059750912-878253056),65536)
        patched=study.patch_controller((study.PRIOR.HOME/'controller.inc').read_text(),overhead)
        self.assertIn('Active residency support limit',patched)
        self.assertIn('library_growth=std::max(library_growth,observed_operational_overhead)',patched)

    def test_residency_pins_selected_intermediate_nodes(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-pinned-20260922.study')
        root=study.PRIOR.WORK/'source/src'
        graph=study.patch_graph((root/'llama-graph.cpp').read_text(encoding='utf-8'))
        for name in ('range','indices','decoded','tile','output'):
            self.assertIn('residency_pin_cuda(scheduler,'+name+')',graph)
        context=study.patch_context((root/'llama-context.cpp').read_text(encoding='utf-8'))
        self.assertIn('residency::exact_weight(layer.attn_norm)',context)
        self.assertIn('whole_ffn && std::strncmp(name,"ffn_",4)==0',context)

    def test_residency_admission_checks_physical_growth_before_upload(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-admission-20260922.study')
        changed=study.patch_controller((study.PRIOR.HOME/'controller.inc').read_text())
        body=changed.split('    void prepare(uint64_t generation)',1)[1].split('    void upload(',1)[0]
        self.assertLess(body.index('RESIDENCY_ALLOCATION'),body.index('uploader=std::thread'))
        self.assertIn('rejected_allocation_bytes=chosen.bytes',body)
        self.assertIn('replicas.clear();weights.clear()',body)
        self.assertIn('residency::round_resident(candidate.bytes,resident_quantum)',changed)

    def test_residency_primes_only_first_original_batch(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-primed-20260922.study')
        changed=study.patch_controller((study.PRIOR.HOME/'controller.inc').read_text())
        body=changed.split('    void prepare(uint64_t generation)',1)[1].split('    void upload(',1)[0]
        self.assertLess(body.index('callback(data)'),body.index('baseline_batch_submitted=true'))
        self.assertLess(body.index('return;'),body.index('state.begin();initialize_compute_library();'))
        self.assertIn('state.released();baseline_batch_submitted=false;',changed)
        self.assertIn('extra_warmup=0',changed)

    def test_residency_library_accounts_for_each_cuda_instance(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-library-20260922.study')
        changed=study.patch_controller((study.PRIOR.HOME/'controller.inc').read_text())
        region=changed.split('void initialize_compute_library()',1)[1].split('std::vector<llama_research_storage>',1)[0]
        self.assertIn('std::unordered_set<ggml_backend_t> initialized',region)
        self.assertIn('initialized.insert(backend)',region)
        self.assertNotIn('break;',region)
        self.assertLess(region.index('RESIDENCY_LIBRARY_INSTANCE'),region.index('library_growth='))

    def test_residency_planner_preview_preserves_live_scheduler(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-planner-20260922.study')
        context=(study.PRIOR.WORK/'source/src/llama-context.cpp').read_text(encoding='utf-8')
        changed=study.patch_context(context)
        body=changed[changed.index('std::vector<size_t> llama_context::research_measure_compute()'):]
        self.assertLess(body.index('residency::preview_scheduler'),body.index('memory->init_full()'))
        self.assertIn('ggml_backend_sched_new(backend_ptrs.data()',body)
        scope=study.patch_scope((study.PRIOR.WORK/'source/include/exact-scope.h').read_text())
        self.assertIn('preview_in_progress=prior_preview;',scope)
        controller=study.patch_controller((study.PRIOR.HOME/'controller.inc').read_text())
        self.assertIn('for (const auto & candidate:options)',controller)
        self.assertIn('RESIDENCY_SEARCH_BEGIN',controller)

    def test_residency_resume_uses_measured_graph_metadata(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-resume-20260922.study')
        original=(study.SERVER.HOME/'controller.inc').read_text()
        fixed=study.patch_controller(original)
        self.assertNotIn('parameters={65536,nullptr,true}',fixed)
        self.assertIn('residency_library_graph initialization;',fixed)
        self.assertIn('ggml_backend_graph_compute(backend,initialization.graph)',fixed)
        header=(study.HOME/'library-graph.h').read_text()
        self.assertIn('ggml_graph_overhead_custom(nodes,false)',header)
        self.assertIn('if (!output || !graph)',header)
        with self.assertRaises(ValueError):
            study.patch_controller(fixed)

    def test_residency_server_runtime_qualification_preserves_original_prompts(self):
        import importlib
        import inspect
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-server-20260922.run')
        source=study.ordered_source(study.QUALIFICATION_ORDER)
        compile(source,'qualification-workload','exec')
        self.assertEqual(study.QUALIFICATION_ORDER,(0,1,10,9))
        original=inspect.getsource(study.ORIGINAL_RUN)
        self.assertEqual(source.replace('        assert len(workloads)==14\n        for index in '+repr(study.QUALIFICATION_ORDER)+':\n            kind,has_image=workloads[index]',
            '        for index,(kind,has_image) in enumerate(workloads):'),original)
        off=study.environment('baseline');on=study.environment('mixed_fill')
        self.assertNotIn('LOCAL_DESK_ENCODER_RECLAIM',off)
        self.assertNotIn('LOCAL_DESK_RESIDENCY_VALIDATE',on)
        self.assertEqual(study.environment('mixed_fill',True)['LOCAL_DESK_RESIDENCY_VALIDATE'],'1')
        with self.assertRaises(ValueError):
            study.activation('','mixed_fill')

    def test_residency_server_request_and_timed_validation_contract(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-server-20260922.study')
        clip=(study.VISION.WORK/'source/tools/mtmd/clip.cpp').read_text(encoding='utf-8')
        changed=study.patch_clip(clip)
        self.assertIn('llama_residency_before_image()',changed)
        self.assertIn('llama_residency_vision_released()',changed)
        self.assertIn('validation_readback=0',changed)
        self.assertNotIn('WaitForSingleObject(event',changed)
        mtmd=(study.SCOPED.WORK/'source/tools/mtmd/mtmd.cpp').read_text(encoding='utf-8')
        self.assertIn('status==0 && n_bitmaps==0',study.patch_mtmd(mtmd))
        transfer=(study.SCOPED.WORK/'source/include/storage-transfer.h').read_text()
        self.assertIn('if (!copy && !verify) return;',study.patch_transfer(transfer))
        self.assertIn('actual(verify ? chunk : 0)',study.patch_transfer(transfer))

    def test_residency_exact_scope_excludes_original_cuda_weights(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-final-20260922.scoped_runtime')
        source=study.DRAIN.WORK/'source/src'
        context=study.patch_context((source/'llama-context.cpp').read_text(encoding='utf-8'))
        graph=study.patch_graph((source/'llama-graph.cpp').read_text(encoding='utf-8'))
        self.assertIn('!residency::exact_weight(weight)',graph)
        self.assertIn('residency::exact_weights=std::move(selected_weights)',context)
        self.assertIn('residency::exact_weights.clear()',context)
        self.assertIn('residency::preview_scope exact_preview',context)
        self.assertLess(context.index('transaction->activate(generation)'),context.index('residency::exact_weights=std::move(selected_weights)'))

    def test_residency_selected_native_state_scope(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-final-20260922.integration')
        source=(study.NATIVE.HOME/'native-runtime.inc').read_text()
        patched=study.selected_runtime(source)
        self.assertIn('llama_research_inventory_selected',patched)
        self.assertIn('if (selected.ffn(layer)) {',patched)
        self.assertIn('if (weights!=3)',patched)
        self.assertIn('if (selected.layer(layer)) originals.emplace_back',patched)
        self.assertIn('selected.validate(cpu_test,model.hparams.n_layer())',patched)
        test=study.selected_test((study.NATIVE.HOME/'layer-context-test.cpp').read_text())
        self.assertIn('selected_layers=12;selected_ffns=3',test)
        with self.assertRaises(ValueError):
            study.selected_runtime(patched)

    def test_layer_exact_graph_keeps_compressed_source_and_fp32_input(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-layer-exact-20260922.study')
        original=(study.NATIVE.WORK/'source/src/llama-graph.cpp').read_text(encoding='utf-8')
        changed=study.patch_graph(original)
        self.assertIn('ggml_get_rows(context,weight,indices)',changed)
        self.assertIn('ggml_mul_mat_set_prec(tile,GGML_PREC_F32)',changed)
        self.assertIn('first+=128',changed)
        self.assertIn('strcmp(ggml_backend_dev_name(device),"CUDA0")',changed)
        self.assertIn('research_compressed_fp32_mm(ctx0, w, cur)',changed)
        with self.assertRaises(ValueError):
            study.patch_graph(changed)

    def test_residency_final_four_arms_and_disjoint_maximum_fill(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-residency-final-20260922.study')
        cost=study.RuntimeCost(64,32,16,0,0,'a'*64)
        ffns=[dict(block=52,bytes=100),dict(block=53,bytes=100),dict(block=56,bytes=120)]
        layers=[dict(block=52,additional_weight_bytes=170,additional_state_bytes=30,state_kind='recurrent'),
            dict(block=53,additional_weight_bytes=180,additional_state_bytes=30,state_kind='recurrent')]
        current=study.GUARD.DEDICATED_CAP-cost.total-430
        self.assertEqual(study.protocol()['arms'],list(study.ARMS))
        self.assertEqual(study.protocol()['total_planned_responses'],56)
        baseline=study.select('baseline',ffns,layers,current,cost)
        self.assertEqual(baseline['operational_total_bytes'],0)
        plain=study.select('whole_ffn_fill',ffns,layers,current,cost)
        self.assertEqual(plain['resident_bytes'],320)
        whole=study.select('whole_layer_fill',ffns,layers,current,cost)
        self.assertEqual(whole['whole_layers'],[52,53])
        mixed=study.select('mixed_fill',ffns,layers,current,cost)
        self.assertEqual(mixed['resident_bytes'],430)
        self.assertEqual(mixed['whole_layers'],[53])
        self.assertEqual(mixed['ffns'],[52,56])
        self.assertFalse(set(mixed['whole_layers']) & set(mixed['ffns']))
        self.assertEqual(mixed['verification_only_device_bytes'],0)
        self.assertFalse(study.select('mixed_fill',ffns,layers,study.GUARD.DEDICATED_CAP,cost)['executable_selection'])
        with self.assertRaises(ValueError):
            study.RuntimeCost(64,32,16,0,0,'unmeasured')

    def test_whole_layer_v2_native_inventory_boundary_contract(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-layer-native-20260922.study')
        source=(study.PRIOR.NATIVE_WORK/'source/src/llama-context.cpp').read_text(encoding='utf-8')
        changed=study.patch_context(source)
        self.assertIn('llama_research_inventory(',changed)
        self.assertIn('entry->lease.permits(generation)',changed)
        self.assertIn('research_storage(layer)',changed)
        self.assertIn('recurrent->r_l.at(layer)',changed)
        self.assertIn('transaction->activate(generation)',changed)
        section=changed[changed.index('llm_graph_result * llama_context::process_ubatch('):]
        self.assertLess(section.index('research_handoff_poll'),section.index('synchronize();'))
        self.assertLess(section.index('ggml_backend_sched_reset'),section.index('research_handoff_paused'))
        self.assertLess(section.index('research_handoff_paused'),section.index('mctx->apply()'))
        with self.assertRaises(ValueError):
            study.patch_context(changed)

    def test_whole_layer_native_boundary_waits_before_memory_apply(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-whole-layer-20260922.study')
        source=(study.ROOT/'workspace/vk-gcn/source/llama.cpp-b11000/src/llama-context.cpp').read_text(encoding='utf-8')
        changed=study.patch_native_context(source)
        section=changed[changed.index('llm_graph_result * llama_context::process_ubatch('):]
        self.assertLess(section.index('LLAMA_RESEARCH_POLL_COPY'),section.index('synchronize();'))
        self.assertLess(section.index('ggml_backend_sched_reset'),section.index('LLAMA_RESEARCH_PAUSED_BOUNDARY'))
        self.assertLess(section.index('LLAMA_RESEARCH_PAUSED_BOUNDARY'),section.index('mctx->apply()'))
        self.assertIn('if (ready < 0) { ret = GGML_STATUS_FAILED; return nullptr; }',section)
        with self.assertRaises(ValueError):
            study.patch_native_context(changed)

    def _whole_layer_handoff_fixture(self):
        import hashlib
        import importlib
        from unittest.mock import Mock
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-whole-layer-20260922.study')
        tensors=tuple((f'blk.52.{name}',b'original',hashlib.sha256(b'original').hexdigest())
            for name in ('ffn_up.weight','ffn_gate.weight','ffn_down.weight','attn_norm.weight','post_attention_norm.weight','ssm_conv1d.weight'))
        image=study.LayerImage((52,),'a'*64,tensors)
        budget=study.layer_budget([dict(block=52,additional_weight_bytes=image.nbytes,
            additional_state_bytes=study.MIB,state_kind='recurrent')],64*study.MIB,128*study.MIB)
        row=dict(at_ms=1000,pid=42,creation_time=1,available_ram_bytes=8*1024**3,
            adapter_dedicated={study.GUARD.RTX:6500*study.MIB},
            process_shared={f'pid_42_{study.GUARD.RTX}':174*study.MIB})
        state=dict(copy_ready=False,revision=1,owner='Vulkan0')
        backend=Mock()
        backend.validate_complete_inventory.return_value=True
        backend.enqueue_original_weights.return_value=('weights','copy_event')
        backend.query.side_effect=lambda event:state['copy_ready'] if event=='copy_event' else True
        backend.verify_original_weights.return_value=True
        def pause(request,blocks,device):
            self.assertEqual(state['owner'],device)
            return study.Boundary(request,blocks,device,state['revision'],((0,state['revision']),),object())
        def capture(boundary):
            payload=str(state['revision']).encode()
            return study.StateImage(boundary,payload,hashlib.sha256(payload).hexdigest())
        def commit(boundary,handle,snapshot,destination):
            self.assertIs(snapshot.boundary,boundary)
            self.assertEqual(snapshot.payload,str(state['revision']).encode())
            state['owner']=destination
            return True
        backend.pause_and_fence.side_effect=pause
        backend.capture_current_state.side_effect=capture
        backend.enqueue_state_restore.return_value='restore_event'
        backend.verify_restored_state.return_value=True
        backend.commit_binding.side_effect=commit
        handoff=study.LayerHandoff(backend,lambda:row,42,1,clock=lambda:1000)
        return study,handoff,backend,image,budget,row,state

    def test_whole_layer_handoff_keeps_compute_running_until_copy_ready(self):
        study,handoff,backend,image,budget,row,state=self._whole_layer_handoff_fixture()
        epoch=handoff.prefetch('request',image,budget,999)
        self.assertFalse(handoff.poll())
        backend.pause_and_fence.assert_not_called()
        backend.capture_current_state.assert_not_called()
        with self.assertRaises(RuntimeError):
            handoff.activate('request',epoch)
        state.update(copy_ready=True,revision=7)
        self.assertTrue(handoff.poll())
        handoff.activate('request',epoch)
        self.assertEqual(handoff.phase,'rtx')
        snapshot=backend.enqueue_state_restore.call_args.args[1]
        self.assertEqual(snapshot.payload,b'7')
        calls=[call[0] for call in backend.mock_calls]
        self.assertLess(calls.index('verify_original_weights'),calls.index('pause_and_fence'))
        self.assertLess(calls.index('wait'),calls.index('verify_restored_state'))
        self.assertLess(calls.index('verify_restored_state'),calls.index('commit_binding'))
        self.assertLess(calls.index('commit_binding'),calls.index('resume'))
        with self.assertRaises(RuntimeError):
            handoff.authorize_vision(850*study.MIB)
        state['revision']=12
        handoff.return_for_vision('request')
        self.assertEqual(backend.enqueue_state_restore.call_args.args[1].payload,b'12')
        self.assertEqual(state['owner'],'Vulkan0')
        self.assertEqual(handoff.phase,'rx')
        self.assertEqual(backend.capture_current_state.call_count,2)
        self.assertTrue(handoff.authorize_vision(850*study.MIB))
        stamps=[event['at_ns'] for event in handoff.events]
        self.assertEqual(stamps,sorted(stamps))
        self.assertEqual(sum(event['kind']=='latest_state_captured' for event in handoff.events),2)

    def test_whole_layer_handoff_cancelled_upload_cannot_switch(self):
        study,handoff,backend,image,budget,row,state=self._whole_layer_handoff_fixture()
        epoch=handoff.prefetch('request',image,budget,999)
        handoff.cancel_pending()
        self.assertFalse(handoff.poll())
        backend.release.assert_not_called()
        state['copy_ready']=True
        self.assertFalse(handoff.poll())
        backend.release.assert_called_once_with('weights')
        backend.pause_and_fence.assert_not_called()
        with self.assertRaises(RuntimeError):
            handoff.activate('request',epoch)

    def test_whole_layer_handoff_rejects_old_snapshot_without_resume(self):
        study,handoff,backend,image,budget,row,state=self._whole_layer_handoff_fixture()
        old_boundary=study.Boundary('request',(52,),'Vulkan0',0,((0,0),),object())
        old_snapshot=backend.capture_current_state(old_boundary)
        backend.capture_current_state.side_effect=None
        backend.capture_current_state.return_value=old_snapshot
        epoch=handoff.prefetch('request',image,budget,999)
        state.update(copy_ready=True,revision=8)
        self.assertTrue(handoff.poll())
        with self.assertRaisesRegex(RuntimeError,'current paused boundary'):
            handoff.activate('request',epoch)
        self.assertEqual(handoff.phase,'fault')
        backend.commit_binding.assert_not_called()
        backend.resume.assert_not_called()
        backend.release.assert_not_called()

    def test_whole_layer_handoff_shared_fault_latches(self):
        study,handoff,backend,image,budget,row,state=self._whole_layer_handoff_fixture()
        epoch=handoff.prefetch('request',image,budget,999)
        state['copy_ready']=True
        self.assertTrue(handoff.poll())
        row['process_shared'][f'pid_42_{study.GUARD.RTX}']=study.GUARD.SHARED_CAP+1
        with self.assertRaises(RuntimeError):
            handoff.activate('request',epoch)
        row['process_shared'][f'pid_42_{study.GUARD.RTX}']=174*study.MIB
        with self.assertRaises(RuntimeError):
            handoff.activate('request',epoch)
        backend.pause_and_fence.assert_not_called()
        backend.resume.assert_not_called()

    def test_whole_layer_fill_accounts_for_state_and_contiguity(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-whole-layer-20260922.study')
        entries=[dict(block=block,additional_weight_bytes=150*study.MIB,
            additional_state_bytes=20*study.MIB,state_kind='recurrent') for block in (52,53,54,56,57)]
        chosen=study.select_layers(entries,7100*study.MIB,64*study.MIB,128*study.MIB)
        self.assertEqual([entry['block'] for entry in chosen['selected']],[52,53,54])
        self.assertEqual(chosen['budget']['state'],60*study.MIB)
        self.assertLessEqual(chosen['budget']['total'],750*study.MIB)
        missing=[dict(entries[0],additional_state_bytes=0)]
        with self.assertRaises(ValueError):
            study.layer_budget(missing,64*study.MIB,128*study.MIB)
        with self.assertRaisesRegex(ValueError,'full RX ownership'):
            study.layer_budget([dict(entries[0],block=55)],64*study.MIB,128*study.MIB)
        with self.assertRaises(ValueError):
            study.select_layers(entries,None,64*study.MIB,128*study.MIB)
        with self.assertRaises(RuntimeError):
            study.require_execution_evidence({'original_weight_hashes_verified':True})
        cases=study.comparison_cases()
        self.assertEqual([case['name'] for case in cases],['whole_ffn_fill','whole_layer_fill'])
        self.assertTrue(all(case['original_compressed_weights'] and not case['row_partition'] for case in cases))

    def test_whole_ffn_budget_and_transfer_boundary(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-whole-ffn-20260922.study')
        values=study.budget()
        self.assertEqual(values['weights'],106536960)
        self.assertLessEqual(values['reservation']+values['headroom'],970*1024**2)
        self.assertLess(values['managed'],130*1024**2)
        self.assertEqual(values['h2d_bytes_per_token'],5120*4)
        self.assertEqual(values['d2h_bytes_per_token'],5120*4)
        self.assertFalse(values['row_partition'])
        self.assertFalse(values['host_merge'])
        self.assertFalse(values['weights_expanded'])

    def test_whole_ffn_fill_uses_only_complete_compressed_groups(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-whole-ffn-20260922.study')
        entries=[dict(block=block,bytes=study.FIRST_FFN_BYTES) for block in study.BLOCKS]
        roomy=study.select_ffns(entries,6500*study.MIB)
        self.assertEqual(len(roomy['selected']),7)
        self.assertLess(roomy['unused_weight_capacity_bytes'],study.FIRST_FFN_BYTES)
        self.assertLessEqual(roomy['budget']['reservation']+study.GUARD.HEADROOM,study.SUPPORT_CAP)
        tight=study.select_ffns(entries,7500*study.MIB)
        self.assertEqual(len(tight['selected']),1)
        self.assertEqual(study.select_ffns(entries,7850*study.MIB)['selected'],[])
        with self.assertRaises(ValueError):
            study.select_ffns(entries,None)
        with self.assertRaises(ValueError):
            study.select_ffns(entries+[entries[0]],6500*study.MIB)

    def test_text_swap_requires_successful_media_free_request(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-text-swap-20260922.study')
        self.assertTrue(study.text_only_request(0,0))
        self.assertFalse(study.text_only_request(1,0))
        self.assertFalse(study.text_only_request(0,2))
        self.assertFalse(study.text_only_request(2,0))
        for count in (None,-1,False):
            with self.assertRaises(ValueError):
                study.text_only_request(count,0)
        source='        mtmd_tokenizer tokenizer(ctx, text, bitmaps, n_bitmaps);\n        return tokenizer.tokenize(output);'
        changed=study.patch_request(source)
        self.assertIn('status == 0 && n_bitmaps == 0 && ctx->ctx_v',changed)
        self.assertLess(changed.index('tokenizer.tokenize(output)'),changed.index('clip_cooperative_request_text'))

    def test_text_swap_source_boundaries_and_multi_cache(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-text-swap-20260922.study')
        clip=study.PRIOR.WORK/'vision-source/tools/mtmd/clip.cpp'
        changed=study.patch_clip(clip.read_text(encoding='utf-8'))
        self.assertIn('if (!ctx || !ctx->buf) return 1;',changed)
        self.assertIn('text_swap_pending=ctx->buf ? ctx : nullptr;',changed)
        self.assertIn('text_swap_detach(this);',changed)
        encoding=changed.index('bool clip_encode(')
        self.assertLess(changed.index('lock(text_swap_mutex)',encoding),changed.index('research_restore_weights(ctx)',encoding))
        wrapper=(study.HOME/'wrapper.inc').read_text()
        self.assertIn('api.submit(key,input.data(),tokens)',wrapper)
        self.assertIn('api.finish(key,gpu_output.data(),tokens)',wrapper)
        self.assertIn('segment.n_nodes=index-first+1',wrapper)
        self.assertEqual((study.RX_ROWS,study.RTX_ROWS),(1280,3840))
        self.assertEqual(study.RTX_ROWS*7480*len(study.BLOCKS),201062400)
        with self.assertRaises(ValueError):
            study.patch_clip(changed)

    def test_text_swap_interleaving_preserves_original_indices(self):
        import importlib
        import inspect
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-text-swap-20260922.study')
        self.assertEqual(sorted(study.ORDER),list(range(14)))
        self.assertEqual(study.normalized_samples(list(study.ORDER)),list(range(14)))
        source=inspect.getsource(study.ORIGINAL_RUN_CANDIDATE)
        changed=study.patch_workload(source)
        compile(changed,'interleaved-research-workload','exec')
        self.assertEqual(changed.replace('        assert len(workloads)==14\n        for index in '+repr(study.ORDER)+':\n            kind,has_image=workloads[index]',
            '        for index,(kind,has_image) in enumerate(workloads):'),source)
        self.assertEqual(study.ORDER[-2:],(13,9))
        with self.assertRaises(ValueError):
            study.normalized_samples([0]*13)

    def test_cooperative_full_graph_split_keeps_waiting_on_rx(self):
        import importlib
        study=importlib.import_module('desktop_agent.data.benchmarks.rtx-cooperative-full-20260922.study')
        wrapper=(study.HOME/'wrapper.inc').read_text()
        self.assertIn('target<0 || !api.ready()',wrapper)
        self.assertLess(wrapper.index('api.submit('),wrapper.index('graph_compute_original(backend,&rx_graph)'))
        self.assertLess(wrapper.index('api.finish('),wrapper.index('ggml_backend_tensor_set(node'))
        self.assertIn('result.ne[0]=rx_rows',wrapper)
        self.assertIn('rx_rows=3840, rtx_rows=1280',wrapper)
        original=(study.VISION.WORK/'source/tools/mtmd/clip.cpp').read_text(encoding='utf-8')
        changed=study.patch_vision(original)
        self.assertIn('cooperative_vision_call("coop_vision_begin")',changed)
        self.assertIn('cooperative_vision_call("coop_vision_released")',changed)
        self.assertNotIn('WaitForSingleObject(event, 10000)',changed)
        lines=['COOP ram_source=GGUF bytes=9574400 sha256=verified rx_weight_reads=0',
            'VISION_RAM copy_cycle=1 host_bytes=625688384 copies=1',
            'COOP numerical_check=pass boundary_values=8 max_abs=1e-7', 'COOP rx_only=not_ready']
        sequence=0
        for epoch in range(4):
            if epoch:
                lines += [f'COOP vision_restore_authorized epoch={epoch} support_live_bytes=0',
                    f'COOP_VISION cycle={epoch+1} phase=before_restore',
                    f'VISION_RAM restore_cycle={epoch+1} restored_bytes=625688384 bitwise_equal=1']
            lines += [f'VISION_RAM parked_cycle={epoch+1} host_bytes=625688384 device_bytes=0',
                f'COOP vision_released epoch={epoch} schedule_prefetch=1',
                f'COOP prefetch_begin epoch={epoch} pinned_bytes=1048576',
                f'COOP prefetch_ready epoch={epoch} compressed_bytes=9574400 event_complete=1']
            for _ in range(2):
                sequence+=1
                lines += [f'COOP submitted epoch={epoch} tokens=128 rtx_rows=1280 cached=1',
                    f'COOP completed={sequence} epoch={epoch} tokens=128 event_complete=1',
                    'COOP split_complete tokens=128 rx_rows=3840 rtx_rows=1280 cuda_spans_rx=1']
        log='\n'.join(lines)
        evidence=study.lifecycle(log)
        self.assertEqual(evidence['reuse_computations'],4)
        self.assertEqual(evidence['cuda_event_spans_rx'],8)
        with self.assertRaises(ValueError):
            study.lifecycle(log.replace('prefetch_ready epoch=0','prefetch_ready epoch=8'))
        with self.assertRaises(ValueError):
            study.lifecycle(log.replace('bitwise_equal=1','bitwise_equal=0'))
        with self.assertRaises(ValueError):
            study.lifecycle(log.replace('cuda_spans_rx=1','cuda_spans_rx=0'))

    def test_fused_iq3_kernel_decoder_matches_official_cpu(self):
        import ctypes
        import importlib
        import numpy as np
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-fused-fp32-20260922.study')
        source = study.ROOT/'workspace/vk-gcn/source/llama.cpp-b11000/ggml/src/ggml-common.h'
        values = [int(value,16) for value in study.codebook(source.read_text(encoding='utf-8'))]
        generator = np.random.default_rng(121)
        raw = generator.integers(0,256,(16,110),dtype=np.uint8)
        raw[:,:2] = np.linspace(-0.02,0.02,16,dtype=np.float16).view(np.uint8).reshape(16,2)
        with os.add_dll_directory(str(study.RUNTIME)):
            core = ctypes.CDLL(str(study.RUNTIME/'ggml-base.dll'))
            expected = study.ASSIST.dequantize(raw,core).reshape(16,256)
        actual = study.cpu_decode(raw,values)
        np.testing.assert_array_equal(expected.view(np.uint32),actual.view(np.uint32))
        with self.assertRaises(ValueError):
            study.codebook('missing codebook')
        command,environment = study.worker_launch()
        self.assertTrue(study.is_diagnostic_command(command))
        self.assertEqual(environment['NVIDIA_TF32_OVERRIDE'],'0')

    def test_native_compressed_gate_does_not_relax_accuracy(self):
        import importlib
        import numpy as np
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-native-kernel-20260922.study')
        reference = np.ones((3,4),dtype=np.float64)
        result = study.compare(reference,np.ones((3,4),dtype=np.float32))
        self.assertTrue(result['passed'])
        changed = np.ones((3,4),dtype=np.float32)
        changed[0,0] += 1e-3
        result = study.compare(reference,changed)
        self.assertFalse(result['passed'])
        self.assertEqual(result['violating_elements'],1)
        command,environment = study.worker_launch()
        self.assertTrue(study.is_diagnostic_command(command))
        self.assertEqual(environment['NVIDIA_TF32_OVERRIDE'],'0')
        self.assertEqual(environment['GGML_CUDA_DISABLE_GRAPHS'],'1')
        state = dict(capacity_bytes=0,busy=False,disabled_reason='',pending_bytes=None,request=None,window=False)
        study.check_phase(5,dict(phase='work_complete',state=state))
        with self.assertRaises(ValueError):
            study.check_phase(5,dict(phase='work_complete',state=dict(state,busy=True)))

    def test_cooperative_cache_ram_source_is_original_file_bytes(self):
        import hashlib
        import importlib
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        import gguf
        module = importlib.import_module('desktop_agent.data.benchmarks.rtx-cooperative-cache-20260922.coordinator')
        raw = np.arange(4*110,dtype=np.int64).astype(np.uint8).reshape(4,110)
        tensor = SimpleNamespace(name='weights',shape=np.array([256,4]),tensor_type=SimpleNamespace(name='IQ3_S'),
            n_bytes=440,data=raw,data_offset=64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.gguf'
            path.write_bytes(b'metadata')
            with patch.object(gguf,'GGUFReader',return_value=SimpleNamespace(tensors=[tensor])),\
                    patch.object(module.psutil,'virtual_memory',return_value=SimpleNamespace(available=8*1024**3)):
                source,manifest = module.load_ram_weights(path,'weights',1,2,hashlib.sha256(raw).hexdigest())
                self.assertEqual(source.data,raw[1:3].tobytes())
                self.assertEqual(manifest['source'],'GGUF_FILE_TO_PRIVATE_RAM')
                self.assertEqual(manifest['bytes'],220)
                with self.assertRaises(ValueError):
                    module.load_ram_weights(path,'weights',1,2,'a'*64)
                with self.assertRaises(ValueError):
                    module.load_ram_weights(path,'weights',3,2,hashlib.sha256(raw).hexdigest())

    def test_cooperative_cache_partition_and_reclaim_evidence(self):
        import importlib
        from unittest.mock import Mock
        module = importlib.import_module('desktop_agent.data.benchmarks.rtx-cooperative-cache-20260922.coordinator')
        self.assertEqual(module.row_partition(5120,1280,False),dict(rx=(0,5120),rtx=None))
        split = module.row_partition(5120,1280,True)
        self.assertEqual(split,dict(rx=(0,3840),rtx=(3840,1280)))
        self.assertEqual(split['rx'][0]+split['rx'][1],split['rtx'][0])
        self.assertEqual(split['rtx'][0]+split['rtx'][1],5120)
        with self.assertRaises(ValueError):
            module.row_partition(5120,5120,True)
        row = dict(at_ms=999,pid=42,creation_time=7.0,available_ram_bytes=8*1024**3,
            adapter_dedicated={module.GUARD.RTX:6000*module.MIB},process_shared={f'pid_42_{module.GUARD.RTX}':174*module.MIB})
        cache = module.CooperativeCache(Mock(),lambda: row,42,7.0,clock=lambda:1000)
        with self.assertRaisesRegex(RuntimeError,'predates'):
            cache.confirm_vision_released(0,1000)
        self.assertEqual(cache.state,'vision')
        with self.assertRaisesRegex(RuntimeError,'stale'):
            cache.confirm_vision_released(0,499)
        row['at_ms'] = 1000
        cache.confirm_vision_released(0,1000)
        self.assertEqual(cache.state,'available')
        for field,value in (('weights',-1),('workspace',0.5),('host_new',None)):
            arguments = dict(weights=1,workspace=1,io=1,library=1)
            arguments[field] = value
            with self.assertRaises(ValueError):
                module.Budget(**arguments)

    def test_cooperative_cache_caps_cancellation_and_stale_leases(self):
        import hashlib
        import importlib
        from unittest.mock import Mock
        module = importlib.import_module('desktop_agent.data.benchmarks.rtx-cooperative-cache-20260922.coordinator')
        def setup():
            backend = Mock()
            backend.enqueue_from_ram.return_value = ('same-address','upload')
            completed = set()
            backend.query.side_effect = lambda event: event in completed
            row = dict(at_ms=1000,pid=42,creation_time=7.0,available_ram_bytes=8*1024**3,
                adapter_dedicated={module.GUARD.RTX:6000*module.MIB},process_shared={f'pid_42_{module.GUARD.RTX}':module.GUARD.SHARED_CAP})
            cache = module.CooperativeCache(backend,lambda: row,42,7.0,clock=lambda:1000)
            data = b'original-compressed-weights'
            weights = module.Weights('a'*64,'blk.52.ffn_down.weight',0,32,'IQ3_S',data,hashlib.sha256(data).hexdigest())
            cache.begin_request('one')
            cache.confirm_vision_released(0,1000)
            return cache,backend,row,weights,completed
        cache,backend,row,weights,completed = setup()
        too_big = module.Budget(module.TARGET_BYTES-module.HEADROOM_BYTES+1,0,0,0)
        self.assertFalse(cache.prefetch(weights,too_big))
        exact = module.Budget(module.TARGET_BYTES-module.HEADROOM_BYTES,0,0,0)
        self.assertTrue(cache.prefetch(weights,exact))
        cache.request_vision()
        self.assertIsNone(cache.choose(weights))
        self.assertFalse(cache.allow_vision_restore(600*module.MIB))
        backend.release.assert_not_called()
        completed.add('upload')
        self.assertTrue(cache.allow_vision_restore(600*module.MIB))
        self.assertEqual(cache.state,'vision')
        backend.release.assert_called_once()
        cache,backend,row,weights,completed = setup()
        budget = module.Budget(64*module.MIB,64*module.MIB,32*module.MIB,96*module.MIB)
        row['adapter_dedicated'][module.GUARD.RTX] = module.GUARD.DEDICATED_CAP-budget.device-module.HEADROOM_BYTES+1
        self.assertFalse(cache.prefetch(weights,budget))
        row['adapter_dedicated'][module.GUARD.RTX] -= 1
        self.assertTrue(cache.prefetch(weights,budget))
        completed.add('upload')
        lease = cache.choose(weights)
        cache.complete(lease,'first')
        completed.add('first')
        cache.poll()
        second = cache.choose(weights)
        with self.assertRaises(RuntimeError):
            cache.complete(lease,'stale')
        cache.complete(second,'second')
        row['process_shared'][f'pid_42_{module.GUARD.RTX}'] += 1
        cache.poll()
        self.assertTrue(cache.request_disabled)
        backend.release.assert_not_called()
        completed.add('second')
        cache.poll()
        backend.release.assert_called_once()
        self.assertEqual(cache.reserved,0)
        row['process_shared'][f'pid_42_{module.GUARD.RTX}'] -= 1
        self.assertIsNone(cache.choose(weights))
        self.assertFalse(cache.prefetch(weights,budget))
        cache,backend,row,weights,completed = setup()
        self.assertTrue(cache.prefetch(weights,budget))
        backend.query.side_effect = RuntimeError('unknown completion')
        cache.poll()
        self.assertTrue(cache.fault)
        backend.release.assert_not_called()
        cache.request_vision()
        self.assertFalse(cache.allow_vision_restore(0))

    def test_cooperative_cache_prefetch_and_vision_handoff(self):
        import hashlib
        import importlib
        from unittest.mock import Mock
        module = importlib.import_module('desktop_agent.data.benchmarks.rtx-cooperative-cache-20260922.coordinator')
        backend = Mock()
        backend.enqueue_from_ram.return_value = ('allocation','upload')
        completed = set()
        backend.query.side_effect = lambda event: event in completed
        sample = dict(at_ms=1000,pid=42,creation_time=7.0,available_ram_bytes=8*1024**3,
            adapter_dedicated={module.GUARD.RTX:6500*module.MIB},process_shared={f'pid_42_{module.GUARD.RTX}':174*module.MIB})
        cache = module.CooperativeCache(backend,lambda: sample,42,7.0,clock=lambda:1000)
        data = b'compressed-weights'
        weights = module.Weights('a'*64,'blk.52.ffn_down.weight',0,32,'IQ3_S',data,hashlib.sha256(data).hexdigest())
        budget = module.Budget(weights=64*module.MIB,workspace=128*module.MIB,io=64*module.MIB,library=96*module.MIB)
        cache.begin_request('one')
        self.assertFalse(cache.prefetch(weights,budget))
        cache.confirm_vision_released(0,1000)
        self.assertTrue(cache.prefetch(weights,budget))
        self.assertIsNone(cache.choose(weights))
        backend.release.assert_not_called()
        completed.add('upload')
        lease = cache.choose(weights)
        self.assertIsNotNone(lease)
        epoch = cache.request_vision()
        self.assertIsNone(cache.choose(weights))
        self.assertFalse(cache.allow_vision_restore(850*module.MIB))
        cache.complete(lease,'compute')
        self.assertFalse(cache.allow_vision_restore(850*module.MIB))
        completed.add('compute')
        self.assertTrue(cache.allow_vision_restore(850*module.MIB))
        backend.release.assert_called_once_with('allocation')
        self.assertEqual(cache.reserved,0)
        with self.assertRaises(RuntimeError):
            cache.confirm_vision_released(0,1000)
        cache.confirm_vision_released(epoch,1000)
        cache.end_request()
        backend.synchronize.assert_not_called()

    def test_full_assist_wrapper_preserves_off_and_fallback(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-full-compare-20260922.study')
        original = 'static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) { return GGML_STATUS_SUCCESS; }\n// Sort the graph for improved parallelism.'
        helper = (study.HOME/'assist.inc').read_text()
        wrapper = (study.HOME/'wrapper.inc').read_text()
        changed = study.patch_host(original,helper,wrapper)
        self.assertIn('graph_compute_original(',changed)
        self.assertIn('target+(assisted ? 1 : 0)',changed)
        self.assertIn('ggml_backend_vk_synchronize(backend);',changed)
        self.assertIn('node->src[1]->ne[1]>=64',changed)
        self.assertIn('node->src[1]->ne[1]<=128',changed)
        self.assertIn('desk_disabled = true;',changed)
        self.assertLess(changed.index('if (!enabled'),changed.index('desk_compute(graph->nodes[target])'))
        kinds = ['warmup','text']+['quality']*8+['vision_quality']*4
        samples = [dict(kind=kind,passed=True,input_sha256=str(index),answer_sha256='answer',usage={'tokens':1},tool='finish',
            seconds=100 if index==0 else 1,timings=dict(prompt_ms=400,predicted_ms=600)) for index,kind in enumerate(kinds)]
        candidate = [dict(sample,seconds=sample['seconds']*2) for sample in samples]
        report = study.compare_samples(samples,candidate)
        self.assertEqual(report['arms']['off']['total_seconds'],13)
        self.assertEqual(report['change_percent']['total_seconds'],100)
        candidate[4]['input_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError,'input_sha256'):
            study.compare_samples(samples,candidate)

    def test_rtx_compressed_layout_and_original_bytes(self):
        import importlib
        import numpy as np
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-assist-compressed-20260922.study')
        self.assertEqual(study.layout(128,17408,128),dict(weights=957440,input=8912896,output=65536,
            decoded=8912896,indices_offset=957440,packed_block=957952))
        tail = study.layout(129,17408,128)
        self.assertEqual(tail['weights'],964920)
        self.assertEqual(tail['indices_offset']%256,0)
        self.assertGreaterEqual(tail['indices_offset'],tail['weights'])
        with self.assertRaises(ValueError):
            study.layout(128,17409,128)
        with self.assertRaises(ValueError):
            study.layout(128,17408,256)
        original = np.arange(110,dtype=np.uint8)
        study.validate_roundtrip(original,original.copy())
        changed = original.copy()
        changed[-1] ^= 1
        with self.assertRaises(ValueError):
            study.validate_roundtrip(original,changed)
        command,environment = study.worker_launch()
        self.assertTrue(study.is_diagnostic_command(command))
        self.assertEqual(environment['NVIDIA_TF32_OVERRIDE'],'0')
        state = dict(capacity_bytes=0,busy=False,disabled_reason='',pending_bytes=None,request=None,window=False)
        study.check_phase(7,dict(phase='work_complete',state=state))
        with self.assertRaises(ValueError):
            study.check_phase(7,dict(phase='work_complete',state=dict(state,request='active')))

    def test_rtx_reuse_invalidates_same_address_after_reallocation(self):
        import importlib
        from types import SimpleNamespace
        from unittest.mock import Mock
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-assist-reuse-20260922.study')
        handles = (10,20,30,40)
        pool = SimpleNamespace(busy=True,disabled_reason='',window=True,request='one',backend=object(),
            blocks=[(handle,study.BUFFER.BLOCK_BYTES) for handle in handles],events=[dict(kind='allocate') for _ in handles])
        cache = study.ResidentWeights()
        load = Mock(return_value=dict(dequantize_us=1,weight_upload_us=2))
        self.assertFalse(cache.ensure(pool,handles,'a'*64,load)['reused'])
        self.assertTrue(cache.ensure(pool,handles,'a'*64,load)['reused'])
        self.assertEqual(load.call_count,1)
        pool.events += [dict(kind='free') for _ in handles]+[dict(kind='allocate') for _ in handles]
        self.assertFalse(cache.ensure(pool,handles,'a'*64,load)['reused'])
        self.assertEqual(load.call_count,2)
        self.assertFalse(cache.ensure(pool,handles,'b'*64,load)['reused'])
        pool.request = 'two'
        self.assertFalse(cache.ensure(pool,handles,'b'*64,load)['reused'])
        replacement = SimpleNamespace(**pool.__dict__)
        self.assertFalse(cache.ensure(replacement,handles,'b'*64,load)['reused'])
        load.side_effect = MemoryError('transfer failed')
        with self.assertRaises(MemoryError):
            cache.ensure(pool,handles,'c'*64,load)
        self.assertIsNone(cache.residency)
        pool.busy = False
        with self.assertRaises(RuntimeError):
            cache.ensure(pool,handles,'a'*64,load)
        command,environment = study.worker_launch()
        self.assertTrue(study.is_diagnostic_command(command))
        self.assertEqual(environment['NVIDIA_TF32_OVERRIDE'],'0')
        state = dict(capacity_bytes=0,busy=False,disabled_reason='',pending_bytes=None,request=None,window=False)
        study.check_phase(11,dict(phase='work_complete',state=state))
        with self.assertRaises(ValueError):
            study.check_phase(11,dict(phase='work_complete',state=dict(state,request='live')))
        calls = []
        for index in range(4):
            if index in (0,3):
                calls += [dict(kind='blas_created',math_mode=2)]+[dict(kind='allocate',bytes=study.BUFFER.BLOCK_BYTES) for _ in range(4)]
                calls.append(dict(kind='upload',role='weights',batch=index,bytes=study.ROWS*study.COLUMNS*4,weight_key='a'*64))
            calls += [dict(kind='upload',role='input',batch=index,bytes=study.TOKENS*study.COLUMNS*4),
                dict(kind='gemm',rows=study.ROWS,tokens=study.TOKENS,columns=study.COLUMNS),dict(kind='download',bytes=study.ROWS*study.TOKENS*4)]
            if index in (2,3):
                calls += [dict(kind='blas_destroyed')]+[dict(kind='free',bytes=study.BUFFER.BLOCK_BYTES) for _ in range(4)]
        calls.append(dict(kind='backend_closed'))
        events = [dict(kind=kind,generation=generation,key='a'*64,weight_handle=20)
            for kind,generation in zip(('load','reuse','reuse','load'),(4,4,4,8))]
        result = dict(allocator_calls=calls,residency_events=events)
        study.validate_reuse(result,'a'*64)
        events[-1]['kind'] = 'reuse'
        with self.assertRaises(ValueError):
            study.validate_reuse(result,'a'*64)

    def test_rtx_assist_row_tiles_and_output_ownership(self):
        import importlib
        import numpy as np
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-assist-matmul-20260922.study')
        self.assertEqual(study.row_tiles(257,128),[(0,128),(128,128),(256,1)])
        layout = study.buffer_layout(17408,128,128)
        self.assertEqual(layout['weights'],8912896)
        dimensions = study.gemm_dimensions(3,5,2)
        self.assertEqual(dimensions,dict(transpose_weights=1,transpose_input=0,rows=3,tokens=2,inner=5,
            weight_stride=5,input_stride=5,output_stride=3))
        weights = np.arange(15,dtype=np.float32).reshape(3,5)
        inputs = np.arange(10,dtype=np.float32).reshape(2,5)
        interpreted = weights.ravel().reshape((5,3),order='F').T @ inputs.ravel().reshape((5,2),order='F')
        np.testing.assert_array_equal(interpreted,weights @ inputs.T)
        with self.assertRaises(ValueError):
            study.buffer_layout(17408,128,257)
        output = np.zeros((5,3),np.float32)
        covered = np.zeros(5,bool)
        study.merge_rows(output,covered,3,np.ones((2,3),np.float32))
        study.merge_rows(output,covered,0,np.ones((3,3),np.float32))
        self.assertTrue(covered.all())
        self.assertTrue(study.accuracy(np.ones((5,3),np.float64),output)['passed'])
        with self.assertRaises(ValueError):
            study.merge_rows(output,covered,1,np.ones((1,3),np.float32))
        output[4,2] = 2
        self.assertFalse(study.accuracy(np.ones((5,3),np.float64),output)['passed'])
        command,environment = study.worker_launch()
        self.assertTrue(study.is_diagnostic_command(command))
        self.assertEqual(environment['NVIDIA_TF32_OVERRIDE'],'0')
        self.assertTrue(study.is_diagnostic_command(['python','-m','desktop_agent.data.benchmarks.rtx-dynamic-buffer-20260922.study','worker']))
        state = dict(capacity_bytes=0,busy=False,disabled_reason='',pending_bytes=None,request=None,window=False)
        study.check_phase(8,dict(phase='work_complete',state=state))
        with self.assertRaises(ValueError):
            study.check_phase(8,dict(phase='work_complete',state=dict(state,busy=True)))
        calls = [dict(kind='blas_created',math_mode=2)]
        calls += [dict(kind='allocate',bytes=study.BUFFER.BLOCK_BYTES) for _ in range(4)]
        calls += [dict(kind='upload',bytes=study.TOKENS*study.COLUMNS*4)]
        for count in (128,128,1):
            calls += [dict(kind='upload',bytes=count*study.COLUMNS*4),dict(kind='gemm',rows=count),dict(kind='download',bytes=count*study.TOKENS*4)]
        calls += [dict(kind='blas_destroyed')]+[dict(kind='free',bytes=study.BUFFER.BLOCK_BYTES) for _ in range(4)]+[dict(kind='backend_closed')]
        study.validate_calls(calls)
        calls.remove(dict(kind='blas_destroyed'))
        calls.insert(len(calls)-1,dict(kind='blas_destroyed'))
        with self.assertRaisesRegex(ValueError,'Workspace freed'):
            study.validate_calls(calls)

    def test_dynamic_buffer_diagnostic_teardown_requires_finished_work(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-dynamic-buffer-20260922.study')
        module = 'desktop_agent.data.benchmarks.rtx-dynamic-buffer-20260922.study'
        self.assertTrue(study.is_diagnostic_command(['python','-m',module,'worker']))
        self.assertTrue(study.is_diagnostic_command(['python','-u','-m',module,'measure']))
        self.assertFalse(study.is_diagnostic_command(['python','-c','inspect '+module]))
        self.assertFalse(study.is_diagnostic_command(['python','-m',module,'audit']))
        original_path = os.environ.get('PATH','')
        command,environment = study.worker_launch()
        self.assertEqual(command[0],study.sys._base_executable)
        self.assertTrue(study.is_diagnostic_command(command))
        self.assertEqual(environment['PATH'],str(study.RUNTIME)+os.pathsep+original_path)
        self.assertEqual(os.environ.get('PATH',''),original_path)
        self.assertIn(os.path.normcase(str(Path(study.psutil.__file__).parent.parent)),
            [os.path.normcase(path) for path in environment['PYTHONPATH'].split(os.pathsep)])
        state = dict(capacity_bytes=0,disabled_reason='',busy=False,pending_bytes=None,request=None,window=False)
        message = dict(phase='work_complete',state=state)
        study.check_phase(6,message)
        for change in (dict(capacity_bytes=16*1024**2),dict(busy=True),dict(pending_bytes=0),dict(request='live'),dict(window=True)):
            with self.assertRaises(ValueError):
                study.check_phase(6,dict(message,state=dict(state,**change)))
        with self.assertRaises(ValueError):
            study.check_phase(5,message)
        sample = dict(at_ms=1000,pid=42,creation_time=7.0,available_ram_bytes=4*1024**3,
            adapter_dedicated={study.GUARD.RTX:100*1024**2,study.GUARD.RX:0},process_shared={})
        with patch.object(study.time,'time',return_value=1.0):
            self.assertTrue(study.memory_reason(sample,42,7.0,require_process=True))
            self.assertEqual(study.memory_reason(sample,42,7.0,require_process=False),'')
            sample['process_shared'] = {f'pid_42_{study.GUARD.RTX}':study.GUARD.SHARED_CAP+1}
            self.assertTrue(study.memory_reason(sample,42,7.0,require_process=False))
            sample['process_shared'] = {}
            sample['adapter_dedicated'][study.GUARD.RTX] = study.GUARD.DEDICATED_CAP+1
            self.assertTrue(study.memory_reason(sample,42,7.0,require_process=False))
        sample['adapter_dedicated'][study.GUARD.RTX] = 100*1024**2
        sample['process_shared'] = {f'pid_42_{study.GUARD.RTX}':174*1024**2}
        sample['process_dedicated'] = {f'pid_42_{study.GUARD.RTX}':80*1024**2}
        points = []
        for phase,capacity in zip(study.PHASES,study.CAPACITIES):
            snapshot = dict(state,capacity_bytes=capacity,busy=phase=='busy_shrink',pending_bytes=32*1024**2 if phase=='busy_shrink' else None)
            points.append(dict(phase=phase,state=snapshot,samples=[sample.copy() for _ in range(3)]))
        calls = [dict(kind='allocate',bytes=study.BUFFER.BLOCK_BYTES) for _ in range(6)]
        calls += [dict(kind='async_fill'),dict(kind='synchronize')]
        calls += [dict(kind='free',bytes=study.BUFFER.BLOCK_BYTES) for _ in range(6)]+[dict(kind='backend_closed')]
        worker = dict(passed=True,deferred_shrink_verified=True,teardown_authorized=True,final_state=state,edge_patterns_verified=2,allocator_calls=calls)
        self.assertEqual(len(study.summarize(points,worker,42)),7)
        calls.remove(dict(kind='synchronize'))
        with self.assertRaisesRegex(ValueError,'before synchronization'):
            study.summarize(points,worker,42)

    def test_dynamic_buffer_sync_failure_preserves_live_buffers(self):
        import importlib
        from unittest.mock import Mock
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-dynamic-buffer-20260922.buffer')
        backend = Mock()
        backend.allocate.side_effect = lambda size: object()
        row = dict(at_ms=1000,pid=42,creation_time=7.0,available_ram_bytes=4*1024**3,
            adapter_dedicated={study.GUARD.RTX:7000*1024**2},process_shared={f'pid_42_{study.GUARD.RTX}':study.GUARD.SHARED_CAP})
        pool = study.DynamicBuffer(backend,lambda: row,42,7.0,clock=lambda:1000)
        pool.begin_request('sync-failure')
        self.assertTrue(pool.open_window())
        pool.resize(32*1024**2)
        pool.acquire()
        pool.resize(0)
        backend.synchronize.side_effect = RuntimeError('unfinished CUDA work')
        with self.assertRaisesRegex(RuntimeError,'unfinished CUDA work'):
            pool.complete()
        self.assertTrue(pool.busy)
        self.assertEqual(pool.capacity,32*1024**2)
        self.assertTrue(pool.disabled_reason)
        backend.free.assert_not_called()
        with self.assertRaises(RuntimeError):
            pool.end_request()
        backend.synchronize.side_effect = None
        pool.complete()
        self.assertEqual(pool.capacity,0)
        self.assertFalse(pool.busy)
        self.assertFalse(pool.open_window())
        pool.end_request()

    def test_dynamic_buffer_pressure_and_invalid_telemetry(self):
        import importlib
        from unittest.mock import Mock
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-dynamic-buffer-20260922.buffer')
        def setup():
            backend = Mock()
            backend.allocate.side_effect = lambda size: object()
            row = dict(at_ms=1000,pid=42,creation_time=7.0,available_ram_bytes=4*1024**3,
                adapter_dedicated={study.GUARD.RTX:7000*1024**2},process_shared={f'pid_42_{study.GUARD.RTX}':study.GUARD.SHARED_CAP})
            pool = study.DynamicBuffer(backend,lambda: row,42,7.0,clock=lambda:1000)
            pool.begin_request('request')
            self.assertTrue(pool.open_window())
            pool.resize(64*1024**2)
            return pool,backend,row
        pool,backend,row = setup()
        row['adapter_dedicated'][study.GUARD.RTX] = study.GUARD.DEDICATED_CAP-study.GUARD.HEADROOM+1
        with self.assertRaisesRegex(RuntimeError,'pressure'):
            pool.acquire()
        self.assertFalse(pool.busy)
        self.assertLess(pool.capacity,64*1024**2)
        self.assertEqual(pool.disabled_reason,'')
        row['adapter_dedicated'][study.GUARD.RTX] = 7000*1024**2
        pool.resize(64*1024**2)
        self.assertEqual(pool.capacity,64*1024**2)
        pool.end_request()
        changes = (dict(at_ms=499),dict(at_ms=1001),dict(pid=43),dict(creation_time=8.0),
            dict(available_ram_bytes=4*1024**3-1),dict(process_shared={}),dict(adapter_dedicated=None))
        for change in changes:
            with self.subTest(change=change):
                pool,backend,row = setup()
                row.update(change)
                pool.poll()
                self.assertTrue(pool.disabled_reason)
                self.assertEqual(pool.capacity,0)
                self.assertFalse(pool.open_window())
                pool.end_request()
        pool,backend,row = setup()
        backend.allocate.side_effect = MemoryError('allocation refused')
        pool.resize(80*1024**2)
        self.assertIn('allocation failed',pool.disabled_reason)
        self.assertEqual(pool.capacity,0)
        pool.end_request()

    def test_dynamic_buffer_safe_release_and_shared_latch(self):
        import importlib
        from unittest.mock import Mock
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-dynamic-buffer-20260922.buffer')
        backend = Mock()
        backend.allocate.side_effect = lambda size: object()
        row = dict(at_ms=1000,pid=42,creation_time=7.0,available_ram_bytes=8*1024**3,
            adapter_dedicated={study.GUARD.RTX:7000*1024**2},process_shared={f'pid_42_{study.GUARD.RTX}':184*1024**2})
        pool = study.DynamicBuffer(backend,lambda: row,42,7.0,clock=lambda:1000)
        pool.begin_request('first')
        self.assertTrue(pool.open_window())
        self.assertEqual(pool.resize(64*1024**2)['capacity_bytes'],64*1024**2)
        self.assertEqual(len(pool.acquire()),4)
        pool.resize(16*1024**2)
        backend.free.assert_not_called()
        pool.complete()
        self.assertEqual(pool.capacity,16*1024**2)
        self.assertEqual(backend.free.call_count,3)
        pool.acquire()
        row['process_shared'][f'pid_42_{study.GUARD.RTX}'] += 1
        pool.poll()
        self.assertIn('shared',pool.disabled_reason)
        self.assertEqual(backend.free.call_count,3)
        row['process_shared'][f'pid_42_{study.GUARD.RTX}'] -= 1
        pool.complete()
        self.assertEqual(pool.capacity,0)
        self.assertFalse(pool.open_window())
        self.assertEqual(pool.resize(64*1024**2)['capacity_bytes'],0)
        pool.end_request()
        with self.assertRaises(ValueError):
            pool.begin_request('first')
        pool.begin_request('second')
        self.assertTrue(pool.open_window())
        pool.end_request()

    def test_server_bridge_separates_cpp_and_context_ownership(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-server-bridge-20260922.study')
        cpp = study.CPP_PREFIX+'test'
        official = {cpp:dict(ordinal=1),'mtmd_free':dict(ordinal=2),'mtmd_helper_gen_audio_free':dict(ordinal=3)}
        research = {'mtmd_free':dict(ordinal=20)}
        result = study.routes({cpp,'mtmd_free'},official,research)
        self.assertEqual(result[cpp]['target'],'mtmd-official.'+cpp)
        self.assertEqual(result['mtmd_free']['target'],'libmtmd.mtmd_free')
        self.assertEqual(result['mtmd_free']['ordinal'],2)
        for required in ({cpp,'mtmd_helper_gen_audio_free'},{cpp,'unknown'},{'mtmd_free'}):
            with self.assertRaises(ValueError):
                study.routes(required,official,research)

    def test_vision_ram_parking_preserves_bytes_before_release(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-vision-ram-20260921.study')
        source = (study.HOME/'parking.inc').read_text()
        study.validate_parking(source)
        for invalid in (source.replace('ggml_backend_tensor_get','missing'),source+'\ncudaHostRegister',source.replace('memcmp(','missing(')):
            with self.assertRaises(ValueError):
                study.validate_parking(invalid)
        self.assertEqual(len(study.EXPECTED),19)
        self.assertEqual(study.EXPECTED[-1],(2,'released'))
        reserve = 300*1024**2
        row = dict(available_ram_bytes=study.PRIOR.AVAILABLE_RAM_MIN_BYTES+study.WEIGHT_BYTES+study.PRIOR.HEADROOM,
            adapter_dedicated={study.PRIOR.RTX:study.PRIOR.DEDICATED_CAP-study.WEIGHT_BYTES-reserve-study.PRIOR.HEADROOM})
        self.assertEqual(study.allocation_reason('before_host_copy',row,reserve),'')
        self.assertEqual(study.allocation_reason('before_restore',row,reserve),'')
        row['available_ram_bytes'] -= 1
        self.assertIn('RAM',study.allocation_reason('before_host_copy',row,reserve))
        row['adapter_dedicated'][study.PRIOR.RTX] += 1
        self.assertIn('dedicated',study.allocation_reason('before_restore',row,reserve))
        inputs = json.dumps(dict(controller_sha256={},runtime_manifest_sha256='hash'))
        with patch.object(study,'CONTEXT_VERIFY_RUNTIME',return_value='checked') as verify,patch.object(study.CONTEXT,'verify_runtime',study.verify_runtime),\
                patch.object(Path,'read_text',return_value=inputs),patch.object(study,'sha256',return_value='hash'):
            self.assertEqual(study.verify_runtime(),'checked')
            verify.assert_called_once_with()
        pid = 123
        key = f'pid_{pid}_{study.PRIOR.RTX}'
        points = []
        for cycle,phase in study.EXPECTED:
            resident = phase not in ('weights_parked','released','before_restore')
            sample = dict(at_ms=100,pid=pid,available_ram_bytes=4*1024**3,
                adapter_dedicated={study.PRIOR.RTX:(800 if resident else 200)*1024**2},
                process_dedicated={key:(780 if resident else 180)*1024**2},process_shared={key:174*1024**2})
            points.append(dict(cycle=cycle,phase=phase,weights_bytes=study.WEIGHT_BYTES if resident else 0,
                compute_bytes=128*1024**2 if phase in ('allocated','encoded') else 0,elapsed_us=1,samples=[sample.copy() for _ in range(3)]))
        report = study.summarize(points,pid)
        self.assertEqual(report[1]['weight_process_drop_bytes'],600*1024**2)
        points[-1]['weights_bytes'] = study.WEIGHT_BYTES
        with self.assertRaises(ValueError):
            study.summarize(points,pid)

    def test_vision_context_resume_only_incomplete_ram_interruption(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-vision-context-20260921.study')
        result = dict(passed=False,owned_process_exited=True,error='Available RAM below8GiB during research')
        study.validate_resume(result,8*1024**3,4*1024**3)
        for changed in (dict(result,passed=True),dict(result,owned_process_exited=False),dict(result,error='GPU cap')):
            with self.assertRaises(ValueError):
                study.validate_resume(changed,8*1024**3,4*1024**3)
        with self.assertRaises(ValueError):
            study.validate_resume(result,8*1024**3,3*1024**3)

    def test_vision_context_lifecycle_preserves_weights_and_limits(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-vision-context-20260921.study')
        helper = (study.HOME/'reclaim.inc').read_text()
        study.validate_helper(helper)
        for modified in (helper.replace('ctx->sched.reset();','ctx->buf.reset();'),
            helper.replace('ggml_backend_synchronize(ctx->backend);','cudaDeviceReset();')):
            with self.assertRaises(ValueError):
                study.validate_helper(modified)
        probe = 'options.image_min_tokens=1024; options.image_max_tokens=1024; parameters.vocab_only=true; iteration<2'
        study.validate_probe(probe)
        with self.assertRaises(ValueError):
            study.validate_probe(probe.replace('options.image_min_tokens=1024;',''))
        limits = 'load_hparams: image_min_pixels:   1048576 (custom value)\nload_hparams: image_max_pixels:   1048576 (custom value)'
        study.validate_image_limits(limits)
        with self.assertRaises(ValueError):
            study.validate_image_limits(limits.replace('image_min_pixels:   1048576','image_min_pixels:   8192'))
        mib = 1024**2
        pid = 123
        key = f'pid_{pid}_{study.PRIOR.RTX}'
        points = []
        for cycle in (1,2):
            for phase,used,compute in zip(study.PHASES,(700,900,1050,850,720,720),(0,200,200,0,0,0)):
                row = dict(at_ms=100,pid=pid,available_ram_bytes=4*1024**3,
                    adapter_dedicated={study.PRIOR.RTX:used*mib},process_dedicated={key:(used-20)*mib},process_shared={key:174*mib})
                points.append(dict(cycle=cycle,phase=phase,compute_bytes=compute*mib,elapsed_us=10,weights_bytes=600*mib,samples=[row.copy() for _ in range(3)]))
        report = study.summarize(points,pid)
        self.assertEqual(report[0]['scheduler_process_drop_bytes'],200*mib)
        self.assertEqual(report[0]['extra_backend_process_drop_bytes'],130*mib)
        with self.assertRaises(ValueError):
            study.summarize(points[:-1],pid)
        with self.assertRaises(ValueError):
            study.summarize(points,pid,available_ram_min_bytes=8*1024**3)
        points[4]['samples'][0]['available_ram_bytes'] = 4*1024**3-1
        with self.assertRaises(ValueError):
            study.summarize(points,pid)

    def test_rtx_reclaim_startup_ram_floor_and_concurrency(self):
        import importlib
        from types import SimpleNamespace
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-encoder-reclaim-20260921.study')
        with patch.object(study.psutil,'process_iter',return_value=[]) as processes,patch.object(study.psutil,'virtual_memory') as memory:
            for available in (8*1024**3-1,4*1024**3+1,4*1024**3):
                memory.return_value.available = available
                study.ensure_idle()
            memory.return_value.available = 4*1024**3-1
            with self.assertRaisesRegex(RuntimeError,'4GiB'):
                study.ensure_idle()
            memory.return_value.available = 4*1024**3
            for name in ('LLAMA-SERVER.EXE','encoder-reclaim-probe.exe','test-fa-ops.exe','test-fa-ops-v2.exe',
                'test-backend-ops.exe','test-ffn-iq4-case.exe','test-output-bounded-case.exe','test-mm-glu-case.exe'):
                with self.subTest(name=name):
                    processes.return_value = [SimpleNamespace(info=dict(name=name))]
                    with self.assertRaisesRegex(RuntimeError,'no concurrent experiment'):
                        study.ensure_idle()

    def test_rtx_reclaim_summary_requires_release_and_unchanged_weights(self):
        import copy
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-encoder-reclaim-20260921.study')
        points = []
        for cycle in (1,2):
            for phase in ('before_allocate','allocated','encoded','released'):
                allocated = phase in ('allocated','encoded')
                row = dict(at_ms=1000,pid=42,adapter_dedicated={study.RTX:(900 if allocated else 800)*study.MIB},
                    process_dedicated={f'pid_42_{study.RTX}':(880 if allocated else 780)*study.MIB},
                    process_shared={f'pid_42_{study.RTX}':184*study.MIB})
                points.append(dict(cycle=cycle,phase=phase,weights_bytes=600*study.MIB,
                    compute_bytes=100*study.MIB if allocated else 0,elapsed_us=1000,samples=[row.copy() for _ in range(3)]))
        summaries = study.summarize_checkpoints(points,42)
        self.assertEqual(summaries[0]['adapter_dedicated_drop_bytes'],100*study.MIB)
        self.assertEqual(summaries[1]['shared_change_bytes'],0)
        with self.assertRaises(ValueError):
            study.summarize_checkpoints(points[:-1],42)
        wrong = copy.deepcopy(points)
        wrong[-1]['weights_bytes'] -= 1
        with self.assertRaises(ValueError):
            study.summarize_checkpoints(wrong,42)
        wrong = copy.deepcopy(points)
        wrong[-1]['compute_bytes'] = 1
        with self.assertRaises(ValueError):
            study.summarize_checkpoints(wrong,42)

    def test_rtx_reclaim_runtime_monitor_enforces_new_shared_cap(self):
        import importlib
        import threading
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-encoder-reclaim-20260921.study')
        monitor = object.__new__(study.ReclaimWatchdog)
        monitor.lock = threading.RLock()
        process = Mock(pid=42)
        process.poll.return_value = None
        monitor.server = SimpleNamespace(process=process)
        monitor.counters = Mock()
        monitor.rows = []
        monitor.reason = ''
        monitor.seen_shared = False
        monitor.creation_time = 7.0
        monitor.stopped = threading.Event()
        monitor.limits = {study.RTX:8151,study.RX:8192}
        monitor.counters.read.return_value = dict(adapter_dedicated={study.RTX:study.DEDICATED_CAP,study.RX:0},
            process_shared={f'pid_42_{study.RTX}':study.SHARED_CAP+1},engine_utilization={})
        with patch.object(study.psutil,'Process') as identity, patch.object(study.psutil,'virtual_memory') as memory:
            identity.return_value.create_time.return_value = 7.0
            memory.return_value.available = 9*1024**3
            monitor.observe(required=True)
        self.assertIn('184',monitor.reason)
        self.assertTrue(monitor.stopped.is_set())
        self.assertEqual(len(monitor.rows),1)
        monitor.reason = ''
        monitor.stopped.clear()
        monitor.counters.read.return_value['process_shared'][f'pid_42_{study.RTX}'] = study.SHARED_CAP
        self.assertEqual(study.AVAILABLE_RAM_MIN_BYTES,4*1024**3)
        with patch.object(study.psutil,'Process') as identity, patch.object(study.psutil,'virtual_memory') as memory:
            identity.return_value.create_time.return_value = 7.0
            for available in (8*1024**3-1,4*1024**3+1,4*1024**3):
                with self.subTest(available=available):
                    memory.return_value.available = available
                    monitor.observe(required=True)
                    self.assertEqual(monitor.reason,'')
                    self.assertFalse(monitor.stopped.is_set())
            memory.return_value.available = 4*1024**3-1
            monitor.observe(required=True)
        self.assertIn('4GiB',monitor.reason)
        self.assertTrue(monitor.stopped.is_set())

    def test_rtx_reclaim_strict_caps_and_fail_closed_admission(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.rtx-encoder-reclaim-20260921.study')
        pid = 42
        shared_key = f'pid_{pid}_{study.RTX}'
        sample = dict(at_ms=1000,pid=pid,adapter_dedicated={study.RTX:study.DEDICATED_CAP},process_shared={shared_key:study.SHARED_CAP})
        self.assertEqual(study.strict_reason(sample,pid,1500),'')
        self.assertTrue(study.strict_reason(sample,pid,1501))
        self.assertTrue(study.strict_reason(sample,pid,999))
        self.assertTrue(study.strict_reason(sample,43,1000))
        self.assertTrue(study.strict_reason(dict(sample,process_shared={}),pid,1000))
        self.assertTrue(study.strict_reason(dict(sample,adapter_dedicated={study.RTX:study.DEDICATED_CAP+1}),pid,1000))
        exceeded = dict(sample,process_shared={shared_key:study.SHARED_CAP+1})
        self.assertIn('shared',study.strict_reason(exceeded,pid,1000))
        gate = study.SupportGate()
        self.assertTrue(gate.check(exceeded,pid,1000))
        self.assertTrue(gate.check(sample,pid,1000))
        safe = dict(sample,adapter_dedicated={study.RTX:study.DEDICATED_CAP-study.HEADROOM})
        self.assertEqual(study.SupportGate().check(safe,pid,1000,0),'')
        self.assertTrue(study.SupportGate().check(safe,pid,1000,1))
        self.assertTrue(study.SupportGate().check(safe,None,1000))
        self.assertTrue(study.admission(0,study.DEDICATED_CAP-study.HEADROOM,study.DEDICATED_CAP,study.SHARED_CAP)['allowed'])
        self.assertFalse(study.admission(1,study.DEDICATED_CAP-study.HEADROOM,study.DEDICATED_CAP,study.SHARED_CAP)['allowed'])
        self.assertFalse(study.admission(0,0,study.DEDICATED_CAP+1,0)['allowed'])
        self.assertFalse(study.admission(0,0,0,study.SHARED_CAP+1)['allowed'])

    def test_q4_input_lut_bitplanes_and_numerical_equivalence(self):
        import importlib
        import numpy as np
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-q4-input-lut-20260921.study')
        codes = np.array(np.unravel_index(np.arange(65536),(16,16,16,16))).T.astype(np.uint32)
        masks = study.nibble_masks(codes)
        words = np.sum(codes << (8*np.arange(4,dtype=np.uint32)),axis=-1,dtype=np.uint32)
        for bit in range(4):
            compressed = ((((words>>bit)&np.uint32(0x01010101))*np.uint32(0x01020408))>>np.uint32(24))&15
            np.testing.assert_array_equal(compressed,masks[:,bit])
        vector = np.random.default_rng(13).normal(0,0.1,512).astype(np.float32)
        raw = study.PRIOR.synthetic_blocks('q4_k',18,13)
        expected = study.PRIOR.native_dequantize(raw,'q4_k').reshape(9,512).astype(np.float64)@vector
        actual = study.q4_lut_dot(raw,vector)
        self.assertTrue(study.PRIOR.accuracy(expected,actual)['passed'])
        np.testing.assert_array_equal(study.q4_lut_dot(raw,np.zeros(512,dtype=np.float32)),np.zeros(9))
        self.assertEqual(study.input_lut(np.zeros(5120)).nbytes,81920)
        for invalid in ([],[1,2,3],[1,2,3,float('nan')]):
            with self.assertRaises(ValueError):
                study.input_lut(invalid)
        fixture = dict(original_weight_bytes=715161600,lut_bytes=81920)
        metrics = dict(timed_repeats=128,lut_builds=128,dispatches=256,batches=16,
            weight_bytes=715161600,lut_bytes=81920,gpu_us=8000,wall_us=8100)
        study.validate_metrics(metrics,fixture)
        for changed in (dict(lut_builds=1),dict(dispatches=128),dict(batches=1),dict(wall_us=1),dict(gpu_us=float('nan'))):
            with self.assertRaises(ValueError):
                study.validate_metrics(dict(metrics,**changed),fixture)

    def test_packed_iq3_preserves_bytes_and_halfword_addresses(self):
        import importlib
        import numpy as np
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-packed-iq3-20260921.study')
        raw = np.random.default_rng(20260921).integers(0,256,(12,110),dtype=np.uint8)
        before = raw.copy()
        packed = study.pack(raw)
        self.assertEqual(packed.nbytes,raw.nbytes)
        np.testing.assert_array_equal(study.unpack(packed),raw)
        np.testing.assert_array_equal(raw,before)
        original = raw.reshape(-1).view('<u2').reshape(-1,55)
        changed = packed.view('<u2')
        addresses = [study.half_address(block,half) for block in range(12) for half in range(55)]
        self.assertEqual(sorted(addresses),list(range(660)))
        for block in range(12):
            for half in range(55):
                self.assertEqual(changed[study.half_address(block,half)],original[block,half])
        for size in (0,110,439,441):
            with self.assertRaises(ValueError):
                study.pack(np.zeros(size,dtype=np.uint8))

    def test_isa_disassembly_counts_static_instructions_not_data_or_visits(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-isa-review-20260921.study')
        text = ('shader main\n  v_rcp_f32 v1, v1 // 0000: 12345678\nlabel_0010:\n'
            '  v_mac_f32 v1, v2, v3 // 0004: 12345678\n'
            '  s_cbranch_scc1 label_0010 // 0008: 12345678\n'
            '  s_endpgm // 000C: 12345678\n  .long 0x12345678\n\x00')
        parsed = study.parse_isa(text)
        self.assertEqual(parsed['static_instructions'],4)
        self.assertEqual(parsed['backward_branch_regions'][0]['start'],4)
        self.assertEqual(parsed['backward_branch_regions'][0]['static_instructions'],2)
        self.assertEqual(parsed['reciprocal_inside_backward_region'],0)
        with self.assertRaises(ValueError):
            study.parse_isa(text.replace('s_cbranch_scc1 label_0010','s_cbranch_scc1 label_FFFF'))
        with self.assertRaises(ValueError):
            study.parse_isa('shader main\n')

    def test_isa_statistics_require_complete_valid_driver_output(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-isa-review-20260921.study')
        stats = dict(vgprs=40,sgprs=32,lds_bytes=17408,lds_limit_bytes=32768,scratch_bytes=0,
            physical_vgprs=256,physical_sgprs=800,available_vgprs=256,available_sgprs=104,workgroup=[256,1,1])
        text = 'STATISTICS '+json.dumps(stats)+'\nCOMPLETE pipelines=1 dispatches=0 buffers=0\n'
        self.assertEqual(study.parse_statistics(text,[256,1,1]),stats)
        with self.assertRaises(ValueError):
            study.parse_statistics(text,[64,1,1])
        with self.assertRaises(ValueError):
            study.parse_statistics(text.replace('COMPLETE','INCOMPLETE_RECORD'),[256,1,1])
        with self.assertRaises(ValueError):
            study.parse_statistics(text.replace('"scratch_bytes": 0','"scratch_bytes": -1'),[256,1,1])

    def test_attn_gate_small_tile_scope_and_environment(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-attn-gate-tile-20260921.study')
        selected = [name for kind,rows,tokens,batches,name in study.cases() if study.in_scope(kind,rows,tokens,5120,batches)]
        self.assertEqual(selected,['gate_input','gate_tail','gate_partial'])
        self.assertFalse(study.in_scope('iq3_xxs',17408,128,5120))
        self.assertFalse(study.in_scope('iq3_xxs',6144,128,6144))
        self.assertEqual(len(study.fixture_rows()),7)
        with patch.dict(os.environ,{study.FLAG:'1','GGML_VK_GCN_MM_GLU':'1'}):
            before = dict(os.environ)
            off,on = study.environment(False),study.environment(True)
            self.assertEqual({key for key in set(off)|set(on) if off.get(key)!=on.get(key)},{study.FLAG})
            self.assertNotIn('GGML_VK_GCN_MM_GLU',on)
            self.assertEqual(before,dict(os.environ))

            control = {name:dict(us=100) for name in ('gate_input','gate_partial')}
            self.assertTrue(study.decision(control,{name:dict(us=94) for name in control})['eligible_for_model'])
            self.assertFalse(study.decision(control,{name:dict(us=99) for name in control})['eligible_for_model'])
            self.assertFalse(study.decision(control,dict(gate_input=dict(us=80),gate_partial=dict(us=102)))['eligible_for_model'])
            log = '\n'.join(f'GCN_PROBE type=iq3_xxs m=6144 n={tokens} k=5120 pipeline=matmul tile=64,64,1 integer=0 dequant_x=0 dequant_y=0 aligned=1 large=0' for tokens in (128,90))
            self.assertEqual(study.activation(log,False,False),[])
            with self.assertRaises(ValueError):
                study.activation(log,True,False)
            on_log = log.replace('tile=64,64,1','tile=32,32,1')
            on_log += '\n'+'\n'.join(f'GCN_ATTN_GATE_SMALL type=iq3_xxs m={rows} n={tokens} k=5120 tile=32,32,1'
                for rows,tokens in ((6144,128),(6145,65),(6144,90)))
            with self.assertRaises(ValueError):
                study.activation(on_log,True,True)
            on_log += '\n'+log.splitlines()[0]
            self.assertEqual(len(study.activation(on_log,True,True)),3)

    def test_prepack_review_counts_shared_intervals_once(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-prepack-20260921.study')
        def node(op,weight='none'):
            return dict(op=op,weight=weight,ne1=128)
        groups = [dict(ns=10,nodes=[node('MUL_MAT','blk.52.ffn_down.weight')]),
            dict(ns=20,nodes=[node('MUL_MAT','blk.52.attn_qkv.weight'),node('SCALE')]),
            dict(ns=3,nodes=[node('CONCAT')]),dict(ns=4,nodes=[node('GATED_DELTA_NET')])]
        summary = study.summarize_intervals([dict(ns=37,groups=groups)])['input']
        self.assertEqual(summary['total_ns'],37)
        self.assertEqual(summary['original_other_ns'],27)
        self.assertEqual(summary['buckets_ns']['other_matmul_including_companions'],20)
        self.assertEqual(summary['remaining_families_ns'],dict(concat=3))
        with self.assertRaises(ValueError):
            study.summarize_intervals([dict(ns=38,groups=groups)])

    def test_prepack_iq3_q4_preserves_native_float_bits(self):
        import importlib
        import numpy as np
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-prepack-20260921.study')
        for kind in study.SIZES:
            raw = study.synthetic_blocks(kind,256,12345)
            before = raw.copy()
            packed = study.pack_blocks(raw,kind)
            expected = study.native_dequantize(raw,kind)
            actual = study.unpack_blocks(packed,kind)
            np.testing.assert_array_equal(raw,before)
            np.testing.assert_array_equal(expected.view(np.uint32),actual.view(np.uint32))
            self.assertEqual(packed.shape,(256,study.SIZES[kind][1]))
        self.assertTrue(study.accuracy(np.zeros(5),np.zeros(5,dtype=np.float32))['passed'])
        self.assertFalse(study.accuracy(np.ones(5),np.zeros(5,dtype=np.float32))['passed'])
        with self.assertRaises(ValueError):
            study.accuracy(np.zeros(5),np.full(5,np.nan))
        with self.assertRaises(ValueError):
            study.accuracy(np.zeros(5),np.zeros(4))

    def test_isolated_abba_changes_exactly_one_flag_and_keeps_one_dll(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-isolated-abba-20260921.study')
        with patch.dict(os.environ,{'GGML_VK_GCN_MM_GLU':'1','GGML_VK_GCN_IQ4_MODE':'acc2','LLAMA_TEST_GCN_IQ4_FFN':'1'}):
            before = dict(os.environ)
            off,on = study.environment(False),study.environment(True)
            study.validate_pair(off,on)
            self.assertEqual(off['GGML_BACKEND_PATH'],on['GGML_BACKEND_PATH'])
            self.assertNotIn('GGML_VK_GCN_IQ4_MODE',on)
            self.assertNotIn('LLAMA_TEST_GCN_IQ4_FFN',on)
            self.assertNotIn('LLAMA_TEST_FA_VEC_DISABLE',on)
            self.assertEqual(before,dict(os.environ))
            with self.assertRaises(ValueError):
                study.validate_pair(off,dict(on,GGML_VK_GCN_IQ3_TPB16='0'))
            with self.assertRaises(ValueError):
                study.validate_pair(on,on)
        self.assertEqual(study.ORDER,('a1','b1','b2','a2'))
        study.activation('',False,numerical=True)
        with self.assertRaises(ValueError):
            study.activation('GCN_MM_GLU m=17408',False,numerical=True)
        log = '\n'.join(f'GCN_MM_GLU m={rows} n={columns} k=5120 role={role} tile=64,64,32 wg=256 lds=17408'
                        for rows,columns,role in ((17408,128,0),(17409,65,0),(17408,90,1)))
        self.assertEqual(len(study.activation(log,True,numerical=True)),3)

    def test_isolated_build_emits_exact_shader_arrays_and_limits_families(self):
        import importlib
        import struct
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-isolated-build-20260921.study')
        words = [0x07230203,0x00010500,0,8,0,(3<<16)|14,0,1,(4<<16)|15,5,1,0,(1<<16)|56]
        data = struct.pack('<'+'I'*len(words),*words)
        text = study.module_definitions('sample_shader',data)
        self.assertIn(f'uint64_t sample_shader_len = {len(data)};',text)
        encoded = text.split('{',1)[1].split('}',1)[0]
        self.assertEqual(bytes(int(value.strip(),16) for value in encoded.split(',')),data)
        word_array = '{'+','.join(hex(word) for word in words)+'}'
        self.assertEqual(study.spirv_include_bytes(word_array),data)
        for invalid in ('[]','{true}','{0x100000000}','{1,2}'):
            with self.assertRaises((ValueError,SyntaxError)):
                study.spirv_include_bytes(invalid)
        with self.assertRaises(ValueError):
            study.module_definitions('bad-symbol',data)
        names = [f'mul_mat_vec_{prefix}iq3_s_{kind}_f32{suffix}'
                 for prefix,kind in (('','f32'),('','f16'),('id_','f32'))
                 for suffix in ('','_subgroup','_subgroup_no_shmem')]
        self.assertEqual(study.family_members(names+['matmul_iq3_s_f32'],'iq3_s'),sorted(names))
        for selected,family in ((names[:-1],'iq3_s'),(names,'iq4_xs'),(names,'iq3_xxs')):
            with self.assertRaises(ValueError):
                study.family_members(selected,family)

    def test_build_audit_validates_spirv_and_compares_content_not_addresses(self):
        import importlib
        import struct
        from types import SimpleNamespace
        from unittest.mock import MagicMock,patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-build-audit-20260921.study')
        words = [0x07230203,0x00010500,0,8,0, (3<<16)|14,0,1,
                 (4<<16)|15,5,1,0, (4<<16)|71,2,1,4, (4<<16)|50,3,2,0, (1<<16)|56]
        data = struct.pack('<'+'I'*len(words),*words)
        record = study.parse_spirv(data)
        self.assertEqual(record['spec_defaults'],{'4':0})
        self.assertEqual(record['instruction_count'],5)
        for invalid in (b'',data[:-1],data[:-4]+b'\x00'*4,data[:20],b'\x00'*4+data[4:]):
            with self.assertRaises(ValueError):
                study.parse_spirv(invalid)
        before = dict(modules=dict(shader=record))
        after = dict(modules=dict(shader=dict(record),new=record))
        result = study.compare_modules(before,after)
        self.assertEqual(result['identical'],1)
        self.assertEqual(result['added'],['new'])
        after['modules']['shader']['sha256'] = 'changed'
        self.assertEqual(study.compare_modules(before,after)['changed'],['shader'])
        self.assertFalse(study.isolation_check(before,after)['generated_shader_isolation_passed'])
        self.assertFalse(study.isolation_check(before,after,['new'])['generated_shader_isolation_passed'])
        self.assertTrue(study.isolation_check(before,after,['shader','new'])['generated_shader_isolation_passed'])
        self.assertTrue(study.isolation_check(before,before)['generated_shader_isolation_passed'])
        self.assertFalse(study.isolation_check(after,before)['generated_shader_isolation_passed'])
        for invalid in ('shad*','unknown'):
            with self.assertRaises(ValueError):
                study.isolation_check(before,after,[invalid])
        base = 0x180000000
        symbols,storage = {},{}
        for family_index,family in enumerate(('iq3_s','iq4_xs')):
            pointers = []
            for index,suffix in enumerate(('', '_subgroup', '_subgroup_no_shmem')):
                name = 'mul_mat_vec_'+family+'_f32_f32'+suffix
                address = 0x1000+family_index*0x4000+index*0x1000
                symbols[name+'_data'] = base+address
                symbols[name+'_len'] = base+address-8
                storage[address] = data
                storage[address-8] = struct.pack('<Q',len(data))
                pointers.append(base+address)
            table = 'arr_dmmv_'+family+'_f32_f32'
            table_address = 0xa000+family_index*0x100
            symbols[table+'_data'] = base+table_address
            symbols[table+'_len'] = base+table_address+32
            storage[table_address] = struct.pack('<3Q',*pointers)
        image = MagicMock()
        image.OPTIONAL_HEADER = SimpleNamespace(ImageBase=base)
        image.get_data.side_effect = lambda offset,size: storage.get(offset,b'')[:size]
        with patch.object(study,'read_symbols',return_value=(symbols,'symbol-hash')), \
             patch.object(study.pefile,'PE',return_value=image),patch.object(study,'sha256',return_value='file-hash'):
            path = study.ROOT/'synthetic.dll'
            extracted = study.extract_modules(path)
            self.assertEqual(extracted['module_count'],6)
            self.assertEqual(len(extracted['pointer_tables']),2)
            tables = study.resolve_reduction_tables(path)
            self.assertEqual(tables['arr_dmmv_iq3_s_f32_f32'][2]['module'],
                             'mul_mat_vec_iq3_s_f32_f32_subgroup_no_shmem')
            self.assertEqual(image.close.call_count,2)

    def test_mm_glu_epilogue_scope_roles_and_isolation(self):
        import importlib
        import math
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-mm-glu-20260921.study')
        text = '\n'.join(f'GCN_MM_GLU m={rows} n={columns} k=5120 role={role} tile=64,64,32 wg=256 lds=17408'
                         for rows,columns,role in ((17408,128,0),(17409,65,0),(17408,90,1)))
        self.assertEqual(len(study.verify_activation(text,True)),3)
        for invalid in (text.replace('n=90','n=64'),text.replace('role=1','role=0'),text+'\nGCN_FFN_IQ4'):
            with self.assertRaises(ValueError):
                study.verify_activation(invalid,True)
        for gate,up in ((-3,2),(0,4),(2,-1)):
            expected = gate/(1+math.exp(-gate))*up
            for role in (0,1):
                current,other = (gate,up) if role else (up,gate)
                gate_value,up_value = (current,other) if role else (other,current)
                self.assertAlmostEqual(gate_value/(1+math.exp(-gate_value))*up_value,expected)
        self.assertEqual(2*64*17*8,17408)
        with patch.dict(os.environ,{'GGML_VK_GCN_FFN_IQ4_TILED':'1','GGML_VK_GCN_IQ3_MEMORY':'reuse'}):
            before = dict(os.environ)
            child = study.environment()
            self.assertEqual(child['GGML_VK_GCN_MM_GLU'],'1')
            self.assertNotIn('GGML_VK_GCN_FFN_IQ4_TILED',child)
            self.assertNotIn('GGML_VK_GCN_IQ3_MEMORY',child)
            self.assertEqual(before,dict(os.environ))

            child = study.environment(model=True)
            self.assertNotIn('LLAMA_TEST_GCN_MM_GLU',child)
            self.assertNotIn('LLAMA_TEST_GCN_IQ4_FFN',child)
            self.assertNotIn('LLAMA_TEST_FA_VEC_DISABLE',child)
            model_log = text.splitlines()[0]+'\nGCN_FA_BR8 active=1 rows=128 kv=3072 dequant=1\n'
            model_log += '\n'.join(f'GCN_IQ3_TPB16 type=iq3_s m={rows} n=1 k={inner} tpb=16 wg=64'
                           for rows,inner in ((17408,5120),(5120,17408)))
            study.model_activation(model_log)
            with self.assertRaises(ValueError):
                study.model_activation(model_log.replace('n=128','n=1'))

    def test_tiled_ffn_budget_mapping_and_generation_exclusion(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-ffn-tiled-20260921.study')
        self.assertEqual(study.design()['shared_bytes'],25344)
        self.assertLess(study.design()['shared_bytes'],32768)
        outputs = [(thread%64,thread//64+4*item) for thread in range(256) for item in range(16)]
        self.assertEqual(len(set(outputs)),4096)
        self.assertEqual(set(outputs),{(row,column) for row in range(64) for column in range(64)})
        loads = [element for thread in range(256) for element in range(thread,2048,256)]
        self.assertEqual(sorted(loads),list(range(2048)))
        text = '\n'.join(f'GCN_FFN_IQ4_TILED m={rows} n={columns} k=5120 tile=64,64,32 wg=256 lds=25344'
                         for rows,columns in ((17408,128),(17409,65)))
        self.assertEqual(len(study.verify_activation(text,True)),2)
        for invalid in (text.replace('n=65','n=1'),text.replace('lds=25344','lds=49152'),text+'\nGCN_IQ3_MEMORY'):
            with self.assertRaises(ValueError):
                study.verify_activation(invalid,True)
        with patch.dict(os.environ,{'GGML_VK_GCN_IQ3_MEMORY':'reuse','GGML_VK_GCN_FFN_IQ4':'1'}):
            child = study.environment()
            self.assertEqual(child['GGML_VK_GCN_FFN_IQ4_TILED'],'1')
            self.assertNotIn('GGML_VK_GCN_IQ3_MEMORY',child)
            self.assertNotIn('GGML_VK_GCN_FFN_IQ4',child)

    def test_iq3_memory_modes_preserve_tpb16_and_input_mapping(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-iq3-memory-20260921.study')
        covered = []
        for thread in range(16):
            subblock,start = thread//2,(thread%2)*2
            cached = [(32*subblock)//4+2*(start+part)+half for part in range(2) for half in range(2)]
            covered.extend(cached)
        self.assertEqual(sorted(covered),list(range(64)))
        for mode in study.MODES:
            text = '\n'.join(f'GCN_IQ3_MEMORY mode={mode} m={rows} n=1 k={inner} reuse={int(mode=="reuse")} direct={int(mode=="direct")} tpb=16 wg=64'
                             for rows,inner in ((17408,5120),(5120,17408),(5121,17408)))
            self.assertEqual(len(study.verify_activation(text,mode,True)),3)
            for invalid in (text.replace('tpb=16','tpb=8'),text.replace('n=1','n=2'),text+'\nGCN_FFN_IQ4'):
                with self.assertRaises(ValueError):
                    study.verify_activation(invalid,mode,True)
        with patch.dict(os.environ,{'GGML_VK_GCN_FFN_IQ4':'1','GGML_VK_GCN_IQ4_MODE':'tpb16','GGML_VK_GCN_IQ3_MEMORY':'direct'}):
            before = dict(os.environ)
            child = study.environment('reuse',model=True)
            self.assertEqual(child['GGML_VK_GCN_IQ3_MEMORY'],'reuse')
            self.assertEqual(child['GGML_VK_GCN_IQ3_TPB16'],'1')
            for key in ('GGML_VK_GCN_FFN_IQ4','GGML_VK_GCN_IQ4_MODE','LLAMA_TEST_FA_VEC_DISABLE'):
                self.assertNotIn(key,child)
            self.assertEqual(before,dict(os.environ))

    def test_ffn_iq4_fusion_activation_exclusions_and_isolation(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-ffn-fusion-20260921.study')
        text = '\n'.join(f'GCN_FFN_IQ4 m={rows} n={columns} k=5120 rows=4 wg=64'
                         for rows,columns in ((17408,1),(17408,128),(17409,65)))
        self.assertEqual(len(study.verify_activation(text,True,True)),3)
        study.verify_activation('',False,False)
        whole = '\n'.join(f'GCN_FFN_WHOLE_GRAPH name=iq4_ffn_{name} nodes=4 repeats_per_submit=1'
                          for name in ('generation','prefill'))
        self.assertEqual(len(study.verify_whole_graph(whole)),2)
        for invalid in ('',whole.replace('nodes=4','nodes=1'),whole.replace('submit=1','submit=281'),whole+'\n'+whole):
            with self.assertRaises(ValueError):
                study.verify_whole_graph(invalid)
        for invalid in (text.replace('n=65','n=2'),text.replace('k=5120','k=17408'),text+'\nGCN_IQ4 mode=tpb16'):
            with self.assertRaises(ValueError):
                study.verify_activation(invalid,True,True)
        with patch.dict(os.environ,{'GGML_VK_GCN_IQ4_MODE':'tpb16','GGML_VK_GCN_FFN_IQ4':'1'}):
            before = dict(os.environ)
            self.assertNotIn('GGML_VK_GCN_FFN_IQ4',study.environment(False))
            self.assertNotIn('GGML_VK_GCN_IQ4_MODE',study.environment(True))
            self.assertEqual(study.environment(True)['GGML_VK_GCN_FFN_IQ4'],'1')
            self.assertEqual(dict(os.environ),before)

    def test_iq4_layout_partition_and_candidate_isolation(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-iq4-layout-20260921.study')
        for threads in (8,16):
            covered = [32*(thread//(threads//8))+4*packed+lane+half*16
                for thread in range(threads)
                for packed in range((thread%(threads//8))*(32//threads),(thread%(threads//8)+1)*(32//threads))
                for lane in range(4) for half in range(2)]
            self.assertEqual(sorted(covered),list(range(256)))
        rows = [line.split() for line in study.fixture_rows()]
        self.assertEqual([(int(row[2]),int(row[3])) for row in rows],[(17408,1),(5120,1),(5121,1),(5120,2)])
        for mode in study.MODES:
            log = '\n'.join(f'GCN_IQ4 mode={mode} m={rows} n=1 k={inner} tpb={16 if mode=="tpb16" else 8} acc2={1 if mode=="acc2" else 0} wg=64'
                            for rows,inner in ((17408,5120),(5120,17408),(5121,17408)))
            self.assertEqual(len(study.verify_activation(log,mode,True)),3)
            with self.assertRaises(ValueError):
                study.verify_activation(log+'\nGCN_DOWN_SPLIT_K4',mode,True)
        with patch.dict(os.environ,{'GGML_VK_GCN_DOWN_SPLIT_K4':'1','GGML_VK_GCN_IQ4_MODE':'acc2'}):
            before = dict(os.environ)
            child = study.environment('tpb16')
            self.assertEqual(child['GGML_VK_GCN_IQ4_MODE'],'tpb16')
            self.assertEqual(child['GGML_VK_GCN_IQ3_TPB16'],'1')
            self.assertNotIn('GGML_VK_GCN_DOWN_SPLIT_K4',child)
            self.assertEqual(dict(os.environ),before)

            model_log = '\n'.join(f'GCN_IQ3_TPB16 type=iq3_s m={rows} n=1 k={inner} tpb=16 wg=64'
                          for rows,inner in ((17408,5120),(5120,17408)))
            model_log += '\nGCN_FA_BR8 active=1 rows=128 kv=3072 dequant=1'
            model_log += '\n'+'\n'.join(f'GCN_IQ4 mode=tpb16 m={rows} n=1 k={inner} tpb=16 acc2=0 wg=64'
                             for rows,inner in ((17408,5120),(5120,17408)))
            self.assertEqual(len(study.model_activation(model_log,'tpb16')),2)
            with self.assertRaises(ValueError):
                study.model_activation(model_log,'acc2')

    def test_splitk_repeat_requires_consistent_gain_and_bounded_resources(self):
        import copy
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-down-splitk-repeat-20260921.study')
        resources = dict(cpu=dict(mean=5,coverage=0.95),available_ram_min_gib=10,
            memory=dict(rss_gib=dict(maximum=2,coverage=1),private_commit_gib=dict(maximum=14,coverage=1)),
            gpu=dict(primary=dict(dedicated_peak_mib=7716,shared_peak_mib=182,usage=dict(coverage=1)),
                     secondary=dict(dedicated_peak_mib=3825,shared_peak_mib=13,usage=dict(coverage=1))))
        runs = [dict(label=label,timings=dict(seconds=120,prompt_seconds=46,image_seconds=66,decode_seconds=73),
                     resources=copy.deepcopy(resources)) for label in study.ORDER]
        for index in (1,2):
            runs[index]['timings'].update(seconds=119,prompt_seconds=45.5,image_seconds=65.8)
            runs[index]['resources']['gpu']['secondary']['dedicated_peak_mib'] += 10
        self.assertTrue(study.promotion_decision(runs)['eligible_for_deployment'])
        for index,key,value in ((2,'seconds',121),(1,'prompt_seconds',46.1),(2,'image_seconds',67)):
            invalid = copy.deepcopy(runs)
            invalid[index]['timings'][key] = value
            self.assertFalse(study.promotion_decision(invalid)['eligible_for_deployment'])
        invalid = copy.deepcopy(runs)
        invalid[1]['resources']['gpu']['secondary']['dedicated_peak_mib'] += 10
        self.assertFalse(study.promotion_decision(invalid)['eligible_for_deployment'])
        invalid = copy.deepcopy(runs)
        invalid[2]['resources']['cpu']['coverage'] = 0.5
        self.assertFalse(study.promotion_decision(invalid)['eligible_for_deployment'])
        with self.assertRaises(ValueError):
            study.promotion_decision(runs[:3])
        with patch.dict(os.environ,{'GGML_VK_GCN_DOWN_SPLIT_K4':'1','GGML_VK_GCN_OUTPUT_ROWS8':'1'}):
            before = dict(os.environ)
            self.assertNotIn('GGML_VK_GCN_DOWN_SPLIT_K4',study.environment('a1'))
            self.assertEqual(study.environment('b1')['GGML_VK_GCN_DOWN_SPLIT_K4'],'1')
            self.assertNotIn('GGML_VK_GCN_OUTPUT_ROWS8',study.environment('b1'))
            self.assertEqual(dict(os.environ),before)

    def test_down_splitk4_preserves_tile_scope_and_isolates_options(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-down-splitk-20260921.study')
        shapes = [(kind,5120,128) for kind in ('iq3_xxs','iq3_s','iq4_xs')]+[('iq3_s',5121,65),('iq4_xs',5120,90)]
        log = '\n'.join(f'GCN_DOWN_SPLIT_K4 type={kind} m={rows} n={columns} k=17408 default=1 split=4 chunk=4352 tile=64,64,1'
                        for kind,rows,columns in shapes)
        self.assertEqual(len(study.verify_activation(log,True)),5)
        self.assertEqual(4*4352,17408)
        self.assertEqual(4352%256,0)
        self.assertEqual(5120*128*4*4,10*1024**2)
        for invalid in (log.replace('split=4','split=2'),log.replace('default=1','default=4'),
                        log.replace('tile=64,64,1','tile=32,32,1'),log.replace('n=90','n=42'),
                        log+'\nGCN_DOWN_SMALL',log+'\nVK_TEST_BOUNDED_TRANSFER'):
            with self.assertRaises(ValueError):
                study.verify_activation(invalid,True)
        with patch.dict(os.environ,{'GGML_VK_GCN_DOWN_SMALL':'1','GGML_VK_GCN_OUTPUT_ROWS8':'1',
                                   'GGML_VK_TEST_BOUNDED_TRANSFER':'1','LLAMA_TEST_BOUNDED_Q4_INIT':'1'}):
            before = dict(os.environ)
            child = study.environment()
            self.assertEqual(child['GGML_VK_GCN_DOWN_SPLIT_K4'],'1')
            self.assertEqual(child['GGML_VK_GCN_IQ3_TPB16'],'1')
            for key in ('GGML_VK_GCN_DOWN_SMALL','GGML_VK_GCN_OUTPUT_ROWS8',
                        'GGML_VK_TEST_BOUNDED_TRANSFER','LLAMA_TEST_BOUNDED_Q4_INIT'):
                self.assertNotIn(key,child)
            self.assertEqual(dict(os.environ),before)
        self.assertNotIn('LLAMA_TEST_FA_VEC_DISABLE',study.environment(model=True))
        model_log = '\n'.join(log.splitlines()[:3])
        model_log += '\n'+'\n'.join(f'GCN_IQ3_TPB16 type=iq3_s m={rows} n=1 k={inner} tpb=16 wg=64'
                                    for rows,inner in ((17408,5120),(5120,17408)))
        model_log += '\nGCN_FA_BR8 active=1 rows=128 kv=3072 heads=24 br=8 bc=32 row_split=4 dequant=1'
        self.assertEqual(len(study.model_activation(model_log)),3)
        for invalid in (model_log.replace('n=128 k=17408','n=42 k=17408'),
                        model_log+'\nGCN_OUTPUT_ROWS8 type=q4_K',
                        model_log.replace('GCN_DOWN_SPLIT_K4','DISABLED_SPLIT_K4')):
            with self.assertRaises(ValueError):
                study.model_activation(invalid)

    def test_bounded_output_preserves_shapes_and_requires_full_transfer_evidence(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-output-bounded-20260921.study')
        log = '\n'.join(f'TEST_BOUNDED_Q4_INIT rows={rows} completed={rows} chunk_rows=819 float_bytes=16773120'
                        for rows in (248320,248321,4096))
        log += '\n'+'\n'.join(f'VK_TEST_BOUNDED_TRANSFER direction=read bytes={rows*2880} chunk=16777216'
                              for rows in (248320,248321))
        self.assertEqual(len(study.verify_bounded(log,True)['initialization']),3)
        for invalid in (log.replace('completed=248321','completed=248320'),
                        log.replace('chunk=16777216','chunk=33554432'),
                        log.replace('direction=read','direction=write'),
                        log.replace('float_bytes=16773120','float_bytes=16777217')):
            with self.assertRaises(ValueError):
                study.verify_bounded(invalid,True)
        for total in (248320,248321,4096):
            chunks = [(begin,min(819,total-begin)) for begin in range(0,total,819)]
            self.assertEqual(sum(count for _,count in chunks),total)
            self.assertEqual(chunks[-1][0]+chunks[-1][1],total)
            self.assertTrue(all(count*5120*4<=study.LIMIT for _,count in chunks))
        with patch.dict(os.environ,{'GGML_VK_GCN_OUTPUT_ROWS8':'1','GGML_VK_TEST_BOUNDED_TRANSFER':'0'}):
            before = dict(os.environ)
            child = study.environment(False)
            self.assertNotIn('GGML_VK_GCN_OUTPUT_ROWS8',child)
            self.assertEqual(child['GGML_VK_TEST_BOUNDED_TRANSFER'],'1')
            self.assertEqual(child['LLAMA_TEST_BOUNDED_Q4_INIT'],'1')
            self.assertEqual(child['GGML_VK_GCN_IQ3_TPB16'],'1')
            self.assertEqual(dict(os.environ),before)

        with patch.dict(os.environ,{'GGML_VK_TEST_BOUNDED_TRANSFER':'1','LLAMA_TEST_BOUNDED_Q4_INIT':'1'}):
            child = study.environment(True,model=True)
            self.assertNotIn('GGML_VK_TEST_BOUNDED_TRANSFER',child)
            self.assertNotIn('LLAMA_TEST_BOUNDED_Q4_INIT',child)
            self.assertNotIn('LLAMA_TEST_FA_VEC_DISABLE',child)
            self.assertEqual(child['GGML_VK_GCN_OUTPUT_ROWS8'],'1')
        model_log = '\n'.join(f'GCN_IQ3_TPB16 type=iq3_s m={rows} n=1 k={inner} tpb=16 wg=64'
                              for rows,inner in ((17408,5120),(5120,17408)))
        model_log += '\nGCN_OUTPUT_ROWS8 type=q4_K m=248320 n=1 k=5120 rows=8 wg=64'
        model_log += '\nGCN_FA_BR8 active=1 rows=128 kv=3072 heads=24 br=8 bc=32 row_split=4 dequant=1'
        study.model_activation(model_log)
        with self.assertRaises(ValueError):
            study.model_activation(model_log+'\nVK_TEST_BOUNDED_TRANSFER direction=read')

    def test_output_rows_candidate_is_exact_head_and_excludes_other_paths(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-output-rows-20260921.study')
        rows = [line.split() for line in study.fixture_rows()]
        self.assertEqual([(int(row[2]),int(row[3])) for row in rows],[(248320,1),(248321,1),(4096,2)])
        self.assertTrue(all(len(row)==27 and row[8]=='12' and row[9]=='5120' for row in rows))
        text = '\n'.join(f'GCN_OUTPUT_ROWS8 type=q4_K m={count} n=1 k=5120 rows=8 wg=64' for count in (248320,248321))
        self.assertEqual(len(study.verify_activation(text,True,True)),2)
        for invalid in (text.replace('rows=8','rows=4'),text.replace('248321','4096'),text+'\nGCN_DOWN_SMALL'):
            with self.assertRaises(ValueError):
                study.verify_activation(invalid,True,True)
        study.verify_activation('',False,False)
        with patch.dict(os.environ,{'GGML_VK_GCN_OUTPUT_ROWS8':'1','GGML_VK_GCN_DOWN_SMALL':'1'}):
            before = dict(os.environ)
            child = study.environment(False)
            self.assertNotIn('GGML_VK_GCN_OUTPUT_ROWS8',child)
            self.assertNotIn('GGML_VK_GCN_DOWN_SMALL',child)
            self.assertEqual(child['GGML_VK_GCN_IQ3_TPB16'],'1')
            self.assertEqual(dict(os.environ),before)

    def test_ffn_layout_candidates_are_disjoint_and_load_partition_is_complete(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-ffn-layout-20260921.study')
        for threads in (8,16):
            covered = [32*(thread//(threads//8))+8*((thread%(threads//8))*(32//threads)+offset)+lane
                       for thread in range(threads) for offset in range(32//threads) for lane in range(8)]
            self.assertEqual(sorted(covered), list(range(256)))
        rows = [line.split() for line in study.down_rows()]
        self.assertEqual([(int(row[2]),int(row[3])) for row in rows], [(5120,128)]*3+[(5121,65),(5120,90),(5120,42)])
        with patch.dict(os.environ, {flag:'1' for flag in study.FLAGS.values()}):
            for variant, flag in study.FLAGS.items():
                child = study.environment(variant)
                self.assertEqual(child[flag], '1')
                for other in set(study.FLAGS.values())-{flag}:
                    self.assertNotIn(other, child)
        log = '\n'.join(f'GCN_IQ3_TPB16 type=iq3_s m={rows} n=1 k={inner} tpb=16 wg=64' for rows,inner in ((17408,5120),(5120,17408),(5121,17408)))
        self.assertEqual(len(study.verify_activation(log,'tpb16',True)),3)
        with self.assertRaises(ValueError):
            study.verify_activation(log+'\nGCN_IQ3_ROWS4','tpb16',True)
        study.model_activation(log)
        for invalid in (log.replace('m=17408','m=17407'),log+'\nGCN_DOWN_SMALL'):
            with self.assertRaises(ValueError):
                study.model_activation(invalid)

    def test_iq3_generation_workgroup_scope_and_fixtures(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-iq3-dmmv-20260921.study')
        rows = [line.split() for line in study.fixture_rows()]
        self.assertEqual(len(rows), 6)
        self.assertEqual([(int(row[2]),int(row[3])) for row in rows], [(17408,1),(5120,1),(17408,1),(5120,1),(5121,1),(5120,2)])
        for row in rows:
            self.assertEqual(len(row), 27)
            self.assertEqual(row[:2], ['29','0'])
            self.assertEqual(row[17], '0')
        text = '\n'.join(f'GCN_IQ3_DMMV mode=large type={kind} m={count} n=1 k={inner} wg=256 reduction=hybrid'
            for kind,count,inner in (('iq3_xxs',17408,5120),('iq3_xxs',5120,17408),('iq3_s',17408,5120),('iq3_s',5120,17408),('iq3_s',5121,17408)))
        self.assertEqual(len(study.verify_activation(text, 'large', True)), 5)
        for invalid in (text.replace('wg=256','wg=64'), text.replace('n=1','n=2'), text+'\nGCN_FA_BR4 active=1'):
            with self.assertRaises(ValueError):
                study.verify_activation(invalid, 'large', True)
        with patch.dict(os.environ, {'GGML_VK_GCN_FA_BR4':'1','GGML_VK_PERF_LOGGER':'1'}):
            before = dict(os.environ)
            child = study.environment('large')
            self.assertEqual(child['GGML_VK_GCN_IQ3_DMMV'], 'large')
            self.assertEqual(child['GGML_VK_GCN_FA_BR8'], '1')
            self.assertNotIn('GGML_VK_GCN_FA_BR4', child)
            self.assertNotIn('GGML_VK_PERF_LOGGER', child)
            self.assertEqual(dict(os.environ), before)

    def test_br4_research_does_not_inherit_other_candidates(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-fa-br4-20260921.study')
        with patch.dict(os.environ, {'GGML_VK_GCN_FA_Q8_DIRECT':'1', 'GGML_VK_GCN_FA_MASK_OPT':'1', 'GGML_VK_PERF_LOGGER':'1'}):
            before = dict(os.environ)
            child = study.environment('causal-prefix')
            self.assertEqual(child['GGML_VK_GCN_FA_BR4'], '1')
            self.assertEqual(child['GGML_VK_GCN_FA_BR8'], '1')
            for key in ('GGML_VK_GCN_FA_Q8_DIRECT', 'GGML_VK_GCN_FA_MASK_OPT', 'GGML_VK_PERF_LOGGER'):
                self.assertNotIn(key, child)
            self.assertEqual(dict(os.environ), before)
        text = '\n'.join(f'GCN_FA_BR4 active=1 rows={rows} kv={kv} heads=24 br=4 bc=32 row_split=4 dequant={dequant}'
            for rows,kv,dequant in ((128,3072,1),(64,3072,1),(128,1024,1),(128,3072,0),(90,3072,1),(65,3071,1)))
        self.assertEqual(len(study.verify_activation(text, True)), 6)
        for invalid in (text.replace('rows=65','rows=63'), text+'\nGCN_FA_BR8 active=1', text.replace('br=4','br=8')):
            with self.assertRaises(ValueError):
                study.verify_activation(invalid, True)

    def test_gcn_q8_direct_preserves_reference_fixture_and_limits_scope(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-fa-q8-direct-20260921.study')
        lines = study.fixture_rows()
        self.assertEqual(lines[0], study.MASK.fixture_rows(74)[0])
        self.assertEqual([int(line.split()[4]) for line in lines], [128, 64, 128, 128])
        self.assertEqual([int(line.split()[4]) for line in study.fixture_rows(True)], [90, 65])
        with self.assertRaisesRegex(ValueError, 'first real prompt'):
            study.verify_model_activation('GCN_MEDIUM_TILE active=1')
        study.verify_model_activation('GCN_FA_BR8 active=1 rows=128 kv=3072 heads=24 br=8 bc=32 row_split=4 dequant=1')
        with self.assertRaises(ValueError):
            study.verify_model_activation('GCN_FA_BR8 active=1 rows=128 dequant=1\nGCN_FA_Q8_DIRECT changed=1')
        log = '\n'.join(f'GCN_FA_Q8_DIRECT changed=1 rows={rows} kv={kv} heads=24 default_dequant=1 dequant=0'
            for rows, kv in ((128, 3072), (64, 3072), (128, 1024)))
        log += '\nGCN_FA_Q8_DIRECT changed=0 rows=128 kv=3072 heads=24 default_dequant=0 dequant=0'
        self.assertEqual(len(study.verify_activation(log, True)), 4)
        for invalid in (log.replace('rows=64', 'rows=63'), log.replace('dequant=0', 'dequant=1'),
                        log+'\nGCN_FA_MASK active=1'):
            with self.assertRaises(ValueError):
                study.verify_activation(invalid, True)
        with patch.dict(os.environ, {'GGML_VK_GCN_FA_MASK_OPT':'1', 'GGML_VK_PERF_LOGGER':'1'}):
            before = dict(os.environ)
            child = study.environment('causal-prefix')
            self.assertEqual(child['GGML_VK_GCN_FA_Q8_DIRECT'], '1')
            self.assertNotIn('GGML_VK_GCN_FA_MASK_OPT', child)
            self.assertNotIn('GGML_VK_PERF_LOGGER', child)
            self.assertEqual(dict(os.environ), before)
        with patch.object(study, 'VARIANT', 'br8'):
            child = study.environment()
            self.assertEqual(child['GGML_VK_GCN_FA_BR8'], '1')
            self.assertNotIn('GGML_VK_GCN_FA_Q8_DIRECT', child)
            br8 = '\n'.join(f'GCN_FA_BR8 active=1 rows={rows} kv={kv} heads=24 br=8 bc=32 row_split=4 dequant={dequant}'
                for rows, kv, dequant in ((128,3072,1), (64,3072,1), (128,1024,1), (128,3072,0)))
            self.assertEqual(len(study.verify_activation(br8, True)), 4)
            with self.assertRaises(ValueError):
                study.verify_activation(br8+'\nGCN_FA_Q8_DIRECT changed=1', True)
            boundary = '\n'.join(f'GCN_FA_BR8 active=1 rows={rows} kv={kv} heads=24 br=8 bc=32 row_split=4 dequant=1'
                for rows, kv in ((90,3072), (65,3071)))
            self.assertEqual(len(study.verify_activation(boundary, True, True)), 2)
            with self.assertRaises(ValueError):
                study.verify_activation(boundary.replace('rows=65', 'rows=64'), True, True)

    def test_gcn_attention_mask_fixtures_and_environment_are_isolated(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-fa-mask-20260921.study')
        lines = [line.split() for line in study.fixture_rows(99)]
        self.assertEqual(len(lines), 4)
        activation = '\n'.join(f'GCN_FA_MASK active=1 rows={rows} kv=3072' for rows in (128, 42, 33))
        study.verify_activation(activation+'\nGCN_FA_MASK active=0 rows=6 kv=3072', True)
        with self.assertRaises(ValueError):
            study.verify_activation(activation+'\nGCN_FA_MASK active=1 rows=6 kv=3072', True)
        self.assertEqual([int(line[4]) for line in lines], [128, 42, 33, 1])
        for line in lines:
            self.assertEqual(list(map(int, line[:4])), [99, 0, 256, 24])
            self.assertEqual(list(map(int, line[6:13])), [5, 1031798784, 0, 0, 10, 0, 4])
            self.assertEqual(len(line), 50)
        with patch.dict(os.environ, {'GGML_VK_PERF_LOGGER':'1', 'GGML_VK_GCN_FA_MASK_OPT':'1', 'GGML_SCHED_COPY_TRACE':'1'}):
            before = dict(os.environ)
            control = study.environment(False, 'random')
            candidate = study.environment(True, 'causal-prefix')
            self.assertNotIn('GGML_VK_GCN_FA_MASK_OPT', control)
            self.assertNotIn('GGML_VK_PERF_LOGGER', candidate)
            self.assertNotIn('GGML_SCHED_COPY_TRACE', candidate)
            self.assertEqual(candidate['GGML_VK_GCN_FA_MASK_OPT'], '1')
            self.assertEqual(candidate['LLAMA_TEST_KQ_MASK_PATTERN'], 'causal-prefix')
            self.assertEqual(dict(os.environ), before)

    def test_vulkan_device_trace_counts_shared_intervals_once(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.vulkan-device-trace-20260921.study')
        lines = ['VK_DEVICE_TRACE kind=begin graph=1 backend=Vulkan0 nodes=2 groups=1 concurrent=1',
            'VK_DEVICE_TRACE kind=group graph=1 query=1 ns=100 members=2',
            'VK_DEVICE_TRACE kind=node graph=1 query=1 member=0 op=MUL_MAT tensor=ffn_gate-51 weight=blk.51.ffn_gate.weight type=iq2_s ne0=17408 ne1=128 ne2=1 ne3=1 k=5120',
            'VK_DEVICE_TRACE kind=node graph=1 query=1 member=1 op=MUL_MAT tensor=ffn_up-51 weight=blk.51.ffn_up.weight type=iq3_xxs ne0=17408 ne1=128 ne2=1 ne3=1 k=5120',
            'VK_DEVICE_TRACE kind=end graph=1 ns=100']
        text = '\n'.join(lines)
        totals = study.aggregate(study.parse_trace(text))
        self.assertEqual(totals, {'input|ffn_gate_up': dict(groups=1, representative_nodes=2, ns=100)})
        raw = (text+'\nidle\n').encode()
        bounds = dict(begin=0, end=raw.index(b'kind=end')-16)
        recovered = study.request_traces(raw, [bounds])
        self.assertEqual(recovered[0]['totals'], totals)
        self.assertGreater(recovered[0]['late_log_bytes'], 0)
        with self.assertRaises(ValueError):
            study.request_traces(raw+raw, [bounds])
        graph = study.parse_trace(text.replace('ne1=128', 'ne1=1'))[0]
        graph['groups'][0]['nodes'].append(dict(op='MUL_MAT', weight='Vulkan0#attn_inp_v_rot#0', ne1=96))
        self.assertEqual(study.graph_phase(graph), 'generation')
        with self.assertRaisesRegex(ValueError, 'boundary'):
            study.validate_tail([graph])
        graph['tail_closed'] = True
        with self.assertRaisesRegex(ValueError, 'Output'):
            study.validate_tail([graph])
        graph['groups'][0]['nodes'].append(dict(weight='output.weight'))
        study.validate_tail([graph])
        for invalid in ('\n'.join(lines[:-1]), '\n'.join(lines[:3]+lines[4:]), text.replace('kind=end graph=1 ns=100', 'kind=end graph=1 ns=101'),
                        text.replace('members=2', 'members=1'), text.replace('concurrent=1', 'concurrent=0'), text+'\n'+text,
                        text.replace('ns=100', 'ns=-1')):
            with self.assertRaises(ValueError):
                study.parse_trace(invalid)

    def test_scheduler_copy_trace_requires_complete_graphs_and_separates_waits(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.scheduler-trace-20260921.study')
        lines = ['SCHED_TRACE kind=begin graph=1 splits=1',
            'SCHED_TRACE kind=copy graph=1 split=0 src=CUDA0 dst=Vulkan0 tensor=norm-51 bytes=1024 ne1=128 mode=blocking reuse_us=3 attempt_us=4 source_wait_us=50 destination_wait_us=6 blocking_copy_us=7',
            'SCHED_TRACE kind=submit graph=1 split=0 backend=Vulkan0 nodes=8 host_us=9 status=0',
            'SCHED_TRACE kind=end graph=1 host_us=80']
        graphs = study.parse_trace('\n'.join(lines))
        group = study.aggregate(graphs)['input|CUDA0|Vulkan0|blocking']
        self.assertEqual(group['source_wait_us'], 50)
        self.assertEqual(group['blocking_copy_us'], 7)
        self.assertEqual(group['bytes'], 1024)
        for invalid in ('\n'.join(lines[:-1]), '\n'.join(lines[1:]), '\n'.join(lines).replace('splits=1', 'splits=2'),
                        '\n'.join(lines).replace('status=0', 'status=1'), '\n'.join(lines).replace('source_wait_us=50', 'source_wait_us=-1')):
            with self.assertRaises(ValueError):
                study.parse_trace(invalid)

    def test_image_mtp_cpu_sleep_only_changes_wait_options(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.mtp-image-20260921.study')
        settings = ('server','model','projector',1024,1024,('-ub','128','--poll','0','--poll-batch','0'))
        candidate = dict(cpu_wait_sleep=True,draft_n_max=2,poll=0,poll_batch=0)
        server = study.ImageMTPServer(candidate)
        with patch.object(study.placement.PlacementServer,'settings_for',return_value=settings):
            changed = server.settings_for()
            self.assertEqual(changed[:-1],settings[:-1])
            self.assertEqual(changed[-1],settings[-1]+('--spec-draft-poll','0','--spec-draft-poll-batch','0'))
            for invalid in (dict(candidate,draft_n_max=3),dict(candidate,poll=50),dict(candidate,poll_batch=1)):
                server.candidate = invalid
                with self.assertRaises(ValueError):
                    server.settings_for()
            server.candidate = dict(cpu_wait_sleep=False)
            self.assertEqual(server.settings_for(),settings)
        server.candidate = candidate
        duplicate = (*settings[:-1],settings[-1]+('--spec-draft-poll','0'))
        with patch.object(study.placement.PlacementServer,'settings_for',return_value=duplicate):
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                server.settings_for()

    def test_image_mtp_evidence_requires_every_original_batch(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.mtp-image-20260921.study')
        rows = [f'MTP_IMAGE processed rows={count} width=5120 positions=original separate_hidden=1 seq=0'
                for count in [512,512,42]*4]
        self.assertEqual(len(study.image_execution_evidence('\n'.join(rows))),12)
        for text in ('\n'.join(rows[:-1]),'\n'.join(rows).replace('width=5120','width=1024'),
                     '\n'.join(rows).replace('seq=0','seq=1'),'\n'.join(rows)+'\nMTP_IMAGE draft decode failed rc=1'):
            with self.assertRaises(ValueError):
                study.image_execution_evidence(text)

    def test_cherrypick_profile_excludes_unfinished_table_and_weights_calls(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.cherrypick-20260921.study')
        group = 'MUL_MAT iq2_s m=17408 n=128 k=5120, MUL_MAT iq3_xxs m=17408 n=128 k=5120'
        text = '\n'.join(['Vulkan Timings:',group+': 2 x 10 us = 20 us','----------------',
            'Vulkan Timings:',group+': 1 x 40 us = 40 us','----------------',
            'Vulkan Timings:',group+': 5 x 99 us = 495 us'])
        result = study.parse_profile(text)
        self.assertEqual(result['complete_tables'],2)
        self.assertTrue(result['incomplete_final_table'])
        self.assertEqual(result['input_ffn_groups'][group],dict(calls=3,total_us=60.0,mean_us=20.0))
        with self.assertRaisesRegex(ValueError,'did not terminate'):
            study.parse_profile('Vulkan Timings:\nVulkan Timings:')

    def test_cherrypick_mtp_confidence_only_changes_draft_option(self):
        import importlib
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.cherrypick-20260921.study')
        settings = ('server','model','projector',1024,1024,('-ub','128','--spec-type','draft-mtp'))
        candidate = dict(spec_type='draft-mtp',mtp_image_checks=False,draft_probability_min=0.5)
        server = study.CherryServer(candidate)
        with patch.object(study.INTEGRATED.IntegratedServer,'settings_for',return_value=settings):
            changed = server.settings_for()
            self.assertEqual(changed[:-1],settings[:-1])
            self.assertEqual(changed[-1],settings[-1]+('--spec-draft-p-min','0.5'))
            server.candidate = dict(candidate,mtp_image_checks=True)
            with self.assertRaises(ValueError):
                server.settings_for()
            server.candidate = dict(candidate,draft_probability_min=0.3)
            with self.assertRaises(ValueError):
                server.settings_for()

    def test_cherrypick_plan_preserves_weights_and_layer_partition(self):
        import importlib
        import re
        study = importlib.import_module('desktop_agent.data.benchmarks.cherrypick-20260921.study')
        weights = {f'blk.{block}.{kind}.weight':dict(bytes=1024,type='F32')
                   for block in range(65) for kind in ('ffn_gate','ffn_up','ffn_down','attn_norm','post_attention_norm')}
        weights.update({name:dict(bytes=2048,type='F32') for name in ('output.weight','output_norm.weight','token_embd.weight')})
        reference = dict(ubatch=128,cache_k='q8_0',cache_v='q8_0',spec_type='none')
        whole = set(range(52,55))|set(range(56,64))
        candidate = study.make_candidate(reference,weights,'test',whole,{51,55})
        self.assertEqual(candidate['tensor_split'],'52,3,1,10')
        self.assertEqual(candidate['estimated_stage_transitions'],3)
        self.assertEqual(candidate['ubatch'],128)
        self.assertFalse(any(name.startswith('blk.64.') for name in candidate['secondary_tensors']))
        total = sum(value['bytes'] for name,value in weights.items() if not name.startswith('blk.64.'))/1024**2
        self.assertEqual(sum(candidate[key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')),total)
        self.assertNotIn('secondary_tensors',reference)
        rules = [rule.rsplit('=',1) for rule in candidate['override'].split(',')]
        self.assertTrue(any(re.fullmatch(pattern,'blk.51.ffn_gate.weight') and device=='Vulkan0' for pattern,device in rules))
        self.assertFalse(any(re.fullmatch(pattern,'blk.51.attn_norm.weight') and device=='Vulkan0' for pattern,device in rules))
        shifted = study.make_candidate(reference,weights,'swap',whole,{50,55})
        self.assertEqual(shifted['estimated_stage_transitions'],5)
        boundary = study.make_candidate(reference,weights,'boundary',whole|{51},{55})
        self.assertEqual(boundary['tensor_split'],'51,4,1,10')
        with self.assertRaises(ValueError):
            study.make_candidate(reference,weights,'bad',whole,{52})

    def test_cherrypick_component_swap_avoids_conflicting_overrides(self):
        import importlib
        import re
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.cherrypick-20260921.study')
        weights = {f'blk.{block}.{kind}.weight':dict(bytes=1024,type='F32')
                   for block in range(65) for kind in ('ffn_gate','ffn_up','ffn_down','attn_norm','post_attention_norm')}
        weights.update({name:dict(bytes=2048,type='F32') for name in ('output.weight','output_norm.weight','token_embd.weight')})
        reference = dict(name='attention55_baseline',whole_blocks=sorted(set(range(52,55))|set(range(56,64))),
            ubatch=128,cache_k='q8_0',cache_v='q8_0',spec_type='none')
        with patch.object(study.placement,'mtp_candidates',return_value=[reference]):
            options = study.extended_candidates(weights)
        swap = options['gate51_down50']
        self.assertNotIn('blk.51.ffn_gate.weight',swap['secondary_tensors'])
        self.assertIn('blk.50.ffn_down.weight',swap['secondary_tensors'])
        rules = [rule.rsplit('=',1) for rule in swap['override'].split(',')]
        self.assertFalse(any(re.fullmatch(pattern,'blk.51.ffn_gate.weight') for pattern,device in rules))
        self.assertTrue(any(re.fullmatch(pattern,'blk.50.ffn_down.weight') and device=='Vulkan0' for pattern,device in rules))
        self.assertIsNone(swap['estimated_stage_transitions'])
        self.assertNotIn('blk.55.ffn_gate.weight',options['pack55_cuda']['secondary_tensors'])
        self.assertIn('blk.51.attn_norm.weight',options['pack55_cuda']['secondary_tensors'])
        for candidate in options.values():
            self.assertEqual(candidate['ubatch'],128)
            self.assertTrue(candidate['stop_on_quality_failure'])
        weights['blk.64.nextn.eh_proj.weight'] = dict(bytes=2048,type='F32')
        with patch.object(study.placement,'mtp_candidates',return_value=[reference]):
            combo = study.combo_candidate(weights)
        self.assertEqual(combo['tensor_split'],'50,5,1,10')
        self.assertEqual(combo['draft_n_max'],3)
        self.assertFalse(combo['mtp_image_checks'])
        self.assertEqual(combo['spec_type'],'draft-mtp')
        self.assertIn('blk.50.attn_norm.weight',combo['secondary_tensors'])
        self.assertIn('blk.64.nextn.eh_proj.weight',combo['secondary_tensors'])
        self.assertEqual(len(combo['secondary_tensors']),len(set(combo['secondary_tensors'])))
        self.assertEqual(sum(combo[key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')),
                         sum(item['bytes'] for item in weights.values())/1024**2)

    def test_gcn_rectangular_iq2_load_bound_preserves_full_tile(self):
        def coordinates(rows,vector_width,bounded):
            result = []
            block_depth,threads = 32,256
            stride = threads*vector_width//block_depth
            for thread in range(threads):
                column = thread//(block_depth//vector_width)
                vector = thread%(block_depth//vector_width)
                for offset in range(0,rows,stride):
                    if not bounded or column+offset<rows:
                        result.extend((column+offset,vector*vector_width+element) for element in range(vector_width))
            return result
        original = coordinates(32,8,False)
        self.assertEqual(sum(row>=32 for row,element in original),1024)
        for width in (4,8):
            guarded = coordinates(32,width,True)
            self.assertEqual(len(guarded),32*32)
            self.assertEqual(set(guarded),{(row,element) for row in range(32) for element in range(32)})
        self.assertEqual(coordinates(64,8,False),coordinates(64,8,True))

    def test_gcn_mtp_automatic_extension_requires_gain_acceptance_and_memory(self):
        import importlib
        from copy import deepcopy
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-integrated-20260921.research-cycle')
        metrics = dict(passed=True,acceptance=0.95,seconds=90,memory={
            'gpu':dict(shared_mib=100,adapter_peak_mib=7000,next_rs_increment_mib=120,capacity_mib=8000)})
        self.assertTrue(study.mtp_extension_gate(metrics,100,95)['proceed'])
        for key,value in (('passed',False),('acceptance',0.8),('seconds',99)):
            self.assertFalse(study.mtp_extension_gate(dict(metrics,**{key:value}),100,95)['proceed'])
        self.assertFalse(study.mtp_extension_gate(metrics,100,90)['proceed'])
        for key,value in (('shared_mib',321),('adapter_peak_mib',7800)):
            changed = deepcopy(metrics)
            changed['memory']['gpu'][key] = value
            self.assertFalse(study.mtp_extension_gate(changed,100,95)['proceed'])

    def test_gcn_mtp_candidate_keeps_prompt_controls_and_blocks_fake_activation(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-integrated-20260921.run-integrated')
        weights = {f'blk.50.ffn_{kind}.weight':dict(bytes=10*1024**2) for kind in ('gate','up','down')}
        weights['blk.64.nextn.eh_proj.weight'] = dict(bytes=100*1024**2)
        baseline = dict(secondary_tensors=['output.weight'],ffn_blocks=[51,55],override='existing',
            primary_weight_mib=6000,secondary_weight_mib=2600,ubatch=128,cache_k='q8_0',cache_v='q8_0',
            devices='CUDA0,Vulkan0,CUDA0,Vulkan0',tensor_split='52,3,1,10')
        candidate = study.mtp_text_candidate(baseline,weights)
        self.assertEqual(candidate['primary_weight_mib'],5970)
        self.assertEqual(candidate['secondary_weight_mib'],2730)
        self.assertEqual(candidate['draft_n_max'],1)
        self.assertFalse(candidate['mtp_image_checks'])
        for key in ('ubatch','cache_k','cache_v','devices','tensor_split'):
            self.assertEqual(candidate[key],baseline[key])
        self.assertEqual(baseline['secondary_tensors'],['output.weight'])
        for length in (2,3):
            extended = study.mtp_text_candidate(baseline,weights,length)
            self.assertEqual(extended,dict(candidate,name=f'attention55_gcn_mtp_rx_n{length}_ffn50',draft_n_max=length))
        with self.assertRaises(ValueError):
            study.mtp_text_candidate(baseline,weights,4)
        arguments = ['--spec-type','draft-mtp','--spec-draft-n-max','1','--spec-draft-type-k','q8_0',
                     '--spec-draft-type-v','q8_0','-ub','128']
        log = "creating MTP draft context\nadding speculative implementation 'draft-mtp'\nn_rs_seq = 1\n"
        study.verify_mtp_activation(arguments,log)
        for length in (2,3):
            extended_args = list(arguments)
            extended_args[extended_args.index('--spec-draft-n-max')+1] = str(length)
            study.verify_mtp_activation(extended_args,log.replace('= 1','= '+str(length)),length)
        for changed,changed_log in ((arguments+['--spec-synth-len','1'],log),(arguments,''),
                                    (arguments,log.replace('= 1','= 2')),(arguments+['-ub','128'],log)):
            with self.assertRaises(ValueError):
                study.verify_mtp_activation(changed,changed_log)

    def test_gcn_research_resume_never_repeats_completed_or_uncertain_stage(self):
        import importlib
        import time
        from unittest.mock import patch
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-integrated-20260921.research-cycle')
        command = ['synthetic-command']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study.write_new(root/'done.completed.json',dict(command=command,exit_code=0))
            study.write_new(root/'incomplete.started.json',dict(command=command))
            study.write_new(root/'failed.completed.json',dict(command=command,exit_code=1))
            with patch.object(study.subprocess,'Popen') as launch:
                study.run_step(root,'done',command,time.monotonic()+1000)
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                    study.run_step(root,'incomplete',command,time.monotonic()+1000)
                with self.assertRaisesRegex(RuntimeError,'failed or changed'):
                    study.run_step(root,'failed',command,time.monotonic()+1000)
                with self.assertRaisesRegex(RuntimeError,'failed or changed'):
                    study.run_step(root,'done',['changed-command'],time.monotonic()+1000)
                with self.assertRaisesRegex(RuntimeError,'time budget'):
                    study.run_step(root,'new',command,time.monotonic()+10)
                launch.assert_not_called()
            self.assertFalse((root/'new.started.json').exists())

    def test_gcn_research_gate_rejects_regressions_and_changed_cases(self):
        import importlib
        study = importlib.import_module('desktop_agent.data.benchmarks.gcn-integrated-20260921.research-cycle')
        baseline = {name:dict(calls=100,us=1000.0) for name in ('iq2','iq3xxs','iq3s','iq4')}
        self.assertFalse(study.select_candidate(baseline,baseline)['promote'])
        faster = {name:dict(value,us=950.0) for name,value in baseline.items()}
        self.assertTrue(study.select_candidate(baseline,faster)['promote'])
        faster['iq2']['us'] = 1060.0
        self.assertFalse(study.select_candidate(baseline,faster)['promote'])
        with self.assertRaisesRegex(ValueError,'case set'):
            study.select_candidate(baseline,dict(iq2=baseline['iq2']))
        with self.assertRaisesRegex(ValueError,'invocation'):
            study.select_candidate(baseline,dict(baseline,iq2=dict(calls=101,us=900.0)))
        with self.assertRaisesRegex(ValueError,'four'):
            study.perf_rows('No test results')

    def test_local_image_cache_boundary_preserves_request_order(self):
        import threading
        from unittest.mock import MagicMock,patch
        from PIL import Image
        from desktop_agent.agent import Model,Settings,pack_messages,system_prompt
        from desktop_agent.protocol import ToolCatalog
        capabilities = dict(screen=True,input=True,browser=True,approval='manual')
        catalog = ToolCatalog(capabilities)
        events = [dict(id=1,role='user',content='Read the supplied image.')]
        image = Image.new('RGB',(64,64),'white')
        model = Model(Settings())
        model.endpoint = 'http://127.0.0.1:8080'
        model.tool_names = catalog.names()
        payloads = []
        for steps in (20,19):
            messages,_,_ = pack_messages(events,system_prompt(capabilities,catalog),len,100000,
                state=dict(window=None,image_attached=True,pending_jobs=[],steps_remaining=steps))
            original = json.dumps(messages,sort_keys=True)
            client = MagicMock()
            client.__enter__.return_value = client
            client.post.side_effect = RuntimeError('offline payload capture')
            with patch('desktop_agent.agent.session',return_value=client):
                with self.assertRaisesRegex(RuntimeError,'offline payload capture'):
                    model.generate(messages,image,threading.Event(),lambda *args:None)
            self.assertEqual(json.dumps(messages,sort_keys=True),original)
            payloads.append(client.post.call_args.kwargs['json'])
        first,second = payloads
        self.assertTrue(first['cache_prompt'])
        self.assertTrue(second['cache_prompt'])
        self.assertEqual(first['messages'][:-2],second['messages'][:-2])
        self.assertNotEqual(first['messages'][-2],second['messages'][-2])
        self.assertEqual(first['messages'][-1],second['messages'][-1])
        self.assertEqual(first['messages'][-1]['content'][-1]['type'],'image_url')
        self.assertEqual({key:value for key,value in first.items() if key!='messages'},
                         {key:value for key,value in second.items() if key!='messages'})

    def test_selected_f16_expansion_preserves_metadata_and_other_tensors(self):
        import numpy as np
        from gguf import GGUFReader,GGUFWriter,GGMLQuantizationType,quantize,dequantize
        from desktop_agent.benchmark_placement import expand_f16_tensors,convert_selected_tensors
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.gguf'
            destination = Path(directory)/'expanded.gguf'
            values = np.linspace(-1,1,256,dtype=np.float32).reshape(4,64)
            packed = quantize(values,GGMLQuantizationType.Q4_0)
            writer = GGUFWriter(str(source),'qwen35')
            writer.add_name('format-fixture')
            writer.add_token_list(['hello','world'])
            writer.add_tensor('selected.weight',packed,raw_dtype=GGMLQuantizationType.Q4_0)
            writer.add_tensor('untouched.weight',packed.copy(),raw_dtype=GGMLQuantizationType.Q4_0)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            report = expand_f16_tensors(source,destination,['selected.weight'])
            before,after = GGUFReader(str(source),'r'),GGUFReader(str(destination),'r')
            self.assertEqual({key:field.contents() for key,field in before.fields.items()},
                             {key:field.contents() for key,field in after.fields.items()})
            self.assertEqual(after.tensors[0].tensor_type,GGMLQuantizationType.F16)
            np.testing.assert_array_equal(after.tensors[0].data,dequantize(packed,GGMLQuantizationType.Q4_0).astype(np.float16))
            np.testing.assert_array_equal(before.tensors[1].data,after.tensors[1].data)
            self.assertEqual(report['expanded'][0]['name'],'selected.weight')
            self.assertEqual(set(report['copied_sha256']),{'untouched.weight'})
            with self.assertRaises(FileExistsError):
                expand_f16_tensors(source,destination,['selected.weight'])
            with self.assertRaisesRegex(ValueError,'must exist'):
                expand_f16_tensors(source,Path(directory)/'invalid.gguf',['missing.weight'])
            self.assertFalse((Path(directory)/'invalid.gguf').exists())
            q8_path = Path(directory)/'q8.gguf'
            q8_report = convert_selected_tensors(source,q8_path,['selected.weight'],'Q8_0')
            q8 = GGUFReader(str(q8_path),'r')
            self.assertEqual(q8.tensors[0].tensor_type,GGMLQuantizationType.Q8_0)
            np.testing.assert_array_equal(q8.tensors[0].data,
                quantize(dequantize(packed,GGMLQuantizationType.Q4_0),GGMLQuantizationType.Q8_0))
            np.testing.assert_array_equal(before.tensors[1].data,q8.tensors[1].data)
            self.assertEqual({key:field.contents() for key,field in before.fields.items()},
                             {key:field.contents() for key,field in q8.fields.items()})
            self.assertLess(q8_report['expanded'][0]['normalized_rmse'],0.01)
            self.assertEqual(q8_report['target_type'],'Q8_0')
            with self.assertRaisesRegex(ValueError,'only F16 and Q8_0'):
                convert_selected_tensors(source,Path(directory)/'unsupported.gguf',['selected.weight'],'Q4_0')
            before.data._mmap.close()
            after.data._mmap.close()
            q8.data._mmap.close()

    def test_dflash_load_guard_requires_real_activation_and_rx_weights(self):
        from desktop_agent.benchmark_placement import verify_dflash_load
        candidate = dict(draft_model='draft.gguf',draft_device='Vulkan0',draft_gpu_layers='all',
            spec_type='draft-dflash',draft_n_max=7,draft_cache_k='f16',draft_cache_v='f16',
            draft_override=r'^.*$=Vulkan0',draft_weight_mib=1000)
        arguments = ['--spec-draft-model','draft.gguf','--spec-draft-device','Vulkan0',
            '--spec-draft-ngl','all','--spec-type','draft-dflash','--spec-draft-n-max','7',
            '--spec-draft-type-k','f16','--spec-draft-type-v','f16','--spec-draft-override-tensor',r'^.*$=Vulkan0']
        log = '\n'.join([
            'load_tensors: Vulkan0 model buffer size = 2600.00 MiB',
            'llama_context: n_rs_seq = 7',
            'loading draft model',
            'DFlash2 conv kernel = 2, group = 16, selector rank = 256, top-k = 16',
            'load_tensors: Vulkan0 model buffer size = 1000.00 MiB',
            'llama_kv_cache: size = 40.00 MiB, K (f16): 20.00 MiB, V (f16): 20.00 MiB',
            "adding speculative implementation 'draft-dflash'",
            'block_size=8, mask_token_id=248070, n_extract=5',
        ])+'\n'
        self.assertEqual(verify_dflash_load(candidate,arguments,log)['weight_buffers_mib'],{'Vulkan0':1000})
        for changed_args,changed_log in (
            (arguments+['--spec-draft-device','Vulkan0'],log),
            (arguments+['--spec-synth-len','8'],log),
            (arguments,log.replace('loading draft model','missing')),
            (arguments,log.replace('n_rs_seq = 7','n_rs_seq = 3')),
            (arguments,log.replace('top-k = 16','top-k = 8')),
            (arguments,log.replace('1000.00','900.00')),
            (arguments,log+'load_tensors: CPU model buffer size = 1.00 MiB\n'),
            (arguments,log.replace('(f16)','(q8_0)'))):
            with self.subTest(arguments=changed_args,log=changed_log):
                with self.assertRaises(ValueError):
                    verify_dflash_load(candidate,changed_args,changed_log)

    def test_dflash_candidates_keep_eleven_blocks_and_compensate_weight_budget(self):
        import re
        import numpy as np
        from desktop_agent.benchmark_placement import dflash_candidates,dflash_headroom_pair,mtp_candidates
        weights = {f'blk.{index}.{kind}.weight':dict(bytes=(index+size)*1024**2)
                   for index in range(65) for kind,size in
                   (('ffn_gate',30),('ffn_up',31),('ffn_down',32),('attn_q',12),('attn_norm',1))}
        weights.update({'output.weight':dict(bytes=700*1024**2),'output_norm.weight':dict(bytes=1024),
                        'token_embd.weight':dict(bytes=400*1024**2),'blk.64.nextn.eh_proj.weight':dict(bytes=80*1024**2)})
        original = next(item for item in mtp_candidates(weights) if item['name']=='attention55_baseline')
        base,swapped = dflash_candidates(weights,'draft.gguf',{'fc.weight':dict(bytes=1024**3)})
        short = dflash_candidates(weights,'draft.gguf',{'fc.weight':dict(bytes=1024**3)},draft_n_max=3)
        self.assertEqual(short,[dict(item,name=item['name']+'_n3',draft_n_max=3) for item in (base,swapped)])
        corrected = dflash_candidates(weights,'draft.gguf',{'fc.weight':dict(bytes=1024**3)},draft_n_max=3,extra_ffn50=True)
        control,headroom = dflash_headroom_pair(weights,'draft.gguf',{'fc.weight':dict(bytes=1024**3)})
        self.assertEqual(control,original)
        self.assertNotIn('draft_model',control)
        added_pair = {f'blk.{block}.ffn_{kind}.weight' for block in (48,49) for kind in ('gate','up','down')}
        pair_mib = sum(weights[name]['bytes'] for name in added_pair)/1024**2
        self.assertEqual(set(headroom['secondary_tensors']),set(corrected[0]['secondary_tensors'])|added_pair)
        self.assertEqual(headroom['primary_weight_mib'],corrected[0]['primary_weight_mib']-pair_mib)
        self.assertEqual(headroom['secondary_weight_mib'],corrected[0]['secondary_weight_mib']+pair_mib)
        self.assertEqual(headroom['extra_vs_previous_a_mib'],pair_mib)
        self.assertEqual(headroom['additional_ffn_blocks'],[48,49,50])
        self.assertEqual(len(headroom['secondary_tensors']),len(set(headroom['secondary_tensors'])))
        for key in ('whole_blocks','devices','tensor_split','cache_k','cache_v','draft_n_max',
                    'draft_cache_k','draft_cache_v','draft_model','draft_device','draft_weight_mib'):
            self.assertEqual(headroom[key],corrected[0][key])
        for name in added_pair:
            self.assertTrue(any(re.fullmatch(rule.rsplit('=',1)[0],name)
                                for rule in headroom['override'].split(',') if rule.endswith('=Vulkan0')))
        for before,after in zip(short,corrected):
            added = {f'blk.50.ffn_{kind}.weight' for kind in ('gate','up','down')}
            extra_mib = sum(weights[name]['bytes'] for name in added)/1024**2
            self.assertEqual(after['name'],before['name']+'_ffn50')
            self.assertEqual(set(after['secondary_tensors']),set(before['secondary_tensors'])|added)
            self.assertEqual(after['primary_weight_mib'],before['primary_weight_mib']-extra_mib)
            self.assertEqual(after['secondary_weight_mib'],before['secondary_weight_mib']+extra_mib)
            self.assertEqual(after['whole_blocks'],before['whole_blocks'])
            self.assertEqual(after['tensor_split'],before['tensor_split'])
            self.assertEqual(after['ffn_blocks'],sorted({50,*before['ffn_blocks']}))
        with self.assertRaisesRegex(ValueError,'lengths'):
            dflash_candidates(weights,'draft.gguf',{'fc.weight':dict(bytes=1024)},draft_n_max=8)
        self.assertEqual(base['secondary_tensors'],original['secondary_tensors'])
        self.assertEqual(base['draft_weight_mib'],1024)
        self.assertEqual(swapped['whole_blocks'],[5,19,33,47,53,56,57,58,60,61,62])
        self.assertLessEqual(swapped['primary_weight_mib'],base['primary_weight_mib'])
        self.assertAlmostEqual(swapped['primary_weight_mib']+swapped['secondary_weight_mib'],
                               base['primary_weight_mib']+base['secondary_weight_mib'])
        splits = np.cumsum(np.array(list(map(int,swapped['tensor_split'].split(','))),dtype=np.float32))/np.float32(66)
        devices = swapped['devices'].split(',')
        self.assertLessEqual(len(devices),15)
        selected = [devices[int(np.searchsorted(splits,np.float32(block)/np.float32(66),side='right'))] for block in range(66)]
        self.assertEqual([block for block in range(64) if selected[block]=='Vulkan0'],swapped['whole_blocks'])
        self.assertEqual(selected[65],'CUDA0')
        self.assertIn(r'^output(_norm)?\.weight$=Vulkan0',swapped['override'])
        assigned = set()
        rules = [rule.rsplit('=',1) for rule in swapped['override'].split(',')]
        for name in weights:
            if name.startswith('blk.64.'):
                continue
            device = selected[int(name.split('.')[1])] if name.startswith('blk.') else selected[65]
            for pattern,override_device in rules:
                if re.fullmatch(pattern,name):
                    device = override_device
                    break
            if device=='Vulkan0':
                assigned.add(name)
        self.assertEqual(assigned,set(swapped['secondary_tensors']))
        self.assertEqual(len(swapped['secondary_tensors']),len(set(swapped['secondary_tensors'])))
        for candidate in (base,swapped):
            self.assertEqual(candidate['secondary_weight_mib'],sum(weights[name]['bytes'] for name in candidate['secondary_tensors'])/1024**2)
            self.assertEqual(candidate['cpu_weight_mib'],original['cpu_weight_mib'])
            self.assertTrue(candidate['verify_dflash'])
            self.assertNotIn('blk.64.nextn.eh_proj.weight',candidate['secondary_tensors'])
        with self.assertRaisesRegex(ValueError,'sharing'):
            dflash_candidates(weights,'draft.gguf',{'output.weight':dict(bytes=1024)})

    def test_draft_model_options_preserve_target_cache_and_device(self):
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import PlacementServer
        candidate = dict(override=r'^token_embd\.weight$=CPU',threads=12,batch_threads=6,ubatch=128,
            cache_k='q8_0',cache_v='q8_0',devices='CUDA0,Vulkan0',spec_type='draft-dflash',
            draft_model='G:/models/dflash2-q4.gguf',draft_device='Vulkan0',draft_gpu_layers='all',
            draft_n_max=7,draft_cache_k='f16',draft_cache_v='f16',draft_override=r'^token_embd\.weight$=CPU')
        settings = Settings()
        options = PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        for flag,value in (('--spec-draft-model',candidate['draft_model']),('--spec-draft-device','Vulkan0'),
                           ('--spec-draft-ngl','all'),('--spec-draft-n-max','7'),('--spec-type','draft-dflash'),
                           ('--spec-draft-type-k','f16'),('--spec-draft-type-v','f16'),
                           ('--spec-draft-override-tensor',candidate['draft_override']),
                           ('-ctk','q8_0'),('-ctv','q8_0'),('--device','CUDA0,Vulkan0')):
            with self.subTest(flag=flag):
                self.assertEqual(options.count(flag),1)
                self.assertEqual(options[options.index(flag)+1],value)

    def test_graphics_whole_blocks_keep_cuda_weight_budget_and_cpu_embedding(self):
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import mtp_candidates,PlacementServer
        weights = {f'blk.{index}.{kind}.weight':dict(bytes=size*1024**2)
                   for index in range(65) for kind,size in
                   (('ffn_gate',30),('ffn_up',30),('ffn_down',30),('ssm_in',25),('attn_norm',1))}
        weights.update({'output.weight':dict(bytes=700*1024**2),'output_norm.weight':dict(bytes=1024),
                        'token_embd.weight':dict(bytes=400*1024**2),'blk.64.nextn.eh_proj.weight':dict(bytes=80*1024**2)})
        candidates = {candidate['name']:candidate for candidate in mtp_candidates(weights)}
        baseline = candidates['graphics_baseline']
        for name in ('graphics_blocks_budget','graphics_blocks_extra1'):
            candidate = candidates[name]
            first = candidate['whole_blocks'][0]
            self.assertEqual(candidate['whole_blocks'],list(range(first,64)))
            self.assertEqual(candidate['ffn_blocks'],candidate['whole_blocks'])
            expected = {tensor for tensor in weights if tensor in ('output.weight','output_norm.weight') or
                        (tensor.startswith('blk.') and first <= int(tensor.split('.')[1]) < 64)}
            self.assertEqual(set(candidate['secondary_tensors']),expected)
            self.assertEqual(candidate['secondary_weight_mib'],sum(weights[tensor]['bytes'] for tensor in expected)/1024**2)
            self.assertLessEqual(candidate['primary_weight_mib'],baseline['primary_weight_mib'])
            self.assertAlmostEqual(candidate['primary_weight_mib']+candidate['secondary_weight_mib'],
                                   baseline['primary_weight_mib']+baseline['secondary_weight_mib'])
            for key in ('cpu_weight_mib','backend_environment','cache_k','cache_v','spec_type','ubatch',
                        'cuda_scale_launch_queues','request_limit','slow_limit','projector_device'):
                self.assertEqual(candidate[key],baseline[key])
            settings = Settings()
            options = PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            self.assertEqual(options[options.index('--override-tensor')+1],r'^token_embd\.weight$=CPU')
            self.assertEqual(options[options.index('--tensor-split')+1],f'{first},{66-first}')
            self.assertTrue(candidate['verify_weight_buffers'])
            import numpy as np
            fraction = np.float32(first)/np.float32(66)
            self.assertEqual([block for block in range(64) if np.float32(block)/np.float32(66)>=fraction],
                             candidate['whole_blocks'])
        self.assertEqual(candidates['graphics_blocks_extra1']['whole_blocks'][0],
                         candidates['graphics_blocks_budget']['whole_blocks'][0]-1)
        for count in (4,8,10,11,12):
            import re
            candidate = candidates[f'graphics_hybrid{count}']
            first = 64-count
            self.assertEqual(candidate['whole_blocks'],list(range(first,64)))
            self.assertLessEqual(candidate['primary_weight_mib'],baseline['primary_weight_mib'])
            self.assertAlmostEqual(candidate['primary_weight_mib']+candidate['secondary_weight_mib'],
                                   baseline['primary_weight_mib']+baseline['secondary_weight_mib'])
            patterns = [item.split('=')[0] for item in candidate['override'].split(',') if item.endswith('=Vulkan0')]
            assigned = {tensor for tensor in weights if tensor in ('output.weight','output_norm.weight') or
                        (tensor.startswith('blk.') and first <= int(tensor.split('.')[1]) < 64) or
                        any(re.fullmatch(pattern,tensor) for pattern in patterns)}
            self.assertEqual(assigned,set(candidate['secondary_tensors']))
            self.assertFalse(any(tensor.startswith('blk.64.') for tensor in assigned))
            self.assertEqual(candidate['tensor_split'],f'{first},{66-first}')

    def test_compact_hybrid9_moves_exact_tail_and_five_ffns(self):
        import re
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import mtp_candidates,PlacementServer
        weights = {f'blk.{index}.{kind}.weight':dict(bytes=(index+size)*1024**2)
                   for index in range(65) for kind,size in
                   (('ffn_gate',30),('ffn_up',31),('ffn_down',32),('attn_q',12),('attn_norm',1))}
        weights.update({'output.weight':dict(bytes=700*1024**2),'output_norm.weight':dict(bytes=1024),
                        'token_embd.weight':dict(bytes=400*1024**2),'blk.64.nextn.eh_proj.weight':dict(bytes=80*1024**2)})
        candidates = {item['name']:item for item in mtp_candidates(weights)}
        candidate = candidates['compact_hybrid9']
        expected = {name for name in weights if name in ('output.weight','output_norm.weight') or
                    (name.startswith('blk.') and 55<=int(name.split('.')[1])<64) or
                    re.fullmatch(r'blk\.5[0-4]\.ffn_(gate|up|down)\.weight',name)}
        self.assertEqual(set(candidate['secondary_tensors']),expected)
        self.assertEqual(len(candidate['secondary_tensors']),len(expected))
        self.assertEqual(candidate['whole_blocks'],list(range(55,64)))
        self.assertEqual(candidate['ffn_blocks'],list(range(50,64)))
        self.assertEqual(candidate['secondary_weight_mib'],sum(weights[name]['bytes'] for name in expected)/1024**2)
        self.assertAlmostEqual(sum(candidate[key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')),
                               sum(candidates['graphics_hybrid8'][key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')))
        for key in ('backend_environment','cache_k','cache_v','ubatch','batch','request_limit','slow_limit','vram_guard','verify_weight_buffers'):
            self.assertEqual(candidate[key],candidates['graphics_hybrid8'][key])
        settings = Settings()
        options = PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        self.assertEqual(options[options.index('--tensor-split')+1],'55,11')
        self.assertEqual(options[options.index('--override-tensor')+1],candidate['override'])
        self.assertEqual(candidates['compact9_no_host_visible'],dict(candidate,name='compact9_no_host_visible',
            backend_environment=dict(candidate['backend_environment'],GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM='1')))
        for name,reference_name in (('compact12_no_host_visible','graphics_hybrid12'),
                                   ('compact13_no_host_visible','graphics_blocks_budget')):
            reference = candidates[reference_name]
            self.assertEqual(candidates[name],dict(reference,name=name,
                backend_environment=dict(reference['backend_environment'],GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM='1')))
        self.assertEqual(candidates['compact9_vk_profile'],dict(candidate,name='compact9_vk_profile',diagnostic_only=True,
            backend_environment=dict(candidate['backend_environment'],GGML_VK_PERF_LOGGER='1',
                GGML_VK_PERF_LOGGER_CONCURRENT='1',GGML_VK_PERF_LOGGER_FREQUENCY='100')))
        hybrid12 = candidates['compact12_no_host_visible']
        self.assertEqual(candidates['hybrid12_baseline'],dict(hybrid12,name='hybrid12_baseline'))
        self.assertEqual(candidates['hybrid12_no_fusion'],dict(hybrid12,name='hybrid12_no_fusion',
            backend_environment=dict(hybrid12['backend_environment'],GGML_VK_DISABLE_FUSION='1')))
        for suffix,cache in (('q4','q4_0'),('f16','f16')):
            changed = candidates['hybrid12_kv_'+suffix]
            self.assertEqual(changed,dict(hybrid12,name='hybrid12_kv_'+suffix,
                cache_k=cache,cache_v=cache,draft_cache_k=cache,draft_cache_v=cache))
            options = PlacementServer(changed).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            for flag in ('-ctk','-ctv','--spec-draft-type-k','--spec-draft-type-v'):
                self.assertEqual(options.count(flag),1)
                self.assertEqual(options[options.index(flag)+1],cache)
        swapped = candidates['hybrid12_ffn_swap51_50']
        swapped_expected = {name for name in hybrid12['secondary_tensors'] if not re.fullmatch(r'blk\.51\.ffn_(gate|up|down)\.weight',name)}
        swapped_expected.update(f'blk.50.ffn_{kind}.weight' for kind in ('gate','up','down'))
        self.assertEqual(set(swapped['secondary_tensors']),swapped_expected)
        self.assertEqual(len(swapped['secondary_tensors']),len(swapped_expected))
        self.assertEqual(swapped['secondary_weight_mib'],sum(weights[name]['bytes'] for name in swapped_expected)/1024**2)
        self.assertAlmostEqual(swapped['primary_weight_mib']+swapped['secondary_weight_mib'],
                               hybrid12['primary_weight_mib']+hybrid12['secondary_weight_mib'])
        self.assertEqual(swapped['ffn_blocks'],sorted({50,*[block for block in hybrid12['ffn_blocks'] if block!=51]}))
        patterns = [item.split('=')[0] for item in swapped['override'].split(',') if item.endswith('=Vulkan0')]
        assigned = {name for name in weights if name in ('output.weight','output_norm.weight') or
                (name.startswith('blk.') and int(name.split('.')[1]) in hybrid12['whole_blocks']) or
                any(re.fullmatch(pattern,name) for pattern in patterns)}
        self.assertEqual(assigned,swapped_expected)
        for key in ('whole_blocks','tensor_split','backend_environment','cache_k','cache_v','vram_guard','verify_weight_buffers'):
            self.assertEqual(swapped[key],hybrid12[key])
        attention = candidates['hybrid12_attention55_cuda']
        returned = {name for name in hybrid12['secondary_tensors'] if re.fullmatch(r'blk\.55\.attn_.*\.weight',name)}
        self.assertTrue(returned)
        self.assertEqual(set(attention['secondary_tensors']),set(hybrid12['secondary_tensors'])-returned)
        delta = sum(weights[name]['bytes'] for name in returned)/1024**2
        self.assertAlmostEqual(attention['primary_weight_mib'],hybrid12['primary_weight_mib']+delta)
        self.assertAlmostEqual(attention['secondary_weight_mib'],hybrid12['secondary_weight_mib']-delta)
        self.assertEqual(attention['whole_blocks'],[block for block in hybrid12['whole_blocks'] if block!=55])
        self.assertEqual(attention['layer_split_blocks'],hybrid12['whole_blocks'])
        self.assertEqual(attention['ffn_blocks'],hybrid12['ffn_blocks'])
        self.assertEqual(attention['override'],hybrid12['override']+r',^blk\.55\.attn_.*\.weight$=CUDA0')
        from unittest.mock import patch
        layer = candidates['hybrid12_attention55_layer']
        self.assertEqual(candidates['attention55_baseline'],dict(layer,name='attention55_baseline'))
        self.assertEqual(candidates['attention55_vk_profile'],dict(layer,name='attention55_vk_profile',
            diagnostic_only=True,diagnostic_image_only=True,stop_on_quality_failure=True,
            backend_environment=dict(layer['backend_environment'],GGML_VK_PERF_LOGGER='1',
                GGML_VK_PERF_LOGGER_CONCURRENT='1',GGML_VK_PERF_LOGGER_FREQUENCY='100')))
        both = candidates['attention55_63_cuda_ffn50']
        moved63 = {name for name in layer['secondary_tensors'] if re.fullmatch(r'blk\.63\.attn_.*\.weight',name)}
        moved63_mib = sum(weights[name]['bytes'] for name in moved63)/1024**2
        compensation50 = {f'blk.50.ffn_{kind}.weight' for kind in ('gate','up','down')}
        compensation50_mib = sum(weights[name]['bytes'] for name in compensation50)/1024**2
        self.assertTrue(moved63)
        self.assertEqual(set(both['secondary_tensors']),(set(layer['secondary_tensors'])-moved63)|compensation50)
        self.assertAlmostEqual(both['primary_weight_mib'],layer['primary_weight_mib']+moved63_mib-compensation50_mib)
        self.assertAlmostEqual(both['secondary_weight_mib'],layer['secondary_weight_mib']-moved63_mib+compensation50_mib)
        self.assertEqual(both['ffn_blocks'],sorted({50,*layer['ffn_blocks']}))
        self.assertEqual(both['additional_ffn_blocks'],[50])
        self.assertTrue(both['stop_on_quality_failure'])
        self.assertEqual(both['whole_blocks'],[block for block in layer['whole_blocks'] if block!=63])
        self.assertEqual(both['returned_attention_blocks'],[55,63])
        self.assertEqual(both['tensor_split'],'52,3,1,7,1,2')
        import numpy as np
        split_both = np.cumsum(np.array([52,3,1,7,1,2],dtype=np.float32))/np.float32(66)
        devices_both = both['devices'].split(',')
        self.assertEqual([block for block in range(64) if devices_both[int(np.searchsorted(
            split_both,np.float32(block)/np.float32(66),side='right'))]=='Vulkan0'],both['whole_blocks'])
        self.assertEqual(candidates['attention55_rx_ffn55_f16'],dict(layer,name='attention55_rx_ffn55_f16',
            required_tensor_types={f'blk.55.ffn_{kind}.weight':'F16' for kind in ('gate','up','down')},
            stop_on_quality_failure=True))
        self.assertEqual(candidates['attention55_rx_ffn55_q8'],dict(layer,name='attention55_rx_ffn55_q8',
            required_tensor_types={f'blk.55.ffn_{kind}.weight':'Q8_0' for kind in ('gate','up','down')},
            stop_on_quality_failure=True))
        self.assertEqual(candidates['attention55_rx_gate51_q8'],dict(layer,name='attention55_rx_gate51_q8',
            required_tensor_types={'blk.51.ffn_gate.weight':'Q8_0'},stop_on_quality_failure=True))
        self.assertEqual(candidates['attention55_ub64'],dict(layer,name='attention55_ub64',ubatch=64))
        self.assertEqual(candidates['attention55_ub64_blocking_schedule'],dict(layer,
            name='attention55_ub64_blocking_schedule',ubatch=64,diagnostic_only=True,diagnostic_image_only=True,
            backend_environment=dict(layer['backend_environment'],CUDA_LAUNCH_BLOCKING='1',
                                     CUDA_LOG_FILE='stderr',GGML_SCHED_DEBUG='2')))
        self.assertEqual(candidates['attention55_ub256'],dict(layer,name='attention55_ub256',ubatch=256))
        virtual = candidates['attention55_virtual_cuda']
        self.assertEqual(virtual,dict(layer,name='attention55_virtual_cuda',
            devices='CUDA0,Vulkan0,CUDA1,Vulkan0',
            backend_environment=dict(layer['backend_environment'],GGML_CUDA_DEVICES='2'),
            expected_weight_buffers_mib={'CPU':layer['cpu_weight_mib'],
                'CUDA0':layer['primary_weight_mib']-delta,'CUDA1':delta,'Vulkan0':layer['secondary_weight_mib']}))
        self.assertAlmostEqual(sum(virtual['expected_weight_buffers_mib'][key] for key in ('CUDA0','CUDA1')),
                               layer['primary_weight_mib'])
        virtual_queue = candidates['attention55_virtual_queue2x']
        self.assertEqual(virtual_queue,dict(virtual,name='attention55_virtual_queue2x',cuda_scale_launch_queues='2x'))
        virtual_q4 = candidates['attention55_virtual_kv_q4']
        self.assertEqual(virtual_q4,dict(virtual,name='attention55_virtual_kv_q4',
            cache_k='q4_0',cache_v='q4_0',draft_cache_k='q4_0',draft_cache_v='q4_0',verify_kv_cache=True))
        q4_options = PlacementServer(virtual_q4).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        for flag in ('-ctk','-ctv','--spec-draft-type-k','--spec-draft-type-v'):
            self.assertEqual(q4_options.count(flag),1)
            self.assertEqual(q4_options[q4_options.index(flag)+1],'q4_0')
        vision1 = candidates['attention55_virtual_vision1']
        self.assertEqual(vision1,dict(virtual,name='attention55_virtual_vision1',projector_device='CUDA1'))
        vision_options = PlacementServer(vision1).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        self.assertEqual(vision_options.count('--mmproj-device'),1)
        self.assertEqual(vision_options[vision_options.index('--mmproj-device')+1],'CUDA1')
        reduced = candidates['attention55_virtual_ffn50']
        reduced_expected = set(virtual['secondary_tensors'])|{name for name in weights if re.fullmatch(r'blk\.50\.ffn_(gate|up|down)\.weight',name)}
        extra_mib = sum(weights[name]['bytes'] for name in reduced_expected-set(virtual['secondary_tensors']))/1024**2
        self.assertEqual(set(reduced['secondary_tensors']),reduced_expected)
        self.assertEqual(len(reduced['secondary_tensors']),len(reduced_expected))
        self.assertEqual(reduced['ffn_blocks'],sorted({50,*virtual['ffn_blocks']}))
        self.assertAlmostEqual(reduced['primary_weight_mib'],virtual['primary_weight_mib']-extra_mib)
        self.assertAlmostEqual(reduced['secondary_weight_mib'],virtual['secondary_weight_mib']+extra_mib)
        self.assertEqual(reduced['expected_weight_buffers_mib'],dict(virtual['expected_weight_buffers_mib'],
            CUDA0=virtual['expected_weight_buffers_mib']['CUDA0']-extra_mib,
            Vulkan0=virtual['expected_weight_buffers_mib']['Vulkan0']+extra_mib))
        for key in ('devices','tensor_split','whole_blocks','layer_split_blocks','projector_device','backend_environment'):
            self.assertEqual(reduced[key],virtual[key])
        self.assertEqual(reduced['override'],virtual['override']+r',^blk\.50\.ffn_(gate|up|down)\.weight$=Vulkan0')
        graph_opt = candidates['attention55_cuda_graph_opt']
        self.assertEqual(graph_opt,dict(layer,name='attention55_cuda_graph_opt',
            backend_environment=dict(layer['backend_environment'],GGML_CUDA_GRAPH_OPT='1')))
        with patch('desktop_agent.models.Path.is_file',return_value=True):
            inherited = dict(os.environ)
            environment = PlacementServer(graph_opt).environment_for('Qwen3.8-27B-UD-Q2_K_XL.gguf','mmproj-Qwen3.8-27B-Q8_0.gguf')
            self.assertEqual(environment['GGML_CUDA_GRAPH_OPT'],'1')
            virtual_environment = PlacementServer(virtual).environment_for('Qwen3.8-27B-UD-Q2_K_XL.gguf','mmproj-Qwen3.8-27B-Q8_0.gguf')
            self.assertEqual(virtual_environment['GGML_CUDA_DEVICES'],'2')
            queue_server = PlacementServer(virtual_queue)
            with patch.dict(os.environ,{'CUDA_SCALE_LAUNCH_QUEUES':'4x'}):
                queue_environment = queue_server.environment_for('Qwen3.8-27B-UD-Q2_K_XL.gguf','mmproj-Qwen3.8-27B-Q8_0.gguf')
                self.assertEqual(os.environ['CUDA_SCALE_LAUNCH_QUEUES'],'4x')
                self.assertEqual(queue_environment,dict(virtual_environment,CUDA_SCALE_LAUNCH_QUEUES='2x'))
                self.assertEqual(queue_server.launch_queue_environment['CUDA_SCALE_LAUNCH_QUEUES'],'2x')
            with patch.dict(os.environ,{'GGML_CUDA_GRAPH_OPT':'1','GGML_CUDA_DEVICES':'2'}):
                environment = PlacementServer(layer).environment_for('Qwen3.8-27B-UD-Q2_K_XL.gguf','mmproj-Qwen3.8-27B-Q8_0.gguf')
                self.assertNotIn('GGML_CUDA_GRAPH_OPT',environment)
                self.assertNotIn('GGML_CUDA_DEVICES',environment)
            self.assertEqual(dict(os.environ),inherited)
        self.assertEqual(layer,dict(attention,name='hybrid12_attention55_layer',
            devices='CUDA0,Vulkan0,CUDA0,Vulkan0',tensor_split='52,3,1,10',
            layer_split_blocks=attention['whole_blocks'],
            override=hybrid12['override']+r',^blk\.55\.(ffn_(gate|up|down)|post_attention_norm)\.weight$=Vulkan0'))
        import numpy as np
        splits = np.cumsum(np.array([52,3,1,10],dtype=np.float32))/np.float32(66)
        devices = layer['devices'].split(',')
        selected = [devices[int(np.searchsorted(splits,np.float32(block)/np.float32(66),side='right'))] for block in range(64)]
        self.assertEqual([block for block,device in enumerate(selected) if device=='Vulkan0'],layer['whole_blocks'])
        patterns = [rule.rsplit('=',1) for rule in layer['override'].split(',')]
        assigned = set()
        for name in weights:
            if name.startswith('blk.') and int(name.split('.')[1])>=64:
                continue
            device = selected[int(name.split('.')[1])] if name.startswith('blk.') else 'Vulkan0'
            for pattern,target in patterns:
                if re.fullmatch(pattern,name):
                    device = target
                    break
            if device=='Vulkan0':
                assigned.add(name)
        self.assertEqual(assigned,set(attention['secondary_tensors']))
        pair = candidates['hybrid12_attention55_59_layer']
        pair_expected = {name for name in hybrid12['secondary_tensors'] if not re.fullmatch(r'blk\.(55|59)\.attn_.*\.weight',name)}
        pair_expected.update(f'blk.50.ffn_{kind}.weight' for kind in ('gate','up','down'))
        self.assertEqual(set(pair['secondary_tensors']),pair_expected)
        self.assertEqual(len(pair['secondary_tensors']),len(pair_expected))
        self.assertAlmostEqual(pair['primary_weight_mib']+pair['secondary_weight_mib'],
                               hybrid12['primary_weight_mib']+hybrid12['secondary_weight_mib'])
        splits = np.cumsum(np.array([52,3,1,3,1,6],dtype=np.float32))/np.float32(66)
        devices = pair['devices'].split(',')
        selected = [devices[int(np.searchsorted(splits,np.float32(block)/np.float32(66),side='right'))] for block in range(64)]
        self.assertEqual([block for block,device in enumerate(selected) if device=='Vulkan0'],pair['layer_split_blocks'])
        assigned = set()
        for name in weights:
            if name.startswith('blk.') and int(name.split('.')[1])>=64:
                continue
            device = selected[int(name.split('.')[1])] if name.startswith('blk.') else 'Vulkan0'
            for pattern,target in (rule.rsplit('=',1) for rule in pair['override'].split(',')):
                if re.fullmatch(pattern,name):
                    device = target
                    break
            if device=='Vulkan0':
                assigned.add(name)
        self.assertEqual(assigned,pair_expected)
        for config in (layer,pair):
            options = PlacementServer(config).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            for flag,key in (('--device','devices'),('--tensor-split','tensor_split'),('--override-tensor','override')):
                self.assertEqual(options.count(flag),1)
                self.assertEqual(options[options.index(flag)+1],config[key])
        for suffix,parts in (('layer',[52,11,1,2]),('tail',[52,11,3])):
            config = candidates['hybrid12_attention63_'+suffix]
            expected = {name for name in hybrid12['secondary_tensors'] if not re.fullmatch(r'blk\.63\.attn_.*\.weight',name)}
            self.assertEqual(set(config['secondary_tensors']),expected)
            self.assertEqual(len(config['secondary_tensors']),len(expected))
            self.assertEqual(config['secondary_weight_mib'],sum(weights[name]['bytes'] for name in expected)/1024**2)
            self.assertAlmostEqual(config['primary_weight_mib']+config['secondary_weight_mib'],
                                   hybrid12['primary_weight_mib']+hybrid12['secondary_weight_mib'])
            self.assertEqual(config['ffn_blocks'],hybrid12['ffn_blocks'])
            splits = np.cumsum(np.array(parts,dtype=np.float32))/np.float32(66)
            devices = config['devices'].split(',')
            selected = [devices[int(np.searchsorted(splits,np.float32(block)/np.float32(66),side='right'))] for block in range(66)]
            self.assertEqual([block for block in range(64) if selected[block]=='Vulkan0'],list(range(52,63)))
            self.assertEqual(selected[65],'Vulkan0' if suffix=='layer' else 'CUDA0')
            assigned = set()
            for name in weights:
                if name.startswith('blk.') and int(name.split('.')[1])>=64:
                    continue
                device = selected[int(name.split('.')[1])] if name.startswith('blk.') else selected[65]
                for pattern,target in (rule.rsplit('=',1) for rule in config['override'].split(',')):
                    if re.fullmatch(pattern,name):
                        device = target
                        break
                if device=='Vulkan0':
                    assigned.add(name)
            self.assertEqual(assigned,expected)
            options = PlacementServer(config).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            for flag,key in (('--device','devices'),('--tensor-split','tensor_split'),('--override-tensor','override')):
                self.assertEqual(options.count(flag),1)
                self.assertEqual(options[options.index(flag)+1],config[key])
        for name in ('hybrid12_baseline','hybrid12_attention55_cuda','hybrid12_attention55_layer','hybrid12_attention55_59_layer',
                     'hybrid12_attention63_layer','hybrid12_attention63_tail','attention55_cuda_graph_opt',
                     'attention55_virtual_cuda','attention55_virtual_vision1'):
            reference = candidates[name]
            diagnostic = candidates[name+'_schedule']
            self.assertEqual(diagnostic,dict(reference,name=name+'_schedule',diagnostic_only=True,diagnostic_image_only=True,
                backend_environment=dict(reference['backend_environment'],GGML_SCHED_DEBUG='2')))
            with patch('desktop_agent.agent.server_environment',return_value={'GGML_SCHED_DEBUG':'inherited'}):
                self.assertEqual(PlacementServer(diagnostic).environment_for(settings.model,settings.projector)['GGML_SCHED_DEBUG'],'2')
                self.assertNotIn('GGML_SCHED_DEBUG',PlacementServer(reference).environment_for(settings.model,settings.projector))
            for config,verbosity in ((diagnostic,'5'),(reference,'4')):
                options = PlacementServer(config).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
                self.assertEqual(options[options.index('-lv')+1],verbosity)
        for suffix,key,value,flag in (('ub64','ubatch',64,'-ub'),('batch128','batch',128,'-b')):
            changed = candidates['hybrid12_'+suffix]
            self.assertEqual(changed,dict(hybrid12,name='hybrid12_'+suffix,**{key:value}))
            options = PlacementServer(changed).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            self.assertEqual(options.count(flag),1)
            self.assertEqual(options[options.index(flag)+1],str(value))
        self.assertEqual(candidates['hybrid12_vk_profile'],dict(hybrid12,name='hybrid12_vk_profile',diagnostic_only=True,
            backend_environment=dict(hybrid12['backend_environment'],GGML_VK_PERF_LOGGER='1',
                GGML_VK_PERF_LOGGER_CONCURRENT='1',GGML_VK_PERF_LOGGER_FREQUENCY='100')))

    def test_optional_ram_cache_defaults_persists_and_reloads_actual_process(self):
        import threading
        from dataclasses import replace
        from unittest.mock import Mock,patch
        from desktop_agent.agent import Model,Settings
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            path.write_text('{}',encoding='utf-8')
            settings = Settings.load(path)
            self.assertEqual(settings.cache_ram_mib,0)
            model = Model(settings)
            processes = []
            def spawn(*args,**kwargs):
                process = Mock()
                process.poll.return_value = None
                processes.append(process)
                return process
            try:
                with patch('desktop_agent.agent.HOME',Path(folder)), \
                     patch('game_agent.runtime.Path.is_file',return_value=True), \
                     patch('desktop_agent.agent.server_environment',return_value={}), \
                     patch('game_agent.runtime.subprocess.Popen',side_effect=spawn) as popen, \
                     patch('game_agent.runtime.session') as client:
                    client.return_value.__enter__.return_value.get.return_value.status_code = 200
                    for index,value in enumerate((0,2048,0)):
                        model.settings = replace(settings,cache_ram_mib=value)
                        model.settings.save(path)
                        self.assertEqual(Settings.load(path),model.settings)
                        model.ensure(threading.Event())
                        arguments = popen.call_args.args[0]
                        self.assertEqual(arguments.count('--cache-ram'),1)
                        self.assertEqual(arguments[arguments.index('--cache-ram')+1],str(value))
                        self.assertEqual(popen.call_count,index+1)
                        if index:
                            processes[index-1].terminate.assert_called_once()
                        model.ensure(threading.Event())
                        self.assertEqual(popen.call_count,index+1)
                for value in (-1,1024,4096,True,'2048',None):
                    replace(settings,cache_ram_mib=value).save(path)
                    with self.assertRaisesRegex(ValueError,'RAM prompt cache'):
                        Settings.load(path)
                    model.server.cache_ram_mib = value
                    with self.assertRaisesRegex(ValueError,'RAM prompt cache'):
                        model.server.settings_for(settings.executable,settings.model,settings.projector)
            finally:
                model.server.close()

    def test_cache_ram_research_value_reaches_process_without_trailing_override(self):
        import threading
        from unittest.mock import Mock,patch
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import PlacementServer
        from desktop_agent.models import Q2_XL
        settings = Settings()
        with tempfile.TemporaryDirectory() as folder:
            for value in (None,0,1024,2048):
                candidate = dict(override=r'^token_embd\.weight$=CPU',threads=12,batch_threads=6,ubatch=128)
                if value is not None:
                    candidate['cache_ram'] = value
                server = PlacementServer(candidate)
                process = Mock()
                process.poll.return_value = None
                try:
                    with patch('game_agent.runtime.Path.is_file',return_value=True), \
                         patch('desktop_agent.agent.server_environment',return_value={}), \
                         patch('game_agent.runtime.subprocess.Popen',return_value=process) as popen, \
                         patch('game_agent.runtime.session') as client:
                        client.return_value.__enter__.return_value.get.return_value.status_code = 200
                        server.start(settings.executable,Q2_XL['model'],Q2_XL['projector'],
                                     Path(folder)/f'{value}.log',threading.Event(),1024,1024)
                        arguments = popen.call_args.args[0]
                        self.assertEqual(arguments.count('--cache-ram'),1)
                        self.assertEqual(arguments[arguments.index('--cache-ram')+1],str(0 if value is None else value))
                finally:
                    server.close()

    def test_context_switch_research_reuses_exact_requests_and_invalidates_changed_record(self):
        from desktop_agent.benchmark_placement import context_switch_workload
        first,answer = context_switch_workload(0)
        repeated,repeated_answer = context_switch_workload(2)
        changed,changed_answer = context_switch_workload(4)
        self.assertEqual((first,answer),(repeated,repeated_answer))
        self.assertEqual(answer,'A731')
        self.assertEqual(changed_answer,'A739')
        self.assertNotEqual(first,changed)
        self.assertEqual(context_switch_workload(4),context_switch_workload(6))
        self.assertEqual(context_switch_workload(1),context_switch_workload(7))
        self.assertTrue(all(event['role']=='user' for event in first+changed))
        self.assertNotIn(answer,first[-1]['content'])
        self.assertEqual(context_switch_workload(0,120),context_switch_workload(2,120))
        self.assertGreater(len(context_switch_workload(0,120)[0][0]['content']),len(first[0]['content']))
        self.assertEqual(context_switch_workload(4,120)[1],'A739')

    def test_q8_backend_research_isolates_child_environment_and_keeps_baseline(self):
        from unittest.mock import patch
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import mtp_candidates,PlacementServer
        weights = {f'blk.{index}.ffn_{kind}.weight':dict(bytes=40*1024**2)
                   for index in range(65) for kind in ('gate','up','down')}
        weights.update({'output.weight':dict(bytes=700*1024**2),'token_embd.weight':dict(bytes=400*1024**2),
                        'blk.64.nextn.eh_proj.weight':dict(bytes=80*1024**2)})
        candidates = {item['name']:item for item in mtp_candidates(weights)}
        reference = candidates['q8_off_baseline']
        self.assertEqual(reference,dict(candidates['mtp_off_q8'],name='q8_off_baseline',backend_environment={}))
        graphics = candidates['q8_off_vk_graphics']
        self.assertEqual(candidates['graphics_baseline'],dict(graphics,name='graphics_baseline'))
        self.assertEqual(candidates['graphics_baseline_end'],dict(graphics,name='graphics_baseline_end'))
        for ubatch in (64,256):
            self.assertEqual(candidates[f'graphics_ub{ubatch}'],dict(graphics,name=f'graphics_ub{ubatch}',ubatch=ubatch))
        self.assertEqual(candidates['graphics_queue2x'],dict(graphics,name='graphics_queue2x',cuda_scale_launch_queues='2x'))
        for count in (20,21):
            candidate = candidates[f'graphics_ffn{count}']
            self.assertEqual(candidate['ffn_blocks'],list(range(64-count,64)))
            self.assertAlmostEqual(graphics['primary_weight_mib']-candidate['primary_weight_mib'],120*(count-19))
            self.assertAlmostEqual(candidate['secondary_weight_mib']-graphics['secondary_weight_mib'],120*(count-19))
            self.assertEqual(candidate['backend_environment'],graphics['backend_environment'])
        settings = Settings()
        for suffix,cache_ram in (('baseline',0),('ram1024',1024),('ram2048',2048),('baseline_end',0)):
            name = 'context_switch_'+suffix
            self.assertEqual(candidates[name],dict(graphics,name=name,context_switch_study=True,cache_ram=cache_ram))
            long_name = 'context_switch_long_'+suffix
            self.assertEqual(candidates[long_name],dict(graphics,name=long_name,context_switch_study=True,
                                                      cache_ram=cache_ram,context_switch_rows=120))
        for name,key,value,flag in (
            ('graphics_batch128','batch',128,'-b'),('graphics_batch256','batch',256,'-b'),
            ('graphics_checkpoint256','checkpoint_min_step',256,'--checkpoint-min-step'),
            ('graphics_checkpoint1024','checkpoint_min_step',1024,'--checkpoint-min-step'),
            ('graphics_no_cache_ram','cache_ram',0,'--cache-ram'),
            ('graphics_cache2048','cache_ram',2048,'--cache-ram'),
            ('graphics_no_op_offload','op_offload',False,'--no-op-offload')):
            candidate = candidates[name]
            self.assertEqual(candidate,dict(graphics,name=name,**{key:value}))
            options = PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            self.assertEqual(options.count(flag),1)
            if key=='op_offload':
                self.assertNotIn('--op-offload',options)
            else:
                self.assertEqual(options[options.index(flag)+1],str(value))
        managed = PlacementServer.research_environment_values
        swapped = candidates['hybrid8_ffn_swap63_40']
        hybrid = candidates['graphics_hybrid8']
        self.assertEqual(swapped['ffn_blocks'],sorted([40]+[block for block in hybrid['ffn_blocks'] if block!=63]))
        self.assertEqual(swapped['whole_blocks'],list(range(56,63)))
        self.assertEqual(swapped['layer_split_blocks'],list(range(56,64)))
        self.assertEqual(swapped['tensor_split'],hybrid['tensor_split'])
        self.assertTrue(swapped['verify_weight_buffers'])
        for kind in ('gate','up','down'):
            self.assertNotIn(f'blk.63.ffn_{kind}.weight',swapped['secondary_tensors'])
            self.assertIn(f'blk.40.ffn_{kind}.weight',swapped['secondary_tensors'])
        self.assertEqual(swapped['backend_environment'],hybrid['backend_environment'])
        self.assertAlmostEqual(sum(swapped[key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')),
                               sum(hybrid[key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')))
        options = PlacementServer(swapped).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        self.assertEqual(options[options.index('--override-tensor')+1],swapped['override'])
        self.assertTrue(swapped['override'].endswith(r'^blk\.63\.ffn_(gate|up|down)\.weight$=CUDA0,^blk\.40\.ffn_(gate|up|down)\.weight$=Vulkan0'))
        unequal_weights = {name:dict(value) for name,value in weights.items()}
        unequal_weights['blk.63.ffn_gate.weight']['bytes'] += 2*1024**2
        unequal = {candidate['name']:candidate for candidate in mtp_candidates(unequal_weights)}
        self.assertAlmostEqual(unequal['hybrid8_ffn_swap63_40']['primary_weight_mib']-
                       unequal['graphics_hybrid8']['primary_weight_mib'],2)
        self.assertAlmostEqual(unequal['graphics_hybrid8']['secondary_weight_mib']-
                       unequal['hybrid8_ffn_swap63_40']['secondary_weight_mib'],2)
        profile = candidates['hybrid8_vk_profile']
        self.assertEqual(profile,dict(candidates['graphics_hybrid8'],name='hybrid8_vk_profile',diagnostic_only=True,
            backend_environment=dict(candidates['graphics_hybrid8']['backend_environment'],
                GGML_VK_PERF_LOGGER='1',GGML_VK_PERF_LOGGER_CONCURRENT='1',GGML_VK_PERF_LOGGER_FREQUENCY='100')))
        with patch.dict(os.environ,{key:'inherited' for key in managed}):
            before = dict(os.environ)
            for candidate in (profile,candidates['hybrid8_baseline']):
                environment = PlacementServer(candidate).environment_for(settings.model,settings.projector)
                self.assertEqual({key:environment.get(key) for key in managed},
                                 {key:candidate['backend_environment'].get(key) for key in managed})
            self.assertEqual(dict(os.environ),before)
        self.assertEqual(candidates['hybrid8_queue2x'],dict(candidates['graphics_hybrid8'],
                         name='hybrid8_queue2x',cuda_scale_launch_queues='2x'))
        for suffix,overrides in (
            ('cuda_graph_opt',{'GGML_CUDA_GRAPH_OPT':'1'}),
            ('submit50',{'GGML_VK_MAX_NODES_PER_SUBMIT':'50'}),
            ('submit200',{'GGML_VK_MAX_NODES_PER_SUBMIT':'200'}),
            ('transfer',{'GGML_VK_ASYNC_USE_TRANSFER_QUEUE':'1'})):
            reference_hybrid = candidates['graphics_hybrid8']
            candidate = candidates['hybrid8_'+suffix]
            self.assertEqual(candidate,dict(reference_hybrid,name='hybrid8_'+suffix,
                             backend_environment=dict(reference_hybrid['backend_environment'],**overrides)))
            with patch.dict(os.environ,{key:'inherited' for key in managed}):
                before = dict(os.environ)
                server = PlacementServer(candidate)
                environment = server.environment_for(settings.model,settings.projector)
                self.assertEqual({key:environment.get(key) for key in managed},
                                 {key:candidate['backend_environment'].get(key) for key in managed})
                self.assertEqual(dict(os.environ),before)
            self.assertEqual(server.settings_for(settings.executable,settings.model,settings.projector,1024,1024),
                             PlacementServer(reference_hybrid).settings_for(settings.executable,settings.model,settings.projector,1024,1024))
        self.assertEqual(candidates['hybrid8_cache2048'],dict(candidates['graphics_hybrid8'],
                         name='hybrid8_cache2048',cache_ram=2048))
        for suffix,cache_ram in (('baseline',0),('ram2048',2048),('baseline_end',0)):
            candidate = candidates['hybrid8_context_'+suffix]
            self.assertEqual(candidate,dict(candidates['graphics_hybrid8'],name='hybrid8_context_'+suffix,
                             context_switch_study=True,cache_ram=cache_ram,context_switch_rows=120))
            options = PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            for flag,value in (('--cache-ram',str(cache_ram)),('--tensor-split','56,10'),('-ub','128')):
                self.assertEqual(options.count(flag),1)
                self.assertEqual(options[options.index(flag)+1],value)
            self.assertTrue(candidate['verify_weight_buffers'])
        for name,ubatch in (('hybrid8_baseline',128),('hybrid8_ub64',64),
                            ('hybrid8_ub256',256),('hybrid8_baseline_end',128)):
            candidate = candidates[name]
            self.assertEqual(candidate,dict(candidates['graphics_hybrid8'],name=name,ubatch=ubatch))
            options = PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            self.assertEqual(options[options.index('-ub')+1],str(ubatch))
            self.assertEqual(options[options.index('--tensor-split')+1],'56,10')
        for name,reference_name in (('graphics_long_baseline','graphics_baseline'),
                                   ('graphics_long_hybrid8','graphics_hybrid8'),
                                   ('graphics_long_baseline_end','graphics_baseline_end')):
            candidate = candidates[name]
            reference_long = candidates[reference_name]
            self.assertEqual(candidate,dict(reference_long,name=name,mtp_study=False,fresh_text_prefill=True))
            self.assertEqual(
                PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024),
                PlacementServer(reference_long).settings_for(settings.executable,settings.model,settings.projector,1024,1024))
            with patch.dict(os.environ,{key:'inherited' for key in managed}):
                before = dict(os.environ)
                environment = PlacementServer(candidate).environment_for(settings.model,settings.projector)
                self.assertEqual({key:environment.get(key) for key in managed},
                                 {key:candidate['backend_environment'].get(key) for key in managed})
                self.assertEqual(dict(os.environ),before)
        for count in (4,8):
            reference_hybrid = candidates[f'graphics_hybrid{count}']
            for suffix,overrides in (('cuda_log',{'CUDA_LOG_FILE':'stderr'}),
                                     ('cuda_blocking',{'CUDA_LOG_FILE':'stderr','CUDA_LAUNCH_BLOCKING':'1'}),
                                     ('cuda_no_graphs',{'CUDA_LOG_FILE':'stderr','GGML_CUDA_DISABLE_GRAPHS':'1'})):
                candidate = candidates[f'graphics_hybrid{count}_{suffix}']
                self.assertEqual(candidate,dict(reference_hybrid,name=f'graphics_hybrid{count}_{suffix}',
                    diagnostic_only=True,backend_environment=dict(reference_hybrid['backend_environment'],**overrides)))
                with patch.dict(os.environ,{key:'inherited' for key in managed}):
                    before = dict(os.environ)
                    server = PlacementServer(candidate)
                    environment = server.environment_for(settings.model,settings.projector)
                    for key in managed:
                        self.assertEqual(environment.get(key),candidate['backend_environment'].get(key))
                    self.assertEqual(dict(os.environ),before)
        with patch.dict(os.environ,{key:'inherited' for key in managed}):
            before = dict(os.environ)
            for name,candidate in candidates.items():
                if not name.startswith(('q8_off_','q8_long_')):
                    continue
                if name!='q8_off_cuda_output':
                    ignored = ('name','backend_environment','checkpoints','batch_threads','mtp_study','fresh_text_prefill')
                    self.assertEqual({key:value for key,value in candidate.items() if key not in ignored},
                                     {key:value for key,value in reference.items() if key not in ignored})
                    self.assertEqual(candidate['batch_threads'],12 if name=='q8_off_graphics_tb12' else 6)
                    self.assertEqual(candidate['mtp_study'],not name.startswith('q8_long_'))
                    self.assertEqual(candidate.get('fresh_text_prefill',False),name.startswith('q8_long_'))
                else:
                    self.assertNotIn('output.weight',candidate['secondary_tensors'])
                    self.assertTrue(candidate['override'].endswith(r'^output\.weight$=CUDA0'))
                    self.assertGreaterEqual(reference['primary_weight_mib']-candidate['primary_weight_mib'],128)
                    self.assertAlmostEqual(sum(candidate[key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')),
                                           sum(reference[key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')))
                server = PlacementServer(candidate)
                options = server.settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
                self.assertEqual(options[options.index('-tb')+1],str(candidate['batch_threads']))
                if 'checkpoints' in candidate:
                    self.assertEqual(options[options.index('--ctx-checkpoints')+1],str(candidate['checkpoints']))
                environment = server.environment_for(settings.model,settings.projector)
                for key in managed:
                    self.assertEqual(environment.get(key),candidate['backend_environment'].get(key))
                    self.assertEqual(server.launch_queue_environment[key],candidate['backend_environment'].get(key))
                self.assertEqual(dict(os.environ),before)
            with self.assertRaisesRegex(ValueError,'Unsupported backend'):
                PlacementServer(dict(reference,backend_environment={'UNVERIFIED_OPTION':'1'})).environment_for(settings.model,settings.projector)

    def test_mtp_off_image_checks_pass_images_and_grade_clicks_without_execution(self):
        from unittest.mock import Mock,patch
        from PIL import Image
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import run_candidate,mtp_workload
        image = Image.new('RGB',(1280,800),'white')
        cases = [dict(name=f'text_{index}',prompt='Calculate.',expected=dict(tool='finish',text='42')) for index in range(8)]
        cases += [dict(name=f'image_{index}',prompt='Read the image.',image=image,
                       expected=dict(tool='finish',text='42')) for index in range(2)]
        cases += [dict(name=f'click_{index}',prompt='Click APPLY.',image=image,
                       expected=dict(tool='desktop_click',arguments={'button':'left','clicks':1},bounds=[100,100,300,300])) for index in range(2)]
        actions = [dict(tool='finish',message=mtp_workload(index)[1]) for index in range(2)]
        actions += [dict(tool='finish',message='42') for index in range(10)]
        actions += [dict(tool='desktop_click',arguments={'x':200,'y':200,'button':'left','clicks':1}),
                    dict(tool='desktop_click',arguments={'x':900,'y':200,'button':'left','clicks':1})]
        metrics = dict(seconds=0.01,usage={'completion_tokens':1},
                       timings={'prompt_ms':1,'predicted_ms':1,'predicted_per_second':1000})
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/'images.log').touch()
            server,model = Mock(),Mock()
            server.process.pid = 42
            server.process.args = ['synthetic']
            repeated_actions = actions+actions[2:]
            model.generate.side_effect = [(action,metrics) for action in repeated_actions]
            with patch('desktop_agent.benchmark_placement.PlacementServer',return_value=server), \
                 patch('desktop_agent.benchmark_placement.Model',return_value=model), \
                 patch('desktop_agent.benchmark_placement.quality_cases',return_value=cases), \
                 patch('desktop_agent.benchmark_placement.pack_messages',return_value=([],0,0)) as pack:
                result = run_candidate(dict(name='images',mtp_study=True,mtp_checks=True,mtp_image_checks=True,
                                            mtp_check_rounds=2,spec_type='none',request_limit=90,slow_limit=75),Settings(),directory,1)
            self.assertNotIn('error',result)
            self.assertEqual(len(result['samples']),26)
            self.assertEqual([sample['kind'] for sample in result['samples']],['warmup','text']+(['quality']*8+['vision_quality']*4)*2)
            self.assertTrue(all(sample['passed'] for index,sample in enumerate(result['samples']) if index not in (13,25)))
            self.assertFalse(result['samples'][-1]['passed'])
            self.assertEqual(result['samples'][-1]['action'],actions[-1])
            for index,call in enumerate(model.generate.call_args_list):
                has_image = index>=2 and (index-2)%12>=8
                self.assertIs(call.args[1],image if has_image else None)
                state = pack.call_args_list[index].kwargs['state']
                self.assertEqual(state['image_attached'],has_image)
                self.assertEqual(state['window']['handle'] if state['window'] else None,73101 if has_image else None)
                if index>=14:
                    self.assertEqual(pack.call_args_list[index].kwargs,pack.call_args_list[index-12].kwargs)
                    self.assertEqual(result['samples'][index]['validation_round'],1)
            progress = json.loads((directory/'images-progress.json').read_text())
            self.assertEqual(len(progress['samples']),26)
            model.close.assert_called_once()
            model.reset_mock()
            model.generate.side_effect = [(action,metrics) for action in actions[:2]+[actions[10]]]
            diagnostic = dict(name='images',mtp_study=True,mtp_checks=True,mtp_image_checks=True,
                              diagnostic_only=True,diagnostic_image_only=True,spec_type='none')
            with patch('desktop_agent.benchmark_placement.PlacementServer',return_value=server), \
                 patch('desktop_agent.benchmark_placement.Model',return_value=model), \
                 patch('desktop_agent.benchmark_placement.quality_cases',return_value=cases), \
                 patch('desktop_agent.benchmark_placement.pack_messages',return_value=([],0,0)):
                result = run_candidate(diagnostic,Settings(),directory,1)
                self.assertNotIn('error',result)
                self.assertEqual([sample['kind'] for sample in result['samples']],['warmup','text','vision_quality'])
                self.assertEqual(result['samples'][-1]['case'],'image_0')
                self.assertTrue(all(sample['passed'] for sample in result['samples']))
                self.assertIs(model.generate.call_args_list[-1].args[1],image)
                model.reset_mock()
                server.reset_mock()
                result = run_candidate(dict(diagnostic,mtp_check_rounds=2),Settings(),directory,1)
                self.assertEqual(result['failed_phase'],'preflight')
                server.start.assert_not_called()
                model.generate.assert_not_called()
                result = run_candidate(dict(diagnostic,spec_type='draft-mtp'),Settings(),directory,1)
                self.assertEqual(result['failed_phase'],'preflight')
                server.start.assert_not_called()
                model.generate.assert_not_called()
                result = run_candidate(dict(diagnostic,backend_environment={'GGML_SCHED_DEBUG':'2'}),Settings(),directory,1)
                self.assertIn('Scheduler node diagnostics were not activated',result['error'])
                model.generate.assert_not_called()

    def test_mtp_preflight_failure_does_not_wait_for_unstarted_sampler(self):
        from unittest.mock import Mock,patch
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import run_candidate
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            server,model,probe,guard = Mock(),Mock(),Mock(),Mock()
            server.process = None
            guard.reason = 'Preflight VRAM estimate exceeds limit'
            guard.path = directory/'guard.json'
            guard.preflight.side_effect = ValueError(guard.reason)
            probe.idle.side_effect = RuntimeError('Sampler never started')
            with patch('desktop_agent.benchmark_placement.PlacementServer',return_value=server), \
                 patch('desktop_agent.benchmark_placement.Model',return_value=model), \
                 patch('desktop_agent.benchmark_resources.GPUWatchdog',return_value=guard):
                result = run_candidate(dict(name='preflight',vram_guard=True),Settings(),directory,1,resource_probe=probe)
            self.assertEqual(result['error'],guard.reason)
            self.assertEqual(result['failed_phase'],'preflight')
            probe.start.assert_not_called()
            probe.idle.assert_not_called()
            probe.close.assert_called_once()
            model.generate.assert_not_called()
            model.close.assert_called_once()

    def test_mtp_research_cleanup_does_not_hide_request_failure(self):
        from unittest.mock import Mock,patch
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import run_candidate
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/'probe.log').touch()
            server,model,probe = Mock(),Mock(),Mock()
            server.process.pid = 42
            server.process.args = ['synthetic']
            server.process.poll.return_value = None
            model.generate.side_effect = RuntimeError('primary request failure')
            def snapshot(label):
                if label=='idle_final':
                    raise PermissionError('process exited during snapshot')
            probe.snapshot.side_effect = snapshot
            with patch('desktop_agent.benchmark_placement.PlacementServer',return_value=server), \
                 patch('desktop_agent.benchmark_placement.Model',return_value=model):
                result = run_candidate(dict(name='probe',mtp_study=True),Settings(),directory,1,resource_probe=probe)
            self.assertEqual(result['error'],'primary request failure')
            self.assertIn('process exited',result['resource_cleanup_error'])
            model.close.assert_called_once()
            probe.close.assert_called_once()

    def test_mtp_research_verifies_loaded_weight_buffers_before_inference(self):
        from unittest.mock import Mock,patch
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import run_candidate
        logical = {'CPU':400,'CUDA0':5960,'CUDA1':40,'Vulkan0':2600}
        for buffers,valid,plan in (({},False,None),({'CPU':400,'CUDA0':6010,'Vulkan0':2590},False,None),
                      ({'CPU':400,'CUDA0':6000,'Vulkan0':2600},True,None),
                      (logical,True,logical),
                      (dict(logical,CUDA0=6000,CUDA1=0),False,logical),
                      ({'CPU':400,'CUDA0':6000,'Vulkan0':2600},False,logical),
                      (dict(logical,CUDA2=1),False,logical)):
            with self.subTest(buffers=buffers),tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                (directory/'placement.log').write_text('\n'.join(
                    f'load_tensors: {device} model buffer size = {size:.2f} MiB'
                    for device,size in buffers.items()),encoding='utf-8')
                server,model = Mock(),Mock()
                server.process.pid = 42
                server.process.args = ['synthetic']
                model.generate.side_effect = RuntimeError('request sentinel')
                candidate = dict(name='placement',mtp_study=True,verify_weight_buffers=True,
                                 cpu_weight_mib=400,primary_weight_mib=6000,secondary_weight_mib=2600)
                if plan is not None:
                    candidate['expected_weight_buffers_mib'] = plan
                with patch('desktop_agent.benchmark_placement.PlacementServer',return_value=server), \
                     patch('desktop_agent.benchmark_placement.Model',return_value=model):
                    result = run_candidate(candidate,Settings(),directory,1)
                if valid:
                    self.assertEqual(result['verified_weight_buffers_mib'],buffers)
                    self.assertEqual(result['error'],'request sentinel')
                    model.generate.assert_called_once()
                else:
                    self.assertIn('Loaded weight placement differs',result['error'])
                    model.generate.assert_not_called()
                model.close.assert_called_once()

    def test_mtp_research_verifies_kv_cache_before_inference(self):
        from unittest.mock import Mock,patch
        from desktop_agent.agent import Settings
        from desktop_agent.benchmark_placement import run_candidate
        candidate = dict(name='kv',mtp_study=True,verify_kv_cache=True,cache_k='q4_0',cache_v='q4_0',
                         draft_cache_k='q4_0',draft_cache_v='q4_0')
        arguments = ['server','-ctk','q4_0','-ctv','q4_0','--spec-draft-type-k','q4_0','--spec-draft-type-v','q4_0']
        active = 'llama_kv_cache: size = 216.00 MiB (12288 cells, 16 layers, 1/1 seqs), K (q4_0): 108.00 MiB, V (q4_0): 108.00 MiB'
        for log,options,valid in ((active,arguments,True),('',arguments,False),
                (active.replace('q4_0','q8_0'),arguments,False),
                (active+'\n'+active.replace('108.00','0.00'),arguments,False),
                (active,arguments+['-ctk','q8_0'],False),
                (active,[value.replace('q4_0','q8_0') for value in arguments],False),
                (active,arguments[:-1],False)):
            with self.subTest(log=log,options=options),tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                (directory/'kv.log').write_text(log,encoding='utf-8')
                server,model = Mock(),Mock()
                server.process.pid = 42
                server.process.args = options
                model.generate.side_effect = RuntimeError('request sentinel')
                with patch('desktop_agent.benchmark_placement.PlacementServer',return_value=server), \
                     patch('desktop_agent.benchmark_placement.Model',return_value=model):
                    result = run_candidate(candidate,Settings(),directory,1)
                if valid:
                    self.assertEqual(result['verified_kv_cache'],dict(cache_k='q4_0',cache_v='q4_0',
                        total_mib=216.0,key_mib=108.0,value_mib=108.0))
                    self.assertEqual(result['error'],'request sentinel')
                    model.generate.assert_called_once()
                else:
                    self.assertIn('KV cache',result['error'])
                    model.generate.assert_not_called()
                model.close.assert_called_once()

    def test_mtp_candidates_budget_head_and_compensate_cuda_with_whole_ffns(self):
        import re
        from desktop_agent.benchmark_placement import mtp_candidates,mtp_workload,PlacementServer
        from desktop_agent.agent import Settings
        weights = {f'blk.{index}.ffn_{kind}.weight':dict(bytes=40*1024**2)
                   for index in range(65) for kind in ('gate','up','down')}
        weights.update({'output.weight':dict(bytes=700*1024**2),'token_embd.weight':dict(bytes=400*1024**2),
                        'blk.64.nextn.eh_proj.weight':dict(bytes=80*1024**2)})
        candidates = {item['name']:item for item in mtp_candidates(weights)}
        baseline = candidates['mtp_off_q8']
        swapped = candidates['hybrid12_ffn_swap51_50']
        self.assertEqual(len(swapped['secondary_tensors']),len(set(swapped['secondary_tensors'])))
        self.assertEqual(len(swapped['ffn_blocks']),len(set(swapped['ffn_blocks'])))
        patterns = [item.split('=')[0] for item in swapped['override'].split(',') if item.endswith('=Vulkan0')]
        assigned = {name for name in weights if name=='output.weight' or
                (name.startswith('blk.') and int(name.split('.')[1]) in swapped['whole_blocks']) or
                any(re.fullmatch(pattern,name) for pattern in patterns)}
        self.assertEqual(assigned,set(swapped['secondary_tensors']))
        self.assertEqual(baseline['ffn_blocks'],list(range(45,64)))
        self.assertEqual(baseline['spec_type'],'none')
        settings = Settings()
        for candidate in candidates.values():
            enabled = candidate['spec_type']=='draft-mtp'
            self.assertEqual(candidate['ubatch'],{'mtp_off_q4_ub64':64,'mtp_off_q4_ub512':512,'graphics_ub64':64,'graphics_ub256':256,
                                                 'hybrid8_ub64':64,'hybrid8_ub256':256,'hybrid12_ub64':64,
                                                 'attention55_ub64':64,'attention55_ub64_blocking_schedule':64,
                                                 'attention55_ub256':256}.get(candidate['name'],128))
            self.assertEqual(candidate['cuda_scale_launch_queues'],'2x' if candidate['name'] in (
                'graphics_queue2x','hybrid8_queue2x','attention55_virtual_queue2x') else '4x')
            self.assertTrue(candidate['vram_guard'])
            self.assertEqual(candidate['request_limit'],90)
            expected = sum(value['bytes'] for value in weights.values())/1024**2
            self.assertAlmostEqual(sum(candidate[key] for key in ('cpu_weight_mib','primary_weight_mib','secondary_weight_mib')),
                                   expected-(0 if enabled else candidate['mtp_weight_mib']))
            options = PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
            self.assertEqual(options[options.index('-ub')+1],str(candidate['ubatch']))
            self.assertEqual(options[options.index('--spec-type')+1],candidate['spec_type'])
            self.assertEqual(options[options.index('-ctk')+1],candidate['cache_k'])
            self.assertEqual(options[options.index('-ctv')+1],candidate['cache_v'])
            self.assertEqual(options[options.index('--spec-draft-type-k')+1],candidate['cache_k'])
            self.assertNotIn('--spec-draft-device',options)
            if candidate['mtp_device']=='CUDA0':
                self.assertGreaterEqual(candidate['compensation_mib'],candidate['mtp_weight_mib']+128)
                self.assertLess(candidate['primary_weight_mib'],baseline['primary_weight_mib'])
                self.assertTrue(candidate['override'].endswith(r'^blk\.64\..*$=CUDA0'))
        off_q4 = candidates['mtp_off_q4']
        self.assertEqual(off_q4['spec_type'],'none')
        self.assertEqual((off_q4['cache_k'],off_q4['cache_v']),('q4_0','q4_0'))
        self.assertEqual(candidates['mtp_off_q4_ub64'],dict(off_q4,name='mtp_off_q4_ub64',ubatch=64))
        self.assertEqual(candidates['mtp_off_q4_ub512'],dict(off_q4,name='mtp_off_q4_ub512',ubatch=512))
        self.assertEqual(candidates['mtp_off_q4_end'],dict(off_q4,name='mtp_off_q4_end'))
        headroom_base = candidates['mtp_cuda_shift_q4_n3']
        for extra in (1,2):
            candidate = candidates[f'mtp_cuda_q4_n3_extra{extra}']
            self.assertEqual(len(candidate['ffn_blocks']),len(headroom_base['ffn_blocks'])+extra)
            self.assertAlmostEqual(candidate['extra_headroom_mib'],120*extra)
            self.assertAlmostEqual(headroom_base['primary_weight_mib']-candidate['primary_weight_mib'],120*extra)
            self.assertAlmostEqual(candidate['secondary_weight_mib']-headroom_base['secondary_weight_mib'],120*extra)
            for key in ('spec_type','draft_n_max','cache_k','cache_v','draft_cache_k','draft_cache_v',
                        'mtp_device','cpu_tensors','threads','batch_threads','tensor_split','projector_device'):
                self.assertEqual(candidate[key],headroom_base[key])
            for name in weights:
                is_moved = name in candidate['secondary_tensors']
                pattern = candidate['override'].split(',')[0].removesuffix('=Vulkan0')
                self.assertEqual(bool(re.fullmatch(pattern,name)),is_moved)
        for name in ('mtp_cuda_shift_q4_n3','mtp_cuda_q4_n3_extra2'):
            reference = candidates[name]
            for count in (1,2):
                candidate = candidates[name+f'_cpu{count}']
                self.assertEqual(candidate['cpu_ffn_blocks'],reference['ffn_blocks'][:count])
                self.assertEqual(candidate['ffn_blocks'],reference['ffn_blocks'][count:])
                self.assertEqual(candidate['offloaded_ffn_blocks'],reference['ffn_blocks'])
                self.assertEqual(len(set(candidate['cpu_tensors']) & set(candidate['secondary_tensors'])),0)
                self.assertAlmostEqual(candidate['cpu_ffn_mib'],120*count)
                self.assertAlmostEqual(candidate['cpu_weight_mib']-reference['cpu_weight_mib'],120*count)
                self.assertAlmostEqual(reference['secondary_weight_mib']-candidate['secondary_weight_mib'],120*count)
                for key in ('primary_weight_mib','mtp_device','draft_n_max','cache_k','cache_v','threads','batch_threads'):
                    self.assertEqual(candidate[key],reference[key])
                for tensor in weights:
                    selected = [device for pattern,device in (part.split('=') for part in candidate['override'].split(','))
                                if re.fullmatch(pattern,tensor)]
                    expected_device = 'CPU' if tensor in candidate['cpu_tensors'] else 'Vulkan0' if tensor in candidate['secondary_tensors'] else 'CUDA0' if tensor.startswith('blk.64.') else None
                    self.assertEqual(selected,[expected_device] if expected_device else [])
        self.assertEqual(len(mtp_workload(1)[1].splitlines()),20)
        self.assertEqual(mtp_workload(0)[1],'391')
        with self.assertRaisesRegex(ValueError,'embedded MTP'):
            mtp_candidates({'token_embd.weight':dict(bytes=1)})

    def test_input_embedding_study_moves_only_token_table_and_accounts_for_rx_memory(self):
        from desktop_agent.benchmark_placement import input_embedding_candidates,launch_queue_candidates,PlacementServer
        from desktop_agent.agent import Settings
        weights = {f'blk.{index}.ffn_{kind}.weight':dict(bytes=1024)
                   for index in range(64) for kind in ('gate','up','down')}
        weights.update({'output.weight':dict(bytes=1024),'token_embd.weight':dict(bytes=4096)})
        reference = next(item for item in launch_queue_candidates(weights) if item['name']=='queue_4x_ub128')
        candidates = input_embedding_candidates(weights)
        self.assertEqual([item['embedding_device'] for item in candidates],['CPU','Vulkan0','CPU'])
        settings = Settings()
        for candidate in candidates:
            with self.subTest(name=candidate['name']):
                for key in ('primary_weight_mib','ffn_blocks','ubatch','cuda_scale_launch_queues','threads',
                            'batch_threads','tensor_split','projector_device','request_limit','vram_guard'):
                    self.assertEqual(candidate[key],reference[key])
                self.assertTrue(candidate['fresh_text_prefill'])
                self.assertEqual(candidate['override'],reference['override']+r',^token_embd\.weight$='+candidate['embedding_device'])
                self.assertEqual(candidate['cpu_weight_mib']+candidate['secondary_weight_mib'],
                                 reference['cpu_weight_mib']+reference['secondary_weight_mib'])
                options = PlacementServer(candidate).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
                self.assertEqual(options[options.index('--override-tensor')+1],candidate['override'])
        self.assertEqual(candidates[1]['cpu_weight_mib'],0)
        self.assertAlmostEqual(candidates[1]['secondary_weight_mib']-candidates[0]['secondary_weight_mib'],4096/1024**2)

    @unittest.skipUnless(os.name=='nt','Windows launcher')
    def test_batch_launcher_detaches_windowless_python_and_preserves_errors(self):
        import shutil
        import subprocess
        with tempfile.TemporaryDirectory(prefix='Local Desk launcher ') as folder:
            root = Path(folder)
            launcher = root/'desktop_agent'
            launcher.mkdir()
            source = Path(__file__).resolve().parents[1]/'desktop_agent'
            shutil.copy2(source/'start.ps1',launcher/'start.ps1')
            interpreter = root/'python environment'
            interpreter.mkdir()
            python = interpreter/'python.exe'
            python.touch()
            pythonw = interpreter/'pythonw.exe'
            pythonw.touch()
            script = '''
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $RedirectStandardOutput, $RedirectStandardError, [switch]$Wait)
    $global:captured = [pscustomobject]@{
        executable=$FilePath; arguments=$ArgumentList; directory=$WorkingDirectory
        stdout=$RedirectStandardOutput; stderr=$RedirectStandardError; wait=[bool]$Wait
    }
}
$ErrorActionPreference = 'Stop'
$before = (Get-Location).Path
& $env:LOCAL_DESK_TEST_LAUNCHER -Detached -PythonPath $env:LOCAL_DESK_TEST_PYTHON
if ((Get-Location).Path -ne $before) { throw 'Launcher changed parent directory' }
$global:captured | ConvertTo-Json -Compress
'''
            environment = dict(os.environ,LOCAL_DESK_TEST_LAUNCHER=str(launcher/'start.ps1'),
                               LOCAL_DESK_TEST_PYTHON=str(python))
            command = ['powershell.exe','-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-Command',script]
            result = subprocess.run(command,env=environment,capture_output=True,text=True,timeout=20,check=True)
            captured = json.loads(result.stdout)
            self.assertEqual(Path(captured['executable']),pythonw)
            self.assertEqual(captured['arguments'],['-m','desktop_agent.app'])
            self.assertEqual(Path(captured['directory']),root)
            self.assertFalse(captured['wait'])
            for key in ('stdout','stderr'):
                self.assertEqual(Path(captured[key]).parent,launcher/'data'/'launcher')
            self.assertNotEqual(captured['stdout'],captured['stderr'])
            pythonw.unlink()
            failure = subprocess.run(command,env=environment,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(failure.returncode,0)
            self.assertIn('pythonw.exe not found',failure.stderr)
            batch = (source/'start.bat').read_text(encoding='utf-8')
            self.assertIn('"%~dp0start.ps1" -Detached %*',batch)
            self.assertIn('if not "%EXIT_CODE%"=="0" (',batch)

    def test_launch_queue_environment_is_child_only_and_baseline_unsets_inherited_value(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.agent import DesktopServer,Settings
        from desktop_agent.benchmark_placement import PlacementServer,launch_queue_candidates
        weights = {f'blk.{index}.ffn_{kind}.weight':dict(bytes=1024)
                   for index in range(64) for kind in ('gate','up','down')}
        weights.update({'output.weight':dict(bytes=1024),'token_embd.weight':dict(bytes=1024)})
        candidates = launch_queue_candidates(weights)
        self.assertEqual([item['cuda_scale_launch_queues'] for item in candidates],[None,'2x','4x','0.5x','0.25x',None,'4x','4x','4x'])
        settings = Settings()
        with patch.dict(os.environ,{'CUDA_SCALE_LAUNCH_QUEUES':'4x'}), \
             patch('desktop_agent.agent.server_environment',return_value={'GGML_BACKEND_PATH':'synthetic-backend','CUDA_SCALE_LAUNCH_QUEUES':'4x','GGML_VK_ALLOW_GRAPHICS_QUEUE':'1'}) as environment, \
             patch('game_agent.runtime.LocalServer.start',return_value='synthetic-endpoint') as start:
            for candidate in candidates:
                server = PlacementServer(candidate)
                self.assertTrue(candidate['vram_guard'])
                self.assertEqual(candidate['request_limit'],90)
                expected_ubatch = {'queue_4x_ub128':128,'queue_4x_ub512':512}.get(candidate['name'],256)
                self.assertEqual(candidate['ubatch'],expected_ubatch)
                for key in ('override','threads','batch_threads','batch','load_mode','tensor_split','ffn_blocks'):
                    self.assertEqual(candidate[key],candidates[0][key])
                options = server.settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
                self.assertEqual(options[options.index('-ub')+1],str(expected_ubatch))
                server.start(settings.executable,settings.model,settings.projector,Path('unused.log'),threading.Event())
                child = start.call_args.kwargs['environment']
                self.assertEqual(child.get('CUDA_SCALE_LAUNCH_QUEUES'),candidate['cuda_scale_launch_queues'])
                self.assertEqual(child['GGML_BACKEND_PATH'],'synthetic-backend')
                self.assertNotIn('GGML_VK_ALLOW_GRAPHICS_QUEUE',child)
                self.assertEqual(server.launch_queue_environment,{'CUDA_SCALE_LAUNCH_QUEUES':candidate['cuda_scale_launch_queues']})
                self.assertEqual(os.environ['CUDA_SCALE_LAUNCH_QUEUES'],'4x')
                self.assertEqual(environment.return_value['CUDA_SCALE_LAUNCH_QUEUES'],'4x')
            legacy = dict(candidates[0])
            legacy.pop('cuda_scale_launch_queues')
            self.assertNotIn('CUDA_SCALE_LAUNCH_QUEUES',PlacementServer(legacy).environment_for(settings.model,settings.projector))
            self.assertEqual(DesktopServer().environment_for(settings.model,settings.projector)['CUDA_SCALE_LAUNCH_QUEUES'],'4x')
            with self.assertRaisesRegex(ValueError,'Unsupported'):
                PlacementServer(dict(candidates[0],cuda_scale_launch_queues='2')).environment_for(settings.model,settings.projector)

    @unittest.skipUnless(os.name=='nt','Windows research module')
    def test_resource_intervals_clip_requests_and_do_not_fill_missing_samples(self):
        from desktop_agent.benchmark_resources import sampled_intervals,cpu_intervals
        rows=[dict(at_ms=index*500,monotonic=index/2,pid=42,value=value,
                   cpu=dict(user=index/2,system=0))
              for index,value in enumerate((10,20,None,40,50))]
        self.assertEqual(sampled_intervals(rows,0.25,1.75,lambda row:row['value'],rate=True),
                         [(0.25,0.5,20),(1.0,1.5,40),(1.5,1.75,50)])
        self.assertEqual(sampled_intervals(rows,0.25,1.75,lambda row:row['value']),
                         [(0.25,0.5,10),(0.5,1.0,20),(1.5,1.75,40)])
        cpu=cpu_intervals(rows,0.25,1.75,12)
        self.assertEqual(len(cpu),1)
        self.assertEqual(cpu[0][:2],(0.5,1.5))
        self.assertAlmostEqual(cpu[0][2],100/12)

    @unittest.skipUnless(os.name=='nt','Windows research module')
    def test_gpu_usage_groups_physical_engines_and_weights_intervals(self):
        from desktop_agent.benchmark_resources import gpu_engine_usage,usage_statistics
        adapter='luid_0x00000000_0x00001234_phys_0'
        counters={f'pid_42_{adapter}_eng_0_engtype_Compute':60,
                  f'pid_99_{adapter}_eng_0_engtype_Compute':20,
                  f'pid_42_{adapter}_eng_1_engtype_Copy':30}
        values=gpu_engine_usage(counters,42)[adapter]
        self.assertEqual(values['total_busiest'],80)
        self.assertEqual(values['server_busiest'],60)
        self.assertEqual(len(values['server_engines']),2)
        result=usage_statistics([(0,1,10),(1,3,90),(3,4,85),(4,5,5),(5,6,95)])
        self.assertAlmostEqual(result['mean'],375/6)
        self.assertEqual(result['p95'],95)
        self.assertEqual(result['high_episodes'],2)
        self.assertEqual(result['high_seconds'],4)
        self.assertIsNone(usage_statistics([]))

    @unittest.skipUnless(os.name=='nt','Windows GPU counters')
    def test_gpu_guard_rejects_margin_shared_memory_and_missing_adapter(self):
        from desktop_agent.benchmark_resources import gpu_limit_reason
        limits={'adapter':8192}
        sample=dict(adapter_dedicated={'adapter':7000*1024**2},process_shared={'pid_42_adapter':12*1024**2})
        self.assertEqual(gpu_limit_reason(sample,42,limits),'')
        self.assertIn('margin',gpu_limit_reason(dict(sample,adapter_dedicated={'adapter':8100*1024**2}),42,limits))
        self.assertIn('shared',gpu_limit_reason(dict(sample,process_shared={'pid_42_adapter':400*1024**2}),42,limits))
        self.assertEqual(gpu_limit_reason(dict(sample,process_shared={'pid_43_adapter':400*1024**2}),42,limits),'')
        self.assertIn('disappeared',gpu_limit_reason(dict(sample,adapter_dedicated={}),42,limits))

    def test_latency_stages_do_not_double_count_vision_and_variants_keep_ffn_count(self):
        from desktop_agent.benchmark_placement import latency_breakdown,latency_candidates,PlacementServer
        from desktop_agent.agent import Settings
        log = '0.01.000.000 I encoding mtmd batch\n0.01.800.000 I decoding image batch 1/3'
        result = latency_breakdown(log,dict(prompt_ms=5000,predicted_ms=2000),7.2,7.1)
        self.assertAlmostEqual(result['vision_encode_seconds'],0.8)
        self.assertAlmostEqual(result['prefill_excluding_vision_seconds'],4.2)
        self.assertAlmostEqual(sum(result[key] for key in ('vision_encode_seconds','prefill_excluding_vision_seconds',
            'reasoning_and_answer_seconds','client_preparation_seconds','transport_and_other_seconds')),7.2)
        weights = {f'blk.{index}.ffn_{kind}.weight':dict(bytes=1024) for index in range(64) for kind in ('gate','up','down')}
        weights.update({'output.weight':dict(bytes=1024),'token_embd.weight':dict(bytes=1024)})
        candidates = {item['name']:item for item in latency_candidates(weights)}
        for name in ('current','early9','middle9','spread9','full_attention9','linear_attention9'):
            self.assertEqual(len(candidates[name]['ffn_blocks']),9)
        self.assertEqual(len(candidates['return2_ub128']['ffn_blocks']),7)
        self.assertEqual(candidates['blocks9']['tensor_split'],'55,10')
        self.assertEqual(candidates['blocks9']['override'],r'^token_embd\.weight$=CPU')
        self.assertEqual(candidates['blocks9']['whole_blocks'],list(range(55,64)))
        self.assertLessEqual(candidates['blocks9']['request_limit'],90)
        self.assertTrue(all(item['vram_guard'] and item['request_limit']==90 for item in candidates.values()))
        self.assertTrue(all(item['primary_weight_mib']+item['secondary_weight_mib']==193/1024 for item in candidates.values()))
        settings = Settings()
        options = PlacementServer(candidates['host']).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        self.assertNotIn('--no-host',options)

    def test_surrounding_capture_clips_screen_and_keeps_input_inside_target(self):
        import threading
        from unittest.mock import Mock,patch
        from PIL import Image
        from game_agent.core import Region
        from desktop_agent.tools import Tools
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            tools = Tools(folder,threading.Event(),Mock(),Mock())
            tools.window = dict(handle=11,pid=22)
            tools.capture_margin = 128
            target = Region(-1800,50,400,600)
            tools.capture_region = Mock(return_value=target)
            with patch('desktop_agent.tools.windows.virtual_screen',return_value=(-1920,0,3840,1080)), patch('desktop_agent.tools.windows.capture',return_value=Image.new('RGB',(648,778))) as capture:
                result = tools.desktop('desktop_capture',{})
            self.assertEqual(capture.call_args.args[0].bbox,(-1920,0,-1272,778))
            reference = tools.coordinate_reference(result.image,None)
            self.assertEqual(reference['source'],'visible_desktop')
            inside = dict(coordinate_space='image_pixels',x=120,y=50)
            self.assertEqual(tools.pointer_region(target,inside).point(120,50),(-1800,50))
            with self.assertRaisesRegex(ValueError,'outside the selected'):
                tools.pointer_region(target,dict(inside,x=0,y=0))
            path = Path(folder)/'settings.json'
            Settings(capture_margin=256).save(path)
            self.assertEqual(Settings.load(path).capture_margin,256)
            self.assertEqual(Settings().capture_margin,0)
            Settings(capture_margin=-1).save(path)
            with self.assertRaises(ValueError):
                Settings.load(path)

    def test_iq2s_rx580_profile_moves_exactly_nine_ffn_blocks_and_output(self):
        import re
        from desktop_agent.agent import Settings, DesktopServer
        from desktop_agent.models import IQ2_S, server_environment
        from game_agent.models import model_server_options
        settings = Settings()
        server = DesktopServer()
        server.context_tokens = 12288
        options = server.settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        for flag,value in (('--device','CUDA0,Vulkan0'),('--mmproj-device','CUDA0'),('--split-mode','layer'),
                           ('--tensor-split','1,0'),('--load-mode','none'),('-t','12'),('-tb','12'),('-ub','128'),
                           ('-ctk','q8_0'),('-ctv','q8_0'),('-c','12288')):
            self.assertEqual(options[options.index(flag)+1],value)
        override = options[options.index('--override-tensor')+1]
        self.assertTrue(override.endswith('=Vulkan0'))
        pattern = override.removesuffix('=Vulkan0')
        for kind in ('gate','up','down'):
            self.assertEqual([block for block in range(65) if re.fullmatch(pattern,f'blk.{block}.ffn_{kind}.weight')],list(range(55,64)))
        self.assertIsNotNone(re.fullmatch(pattern,'output.weight'))
        for name in ('token_embd.weight','blk.55.attn_q.weight','blk.55.ssm_out.weight'):
            self.assertIsNone(re.fullmatch(pattern,name))
        self.assertNotIn('--ctx-checkpoints',options)
        self.assertEqual(settings.local_label,IQ2_S['label'])
        self.assertIsNone(server_environment(settings.model,'mmproj-F16.gguf'))
        shared = model_server_options(settings.model,settings.projector,1024)
        self.assertNotIn('--device',shared)
        self.assertTrue(shared[shared.index('--override-tensor')+1].endswith('=CPU'))

    def test_pixel_desktop_delivery_uses_scaled_image_and_refuses_stale_frame(self):
        import threading
        from unittest.mock import Mock,patch
        from PIL import Image
        from game_agent.core import Region
        from desktop_agent.tools import Tools
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.protocol import normalize_call,require_pixel_coordinates
        with tempfile.TemporaryDirectory() as folder:
            tools = Tools(folder,threading.Event(),lambda *args:True,lambda *args:None)
            tools.window = dict(handle=17,pid=23)
            region = Region(-1200,100,1920,1080)
            image = Image.new('RGB',(1920,1080),'white')
            image.info['desktop_frame'] = dict(source='selected_window',handle=17,pid=23,bounds=region.bbox)
            reference = tools.coordinate_reference(image,1280)
            self.assertEqual((reference['width'],reference['height']),(1280,720))
            self.assertEqual(image.size,(1920,1080))
            action = normalize_call(dict(tool='desktop_click',arguments=dict(x=1279,y=719,coordinate_space='image_pixels')))
            require_pixel_coordinates(action)
            tools.focus_window = Mock(return_value=(region,Mock(title='Synthetic')))
            with patch('desktop_agent.tools.windows.DesktopInput') as backend, patch('desktop_agent.tools.windows.virtual_screen',return_value=(-1920,0,3840,2160)), patch('desktop_agent.tools.windows.cursor_position',return_value=(719,1179)):
                result = tools.desktop('desktop_click',action['arguments'])
                self.assertEqual(json.loads(result.text)['requested_desktop'],[719,1179])
                self.assertEqual(json.loads(result.text)['coordinate_space'],'image_pixels')
                backend.return_value.down.assert_called_once_with('click','left')
                backend.reset_mock()
                tools.focus_window.return_value = (Region(-1200,100,1919,1080),Mock(title='Changed'))
                with self.assertRaisesRegex(ValueError,'moved or resized'):
                    tools.desktop('desktop_click',action['arguments'])
                backend.return_value.down.assert_not_called()
            original = tools.coordinate_reference(image,None)
            self.assertEqual((original['width'],original['height']),(1920,1080))
            tools.coordinate_reference(Image.new('RGB',(100,100)),None)
            with self.assertRaisesRegex(ValueError,'not a desktop'):
                tools.pointer_region(region,action['arguments'])
            runner = ToolRunner(folder,threading.Event(),lambda *args:True,lambda *args:None)
            try:
                runner.window = dict(handle=17,pid=23)
                runner.coordinate_frame = dict(source='selected_window',bounds=region.bbox,image_size=[1280,720],handle=17,pid=23)
                worker = runner.snapshot(threading.Event())
                self.assertEqual(worker.coordinate_frame,runner.coordinate_frame)
                self.assertIsNot(worker.coordinate_frame,runner.coordinate_frame)
            finally:
                runner.close()
            with self.assertRaisesRegex(ValueError,'image_pixels'):
                require_pixel_coordinates(normalize_call(dict(tool='desktop_click',arguments=dict(x=500,y=500))))

    def test_pixel_coordinate_mapping_and_legacy_separation(self):
        from game_agent.core import Region
        from desktop_agent.coordinates import input_region
        from desktop_agent.protocol import normalize_call, compact_parameters
        window = dict(handle=10,pid=20)
        region = Region(-500,100,1920,1080)
        frame = dict(source='selected_window',handle=10,pid=20,bounds=region.bbox,image_size=[1280,720])
        arguments = dict(x=1279,y=719,coordinate_space='image_pixels')
        pixels = input_region(region,arguments,frame,window)
        self.assertEqual(pixels.point(1279,719),(1419,1179))
        self.assertEqual(pixels.point(0,0),(-500,100))
        with self.assertRaises(ValueError):
            pixels.point(1280,0)
        with self.assertRaisesRegex(ValueError,'moved or resized'):
            input_region(Region(-499,100,1920,1080),arguments,frame,window)
        with self.assertRaisesRegex(ValueError,'different'):
            input_region(region,arguments,frame,dict(handle=11,pid=20))
        self.assertIs(input_region(region,dict(x=500,y=500),frame,window),region)
        desktop = dict(source='visible_desktop',handle=10,pid=20,bounds=[-1920,0,1920,1080],
                       target_bounds=region.bbox,image_size=[3840,1080])
        desktop_pixels = input_region(region,dict(coordinate_space='image_pixels',x=1420,y=100),desktop,window)
        self.assertEqual(desktop_pixels.point(1420,100),(-500,100))
        with self.assertRaisesRegex(ValueError,'outside the selected'):
            desktop_pixels.point(0,0)
        with self.assertRaisesRegex(ValueError,'outside the reference'):
            input_region(region,dict(coordinate_space='image_pixels',x=10,y=10,end_x=1280,end_y=10),frame,window)
        action = normalize_call(dict(tool='desktop_click',arguments=arguments))
        self.assertEqual(action['arguments']['x'],1279)
        self.assertIn('coordinate_space',compact_parameters('desktop_click')['required'])
        with self.assertRaises(ValueError):
            normalize_call(dict(tool='desktop_click',arguments=dict(x=1279,y=719)))

    def test_image_resize_setting_roundtrips_and_rejects_invalid_sizes(self):
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            self.assertEqual(Settings().image_max_edge,1280)
            for size in (0,1280):
                settings = Settings(image_max_edge=size)
                settings.save(path)
                self.assertEqual(Settings.load(path).image_max_edge,size)
            for size in (-1,True,99999):
                path.write_text(json.dumps(dict(image_max_edge=size)),encoding='utf-8')
                with self.assertRaises(ValueError):
                    Settings.load(path)

    def test_q2xl_profile_is_optional_and_preserves_iq2s_default(self):
        import re
        from desktop_agent.agent import DesktopServer, Settings
        from desktop_agent.models import MODEL_PRESETS, Q2_XL
        from game_agent.models import MODEL_PRESETS as shared_presets
        self.assertNotIn('qwen38_27b_q2_k_xl',shared_presets)
        self.assertIn('qwen38_27b_q2_k_xl',MODEL_PRESETS)
        original = Settings(context_tokens=12288,reasoning_enabled=True,reasoning_effort='low')
        self.assertEqual(Path(original.model).name,'Qwen3.8-27B-UD-IQ2_S.gguf')
        with tempfile.TemporaryDirectory() as folder:
            for key in ('model','projector'):
                (Path(folder)/Q2_XL[key]).touch()
            selected = original.with_local_preset('qwen38_27b_q2_k_xl',folder)
            self.assertEqual(selected.context_tokens,8192)
            self.assertEqual(selected.local_image_tokens,(1024,1024))
            self.assertEqual(selected.reasoning_effort,'low')
            self.assertEqual(selected.local_label,Q2_XL['label'])
            self.assertEqual(original.context_tokens,12288)
            path = Path(folder)/'settings.json'
            selected.save(path)
            self.assertEqual(Settings.load(path),selected)
            options = DesktopServer().settings_for(selected.executable,selected.model,selected.projector,1024,1024)[-1]
            for flag,value in (('--device','CUDA0,Vulkan0,CUDA0,Vulkan0'),('--mmproj-device','CUDA0'),
                               ('--load-mode','none'),('-ngl','all'),('-c','8192'),('-ub','128'),
                               ('-t','12'),('-tb','6'),('-b','512'),('--tensor-split','52,3,1,10'),
                               ('--split-mode','layer'),('-ctk','q8_0'),('-ctv','q8_0'),('--spec-type','none')):
                self.assertEqual(options.count(flag),1)
                self.assertEqual(options[options.index(flag)+1],value)
            embedding,ffn,attention_ffn = options[options.index('--override-tensor')+1].split(',')
            self.assertEqual(embedding,r'^token_embd\.weight$=CPU')
            pattern = ffn.removesuffix('=Vulkan0')
            for kind in ('gate','up','down'):
                self.assertEqual([block for block in range(65) if re.fullmatch(pattern,f'blk.{block}.ffn_{kind}.weight')],[51])
            self.assertIsNone(re.fullmatch(pattern,'output.weight'))
            self.assertIsNone(re.fullmatch(pattern,'blk.49.attn_q.weight'))
            pattern = attention_ffn.removesuffix('=Vulkan0')
            for kind in ('ffn_gate','ffn_up','ffn_down','post_attention_norm'):
                self.assertEqual([block for block in range(65) if re.fullmatch(pattern,f'blk.{block}.{kind}.weight')],[55])
            self.assertIsNone(re.fullmatch(pattern,'blk.55.attn_q.weight'))
            self.assertNotIn('--ctx-checkpoints',options)
            with self.assertRaisesRegex(ValueError,'matching projector'):
                DesktopServer().settings_for(selected.executable,selected.model,'mmproj-F16.gguf',1024,1024)

    def test_q2xl_hybrid12_reaches_process_and_preserves_historical_research(self):
        import threading
        from unittest.mock import Mock,patch
        from desktop_agent.agent import DesktopServer, Settings
        from desktop_agent.benchmark_placement import PlacementServer,mtp_candidates
        from desktop_agent.models import Q2_XL,VULKAN_BACKEND,Q2_TPB16_BACKEND
        weights = {f'blk.{index}.ffn_{kind}.weight':dict(bytes=40*1024**2)
                   for index in range(65) for kind in ('gate','up','down')}
        weights.update({'output.weight':dict(bytes=700*1024**2),'token_embd.weight':dict(bytes=400*1024**2),
                        'blk.64.nextn.eh_proj.weight':dict(bytes=80*1024**2)})
        baseline = next(candidate for candidate in mtp_candidates(weights) if candidate['name']=='graphics_baseline')
        settings = Settings()
        with tempfile.TemporaryDirectory() as folder:
            for server,split in ((DesktopServer(),'52,3,1,10'),(PlacementServer(baseline),'1,0')):
                server.context_tokens=12288
                server.use_front_residency=False
                process = Mock()
                process.poll.return_value = None
                try:
                    with patch('game_agent.runtime.Path.is_file',return_value=True), \
                         patch('game_agent.runtime.subprocess.Popen',return_value=process) as popen, \
                         patch('game_agent.runtime.session') as client:
                        client.return_value.__enter__.return_value.get.return_value.status_code = 200
                        server.start(settings.executable,Q2_XL['model'],Q2_XL['projector'],
                                     Path(folder)/f'{split}.log',threading.Event(),1024,1024)
                        arguments = popen.call_args.args[0]
                        for flag,value in (('--tensor-split',split),('--mmproj-device','CUDA0'),
                                           ('--device','CUDA0,Vulkan0' if split=='1,0' else 'CUDA0,Vulkan0,CUDA0,Vulkan0'),
                                           ('--cache-ram','0'),('-ctk','q8_0'),('-ctv','q8_0')):
                            self.assertEqual(arguments.count(flag),1)
                            self.assertEqual(arguments[arguments.index(flag)+1],value)
                        self.assertEqual(arguments.count('--override-tensor'),1)
                        override = arguments[arguments.index('--override-tensor')+1]
                        environment = popen.call_args.kwargs['env']
                        if split=='1,0':
                            self.assertEqual(override,baseline['override'])
                            self.assertNotIn('GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM',environment)
                            self.assertEqual(environment['GGML_BACKEND_PATH'],str(VULKAN_BACKEND))
                            self.assertNotIn('GGML_VK_GCN_MEDIUM_TILE',environment)
                            self.assertNotIn('GGML_VK_GCN_FA_BR8',environment)
                            self.assertNotIn('GGML_VK_GCN_IQ3_TPB16',environment)
                            self.assertNotIn('GGML_VK_VISIBLE_DEVICES',environment)
                        else:
                            self.assertIn(r'^token_embd\.weight$=CPU',override)
                            self.assertIn(r'^blk\.51\.ffn_',override)
                            self.assertIn(r'^blk\.55\.(ffn_(gate|up|down)|post_attention_norm)\.weight$=Vulkan0',override)
                            self.assertEqual(environment['GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM'],'1')
                            self.assertEqual(environment['GGML_BACKEND_PATH'],str(Q2_TPB16_BACKEND))
                            self.assertEqual(environment['GGML_VK_GCN_MEDIUM_TILE'],'1')
                            self.assertEqual(environment['GGML_VK_GCN_FA_BR8'],'1')
                            self.assertEqual(environment['GGML_VK_GCN_IQ3_TPB16'],'1')
                            self.assertEqual(environment['GGML_VK_VISIBLE_DEVICES'],'0')
                            self.assertEqual(arguments.count('--spec-type'),1)
                            self.assertEqual(arguments[arguments.index('--spec-type')+1],'none')
                finally:
                    server.close()

    def test_q2xl_backend_environment_is_child_only_and_missing_backend_fails(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.agent import DesktopServer, Settings
        from desktop_agent.models import Q2_XL
        from game_agent.runtime import LocalServer
        with tempfile.TemporaryDirectory() as folder:
            backend,medium = (Path(folder)/name for name in ('ggml-vulkan.dll','ggml-vulkan-medium.dll'))
            for library in (backend,medium):
                library.touch()
            before = dict(os.environ)
            inherited_values = dict(CUDA_SCALE_LAUNCH_QUEUES='0.5x',GGML_VK_ALLOW_GRAPHICS_QUEUE='inherited',
                GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM='inherited',GGML_VK_GCN_MEDIUM_TILE='row4',
                GGML_VK_GCN_LARGE_TILE='1',GGML_VK_GCN_PROBE='1',LLAMA_MTP_IMAGE_PACKED='1',
                GGML_VK_GCN_FA_Q8_DIRECT='1',GGML_VK_GCN_FA_MASK_OPT='1',GGML_VK_PERF_LOGGER='1',
                GGML_VK_PERF_LOGGER_DETAILS='1',GGML_SCHED_COPY_TRACE='1',GGML_VK_GCN_FA_BR8='0',
                GGML_VK_GCN_FA_BR4='1',GGML_VK_GCN_IQ3_ROWS4='1',GGML_VK_GCN_DOWN_SMALL='1',
                GGML_VK_GCN_IQ3_DMMV='large',GGML_VK_GCN_IQ3_TPB16='0')
            with patch.multiple('desktop_agent.models',VULKAN_BACKEND=backend,Q2_TPB16_BACKEND=medium), \
                 patch.dict(os.environ,inherited_values), \
                 patch.object(LocalServer,'start',return_value='endpoint') as start:
                inherited = dict(os.environ)
                server = DesktopServer()
                server.context_tokens=12288
                server.use_front_residency=False
                server.start('server',Q2_XL['model'],Q2_XL['projector'],Path(folder)/'log',threading.Event(),1024,1024)
                self.assertEqual(start.call_args.kwargs['environment']['GGML_BACKEND_PATH'],str(medium))
                self.assertEqual(start.call_args.kwargs['environment']['CUDA_SCALE_LAUNCH_QUEUES'],'4x')
                self.assertEqual(start.call_args.kwargs['environment']['GGML_VK_ALLOW_GRAPHICS_QUEUE'],'1')
                self.assertEqual(start.call_args.kwargs['environment']['GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM'],'1')
                self.assertEqual(start.call_args.kwargs['environment']['GGML_VK_GCN_MEDIUM_TILE'],'1')
                self.assertEqual(start.call_args.kwargs['environment']['GGML_VK_GCN_FA_BR8'],'1')
                self.assertEqual(start.call_args.kwargs['environment']['GGML_VK_GCN_IQ3_TPB16'],'1')
                self.assertEqual(start.call_args.kwargs['environment']['GGML_VK_VISIBLE_DEVICES'],'0')
                for key in ('GGML_VK_GCN_LARGE_TILE','GGML_VK_GCN_PROBE','LLAMA_MTP_IMAGE_PACKED',
                            'GGML_VK_GCN_FA_Q8_DIRECT','GGML_VK_GCN_FA_MASK_OPT','GGML_VK_PERF_LOGGER',
                            'GGML_VK_PERF_LOGGER_DETAILS','GGML_SCHED_COPY_TRACE','GGML_VK_GCN_FA_BR4',
                            'GGML_VK_GCN_IQ3_ROWS4','GGML_VK_GCN_DOWN_SMALL','GGML_VK_GCN_IQ3_DMMV'):
                    self.assertNotIn(key,start.call_args.kwargs['environment'])
                self.assertEqual(dict(os.environ),inherited)
                server.start('server',Settings.model,Settings.projector,Path(folder)/'log',threading.Event(),1024,1024)
                self.assertEqual(start.call_args.kwargs['environment']['GGML_BACKEND_PATH'],str(backend))
                self.assertEqual(start.call_args.kwargs['environment']['CUDA_SCALE_LAUNCH_QUEUES'],'4x')
                self.assertEqual(start.call_args.kwargs['environment']['GGML_VK_ALLOW_GRAPHICS_QUEUE'],'inherited')
                self.assertEqual(start.call_args.kwargs['environment']['GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM'],'inherited')
                self.assertEqual(dict(os.environ),inherited)
                server.start('server',Settings.model,'mmproj-F16.gguf',Path(folder)/'log',threading.Event(),1024,1024)
                self.assertIsNone(start.call_args.kwargs['environment'])
                server.start('server',Settings.model.replace('IQ2_S','IQ2_XXS'),Settings.projector,Path(folder)/'log',threading.Event(),1024,1024)
                self.assertIsNone(start.call_args.kwargs['environment'])
                backend.unlink()
                medium.unlink()
                start.reset_mock()
                with self.assertRaisesRegex(FileNotFoundError,'Vulkan backend'):
                    server.start('server',Q2_XL['model'],Q2_XL['projector'],Path(folder)/'log',threading.Event(),1024,1024)
                with self.assertRaisesRegex(FileNotFoundError,'Vulkan backend'):
                    server.start('server',Settings.model,Settings.projector,Path(folder)/'log',threading.Event(),1024,1024)
                start.assert_not_called()
            self.assertEqual(dict(os.environ),before)

    def test_quantization_quality_grading_requires_correct_arguments_and_bounds(self):
        from desktop_agent.benchmark_placement import grade_quality
        self.assertTrue(grade_quality({'tool':'finish','message':' 391 '},{'tool':'finish','text':'391'}))
        self.assertFalse(grade_quality({'tool':'finish','message':'391 extra'},{'tool':'finish','text':'391'}))
        self.assertTrue(grade_quality({'tool':'finish','message':'5827, 6'},dict(tool='finish',text='5827,6',strip_spaces=True)))
        expected = dict(tool='desktop_click',arguments={'button':'left','clicks':1},bounds=[100,200,300,400])
        action = dict(tool='desktop_click',arguments={'x':200,'y':300,'button':'left','clicks':1})
        self.assertTrue(grade_quality(action,expected))
        self.assertFalse(grade_quality(dict(action,arguments=dict(action['arguments'],x=900)),expected))
        self.assertFalse(grade_quality(dict(action,arguments=dict(action['arguments'],button='right')),expected))

    def test_quantization_candidate_budget_excludes_mtp_and_keeps_vision_cuda(self):
        from desktop_agent.benchmark_placement import quantization_candidates
        weights = {f'blk.{index}.ffn_{kind}.weight':dict(bytes=40*1024**2)
                   for index in range(65) for kind in ('gate','up','down')}
        weights.update({'output.weight':dict(bytes=700*1024**2),'token_embd.weight':dict(bytes=400*1024**2)})
        candidates = quantization_candidates(weights)
        for candidate in candidates:
            self.assertEqual(candidate['projector_device'],'CUDA0')
            self.assertEqual(candidate['load_mode'],'none')
            self.assertLessEqual(candidate['primary_weight_mib'],6050)
            self.assertEqual(candidate['primary_weight_mib']+candidate['secondary_weight_mib'],64*120+700)
            self.assertNotIn('blk.64',candidate['override'])
        with self.assertRaises(ValueError):
            quantization_candidates({'output.weight':dict(bytes=1),'token_embd.weight':dict(bytes=1)})

    def test_local_load_mode_is_none_without_changing_model_placement(self):
        from desktop_agent.agent import DesktopServer, Settings
        from desktop_agent.models import MODEL_PRESETS, model_server_options, uses_dual_gpu
        for preset in MODEL_PRESETS.values():
            with self.subTest(model=preset['model']):
                settings = Settings(model=preset['model'],projector=preset['projector'])
                shared = model_server_options(settings.model,settings.projector,preset['image_max_tokens'])
                options = DesktopServer().settings_for(settings.executable,settings.model,settings.projector,
                    preset['image_min_tokens'],preset['image_max_tokens'])[-1]
                self.assertEqual(options.count('--load-mode'),1)
                self.assertEqual(options[options.index('--load-mode')+1],'none')
                self.assertNotIn('--load-mode',shared)
                self.assertNotIn('--ctx-checkpoints',options)
                self.assertEqual('--device' in options,uses_dual_gpu(settings.model,settings.projector))
                for flag in ('--override-tensor','-ngl','-b','-ub','-ctk','-ctv','-t','-tb'):
                    if flag in shared:
                        self.assertEqual(options[options.index(flag)+1],shared[shared.index(flag)+1])

    @unittest.skipUnless(os.name == 'nt','Windows working-set APIs required')
    def test_resource_probe_classifies_resident_pages_without_double_counting(self):
        import importlib.util
        if any(importlib.util.find_spec(name) is None for name in ('numpy','psutil')):
            self.skipTest('Optional resource research dependencies are not installed')
        from desktop_agent.benchmark_resources import memory_snapshot, system_memory
        snapshot = memory_snapshot(os.getpid())
        accounted = sum(item['resident'] for item in snapshot['totals'].values())
        self.assertEqual(accounted+snapshot['unclassified_resident'],snapshot['queried_resident'])
        self.assertGreater(snapshot['totals']['private']['resident'],0)
        for item in snapshot['totals'].values():
            self.assertLessEqual(item['nonshareable_resident'],item['resident'])
            self.assertLessEqual(item['resident'],item['virtual_committed'])
        memory = system_memory()
        self.assertGreater(memory['physical_total'],memory['physical_available'])

    def test_placement_probe_changes_only_selected_runtime_options(self):
        from desktop_agent.benchmark_placement import PlacementServer
        from desktop_agent.agent import DesktopServer, Settings
        settings = Settings()
        candidate = dict(override=r'^output\.weight$=CPU',threads=10,batch_threads=12,ubatch=128)
        probe = PlacementServer(candidate)
        before = DesktopServer().settings_for(settings.executable,settings.model,settings.projector,1024,1024)
        after = probe.settings_for(settings.executable,settings.model,settings.projector,1024,1024)
        options = list(after[-1])
        for flag,key in (('--override-tensor','override'),('-t','threads'),('-tb','batch_threads'),('-ub','ubatch')):
            self.assertEqual(options[options.index(flag)+1],str(candidate[key]))
            options[options.index(flag)+1] = before[-1][before[-1].index(flag)+1]
        self.assertEqual(options[-2:],['-lv','4'])
        self.assertEqual(tuple(options[:-2]),before[-1])
        self.assertEqual(after[:-1],before[:-1])

    def test_secondary_placement_keeps_primary_and_moves_only_requested_tensors(self):
        from desktop_agent.benchmark_placement import PlacementServer, secondary_candidates
        from desktop_agent.agent import Settings
        weights = {name:dict(bytes=1024) for name in ('output.weight','token_embd.weight',
            'blk.54.ffn_gate.weight','blk.55.ffn_gate.weight','blk.63.ffn_down.weight')}
        candidates = {candidate['name']:candidate for candidate in secondary_candidates(weights,'CUDA0','Vulkan0')}
        self.assertEqual(candidates['primary_control']['devices'],'CUDA0')
        self.assertEqual(candidates['primary_control']['projector_device'],'CUDA0')
        self.assertEqual(candidates['primary_control']['gpu_layers'],65)
        self.assertEqual(candidates['secondary_ffn']['cpu_tensors'],['output.weight'])
        self.assertEqual(candidates['secondary_output']['secondary_tensors'],['output.weight'])
        combined = candidates['secondary_ffn_output']
        self.assertEqual(combined['cpu_tensors'],[])
        self.assertEqual(combined['secondary_weight_mib'],0.003)
        self.assertNotIn('token_embd',combined['override'])
        self.assertNotIn('54',combined['override'])
        settings = Settings()
        options = PlacementServer(combined).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        self.assertEqual(options[options.index('--device')+1],'CUDA0,Vulkan0')
        self.assertEqual(options[options.index('--split-mode')+1],'layer')
        self.assertEqual(options[options.index('--tensor-split')+1],'1,0')
        self.assertEqual(options.count('--load-mode'),1)
        self.assertEqual(options[options.index('--load-mode')+1],'auto')
        self.assertIn('=Vulkan0',options[options.index('--override-tensor')+1])
        self.assertEqual(candidates['secondary_tail55']['tensor_split'],'55,10')
        self.assertEqual(candidates['secondary_tail55']['override'],r'^token_embd\.weight$=CPU')
        self.assertEqual(candidates['secondary_ffn_output_ub64']['ubatch'],64)
        self.assertEqual(combined['ubatch'],256)
        self.assertEqual(candidates['primary_control_batch12']['batch_threads'],12)
        self.assertEqual(candidates['secondary_ffn61_output']['cpu_tensors'],['blk.55.ffn_gate.weight'])
        self.assertEqual(candidates['secondary_ffn61_output']['secondary_tensors'],['blk.63.ffn_down.weight','output.weight'])
        vision = candidates['vision_secondary_cuda_output']
        self.assertEqual(vision['projector_device'],'Vulkan0')
        self.assertNotIn('output.weight',vision['secondary_tensors'])
        self.assertIn(r'^output\.weight$=CUDA0',vision['override'])
        options = PlacementServer(vision).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        self.assertEqual(options[options.index('--mmproj-device')+1],'Vulkan0')
        self.assertEqual(candidates['vision_secondary_cpu_tail']['devices'],'CUDA0')
        self.assertEqual(candidates['vision_secondary_cpu_tail']['override'],candidates['primary_control']['override'])
        self.assertNotIn('blk.55.ffn_gate.weight',candidates['vision_secondary_cuda_ffn61']['secondary_tensors'])
        self.assertEqual(candidates['gpu_tail_threads6']['threads'],6)
        controlled = dict(combined,poll=0,poll_batch=0,checkpoints=0,load_mode='none')
        options = PlacementServer(controlled).settings_for(settings.executable,settings.model,settings.projector,1024,1024)[-1]
        self.assertEqual(options.count('--load-mode'),1)
        for flag,expected in (('--poll','0'),('--poll-batch','0'),('--ctx-checkpoints','0'),('--load-mode','none')):
            self.assertEqual(options[options.index(flag)+1],expected)
        unknown = 'Qwen3.8-27B-UD-Q2_K_XL.gguf'
        research = dict(combined,load_mode='none',gpu_layers=65,batch=512,cache_k='q8_0',cache_v='q8_0',
                        fit_target=128,no_host=True)
        options = PlacementServer(research).settings_for(settings.executable,unknown,settings.projector,1024,1024)[-1]
        for flag,expected in (('-ngl','65'),('-b','512'),('-ctk','q8_0'),('-ctv','q8_0'),('--load-mode','none')):
            self.assertEqual(options.count(flag),1)
            self.assertEqual(options[options.index(flag)+1],expected)
        self.assertEqual(options.count('--no-host'),1)
        self.assertEqual(options[options.index('--reasoning-format')+1],'deepseek')

    def test_qwen_effort_settings_migrate_and_roundtrip(self):
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            for tokens,expected in ((128,'low'),(256,'medium'),(512,'xhigh'),(1024,'xhigh')):
                path.write_text(json.dumps({'reasoning_tokens':tokens,'reasoning_enabled':True}),encoding='utf-8')
                settings = Settings.load(path)
                self.assertEqual(settings.reasoning_effort,expected)
                selected = settings.with_reasoning_level('medium')
                self.assertEqual(selected.reasoning_tokens,tokens)
                selected.save(path)
                self.assertEqual(Settings.load(path),selected)
            path.write_text(json.dumps({'reasoning_effort':'high'}),encoding='utf-8')
            with self.assertRaises(ValueError):
                Settings.load(path)

    def test_reasoning_budget_dialog_and_escape_policy(self):
        import tkinter as tk
        from unittest.mock import Mock
        from desktop_agent.agent import Settings
        from desktop_agent.app import Console
        from game_agent.core import interruption_reason, INPUT_TAG
        for message in (0x100,0x104):
            self.assertIsNone(interruption_reason(0,message,0x1B,False,False))
            self.assertEqual(interruption_reason(0,message,0x77,False,False),'Emergency stop (F8)')
            self.assertIsNotNone(interruption_reason(0,message,0x1B,False))
            self.assertIsNone(interruption_reason(INPUT_TAG,message,0x77,False,False))
        root=tk.Tk()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root.busy=False;root.settings=Settings(reasoning_enabled=True,reasoning_effort='xhigh')
                root.data=Path(folder);root.status=tk.StringVar(master=root);root.enqueue=Mock()
                root.update()
                dialog=Console.reasoning_budget_dialog(root)
                dialog.geometry('440x200');root.update()
                controls={child.winfo_name():child for frame in dialog.winfo_children() for child in frame.winfo_children()}
                self.assertEqual(controls['budget'].get(),'2048')
                controls['budget'].set('-2');controls['save'].invoke();root.update()
                self.assertTrue(dialog.winfo_exists())
                self.assertFalse((root.data/'settings.json').exists())
                for name in ('budget','save','cancel','defaults'):
                    control=controls[name]
                    self.assertTrue(control.winfo_ismapped())
                    self.assertLessEqual(control.winfo_rooty()+control.winfo_height(),dialog.winfo_rooty()+dialog.winfo_height())
                    self.assertLessEqual(control.winfo_rootx()+control.winfo_width(),dialog.winfo_rootx()+dialog.winfo_width())
                controls['budget'].set('1024');controls['save'].invoke();root.update()
                self.assertFalse(dialog.winfo_exists())
                saved=Settings.load(root.data/'settings.json')
                self.assertEqual((saved.reasoning_budget_tokens,saved.reasoning_effort),(1024,'xhigh'))
                root.enqueue.assert_called_once_with('reasoning',root.settings)
                before=(root.data/'settings.json').read_bytes()
                dialog=Console.reasoning_budget_dialog(root);root.update()
                controls={child.winfo_name():child for frame in dialog.winfo_children() for child in frame.winfo_children()}
                controls['defaults'].invoke();self.assertEqual(controls['budget'].get(),'2048')
                controls['cancel'].invoke();root.update()
                self.assertEqual((root.data/'settings.json').read_bytes(),before)
                root.busy=True
                self.assertIsNone(Console.reasoning_budget_dialog(root))
        finally:
            root.destroy()

    def test_reasoning_budget_settings_validate_and_preserve_effort(self):
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            path.write_text(json.dumps({'reasoning_effort':'xhigh','reasoning_enabled':True}),encoding='utf-8')
            self.assertEqual(Settings.load(path).reasoning_budget_tokens,2048)
            for budget in (-1,0,1,2048,65536):
                settings = Settings(reasoning_effort='xhigh',reasoning_enabled=True,reasoning_budget_tokens=budget)
                settings.save(path)
                loaded = Settings.load(path)
                self.assertEqual(loaded,settings)
                self.assertEqual(loaded.with_reasoning_level('low').reasoning_budget_tokens,budget)
            for budget in (-2,65537,True,1.5,'2048',None):
                path.write_text(json.dumps({'reasoning_budget_tokens':budget}),encoding='utf-8')
                with self.assertRaises(ValueError):
                    Settings.load(path)

    def test_qwen27_reasoning_effort_and_other_local_unlimited_budget(self):
        from desktop_agent.agent import Settings, DesktopServer
        from game_agent.models import MODEL_PRESETS
        for name in ('qwen38_27b_iq2_s','qwen38_27b_iq2_xxs'):
            preset = MODEL_PRESETS[name]
            settings = Settings(model=preset['model'],projector=preset['projector'])
            limits = []
            for level in ('low','medium','xhigh'):
                selected = settings.with_reasoning_level(level)
                self.assertEqual(selected.reasoning_effort,level)
                self.assertTrue(selected.reasoning_enabled)
                limits.append(selected.output_budget)
                options = DesktopServer().settings_for(selected.executable,selected.model,selected.projector,
                    1024,1024,True,selected.reasoning_tokens)[-1]
                self.assertNotIn('--reasoning-budget',options)
                self.assertEqual(options[options.index('--reasoning-format')+1],'deepseek')
            self.assertEqual(len(set(limits)),1)
            server = DesktopServer()
            self.assertEqual(server.settings_for(settings.executable,settings.model,settings.projector,1024,1024,True,128),
                             server.settings_for(settings.executable,settings.model,settings.projector,1024,1024,False,1024))
        preset = MODEL_PRESETS['qwen38_nvfp4']
        small = Settings(model=preset['model'],projector=preset['projector']).with_reasoning_level('extended')
        self.assertEqual(small.reasoning_tokens,1024)
        options = DesktopServer().settings_for(small.executable,small.model,small.projector,192,384,True,small.reasoning_tokens)[-1]
        self.assertEqual(options[options.index('--reasoning-budget')+1],'-1')
        self.assertEqual(small.reasoning_levels,('unlimited',))
        self.assertEqual(small.reasoning_level,'unlimited')
        self.assertTrue(small.with_reasoning_level('unlimited').reasoning_enabled)

    def test_tool_catalog_discovers_groups_and_enforces_permissions(self):
        from desktop_agent.protocol import ToolCatalog, compact_schema
        catalog = ToolCatalog(dict(screen=True,input=True,browser=False))
        self.assertIn('desktop_click',catalog.names())
        self.assertNotIn('desktop_record',catalog.names())
        self.assertNotIn('desktop_input',catalog.names())
        self.assertNotIn('browser_read',catalog.names())
        with self.assertRaises(ValueError):
            catalog.normalize({'tool':'desktop_record','arguments':{'seconds':2}})
        catalog.load(['recording','advanced_input'])
        self.assertIn('desktop_record',catalog.names())
        self.assertIn('desktop_input',catalog.names())
        self.assertNotIn('browser_record',catalog.names())
        with self.assertRaises(ValueError):
            catalog.load(['browser'])
        disabled = ToolCatalog(dict(screen=False,input=False,browser=False),'record browser')
        self.assertEqual(set(disabled.names()),{'finish','load_tool_group'})
        with self.assertRaises(ValueError):
            disabled.load(['recording'])
        discovered = ToolCatalog(dict(screen=True,input=True,browser=True),'Record the browser with attachment',True)
        self.assertTrue({'recording','browser','files'} <= discovered.loaded)
        self.assertNotIn('job_start',json.dumps(compact_schema(discovered.names())))

    def test_compact_calls_fill_defaults_without_weakening_validation(self):
        from desktop_agent.protocol import normalize_call, compact_schema
        clicked = normalize_call({'tool':'desktop_click','arguments':{'x':100,'y':200}})
        self.assertEqual(clicked['arguments'],dict(x=100,y=200,button='left',clicks=1))
        self.assertEqual(normalize_call({'tool':'finish','arguments':{'text':'Done'}})['message'],'Done')
        self.assertEqual(normalize_call({'tool':'desktop_key_queue','arguments':{'steps':[{'key':'Tab'}]}})['arguments']['steps'][0]['delay_ms'],0)
        for name,amount in (('desktop_scroll',-2),('browser_scroll',2)):
            arguments = dict(direction='down',amount=2,**({'x':100,'y':200} if name.startswith('desktop') else {}))
            self.assertEqual(normalize_call({'tool':name,'arguments':arguments})['arguments']['amount'],amount)
        key = normalize_call({'tool':'desktop_input','arguments':{'kind':'key','key':'Delete'}})
        self.assertTrue(approval_reason(key,'routine'))
        for call in ({'tool':'desktop_click','arguments':{'x':100}},
                     {'tool':'desktop_input','arguments':{'kind':'click'}},
                     {'tool':'desktop_input','arguments':{'kind':'text','mode':'message','text':'Hello'}},
                     {'tool':'desktop_key','arguments':{'key':'Unknown'}},
                     {'tool':'desktop_click','arguments':{'x':True,'y':0}},
                     {'tool':'finish','arguments':{'text':'Done'},'risk':'routine'}):
            with self.assertRaises(ValueError):
                normalize_call(call)
        with self.assertRaises(ValueError):
            normalize_call({'tool':'desktop_capture'},allowed={'finish'})
        nested = normalize_call({'tool':'desktop_record','arguments':{'seconds':2,'background':True}})
        self.assertEqual(nested['tool'],'job_start')
        self.assertEqual(nested['arguments']['action']['arguments'],{'seconds':2,'fps':5})
        schema = compact_schema(['finish','desktop_key'])
        self.assertNotIn('risk',json.dumps(schema))

    def test_local_presets_pair_files_reset_defaults_and_preserve_other_settings(self):
        from desktop_agent.agent import Settings
        from game_agent.models import MODEL_PRESETS
        with tempfile.TemporaryDirectory() as folder:
            for name in ('qwen38_nvfp4','qwen38_27b_iq2_s'):
                preset = MODEL_PRESETS[name]
                for field in ('model','projector'):
                    (Path(folder)/preset[field]).touch()
            original = Settings(backend='api',context_tokens=12288,reasoning_enabled=True,reasoning_tokens=128)
            small = original.with_local_preset('qwen38_nvfp4',folder)
            self.assertEqual(small.backend,'local')
            self.assertEqual(small.context_tokens,16384)
            self.assertEqual(small.local_image_tokens,(192,384))
            self.assertIn('Distill-Heretic',small.local_label)
            self.assertEqual(Path(small.projector).name,'mmproj-qwen38-BF16.gguf')
            large = small.with_local_preset('qwen38_27b_iq2_s')
            self.assertEqual(large.context_tokens,8192)
            self.assertEqual(large.local_image_tokens,(1024,1024))
            self.assertTrue(large.reasoning_enabled)
            self.assertEqual(large.reasoning_tokens,128)
            self.assertEqual(large.api,original.api)
            self.assertEqual(original.backend,'api')
            path = Path(folder)/'settings.json'
            small.save(path)
            self.assertEqual(Settings.load(path),small)
            with self.assertRaises(FileNotFoundError):
                original.with_local_preset('ui_venus',folder)
            with self.assertRaises(ValueError):
                original.with_local_preset('missing',folder)

    def test_local_model_uses_its_own_image_tokens_and_server_options(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import DesktopServer, Model, Settings
        from game_agent.models import MODEL_PRESETS
        for name in ('qwen38_27b_iq2_s','qwen38_nvfp4'):
            with self.subTest(preset=name):
                preset = MODEL_PRESETS[name]
                settings = Settings(model=preset['model'],projector=preset['projector'],context_tokens=12288)
                model = Model(settings)
                model.server.start = Mock(return_value='http://127.0.0.1:1234')
                model.ensure(threading.Event())
                minimum,maximum = preset['image_min_tokens'],preset['image_max_tokens']
                self.assertEqual(model.server.start.call_args.args[5:7],(minimum,maximum))
                options = model.server.settings_for(settings.executable,settings.model,settings.projector,minimum,maximum)[-1]
                self.assertEqual(options[options.index('-c')+1],'12288')
                self.assertEqual('--override-tensor' in options,name == 'qwen38_27b_iq2_s')
                self.assertEqual('--no-host' in options,name == 'qwen38_27b_iq2_s')
            previous_preset = MODEL_PRESETS['qwen38_nvfp4' if name == 'qwen38_27b_iq2_s' else 'qwen38_27b_iq2_s']
            previous = Settings(model=previous_preset['model'],projector=previous_preset['projector'])
            model.server.loaded_settings = model.server.settings_for(previous.executable,previous.model,previous.projector,*previous.local_image_tokens)
            model.server.close = Mock()
            model.ensure(threading.Event())
            model.server.close.assert_called_once()
            model.server.close.reset_mock()
            model.server.loaded_settings = model.server.settings_for(settings.executable,settings.model,settings.projector,minimum,maximum)
            model.ensure(threading.Event())
            model.server.close.assert_not_called()
        with self.assertRaisesRegex(ValueError,'matching projector'):
            DesktopServer().settings_for('server.exe',MODEL_PRESETS['qwen38_nvfp4']['model'],Settings().projector,192,384)

    def test_context_setting_changes_only_desktop_server_and_validates_range(self):
        from desktop_agent.agent import DesktopServer, Settings, validate_context
        from game_agent.runtime import LocalServer
        settings = Settings()
        server = DesktopServer()
        server.context_tokens = 16384
        changed = server.settings_for(settings.executable,settings.model,settings.projector,1024,1024)
        original = LocalServer.settings_for(settings.executable,settings.model,settings.projector,1024,1024)
        self.assertEqual(changed[-1][changed[-1].index('-c')+1],'16384')
        self.assertEqual(original[-1][original[-1].index('-c')+1],'8192')
        for invalid in (True,1024,8193,66560):
            with self.assertRaises(ValueError):
                validate_context(invalid)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'settings.json'
            Settings(context_tokens=12288).save(path)
            self.assertEqual(Settings.load(path).context_tokens,12288)

    def test_xxs_retains_six_cpu_ffn_blocks_with_q8_projector(self):
        import re
        from desktop_agent.agent import DesktopServer
        from game_agent.models import MODEL_PRESETS
        from game_agent.runtime import LocalServer
        from desktop_agent.models import model_server_options
        server = DesktopServer()
        for name in ('qwen38_27b_iq2_xxs','qwen38_27b_iq2_s'):
            preset = MODEL_PRESETS[name]
            for projector in (preset['projector'],'mmproj-F16.gguf'):
                original = LocalServer.settings_for('server.exe',preset['model'],projector,1024,1024)
                for context in (8192,12288):
                    server.context_tokens = context
                    changed = server.settings_for('server.exe',preset['model'],projector,1024,1024)
                    options = list(changed[-1])
                    pattern = options[options.index('--override-tensor')+1]
                    if name == 'qwen38_27b_iq2_xxs' and projector == preset['projector']:
                        regex = pattern.removesuffix('=CPU')
                        self.assertEqual([block for block in range(64) if re.fullmatch(regex,f'blk.{block}.ffn_gate.weight')],list(range(58,64)))
                        self.assertIsNone(re.fullmatch(regex,'output.weight'))
                        self.assertIsNone(re.fullmatch(regex,'blk.60.attn_q.weight'))
                    options[options.index('-c')+1] = '8192'
                    for flag,value in (('--load-mode','none'),('--cache-ram','0'),
                                       ('--reasoning','auto'),('--reasoning-format','deepseek')):
                        self.assertEqual(options.count(flag),1)
                        index = options.index(flag)
                        self.assertEqual(options[index+1],value)
                        del options[index:index+2]
                    expected = tuple(model_server_options(preset['model'],projector,1024)) if name=='qwen38_27b_iq2_s' and projector==preset['projector'] else original[-1]
                    self.assertEqual(tuple(options),expected)

    def test_reasoning_presets_and_model_target_selection(self):
        import threading
        from unittest.mock import Mock, patch
        from desktop_agent.agent import Settings, REASONING_LEVELS
        from desktop_agent.tools import Tools
        for level, tokens in REASONING_LEVELS.items():
            settings = Settings(model='Qwen3.8-9B-distill-heretic_nvfp4_q4_k_m.gguf').with_reasoning_level(level)
            self.assertTrue(settings.reasoning_enabled)
            self.assertEqual(settings.reasoning_tokens,tokens)
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        self.assertTrue(tools.allow_screen and tools.allow_input and tools.allow_browser)
        row = {'handle':12,'pid':34,'title':'Editor'}
        with patch('desktop_agent.tools.list_windows',return_value=[row]):
            self.assertIn('Editor',tools.execute(dict(message='Find window',tool='window_list',arguments={},risk='routine')).text)
            tools.execute(dict(message='Choose window',tool='window_select',arguments={'handle':12,'pid':34},risk='routine'))
            self.assertEqual(tools.window,row)
            with self.assertRaises(ValueError):
                tools.execute(dict(message='Choose window',tool='window_select',arguments={'handle':12,'pid':99},risk='routine'))

    def test_video_limits_and_capture_permission_without_input(self):
        import threading
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        for tool in ('desktop_record','browser_record'):
            action = dict(message='Observe motion',tool=tool,arguments={'seconds':5,'fps':5},risk='routine')
            validate_action(action)
            self.assertFalse(approval_reason(action,'routine'))
            for arguments in ({'seconds':0,'fps':5},{'seconds':31,'fps':5},{'seconds':5,'fps':16},
                              {'seconds':5,'fps':True},{'seconds':1}):
                with self.assertRaises(ValueError):
                    validate_action(dict(action,arguments=arguments))
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        tools.window = {'handle':1,'pid':2}
        tools.desktop = Mock(return_value='recorded')
        tools.allow_screen = tools.allow_input = False
        action['tool'] = 'desktop_record'
        with patch('desktop_agent.tools.windows.window_title',return_value='Video test'):
            with self.assertRaisesRegex(ValueError,'Screen capture disabled'):
                tools.execute(action)
            tools.allow_screen = True
            self.assertEqual(tools.execute(action),'recorded')
        self.assertFalse(tools.allow_input)

    def test_key_queues_validate_every_step_and_preserve_risk_checks(self):
        for tool in ('desktop_key_queue','browser_key_queue'):
            action = dict(message='Use focused field',tool=tool,risk='routine',
                          arguments={'steps':[{'key':'Tab','delay_ms':0},{'key':'Enter','delay_ms':250}]})
            self.assertEqual(validate_action(action),action)
            self.assertFalse(approval_reason(action,'routine'))
            self.assertTrue(approval_reason(action,'manual'))
            for key in ('Delete','Shift+Delete','Alt+F4','Control+w','Control+Enter'):
                self.assertTrue(approval_reason(dict(action,arguments={'steps':[{'key':key,'delay_ms':100}]}),'routine'))
            for steps in ([],[{'key':'Tab'}],[{'key':'Tab','delay_ms':True}],
                          [{'key':'Tab','delay_ms':-1}],[{'key':'Tab','delay_ms':5001}],
                          [{'key':'Unknown','delay_ms':0}],[{'key':'Tab','delay_ms':0,'extra':1}],
                          [{'key':'Tab','delay_ms':0}]*33,[{'key':'Tab','delay_ms':5000}]*13):
                with self.assertRaises(ValueError):
                    validate_action(dict(action,arguments={'steps':steps}))

    def test_no_observation_id_and_ordinary_keys_can_be_automatic(self):
        self.assertNotIn('observation',json.dumps(action_schema()))
        for key in ('Enter','Tab','Control+a','Control+v','Control+s'):
            action = dict(message='Use focused field',tool='desktop_key',risk='routine',arguments={'key':key})
            self.assertFalse(approval_reason(action,'routine'))
            self.assertTrue(approval_reason(action,'manual'))
            self.assertFalse(approval_reason(action,'routine',{'submit':True,'label':'Ordinary field'}))
        for key in ('Delete','Shift+Delete','Alt+F4','Control+w'):
            self.assertTrue(approval_reason(dict(action,arguments={'key':key}),'routine'))

    def test_sensitive_intent_and_actual_browser_target_override_auto_mode(self):
        action = dict(message='Click control',tool='browser_click',risk='routine',arguments={'selector':'#button'})
        self.assertFalse(approval_reason(action,'routine',{'label':'Next page'}))
        self.assertTrue(approval_reason(action,'routine',{'label':'Delete account'}))
        self.assertTrue(approval_reason(action,'routine',{'submit':True}))
        self.assertFalse(approval_reason(action,'routine',{'submit':True,'search':True}))
        self.assertTrue(approval_reason(dict(action,risk='sensitive'),'routine'))
        enter = dict(message='Use focused field',tool='browser_key',risk='routine',arguments={'key':'Enter'})
        self.assertFalse(approval_reason(enter,'routine',{'submit':True,'label':'Continue'}))
        self.assertTrue(approval_reason(enter,'routine',{'submit':True,'label':'Send message'}))

    def test_validation_rejects_executable_urls_and_bad_inputs(self):
        for url in ('file:///C:/secret','javascript:alert(1)','http://user:secret@example.com'):
            with self.assertRaises(ValueError):
                web_url(url)
        action = dict(message='Click',tool='desktop_click',risk='routine',arguments=dict(x=0,y=0,button='left',clicks=1))
        for value in (True,-1,1001,'300'):
            with self.assertRaises(ValueError):
                validate_action(dict(action,arguments=dict(action['arguments'],x=value)))


class StoreTests(unittest.TestCase):
    def test_agent_continues_after_evicting_an_oversized_previous_turn(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Settings
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            old_request = store.append(identifier,'user','Old request '*6000)
            old_reply = store.append(identifier,'assistant','Old reply '*6000)
            model = Mock(settings=Settings(context_tokens=8192),count=lambda text:len(text)//4,partial='')
            model.generate.return_value = ({'tool':'finish','arguments':{'text':'Continued'}},{})
            tools = Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='routine')
            Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Current task')
            model.generate.assert_called_once()
            self.assertNotIn('Old reply',json.dumps(model.generate.call_args.args[0]))
            self.assertEqual(store.context_selection(identifier)['excluded_ids'],[old_request,old_reply])
            self.assertEqual(len(store.events(identifier)),4)
            self.assertEqual(json.loads(store.events(identifier)[-1]['content'])['message'],'Continued')

    def test_context_selection_persists_separately_and_clears_on_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier,other = store.create(),store.create()
            request = store.append(identifier,'user','Old request')
            store.append(identifier,'assistant','Answer')
            original = store.events(identifier)
            selection = dict(excluded_ids=[request],tokens=700,budget=1000,through_id=2)
            store.save_context_selection(identifier,selection)
            self.assertEqual(Store(folder).context_selection(identifier),selection)
            self.assertEqual(store.context_selection(other),{})
            self.assertEqual(store.events(identifier),original)
            store.edit_message(identifier,request,'Updated request')
            self.assertEqual(store.context_selection(identifier),{})
            store.save_context_selection(identifier,selection)
            store.delete_session(identifier)
            self.assertEqual(store.context_selection(identifier),{})

    def test_context_evicts_previous_turn_and_old_tool_pairs_without_deleting_events(self):
        from copy import deepcopy
        from desktop_agent.agent import pack_messages
        from desktop_agent.api import native_history
        events = [dict(id=1,role='user',content='Old request '*200),
                  dict(id=2,role='assistant',content='Old answer '*200),
                  dict(id=3,role='user',content='Current request')]
        for index in range(3):
            events += [dict(id=4+index*2,role='assistant',content=json.dumps({'tool':'browser_read','arguments':{}})),
                       dict(id=5+index*2,role='tool',content='Page '+str(index)+' '+('body '*80),metadata={'tool':'browser_read','status':'delivered'})]
        original = deepcopy(events)
        latest = [events[2],*events[-2:]]
        _,_,needed = pack_messages(latest,'system',len,20000,state={})
        selection = {}
        messages,dropped,used = pack_messages(events,'system',len,needed+240,state={},selection=selection)
        self.assertLessEqual(used,needed+240)
        self.assertGreater(dropped,0)
        self.assertEqual(selection['excluded_ids'],[1,2,4,5,6,7])
        self.assertEqual([message.get('_event_id') for message in messages[1:-1]],[3,8,9])
        self.assertEqual(events,original)
        self.assertNotIn('Page 0',json.dumps(messages))
        self.assertIn('Page 2',json.dumps(messages))
        wire = native_history(messages[1:])
        self.assertEqual(wire[1]['tool_calls'][0]['id'],wire[2]['tool_call_id'])
        self.assertEqual(wire[2]['role'],'tool')
        for message in wire:
            self.assertNotIn('_event_id',message)
        self.assertNotIn('_event_id',json.loads(wire[2]['content']))
        expanded = {}
        pack_messages(events,'system',len,20000,selection=expanded)
        self.assertEqual(expanded['excluded_ids'],[])

    def test_context_can_exclude_single_oversized_result_but_keeps_current_request(self):
        from desktop_agent.agent import pack_messages
        events = [dict(id=1,role='user',content='Read the document'),
                  dict(id=2,role='assistant',content=json.dumps({'tool':'file_read','arguments':{'path':'C:/test.txt'}})),
                  dict(id=3,role='tool',content='Large document '*1000,metadata={'tool':'file_read','status':'delivered'})]
        selection = {}
        messages,_,used = pack_messages(events,'system',len,700,state={},selection=selection)
        self.assertLessEqual(used,700)
        self.assertEqual(selection['excluded_ids'],[2,3])
        self.assertEqual(messages[1]['content'],'Read the document')
        self.assertIn('Retrieve full logs',messages[-1]['content'])
        with self.assertRaisesRegex(ValueError,'Current user request'):
            pack_messages([dict(id=1,role='user',content='long '*1000)],'system',len,700)

    def test_cache_prefix_stays_fixed_when_only_runtime_state_changes(self):
        from desktop_agent.agent import pack_messages, system_prompt
        from desktop_agent.protocol import ToolCatalog
        capabilities = dict(screen=True,input=True,browser=False,approval='routine',window=None,history_file='first.jsonl')
        catalog = ToolCatalog(capabilities)
        first_system = system_prompt(capabilities,catalog)
        changed = dict(capabilities,window={'handle':12,'pid':34,'title':'Test'},history_file='second.jsonl')
        self.assertEqual(first_system,system_prompt(changed,catalog))
        events = [dict(id=1,role='user',content='Inspect the selected window'),
                  dict(id=2,role='assistant',content=json.dumps(dict(tool='desktop_capture',arguments={}))),
                  dict(id=3,role='tool',content='Captured',metadata={'tool':'desktop_capture','status':'delivered'})]
        first,_,_ = pack_messages(events,first_system,len,20000,state={'image_attached':True,'steps_remaining':19})
        second,_,_ = pack_messages(events,first_system,len,20000,state={'image_attached':False,'steps_remaining':18})
        self.assertEqual(first[:-1],second[:-1])
        self.assertNotEqual(first[-1]['content'],second[-1]['content'])
        self.assertEqual(first[-1]['role'],'user')
        self.assertTrue(first[-1]['_status'])
        self.assertNotIn('steps_remaining',first[0]['content'])
        extended = [*events,dict(id=4,role='assistant',content=json.dumps({'tool':'desktop_key','arguments':{'key':'Tab'}})),
                dict(id=5,role='tool',content='Delivered',metadata={'tool':'desktop_key','status':'delivered'})]
        later,_,_ = pack_messages(extended,first_system,len,20000,state={'image_attached':False,'steps_remaining':17})
        self.assertEqual(first[:-1],later[:len(first)-1])
        old = [dict(id=-1,role='user',content='old '*500),dict(id=0,role='assistant',content='Old reply')]
        protected,_,protected_size = pack_messages(events,first_system,len,20000)
        trimmed,dropped,_ = pack_messages(old+events+[dict(id=6,role='user',content='Continue')],first_system,len,protected_size+200)
        self.assertGreater(dropped,0)
        self.assertEqual(trimmed[0]['content'],first_system)
        self.assertIn('omitted_messages',trimmed[-1]['content'])

    def test_partial_reasoning_is_preserved_but_not_replayed_to_model(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Settings, pack_messages
        from game_agent.core import Halted
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            model = Mock(settings=Settings(),count=len,partial='',reasoning_text='Synthetic interrupted reasoning')
            model.generate.side_effect = Halted('Stopped')
            tools = Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='routine')
            Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Hello')
            last = store.events(identifier)[-1]
            self.assertEqual(last['metadata']['metrics']['reasoning'],model.reasoning_text)
            self.assertTrue(last['metadata']['metrics']['reasoning_partial'])
            messages,_,_ = pack_messages(store.events(identifier),'system',len,20000)
            self.assertNotIn(model.reasoning_text,json.dumps(messages))
            tools.execute.assert_not_called()

    def test_qwen_effort_payload_and_reasoning_stream_are_saved_separately(self):
        from dataclasses import replace
        import threading
        from unittest.mock import Mock, MagicMock, patch
        from desktop_agent.agent import Model, Settings
        from game_agent.models import MODEL_PRESETS
        synthetic_reasoning = 'Synthetic reasoning visibility test.'
        reply = {'tool':'finish','arguments':{'text':'Done'}}
        settings = Settings(reasoning_enabled=True)
        for level in ('low','medium','xhigh','none','9b'):
            selected = replace(settings,reasoning_effort=level if level in ('low','medium','xhigh') else 'medium',reasoning_enabled=level!='none')
            if level == '9b':
                preset = MODEL_PRESETS['qwen38_nvfp4']
                selected = replace(selected,model=preset['model'],projector=preset['projector'])
            response = MagicMock()
            response.__enter__.return_value = response
            events = [{'choices':[{'delta':{'reasoning_content':synthetic_reasoning},'finish_reason':None}]},
                      {'choices':[{'delta':{'content':json.dumps(reply)},'finish_reason':'stop'}]}]
            response.iter_lines.return_value = ['data: '+json.dumps(event) for event in events]+['data: [DONE]']
            client = MagicMock()
            client.__enter__.return_value = client
            client.post.return_value = response
            model = Model(selected)
            model.endpoint = 'http://127.0.0.1:1234'
            notify = Mock()
            with patch('desktop_agent.agent.session',return_value=client):
                action,metrics = model.generate([],None,threading.Event(),notify)
            payload = client.post.call_args.kwargs['json']
            if level == '9b':
                self.assertNotIn('reasoning_effort',payload)
                self.assertNotIn('reasoning_budget_tokens',payload)
            else:
                self.assertEqual(payload['reasoning_effort'],level)
                self.assertNotIn('reasoning_budget',payload)
                self.assertEqual(payload['reasoning_budget_tokens'],0 if level == 'none' else 2048)
                self.assertEqual(metrics['reasoning_budget_tokens'],payload['reasoning_budget_tokens'])
                self.assertEqual(payload['max_tokens'],-1)
            self.assertEqual(metrics['reasoning'],synthetic_reasoning)
            notify.assert_any_call('reasoning',synthetic_reasoning)
            self.assertEqual(action['message'],'Done')
            self.assertNotIn(synthetic_reasoning,action['message'])

    def test_reasoning_budget_keeps_answer_completion_and_cancellation_contract(self):
        import threading
        from unittest.mock import Mock, MagicMock, patch
        from desktop_agent.agent import Model, Settings, SYSTEM
        from game_agent.core import Halted
        self.assertIn('promptly use permitted search/retrieval tools',SYSTEM)
        self.assertIn('never bypass permissions or fabricate facts',SYSTEM)
        for budget in (-1,0,4,4096):
            for finish in ('stop','length','cancel'):
                with self.subTest(budget=budget,finish=finish):
                    stopped=threading.Event()
                    response=MagicMock();response.__enter__.return_value=response
                    def stream(**kwargs):
                        yield 'data: '+json.dumps({'choices':[{'delta':{'reasoning_content':'Synthetic reasoning.'}}]})
                        if finish == 'cancel':
                            stopped.set()
                        yield 'data: '+json.dumps({'choices':[{'delta':{'content':json.dumps({'tool':'finish','arguments':{'text':'Answer'}})},'finish_reason':finish}]})
                        yield 'data: [DONE]'
                    response.iter_lines.side_effect=stream
                    client=MagicMock();client.__enter__.return_value=client;client.post.return_value=response
                    model=Model(Settings(reasoning_enabled=True,reasoning_effort='xhigh',reasoning_budget_tokens=budget))
                    model.endpoint='http://127.0.0.1:1234'
                    with patch('desktop_agent.agent.session',return_value=client),patch.object(model,'cancel') as cancel:
                        if finish == 'stop':
                            action,metrics=model.generate([],None,stopped,Mock())
                            self.assertEqual(action['message'],'Answer')
                            self.assertEqual(metrics['reasoning_budget_tokens'],budget)
                        else:
                            with self.assertRaises(Halted if finish == 'cancel' else ValueError):
                                model.generate([],None,stopped,Mock())
                        cancel.assert_not_called()
                    payload=client.post.call_args.kwargs['json']
                    self.assertEqual(payload['reasoning_budget_tokens'],budget)
                    self.assertEqual(payload['reasoning_effort'],'xhigh')
                    self.assertEqual(payload['max_tokens'],-1)
                    self.assertTrue(payload['chat_template_kwargs']['enable_thinking'])

    def test_click_diagnostic_records_normalized_bounds_and_pointer_without_remapping(self):
        import threading
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        from game_agent.core import Region
        tools = Tools('.',threading.Event(),Mock(),Mock())
        tools.window = {'handle':12,'pid':34}
        tools.focus_window = Mock(return_value=(Region(-1600,100,800,600),Mock(title='Synthetic')))
        with patch('desktop_agent.tools.windows.DesktopInput') as backend, \
             patch('desktop_agent.tools.windows.virtual_screen',return_value=(-1920,0,3840,1200)), \
             patch('desktop_agent.tools.windows.cursor_position',side_effect=[(-1200,250),(-1199,250)]):
            result = tools.desktop('desktop_click',dict(x=500,y=250,button='left',clicks=1))
            receipt = json.loads(result.text)
            self.assertEqual(receipt['normalized'],[500,250])
            self.assertEqual(receipt['bounds'],[-1600,100,-800,700])
            self.assertEqual(receipt['requested_desktop'],[-1200,250])
            self.assertEqual(receipt['pointer_before_press'],[[-1200,250]])
            self.assertEqual(receipt['pointer_after_action'],[-1199,250])
            self.assertEqual(receipt['handle'],12)
            backend.return_value.down.assert_called_once_with('click','left')
            backend.return_value.up.assert_called_once_with('click','left')

    def test_agent_loads_groups_then_executes_compact_calls_without_extra_images(self):
        import threading
        from unittest.mock import Mock
        from PIL import Image
        from desktop_agent.agent import Agent, Settings
        from desktop_agent.tools import ToolResult
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            screenshot = Image.new('RGB',(20,20),'white')
            model = Mock(settings=Settings(context_tokens=16384),count=lambda value:len(value)//4,partial='')
            model.generate.side_effect = [
                ({'tool':'load_tool_group','arguments':{'groups':['advanced_input','recording']}},{}),
                ({'tool':'desktop_capture','arguments':{}},{}),
                ({'tool':'desktop_input','arguments':{'kind':'key','key':'Tab'}},{}),
                ({'tool':'finish','arguments':{'text':'Done'}},{})]
            tools = Mock(allow_screen=True,allow_input=True,allow_browser=False,window=None,mode='routine')
            tools.execute.side_effect = [ToolResult('Screenshot',screenshot),ToolResult('Delivered')]
            Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Proceed with this task')
            self.assertEqual(tools.execute.call_count,2)
            self.assertEqual(tools.execute.call_args.args[0]['arguments']['mode'],'general')
            calls = model.generate.call_args_list
            self.assertEqual([call.args[1] for call in calls],[None,None,screenshot,screenshot])
            self.assertNotIn('desktop_input:',calls[0].args[0][0]['content'])
            self.assertIn('advanced_input',calls[0].args[0][0]['content'])
            self.assertIn('desktop_input:',calls[1].args[0][0]['content'])
            self.assertNotIn('browser_record:',calls[1].args[0][0]['content'])
            self.assertEqual(json.loads(store.events(identifier)[-1]['content'])['message'],'Done')

    def test_agent_blocks_unloaded_call_then_allows_discovery(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Settings
        from desktop_agent.tools import ToolResult
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            model = Mock(settings=Settings(),count=lambda value:len(value)//4,partial='')
            model.generate.side_effect = [({'tool':'browser_read','arguments':{}},{}),
                ({'tool':'load_tool_group','arguments':{'groups':['browser']}},{}),
                ({'tool':'browser_read','arguments':{}},{}),({'tool':'finish','arguments':{'text':'Done'}},{})]
            tools = Mock(allow_screen=False,allow_input=False,allow_browser=True,window=None,mode='routine')
            tools.execute.return_value = ToolResult('Page data')
            Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Proceed')
            tools.execute.assert_called_once()
            self.assertTrue(any(event['metadata'].get('tool') == 'call_validation' for event in store.events(identifier)))

    def test_completed_background_jobs_are_collected_once_by_software(self):
        import threading
        from concurrent.futures import Future
        from unittest.mock import Mock
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.tools import ToolResult
        with tempfile.TemporaryDirectory() as folder:
            runner = ToolRunner(folder,threading.Event(),Mock(),Mock())
            try:
                future = Future()
                future.set_result(ToolResult('Completed'))
                runner.background['test'] = dict(action={'tool':'desktop_record'},future=future,
                                                stop=threading.Event(),ready=threading.Event())
                self.assertTrue(runner.job_summary()[0]['done'])
                result = runner.collect_ready()
                self.assertEqual(result[0].job_id,'test')
                self.assertEqual(result[0].tool,'desktop_record')
                self.assertEqual(runner.collect_ready(),[])
            finally:
                runner.close()

    def test_compact_request_intent_still_requires_approval(self):
        from desktop_agent.protocol import normalize_call
        action = normalize_call({'tool':'desktop_click','arguments':{'x':0,'y':0}})
        self.assertTrue(approval_reason(action,'routine',{'request_intent':'Delete this account'}))
        self.assertTrue(approval_reason(action,'manual'))
        self.assertTrue(approval_reason(action,'routine',{'label':'Pay now'}))
        self.assertFalse(approval_reason(action,'routine',{'request_intent':'Select the text'}))

    def test_context_excerpts_keep_original_logs_and_mark_tool_results(self):
        from desktop_agent.agent import pack_messages
        from desktop_agent.api import native_history
        events = [dict(id=1,role='user',content='Inspect'),
                  dict(id=2,role='assistant',content=json.dumps(dict(message='Read',tool='browser_read',arguments={},risk='routine'))),
                  dict(id=3,role='system',content='Automatic approval',metadata={'status':'approval'}),
                  dict(id=4,role='tool',content='DATA '*2000,metadata={'tool':'browser_read','status':'delivered'})]
        messages,_,_ = pack_messages(events,'system',len,budget=10000)
        result = json.loads(messages[-1]['content'])
        self.assertEqual(result['tool'],'browser_read')
        self.assertTrue(result['excerpt'])
        self.assertEqual(len(result['result']),1800)
        self.assertEqual(len(events[-1]['content']),10000)
        self.assertNotIn('risk',messages[2]['content'])
        native = native_history(messages[1:])
        self.assertEqual(native[1]['role'],'assistant')
        self.assertEqual(native[2]['role'],'tool')
        events[-1]['content'] = json.dumps({'scope':'managed_browser_only','text':'long '*2000,
                          'controls':[{'selector':'#keep-me','label':'Next'}]})
        events[-1]['metadata']['image'] = 'retained-capture.png'
        messages,_,_ = pack_messages(events,'system',len,budget=10000)
        result = json.loads(messages[-1]['content'])
        self.assertEqual(result['result']['controls'][0]['selector'],'#keep-me')
        self.assertEqual(result['image'],'retained-capture.png')
        events[-1]['content'] = 'full file page '*300
        events[-1]['metadata']['tool'] = 'file_read'
        messages,_,_ = pack_messages(events,'system',len,budget=10000)
        self.assertEqual(json.loads(messages[-1]['content'])['result'],events[-1]['content'])

    def test_history_compaction_preserves_failures_files_and_raw_records(self):
        from copy import deepcopy
        from desktop_agent.agent import pack_messages
        events = [dict(id=1,role='user',content='Inspect'),
            dict(id=2,role='system',content=json.dumps({'tool':'desktop_capture','mode':'automatic','reason':''}),metadata={'status':'approval'}),
            dict(id=3,role='tool',content=json.dumps({'source':'selected_window','handle':12,'bounds':[100,200,500,800],
                'width':400,'height':600,'notice':'Repeated capture explanation'}),metadata={'tool':'desktop_capture','status':'delivered','image':'C:/capture.png'}),
            dict(id=4,role='system',content=json.dumps({'tool':'desktop_click','mode':'denied','reason':'Payment'}),metadata={'status':'approval'}),
            dict(id=5,role='tool',content='Partially sent; do not replay '+'x'*2000,metadata={'tool':'desktop_key_queue','status':'stopped','error':'Different error'}),
            dict(id=6,role='tool',content=json.dumps({'text':'page '*700,'next_offset':3500}),metadata={'tool':'session_history','status':'delivered'}),
            dict(id=7,role='tool',content=json.dumps({'video':'C:/video.mp4','sample_times':[0,1],'reason':'cancelled'}),metadata={'tool':'desktop_record','status':'stopped','video':'C:/video.mp4','job_id':'job-one'})]
        original = deepcopy(events)
        messages,_,_ = pack_messages(events,'system',len,20000)
        self.assertEqual(events,original)
        records = [json.loads(message['content']) for message in messages[2:]]
        self.assertEqual(len(records),5)
        capture = records[0]
        self.assertEqual(capture['result']['bounds'],[100,200,500,800])
        self.assertEqual(capture['image'],'C:/capture.png')
        self.assertNotIn('width',capture['result'])
        self.assertNotIn('notice',capture['result'])
        self.assertEqual(records[1]['result']['mode'],'denied')
        self.assertEqual(records[2]['result'],original[4]['content'])
        self.assertEqual(records[2]['error'],'Different error')
        self.assertEqual(records[3]['result']['next_offset'],3500)
        self.assertEqual(records[3]['result']['text'],'page '*700)
        self.assertEqual(messages[-1]['content'].count('C:/video.mp4'),1)
        self.assertEqual(records[-1]['job_id'],'job-one')
        error_text = 'Input stopped; do not replay '+('sensitive failure '*200)
        error_event = dict(id=8,role='tool',content=error_text,metadata={'tool':'desktop_click','status':'error','error':error_text})
        messages,_,_ = pack_messages([events[0],error_event],'system',len,20000)
        failed = json.loads(messages[-1]['content'])
        self.assertEqual(failed['result'],error_text)
        self.assertNotIn('error',failed)
        self.assertNotIn('excerpt',failed)

    def test_context_count_excludes_internal_native_call_sidecars(self):
        from desktop_agent.agent import pack_messages
        from unittest.mock import Mock
        events = [dict(role='user',content='Inspect'),dict(role='assistant',content=json.dumps(dict(tool='desktop_click',arguments={'x':100,'y':200})))]
        count = Mock(side_effect=len)
        messages,_,used = pack_messages(events,'system',count,20000)
        self.assertIn('_call',messages[-1])
        counted = count.call_args.args[0]
        for message in json.loads(counted):
            self.assertNotIn('_call',message)
            self.assertNotIn('_internal_tool',message)
        call_record=json.loads(json.loads(counted)[-1]['content'])
        self.assertNotIn('_call',call_record)
        self.assertNotIn('_internal_tool',call_record)
        self.assertEqual(used,len(counted))

    def test_window_capture_process_decodes_pixels_and_rejects_invalid_output(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.tools import capture_window_image
        for code,pixels in ((0,b'\xff\x00\x00\x00\x00\xff'),(1,b''),(0,b'bad')):
            with self.subTest(code=code,length=len(pixels)), patch('desktop_agent.tools.subprocess.Popen') as spawn:
                process = spawn.return_value.__enter__.return_value
                process.communicate.return_value = pixels,None
                process.poll.return_value = process.returncode = code
                if code == 0 and len(pixels) == 6:
                    image = capture_window_image(12,(-2,0,0,1),threading.Event())
                    self.assertEqual(image.size,(2,1))
                    self.assertEqual(list(image.getdata()),[(255,0,0),(0,0,255)])
                else:
                    with self.assertRaises(OSError):
                        capture_window_image(12,(-2,0,0,1),threading.Event())
                process.kill.assert_not_called()

    def test_window_capture_process_is_reaped_after_timeout_or_cancellation(self):
        import subprocess
        import threading
        from unittest.mock import patch
        from desktop_agent.tools import capture_window_image
        from game_agent.core import Halted
        for cancelled in (False,True):
            stopped = threading.Event()
            with self.subTest(cancelled=cancelled), patch('desktop_agent.tools.subprocess.Popen') as spawn, \
                 patch('desktop_agent.tools.time.monotonic',side_effect=[100,100,104]):
                process = spawn.return_value.__enter__.return_value
                process.poll.return_value = None
                def communicate(**kwargs):
                    if 'timeout' in kwargs:
                        if cancelled:
                            stopped.set()
                        raise subprocess.TimeoutExpired('window capture',0.05)
                    return b'',None
                process.communicate.side_effect = communicate
                with self.assertRaisesRegex(Halted if cancelled else ValueError,'Stopped' if cancelled else 'timed out'):
                    capture_window_image(12,(0,0,2,1),stopped)
                process.kill.assert_called_once_with()
                self.assertEqual(process.communicate.call_count,2)
        stopped.set()
        with patch('desktop_agent.tools.subprocess.Popen') as spawn:
            with self.assertRaises(Halted):
                capture_window_image(12,(0,0,2,1),stopped)
            spawn.assert_not_called()

    def test_window_renderer_releases_gdi_resources_when_window_refuses_capture(self):
        from unittest.mock import patch, call
        from desktop_agent.tools import render_window_image
        with patch('desktop_agent.tools._capture_user32') as native, \
             patch('desktop_agent.tools._capture_gdi32') as graphics:
            native.GetWindowDC.return_value = 10
            graphics.CreateCompatibleDC.return_value = 20
            graphics.CreateCompatibleBitmap.return_value = 30
            graphics.SelectObject.return_value = 40
            native.PrintWindow.return_value = False
            with self.assertRaises(OSError):
                render_window_image(12,(0,0,400,300))
            native.PrintWindow.assert_called_once_with(12,20,2)
            self.assertEqual(graphics.SelectObject.call_args_list,[call(20,30),call(20,40)])
            graphics.DeleteObject.assert_called_once_with(30)
            graphics.DeleteDC.assert_called_once_with(20)
            native.ReleaseDC.assert_called_once_with(12,10)

    def test_physical_pixel_context_restores_thread_dpi_even_on_failure(self):
        import ctypes
        from unittest.mock import patch
        from desktop_agent.tools import physical_pixels
        previous = ctypes.c_void_p(-1).value
        with patch('desktop_agent.tools._dpi_user32.SetThreadDpiAwarenessContext',return_value=previous) as setter:
            with self.assertRaisesRegex(RuntimeError,'test failure'):
                with physical_pixels():
                    self.assertEqual(setter.call_args.args[0].value,ctypes.c_void_p(-4).value)
                    raise RuntimeError('test failure')
            self.assertEqual(setter.call_count,2)
            self.assertEqual(setter.call_args.args[0].value,previous)

    def test_hwnd_capture_rejects_wrong_size_and_never_falls_back_to_desktop(self):
        import threading
        from unittest.mock import Mock, patch
        from PIL import Image
        from desktop_agent.tools import Tools
        from game_agent.core import Region
        tools = Tools('.',threading.Event(),Mock(),Mock())
        tools.window = {'handle':12,'pid':34}
        tools.capture_region = Mock(return_value=Region(100,100,400,600))
        with patch('desktop_agent.tools.windows.window_rect',return_value=(100,100,500,700)), \
               patch('desktop_agent.tools.capture_window_image',return_value=Image.new('RGB',(320,480))) as grab, \
             patch('desktop_agent.tools.windows.capture') as desktop:
            with self.assertRaisesRegex(ValueError,'pixel size differs'):
                tools.capture_window()
            grab.side_effect = OSError('Unsupported')
            with self.assertRaisesRegex(ValueError,'unsupported'):
                tools.capture_window()
            desktop.assert_not_called()

    def test_selected_window_capture_reads_hwnd_surface_not_neighboring_desktop(self):
        import threading
        from unittest.mock import Mock, patch
        from PIL import Image
        from desktop_agent.tools import Tools
        target_image = Image.new('RGB',(392,642),'white')
        wrong_image = Image.new('RGB',(392,642),'green')
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        tools.window = {'handle':12,'pid':34}
        def process_id(handle,pointer):
            pointer._obj.value = 34
        with patch('desktop_agent.tools.windows.user32') as native, \
             patch('desktop_agent.tools.windows.virtual_screen',return_value=(-1920,0,3840,1200)), \
             patch('desktop_agent.tools.windows.window_rect',return_value=(1136,218,1528,860)), \
             patch('desktop_agent.tools.windows.window_title',return_value='Test window'), \
             patch('desktop_agent.tools.capture_window_image',return_value=target_image) as grab, \
             patch('desktop_agent.tools.windows.capture',return_value=wrong_image) as screen_capture:
            native.IsWindow.return_value = True
            native.IsIconic.return_value = False
            native.GetWindowThreadProcessId.side_effect = process_id
            result = tools.execute(dict(message='Capture selected window',tool='desktop_capture',arguments={},risk='routine'))
            grab.assert_called_once_with(12,(1136,218,1528,860),tools.stopped)
            screen_capture.assert_not_called()
            self.assertEqual(result.image.getpixel((196,321)),(255,255,255))
            native.SetForegroundWindow.assert_not_called()
            native.ShowWindow.assert_not_called()

    def test_input_target_ignores_unrelated_overlay_but_blocks_covered_click(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.tools import InputTarget, visible_window_region
        from game_agent import windows
        from game_agent.core import Halted
        def process_id(handle,pointer):
            pointer._obj.value = 34
        with patch('desktop_agent.tools.windows.user32') as native, \
             patch('desktop_agent.tools.windows.window_rect',return_value=(-10,-5,810,605)), \
             patch('desktop_agent.tools.windows.virtual_screen',return_value=(0,0,1920,1080)), \
             patch('desktop_agent.tools.windows.window_title',return_value='Test'), \
             patch('desktop_agent.tools.windows.at_point',return_value=99) as at_point:
            native.IsWindow.return_value = True
            native.IsIconic.return_value = False
            native.GetForegroundWindow.return_value = 12
            native.GetWindowThreadProcessId.side_effect = process_id
            region = visible_window_region(12)
            self.assertEqual(region.bbox,(0,0,810,605))
            target = InputTarget({'handle':12,'pid':34},region)
            self.assertTrue(target.valid())
            at_point.assert_not_called()
            backend = windows.DesktopInput(region,target,threading.Event())
            with self.assertRaises(Halted):
                backend._check((100,100))
            at_point.return_value = 12
            backend._check((100,100))
            native.GetForegroundWindow.return_value = 99
            self.assertFalse(target.valid())

    def test_message_mode_uses_same_uncropped_coordinates_as_capture(self):
        import threading
        from unittest.mock import patch
        from desktop_agent.input_modes import message_input
        def process_id(handle,pointer):
            pointer._obj.value = 34
        with patch('desktop_agent.input_modes.windows.user32') as native, \
             patch('desktop_agent.tools.windows.window_rect',return_value=(0,0,800,600)), \
             patch('desktop_agent.tools.windows.virtual_screen',return_value=(0,0,1920,1080)):
            native.IsWindow.return_value = True
            native.IsIconic.return_value = False
            native.GetWindowThreadProcessId.side_effect = process_id
            native.ChildWindowFromPointEx.return_value = 12
            native.ScreenToClient.return_value = True
            native.SendMessageTimeoutW.return_value = 1
            for horizontal,vertical,expected in ((0,0,(0,0)),(1000,1000,(799,599))):
                message_input({'handle':12,'pid':34},dict(kind='click',button='left',x=horizontal,y=vertical,hold_ms=0),threading.Event())
                location = native.SendMessageTimeoutW.call_args.args[3]
                self.assertEqual((location&0xffff,(location>>16)&0xffff),expected)

    def test_model_image_resize_keeps_both_screen_edges(self):
        import base64
        from io import BytesIO
        from PIL import Image, ImageDraw
        from game_agent.vision import encode_image
        image = Image.new('RGB',(1920,1080),'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((0,0,31,1079),fill='red')
        draw.rectangle((1888,0,1919,1079),fill='blue')
        encoded,metadata = encode_image(image,1280,90,'rgb')
        with Image.open(BytesIO(base64.b64decode(encoded.split(',',1)[1]))) as resized:
            self.assertEqual(resized.size,(1280,720))
            red,green,blue = resized.getpixel((0,360))
            self.assertGreater(red,max(green,blue)+150)
            red,green,blue = resized.getpixel((1279,360))
            self.assertGreater(blue,max(red,green)+150)

    def test_selected_capture_does_not_require_unobscured_game_window_or_crop_edges(self):
        from unittest.mock import Mock, patch
        import threading
        from PIL import Image
        from desktop_agent.tools import Tools
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        tools.window = {'handle':12,'pid':34}
        tools.allow_input = False
        def process_id(handle,pointer):
            pointer._obj.value = 34
        with patch('desktop_agent.tools.windows.user32') as native, \
             patch('desktop_agent.tools.windows.virtual_screen',return_value=(-1920,0,3840,1080)), \
             patch('desktop_agent.tools.windows.window_rect',return_value=(-1600,100,-800,700)), \
             patch('desktop_agent.tools.windows.Target',side_effect=AssertionError('Game checks must not run')), \
             patch('desktop_agent.tools.capture_window_image',return_value=Image.new('RGB',(800,600))) as grab, \
             patch('desktop_agent.tools.windows.capture',return_value=Image.new('RGB',(800,600))) as capture:
            native.IsWindow.return_value = True
            native.IsIconic.return_value = False
            native.GetWindowThreadProcessId.side_effect = process_id
            result = tools.execute(dict(message='Capture',tool='desktop_capture',arguments={},risk='routine'))
            grab.assert_called_once_with(12,(-1600,100,-800,700),tools.stopped)
            capture.assert_not_called()
            self.assertEqual(json.loads(result.text)['bounds'],[-1600,100,-800,700])
            self.assertEqual(result.image.size,(800,600))
            self.assertEqual(json.loads(result.text)['capture_method'],'hwnd')
            native.SetForegroundWindow.assert_not_called()
            native.ShowWindow.assert_not_called()
            tools.window = None
            tools.execute(dict(message='Capture',tool='desktop_capture',arguments={},risk='routine'))
            self.assertEqual(capture.call_args.args[0].bbox,(-1920,0,1920,1080))

    def test_message_edits_deletion_and_replay_isolate_sessions_and_files(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier, other = store.create('Original'),store.create('Other')
            first = store.append(identifier,'user','First request')
            capture = store.artifact_directory(identifier)/'capture.png'
            Image.new('RGB',(20,20),'white').save(capture)
            store.append(identifier,'tool','Captured',image=str(capture),tool='desktop_capture')
            answer = store.append(identifier,'assistant',json.dumps(dict(message='First answer',tool='finish',arguments={},risk='routine')),metrics={'seconds':2})
            attachment = store.artifact_directory(identifier)/'attached.txt'
            attachment.write_text('attached',encoding='utf-8')
            second = store.append(identifier,'user','Second request',attachments=[{'path':str(attachment),'name':'attached.txt'}])
            reply = store.append(identifier,'assistant',json.dumps(dict(message='Second answer',tool='finish',arguments={},risk='routine')))
            store.edit_message(identifier,second,'Revised request')
            store.edit_message(identifier,answer,'Edited answer')
            self.assertNotIn('metrics',store.message(identifier,answer)[0]['metadata'])
            self.assertEqual(json.loads(store.message(identifier,answer)[0]['content'])['tool'],'finish')
            with self.assertRaises(ValueError):
                store.edit_message(other,second,'Cross-session change')
            store.append(identifier,'user','Later request')
            replayed,prompt,files = store.prepare_replay(identifier,reply)
            self.assertEqual(replayed,identifier)
            self.assertEqual(len(store.sessions()),2)
            self.assertEqual(prompt,'Revised request')
            self.assertEqual(len(store.events(identifier)),3)
            self.assertEqual(Path(files[0]),attachment.resolve())
            self.assertNotIn('Later request',store.history_text(identifier))
            second = store.append(identifier,'user',prompt,attachments=[{'path':files[0]}])
            reply = store.append(identifier,'assistant',json.dumps(dict(message='Retried',tool='finish')))
            store.delete_message(identifier,first)
            self.assertEqual([event['id'] for event in store.events(identifier)],[second,reply])
            self.assertNotIn('First request',store.history_text(identifier))
            store.delete_message(identifier,reply)
            self.assertEqual([event['id'] for event in store.events(identifier)],[second])
            store.delete_session(identifier)
            self.assertFalse(capture.exists())
            self.assertFalse(Path(files[0]).is_file())
            self.assertEqual(store.events(identifier),[])
            self.assertIn(other,[row['id'] for row in store.sessions()])

    def test_replay_validation_fails_before_deleting_conversation(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            request = store.append(identifier,'user','Retry',attachments=[{'path':str(Path(folder)/'missing.txt')}])
            reply = store.append(identifier,'assistant',json.dumps(dict(message='Old reply',tool='finish')))
            with self.assertRaises(ValueError):
                store.prepare_replay(identifier,request)
            self.assertEqual([event['id'] for event in store.events(identifier)],[request,reply])

    def test_message_mutations_reject_tool_calls_and_failed_file_delete_keeps_session(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            event_id = store.append(identifier,'assistant',json.dumps(dict(message='Click',tool='desktop_click')))
            store.artifact_directory(identifier)
            with self.assertRaises(ValueError):
                store.edit_message(identifier,event_id,'Fake success')
            with self.assertRaises(ValueError):
                store.delete_message(identifier,event_id)
            with patch('desktop_agent.store.shutil.rmtree',side_effect=PermissionError('File is open')):
                with self.assertRaises(PermissionError):
                    store.delete_session(identifier)
            self.assertIn(identifier,[row['id'] for row in store.sessions()])
            with self.assertRaises(ValueError):
                store.delete_session('../elsewhere')

    def test_visible_desktop_capture_needs_no_target_and_never_changes_focus(self):
        import threading
        from unittest.mock import Mock, patch
        from PIL import Image
        from desktop_agent.tools import Tools
        from desktop_agent.protocol import CAPTURE_TOOLS
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        tools.allow_input = False
        tools.focus_window = Mock(side_effect=AssertionError('Must not focus'))
        action = dict(message='View current screen',tool='desktop_screen_capture',arguments={},risk='routine')
        screenshot = Image.new('RGB',(3840,1080),'white')
        with patch('desktop_agent.tools.windows.virtual_screen',return_value=(-1920,0,3840,1080)), \
             patch('desktop_agent.tools.windows.capture',return_value=screenshot) as capture, \
             patch('desktop_agent.tools.windows.user32') as native:
            result = tools.execute(action)
            self.assertIs(result.image,screenshot)
            self.assertEqual(capture.call_args.args[0].bbox,(-1920,0,1920,1080))
            native.SetForegroundWindow.assert_not_called()
            native.ShowWindow.assert_not_called()
            tools.focus_window.assert_not_called()
            tools.allow_screen = False
            with self.assertRaisesRegex(ValueError,'Screen capture disabled'):
                tools.execute(action)
        self.assertIn('desktop_screen_capture',CAPTURE_TOOLS)

    def test_timed_holds_and_drag_release_on_cancellation(self):
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        from desktop_agent.input_modes import timed_drag
        from game_agent.core import Halted, Region
        stopped = Mock(wait=Mock(return_value=True))
        tools = Tools('.',stopped,Mock(),Mock())
        for key in ('W','a','1','F5','F12','Shift+ArrowRight'):
            validate_action(dict(message='Hold',tool='desktop_hold',arguments=dict(kind='key',mode='device',key=key,x=0,y=0,button='left',hold_ms=1000),risk='routine'))
        tools.focus_window = Mock(return_value=(Region(0,0,800,600),Mock(title='Test')))
        backend = Mock()
        with patch('desktop_agent.tools.windows.DesktopInput',return_value=backend):
            with self.assertRaises(Halted):
                tools.desktop('desktop_hold',dict(kind='key',mode='general',key='Control+a',hold_ms=1000))
            events = [call.args[0].data.keyboard for call in backend._send.call_args_list]
            self.assertEqual([(event.wVk,event.dwFlags) for event in events],[(17,0),(65,0),(65,2),(17,2)])
        backend.reset_mock()
        with patch('desktop_agent.input_modes.move_pointer'):
            with self.assertRaises(Halted):
                timed_drag(backend,(10,10),(100,100),'left',1000,stopped)
        backend.down.assert_called_once_with('click','left')
        backend.up.assert_called_once_with('click','left')
        stopped.wait.return_value = False
        backend.reset_mock()
        with patch('desktop_agent.input_modes.move_pointer') as move:
            timed_drag(backend,(10,10),(100,100),'left',100,stopped)
            self.assertEqual(move.call_args.args[1],(100,100))
        backend.up.assert_called_once_with('click','left')

    def test_attachments_are_copied_scoped_and_read_without_execution(self):
        from PIL import Image
        from desktop_agent.attachments import save_attachments, registered_path, read_attachment, view_attachment
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source'
            source.mkdir()
            document = source/'notes with spaces.txt'
            document.write_text('Hello \ud55c\uae00',encoding='utf-8')
            picture = source/'picture.png'
            Image.new('RGB',(80,60),'red').save(picture)
            store = Store(Path(folder)/'data')
            identifier = store.create()
            files = save_attachments([document,picture],store.artifact_directory(identifier))
            store.append(identifier,'user','Inspect these',attachments=files)
            document.unlink()
            path = registered_path(store.events(identifier),files[0]['path'])
            self.assertEqual(json.loads(read_attachment(path,0,5).text)['text'],'Hello')
            self.assertEqual(view_attachment(Path(files[1]['path'])).image.size,(80,60))
            with self.assertRaises(ValueError):
                registered_path(store.events(store.create()),files[0]['path'])
            with self.assertRaises(ValueError):
                registered_path(store.events(identifier),str(picture))
            archive = store.archive_history(identifier)
            self.assertIn(files[0]['path'],json.loads(archive.read_text(encoding='utf-8'))['metadata']['attachments'][0]['path'])

    def test_paths_failures_and_previous_calls_reach_model_without_image_reload(self):
        from desktop_agent.agent import pack_messages
        events = [dict(role='user',content='Capture'),
                  dict(role='assistant',content='{"tool":"desktop_capture"}'),
                  dict(role='tool',content='Captured',metadata={'image':'C:/session/capture.png','status':'delivered'}),
                  dict(role='tool',content='Selector missing',metadata={'tool':'browser_click','status':'error'}),
                  dict(role='user',content='What happened?')]
        messages, dropped, used = pack_messages(events,'system',len,budget=4000)
        serialized = json.dumps(messages)
        self.assertIn('C:/session/capture.png',serialized)
        self.assertIn('Selector missing',serialized)
        self.assertIn('browser_click',serialized)
        self.assertNotIn('image_url',serialized)
        self.assertEqual(dropped,0)

    def test_agent_reads_attached_file_and_retains_registered_path(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Settings
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'note.txt'
            source.write_text('CEDAR-42',encoding='utf-8')
            store = Store(Path(folder)/'store')
            identifier = store.create()
            model = Mock(settings=Settings(),count=lambda text:len(text)//4)
            calls = []
            def generate(messages,image,*args):
                calls.append(image)
                if len(calls) == 1:
                    path = store.events(identifier)[0]['metadata']['attachments'][0]['path']
                    return dict(message='Read attachment',tool='file_read',arguments={'path':path,'offset':0,'limit':100},risk='routine'),{}
                self.assertIn('CEDAR-42',json.dumps(messages))
                return dict(message='CEDAR-42',tool='finish',arguments={},risk='routine'),{}
            model.generate.side_effect = generate
            tools = Mock(allow_screen=True,allow_input=True,allow_browser=True,window=None,mode='routine')
            Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Read attachment',[str(source)])
            self.assertEqual(calls,[None,None])
            tools.execute.assert_not_called()

    def test_background_jobs_validate_overlap_cancel_and_report_failures(self):
        import threading
        from unittest.mock import Mock, patch
        from desktop_agent.jobs import ToolRunner
        from desktop_agent.tools import ToolResult
        entered, release = threading.Event(),threading.Event()
        def execute(worker,action):
            if action['tool'] == 'desktop_capture':
                entered.set()
                self.assertTrue(release.wait(3))
                return ToolResult('Captured')
            if action['tool'] == 'desktop_key_queue':
                worker.stopped.wait(3)
                worker.check()
            release.set()
            return ToolResult('Clicked')
        with tempfile.TemporaryDirectory() as folder:
            runner = ToolRunner(folder,threading.Event(),Mock(return_value=True),Mock())
            try:
                nested = dict(message='Capture',tool='desktop_capture',arguments={},risk='routine')
                action = dict(message='Start',tool='job_start',arguments={'action':nested},risk='routine')
                with self.assertRaises(ValueError):
                    validate_action(dict(action,arguments={'action':action}))
                with patch('desktop_agent.jobs.Tools.execute',execute):
                    identifier = json.loads(runner.execute(action).text)['job_id']
                    self.assertTrue(entered.wait(2))
                    runner.execute(dict(message='Click',tool='desktop_key',arguments={'key':'Tab'},risk='routine'))
                    result = runner.execute(dict(message='Collect',tool='job_result',arguments={'job_id':identifier},risk='routine'))
                    self.assertEqual(result.tool,'desktop_capture')
                    self.assertFalse(runner.background)
                    nested = dict(message='Wait',tool='desktop_key_queue',arguments={'steps':[{'key':'Tab','delay_ms':3000}]},risk='routine')
                    identifier = json.loads(runner.execute(dict(action,arguments={'action':nested})).text)['job_id']
                    runner.execute(dict(message='Cancel',tool='job_cancel',arguments={'job_id':identifier},risk='routine'))
                    result = runner.execute(dict(message='Collect',tool='job_result',arguments={'job_id':identifier},risk='routine'))
                    self.assertTrue(result.interrupted)
            finally:
                release.set()
                runner.close()

    def test_device_scancodes_release_on_cancel_and_message_mode_does_not_focus(self):
        from unittest.mock import Mock, patch
        from desktop_agent.input_modes import scan_key
        from desktop_agent.tools import Tools
        from game_agent.core import Halted
        backend = Mock()
        stop = Mock(wait=Mock(return_value=True))
        with patch('desktop_agent.input_modes.windows.user32.MapVirtualKeyW',side_effect=lambda code,mode:code+1):
            with self.assertRaises(Halted):
                scan_key(backend,'Control+ArrowRight',200,stop)
        events = [call.args[0].data.keyboard for call in backend._send.call_args_list]
        self.assertEqual([(event.wVk,event.wScan,event.dwFlags) for event in events],[(0,18,8),(0,40,9),(0,40,11),(0,18,10)])
        tools = Tools('.',Mock(),Mock(),Mock())
        tools.window = {'handle':12,'pid':34}
        tools.focus_window = Mock()
        with patch('desktop_agent.input_modes.message_input',return_value='Sent') as sender:
            result = tools.desktop('desktop_input',{'mode':'message','kind':'text','text':'Hello'})
            self.assertEqual(result.text,'Sent')
            sender.assert_called_once()
            tools.focus_window.assert_not_called()

    def test_video_artifact_and_latest_sample_image_are_preserved(self):
        import threading
        from unittest.mock import Mock
        from PIL import Image
        from desktop_agent.agent import Agent, Settings
        from desktop_agent.tools import ToolResult
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            model = Mock(settings=Settings(),count=lambda text:len(text)//4)
            record = dict(message='Record motion',tool='desktop_record',arguments={'seconds':2,'fps':4},risk='routine')
            finish = dict(message='Observed',tool='finish',arguments={},risk='routine')
            model.generate.side_effect = [(record,{}),(finish,{}),(finish,{})]
            tools = Mock(allow_screen=True,allow_input=False,allow_browser=False,window={'handle':1,'pid':2},mode='routine')
            samples = Image.new('RGB',(1280,768),'white')
            video = str(store.artifact_directory(identifier)/'sample.mp4')
            tools.execute.return_value = ToolResult('Recorded',samples,video)
            agent = Agent(store,model,tools,threading.Event(),Mock())
            agent.run(identifier,'Record a video')
            agent.run(identifier,'Continue')
            self.assertEqual([call.args[1] for call in model.generate.call_args_list[:2]],[None,samples])
            self.assertEqual(model.generate.call_args_list[2].args[1].tobytes(),samples.tobytes())
            event = next(event for event in store.events(identifier) if event['role'] == 'tool')
            self.assertEqual(event['metadata']['video'],video)
            self.assertTrue(Path(event['metadata']['image']).is_file())
            interrupted = store.create()
            model.generate.side_effect = [(record,{})]
            tools.execute.return_value = ToolResult('Partial recording',samples,video,'Recording stopped')
            agent.run(interrupted,'Record again')
            event = next(event for event in store.events(interrupted) if event['role'] == 'tool')
            self.assertEqual(event['metadata']['status'],'stopped')
            self.assertEqual(event['metadata']['video'],video)

    def test_video_encoder_saves_playable_clip_and_partial_clip_on_stop(self):
        import cv2
        import threading
        from unittest.mock import Mock, patch
        from PIL import Image
        from desktop_agent.tools import Tools
        clock = [0.0]
        stopped = Mock(is_set=Mock(return_value=False))
        def wait(delay):
            clock[0] += delay
            return False
        stopped.wait.side_effect = wait
        with tempfile.TemporaryDirectory() as folder:
            tools = Tools(folder,stopped,Mock(),Mock())
            frames = [Image.new('RGB',(320,180),color) for color in ('red','green','blue','white')]
            with patch('desktop_agent.tools.time.monotonic',side_effect=lambda:clock[0]):
                result = tools.record_video(Mock(side_effect=frames),{'seconds':1,'fps':4},'Test clip')
            self.assertTrue(Path(result.video).is_file())
            self.assertEqual(result.image.size,(1280,768))
            capture = cv2.VideoCapture(result.video)
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),4)
                self.assertEqual(capture.get(cv2.CAP_PROP_FPS),4)
                decoded = []
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    decoded.append(frame)
                self.assertEqual(len(decoded),4)
                self.assertFalse((decoded[0] == decoded[-1]).all())
            finally:
                capture.release()
            real_stop = threading.Event()
            tools.stopped = real_stop
            tools.notify = lambda kind,value:real_stop.set()
            result = tools.record_video(lambda:frames[0],{'seconds':5,'fps':5},'Stopped clip')
            self.assertTrue(result.interrupted)
            capture = cv2.VideoCapture(result.video)
            try:
                self.assertTrue(capture.read()[0])
            finally:
                capture.release()

    def test_performance_footer_uses_server_timings_not_cached_prompt_wall_time(self):
        from desktop_agent.app import performance_text
        text = performance_text({'timings':{'prompt_per_second':400.25,'predicted_per_second':8.5},
                                 'seconds':9.3,'task_seconds':22.8})
        self.assertIn('400.2 tok/s',text)
        self.assertIn('8.5 tok/s',text)
        self.assertIn('9.3',text)
        self.assertIn('22.8',text)
        text = performance_text({'timings':{'prompt_n':12,'prompt_ms':30,'predicted_n':20,'predicted_ms':2000}})
        self.assertIn('400.0 tok/s',text)
        self.assertIn('10.0 tok/s',text)
        legacy = performance_text({'usage':{'prompt_tokens':8192,'completion_tokens':80},'seconds':10})
        self.assertNotIn('tok/s',legacy)
        self.assertEqual(performance_text({}),'')
        self.assertNotIn('tok/s',performance_text({'timings':{'prompt_per_second':float('nan'),'predicted_ms':0}}))

    def test_reply_context_prefers_complete_server_usage_without_double_counting(self):
        from unittest.mock import Mock
        from desktop_agent.agent import response_context
        from desktop_agent.app import performance_text
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        count = Mock(side_effect=AssertionError('Reported usage needs no tokenization'))
        metrics = {'usage':{'prompt_tokens':4000,'completion_tokens':100,'total_tokens':9999,
                            'cached_tokens':3000,'thinking_tokens':50}}
        context = response_context(metrics,[],action,count,4500,8192,True)
        self.assertEqual(context['context_tokens'],4100)
        self.assertEqual(context['context_input_tokens'],4000)
        self.assertEqual(context['context_output_tokens'],100)
        self.assertEqual(context['context_source'],'server_usage')
        self.assertIn('4,100 / 8,192 tok',performance_text(context))
        self.assertIn('\uc11c\ubc84 \ubcf4\uace0',performance_text(context))
        count.assert_not_called()

    def test_reply_context_fallback_is_labeled_estimated(self):
        from unittest.mock import Mock
        from desktop_agent.agent import response_context
        from desktop_agent.app import performance_text
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        for usage in ({},{'prompt_tokens':True,'completion_tokens':float('nan')},
                      {'prompt_tokens':-1,'completion_tokens':1.5}):
            count = Mock(return_value=1100)
            result = response_context({'usage':usage},[],action,count,1000,8192,True)
            self.assertEqual(result['context_tokens'],2636)
            self.assertEqual(result['context_source'],'estimate')
            self.assertIn('\ucd94\uc815',performance_text(result))
            count.assert_called_once()
        count = Mock(return_value=1100)
        result = response_context({'usage':{'prompt_tokens':2000}},[],action,count,1000,8192,False)
        self.assertEqual(result['context_tokens'],2100)
        count.side_effect = RuntimeError('Tokenizer unavailable')
        result = response_context({},[],action,count,1000,8192,False)
        self.assertGreater(result['context_tokens'],1000)
        self.assertEqual(result['context_source'],'estimate')

    def test_stream_preserves_server_timings_and_usage_from_final_chunks(self):
        import itertools
        import threading
        from unittest.mock import Mock, MagicMock, patch
        from desktop_agent.agent import Model, Settings
        action = dict(message='Done',tool='finish',arguments={},risk='routine')
        timings = dict(prompt_n=8,prompt_ms=20,prompt_per_second=400,predicted_per_second=8.5)
        events = [dict(choices=[dict(delta={'content':json.dumps(action)},finish_reason=None)]),
                  dict(choices=[dict(delta={},finish_reason='stop')],timings=timings),
                  dict(choices=[],usage={'prompt_tokens':808,'completion_tokens':30})]
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = ['data: '+json.dumps(event) for event in events]+['data: [DONE]']
        client = MagicMock()
        client.__enter__.return_value = client
        client.post.return_value = response
        model = Model(Settings())
        model.endpoint = 'http://127.0.0.1:1234'
        with patch('desktop_agent.agent.session',return_value=client), \
             patch('desktop_agent.agent.time.monotonic',side_effect=itertools.count(10,0.2)):
            result, metrics = model.generate([],None,threading.Event(),Mock())
        self.assertEqual(result,action)
        self.assertTrue(client.post.call_args.kwargs['json']['timings'])
        self.assertEqual(metrics['timings'],timings)
        self.assertEqual(metrics['usage']['prompt_tokens'],808)
        self.assertGreater(metrics['seconds'],metrics['first_token_seconds'])
        self.assertEqual(client.post.call_args.kwargs['json']['max_tokens'],-1)
        for guarded,elapsed in ((False,181),(False,86400),(True,600.001),(True,86400)):
            with self.subTest(guarded=guarded,elapsed=elapsed):
                model.server.residency_guard=Mock(reason='') if guarded else None
                with patch('desktop_agent.agent.session',return_value=client), \
                     patch('desktop_agent.agent.time.monotonic',side_effect=itertools.chain([0],itertools.repeat(elapsed))):
                    result,_=model.generate([],None,threading.Event(),Mock())
                    self.assertEqual(result,action)
                self.assertEqual(client.post.call_args.kwargs['timeout'],(3,None))
                if guarded:
                    model.server.residency_guard.begin_request.assert_called_once()
                    model.server.residency_guard.end_request.assert_called_once()

    def test_chat_groups_progress_and_audits_but_preserves_replies_errors_and_partial(self):
        from desktop_agent.app import conversation_items
        events = [dict(id=1,role='user',content='Read page'),
                  dict(id=2,role='assistant',content=json.dumps(dict(message='Reading',tool='desktop_capture'))),
                  dict(id=3,role='system',content='Approval JSON',metadata={'status':'approval'}),
                  dict(id=4,role='tool',content='Captured',metadata={'tool':'desktop_capture'}),
                  dict(id=5,role='assistant',content=json.dumps(dict(message='**Answer**',tool='finish'))),
                  dict(id=6,role='system',content='Stopped',metadata={'status':'stopped'}),
                  dict(id=7,role='assistant',content='Partial',metadata={'partial':True})]
        items = conversation_items(events)
        self.assertEqual([item['role'] for item in items],['user','activity','assistant','system','assistant'])
        self.assertEqual(len(items[1]['events']),3)
        self.assertEqual(items[2]['content'],'**Answer**')
        self.assertEqual(items[-1]['content'],'Partial')
        self.assertEqual(len(events),7)
        self.assertEqual(conversation_items(events[:4])[-1]['role'],'activity')

    def test_native_key_queue_waits_in_order_and_cancels_without_later_input(self):
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        from game_agent.core import Region, Halted
        stopped = Mock(is_set=Mock(return_value=False),wait=Mock(return_value=False))
        tools = Tools('.',stopped,Mock(),Mock())
        tools.focus_window = Mock(return_value=(Region(0,0,800,600),Mock(title='Editor')))
        backend = Mock()
        timeline = []
        stopped.wait.side_effect = lambda delay:timeline.append(('wait',delay)) or False
        backend._send.side_effect = lambda event:timeline.append(('key',event.data.keyboard.wVk,event.data.keyboard.dwFlags))
        steps = [{'key':'Tab','delay_ms':150},{'key':'Control+a','delay_ms':300}]
        with patch('desktop_agent.tools.windows.DesktopInput',return_value=backend):
            tools.desktop('desktop_key_queue',{'steps':steps})
            self.assertEqual(timeline,[('wait',0.15),('key',9,0),('key',9,2),('wait',0.3),
                                       ('key',17,0),('key',65,0),('key',65,2),('key',17,2)])
            tools.focus_window.assert_called_once()
            timeline.clear()
            stopped.wait.side_effect = [False,True]
            backend.reset_mock()
            with self.assertRaises(Halted):
                tools.desktop('desktop_key_queue',{'steps':steps})
            self.assertEqual(timeline,[('key',9,0),('key',9,2)])
            backend.release_all.assert_called_once()

    def test_keyboard_target_allows_geometry_change_but_never_focus_loss(self):
        from unittest.mock import patch
        from desktop_agent.tools import InputTarget
        from game_agent.core import Region
        def process_id(handle,pointer):
            pointer._obj.value = 34
        with patch('desktop_agent.tools.windows.user32') as native, \
             patch('desktop_agent.tools.windows.window_rect',return_value=(0,0,800,600)) as rectangle, \
             patch('desktop_agent.tools.windows.window_title',return_value='Test'):
            native.GetWindowThreadProcessId.side_effect = process_id
            native.IsWindow.return_value = True
            native.IsIconic.return_value = False
            native.GetForegroundWindow.return_value = 12
            target = InputTarget({'handle':12,'pid':34},Region(0,0,800,600))
            rectangle.return_value = (50,50,850,650)
            self.assertFalse(target.valid())
            self.assertIn('moved or resized',target.failure_reason)
            target.require_geometry = False
            self.assertTrue(target.valid())
            native.GetForegroundWindow.return_value = 99
            self.assertFalse(target.valid())
            self.assertIn('lost focus',target.failure_reason)

    def test_key_queue_reports_partial_delivery_without_replaying(self):
        import threading
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        from game_agent.core import Region, Halted
        from game_agent.windows import TargetUnavailable
        tools = Tools('.',threading.Event(),Mock(),Mock())
        target = Mock(title='Editor',failure_reason='Selected window lost focus')
        tools.focus_window = Mock(return_value=(Region(0,0,800,600),target))
        tools.press_desktop_key = Mock(side_effect=[None,TargetUnavailable('Game window temporarily unavailable')])
        with patch('desktop_agent.tools.windows.DesktopInput') as backend:
            with self.assertRaisesRegex(Halted,'lost focus; 1/2 key steps completed'):
                tools.desktop('desktop_key_queue',{'steps':[{'key':'Enter','delay_ms':0}]*2})
            self.assertFalse(target.require_geometry)
            self.assertEqual(tools.press_desktop_key.call_count,2)
            backend.return_value.release_all.assert_called_once()

    def test_capture_focus_preserves_maximized_window_and_restores_only_minimized(self):
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        tools = Tools('.',Mock(wait=Mock(return_value=False),is_set=Mock(return_value=False)),Mock(),Mock())
        tools.window = {'handle':12,'pid':34}
        def process_id(handle, pointer):
            pointer._obj.value = 34
        with patch('desktop_agent.tools.windows.user32') as native, \
             patch('desktop_agent.tools.windows.window_rect',return_value=(0,0,1920,1080)), \
               patch('desktop_agent.tools.windows.virtual_screen',return_value=(0,0,1920,1080)), \
               patch('desktop_agent.tools.InputTarget',return_value=Mock(handle=12,pid=34,valid=Mock(return_value=True))):
            native.IsWindow.return_value = True
            native.GetForegroundWindow.return_value = 12
            native.GetWindowThreadProcessId.side_effect = process_id
            native.IsIconic.return_value = False
            region, target = tools.focus_window()
            self.assertEqual(region.bbox,(0,0,1920,1080))
            native.ShowWindow.assert_not_called()
            native.SetForegroundWindow.assert_called_once_with(12)
            native.IsIconic.return_value = True
            tools.focus_window()
            native.ShowWindow.assert_called_once_with(12,9)

    def test_focus_fallback_detaches_threads_and_never_restores_maximized_window(self):
        from unittest.mock import Mock, patch, call
        from desktop_agent.tools import Tools
        tools = Tools('.',Mock(wait=Mock(return_value=False),is_set=Mock(return_value=False)),Mock(),Mock())
        tools.window = {'handle':12,'pid':34}
        def thread_id(handle,pointer):
            if pointer is not None:
                pointer._obj.value = 34
            return 222
        with patch('desktop_agent.tools.windows.user32') as native, \
             patch('desktop_agent.tools.windows.kernel32') as kernel, \
             patch('desktop_agent.tools.windows.window_rect',return_value=(0,0,1920,1080)), \
               patch('desktop_agent.tools.windows.virtual_screen',return_value=(0,0,1920,1080)), \
               patch('desktop_agent.tools.InputTarget',return_value=Mock(handle=12,pid=34,valid=Mock(return_value=True))):
            native.IsWindow.return_value = True
            native.IsIconic.return_value = False
            native.GetForegroundWindow.return_value = 99
            native.GetWindowThreadProcessId.side_effect = thread_id
            native.AttachThreadInput.return_value = True
            kernel.GetCurrentThreadId.return_value = 111
            tools.focus_window()
            native.ShowWindow.assert_not_called()
            self.assertEqual(native.AttachThreadInput.call_args_list,[call(111,222,True),call(111,222,False)])
            native.BringWindowToTop.assert_called_once_with(12)

    def test_cancelled_task_does_not_execute_generated_tool(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Settings
        from game_agent.core import Halted
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            stopped = threading.Event()
            model = Mock(settings=Settings(),count=lambda text:len(text)//4,partial='Interrupted answer')
            def generate(*args):
                stopped.set()
                raise Halted('Stopped')
            model.generate.side_effect = generate
            tools = Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='manual')
            Agent(store,model,tools,stopped,Mock()).run(identifier,'Task')
            tools.execute.assert_not_called()
            self.assertTrue(any(event['metadata'].get('partial') for event in store.events(identifier)))

    def test_native_unicode_and_keys_release_without_a_capture(self):
        import threading
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        from game_agent.core import Region, Halted
        stopped = threading.Event()
        tools = Tools('.',stopped,Mock(),Mock())
        tools.focus_window = Mock(return_value=(Region(0,0,800,600),Mock(title='Editor')))
        backend = Mock()
        with patch('desktop_agent.tools.windows.DesktopInput',return_value=backend):
            tools.desktop('desktop_type',{'text':'A\ud55c'})
            events = [call.args[0].data.keyboard for call in backend._send.call_args_list]
            self.assertEqual([(event.wScan,event.dwFlags) for event in events],[(65,4),(65,6),(0xD55C,4),(0xD55C,6)])
            backend.reset_mock()
            tools.desktop('desktop_key',{'key':'Control+a'})
            events = [call.args[0].data.keyboard for call in backend._send.call_args_list]
            self.assertEqual([(event.wVk,event.dwFlags) for event in events],[(17,0),(65,0),(65,2),(17,2)])
            backend.reset_mock()
            backend._check.side_effect = [None,Halted('Stopped')]
            with self.assertRaises(Halted):
                tools.desktop('desktop_key',{'key':'Control+a'})
            events = [call.args[0].data.keyboard for call in backend._send.call_args_list]
            self.assertEqual([(event.wVk,event.dwFlags) for event in events],[(17,0),(17,2)])
            backend.release_all.assert_called_once()

    def test_capture_persists_until_replaced_and_rehydrates_per_session(self):
        import threading
        from unittest.mock import Mock
        from PIL import Image
        from desktop_agent.agent import Agent, Settings
        from desktop_agent.tools import ToolResult
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            model = Mock(settings=Settings(),count=lambda text:len(text)//4)
            def action(tool):
                return dict(message=tool,tool=tool,arguments={},risk='routine'),{}
            model.generate.side_effect = [action('browser_capture'),action('browser_read'),action('finish'),action('finish')]
            tools = Mock(allow_screen=True,allow_input=False,allow_browser=True,window=None,mode='routine')
            screenshot = Image.new('RGB',(1920,1080),'white')
            tools.execute.side_effect = [ToolResult('Capture',screenshot),ToolResult('Text only')]
            agent = Agent(store,model,tools,threading.Event(),Mock())
            agent.run(identifier,'Inspect page')
            agent.run(identifier,'Continue conversation')
            images = [call.args[1] for call in model.generate.call_args_list]
            self.assertEqual(images[:3],[None,screenshot,screenshot])
            self.assertEqual(images[3].tobytes(),screenshot.tobytes())
            self.assertEqual(agent.image_source['tool'],'browser_capture')
            model.generate.side_effect = [action('browser_capture'),action('browser_read'),action('finish')]
            replacement = Image.new('RGB',(300,200),'red')
            tools.execute.side_effect = [ToolResult('New capture',replacement),RuntimeError('Read failed')]
            agent.run(identifier,'Capture the browser page again')
            self.assertIs(model.generate.call_args_list[-2].args[1],replacement)
            self.assertIs(model.generate.call_args_list[-1].args[1],replacement)
            reopened = Agent(Store(folder),model,tools,threading.Event(),Mock())
            self.assertEqual(reopened.latest_capture(identifier).tobytes(),replacement.tobytes())
            other = store.create()
            self.assertIsNone(reopened.latest_capture(other))
            tools.allow_browser=False
            self.assertIsNone(reopened.latest_capture(identifier))
            tools.allow_browser=True
            saved = [event['metadata']['image'] for event in store.events(identifier) if event['metadata'].get('image')]
            self.assertEqual(len(saved),2)
            self.assertTrue(Path(saved[0]).is_file())
            with Image.open(saved[0]) as archived:
                self.assertEqual(archived.size,(1920,1080))
            Path(saved[-1]).unlink()
            self.assertIsNone(reopened.latest_capture(identifier))

    def test_reasoning_toggle_reloads_the_matching_server_settings(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Model, Settings
        model = Model(Settings(reasoning_enabled=True))
        model.server = Mock(loaded_settings=('old',))
        model.server.settings_for.return_value = ('new',)
        model.ensure(threading.Event())
        model.server.close.assert_called_once()
        self.assertEqual(model.server.start.call_args.args[-2:],(True,256))

    def test_context_drops_old_turns_without_deleting_saved_history(self):
        from desktop_agent.agent import pack_messages
        events = [dict(role='user',content='old'*100),dict(role='assistant',content='done'),
              dict(role='user',content='previous'),dict(role='tool',content='previous error'),
                  dict(role='user',content='current task'),dict(role='tool',content='result')]
        _,_,protected_size = pack_messages(events[-4:],'system',len,budget=5000)
        messages, dropped, used = pack_messages(events,'system',len,budget=protected_size+300)
        self.assertGreater(dropped,0)
        self.assertIn('current task',json.dumps(messages))
        self.assertNotIn('oldold',json.dumps(messages))
        self.assertEqual(len(events),6)
        self.assertIn('previous error',json.dumps(messages))
        with self.assertRaises(ValueError):
            pack_messages(events[-4:],'system',len,budget=100)

    def test_agent_runs_only_after_input_and_preserves_tool_results(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.agent import Agent, Settings
        from desktop_agent.tools import ToolResult
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            identifier = store.create()
            model = Mock(settings=Settings(),count=lambda text:len(text)//4)
            model.generate.side_effect = [
                (dict(message='Read',tool='browser_read',arguments={},risk='routine'),{}),
                (dict(message='Done',tool='finish',arguments={},risk='routine'),{})]
            tools = Mock(allow_screen=False,allow_input=False,allow_browser=True,window=None,mode='routine')
            tools.execute.return_value = ToolResult('Page content')
            agent = Agent(store,model,tools,threading.Event(),Mock())
            model.ensure.assert_not_called()
            agent.run(identifier,'Read the current page')
            self.assertEqual([event['role'] for event in store.events(identifier)],['user','assistant','tool','assistant'])
            metrics = store.events(identifier)[-1]['metadata']['metrics']
            self.assertEqual(metrics['request_count'],2)
            self.assertGreaterEqual(metrics['task_seconds'],0)
            self.assertEqual(metrics['context_tokens'],metrics['context_input_tokens']+metrics['context_output_tokens'])
            self.assertEqual(metrics['context_limit'],8192)
            self.assertEqual(metrics['context_source'],'estimate')
            self.assertEqual(Store(folder).events(identifier)[-1]['metadata']['metrics']['context_tokens'],metrics['context_tokens'])
            tools.execute.assert_called_once()

    def test_agent_ignores_legacy_task_deadline_after_loading_and_compaction(self):
        import itertools
        import threading
        from unittest.mock import Mock,patch
        from desktop_agent.agent import Agent,Model,Settings
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder)
            identifier=store.create()
            model=Model(Settings(max_task_seconds=600))
            model.ensure=Mock()
            model.count=lambda text:len(text)//4
            action=dict(message='Done',tool='finish',arguments={},risk='routine')
            model.generate=Mock(return_value=(action,{}))
            tools=Mock(allow_screen=False,allow_input=False,allow_browser=False,window=None,mode='routine')
            with patch('desktop_agent.compaction.Compactor.prepare',side_effect=lambda identifier,events,*args:(events,None,set())) as compact, \
                 patch('desktop_agent.agent.time.monotonic',side_effect=itertools.count(0,86400)):
                Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Finish')
            compact.assert_called_once()
            model.generate.assert_called_once()
            self.assertEqual(json.loads(store.events(identifier)[-1]['content'])['tool'],'finish')
            tools.execute.assert_not_called()

    def test_repeated_capture_guidance_and_bounded_recovery(self):
        import threading
        from unittest.mock import Mock
        from PIL import Image
        from desktop_agent.agent import Agent, Settings
        from desktop_agent.tools import ToolResult
        capture = dict(message='Inspect',tool='desktop_screen_capture',arguments={},risk='routine')
        finish = dict(message='Observed result',tool='finish',arguments={},risk='routine')
        for recovered in (True,False):
            with self.subTest(recovered=recovered), tempfile.TemporaryDirectory() as folder:
                store = Store(folder)
                identifier = store.create()
                model = Mock(settings=Settings(),count=lambda text:len(text)//4,partial='')
                model.generate.side_effect = [(capture,{})]*4+[(finish if recovered else capture,{})]
                tools = Mock(allow_screen=True,allow_input=False,allow_browser=False,window=None,mode='routine')
                screenshot = Image.new('RGB',(80,60),'white')
                tools.execute.return_value = ToolResult('Capture delivered',screenshot)
                Agent(store,model,tools,threading.Event(),Mock()).run(identifier,'Inspect current screen')
                self.assertEqual(tools.execute.call_count,3)
                calls = model.generate.call_args_list
                self.assertEqual([call.args[1] for call in calls],[None,screenshot,screenshot,screenshot,screenshot])
                self.assertIn('latest saved tool capture',calls[1].args[0][-1]['content'])
                self.assertIn('LOOP RECOVERY',calls[2].args[0][-1]['content'])
                self.assertEqual(calls[0].args[0][0],calls[2].args[0][0])
                self.assertIn('Duplicate request not executed',json.dumps(calls[4].args[0]))
                events = store.events(identifier)
                if recovered:
                    self.assertEqual(json.loads(events[-1]['content'])['tool'],'finish')
                else:
                    self.assertIn('No progress after loop recovery',events[-1]['content'])

    def test_browser_search_falls_back_once_and_reports_blocks(self):
        import threading
        from unittest.mock import Mock
        from urllib.parse import parse_qs,urlsplit
        from game_agent.core import Halted
        from desktop_agent.tools import Tools,ToolResult
        for google_status,google_url,body in (
                (429,'https://www.google.com/search?q=test',''),
                (200,'https://www.google.com/sorry/index',''),
                (200,'https://www.google.com/search?q=test','Our systems have detected unusual traffic')):
            with self.subTest(status=google_status,url=google_url),tempfile.TemporaryDirectory() as folder:
                tools=Tools(folder,threading.Event(),Mock(),Mock())
                tools.page=Mock()
                tools.page.locator.return_value.count.return_value=0
                tools.page.goto.side_effect=[Mock(status=google_status),Mock(status=200),Mock(status=200)]
                def snapshot(url,text):
                    return ToolResult(json.dumps(dict(url=url,text=text,controls=[],title='Search')))
                tools.browser_snapshot=Mock(side_effect=[snapshot(google_url,body),
                    snapshot('https://www.bing.com/search?q=test','Results'),snapshot('https://www.bing.com/search?q=next','Next')])
                query='A&B + \uac80\uc0c9'
                result=tools.browser_search(query)
                data=json.loads(result.text)
                self.assertFalse(result.error)
                self.assertEqual(data['search']['provider'],'bing')
                self.assertEqual(len(data['search']['attempts']),2)
                self.assertEqual(data['text'],'Results')
                for call in tools.page.goto.call_args_list:
                    self.assertEqual(parse_qs(urlsplit(call.args[0]).query)['q'],[query])
                tools.browser_search('next')
                self.assertEqual(tools.page.goto.call_count,3)
                self.assertEqual(urlsplit(tools.page.goto.call_args.args[0]).hostname,'www.bing.com')
                tools.page.goto.side_effect=None
                tools.page.goto.return_value=Mock(status=403)
                tools.browser_snapshot.side_effect=None
                tools.browser_snapshot.return_value=snapshot('https://www.bing.com/search?q=test','Blocked')
                blocked=tools.browser_search('test')
                self.assertTrue(blocked.error)
                self.assertEqual(json.loads(blocked.text)['text'],'')
                tools.stopped.set()
                with self.assertRaises(Halted): tools.browser_search('cancelled')
                self.assertEqual(tools.page.goto.call_count,4)

    def test_browser_search_keeps_successful_google_results(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.tools import Tools,ToolResult
        with tempfile.TemporaryDirectory() as folder:
            tools=Tools(folder,threading.Event(),Mock(),Mock())
            tools.page=Mock()
            tools.page.goto.return_value=Mock(status=200)
            tools.page.locator.return_value.count.return_value=0
            tools.browser_snapshot=Mock(return_value=ToolResult(json.dumps(dict(
                url='https://www.google.com/search?q=captcha',text='Articles about CAPTCHA',controls=[],title='Search'))))
            data=json.loads(tools.browser_search('captcha').text)
            self.assertEqual(data['search']['provider'],'google')
            self.assertEqual(data['text'],'Articles about CAPTCHA')
            tools.page.goto.assert_called_once()

    def test_desktop_input_is_independent_of_capture_permission(self):
        import threading
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        tools.window = {'handle':1,'pid':2}
        tools.allow_input = True
        tools.allow_screen = False
        tools.desktop = Mock(return_value='done')
        with patch('desktop_agent.tools.windows.window_title',return_value='Editor'):
            result = tools.execute(dict(message='Type text',tool='desktop_key',arguments={'key':'Tab'},risk='routine'))
        self.assertEqual(result,'done')
        self.assertFalse(tools.allow_screen)

    def test_sessions_survive_restart_and_are_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            first, second = store.create('First'), store.create('Second')
            store.append(first, 'user', 'Open a page')
            store.append(first, 'assistant', 'Opening', tool='browser_open')
            store.append(first, 'tool', 'Denied', status='denied')
            store.append(second, 'user', 'Other task')
            reopened = Store(folder)
            self.assertEqual(len(reopened.events(first)), 3)
            self.assertEqual(reopened.events(second)[0]['content'], 'Other task')
            reopened.rename(first, 'Renamed')
            target = Path(folder) / 'export.json'
            reopened.export(first, target)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8'))['session']['title'], 'Renamed')
            self.assertTrue(reopened.artifact_directory(first).is_dir())
            with self.assertRaises(ValueError):
                reopened.artifact_directory('../elsewhere')

    def test_partial_stream_is_preserved_but_not_replayed_as_complete(self):
        events = [dict(role='user', content='First'), dict(role='assistant', content='Done'),
                  dict(role='user', content='Second'),
                  dict(role='assistant', content='Interrupted', metadata={'partial': True})]
        turns = conversation_turns(events)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[-1], [dict(role='user', content='Second')])


class LocalSettingsTests(unittest.TestCase):
    def test_dialog_placement_tracks_parent_and_negative_monitor(self):
        from unittest.mock import Mock,patch
        from desktop_agent.app import place_dialog
        parent = Mock()
        parent.winfo_width.return_value=1000
        parent.winfo_height.return_value=700
        dialog = Mock()
        dialog.minsize.return_value=(400,240)
        for origin,area,expected in (
                ((500,200),(0,0,1920,1040),'600x400+700+350'),
                ((-1700,100),(-1920,0,0,1040),'600x400+-1500+250'),
                ((1850,950),(0,0,1920,1040),'600x400+1308+616')):
            with self.subTest(origin=origin):
                parent.winfo_rootx.return_value,parent.winfo_rooty.return_value=origin
                with patch('desktop_agent.app.dialog_work_area',return_value=area):
                    place_dialog(dialog,parent,600,400)
                dialog.geometry.assert_called_with(expected)
                dialog.transient.assert_called_with(parent)
        dialog.deiconify.assert_called()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST')=='1','Requires Tk desktop')
    def test_linked_history_chat_small_fixture(self):
        from unittest.mock import patch
        from desktop_agent.app import Console
        from desktop_agent.agent import Settings
        from desktop_agent.compaction import source_hash
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Console,'work'),patch.object(Console,'refresh_windows'):
                app=Console(folder)
                try:
                    app.geometry('960x640+30+30')
                    request=app.store.append(app.identifier,'user','Run two checks')
                    public_note='The first check returned. I will verify the second result.'
                    calls=[]
                    for name in ('first','second'):
                        calls.append(app.store.append(app.identifier,'assistant',json.dumps(dict(message=public_note if name=='second' else name,tool='terminal_start',arguments=dict(command='Write-Output '+name,cwd='.',timeout_seconds=10),risk='routine')),call_id=name))
                    app.store.append(app.identifier,'tool',json.dumps(dict(status='running',execution_id='first')),tool='terminal_start',status='delivered',call_id='first')
                    app.store.append(app.identifier,'system',json.dumps(dict(status='completed',exit_code=0,execution_id='second',output_bytes=8)),tool='terminal_start',status='terminal',call_id='second')
                    app.expanded_activity.add((app.identifier,'tool',calls[1]))
                    app.render();app.update()
                    text=app.transcript.get('1.0','end')
                    self.assertIn('terminal_start #second',text)
                    self.assertIn('\uc2e4\ud589 \uc644\ub8cc',text)
                    self.assertIn('exit_code',text)
                    self.assertIn(public_note,text)
                    self.assertIn('\uc6d0\ubb38 \ubcf4\uae30',text)
                    self.assertLess(len(text),3500)
                    self.assertTrue(app.entry.winfo_viewable())
                    self.assertLessEqual(app.entry.winfo_rooty()+app.entry.winfo_height(),app.winfo_rooty()+app.winfo_height())
                    compact_control=app.permissions[-1]
                    self.assertTrue(compact_control.winfo_viewable())
                    self.assertLessEqual(compact_control.winfo_rootx()+compact_control.winfo_width(),app.winfo_rootx()+app.winfo_width())
                    app.auto_compact.set(False);app.compaction_changed()
                    self.assertFalse(Settings.load(Path(folder)/'settings.json').auto_compact)
                    app.auto_compact.set(True);app.compaction_changed()
                    self.assertTrue(Settings.load(Path(folder)/'settings.json').auto_compact)
                    events=app.store.events(app.identifier)
                    app.store.save_context_summary(app.identifier,dict(text='Two checks were requested.',covered_ids=[request],source_sha256=source_hash(events,[request])))
                    app.show_context_summary();app.update()
                    import tkinter as tk
                    self.assertTrue(any(isinstance(child,tk.Toplevel) for child in app.winfo_children()))
                finally:
                    app.stopped.set();app.monitor.close();app.destroy()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST')=='1','Requires Tk desktop')
    def test_dark_tool_picker_save_reopen_and_small_layout(self):
        from unittest.mock import patch
        from tkinter import ttk
        from PIL import ImageGrab
        from desktop_agent.app import Console,COLORS
        from desktop_agent.agent import Settings
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'work';root.mkdir()
            with patch.object(Console,'work'),patch.object(Console,'refresh_windows'):
                app=Console(str(Path(folder)/'data'))
                try:
                    app.update()
                    self.assertEqual(app.transcript.cget('background'),COLORS['editor'])
                    def screenshot(widget,name):
                        directory=os.environ.get('DESKTOP_AGENT_UI_ARTIFACTS')
                        if not directory:
                            return
                        widget.lift();widget.attributes('-topmost',True)
                        app.update()
                        errors=[]
                        def capture():
                            try:
                                path=Path(directory)/name
                                path.parent.mkdir(parents=True,exist_ok=True)
                                self.assertFalse(path.exists())
                                image=ImageGrab.grab(bbox=(widget.winfo_rootx(),widget.winfo_rooty(),
                                    widget.winfo_rootx()+widget.winfo_width(),widget.winfo_rooty()+widget.winfo_height()))
                                self.assertGreater(len(image.resize((100,100)).getcolors(10001)),8)
                                image.save(path)
                            except Exception as error:
                                errors.append(error)
                            finally:
                                app.quit()
                        app.after(120,capture);app.mainloop()
                        widget.attributes('-topmost',False)
                        if errors: raise errors[0]
                    for width,height in ((1180,800),(960,640)):
                        app.geometry(f'{width}x{height}+30+30');app.update()
                        for widget in (app.entry,app.send_button,app.tool_picker_button,app.execution_button):
                            self.assertTrue(widget.winfo_viewable())
                            self.assertLessEqual(widget.winfo_rootx()+widget.winfo_width(),app.winfo_rootx()+app.winfo_width())
                            self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),app.winfo_rooty()+app.winfo_height())
                        screenshot(app,f'main-{width}.png')
                    dialog=app.tools_dialog();app.update()
                    widgets=list(descendants(dialog))
                    tree=next(widget for widget in widgets if isinstance(widget,ttk.Treeview))
                    root_entry=next(widget for widget in widgets if isinstance(widget,ttk.Entry) and not isinstance(widget,ttk.Combobox))
                    root_entry.insert(0,str(root))
                    self.assertEqual(tree.item('terminal_start','values')[0],'\uc0ac\uc6a9 \uc548 \ud568')
                    tree.selection_set('group:terminal');app.update()
                    selector=next(widget for widget in widgets if isinstance(widget,ttk.Combobox))
                    selector.set('\ub9e4\ubc88 \ud655\uc778');selector.event_generate('<<ComboboxSelected>>');app.update()
                    for width,height in ((800,650),(620,480)):
                        dialog.geometry(f'{width}x{height}+70+70');app.update()
                        save=next(widget for widget in widgets if isinstance(widget,ttk.Button) and widget.cget('text')=='\uc800\uc7a5')
                        self.assertTrue(save.winfo_viewable())
                        self.assertLessEqual(save.winfo_rooty()+save.winfo_height(),dialog.winfo_rooty()+dialog.winfo_height())
                        screenshot(dialog,f'tool-picker-{width}.png')
                    save.invoke();app.update()
                    loaded=Settings.load(Path(folder)/'data/settings.json')
                    self.assertEqual(loaded.workspace_root,str(root))
                    self.assertEqual(loaded.tool_policies['terminal_start'],'ask')
                    self.assertEqual(app.permission_snapshot()['tool_policies'],loaded.tool_policies)
                    dialog=app.tools_dialog();app.update()
                    tree=next(widget for widget in descendants(dialog) if isinstance(widget,ttk.Treeview))
                    self.assertIn('\ud544\uc218',tree.item('terminal_start','values')[0])
                    dialog.destroy()
                    viewer=app.execution_dialog();viewer.geometry('620x420+70+70');app.update()
                    screenshot(viewer,'execution-empty-620.png')
                    viewer.destroy()
                finally:
                    app.stopped.set();app.monitor.close();app.destroy()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST')=='1','Requires Tk desktop')
    def test_chat_composer_tools_and_stream_tail_are_stable(self):
        from desktop_agent.app import Console,tool_rows,display_excerpt
        with tempfile.TemporaryDirectory() as folder:
            app=Console(folder)
            try:
                app.update()
                app.entry.insert('1.0','first');app.entry.edit_separator();app.entry.changed()
                app.entry.insert('end',' second');app.entry.edit_separator();app.entry.changed()
                app.entry.history(app.entry.edit_undo);app.entry.changed()
                self.assertEqual(app.prompt.get(),'first')
                app.entry.history(app.entry.edit_redo);app.entry.changed()
                self.assertEqual(app.prompt.get(),'first second')
                request=app.store.append(app.identifier,'user','Copy this user request')
                call=app.store.append(app.identifier,'assistant',json.dumps(dict(tool='browser_search',arguments={'query':'test'},message='Search')))
                app.render();app.update()
                self.assertIn('browser_search',app.transcript.get('1.0','end'))
                app.copy_message(request)
                self.assertEqual(app.clipboard_get(),'Copy this user request')
                self.assertTrue(app.transcript.tag_ranges('menu_copy_'+str(request)))
                large='<html><body>Result</body></html>'*10000
                app.store.append(app.identifier,'tool',large,tool='browser_search',status='delivered')
                app.render();app.update()
                self.assertEqual(tool_rows(app.store.events(app.identifier))[-1]['status'],'delivered')
                app.toggle_activity((app.identifier,'tool',call));app.update()
                self.assertLess(len(app.transcript.get('1.0','end')),8000)
                self.assertLess(len(display_excerpt(large)),12100)
                app.live_reasoning=dict(session_id=app.identifier,key=('live',request,0),enabled=True,text='line\n'*100)
                app.update_live_reasoning();app.toggle_live_reasoning();app.update()
                app.transcript.tag_remove('sel','1.0','end')
                app.transcript.yview_moveto(1);app.update()
                for index in range(20):
                    app.live_reasoning['text']+='new line '+str(index)+'\n'
                    app.update_live_reasoning();app.update()
                    self.assertGreaterEqual(app.transcript.yview()[1],.999)
                    self.assertEqual(app.transcript.get('live_reasoning_start','live_reasoning_end').count('new line '+str(index)+'\n'),1)
                    self.assertEqual(app.transcript.get('tail_padding','end-1c'),'\n\n\n')
                app.live_reasoning['text']=large
                app.update_live_reasoning();app.update()
                self.assertLess(len(app.transcript.get('live_reasoning_start','live_reasoning_end')),12400)
            finally:
                app.close();app.mainloop()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST')=='1','Requires Tk desktop')
    def test_capture_margin_opt_in_persists_and_snapshots_without_changing_model(self):
        from unittest.mock import patch
        from desktop_agent.app import Console
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            try:
                app.update()
                original = app.settings.model
                self.assertEqual(app.capture_margin.get(),'0 px')
                with patch('desktop_agent.app.messagebox.askyesno',return_value=False):
                    app.capture_margin.set('256 px')
                    app.capture_margin_changed()
                self.assertEqual(app.settings.capture_margin,0)
                self.assertEqual(app.capture_margin.get(),'0 px')
                with patch('desktop_agent.app.messagebox.askyesno',return_value=True):
                    app.capture_margin.set('256 px')
                    app.capture_margin_changed()
                self.assertEqual(Settings.load(Path(folder)/'settings.json').capture_margin,256)
                self.assertEqual(app.permission_snapshot()['capture_margin'],256)
                self.assertEqual(app.settings.model,original)
                app.set_busy(True)
                self.assertEqual(str(app.margin_selector.cget('state')),'disabled')
                app.set_busy(False)
                self.assertEqual(str(app.margin_selector.cget('state')),'readonly')
                app.capture_margin.set('0 px')
                app.capture_margin_changed()
                self.assertEqual(app.permission_snapshot()['capture_margin'],0)
            finally:
                app.close()
                app.mainloop()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_original_image_option_saves_without_switching_model(self):
        from unittest.mock import Mock
        from desktop_agent.app import Console
        from desktop_agent.agent import Settings
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            try:
                app.update()
                original = app.settings.model
                app.enqueue = Mock()
                app.image_selector.set('Original')
                app.image_selector.event_generate('<<ComboboxSelected>>')
                app.update()
                self.assertEqual(app.settings.image_max_edge,0)
                self.assertEqual(Settings.load(Path(folder)/'settings.json').image_max_edge,0)
                self.assertEqual(app.settings.model,original)
                app.enqueue.assert_called_with('image_settings',app.settings)
                app.set_busy(True)
                self.assertEqual(str(app.image_selector.cget('state')),'disabled')
                app.set_busy(False)
                self.assertEqual(str(app.image_selector.cget('state')),'readonly')
            finally:
                app.close()
                app.mainloop()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_chat_updates_preserve_view_selection_and_drag(self):
        from desktop_agent.app import Console
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            try:
                app.update()
                first = app.identifier
                other = app.store.create('Other')
                for identifier in (first,other):
                    app.store.append(identifier,'user','Long synthetic conversation')
                    app.store.append(identifier,'assistant',json.dumps(dict(tool='finish',message='Visible row\n\n'*90)))
                app.identifier = other
                app.render()
                app.update()
                self.assertGreater(app.transcript.yview()[1],0.98)
                app.transcript.yview_moveto(0.3)
                app.update()
                top = app.transcript.index('@0,0')
                app.transcript.tag_add('sel',top,top+' + 7 chars')
                selection = app.transcript.get('sel.first','sel.last')
                app.begin_transcript_selection()
                app.store.append(other,'assistant',json.dumps(dict(tool='finish',message='New answer\n'*30)))
                app.render()
                self.assertNotIn('New answer',app.transcript.get('1.0','end'))
                app.end_transcript_selection()
                app.update()
                self.assertIn('New answer',app.transcript.get('1.0','end'))
                self.assertEqual(app.transcript.index('@0,0'),top)
                self.assertEqual(app.transcript.get('sel.first','sel.last'),selection)
                app.transcript.event_generate('<<Copy>>')
                self.assertEqual(app.clipboard_get(),selection)
                from types import SimpleNamespace
                from unittest.mock import patch
                with patch.object(app.message_context,'tk_popup'):
                    app.busy = True
                    app.show_selection_menu(SimpleNamespace(x_root=0,y_root=0))
                    self.assertEqual(app.message_context.entrycget(0,'state'),'normal')
                    app.message_context.invoke(0)
                    self.assertEqual(app.clipboard_get(),selection)
                    app.busy = False
                app.live_reasoning = dict(session_id=other,key=('live',1,0),enabled=True,text='Streaming reasoning')
                app.update_live_reasoning()
                app.update()
                self.assertEqual(app.transcript.index('@0,0'),top)
                self.assertEqual(app.transcript.get('sel.first','sel.last'),selection)
                app.transcript.see('end')
                app.update()
                bottom_top = app.transcript.index('@0,0')
                app.store.append(other,'assistant',json.dumps(dict(tool='finish',message='Later answer\n'*40)))
                app.render()
                app.update()
                self.assertEqual(app.transcript.index('@0,0'),bottom_top)
                self.assertLess(app.transcript.yview()[1],0.98)
                app.identifier = first
                app.render()
                app.update()
                self.assertGreater(app.transcript.yview()[1],0.98)
                self.assertFalse(app.transcript.tag_ranges('sel'))
            finally:
                app.close()
                app.mainloop()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_inline_reasoning_resets_per_request_and_keeps_partial_session_scope(self):
        import tkinter as tk
        from desktop_agent.app import Console
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            try:
                app.update()
                session = app.identifier
                prompt = app.store.append(session,'user','Synthetic request')
                app.render()
                app.live_reasoning = dict(session_id=session,key=('live',prompt,0),text='',enabled=True)
                app.update_live_reasoning()
                self.assertIn('[\uc0ac\uace0\uc911]',app.transcript.get('1.0','end'))
                app.toggle_live_reasoning()
                app.live_reasoning['text'] = 'Partial synthetic reasoning.'
                app.update_live_reasoning()
                self.assertIn('Partial synthetic reasoning.',app.transcript.get('1.0','end'))
                app.identifier = app.store.create('Other session')
                app.render()
                self.assertNotIn('Partial synthetic reasoning.',app.transcript.get('1.0','end'))
                self.assertNotIn('[\uc0ac\uace0\uc911]',app.transcript.get('1.0','end'))
                app.identifier = session
                stopped = app.store.append(session,'system','Stopped',status='stopped',
                    metrics=dict(reasoning='Partial synthetic reasoning.',reasoning_partial=True))
                app.complete_live_reasoning(dict(session_id=session,event_id=stopped))
                app.render()
                self.assertIn('(\ubbf8\uc644\ub8cc)',app.transcript.get('1.0','end'))
                self.assertIn('Partial synthetic reasoning.',app.transcript.get('1.0','end'))
                app.show_reasoning(stopped)
                self.assertNotIn('Partial synthetic reasoning.',app.transcript.get('1.0','end'))
                app.live_reasoning = dict(session_id=session,key=('live',prompt,1),text='Next response reasoning.')
                app.update_live_reasoning()
                self.assertNotIn('Next response reasoning.',app.transcript.get('1.0','end'))
                app.toggle_live_reasoning()
                self.assertIn('Next response reasoning.',app.transcript.get('1.0','end'))
                app.complete_live_reasoning()
                app.update_live_reasoning()
                self.assertNotIn('Next response reasoning.',app.transcript.get('1.0','end'))
                self.assertFalse(any(isinstance(widget,tk.Toplevel) for widget in app.winfo_children()))
                self.assertNotIn('Next response reasoning.',app.store.history_text(session))
            finally:
                app.close()
                app.mainloop()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_excluded_history_stays_visible_and_marker_clears_when_reincluded(self):
        from desktop_agent.app import Console
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            try:
                request = app.store.append(app.identifier,'user','Old visible request')
                call = app.store.append(app.identifier,'assistant',json.dumps(dict(message='Read',tool='browser_read',arguments={},risk='routine')))
                result = app.store.append(app.identifier,'tool','Old visible page',tool='browser_read',status='delivered')
                reply = app.store.append(app.identifier,'assistant',json.dumps(dict(message='Old visible reply',tool='finish',arguments={},risk='routine')))
                app.store.save_context_selection(app.identifier,dict(excluded_ids=[request,call,result,reply]))
                app.render()
                displayed = app.transcript.get('1.0','end')
                self.assertIn('Old visible request',displayed)
                self.assertIn('Old visible reply',displayed)
                self.assertIn('\ubb38\ub9e5 \uc81c\uc678',displayed)
                app.toggle_activity((app.identifier,call))
                self.assertIn('Old visible page',app.transcript.get('1.0','end'))
                app.store.save_context_selection(app.identifier,dict(excluded_ids=[]))
                app.render()
                self.assertNotIn('\ubb38\ub9e5 \uc81c\uc678',app.transcript.get('1.0','end'))
                self.assertEqual(len(app.store.events(app.identifier)),4)
            finally:
                app.close()
                app.mainloop()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_qwen_effort_controls_and_reasoning_history_visibility(self):
        import tkinter as tk
        import time
        from unittest.mock import Mock
        from desktop_agent.app import Console
        from desktop_agent.agent import Settings
        from game_agent.models import MODEL_PRESETS
        with tempfile.TemporaryDirectory() as folder:
            app = Console(folder)
            try:
                app.update()
                app.enqueue = Mock()
                self.assertEqual(tuple(app.reason_selector.cget('values')),('low','medium','xhigh'))
                app.reason_level.set('xhigh')
                app.reasoning_level_changed()
                app.enqueue.assert_called_with('reasoning',app.settings)
                self.assertEqual(app.settings.reasoning_effort,'xhigh')
                app.settings_dialog()
                dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
                app.update()
                label = dialog.grid_slaves(row=4,column=0)[0]
                self.assertEqual(label.cget('text'),'reasoning_effort')
                self.assertEqual(tuple(dialog.grid_slaves(row=4,column=1)[0].cget('values')),('low','medium','xhigh'))
                dialog.destroy()
                app.store.append(app.identifier,'user','Synthetic task')
                summary = 'Synthetic reasoning content for the UI test.'
                tool_id = app.store.append(app.identifier,'assistant',json.dumps(dict(message='Capture',tool='desktop_capture',arguments={},risk='routine')),metrics={'reasoning':summary})
                reply_id = app.store.append(app.identifier,'assistant',json.dumps(dict(message='Done',tool='finish',arguments={},risk='routine')),metrics={'reasoning':summary})
                app.render()
                self.assertTrue(app.transcript.tag_ranges('reasoning_'+str(reply_id)))
                self.assertNotIn(summary,app.transcript.get('1.0','end'))
                app.show_reasoning(reply_id)
                self.assertIn(summary,app.transcript.get('1.0','end'))
                self.assertFalse(any(isinstance(widget,tk.Toplevel) for widget in app.winfo_children()))
                self.assertEqual(str(app.transcript.cget('state')),'disabled')
                app.show_reasoning(reply_id)
                self.assertNotIn(summary,app.transcript.get('1.0','end'))
                app.toggle_activity((app.identifier,tool_id))
                self.assertTrue(app.transcript.tag_ranges('reasoning_'+str(tool_id)))
                self.assertFalse(hasattr(app,'reason_log'))
                app.events.put(('reasoning_start',dict(session_id=app.identifier,request_id=1,step=0)))
                app.events.put(('reasoning',summary))
                deadline = time.monotonic()+2
                while not app.transcript.tag_ranges('reasoning_live'):
                    app.update()
                    self.assertLess(time.monotonic(),deadline)
                self.assertIn('[\uc0ac\uace0\uc911]',app.transcript.get('1.0','end'))
                self.assertNotIn(summary,app.transcript.get('1.0','end'))
                app.toggle_live_reasoning()
                self.assertIn(summary,app.transcript.get('1.0','end'))
                app.events.put(('reasoning',summary+' Updated.'))
                while 'Updated.' not in app.transcript.get('1.0','end'):
                    app.update()
                    self.assertLess(time.monotonic(),deadline)
                live_id = app.store.append(app.identifier,'assistant',json.dumps(dict(message='Final',tool='finish')),
                    metrics=dict(reasoning=summary+' Updated.'))
                app.events.put(('reasoning_saved',dict(session_id=app.identifier,event_id=live_id)))
                while app.live_reasoning is not None:
                    app.update()
                    self.assertLess(time.monotonic(),deadline)
                self.assertIn((app.identifier,live_id),app.expanded_reasoning)
                self.assertIn('Updated.',app.transcript.get('1.0','end'))
                self.assertNotIn('[\uc0ac\uace0\uc911]',app.transcript.get('1.0','end'))
                small = MODEL_PRESETS['qwen38_nvfp4']
                app.settings = Settings(model=small['model'],projector=small['projector'])
                app.update_backend_display()
                self.assertIn('extended',app.reason_selector.cget('values'))
            finally:
                app.close()
                app.mainloop()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_UI_TEST') == '1','Requires Tk desktop')
    def test_local_preset_dialog_previews_pairs_saves_and_cancels(self):
        import tkinter as tk
        from tkinter import ttk
        from unittest.mock import Mock, patch
        from desktop_agent.app import Console
        from desktop_agent.agent import Settings
        from desktop_agent.models import MODEL_PRESETS
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder)/'server.exe'
            executable.touch()
            for name in ('qwen38_27b_iq2_s','qwen38_nvfp4','qwen38_27b_q2_k_xl'):
                for field in ('model','projector'):
                    (Path(folder)/MODEL_PRESETS[name][field]).touch()
            settings = Settings(executable=str(executable)).with_local_preset('qwen38_27b_iq2_s',folder)
            settings.save(Path(folder)/'settings.json')
            app = Console(folder)
            try:
                app.update()
                app.enqueue = Mock()
                app.settings_dialog()
                dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
                selector = dialog.grid_slaves(row=0,column=1)[0]
                preview = next(widget for widget in dialog.winfo_children() if isinstance(widget,ttk.Label) and widget.cget('text') == 'Model options')
                preview = dialog.grid_slaves(row=int(preview.grid_info()['row']),column=1)[0]
                text = next(widget for widget in preview.winfo_children() if isinstance(widget,tk.Text))
                self.assertIn('--override-tensor',text.get('1.0','end'))
                cache = dialog.grid_slaves(row=7,column=1)[0]
                self.assertFalse(cache.instate(['selected']))
                self.assertIn('--cache-ram 0',text.get('1.0','end'))
                cache.invoke()
                self.assertIn('--cache-ram 2048',text.get('1.0','end'))
                self.assertEqual(app.settings.cache_ram_mib,0)
                self.assertIn(MODEL_PRESETS['qwen38_27b_q2_k_xl']['label'],selector.cget('values'))
                selector.set(MODEL_PRESETS['qwen38_27b_q2_k_xl']['label'])
                selector.event_generate('<<ComboboxSelected>>')
                app.update()
                self.assertIn('--device CUDA0,Vulkan0',text.get('1.0','end'))
                self.assertIn('--mmproj-device CUDA0',text.get('1.0','end'))
                self.assertIn('--load-mode none',text.get('1.0','end'))
                self.assertEqual(app.settings,settings)
                selector.set(MODEL_PRESETS['qwen38_nvfp4']['label'])
                selector.event_generate('<<ComboboxSelected>>')
                app.update()
                self.assertNotIn('--override-tensor',text.get('1.0','end'))
                self.assertIn('--image-max-tokens 384',text.get('1.0','end'))
                self.assertIn('-c 16384',text.get('1.0','end'))
                self.assertEqual(app.settings,settings)
                kv = dialog.grid_slaves(row=8,column=1)[0]
                kv.set('4-bit (Q4_0)')
                kv.event_generate('<<ComboboxSelected>>')
                app.update()
                self.assertIn('-ctk q4_0',text.get('1.0','end'))
                buttons = dialog.grid_slaves(row=10,column=0)[0]
                save = next(widget for widget in buttons.winfo_children() if widget.cget('text') == 'Save and unload')
                for geometry in ('940x550','740x500'):
                    dialog.geometry(geometry)
                    app.update()
                    self.assertTrue(save.winfo_ismapped())
                    self.assertTrue(cache.winfo_ismapped())
                    self.assertLessEqual(cache.winfo_rooty()+cache.winfo_height(),preview.winfo_rooty())
                    self.assertLessEqual(save.winfo_rooty()+save.winfo_height(),dialog.winfo_rooty()+dialog.winfo_height())
                with patch('desktop_agent.app.messagebox.askyesno',side_effect=[True,False]):
                    save.invoke()
                self.assertEqual(app.settings,settings)
                self.assertTrue(dialog.winfo_exists())
                with patch('desktop_agent.app.messagebox.askyesno',return_value=True):
                    save.invoke()
                self.assertEqual(app.settings.cache_ram_mib,2048)
                self.assertEqual(app.settings.kv_cache_type,'q4_0')
                self.assertEqual(Path(app.settings.model).name,MODEL_PRESETS['qwen38_nvfp4']['model'])
                self.assertEqual(app.settings.local_image_tokens,(192,384))
                self.assertIn('Distill-Heretic',app.title())
                self.assertEqual(Settings.load(Path(folder)/'settings.json'),app.settings)
                app.enqueue.assert_called_once_with('settings',app.settings)
                saved = app.settings
                app.settings_dialog()
                dialog = next(widget for widget in app.winfo_children() if isinstance(widget,tk.Toplevel))
                cache = dialog.grid_slaves(row=7,column=1)[0]
                self.assertTrue(cache.instate(['selected']))
                cache.invoke()
                selector = dialog.grid_slaves(row=0,column=1)[0]
                selector.set(MODEL_PRESETS['qwen38_27b_iq2_s']['label'])
                selector.event_generate('<<ComboboxSelected>>')
                self.assertEqual(dialog.grid_slaves(row=8,column=1)[0].get(),'4-bit (Q4_0)')
                buttons = dialog.grid_slaves(row=10,column=0)[0]
                next(widget for widget in buttons.winfo_children() if widget.cget('text') == 'Cancel').invoke()
                self.assertEqual(app.settings,saved)
            finally:
                app.close()
                app.mainloop()


class BrowserTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_WINDOW_TEST') == '1', 'Requires visible Windows desktop')
    def test_image_to_click_binding_matches_received_pixels(self):
        if os.environ.get('DESKTOP_AGENT_CLICK_TEST_CHILD') != '1':
            import subprocess
            import sys
            completed = subprocess.run(
                [sys.executable,'-m','unittest','discover','-s','tests','-p','test_desktop_agent.py','-k','image_to_click_binding','-v'],
                cwd=Path(__file__).resolve().parents[1],
                env=dict(os.environ,DESKTOP_AGENT_CLICK_TEST_CHILD='1'),
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=40,
                creationflags=subprocess.CREATE_NO_WINDOW)
            self.assertEqual(completed.returncode,0,completed.stdout.decode('utf-8',errors='replace'))
            print(completed.stdout.decode('utf-8',errors='replace'),flush=True)
            return
        import base64
        import ctypes
        from ctypes import wintypes
        from concurrent.futures import ThreadPoolExecutor
        from io import BytesIO
        import threading
        import time
        import tkinter as tk
        from unittest.mock import Mock
        import numpy as np
        from PIL import Image
        from desktop_agent.tools import Tools
        from game_agent import windows
        from game_agent.vision import encode_image
        windows.enable_dpi()
        native = windows.user32
        native.SetWindowPos.argtypes = [wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.UINT]
        root = tk.Tk()
        root.title('Local Desk coordinate binding test')
        width,height = min(1500,root.winfo_screenwidth()-180),min(700,root.winfo_screenheight()-180)
        root.geometry(f'{width}x{height}+100+100')
        root.attributes('-topmost',True)
        canvas = tk.Canvas(root,bg='white',highlightthickness=0)
        canvas.pack(fill='both',expand=True)
        received = []
        canvas.bind('<Button-1>',lambda event:received.append((event.x_root,event.y_root)))
        root.update()
        handle = native.GetAncestor(root.winfo_id(),2)
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        tools.window = {'handle':handle,'pid':os.getpid()}
        samples = []
        def wait_for(future):
            deadline = time.monotonic()+8
            while not future.done():
                root.update()
                if time.monotonic() >= deadline:
                    self.fail('Coordinate probe timed out')
            return future.result()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                left,top,screen_width,screen_height = windows.virtual_screen()
                origins = [(100,100)]
                if left < 0:
                    origins.append((left+100,top+100))
                for horizontal,vertical in origins:
                    native.SetWindowPos(handle,None,horizontal,vertical,0,0,0x0015)
                    root.update()
                    for index,(fraction_x,fraction_y) in enumerate(((0.05,0.08),(0.95,0.08),(0.5,0.5),(0.05,0.92),(0.95,0.92))):
                        marker_x,marker_y = round(fraction_x*canvas.winfo_width()),round(fraction_y*canvas.winfo_height())
                        canvas.delete('all')
                        canvas.create_rectangle(marker_x-10,marker_y-10,marker_x+10,marker_y+10,fill='#ff00ff',outline='')
                        root.update()
                        expected = (canvas.winfo_rootx()+marker_x,canvas.winfo_rooty()+marker_y)
                        capture = wait_for(pool.submit(tools.desktop,'desktop_capture',{}))
                        edge = 1280 if index%2==0 else None
                        tools.coordinate_reference(capture.image,edge)
                        encoded,_ = encode_image(capture.image,edge,90,'rgb')
                        with Image.open(BytesIO(base64.b64decode(encoded.split(',',1)[1]))) as resized:
                            pixels = np.asarray(resized)
                            rows,columns = np.where((pixels[:,:,0]>200)&(pixels[:,:,1]<70)&(pixels[:,:,2]>180))
                            self.assertGreater(len(rows),10)
                            image_point = (round(float(columns.mean())),round(float(rows.mean())))
                        received.clear()
                        clicked = wait_for(pool.submit(tools.desktop,'desktop_click',dict(x=image_point[0],y=image_point[1],button='left',clicks=1,coordinate_space='image_pixels')))
                        root.update()
                        self.assertEqual(len(received),1)
                        receipt = json.loads(clicked.text)
                        self.assertEqual(receipt['bounds'],json.loads(capture.text)['bounds'])
                        self.assertLessEqual(max(abs(receipt['pointer_before_press'][0][axis]-received[0][axis]) for axis in (0,1)),1)
                        error = tuple(received[0][axis]-expected[axis] for axis in (0,1))
                        samples.append(dict(origin=[horizontal,vertical],image_pixels=image_point,max_edge=edge,error_pixels=error))
                        self.assertLessEqual(max(abs(value) for value in error),2,str(samples[-1]))
        finally:
            tools.stopped.set()
            root.destroy()
        print('IMAGE_CLICK_BINDING '+json.dumps(samples),flush=True)

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_WINDOW_TEST') == '1', 'Requires visible Windows desktop')
    def test_worker_hwnd_capture_uses_physical_pixels_and_restores_unaware_context(self):
        if os.environ.get('DESKTOP_AGENT_WORKER_TEST_CHILD') != '1':
            import subprocess
            import sys
            completed = subprocess.run(
                [sys.executable,'-m','unittest','discover','-s','tests','-p','test_desktop_agent.py','-k','worker_hwnd','-v'],
                cwd=Path(__file__).resolve().parents[1],
                env=dict(os.environ,DESKTOP_AGENT_WORKER_TEST_CHILD='1'),
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW)
            self.assertEqual(completed.returncode,0,completed.stdout.decode('utf-8',errors='replace'))
            return
        import ctypes
        from ctypes import wintypes
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import time
        import tkinter as tk
        from unittest.mock import Mock
        from desktop_agent.tools import Tools, _dpi_user32, visible_window_region
        from game_agent import windows
        windows.enable_dpi()
        native = windows.user32
        native.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        native.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        native.SetWindowPos.argtypes = [wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.UINT]
        root = tk.Tk()
        root.overrideredirect(True)
        root.geometry('400x240+100+100')
        root.configure(background='#ff0000')
        root.attributes('-topmost',True)
        tk.Frame(root,bg='#0000ff').place(relx=0.5,y=0,relwidth=0.5,relheight=1)
        root.update_idletasks()
        handle = native.GetAncestor(root.winfo_id(),2)
        left,top,width,height = windows.virtual_screen()
        native.SetWindowPos(handle,None,left+100,top+100,400,240,0x0014)
        root.update_idletasks()
        expected = visible_window_region(handle)
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        tools.window = {'handle':handle,'pid':os.getpid()}
        def worker():
            previous = _dpi_user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-1))
            try:
                result = tools.desktop('desktop_capture',{})
                after = native.GetAwarenessFromDpiAwarenessContext(native.GetThreadDpiAwarenessContext())
                return result,after
            finally:
                _dpi_user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))
        root.update()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker)
            try:
                deadline = time.monotonic()+10
                while not future.done():
                    root.update()
                    if time.monotonic() >= deadline:
                        self.fail('Worker window capture did not finish')
                result,after = future.result()
                self.assertEqual(after,0)
                self.assertEqual(json.loads(result.text)['bounds'],list(expected.bbox))
                self.assertEqual(result.image.size,(expected.width,expected.height))
                self.assertEqual(result.image.getpixel((2,120)),(255,0,0))
                self.assertEqual(result.image.getpixel((397,120)),(0,0,255))
            finally:
                tools.stopped.set()
                root.destroy()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_WINDOW_TEST') == '1', 'Requires visible Windows desktop')
    def test_actual_covered_window_captures_both_edges_and_records_without_focus(self):
        from concurrent.futures import ThreadPoolExecutor
        import tkinter as tk
        import threading
        import time
        from unittest.mock import Mock
        from desktop_agent.tools import Tools
        from game_agent import windows
        windows.enable_dpi()
        with tempfile.TemporaryDirectory() as folder:
            root = tk.Tk()
            root.overrideredirect(True)
            root.geometry('640x360+100+100')
            root.configure(background='white')
            root.attributes('-topmost',True)
            tk.Frame(root,bg='#ff0000').place(x=0,y=0,width=16,relheight=1)
            tk.Frame(root,bg='#0000ff').place(relx=1,y=0,width=16,relheight=1,anchor='ne')
            overlay = tk.Toplevel(root)
            overlay.overrideredirect(True)
            overlay.geometry('160x120+340+220')
            overlay.configure(background='#00ff00')
            overlay.attributes('-topmost',True)
            root.update_idletasks()
            handle = windows.user32.GetAncestor(root.winfo_id(),2)
            overlay.lift()
            tools = Tools(folder,threading.Event(),Mock(return_value=True),Mock())
            tools.window = {'handle':handle,'pid':os.getpid()}
            tools.focus_window = Mock(side_effect=AssertionError('Capture must not focus'))
            def verify():
                result = tools.execute(dict(message='Capture selected window',tool='desktop_capture',arguments={},risk='routine'))
                self.assertEqual(result.image.size,(640,360))
                self.assertEqual(result.image.getpixel((0,180)),(255,0,0))
                self.assertEqual(result.image.getpixel((639,180)),(0,0,255))
                self.assertEqual(result.image.getpixel((320,180)),(255,255,255))
                desktop = tools.execute(dict(message='Capture desktop',tool='desktop_screen_capture',arguments={},risk='routine'))
                bounds = json.loads(desktop.text)['bounds']
                self.assertEqual(desktop.image.getpixel((420-bounds[0],280-bounds[1])),(0,255,0))
                result.image.save(Path(__file__).resolve().parents[1]/'desktop_agent'/'desktop-capture-edges.png')
                video = tools.execute(dict(message='Record selected window',tool='desktop_record',arguments={'seconds':1,'fps':2},risk='routine'))
                self.assertTrue(Path(video.video).is_file())
                self.assertFalse(video.interrupted)
                tools.focus_window.assert_not_called()
            root.update()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(verify)
                try:
                    deadline = time.monotonic()+10
                    while not future.done():
                        root.update()
                        if time.monotonic() >= deadline:
                            self.fail('Covered window capture did not finish')
                    future.result()
                finally:
                    tools.stopped.set()
                    root.destroy()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_WINDOW_TEST') == '1', 'Requires visible Windows desktop')
    def test_actual_visible_desktop_capture_keeps_foreground_and_pointer(self):
        import threading
        from unittest.mock import Mock, patch
        from desktop_agent.tools import Tools
        from game_agent import windows
        windows.enable_dpi()
        tools = Tools('.',threading.Event(),Mock(return_value=True),Mock())
        tools.allow_input = False
        with patch.object(windows.user32,'SetForegroundWindow') as focus, \
             patch.object(windows.user32,'ShowWindow') as show, \
             patch.object(windows.user32,'SetCursorPos') as move, \
             patch.object(windows.user32,'SendInput') as send:
            result = tools.execute(dict(message='View current desktop',tool='desktop_screen_capture',arguments={},risk='routine'))
            focus.assert_not_called()
            show.assert_not_called()
            move.assert_not_called()
            send.assert_not_called()
        self.assertEqual(result.image.size,windows.virtual_screen()[2:])
        self.assertIsNone(tools.window)

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_BROWSER_TEST') == '1', 'Requires installed Chromium')
    def test_browser_hold_and_drag_release_in_actual_page(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.tools import Tools
        with tempfile.TemporaryDirectory() as folder:
            tools = Tools(folder,threading.Event(),Mock(return_value=True),Mock(),headless=True)
            try:
                tools.ensure_browser()
                tools.page.set_content('<input id="field"><div id="source" style="width:100px;height:80px;background:red">Source</div>'
                    '<div id="target" style="margin-left:200px;width:100px;height:80px;background:blue">Target</div>'
                    '<script>window.events=[]; for (const name of ["keydown","keyup","mousedown","mouseup"]) '
                    'document.addEventListener(name,event=>window.events.push(name));</script>')
                tools.execute(dict(message='Hold key',tool='browser_hold',arguments=dict(selector='#field',kind='key',key='Control+a',button='left',hold_ms=50),risk='routine'))
                tools.execute(dict(message='Drag',tool='browser_drag',arguments=dict(selector='#source',target_selector='#target',duration_ms=100),risk='routine'))
                events = tools.page.evaluate('window.events')
                self.assertEqual(events.count('keydown'),2)
                self.assertEqual(events.count('keyup'),2)
                self.assertEqual(events[-2:],['mousedown','mouseup'])
            finally:
                tools.close()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_WINDOW_TEST') == '1', 'Requires visible Windows desktop')
    def test_window_message_text_reaches_native_child_without_cursor_or_focus_change(self):
        import ctypes
        from ctypes import wintypes
        import threading
        from game_agent import windows
        from desktop_agent.input_modes import message_input
        native = windows.user32
        native.CreateWindowExW.argtypes = [wintypes.DWORD,wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.DWORD,
            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.HWND,wintypes.HMENU,wintypes.HINSTANCE,ctypes.c_void_p]
        native.CreateWindowExW.restype = wintypes.HWND
        native.DestroyWindow.argtypes = [wintypes.HWND]
        parent = native.CreateWindowExW(0,'STATIC','Input test',0x00cf0000,100,100,400,250,None,None,None,None)
        self.assertTrue(parent)
        try:
            child = native.CreateWindowExW(0,'EDIT','',0x50800004,10,10,350,170,parent,None,None,None)
            self.assertTrue(child)
            native.ShowWindow(parent,4)
            before = windows.cursor_position(),native.GetForegroundWindow()
            arguments = dict(kind='text',x=400,y=400,text='Hello \ud55c\uae00',key='',button='left',amount=0,hold_ms=0)
            result = message_input({'handle':parent,'pid':os.getpid()},arguments,threading.Event())
            self.assertIn('NOT guaranteed',result)
            self.assertEqual(windows.window_title(child),'Hello \ud55c\uae00')
            self.assertEqual((windows.cursor_position(),native.GetForegroundWindow()),before)
        finally:
            native.DestroyWindow(parent)

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_BROWSER_TEST') == '1', 'Requires installed Chromium')
    def test_background_recording_captures_simultaneous_browser_click(self):
        import cv2
        import threading
        from unittest.mock import Mock
        from desktop_agent.jobs import ToolRunner
        with tempfile.TemporaryDirectory() as folder:
            runner = ToolRunner(folder,threading.Event(),Mock(return_value=True),Mock(),headless=True)
            def action(tool,arguments=None):
                return dict(message='Use test page',tool=tool,arguments=arguments or {},risk='routine')
            try:
                runner.execute(action('browser_read'))
                runner.browser_lane.submit(lambda:runner.browser_tools.page.set_content(
                    '<body style="background:rgb(220,20,20)"><button id="next" '
                    'onclick="document.body.style.background=\'rgb(20,20,220)\'">Next</button></body>')).result()
                started = runner.execute(action('job_start',{'action':action('browser_record',{'seconds':3,'fps':6})}))
                data = json.loads(started.text)
                self.assertTrue(data['ready'])
                identifier = data['job_id']
                self.assertFalse(runner.background[identifier]['future'].done())
                runner.execute(action('browser_click',{'selector':'#next'}))
                self.assertFalse(runner.background[identifier]['future'].done())
                result = runner.execute(action('job_result',{'job_id':identifier}))
                self.assertFalse(result.error,result.text)
                capture = cv2.VideoCapture(result.video)
                try:
                    frames = []
                    while True:
                        success,frame = capture.read()
                        if not success:
                            break
                        frames.append(frame)
                    self.assertEqual(len(frames),18)
                    self.assertGreater(float(frames[-1].mean(axis=(0,1))[0])-float(frames[0].mean(axis=(0,1))[0]),100)
                finally:
                    capture.release()
                self.assertIn('rgb(20, 20, 220)',runner.browser_lane.submit(lambda:runner.browser_tools.page.locator('body').get_attribute('style')).result())
            finally:
                runner.close()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_WINDOW_TEST') == '1', 'Requires visible Windows desktop')
    def test_maximized_chromium_capture_keeps_window_geometry_and_native_pixels(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.tools import Tools, list_windows
        from game_agent import windows
        windows.enable_dpi()
        with tempfile.TemporaryDirectory() as folder:
            tools = Tools(folder,threading.Event(),Mock(return_value=True),Mock())
            tools.allow_screen = True
            tools.mode = 'routine'
            try:
                existing_handles = {row['handle'] for row in list_windows(os.getpid())}
                tools.ensure_browser()
                tools.page.set_content('<title>Local Desk Capture Probe</title><body style="margin:0;background:#fafafa">'
                    '<section style="background:#135c43;color:white;padding:50px;font:36px sans-serif">'
                    'ORBIT READY<br>Capture remains maximized</section></body>')
                tools.page.bring_to_front()
                new_windows = [row for row in list_windows(os.getpid()) if row['handle'] not in existing_handles]
                self.assertEqual(len(new_windows),1)
                tools.window = new_windows[0]
                handle = tools.window['handle']
                windows.user32.ShowWindow(handle,3)
                tools.focus_window()
                baseline = windows.window_rect(handle)
                visible = tools.capture_region()
                for attempt in range(2):
                    result = tools.execute(dict(message='Inspect screen',tool='desktop_capture',arguments={},risk='routine'))
                    self.assertTrue(windows.user32.IsZoomed(handle))
                    self.assertEqual(windows.window_rect(handle),baseline)
                    self.assertEqual(result.image.size,(visible.width,visible.height))
                    pixels = list(result.image.resize((100,100)).getdata())
                    self.assertGreater(sum(green > red+30 and green > blue+10 for red,green,blue in pixels),100)
                result.image.save(Path(__file__).resolve().parents[1]/'desktop_agent'/'desktop-window-probe.png')
                result = tools.execute(dict(message='Record motion',tool='desktop_record',arguments={'seconds':1,'fps':3},risk='routine'))
                self.assertTrue(Path(result.video).is_file())
                self.assertFalse(result.interrupted)
                self.assertEqual(windows.window_rect(handle),baseline)
            finally:
                tools.close()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_BROWSER_TEST') == '1', 'Requires installed Chromium')
    def test_real_browser_video_records_changing_frames(self):
        import cv2
        import threading
        from unittest.mock import Mock
        from desktop_agent.tools import Tools
        with tempfile.TemporaryDirectory() as folder:
            tools = Tools(folder,threading.Event(),Mock(return_value=True),Mock(),headless=True)
            tools.mode = 'routine'
            try:
                tools.ensure_browser()
                tools.page.set_content('<style>@keyframes change {from {background:#ed3030} to {background:#3030ed}}'
                    'body {animation:change 1s linear infinite alternate}</style><h1>Video fixture</h1>')
                result = tools.execute(dict(message='Observe animation',tool='browser_record',arguments={'seconds':1,'fps':4},risk='routine'))
                capture = cv2.VideoCapture(result.video)
                try:
                    frames = []
                    while True:
                        success, frame = capture.read()
                        if not success:
                            break
                        frames.append(frame)
                    self.assertEqual(len(frames),4)
                    self.assertGreater(abs(float(frames[0].mean(axis=(0,1))[0])-float(frames[-1].mean(axis=(0,1))[0])),5)
                finally:
                    capture.release()
                self.assertFalse(result.interrupted)
            finally:
                tools.close()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_BROWSER_TEST') == '1', 'Requires installed Chromium')
    def test_browser_key_queue_rechecks_each_new_focused_target_and_cancels(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.tools import Tools
        from game_agent.core import Halted
        with tempfile.TemporaryDirectory() as folder:
            approval = Mock(return_value=False)
            stopped = threading.Event()
            tools = Tools(folder,stopped,approval,Mock(),headless=True)
            tools.mode = 'routine'
            try:
                tools.ensure_browser()
                blank = json.loads(tools.execute(dict(message='Read',tool='browser_read',arguments={},risk='routine')).text)
                self.assertEqual(blank['scope'],'managed_browser_only')
                self.assertIn('desktop_capture',blank['notice'])
                tools.page.set_content('<input id="first"><button id="next" onclick="this.textContent=\'Ready\'">Next</button>'
                    '<button id="danger" onclick="this.textContent=\'Gone\'">Delete account</button><input id="secret" type="password">')
                tools.page.locator('#first').focus()
                def call(steps):
                    return tools.execute(dict(message='Use controls',tool='browser_key_queue',arguments={'steps':steps},risk='routine'))
                call([{'key':'Tab','delay_ms':10},{'key':'Enter','delay_ms':20}])
                self.assertEqual(tools.page.locator('#next').inner_text(),'Ready')
                approval.assert_not_called()
                with self.assertRaises(Halted):
                    call([{'key':'Tab','delay_ms':0},{'key':'Enter','delay_ms':0}])
                approval.assert_called_once()
                self.assertEqual(tools.page.locator('#danger').inner_text(),'Delete account')
                tools.page.locator('#secret').focus()
                with self.assertRaises(Halted):
                    call([{'key':'Control+v','delay_ms':0}])
                self.assertEqual(tools.page.locator('#secret').input_value(),'')
                tools.page.locator('#first').focus()
                tools.notify = lambda kind,value:stopped.set() if kind == 'status' else None
                with self.assertRaises(Halted):
                    call([{'key':'Tab','delay_ms':5000}])
                self.assertTrue(tools.page.locator('#first').evaluate('element => element === document.activeElement'))
            finally:
                tools.close()

    @unittest.skipUnless(os.environ.get('DESKTOP_AGENT_BROWSER_TEST') == '1', 'Requires installed Chromium')
    def test_real_browser_tools_capture_independently_and_confirm_risk(self):
        import threading
        from unittest.mock import Mock
        from desktop_agent.tools import Tools
        from game_agent.core import Halted
        with tempfile.TemporaryDirectory() as folder:
            approval = Mock(return_value=False)
            tools = Tools(folder,threading.Event(),approval,Mock(),headless=True)
            tools.mode = 'routine'
            try:
                tools.ensure_browser()
                tools.page.set_content('<html><body><h1>Local test</h1><input id="query" type="search">'
                    '<button id="next" type="button" onclick="document.querySelector(\'h1\').textContent=\'Advanced\'">Next page</button>'
                    '<button id="delete" onclick="document.querySelector(\'h1\').textContent=\'Deleted\'">Delete account</button></body></html>')
                def call(tool,arguments=None):
                    return tools.execute(dict(message='Use page control',tool=tool,arguments=arguments or {},risk='routine'))
                read = call('browser_read')
                self.assertIsNone(read.image)
                self.assertIn('Local test',read.text)
                self.assertIsNone(call('browser_type',{'selector':'#query','text':'Test text'}).image)
                self.assertEqual(tools.page.locator('#query').input_value(),'Test text')
                call('browser_key',{'key':'Enter'})
                call('browser_click',{'selector':'#next'})
                self.assertEqual(tools.page.locator('h1').inner_text(),'Advanced')
                approval.assert_not_called()
                with self.assertRaises(Halted):
                    call('browser_click',{'selector':'#delete'})
                self.assertEqual(tools.page.locator('h1').inner_text(),'Advanced')
                approval.return_value = True
                capture = call('browser_capture')
                self.assertEqual(capture.image.size,(1280,800))
                self.assertGreater(len(set(capture.image.resize((50,50)).getdata())),1)
            finally:
                tools.close()


if __name__ == '__main__':
    unittest.main()