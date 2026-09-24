from copy import deepcopy
import json

from desktop_agent.protocol import compact_call,normalize_call


def raw_reference(event_id):
    return dict(tool='session_record',arguments=dict(event_id=event_id,offset=0,limit=6000))


def parsed_result(event):
    try:
        return json.loads(event['content'])
    except (ValueError,TypeError):
        return event['content']


def text_excerpt(text, limit):
    if len(text) <= limit:
        return text
    marker = '\n[... omitted ...]\n'
    head = (limit-len(marker))*2//3
    return text[:head]+marker+text[-(limit-head-len(marker)):]


def preview(value, limit=600):
    if isinstance(value,str):
        return value if len(value)<=limit else {'excerpt':text_excerpt(value,limit),'total_characters':len(value)}
    if isinstance(value,dict):
        return {key:preview(item,limit) for key,item in value.items()}
    if isinstance(value,list):
        return [preview(item,limit) for item in value[:6]]+([{'omitted_items':len(value)-6}] if len(value)>6 else [])
    return value


def linked_events(events):
    rows = []
    calls,pending,jobs = {},{},{}
    for source in events:
        event = dict(source,metadata=dict(source.get('metadata',{})))
        metadata = event['metadata']
        if event['role']=='user':
            pending.clear()
        if event['role']=='assistant' and not metadata.get('partial'):
            try:
                action = normalize_call(json.loads(event.get('raw_content',event['content'])))
                if action['tool']!='finish':
                    identifier = metadata.get('call_id') or ('legacy_'+str(event['id']) if event.get('id') is not None else '')
                    if identifier:
                        metadata['call_id'] = identifier
                        calls[identifier] = dict(event_id=event.get('id'),action=action,call=compact_call(action))
                        pending[identifier] = calls[identifier]
            except (ValueError,TypeError,KeyError):
                pass
        elif event['role'] in ('tool','system'):
            identifier = metadata.get('call_id','')
            result = parsed_result(event)
            if event['role']=='tool' and not identifier:
                tool = metadata.get('requested_tool',metadata.get('tool'))
                job_id = metadata.get('job_id')
                candidates = [key for key,call in pending.items() if call['action']['tool']==tool
                              or (call['action']['tool']=='job_result' and job_id
                                  and call['action']['arguments'].get('job_id')==job_id)]
                if len(candidates)==1:
                    identifier = candidates[0]
                elif not candidates and job_id:
                    identifier = jobs.get(job_id,'')
                if identifier:
                    metadata['call_id'] = identifier
                    metadata['link_basis'] = 'legacy_unambiguous'
            call = calls.get(identifier)
            if call:
                metadata['call_event_id'] = call['event_id']
                metadata['call_tool'] = call['call']['tool']
                metadata['call_internal_tool'] = call['action']['tool']
                metadata['call_arguments'] = call['call']['arguments']
                if event['role']=='tool':
                    pending.pop(identifier,None)
                    if call['action']['tool']=='job_start' and isinstance(result,dict) and result.get('job_id'):
                        jobs[result['job_id']] = identifier
            rows.append(event)
            continue
        rows.append(event)
    return rows


def execution_status(event, result=None):
    metadata = event.get('metadata',{})
    status = metadata.get('status','unknown')
    if status in ('error','stopped','blocked'):
        return {'error':'failed','stopped':'cancelled','blocked':'blocked'}[status]
    result = parsed_result(event) if result is None else result
    tool = metadata.get('tool','')
    if tool.startswith('terminal_') or status=='terminal':
        return result.get('status','unknown') if isinstance(result,dict) else 'unknown'
    if tool=='workspace_apply_patch' or status=='file_change':
        return result.get('status','unknown') if isinstance(result,dict) else 'unknown'
    if tool=='job_start' and isinstance(result,dict):
        return 'ready_to_collect' if result.get('status')=='completed' else result.get('status','unknown')
    if tool=='job_cancel':
        return 'cancel_requested'
    return 'returned' if status=='delivered' else status


def result_record(event):
    metadata = event.get('metadata',{})
    result = deepcopy(parsed_result(event))
    tool,status = metadata.get('tool'),metadata.get('status')
    if event['role']=='system' and status=='approval' and isinstance(result,dict) and result.get('mode') in ('automatic','confirmed'):
        return None
    record = dict(kind='tool_result' if event['role']=='tool' else 'tool_update' if status in ('terminal','file_change') else 'record')
    for key in ('tool','status','call_id','origin_call_id','requested_tool','call_event_id','link_basis'):
        if metadata.get(key) is not None:
            record[key] = metadata[key]
    if event.get('id') is not None:
        record['id'] = event['id']
        record['raw_ref'] = raw_reference(event['id'])
    if metadata.get('call_tool'):
        record['call'] = dict(tool=metadata['call_tool'],arguments=preview(metadata.get('call_arguments',{})))
        if metadata.get('call_event_id') is not None:
            record['call']['raw_ref'] = raw_reference(metadata['call_event_id'])
    record['execution_status'] = execution_status(event,result)
    summary = {}
    if isinstance(result,dict):
        for key in ('path','sha256','before_sha256','after_sha256','change_id','execution_id','job_id','status','exit_code',
                    'output_bytes','total_lines','scanned','truncated','diff_truncated','ready','offset','next_offset','error','reason','url','title'):
            if key in result:
                summary[key] = preview(result[key],300)
        for key in ('matches','controls','files','results'):
            if isinstance(result.get(key),list):
                summary[key+'_count'] = len(result[key])
        for key in ('text','diff'):
            if isinstance(result.get(key),str):
                summary[key+'_characters'] = len(result[key])
    elif isinstance(result,list):
        summary['item_count'] = len(result)
    elif isinstance(result,str):
        summary['text_characters'] = len(result)
        if status in ('error','stopped','blocked'):
            summary['error'] = text_excerpt(result,600)
    if metadata.get('error'):
        summary['error'] = text_excerpt(str(metadata['error']),600)
    record['summary'] = summary
    shortened = False
    if status=='delivered' and tool not in ('file_read','session_history','session_record'):
        if isinstance(result,dict):
            if tool in ('desktop_capture','desktop_screen_capture','desktop_click'):
                result.pop('notice',None)
                if len(result.get('bounds',[]))==4:
                    result.pop('width',None);result.pop('height',None)
            limit = 1200 if result.get('scope')=='managed_browser_only' else 1800
            for key in ('text','diff'):
                if isinstance(result.get(key),str) and len(result[key])>limit:
                    result[key] = text_excerpt(result[key],limit)
                    shortened = True
            for key in ('command','cwd'):
                if isinstance(result.get(key),str) and len(result[key])>600:
                    result[key] = text_excerpt(result[key],600)
                    shortened = True
            for key in ('matches','controls','files','results'):
                if isinstance(result.get(key),list) and len(result[key])>12:
                    result[key] = result[key][:12]
                    shortened = True
        elif isinstance(result,str) and len(result)>1800:
            result = text_excerpt(result,1800)
            shortened = True
        elif isinstance(result,list) and len(result)>12:
            result = result[:12]
            shortened = True
    record['result'] = result
    if shortened:
        record.update(excerpt=True,excerpt_note='Preview only. Use raw_ref to read this exact stored result; do not rerun the tool to recover omitted text.')
    for key in ('image','video','attachments','job_id','error','edited','api_request'):
        if key in metadata and not (isinstance(result,dict) and result.get(key)==metadata[key]) and not (key=='error' and result==metadata[key]):
            record[key] = metadata[key]
    return record