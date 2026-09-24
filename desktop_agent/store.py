import json
from pathlib import Path
import shutil
import sqlite3
import uuid


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / 'sessions.sqlite3'
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL,
                    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE INDEX IF NOT EXISTS event_session ON events(session_id, id);
                CREATE TABLE IF NOT EXISTS context_selections (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    selection TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS context_summaries (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL);
            ''')

    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    def create(self, title='New conversation'):
        identifier = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute('INSERT INTO sessions(id,title) VALUES (?,?)', (identifier, title[:160]))
        return identifier

    def sessions(self):
        with self.connect() as connection:
            return [dict(row) for row in connection.execute('SELECT * FROM sessions ORDER BY updated DESC, rowid DESC')]

    def rename(self, identifier, title):
        title = title.strip()
        if not title:
            raise ValueError('Session title is empty')
        with self.connect() as connection:
            connection.execute('UPDATE sessions SET title=?,updated=CURRENT_TIMESTAMP WHERE id=?', (title[:160], identifier))

    def append(self, identifier, role, content, **metadata):
        if role not in ('user', 'assistant', 'tool', 'system'):
            raise ValueError('Invalid event role')
        with self.connect() as connection:
            cursor = connection.execute('INSERT INTO events(session_id,role,content,metadata) VALUES (?,?,?,?)',
                (identifier, role, content, json.dumps(metadata, ensure_ascii=False)))
            connection.execute('UPDATE sessions SET updated=CURRENT_TIMESTAMP WHERE id=?', (identifier,))
            return cursor.lastrowid

    def events(self, identifier):
        with self.connect() as connection:
            rows = connection.execute('SELECT * FROM events WHERE session_id=? ORDER BY id', (identifier,)).fetchall()
        return [dict(row, metadata=json.loads(row['metadata'])) for row in rows]

    def artifact_directory(self, identifier):
        with self.connect() as connection:
            if connection.execute('SELECT 1 FROM sessions WHERE id=?', (identifier,)).fetchone() is None:
                raise ValueError('Unknown session')
        directory = self.directory / 'artifacts' / identifier
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_context_selection(self, identifier, selection):
        with self.connect() as connection:
            connection.execute('INSERT OR REPLACE INTO context_selections(session_id,selection) VALUES (?,?)',
                               (identifier,json.dumps(selection)))

    def context_selection(self, identifier):
        with self.connect() as connection:
            row = connection.execute('SELECT selection FROM context_selections WHERE session_id=?',(identifier,)).fetchone()
        return json.loads(row['selection']) if row else {}

    def save_context_summary(self, identifier, summary):
        with self.connect() as connection:
            connection.execute('INSERT OR REPLACE INTO context_summaries(session_id,summary) VALUES (?,?)',
                               (identifier,json.dumps(summary,ensure_ascii=False)))

    def context_summary(self, identifier):
        with self.connect() as connection:
            row=connection.execute('SELECT summary FROM context_summaries WHERE session_id=?',(identifier,)).fetchone()
        return json.loads(row['summary']) if row else None

    def export(self, identifier, path):
        payload = {'session': next(row for row in self.sessions() if row['id'] == identifier),
                   'events': self.events(identifier)}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def history_text(self, identifier):
        return '\n'.join(json.dumps(event,ensure_ascii=False) for event in self.events(identifier))

    def record_page(self, identifier, event_id, offset=0, limit=6000):
        if type(event_id) is not int or event_id<1 or type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=64000:
            raise ValueError('Invalid session record page')
        with self.connect() as connection:
            row=connection.execute('SELECT * FROM events WHERE session_id=? AND id=?',(identifier,event_id)).fetchone()
        if row is None:
            raise ValueError('Record not found in this session')
        metadata=json.loads(row['metadata'])
        content=row['content']
        end=min(len(content),offset+limit)
        return dict(event_id=event_id,role=row['role'],
                    metadata={key:metadata[key] for key in ('tool','status','call_id','origin_call_id','job_id','requested_tool') if key in metadata},
                    text=content[offset:end],offset=offset,next_offset=end if end<len(content) else None,
                    total_characters=len(content),offset_unit='characters')

    def archive_history(self, identifier):
        path = self.artifact_directory(identifier)/'history.jsonl'
        path.write_text(self.history_text(identifier),encoding='utf-8')
        return path

    def message(self, identifier, event_id):
        event = next((event for event in self.events(identifier) if event['id'] == event_id),None)
        if event is None or event['role'] not in ('user','assistant'):
            raise ValueError('Select a user message or assistant reply')
        content = event['content']
        if event['role'] == 'assistant' and not event['metadata'].get('partial'):
            try:
                action = json.loads(content)
            except ValueError:
                action = None
            if isinstance(action,dict) and 'tool' in action:
                if action['tool'] != 'finish':
                    raise ValueError('Tool invocation records cannot be edited as replies')
                content = action['message']
        return event,content

    def edit_message(self, identifier, event_id, content):
        event, previous = self.message(identifier,event_id)
        limit = 12000 if event['role'] == 'user' else 5000
        if not isinstance(content,str) or not content.strip() or len(content) > limit:
            raise ValueError(f'Message must contain 1..{limit} characters')
        stored = content
        if event['role'] == 'assistant' and not event['metadata'].get('partial'):
            try:
                action = json.loads(event['content'])
            except ValueError:
                action = None
            if isinstance(action,dict) and action.get('tool') == 'finish':
                action['message'] = content
                stored = json.dumps(action,ensure_ascii=False)
            else:
                stored = json.dumps(dict(message=content,tool='finish',arguments={},risk='routine'),ensure_ascii=False)
        metadata = dict(event['metadata'],edited=True)
        metadata.pop('metrics',None)
        with self.connect() as connection:
            connection.execute('DELETE FROM context_selections WHERE session_id=?',(identifier,))
            connection.execute('UPDATE events SET content=?,metadata=? WHERE session_id=? AND id=?',
                               (stored,json.dumps(metadata,ensure_ascii=False),identifier,event_id))
            connection.execute('UPDATE sessions SET updated=CURRENT_TIMESTAMP WHERE id=?',(identifier,))
        self.archive_history(identifier)

    def delete_message(self, identifier, event_id):
        event, content = self.message(identifier,event_id)
        with self.connect() as connection:
            connection.execute('DELETE FROM context_selections WHERE session_id=?',(identifier,))
            if event['role'] == 'user':
                next_user = connection.execute("SELECT MIN(id) FROM events WHERE session_id=? AND role='user' AND id>?",
                                               (identifier,event_id)).fetchone()[0]
                connection.execute('DELETE FROM events WHERE session_id=? AND id>=? AND (? IS NULL OR id<?)',
                                   (identifier,event_id,next_user,next_user))
            else:
                connection.execute('DELETE FROM events WHERE session_id=? AND id=?',(identifier,event_id))
            connection.execute('UPDATE sessions SET updated=CURRENT_TIMESTAMP WHERE id=?',(identifier,))
        self.archive_history(identifier)

    def replay_request(self, identifier, event_id):
        self.message(identifier,event_id)
        requests = [event for event in self.events(identifier) if event['id'] <= event_id and event['role'] == 'user']
        if not requests:
            raise ValueError('No user request precedes this reply')
        return requests[-1]

    def prepare_replay(self, identifier, event_id):
        request = self.replay_request(identifier,event_id)
        source_directory = self.artifact_directory(identifier).resolve()
        attachments = []
        for item in request['metadata'].get('attachments',[]):
            path = Path(item['path']).resolve()
            if not path.is_relative_to(source_directory) or not path.is_file():
                raise ValueError('Replay attachment is missing or outside its session')
            attachments.append(str(path))
        with self.connect() as connection:
            connection.execute('DELETE FROM context_selections WHERE session_id=?',(identifier,))
            connection.execute('DELETE FROM events WHERE session_id=? AND id>=?',(identifier,request['id']))
            connection.execute('UPDATE sessions SET updated=CURRENT_TIMESTAMP WHERE id=?',(identifier,))
        self.archive_history(identifier)
        return identifier,request['content'],attachments

    def delete_session(self, identifier):
        with self.connect() as connection:
            if connection.execute('SELECT 1 FROM sessions WHERE id=?',(identifier,)).fetchone() is None:
                raise ValueError('Unknown session')
            root = (self.directory/'artifacts').resolve()
            directory = root/identifier
            if directory.resolve().parent != root or directory.is_symlink():
                raise ValueError('Invalid session directory')
            if directory.exists():
                shutil.rmtree(directory)
            connection.execute('DELETE FROM sessions WHERE id=?',(identifier,))


def conversation_turns(events):
    turns = []
    for event in events:
        if event['role'] == 'user':
            turns.append([])
        if turns and not event.get('metadata', {}).get('partial'):
            turns[-1].append({'role': event['role'], 'content': event['content']})
    return turns