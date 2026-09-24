import json
from pathlib import Path
import shutil
import uuid
from zipfile import ZipFile
from xml.etree import ElementTree

from PIL import Image, ImageDraw, ImageOps
from desktop_agent.tools import ToolResult


MAX_FILE_BYTES = 100*1024*1024
IMAGE_TYPES = {'.png','.jpg','.jpeg','.webp','.bmp','.gif'}
VIDEO_TYPES = {'.mp4','.avi','.mkv','.webm','.mov'}
TEXT_TYPES = {'.txt','.md','.csv','.json','.jsonl','.py','.js','.ts','.html','.css','.xml','.yaml','.yml','.log','.sql'}


def save_attachments(paths, directory):
    if len(paths) > 8:
        raise ValueError('Maximum 8 attachments per message')
    sources = [Path(path).resolve() for path in paths]
    for source in sources:
        if not source.is_file() or source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('File missing or larger than 100MB: '+source.name)
    if sum(source.stat().st_size for source in sources) > 200*1024*1024:
        raise ValueError('Attachments exceed 200MB per message')
    folder = Path(directory)/'attachments'
    folder.mkdir(parents=True,exist_ok=True)
    saved = []
    try:
        for source in sources:
            destination = folder/(uuid.uuid4().hex+'_'+source.name)
            shutil.copyfile(source,destination)
            saved.append({'name':source.name,'path':str(destination.resolve()),'bytes':destination.stat().st_size,
                          'type':source.suffix.lower() or 'binary'})
    except Exception:
        for item in saved:
            Path(item['path']).unlink(missing_ok=True)
        raise
    return saved


def registered_path(events, value):
    requested = Path(value).resolve()
    allowed = []
    for event in events:
        metadata = event.get('metadata',{})
        allowed.extend(item['path'] for item in metadata.get('attachments',[]))
        allowed.extend(metadata[key] for key in ('image','video') if metadata.get(key))
    if requested not in [Path(path).resolve() for path in allowed] or not requested.is_file():
        raise ValueError('File is not an attachment/artifact of this session')
    return requested


def read_attachment(path, offset, limit):
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError('Encrypted PDF: provide an unlocked copy')
        chunks, length = [],0
        for index, page in enumerate(reader.pages):
            chunk = f'\n[Page {index+1}]\n'+(page.extract_text() or '')
            chunks.append(chunk)
            length += len(chunk)
            if length > offset+limit:
                break
        text = ''.join(chunks)
        complete = index+1 == len(reader.pages) if reader.pages else True
    elif suffix == '.docx':
        with ZipFile(path) as archive:
            info = archive.getinfo('word/document.xml')
            if info.file_size > 20*1024*1024:
                raise ValueError('DOCX text is too large')
            root = ElementTree.fromstring(archive.read(info))
        namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        text = '\n'.join(''.join(node.text or '' for node in paragraph.iter(namespace+'t')) for paragraph in root.iter(namespace+'p'))
        complete = True
    elif suffix in TEXT_TYPES:
        data = path.read_bytes()
        encodings = ('utf-16',) if data.startswith((b'\xff\xfe',b'\xfe\xff')) else ('utf-8-sig','cp949')
        for encoding in encodings:
            try:
                text = data.decode(encoding)
                break
            except UnicodeError:
                continue
        else:
            raise ValueError('Unsupported text encoding')
        complete = True
    else:
        return ToolResult(json.dumps({'path':str(path),'bytes':path.stat().st_size,
            'type':suffix,'notice':'No text parser. For images/video use file_view; binary files are never executed.'}))
    end = offset+limit
    return ToolResult(json.dumps({'path':str(path),'offset':offset,'text':text[offset:end],
        'next_offset':None if complete and end >= len(text) else end},ensure_ascii=False))


def view_attachment(path):
    if path.suffix.lower() in IMAGE_TYPES:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        return ToolResult('Explicitly viewed attachment: '+str(path),image)
    if path.suffix.lower() in VIDEO_TYPES:
        import cv2
        capture = cv2.VideoCapture(str(path))
        sheet = Image.new('RGB',(1280,768),'#eef1f4')
        draw = ImageDraw.Draw(sheet)
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or count <= 0 or fps <= 0:
                raise ValueError('Video cannot be decoded')
            for index in range(4):
                frame_index = round((count-1)*index/3)
                capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
                success,frame = capture.read()
                if not success:
                    raise ValueError('Video frame cannot be decoded')
                image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                image.thumbnail((640,360))
                left,top = index%2*640,index//2*384
                sheet.paste(image,(left,top+24))
                draw.text((left+8,top+6),f'{frame_index/fps:.2f}s',fill='#253342')
        finally:
            capture.release()
        return ToolResult('Explicit video sample view (4 frames only, no audio): '+str(path),sheet)
    raise ValueError('Visual preview supports images/videos; use file_read for documents')