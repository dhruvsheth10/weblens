"""Opt-in, bounded local collection of actual Reddit image bytes and observations.

Nothing here labels images from their model scores or subreddit membership.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urlparse
import uuid
from PIL import Image
from .image_io import decode_image, MAX_BYTES, sigmoid

DEFAULT_ROOT=Path(__file__).resolve().parents[1]/'artifacts/reddit-recordings'
ID=re.compile(r'^[a-f0-9]{32}$')
IMAGE_HOSTS={'i.redd.it','preview.redd.it','external-preview.redd.it'}
POST_HOSTS={'reddit.com','www.reddit.com','old.reddit.com','new.reddit.com'}

def safe_url(value, hosts):
    if not isinstance(value,str) or len(value)>4096:raise ValueError('Invalid source URL.')
    url=urlparse(value)
    if url.scheme!='https' or url.hostname not in hosts or url.username or url.password or url.port not in {None,443}:
        raise ValueError('Unsupported source URL.')
    return value

def prediction(value, expected):
    if not isinstance(value,dict):raise ValueError('Missing model observation.')
    if any(value.get(k)!=expected[k] for k in ['model','modelHash','pipeline']):raise ValueError('Recording detector identity mismatch.')
    for key in ['score','logit']:
        if isinstance(value.get(key),bool) or not isinstance(value.get(key),(int,float)) or not math.isfinite(value[key]):
            raise ValueError('Non-finite model observation.')
    if not 0<=value['score']<=1 or abs(sigmoid(value['logit'])-value['score'])>1e-7:
        raise ValueError('Invalid model score/logit observation.')
    if value.get('variant')!='original':raise ValueError('Only actual fetched bytes can be recorded.')
    if expected['model'].startswith('community-') and value.get('uncertainThreshold')!=expected.get('uncertainThreshold'):
        raise ValueError('Uncertainty policy identity mismatch.')
    return value

class RecordingStore:
    def __init__(self,root=DEFAULT_ROOT):
        self.root=Path(root);self.lock=threading.Lock()

    @contextmanager
    def connect(self):
        if self.root.is_symlink():raise ValueError('Recording root cannot be a symlink.')
        self.root.mkdir(parents=True,exist_ok=True)
        objects=self.root/'objects'
        if objects.is_symlink():raise ValueError('Recording objects cannot be a symlink.')
        objects.mkdir(exist_ok=True)
        path=self.root/'records.sqlite3'
        if path.is_symlink():raise ValueError('Recording index cannot be a symlink.')
        db=sqlite3.connect(path);db.row_factory=sqlite3.Row
        db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created REAL, active INTEGER, captures INTEGER DEFAULT 0, bytes INTEGER DEFAULT 0, max_captures INTEGER, max_bytes INTEGER)')
        db.execute('CREATE TABLE IF NOT EXISTS captures (id TEXT PRIMARY KEY, session TEXT, created REAL, pair_key TEXT, metadata TEXT, preview TEXT, original TEXT, label TEXT DEFAULT "unknown", category TEXT DEFAULT "", evidence TEXT DEFAULT "", UNIQUE(session,pair_key))')
        db.commit()
        try:
            with db:yield db
        finally:db.close()

    @staticmethod
    def check_id(value):
        if not isinstance(value,str) or not ID.fullmatch(value):raise ValueError('Invalid recording ID.')

    def start(self,max_captures=1000,max_bytes=2*1024**3):
        with self.lock,self.connect() as db:
            db.execute('UPDATE sessions SET active=0')
            key=uuid.uuid4().hex
            db.execute('INSERT INTO sessions(id,created,active,max_captures,max_bytes) VALUES(?,?,1,?,?)',(key,time.time(),max_captures,max_bytes))
            return dict(db.execute('SELECT * FROM sessions WHERE id=?',(key,)).fetchone())

    def sessions(self):
        # Merely inspecting status does not create a recording directory/database.
        if not (self.root/'records.sqlite3').is_file():return []
        with self.lock,self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM sessions ORDER BY created DESC LIMIT 30')]

    def stop(self,key):
        self.check_id(key)
        with self.lock,self.connect() as db:
            if db.execute('UPDATE sessions SET active=0 WHERE id=?',(key,)).rowcount!=1:raise ValueError('Unknown recording session.')

    def validate_metadata(self,meta,expected,has_original):
        if not isinstance(meta,dict):raise ValueError('Invalid capture metadata.')
        if meta.get('schemaVersion')!=1:raise ValueError('Unknown capture schema.')
        for name in ['preview','original']:
            part=meta.get(name)
            if name=='original' and not has_original:
                if part is not None:raise ValueError('Original metadata has no matching bytes.')
                continue
            if not isinstance(part,dict):raise ValueError(f'Missing {name} metadata.')
            safe_url(part.get('url'),IMAGE_HOSTS);prediction(part.get('prediction'),expected)
        if meta.get('selectedPrediction') is not None:prediction(meta['selectedPrediction'],expected)
        if meta.get('postUrl'):
            url=safe_url(meta['postUrl'],POST_HOSTS)
            if not re.search(r'/(comments|gallery)/',urlparse(url).path):raise ValueError('Expected a Reddit post permalink.')
        if meta.get('subreddit') is not None and (not isinstance(meta['subreddit'],str) or len(meta['subreddit'])>128):raise ValueError('Invalid subreddit.')
        if meta.get('policy') not in {'preview-first','prefer-original','displayed'}:raise ValueError('Unknown fetch policy.')
        if not isinstance(meta.get('threshold'),(int,float)) or isinstance(meta['threshold'],bool) or not math.isfinite(meta['threshold']) or not 0<=meta['threshold']<=1:raise ValueError('Invalid operating point.')
        # Labels/evidence are assigned in a separate review action, never imported from a page.
        if any(k in meta for k in ['label','evidence','category']):raise ValueError('Labels require an explicit review action.')
        # Reject non-finite values in nested observations before writing objects.
        json.dumps(meta,allow_nan=False)
        return meta

    def image_info(self,data):
        if not data or len(data)>MAX_BYTES:raise ValueError('Recording image must be 1 byte to 20 MB.')
        _,size=decode_image(data)
        with Image.open(io.BytesIO(data)) as im:
            suffix={'JPEG':'jpg','MPO':'jpg','PNG':'png','WEBP':'webp','AVIF':'avif'}[im.format]
        return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'width':size[0],'height':size[1],'file':hashlib.sha256(data).hexdigest()+'.'+suffix}

    def write_object(self,info,data):
        path=self.root/'objects'/info['file']
        if path.is_symlink():raise ValueError('Recording image cannot be a symlink.')
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Stored image checksum mismatch.')
            return
        temp=path.with_name('.'+uuid.uuid4().hex)
        try:
            with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(temp,path)
        finally:temp.unlink(missing_ok=True)

    def add(self,session,meta,preview,original,expected):
        self.check_id(session);self.validate_metadata(meta,expected,original is not None)
        preview_info=self.image_info(preview);original_info=self.image_info(original) if original is not None else None
        pair_key=hashlib.sha256(json.dumps([preview_info['sha256'],original_info['sha256'] if original_info else None,meta.get('postUrl')],separators=(',',':')).encode()).hexdigest()
        byte_count=len(preview)+(len(original) if original is not None else 0)
        with self.lock,self.connect() as db:
            state=db.execute('SELECT * FROM sessions WHERE id=?',(session,)).fetchone()
            if state is None or not state['active']:raise ValueError('Recording session is stopped or missing.')
            existing=db.execute('SELECT id FROM captures WHERE session=? AND pair_key=?',(session,pair_key)).fetchone()
            if existing:return {'id':existing['id'],'sessionId':session,'duplicate':True}
            if state['captures']>=state['max_captures'] or state['bytes']+byte_count>state['max_bytes']:
                db.execute('UPDATE sessions SET active=0 WHERE id=?',(session,));db.commit()
                raise ValueError('Recording limit reached; session stopped. Start a new session to continue.')
            self.write_object(preview_info,preview)
            if original_info:self.write_object(original_info,original)
            stored={**meta,'observations':'client-observed local model outputs; not ground-truth labels','preview':{**meta['preview'],**preview_info},
                    'original':{**meta['original'],**original_info} if original_info else None}
            key=uuid.uuid4().hex
            db.execute('INSERT INTO captures(id,session,created,pair_key,metadata,preview,original) VALUES(?,?,?,?,?,?,?)',
                       (key,session,time.time(),pair_key,json.dumps(stored,allow_nan=False),preview_info['file'],original_info['file'] if original_info else None))
            db.execute('UPDATE sessions SET captures=captures+1,bytes=bytes+? WHERE id=?',(byte_count,session))
            return {'id':key,'sessionId':session,'duplicate':False}

    def captures(self,session,offset=0,limit=50):
        self.check_id(session)
        if not (self.root/'records.sqlite3').is_file():raise ValueError('Unknown recording session.')
        with self.lock,self.connect() as db:
            rows=db.execute('SELECT * FROM captures WHERE session=? ORDER BY created DESC LIMIT ? OFFSET ?',(session,limit,offset))
            return [{k:r[k] for k in ['id','session','created','label','category','evidence']}|{'metadata':json.loads(r['metadata'])} for r in rows]

    def label(self,session,capture,label,category,evidence):
        self.check_id(session);self.check_id(capture)
        if label not in {'unknown','real','ai'} or not isinstance(category,str) or len(category)>128 or not isinstance(evidence,str) or len(evidence)>4000:raise ValueError('Invalid review fields.')
        if label!='unknown' and not evidence.strip():raise ValueError('Provide evidence before assigning a real/AI label.')
        with self.lock,self.connect() as db:
            if db.execute('UPDATE captures SET label=?,category=?,evidence=? WHERE id=? AND session=?',(label,category,evidence,capture,session)).rowcount!=1:raise ValueError('Unknown capture.')

    def image_path(self,session,capture,kind):
        self.check_id(session);self.check_id(capture)
        if kind not in {'preview','original'}:raise ValueError('Unknown image role.')
        with self.lock,self.connect() as db:
            row=db.execute('SELECT * FROM captures WHERE id=? AND session=?',(capture,session)).fetchone()
            if row is None or not row[kind]:raise ValueError('Image unavailable.')
            path=self.root/'objects'/row[kind]
            if path.is_symlink() or not path.is_file():raise ValueError('Recording object unavailable.')
            return path
