import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from backend.app import app as production_app
from fastapi import FastAPI
from backend.recording_api import router

# Retain archive module contract tests in an isolated app. Production no longer
# mounts this router and cannot accept captures, even from older extensions.
app=FastAPI()
app.include_router(router)
from backend.recordings import RecordingStore

PROFILE=json.loads((Path(__file__).parents[1]/'models/community-ferret-profile.json').read_text())
OBSERVATION={k:PROFILE[k] for k in ['model','modelHash','pipeline','uncertainThreshold']}|{'score':.5,'logit':0.,'variant':'original'}
def image(color):
    buffer=io.BytesIO();Image.new('RGB',(256,256),color).save(buffer,'PNG');return buffer.getvalue()
PREVIEW=image('red');ORIGINAL=image('blue')
def metadata():
    return {'schemaVersion':1,'policy':'preview-first','threshold':.5,'postUrl':'https://www.reddit.com/r/pics/comments/abc123/example/',
            'subreddit':'r/pics','preview':{'url':'https://preview.redd.it/abc123def456.png?width=640','prediction':copy.deepcopy(OBSERVATION)},
            'original':{'url':'https://i.redd.it/abc123def456.png','prediction':copy.deepcopy(OBSERVATION)}}

class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'recordings';self.store=RecordingStore(self.root)
    def tearDown(self):self.temp.cleanup()
    def add(self,session,meta=None):return self.store.add(session,meta or metadata(),PREVIEW,ORIGINAL,PROFILE)
    def test_retired_detector_recordings_remain_readable_and_new_mirror_identity_is_accepted(self):
        from backend.ferretnet import MODEL_HASH,PIPELINE
        from backend.mirror_profile import profile
        client=TestClient(app);headers={'Origin':'http://127.0.0.1:5173'}
        student=json.loads((Path(__file__).parents[1]/'models/reddit-student-profile.json').read_text())
        for detector in [{'model':'ferretnet-b-v1','modelHash':MODEL_HASH,'pipeline':PIPELINE},profile(),profile(True),student]:
            with patch('backend.recording_api.store',self.store):
                session=self.store.start()['id'];meta=metadata();meta.pop('original')
                meta['preview']['prediction']={k:detector[k] for k in ['model','modelHash','pipeline']}|{'score':.5,'logit':0.,'variant':'original'}
                response=client.post(f'/api/recordings/{session}/captures',headers=headers,data={'metadata':json.dumps(meta)},files={'preview':('p.png',PREVIEW,'image/png')})
                self.assertEqual(response.status_code,200,response.text)
                capture=response.json()['id']
                self.assertEqual(client.get(f'/api/recordings/{session}/captures/{capture}/preview').content,PREVIEW)
                self.store.stop(session)

    def test_production_recording_routes_are_disabled(self):
        client=TestClient(production_app)
        self.assertEqual(client.get('/api/recordings').status_code,404)
        self.assertEqual(client.post('/api/recordings/sessions').status_code,404)
        self.assertEqual(client.post('/api/recordings/'+'a'*32+'/captures').status_code,404)
        self.assertEqual(self.store.sessions(),[])

    def test_status_is_read_only_until_explicit_start(self):
        self.assertEqual(self.store.sessions(),[]);self.assertFalse(self.root.exists())
    def test_exact_bytes_deduplication_and_unknown_labels(self):
        session=self.store.start()['id'];result=self.add(session);duplicate=self.add(session)
        self.assertEqual(result['id'],duplicate['id']);self.assertTrue(duplicate['duplicate'])
        row=self.store.captures(session)[0];self.assertEqual(row['label'],'unknown')
        self.assertEqual(self.store.image_path(session,result['id'],'preview').read_bytes(),PREVIEW)
        self.assertEqual(self.store.image_path(session,result['id'],'original').read_bytes(),ORIGINAL)
        self.assertEqual(self.store.sessions()[0]['captures'],1)
        self.assertEqual(row['metadata']['preview']['width'],256)
    def test_no_automatic_labels_and_identity_corruption_rejected(self):
        session=self.store.start()['id']
        for change in [lambda m:m.update(label='ai'),lambda m:m['preview']['prediction'].update(modelHash='wrong'),
                       lambda m:m['preview']['prediction'].update(score=float('nan')),
                       lambda m:m['original'].update(url='https://evil.test/x.png')]:
            meta=metadata();change(meta)
            with self.assertRaises(ValueError):self.add(session,meta)
        self.assertEqual(self.store.sessions()[0]['captures'],0)
    def test_limits_stop_collection_and_keep_previous_records(self):
        session=self.store.start(max_captures=1)['id'];self.add(session)
        meta=metadata();meta['postUrl']='https://www.reddit.com/r/pics/comments/other123/example/'
        with self.assertRaisesRegex(ValueError,'limit reached'):self.add(session,meta)
        self.assertEqual(self.store.sessions()[0]['active'],0);self.assertEqual(len(self.store.captures(session)),1)
        session=self.store.start(max_bytes=1)['id']
        with self.assertRaisesRegex(ValueError,'limit reached'):self.add(session)
        self.assertEqual(self.store.sessions()[0]['active'],0)
    def test_stopping_session_and_starting_another_prevents_late_captures(self):
        first=self.store.start()['id'];second=self.store.start()['id']
        with self.assertRaisesRegex(ValueError,'stopped'):self.add(first)
        self.store.stop(second)
        with self.assertRaisesRegex(ValueError,'stopped'):self.add(second)
    def test_review_requires_evidence_and_capture_belongs_to_session(self):
        session=self.store.start()['id'];result=self.add(session)
        with self.assertRaisesRegex(ValueError,'evidence'):self.store.label(session,result['id'],'ai','artwork','')
        self.store.label(session,result['id'],'ai','artwork','My generator export record')
        self.assertEqual(self.store.captures(session)[0]['label'],'ai')
        with self.assertRaisesRegex(ValueError,'Unknown capture'):self.store.label('a'*32,result['id'],'real','photo','Creator original')
    def test_invalid_images_and_symlinks_fail_without_records(self):
        session=self.store.start()['id']
        with self.assertRaises(ValueError):self.store.add(session,metadata(),b'not an image',ORIGINAL,PROFILE)
        meta=metadata();meta['original']=None
        result=self.store.add(session,meta,PREVIEW,None,PROFILE)
        with self.assertRaisesRegex(ValueError,'unavailable'):self.store.image_path(session,result['id'],'original')
        path=self.store.image_path(session,result['id'],'preview');path.unlink();path.symlink_to(Path(self.temp.name)/'outside')
        with self.assertRaisesRegex(ValueError,'unavailable'):self.store.image_path(session,result['id'],'preview')
    def test_api_upload_review_export_and_origin_guard(self):
        with patch('backend.recording_api.store',self.store):
            client=TestClient(app);headers={'Origin':'http://127.0.0.1:5173'}
            self.assertEqual(client.get('/api/recordings').json()['sessions'],[]);self.assertFalse(self.root.exists())
            self.assertEqual(client.post('/api/recordings/sessions',headers={'Origin':'https://evil.test'}).status_code,403)
            session=client.post('/api/recordings/sessions',headers=headers).json()['id'];base=f'/api/recordings/{session}'
            response=client.post(base+'/captures',headers=headers,data={'metadata':json.dumps(metadata())},files={'preview':('preview.png',PREVIEW,'image/png'),'original':('original.png',ORIGINAL,'image/png')})
            self.assertEqual(response.status_code,200,response.text);capture=response.json()['id']
            self.assertEqual(client.get(base+f'/captures/{capture}/preview').content,PREVIEW)
            self.assertEqual(client.post(base+f'/captures/{capture}/label',headers=headers,json={'label':'real'}).status_code,422)
            self.assertEqual(client.post(base+f'/captures/{capture}/label',headers=headers,json={'label':'real','evidence':'Creator source'}).status_code,200)
            manifest=client.get(base+'/manifest').json();self.assertEqual(manifest['captures'][0]['label'],'real')
            self.assertIn('not a validation',manifest['purpose'])
            self.assertEqual(client.post(base+'/stop',headers=headers).status_code,200)
    def test_api_malformed_nested_metadata_returns_422_not_500(self):
        with patch('backend.recording_api.store',self.store):
            client=TestClient(app)
            for meta in [{'preview':None},{'preview':{'prediction':None}},[],{'preview':'bad'}]:
                response=client.post('/api/recordings/'+'a'*32+'/captures',headers={'Origin':'http://127.0.0.1:5173'},data={'metadata':json.dumps(meta)},files={'preview':('p.png',PREVIEW,'image/png')})
                self.assertEqual(response.status_code,422,response.text)

if __name__=='__main__':unittest.main()
