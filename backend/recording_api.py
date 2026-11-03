"""Local collection/review endpoints; separately mounted from model inference."""
import asyncio
import json
import re
from fastapi import APIRouter,File,Form,HTTPException,Request,UploadFile
from fastapi.responses import FileResponse
from .recordings import RecordingStore
from .image_io import MAX_BYTES
from .assets import MODELS

router=APIRouter(prefix='/api/recordings')
store=RecordingStore()

def local_origin(request):
    origin=request.headers.get('origin','')
    if origin not in {'http://127.0.0.1:5173','http://localhost:5173','http://127.0.0.1:8000','http://localhost:8000'} and not re.fullmatch(r'chrome-extension://[a-p]{32}',origin):
        raise HTTPException(403,'Recording changes require the local image lab or extension origin.')

def checked(fn,*args):
    try:return fn(*args)
    except (ValueError,OSError) as error:raise HTTPException(422,str(error)) from error

@router.get('')
def sessions():return {'sessions':checked(store.sessions),'root':str(store.root),'labels':'unknown until reviewed'}

@router.post('/sessions')
def start(request:Request):
    local_origin(request);return checked(store.start)

@router.post('/{session}/stop')
def stop(session:str,request:Request):
    local_origin(request);checked(store.stop,session);return {'stopped':True}

@router.post('/{session}/captures')
async def capture(session:str,request:Request,metadata:str=Form(...),preview:UploadFile=File(...),original:UploadFile|None=File(None)):
    local_origin(request)
    try:
        preview_data=await preview.read(MAX_BYTES+1)
        original_data=await original.read(MAX_BYTES+1) if original else None
    finally:
        await preview.close()
        if original:await original.close()
    if len(metadata)>100000:raise HTTPException(422,'Capture metadata exceeds 100 KB.')
    try:meta=json.loads(metadata)
    except ValueError as error:raise HTTPException(422,'Invalid capture JSON.') from error
    part=meta.get('preview') if isinstance(meta,dict) else None
    observation=part.get('prediction') if isinstance(part,dict) else None
    model=observation.get('model') if isinstance(observation,dict) else None
    names={'reddit-sherlock-2':'sherlock-2-profile.json','reddit-sherlock-1.0':'sherlock-profile.json','reddit-student-last2-v1':'reddit-student-profile.json','mirror-h-v1':'mirror-profile.json','mirror-community-v1':'mirror-community-profile.json','community-ferret-preview-v1':'community-ferret-profile.json','community-preview-v1':'community-profile.json',
           'ferretnet-opensynthid-v1':'hybrid-profile.json','fiser-dear-r-v1':'profile.json'}
    if model=='ferretnet-b-v1':
        from .ferretnet import MODEL_ID,MODEL_HASH,PIPELINE
        expected={'model':MODEL_ID,'modelHash':MODEL_HASH,'pipeline':PIPELINE}
    elif model in names:
        expected=json.loads((MODELS/names[model]).read_text())
        if model=='fiser-dear-r-v1':expected={**expected,'model':model,'pipeline':'fiser-layer18-224-dear-native768-grid4-platt-mean-v1'}
    else:raise HTTPException(422,'Unknown recording detector.')
    return await asyncio.to_thread(checked,store.add,session,meta,preview_data,original_data,expected)

@router.get('/{session}/captures')
def captures(session:str,offset:int=0):
    if offset<0:raise HTTPException(422,'Invalid pagination offset.')
    return {'captures':checked(store.captures,session,offset,50)}

@router.post('/{session}/captures/{capture}/label')
async def label(session:str,capture:str,request:Request):
    local_origin(request)
    try:body=await request.json()
    except ValueError as error:raise HTTPException(422,'Invalid review JSON.') from error
    if not isinstance(body,dict):raise HTTPException(422,'Invalid review.')
    checked(store.label,session,capture,body.get('label'),body.get('category',''),body.get('evidence',''));return {'saved':True}

@router.get('/{session}/captures/{capture}/{kind}')
def image(session:str,capture:str,kind:str):
    return FileResponse(checked(store.image_path,session,capture,kind),headers={'Cache-Control':'no-store'})

@router.get('/{session}/manifest')
def manifest(session:str):
    captures=[];offset=0
    while True:
        rows=checked(store.captures,session,offset,50);captures.extend(rows);offset+=len(rows)
        if len(rows)<50:break
    return {'schemaVersion':1,'session':session,'purpose':'development collection; not a validation split',
            'root':str(store.root),'captures':captures,'note':'Unreviewed labels are unknown. Model observations and subreddit names are not ground truth.'}
