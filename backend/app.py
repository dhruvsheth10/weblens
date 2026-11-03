from __future__ import annotations

import asyncio
import logging
import json
import time
import io
from contextlib import asynccontextmanager
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .image_io import MAX_BYTES
from .assets import MODELS
from .previews import predict_image
from .ferretnet import (
    FerretNetEngine,
    MODEL_HASH as FERRETNET_HASH,
    MODEL_ID as FERRETNET_MODEL_ID,
    PIPELINE as FERRETNET_PIPELINE,
    THRESHOLD as FERRETNET_THRESHOLD,
)

logger = logging.getLogger("uvicorn.error")
engine: FerretNetEngine | None = None
LEGACY_MODEL_ID = 'fiser-dear-r-v1'
LEGACY_PIPELINE = 'fiser-layer18-224-dear-native768-grid4-platt-mean-v1'
legacy_engine = None
hybrid_engine = None
community_engine = None
community_ferret_engine = None
mirror_engine = None
mirror_community_engine = None
student_engine = None
sherlock_engine = None
sherlock2_engine = None
STUDENT_PROFILE = json.loads((MODELS/'reddit-student-profile.json').read_text())
SHERLOCK_PROFILE = json.loads((MODELS/'sherlock-profile.json').read_text())
SHERLOCK2_PROFILE = json.loads((MODELS/'sherlock-2-profile.json').read_text())
MIRROR_PROFILE = json.loads((MODELS/'mirror-profile.json').read_text())
MIRROR_COMMUNITY_PROFILE = json.loads((MODELS/'mirror-community-profile.json').read_text())
HYBRID_PROFILE = json.loads((MODELS/'hybrid-profile.json').read_text())
COMMUNITY_PROFILE = json.loads((MODELS/'community-profile.json').read_text())
COMMUNITY_FERRET_PROFILE = json.loads((MODELS/'community-ferret-profile.json').read_text())
inference_lock = asyncio.Lock()
load_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, legacy_engine, hybrid_engine, community_engine, community_ferret_engine, mirror_engine, mirror_community_engine, student_engine, sherlock_engine, sherlock2_engine
    started = time.perf_counter()
    logger.info("Loading FerretNet-B from verified local weights")
    engine = await asyncio.to_thread(FerretNetEngine)
    logger.info("FerretNet-B ready on %s in %.2fs; large experimental detectors load on demand",
                engine.device, time.perf_counter() - started)
    # Prepare the prototype default before the launcher opens the image lab.
    # This changes loading order only; all frozen profiles and scores are retained.
    prototype=await selected_engine(SHERLOCK2_PROFILE['model'])
    # Pay device/kernel initialization during startup, rather than on a user's
    # first image. This in-memory warmup is never saved, labeled, or used to fit.
    warmup=io.BytesIO();Image.new('RGB',(256,256),(128,128,128)).save(warmup,'PNG')
    await asyncio.to_thread(predict_image,prototype,warmup.getvalue(),'original')
    logger.info('Sherlock 2 ready in %.2fs total',time.perf_counter()-started)
    yield
    engine = None
    legacy_engine = None
    hybrid_engine = None
    community_engine = None
    community_ferret_engine = None
    mirror_engine = None
    mirror_community_engine = None
    student_engine = None
    sherlock_engine = None
    sherlock2_engine = None


app = FastAPI(title="Reddit Lens local detector", lifespan=lifespan)
# Pair collection is disabled. Archived files remain on disk, but no
# recording endpoints are mounted in the running application.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                   allow_origin_regex=r"chrome-extension://[a-p]{32}", allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/api/health")
async def health():
    if engine is None:
        raise HTTPException(503, "Models are loading")
    return {
        "protocolVersion": 2,
        "status": "ready",
        "model": FERRETNET_MODEL_ID,
        "modelHash": FERRETNET_HASH,
        "pipeline": FERRETNET_PIPELINE,
        "device": engine.device.type,
        "threshold": FERRETNET_THRESHOLD,
        "availableModels": {
            COMMUNITY_FERRET_PROFILE['model']: {
                'modelHash':COMMUNITY_FERRET_PROFILE['modelHash'],
                'status':'ready' if community_ferret_engine is not None else 'available',
                'threshold':COMMUNITY_FERRET_PROFILE['threshold'],'pipeline':COMMUNITY_FERRET_PROFILE['pipeline'],
                'uncertainThreshold':COMMUNITY_FERRET_PROFILE['uncertainThreshold'],
            },
            COMMUNITY_PROFILE['model']: {
                'modelHash':COMMUNITY_PROFILE['modelHash'],
                'status':'ready' if community_engine is not None else 'available',
                'threshold':COMMUNITY_PROFILE['threshold'],'pipeline':COMMUNITY_PROFILE['pipeline'],
                'uncertainThreshold':COMMUNITY_PROFILE['uncertainThreshold'],
            },
            HYBRID_PROFILE['model']: {
                'modelHash':HYBRID_PROFILE['modelHash'],
                "status": "ready" if hybrid_engine is not None else "available",
                "threshold": HYBRID_PROFILE['threshold'], "pipeline": HYBRID_PROFILE['pipeline'],
            },
            STUDENT_PROFILE['model']: {'modelHash':STUDENT_PROFILE['modelHash'],'pipeline':STUDENT_PROFILE['pipeline'],'threshold':STUDENT_PROFILE['threshold'],'status':'ready' if student_engine is not None else 'available'},
            SHERLOCK2_PROFILE['model']: {'modelHash':SHERLOCK2_PROFILE['modelHash'],'pipeline':SHERLOCK2_PROFILE['pipeline'],'threshold':SHERLOCK2_PROFILE['threshold'],'status':'ready' if sherlock2_engine is not None else 'available'},
            SHERLOCK_PROFILE['model']: {'modelHash':SHERLOCK_PROFILE['modelHash'],'pipeline':SHERLOCK_PROFILE['pipeline'],'threshold':SHERLOCK_PROFILE['threshold'],'status':'ready' if sherlock_engine is not None else 'available'},
            MIRROR_PROFILE['model']: {**MIRROR_PROFILE,'status':'ready' if mirror_engine is not None else 'available'},
            MIRROR_COMMUNITY_PROFILE['model']: {**MIRROR_COMMUNITY_PROFILE,'status':'ready' if mirror_community_engine is not None else 'available'},
        },
        "offline": True,
    }


async def selected_engine(model: str):
    global legacy_engine, hybrid_engine, community_engine, community_ferret_engine, mirror_engine, mirror_community_engine, student_engine, sherlock_engine, sherlock2_engine
    if model == SHERLOCK2_PROFILE['model']:
        if sherlock2_engine is None:
            async with load_lock:
                if sherlock2_engine is None:
                    from .reddit_research import RedditStudentEngine
                    logger.info('Loading pinned Sherlock 2 Reddit student')
                    sherlock2_engine=await asyncio.to_thread(RedditStudentEngine,SHERLOCK2_PROFILE)
        return sherlock2_engine
    if model == SHERLOCK_PROFILE['model']:
        if sherlock_engine is None:
            async with load_lock:
                if sherlock_engine is None:
                    from .reddit_research import RedditStudentEngine
                    logger.info('Loading pinned Sherlock 1.0 Reddit student')
                    sherlock_engine=await asyncio.to_thread(RedditStudentEngine,SHERLOCK_PROFILE)
        return sherlock_engine
    if model == STUDENT_PROFILE['model']:
        if student_engine is None:
            async with load_lock:
                if student_engine is None:
                    from .reddit_research import RedditStudentEngine
                    logger.info('Loading pinned experimental Reddit preview student')
                    student_engine=await asyncio.to_thread(RedditStudentEngine,STUDENT_PROFILE)
        return student_engine
    if model == MIRROR_PROFILE['model']:
        if mirror_engine is None:
            async with load_lock:
                if mirror_engine is None:
                    from .mirror import MirrorEngine
                    logger.info('Loading verified local MIRROR H+ checkpoint (3.45 GB)')
                    mirror_engine=await asyncio.to_thread(MirrorEngine)
        return mirror_engine
    if model == MIRROR_COMMUNITY_PROFILE['model']:
        mirror=await selected_engine(MIRROR_PROFILE['model'])
        community=await selected_engine(COMMUNITY_FERRET_PROFILE['model'])
        if mirror_community_engine is None:
            async with load_lock:
                if mirror_community_engine is None:
                    from .mirror import MirrorCommunityEngine
                    mirror_community_engine=MirrorCommunityEngine(mirror,community)
        return mirror_community_engine
    if model == COMMUNITY_PROFILE['model']:
        if community_engine is None:
            async with load_lock:
                if community_engine is None:
                    from .community import CommunityEngine
                    logger.info('Loading verified Community Forensics + development-trained preview head')
                    community_engine=await asyncio.to_thread(CommunityEngine)
        return community_engine
    if model == COMMUNITY_FERRET_PROFILE['model']:
        semantic=await selected_engine(COMMUNITY_PROFILE['model'])
        if engine is None:raise HTTPException(503,'FerretNet-B is loading')
        if community_ferret_engine is None:
            async with load_lock:
                if community_ferret_engine is None:
                    from .community import CommunityFerretEngine
                    community_ferret_engine=await asyncio.to_thread(CommunityFerretEngine,semantic,engine)
        return community_ferret_engine
    if model == HYBRID_PROFILE['model']:
        if engine is None:
            raise HTTPException(503, "FerretNet-B is loading")
        if hybrid_engine is None:
            async with load_lock:
                if hybrid_engine is None:
                    from .hybrid import HybridEngine
                    logger.info("Loading OpenSynthID for the local combination")
                    hybrid_engine = await asyncio.to_thread(HybridEngine,engine)
        return hybrid_engine
    raise HTTPException(422, "Unknown or retired detector. Choose a model listed in Settings.")


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), variant: str = Form("original"),
                 model: str = Form(SHERLOCK2_PROFILE['model'])):
    try:
        data = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(413, "Choose an image smaller than 20 MB.")
    async with inference_lock:
        try:
            detector = await selected_engine(model)
            return await asyncio.to_thread(predict_image, detector, data, variant)
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            logger.exception("Local model inference failed")
            raise HTTPException(500, "Local inference failed. Check the backend terminal.") from error
