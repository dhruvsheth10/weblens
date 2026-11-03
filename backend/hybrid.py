"""Frozen local FerretNet/OpenSynthID combination. No external inference calls."""
from __future__ import annotations

import hashlib
import json
import math
import time

import cv2
import numpy as np
import torch

from .assets import MODELS, OPENSYNTHID_SHA, FERRETNET_SHA, sha256
from .ferretnet import FerretNetEngine
from .image_io import decode_image, sigmoid
from .vendor.opensynthid import build_model, build_carrier_mask, fft_log_magnitude, wavelet_denoise

PROFILE = json.loads((MODELS / 'hybrid-profile.json').read_text())
MODEL_ID = PROFILE['model']
MODEL_HASH = PROFILE['modelHash']
PIPELINE = PROFILE['pipeline']
THRESHOLD = PROFILE['threshold']


def validate_profile():
    value = {k:v for k,v in PROFILE.items() if k != 'modelHash'}
    expected = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if expected != MODEL_HASH:
        raise RuntimeError('Hybrid profile checksum mismatch.')
    if PROFILE['checkpointSha256'] != {'opensynthid':OPENSYNTHID_SHA,'ferretnet':FERRETNET_SHA}:
        raise RuntimeError('Hybrid checkpoint identity mismatch.')


def centered_fusion(ferret_logit, opensynthid_logit):
    if not all(math.isfinite(x) for x in (ferret_logit,opensynthid_logit)):
        raise ValueError('A component returned a non-finite logit.')
    cuts = PROFILE['componentThresholds']
    offsets = {k:math.log(v/(1-v)) for k,v in cuts.items()}
    logit = max(ferret_logit-offsets['ferretnet'],opensynthid_logit-offsets['opensynthid'])
    return sigmoid(logit), logit


def preprocess(image):
    """Exact upstream six-channel feature construction, without reading a file twice."""
    image = cv2.resize(image,(512,512))
    pixels = image.astype(np.float32)/255.
    residual = np.zeros_like(pixels)
    for c in range(3):
        residual[:,:,c] = pixels[:,:,c]-wavelet_denoise(pixels[:,:,c])
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    channels = np.concatenate([pixels.transpose(2,0,1),residual.mean(2)[None],
                               fft_log_magnitude(gray)[None],build_carrier_mask(512)[None]])
    channels = np.nan_to_num(channels,nan=0.,posinf=1.,neginf=-1.)
    return torch.from_numpy(channels).unsqueeze(0).float()


class HybridEngine:
    def __init__(self, ferret=None):
        validate_profile()
        path = MODELS/'opensynthid.pt'
        if not path.is_file() or sha256(path) != OPENSYNTHID_SHA:
            raise RuntimeError('OpenSynthID weights missing or checksum mismatch. Run npm run setup.')
        self.ferret = ferret or FerretNetEngine()
        self.device = self.ferret.device
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['channels'] != 6 or checkpoint['args']['size'] != 512:
            raise RuntimeError('Unsupported OpenSynthID input configuration.')
        self.model = build_model(total_channels=6,pretrained=False,backbone=checkpoint['args']['backbone'])
        self.model.load_state_dict(checkpoint['model_state'],strict=True)
        self.model.to(self.device).eval()
        cv2.setNumThreads(1)

    @torch.inference_mode()
    def predict(self, data: bytes, variant='original'):
        started = time.perf_counter()
        # Ferret validates the image size/format and keeps its existing decoder.
        result = self.ferret.predict(data,variant)
        if variant == 'original':
            bgr = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('OpenSynthID could not decode this image.')
            image = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        else:
            image = np.asarray(decode_image(data,variant)[0])
        open_logit = float(self.model(preprocess(image).to(self.device)).item())
        score, logit = centered_fusion(result['logit'],open_logit)
        result.update(model=MODEL_ID,modelHash=MODEL_HASH,pipeline=PIPELINE,score=score,logit=logit,
                      elapsedMs=round((time.perf_counter()-started)*1000),watermarkVerified=False,
                      decisionRule='Either component exceeds its frozen threshold; combined threshold 0.5')
        result['components']['ferretnet_b']['threshold'] = PROFILE['componentThresholds']['ferretnet']
        result['components']['opensynthid'] = {
            'logit':open_logit,'score':sigmoid(open_logit),'weight':1.,
            'threshold':PROFILE['componentThresholds']['opensynthid'],
            'resize':[512,512],'watermarkVerified':False,
        }
        return result
