"""Verified Community Forensics ViT-S/224 and a portable development-trained head."""
from __future__ import annotations
import json
import os
import time

import numpy as np
import timm
import torch
from safetensors.torch import load_file
from torchvision.transforms import v2 as T
from .assets import MODELS, sha256
from .image_io import decode_image, sigmoid

from .semantic_profile import CHECKPOINT_SHA, MODEL_ID, PIPELINE, canonical_hash, validate_profile
PROFILE_PATH = MODELS/'community-profile.json'


class CommunityBackbone:
    def __init__(self):
        path = MODELS/'community224.safetensors'
        if not path.is_file() or sha256(path) != CHECKPOINT_SHA:
            raise RuntimeError('Community Forensics weights missing or checksum mismatch. Run npm run setup.')
        requested = os.environ.get('LENS_DEVICE','auto')
        device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if requested == 'auto' else requested
        if device not in {'cpu','mps','cuda'}:
            raise ValueError('LENS_DEVICE must be auto, cpu, mps, or cuda')
        self.device = torch.device(device)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=False)
        self.model.head = torch.nn.Linear(384,1)
        weights = load_file(str(path))
        if not all(key.startswith('vit.') for key in weights):
            raise ValueError('Unsupported Community Forensics checkpoint keys.')
        self.model.load_state_dict({key[4:]:value for key,value in weights.items()},strict=True)
        self.model.to(self.device).eval()
        # Matches the authors' 224 evaluation path, not the unrelated 384 port.
        self.transform = T.Compose([T.ToImage(),T.Resize(256),T.CenterCrop(224),
            T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225])])

    @torch.inference_mode()
    def encode(self, image):
        pixels = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model.forward_head(self.model.forward_features(pixels),pre_logits=True)
        return features[0].cpu().numpy().astype(np.float64)


class CommunityEngine(CommunityBackbone):
    def __init__(self, profile=None):
        self.profile = validate_profile(profile or json.loads(PROFILE_PATH.read_text()))
        if self.profile['kind']!='semantic-head':raise ValueError('Expected standalone semantic profile.')
        super().__init__()
        self.weight = np.array(self.profile['head']['weight'],dtype=np.float64)
        self.bias = self.profile['head']['bias']

    def predict(self, data: bytes, variant='original'):
        started = time.perf_counter()
        image,(width,height) = decode_image(data,variant)
        return self.predict_decoded(image,(width,height),variant,started)

    def predict_decoded(self,image,size,variant='original',started=None):
        started=time.perf_counter() if started is None else started
        width,height=size
        features = self.encode(image)
        logit = float(features @ self.weight+self.bias)
        score = sigmoid(logit)
        return {'model':self.profile['model'],'modelHash':self.profile['modelHash'],
            'pipeline':self.profile['pipeline'],'score':score,'logit':logit,'width':width,'height':height,
            'elapsedMs':round((time.perf_counter()-started)*1000),'engine':self.device.type,'variant':variant,
            'watermarkVerified':False,
            'components':{'community_forensics':{'score':score,'logit':logit,'weight':1.,
                'resizeShortestEdge':256,'crop':[224,224],'frozenBackbone':True}},
            'uncertainThreshold':self.profile['uncertainThreshold']}


class CommunityFerretEngine:
    """Frozen OR at both component cuts; uncertain band comes from the semantic head."""
    def __init__(self,community=None,ferret=None,profile=None):
        from .ferretnet import FerretNetEngine
        self.profile=validate_profile(profile or json.loads((MODELS/'community-ferret-profile.json').read_text()))
        if self.profile['kind']!='semantic-hybrid':raise ValueError('Expected semantic hybrid profile.')
        semantic_profile={k:v for k,v in self.profile.items() if k not in {'componentThresholds','semanticProfileHash'}}
        semantic_profile.update(kind='semantic-head',model='community-preview-v1',pipeline=PIPELINE)
        semantic_profile['modelHash']=canonical_hash(semantic_profile)
        if semantic_profile['modelHash']!=self.profile['semanticProfileHash']:
            raise ValueError('Hybrid semantic-head identity mismatch.')
        self.community=community or CommunityEngine(profile=semantic_profile)
        if self.community.profile['modelHash']!=self.profile['semanticProfileHash']:
            raise ValueError('Loaded semantic head differs from the combination profile.')
        self.ferret=ferret or FerretNetEngine();self.device=self.community.device

    def predict(self,data:bytes,variant='original'):
        started=time.perf_counter();image,size=decode_image(data,variant)
        semantic=self.community.predict_decoded(image,size,variant)
        ferret=self.ferret.predict_decoded(image,size,variant)
        threshold=self.profile['componentThresholds']['ferret']
        ferret_offset=float(np.log(threshold/(1-threshold)))
        logit=max(semantic['logit'],ferret['logit']-ferret_offset)
        result={**semantic,'model':self.profile['model'],'modelHash':self.profile['modelHash'],
                'pipeline':self.profile['pipeline'],'score':sigmoid(logit),'logit':logit,
                'elapsedMs':round((time.perf_counter()-started)*1000),
                'components':{**semantic['components'],**ferret['components']},
                'decisionRule':'Either component exceeds its frozen high cutoff; uncertain band uses only the semantic head'}
        result['components']['ferretnet_b']['threshold']=threshold
        # A below-cutoff Ferret score must not manufacture semantic uncertainty.
        result['semanticScore']=semantic['score']
        return result
