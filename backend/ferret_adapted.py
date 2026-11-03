"""Portable FerretNet feature heads and crop/scale experiments.

Only pixel-derived features enter the classifier. Profiles contain numerical
parameters and a pinned checkpoint, never executable code or source metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2 as transforms

from .assets import FERRETNET_SHA, MODELS
from .ferretnet import FerretNetEngine
from .image_io import decode_image, sigmoid

PROFILE_PATH = MODELS / 'ferretnet-adapted.json'
REPRESENTATION_DIMENSIONS = {
    'center256': 384, 'native5_mean': 384, 'native9_mean': 384,
    'native5_mean_std': 768, 'multiscale_native': 768,
    'resize5_mean': 384, 'native_resize_means': 768, 'all_scales': 1152,
}
FIXED_REPRESENTATIONS = {
    'center256', 'native5_mean', 'native5_median', 'native5_max',
    'native9_mean', 'native9_median', 'native9_max',
    'resize5_mean', 'resize5_median', 'resize5_max',
    'center512', 'native_resize_blend',
}


def profile_hash(profile: dict) -> str:
    payload = {key: value for key, value in profile.items() if key != 'modelHash'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_profile(profile: dict) -> dict:
    if profile.get('schemaVersion') != 1 or profile.get('checkpointSha256') != FERRETNET_SHA:
        raise ValueError('Expected a FerretNet-B profile with the pinned released checkpoint.')
    if profile.get('kind') not in {'probe', 'fixed'}:
        raise ValueError('Profile kind must be probe or fixed.')
    representation = profile.get('representation')
    if profile['kind'] == 'probe':
        dimension = REPRESENTATION_DIMENSIONS.get(representation)
        head = profile.get('head', {})
        weights = head.get('weight', [])
        if dimension is None or len(weights) != dimension or not all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in weights):
            raise ValueError('Head dimensions or weights are invalid.')
        if not isinstance(head.get('bias'), (int,float)) or isinstance(head.get('bias'),bool) or not math.isfinite(head['bias']):
            raise ValueError('Head bias must be finite.')
    elif representation not in FIXED_REPRESENTATIONS:
        raise ValueError('Unknown fixed aggregation.')
    threshold = profile.get('threshold')
    if not isinstance(threshold, (int,float)) or isinstance(threshold,bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and between zero and one.')
    for field in ['model', 'pipeline']:
        if not isinstance(profile.get(field),str) or not profile[field]:
            raise ValueError('Model and pipeline identities are required.')
    if profile.get('modelHash') != profile_hash(profile):
        raise ValueError('Adapted profile checksum mismatch.')
    return profile


def crop_regions(image: Image.Image, size: int, grid: int) -> list[Image.Image]:
    width, height = image.size
    if min(width,height) < size:
        dw, dh = max(0,size-width), max(0,size-height)
        image = ImageOps.expand(image,(dw//2,dh//2,(dw+1)//2,(dh+1)//2),fill=0)
        width,height=image.size
    xs,ys=[0,round((width-size)/2),width-size],[0,round((height-size)/2),height-size]
    coordinates=[(x,y) for y in ys for x in xs]
    if grid==5:
        coordinates=[coordinates[i] for i in [0,2,4,6,8]]
    elif grid==1:
        coordinates=[coordinates[4]]
    return [image.crop((x,y,x+size,y+size)) for x,y in coordinates]


class FerretNetAdaptedEngine(FerretNetEngine):
    def __init__(self, profile_path: Path = PROFILE_PATH, profile: dict | None = None):
        self.profile=validate_profile(profile if profile is not None else json.loads(profile_path.read_text()))
        super().__init__()
        self.normalize=transforms.Compose([
            transforms.ToImage(),transforms.ToDtype(torch.float32,scale=True),
            transforms.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711]),
        ])
        if self.profile['kind']=='probe':
            self.head_weight=np.asarray(self.profile['head']['weight'],dtype=np.float64)
            self.head_bias=float(self.profile['head']['bias'])

    @torch.inference_mode()
    def encode_regions(self, images):
        tensors=torch.stack([self.normalize(image) for image in images]).to(self.device)
        features=self.model.forward_features(tensors)
        logits=self.model.logit(features).flatten()
        return features.cpu().numpy(),logits.cpu().numpy()

    @torch.inference_mode()
    def predict(self, data: bytes, variant: str = 'original') -> dict:
        started=time.perf_counter()
        image,(width,height)=decode_image(data,variant)
        name=self.profile['representation']
        need_native=not name.startswith('resize5') and name!='center512'
        need_resized=name.startswith('resize5') or name in {'native_resize_means','all_scales','native_resize_blend'}
        need_large=name in {'center512','multiscale_native','all_scales','native_resize_blend'}
        crop_count=0
        native=native_logits=resized=resize_logits=large=large_logits=None
        if need_native:
            grid=1 if name=='center256' else 5 if name.startswith('native5') else 9
            native,native_logits=self.encode_regions(crop_regions(image,256,grid))
            crop_count+=grid
        if need_resized:
            scale=min(512/min(image.size),2048/max(image.size))
            scaled=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.BILINEAR)
            resized,resize_logits=self.encode_regions(crop_regions(scaled,256,5))
            crop_count+=5
        if need_large:
            large,large_logits=self.encode_regions(crop_regions(image,512,1))
            crop_count+=1
        if self.profile['kind']=='probe':
            if name=='center256':
                features=native[0]
            elif name in {'native5_mean','native9_mean'}:
                features=native.mean(0)
            elif name=='native5_mean_std':
                features=np.concatenate([native.mean(0),native.std(0)])
            elif name=='multiscale_native':
                features=np.concatenate([native.mean(0),large[0]])
            elif name=='resize5_mean':
                features=resized.mean(0)
            elif name=='native_resize_means':
                features=np.concatenate([native.mean(0),resized.mean(0)])
            elif name=='all_scales':
                features=np.concatenate([native.mean(0),resized.mean(0),large[0]])
            logit=float(features.astype(np.float64)@self.head_weight+self.head_bias)
        elif name=='center256':
            logit=float(native_logits[0])
        elif name=='center512':
            logit=float(large_logits[0])
        elif name=='native_resize_blend':
            logit=float((native_logits.mean()+resize_logits.mean()+large_logits[0])/3)
        else:
            values=resize_logits if name.startswith('resize5') else native_logits
            method=name.rsplit('_',1)[1]
            logit=float(values.mean() if method=='mean' else np.median(values) if method=='median' else values.max())
        score=sigmoid(logit)
        return {
            'model':self.profile['model'],'modelHash':self.profile['modelHash'],
            'pipeline':self.profile['pipeline'],'score':score,'logit':logit,
            'width':width,'height':height,'elapsedMs':round((time.perf_counter()-started)*1000),
            'engine':self.device.type,'variant':variant,
            'components':{'ferretnet_adapted':{
                'logit':logit,'score':score,'weight':1.0,
                'representation':name,'crops':crop_count,
                'classifier':'frozen-backbone-linear-head' if self.profile['kind']=='probe' else 'released-head-aggregation',
                'usesResize':need_resized,'checkpointSha256':FERRETNET_SHA,
            }},
        }
