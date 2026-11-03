"""Deterministic deployment stress tests, not a claim to reproduce Reddit's codec.

Both labels receive identical width-limited, aspect-preserving WebP quality 80.
The original detector implementations and original-byte paths stay unchanged.
"""
from __future__ import annotations
import io
import time
import numpy as np
from PIL import Image

PREVIEW_WIDTHS = {'reddit-webp640': 640, 'reddit-webp960': 960}
VARIANTS = ('original', 'jpeg75', 'resize512', *PREVIEW_WIDTHS)


def preview_bytes(data: bytes, variant: str):
    from .image_io import decode_image
    if variant not in PREVIEW_WIDTHS:
        raise ValueError('Unknown preview transformation.')
    image, original_size = decode_image(data)
    encoded,size = encode_preview(image,variant)
    return encoded, original_size, size


def encode_preview(image, variant):
    scale=min(1.,PREVIEW_WIDTHS[variant]/image.width)
    size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    image=image.resize(size,Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, 'WEBP', quality=80, method=4)
    return stream.getvalue(),size


def transform_image(image,variant):
    encoded,size=encode_preview(image,variant)
    with Image.open(io.BytesIO(encoded)) as decoded:
        return decoded.convert('RGB'),size


def predict_image(engine, data: bytes, variant='original'):
    if variant not in VARIANTS:
        raise ValueError('Unknown transformation.')
    if variant not in PREVIEW_WIDTHS:
        return engine.predict(data, variant)
    if engine.__class__.__module__ == 'backend.engine':
        # The legacy file is checksum-pinned. Reuse its exact feature/head and
        # aggregation methods while validating the source before transformation.
        from .image_io import decode_image, sigmoid
        from .engine import combine, MODEL_ID, PIPELINE, LAYER
        started=time.perf_counter()
        image,original_size=decode_image(data,variant)
        size=image.size
        feature=engine.encode([image])[0]
        fiser=float(feature.astype(np.float64)@engine.weight+engine.bias)
        dear,regions=engine.dear_logit(image)
        score,logit,calibrated=combine(fiser,dear,engine.profile['calibration'])
        result={'model':MODEL_ID,'modelHash':engine.profile['modelHash'],'pipeline':PIPELINE,
                'score':score,'logit':logit,'width':original_size[0],'height':original_size[1],
                'elapsedMs':round((time.perf_counter()-started)*1000),'engine':engine.device.type,
                'variant':variant,'components':{
                    'fiser':{'logit':fiser,'score':sigmoid(fiser),'calibratedScore':calibrated['fiser'],'weight':.5,'head':'trained-linear','layer':LAYER},
                    'dear_r':{'logit':dear,'score':sigmoid(dear),'calibratedScore':calibrated['dear_r'],'weight':.5,'regions':regions}}}
    else:
        # Validate original dimensions before shrinking; a generated narrow
        # preview is valid even when its shorter edge drops below 128 pixels.
        result=engine.predict(data,variant)
        original_size=(result['width'],result['height'])
        scale=min(1.,PREVIEW_WIDTHS[variant]/original_size[0])
        size=(max(1,round(original_size[0]*scale)),max(1,round(original_size[1]*scale)))
    result.update(variant=variant, width=original_size[0], height=original_size[1],
                  transformation={'codec':'webp', 'quality':80, 'size':list(size),
                                  'simulation':True, 'resize':'lanczos-width-no-upscale-v1'})
    return result
