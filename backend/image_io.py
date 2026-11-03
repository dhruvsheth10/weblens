"""Image decoding without importing the legacy detector's ML libraries.

The legacy engine retains its pinned implementation for artifact compatibility.
Standard-image decoding here matches it; MPO JPEGs use their primary frame.
"""
from __future__ import annotations

import io
import math

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000


def sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError('A model returned a non-finite logit')
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def decode_image(data: bytes, variant: str = 'original') -> tuple[Image.Image, tuple[int, int]]:
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Choose an image smaller than 20 MB.')
    if variant not in {'original', 'jpeg75', 'resize512', 'reddit-webp640', 'reddit-webp960'}:
        raise ValueError('Unknown transformation.')
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format not in {'JPEG', 'MPO', 'PNG', 'WEBP', 'AVIF'}:
                raise ValueError('Choose a JPEG, PNG, WebP or AVIF image.')
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError('Image exceeds the 40 megapixel limit.')
            if min(opened.size) < 128:
                raise ValueError('Image too small: at least 128 pixels on each edge is required.')
            image = ImageOps.exif_transpose(opened)
            if image.mode in {'RGBA', 'LA'} or 'transparency' in image.info:
                rgba = image.convert('RGBA')
                background = Image.new('RGBA', rgba.size, 'white')
                image = Image.alpha_composite(background, rgba).convert('RGB')
            else:
                image = image.convert('RGB')
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('The image could not be decoded.') from error
    original_size = image.size
    if variant == 'jpeg75':
        encoded = io.BytesIO()
        image.save(encoded, 'JPEG', quality=75)
        encoded.seek(0)
        with Image.open(encoded) as transformed:
            image = transformed.convert('RGB')
    elif variant == 'resize512':
        scale = min(1, 512 / max(image.size))
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    elif variant in {'reddit-webp640','reddit-webp960'}:
        from .previews import transform_image
        image,_=transform_image(image,variant)
    return image, original_size
