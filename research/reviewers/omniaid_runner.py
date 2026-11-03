"""OmniAID inference using reward/clean_test.py preprocessing and model classes."""
from __future__ import annotations

import gc
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import torch
from PIL import Image

REWARD = Path(__file__).resolve().parent / 'third_party' / 'OmniAID' / 'reward'
sys.path.insert(0, str(REWARD))
from clean_test import BACKBONES, process_images

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'research/reviewers/third_party/weights'
NOTE = WEIGHTS / 'download-note.json'


def _load_hyphen_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_blob(path: Path) -> tuple[dict, dict]:
    try:
        blob = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        info = {'weightsOnly': True, 'mmap': True}
    except Exception as error:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        info = {'weightsOnly': False, 'weightsOnlyError': f'{type(error).__name__}: {error}'}
    if isinstance(blob, dict) and 'model' in blob and isinstance(blob['model'], dict):
        blob = blob['model']
    cleaned = {key[len('module.'):] if key.startswith('module.') else key: value for key, value in blob.items()}
    return cleaned, info


def _variant() -> str:
    if not NOTE.exists():
        raise FileNotFoundError(NOTE)
    note = json.loads(NOTE.read_text())
    variant = note.get('omniaidVariant')
    if variant not in {'dino', 'clip'}:
        raise ValueError(f'Unexpected OmniAID variant {variant}')
    return variant


def build(device: torch.device):
    variant = _variant()
    backbone = 'dino' if variant == 'dino' else 'clip'
    spec = BACKBONES[backbone]
    filename = 'ckpt/checkpoint_omniaid_dino_v2.pth' if backbone == 'dino' else 'ckpt/checkpoint_omniaid_v2.pth'
    checkpoint = WEIGHTS / 'omniaid' / filename
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    config = types.SimpleNamespace(**spec['config_kwargs'])
    if backbone == 'dino':
        module = _load_hyphen_module(REWARD / 'omniaid-dino.py', 'omniaid_dino_reward')
        model = module.OmniAID_DINO(config)
        load_note = 'Loaded reward/omniaid-dino.py via spec_from_file_location because clean_test import_module("omniaid-dino") is not a legal module name.'
    else:
        from omniaid import OmniAID
        model = OmniAID(config)
        load_note = 'DINOv3 backbone was gated or unlicensed, so this run uses the CLIP ViT-L/14@336 variant.'
    state, info = _checkpoint_blob(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    if any(key == 'head.weight' or key.endswith('.head.weight') for key in missing):
        raise RuntimeError(f'OmniAID head did not load; missing {missing[:20]}')
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured: dict = {}

    def remember(_module, _inputs, output):
        captured['logits'] = output.detach()

    model.head.register_forward_hook(remember)
    resolution = spec['image_resolution']
    mean = spec['mean']
    std = spec['std']
    info.update({
        'variant': variant,
        'checkpoint': str(checkpoint),
        'missing': len(missing),
        'unexpected': len(unexpected),
        'missingSample': list(missing)[:12],
        'unexpectedSample': list(unexpected)[:12],
        'device': device.type,
        'preprocessing': (
            f"Resize(({resolution}, {resolution})) bilinear default, ToTensor, "
            f"mean {mean}, std {std}. Source: OmniAID/reward/clean_test.py and data/datasets.py v2 eval transform."
        ),
        'resolution': resolution,
        'mean': mean,
        'std': std,
        'patches': [],
        'loadNote': load_note,
        'downloadNote': json.loads(NOTE.read_text()) if NOTE.exists() else None,
        'logitDefinition': 'class-1 logit minus class-0 logit from the linear head, captured before the official softmax',
    })

    def predict(image: Image.Image) -> tuple[float, float]:
        started = time.perf_counter()
        pixels = process_images([image], resolution, mean, std).to(device)
        with torch.inference_mode():
            model(pixels)
        if device.type == 'mps':
            torch.mps.synchronize()
        logits = captured['logits']
        score = float((logits[0, 1] - logits[0, 0]).detach().float().cpu())
        return score, (time.perf_counter() - started) * 1000

    return predict, info
