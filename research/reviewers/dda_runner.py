"""DDA inference using the official Inference/ model and preprocessing."""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

INFERENCE = Path(__file__).resolve().parent / 'third_party' / 'Dual-Data-Alignment' / 'Inference'
sys.path.insert(0, str(INFERENCE))
from models.dinov2_models_lora import DINOv2ModelWithLoRA

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'research/reviewers/third_party/weights/dda/DDA_ckpt.pth'
TORCH_HOME = ROOT / 'research/reviewers/third_party/torch_home'

# Official Inference/inference.py. Center crop only: no resize.
PREPROCESS = transforms.Compose([
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])


def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    try:
        blob = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        info = {'weightsOnly': True, 'mmap': True}
    except Exception as error:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        info = {'weightsOnly': False, 'weightsOnlyError': f'{type(error).__name__}: {error}'}
    if isinstance(blob, dict) and isinstance(blob.get('model'), dict):
        state = blob['model']
    else:
        state = blob
    cleaned = {}
    for key, value in state.items():
        cleaned[key[len('module.'):] if key.startswith('module.') else key] = value
    info['strippedModulePrefix'] = any(key.startswith('module.') for key in state)
    return cleaned, info


def build(device: torch.device):
    os.environ['TORCH_HOME'] = str(TORCH_HOME)
    TORCH_HOME.mkdir(parents=True, exist_ok=True)
    model = DINOv2ModelWithLoRA(name='dinov2_vitl14', lora_rank=8, lora_alpha=1, lora_targets=None)
    state, info = _load_checkpoint(CHECKPOINT)
    if not any(key.endswith('fc.weight') for key in state):
        raise RuntimeError('DDA checkpoint has no classifier')
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    if any(key.endswith('fc.weight') for key in missing):
        raise RuntimeError(f'DDA classifier did not load; missing {missing[:20]}')
    model.to(device)
    model.eval()
    info.update({
        'checkpoint': str(CHECKPOINT),
        'missing': len(missing),
        'unexpected': len(unexpected),
        'missingSample': list(missing)[:12],
        'unexpectedSample': list(unexpected)[:12],
        'device': device.type,
        'preprocessing': 'CenterCrop(336), ToTensor, CLIP mean/std. No resize. Source: Dual-Data-Alignment/Inference/inference.py',
        'patches': [],
        'backbone': 'torch.hub facebookresearch/dinov2 dinov2_vitl14',
        'lora': {'rank': 8, 'alpha': 1, 'targets': ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']},
    })

    def predict(image: Image.Image) -> tuple[float, float]:
        started = time.perf_counter()
        pixels = PREPROCESS(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(pixels)
        if device.type == 'mps':
            torch.mps.synchronize()
        logit = float(output.detach().float().flatten()[0].cpu())
        return logit, (time.perf_counter() - started) * 1000

    return predict, info
