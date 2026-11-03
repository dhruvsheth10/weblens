"""Download the two reviewer checkpoints into third_party/weights.

Stops before a file if free space would fall below 20 GB. Probes the gated
DINOv3 backbone and records whether the CLIP fallback is required.
"""
from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'research/reviewers/third_party/weights'
FLOOR_BYTES = 20 * (1024 ** 3)
NOTE = WEIGHTS / 'download-note.json'


def free_bytes() -> int:
    return shutil.disk_usage(Path.home()).free


def ensure_room(needed: int, label: str) -> None:
    free = free_bytes()
    print(f'free={free / 1024**3:.2f} GiB needed={needed / 1024**3:.2f} GiB for {label}', flush=True)
    if free - needed < FLOOR_BYTES:
        raise SystemExit(f'Refusing {label}: free space would drop below 20 GB')


def main() -> None:
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    note: dict = {'dinov3': None, 'files': []}
    ensure_room(2_000_000, 'DINOv3 config probe')
    try:
        path = hf_hub_download(
            'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'config.json',
            local_dir=str(WEIGHTS / 'dinov3-probe'),
        )
        note['dinov3'] = {'gated': False, 'config': path}
        print('DINOV3_OK', path, flush=True)
    except Exception as error:
        note['dinov3'] = {'gated': True, 'error': f'{type(error).__name__}: {error}'}
        print('DINOV3_BLOCKED', note['dinov3']['error'][:1000], flush=True)

    dda_size = 1_255_621_296
    ensure_room(dda_size, 'DDA_ckpt.pth')
    dda = hf_hub_download(
        'Junwei-Xi/Dual-Data-Alignment',
        'DDA_ckpt.pth',
        local_dir=str(WEIGHTS / 'dda'),
    )
    note['files'].append({'path': dda, 'bytes': Path(dda).stat().st_size})
    print('DDA', dda, Path(dda).stat().st_size, flush=True)

    if note['dinov3']['gated']:
        filename = 'ckpt/checkpoint_omniaid_v2.pth'
        size = 3_241_518_989
        variant = 'clip'
    else:
        filename = 'ckpt/checkpoint_omniaid_dino_v2.pth'
        size = 3_238_483_725
        variant = 'dino'
    config = {
        'dino': 'config/config_omniaid_dino_v2.json',
        'clip': 'config/config_omniaid_v2.json',
    }[variant]
    ensure_room(size, filename)
    ckpt = hf_hub_download('Yunncheng/OmniAID', filename, local_dir=str(WEIGHTS / 'omniaid'))
    cfg = hf_hub_download('Yunncheng/OmniAID', config, local_dir=str(WEIGHTS / 'omniaid'))
    note['omniaidVariant'] = variant
    note['files'].append({'path': ckpt, 'bytes': Path(ckpt).stat().st_size})
    note['files'].append({'path': cfg, 'bytes': Path(cfg).stat().st_size})
    note['freeBytesAfter'] = free_bytes()
    NOTE.write_text(json.dumps(note, indent=2) + '\n')
    print('OMNI', variant, ckpt, Path(ckpt).stat().st_size, flush=True)
    print('FREE_AFTER', note['freeBytesAfter'] / 1024**3, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
