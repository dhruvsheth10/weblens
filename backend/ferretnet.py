"""FerretNet-B inference using the authors' released Apache-2.0 checkpoint."""
from __future__ import annotations

import os
import time

import torch
from torch import nn
from torch.nn import functional as functional
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import center_crop

from .assets import FERRETNET_SHA, MODELS, sha256
from .image_io import decode_image, sigmoid

MODEL_ID = "ferretnet-b-v1"
MODEL_HASH = FERRETNET_SHA
PIPELINE = "ferretnet-b-median3-centercrop256-v1"
# Chosen on the existing 60-real/60-AI calibration split with zero observed
# false positives. It is an operating point, not a probability or guarantee.
THRESHOLD = 0.8304235935211183


class MaskMedianValues(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        padded = functional.pad(
            tensor,
            (self.padding, self.padding, self.padding, self.padding),
            mode="constant",
            value=0.0,
        )
        patches = functional.unfold(padded, kernel_size=self.kernel_size)
        patches = patches.view(batch, channels, self.kernel_size**2, -1)
        # This matches the released checkpoint's local-pixel-dependency filter.
        patches[:, :, (self.kernel_size**2) // 2, :] = 0.0
        medians = patches.median(dim=2).values
        return medians.view(batch, channels, height, width)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.depth_wise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding=padding,
            groups=in_channels, bias=False,
        )
        self.point_wise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.point_wise(self.depth_wise(tensor))


class DilatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=1,
            groups=in_channels, dilation=1, bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=2,
            groups=in_channels, dilation=2, bias=False,
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv1(tensor) + self.conv2(tensor))


class DSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            DilatedConv2d(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layers(tensor) + self.shortcut(tensor))


class FerretNetB(nn.Module):
    def __init__(self):
        super().__init__()
        dimension = 96
        self.local_dependency = MaskMedianValues(3)
        self.cbr1 = nn.Sequential(
            nn.Conv2d(3, dimension // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dimension // 2),
            nn.ReLU(inplace=True),
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(dimension // 2, dimension, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dimension),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        for depth in (2, 2):
            blocks.append(nn.Sequential(
                DSBlock(dimension, dimension * 2, stride=2),
                *[DSBlock(dimension * 2, dimension * 2, stride=1) for _ in range(depth - 1)],
            ))
            dimension *= 2
        blocks.append(nn.Conv2d(dimension, dimension, 1, bias=False))
        self.feature = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit = nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(dimension, 1))

    def forward_features(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor - self.local_dependency(tensor)
        tensor = self.cbr2(self.cbr1(tensor))
        tensor = self.feature(tensor)
        return self.avg_pool(tensor).flatten(1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.logit(self.forward_features(tensor))


class FerretNetEngine:
    def __init__(self):
        weight_path = MODELS / "ferretnet_b.pth"
        if sha256(weight_path) != FERRETNET_SHA:
            raise RuntimeError("FerretNet checksum mismatch. Run npm run setup.")
        requested = os.environ.get("LENS_DEVICE", "auto")
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ) if requested == "auto" else requested
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("LENS_DEVICE must be auto, cpu, mps, or cuda")
        self.device = torch.device(device)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.model = FerretNetB()
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.CenterCrop((256, 256)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    @torch.inference_mode()
    def predict(self, data: bytes, variant: str = "original") -> dict:
        started = time.perf_counter()
        image, (width, height) = decode_image(data, variant)
        return self.predict_decoded(image,(width,height),variant,started)

    @torch.inference_mode()
    def predict_decoded(self,image,size,variant='original',started=None):
        started=time.perf_counter() if started is None else started
        width,height=size
        # Crop before allocating a tensor of the full decoded image. The PIL
        # crop uses exactly the same rounding/padding as the original transform.
        pixels = self.transform(center_crop(image, [256, 256])).unsqueeze(0).to(self.device)
        logit = float(self.model(pixels)[0, 0].item())
        score = sigmoid(logit)
        return {
            "model": MODEL_ID,
            "modelHash": MODEL_HASH,
            "pipeline": PIPELINE,
            "score": score,
            "logit": logit,
            "width": width,
            "height": height,
            "elapsedMs": round((time.perf_counter() - started) * 1000),
            "engine": self.device.type,
            "variant": variant,
            "components": {
                "ferretnet_b": {
                    "logit": logit,
                    "score": score,
                    "weight": 1.0,
                    "crop": [256, 256],
                    "localPixelDependency": "masked-median-3x3",
                }
            },
        }
