#!/usr/bin/env python3
"""
CLI for SynthID watermark inference on single images.

Vendored from fyxme/opensynthid-detect-0.1, revision
e1384859752c8311b8a168125ac16a3eb240d7f5 (model card: Apache-2.0).
This is an unofficial classifier, not a verified watermark decoder.

Example:
  python infer.py --checkpoint model.pt --image path/to/image.png --size 512
"""

import argparse

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Utility functions (from model/dataset.py)
# ---------------------------------------------------------------------------

def wavelet_denoise(channel, wavelet="db4", level=3):
    channel = np.nan_to_num(channel, nan=0.0, posinf=1.0, neginf=0.0)
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    detail = coeffs[-1][0]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(channel.size))

    new_coeffs = [coeffs[0]]
    for details in coeffs[1:]:
        with np.errstate(invalid="ignore", divide="ignore"):
            new_details = tuple(pywt.threshold(d, threshold, mode="soft") for d in details)
        new_coeffs.append(new_details)

    denoised = pywt.waverec2(new_coeffs, wavelet)
    return denoised[: channel.shape[0], : channel.shape[1]]


def build_carrier_mask(size: int) -> np.ndarray:
    """Binary mask with known SynthID carrier frequencies marked (symmetric)."""
    carriers = [(14, 14), (-14, -14), (126, 14), (-126, -14), (98, -14), (-98, 14), (128, 128), (-128, -128)]
    mask = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    for fy, fx in carriers:
        y, x = c + fy, c + fx
        if 0 <= y < size and 0 <= x < size:
            mask[y, x] = 1.0
        # symmetric positions
        y2, x2 = c - fy, c - fx
        if 0 <= y2 < size and 0 <= x2 < size:
            mask[y2, x2] = 1.0
    return mask


def fft_log_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute log-magnitude FFT channel normalized to [0,1]."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    log_mag = np.log1p(mag)
    log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
    return log_mag.astype(np.float32)


# ---------------------------------------------------------------------------
# Model (from model/model.py)
# ---------------------------------------------------------------------------

class DualStreamWatermarkNet(nn.Module):
    """RGB+residual branch + frequency branch fused for watermark detection."""

    def __init__(
        self,
        spatial_in: int = 4,
        freq_in: int = 1,
        hidden_dim: int = 256,
        pretrained: bool = False,
        backbone: str = "resnet18",
    ):
        super().__init__()
        # Spatial branch (ResNet18/34 backbone)
        if backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.spatial = models.resnet34(weights=weights)
        else:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.spatial = models.resnet18(weights=weights)

        self.spatial.conv1 = nn.Conv2d(spatial_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.spatial.fc = nn.Identity()

        # Frequency branch (lightweight)
        freq_layers = [
            nn.Conv2d(freq_in, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ]
        self.freq = nn.Sequential(*freq_layers)

        # Fusion head
        fusion_dim = 512 + 128  # resnet18 penultimate is 512
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x[:, :4, :, :]
        freq = x[:, 4:, :, :]

        s_feat = self.spatial(spatial)  # (B, 512)
        f_feat = self.freq(freq).flatten(1)  # (B, 128)

        fused = torch.cat([s_feat, f_feat], dim=1)
        logit = self.classifier(fused).squeeze(1)
        return logit


def build_model(total_channels: int, hidden_dim: int = 256, pretrained: bool = False, backbone: str = "resnet34") -> DualStreamWatermarkNet:
    """Helper to build the model with inferred branch channels."""
    spatial_in = 4
    freq_in = max(1, total_channels - spatial_in)
    return DualStreamWatermarkNet(
        spatial_in=spatial_in,
        freq_in=freq_in,
        hidden_dim=hidden_dim,
        pretrained=pretrained,
        backbone=backbone,
    )


def preprocess(path: str, size: int = 512, use_fft: bool = True, use_carrier: bool = True) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img_f = img.astype(np.float32) / 255.0
    residual = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        residual[:, :, c] = img_f[:, :, c] - wavelet_denoise(img_f[:, :, c])
    residual_gray = residual.mean(axis=2)
    channels = [img_f.transpose(2, 0, 1), residual_gray[None, :, :]]
    if use_fft:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        channels.append(fft_log_magnitude(gray)[None, :, :])
    if use_carrier:
        channels.append(build_carrier_mask(size)[None, :, :])
    x = np.concatenate(channels, axis=0)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.from_numpy(x).unsqueeze(0).float()


def main():
    parser = argparse.ArgumentParser(description="SynthID watermark inference.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--image", type=str, required=True, help="Path to image.")
    parser.add_argument("--size", type=int, default=512, help="Resize target.")
    parser.add_argument("--no-fft", action="store_true", help="Disable FFT channel.")
    parser.add_argument("--no-carrier-mask", action="store_true", help="Disable carrier mask channel.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    total_channels = ckpt.get("channels", 6)
    model = build_model(total_channels=total_channels, pretrained=False, backbone=ckpt.get("args", {}).get("backbone", "resnet34"))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    x = preprocess(args.image, size=args.size, use_fft=not args.no_fft, use_carrier=not args.no_carrier_mask)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    print(f"Watermark probability: {prob:.4f}")
    print(f"Decision (threshold 0.5): {'WATERMARKED' if prob >= 0.5 else 'CLEAN'}")


if __name__ == "__main__":
    main()
