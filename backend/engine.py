from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel

from .assets import DEAR_SHA, FISER_SHA, MODELS, ROOT, sha256

sys.path.insert(0, str(ROOT / "vendor"))
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
MODEL_ID = "fiser-dear-r-v1"
PIPELINE = "fiser-layer18-224-dear-native768-grid4-platt-mean-v1"
LAYER = 18
TILE = 768


def sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("A model returned a non-finite logit")
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def score_logit(score: float) -> float:
    return math.log(score / (1 - score))


def combine(fiser: float, dear: float, calibration: dict) -> tuple[float, float, dict]:
    scores = {}
    for name, logit in (("fiser", fiser), ("dear_r", dear)):
        parameters = calibration[name]
        scores[name] = sigmoid(parameters["a"] * logit + parameters["b"])
    score = max(1e-12, min(1 - 1e-12, (scores["fiser"] + scores["dear_r"]) / 2))
    return score, score_logit(score), scores


def decode_image(data: bytes, variant: str = "original") -> tuple[Image.Image, tuple[int, int]]:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose an image smaller than 20 MB.")
    if variant not in {"original", "jpeg75", "resize512"}:
        raise ValueError("Unknown transformation.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP", "AVIF"}:
                raise ValueError("Choose a JPEG, PNG, WebP or AVIF image.")
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 40 megapixel limit.")
            if min(opened.size) < 128:
                raise ValueError("Image too small: at least 128 pixels on each edge is required.")
            image = ImageOps.exif_transpose(opened)
            # Pillow's model reference RGB conversion is used consistently for both heads.
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The image could not be decoded.") from error
    original_size = image.size
    if variant == "jpeg75":
        encoded = io.BytesIO()
        image.save(encoded, "JPEG", quality=75)
        encoded.seek(0)
        with Image.open(encoded) as transformed:
            image = transformed.convert("RGB")
    elif variant == "resize512":
        scale = min(1, 512 / max(image.size))
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    return image, original_size


def native_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Bound memory while retaining native-resolution pixels, with deterministic coverage."""
    def positions(length: int) -> list[int]:
        if length <= TILE:
            return [0]
        count = min(4, math.ceil(length / TILE))
        return [round(index * (length - TILE) / (count - 1)) for index in range(count)]
    return [(x, y, min(width, x + TILE), min(height, y + TILE)) for y in positions(height) for x in positions(width)]


class ImageEngine:
    def __init__(self, with_head: bool = True, with_dear: bool = True):
        # Reuse hashes only within this initialization, never across restarts.
        verified = {"models/fiser/model.safetensors": sha256(MODELS / "fiser/model.safetensors")}
        if verified["models/fiser/model.safetensors"] != FISER_SHA:
            raise RuntimeError("FiSeR checksum mismatch. Run npm run setup.")
        requested = os.environ.get("LENS_DEVICE", "auto")
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if requested == "auto" else requested
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("LENS_DEVICE must be auto, cpu, mps, or cuda")
        self.device = torch.device(device)
        if self.device.type == "mps":
            torch.mps.set_per_process_memory_fraction(.55)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.processor = AutoImageProcessor.from_pretrained(MODELS / "fiser", local_files_only=True)
        self.encoder = AutoModel.from_pretrained(MODELS / "fiser", local_files_only=True, dtype=torch.float32, attn_implementation="sdpa")
        # hidden_states[18] is the output of block 18, before the final norm.
        # Later blocks cannot affect it. Drop them before transferring to the GPU.
        self.encoder.model.layer = self.encoder.model.layer[:LAYER]
        self.encoder = self.encoder.to(self.device).eval()
        self.dear = None
        if with_dear:
            verified["models/dear_r.pth"] = sha256(MODELS / "dear_r.pth")
            if verified["models/dear_r.pth"] != DEAR_SHA:
                raise RuntimeError("DEAR-r checksum mismatch. Run npm run setup.")
            from dear.nn_classifier.resnet import resnet50
            from dear.detector.corvi_mask_gated_detector import GatedResNet
            self.dear = GatedResNet(resnet50(pretrained=False, stride0=1, dropout=0).change_output(1))
            checkpoint = torch.load(MODELS / "dear_r.pth", map_location="cpu", weights_only=True)
            self.dear.load_state_dict(checkpoint["model"], strict=True)
            self.dear = self.dear.to(self.device).eval()
            if not torch.any(self.dear.gate.gate == 0) or not torch.any(self.dear.gate.gate == 1):
                raise RuntimeError("DEAR-r's trained channel gate was not restored")
        self.head = None
        self.profile = None
        if with_head:
            self.profile = json.loads((MODELS / "profile.json").read_text())
            if self.profile["pipeline"] != PIPELINE or self.profile["model"] != MODEL_ID:
                raise RuntimeError("Inference profile is incompatible. Run npm run setup.")
            for relative, expected in self.profile["artifacts"].items():
                actual = verified[relative] if relative in verified else sha256(ROOT / relative)
                if actual != expected:
                    raise RuntimeError(f"Model artifact checksum mismatch: {relative}")
            self.head = json.loads((MODELS / "fiser-head.json").read_text())
            self.weight = np.asarray(self.head["weight"], dtype=np.float64)
            self.bias = float(self.head["bias"])

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        output = self.encoder(pixel_values=inputs["pixel_values"].to(self.device), output_hidden_states=True, return_dict=True)
        features = torch.nn.functional.normalize(output.hidden_states[LAYER][:, 0, :].float(), dim=-1)
        return features.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def dear_logit(self, image: Image.Image) -> tuple[float, list]:
        if self.dear is None:
            raise RuntimeError("DEAR-r is not loaded")
        regions = native_regions(*image.size)
        logits = []
        mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
        for region in regions:
            pixels = np.asarray(image.crop(region), dtype=np.float32).copy() / 255
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
            value = float(self.dear((tensor - mean) / std).item())
            if not math.isfinite(value):
                raise RuntimeError("DEAR-r returned a non-finite logit")
            logits.append(value)
        result = float(np.mean(logits))
        if self.device.type == "mps":
            # Different native image shapes otherwise retain large cached Metal buffers.
            del tensor, pixels
            torch.mps.empty_cache()
        return result, regions

    def predict(self, data: bytes, variant: str = "original") -> dict:
        started = time.perf_counter()
        image, (width, height) = decode_image(data, variant)
        feature = self.encode([image])[0]
        fiser = float(feature.astype(np.float64) @ self.weight + self.bias)
        dear, regions = self.dear_logit(image)
        score, logit, calibrated = combine(fiser, dear, self.profile["calibration"])
        return {"model": MODEL_ID, "modelHash": self.profile["modelHash"], "pipeline": PIPELINE,
                "score": score, "logit": logit, "width": width, "height": height,
                "elapsedMs": round((time.perf_counter() - started) * 1000), "engine": self.device.type,
                "variant": variant, "components": {
                    "fiser": {"logit": fiser, "score": sigmoid(fiser), "calibratedScore": calibrated["fiser"], "weight": .5, "head": "trained-linear", "layer": LAYER},
                    "dear_r": {"logit": dear, "score": sigmoid(dear), "calibratedScore": calibrated["dear_r"], "weight": .5, "regions": regions}}}
