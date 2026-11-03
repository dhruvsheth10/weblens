"""Reddit student v2: fine-tune every layer of the Community ViT-S/16 on the archive collection.

    backend/.venv/bin/python -m research.reddit_v2.train --smoke
    backend/.venv/bin/python -m research.reddit_v2.train
    backend/.venv/bin/python -m research.reddit_v2.train --final    # once, after training is frozen

Splits are fixed before training from author/source hashes only:
  fit     archive "train" authors (90%), old Reddit development previews, calibration development,
          external research images (OpenFake, NTIRE 2026; 90% of prompt/image groups)
  select  archive "train" authors (10%), old Reddit tuning previews, calibration tuning, and 10% of
          external groups. Only the Reddit/calibration part picks the epoch and threshold; the
          external part is reported alongside.
  final   archive "test" and "test-unseen-source". Scored only by --final. Authors with any
          final image are dropped from fit and select.
Never read: the old Reddit holdout, calibration audit, the benchmark validation folder.

Training reads CDN previews, never Reddit originals: AI originals are mostly PNG and old
real originals mostly JPEG, a format shortcut that Reddit's WebP previews do not carry.
Calibration files are originals, so they are always downscaled and re-encoded first.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from backend.community import CommunityBackbone
from backend.image_io import decode_image
from backend.semantic_profile import CHECKPOINT_SHA
from research.reddit_adaptation.common import cutoff_at_category_fpr, digest, load, safe_data, summarize, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/reddit-student-v2-2026-10-16"
ARCHIVE = ROOT / "datasets/reddit/archive-2026-10-16"
OLD_REDDIT = ROOT / "datasets/reddit/data"
CALIBRATION = Path("/Users/dhruvsheth/Downloads/benchmark/calibration")
FROZEN_SPLITS = ROOT / "artifacts/mirror-evaluation-2026-10-13/reddit-splits.json"
EXTERNAL = ROOT / "datasets/external/data-2026-10-16"
SELECT_DOMAINS = ("archive", "reddit-old", "calibration")
MEAN, STD = [.485, .456, .406], [.229, .224, .225]

PLAN = {
    "seed": 20261016,
    "backboneSha256": CHECKPOINT_SHA,
    "selectPercent": 10,
    "domainMix": {"archive": .45, "reddit-old": .15, "calibration": .15, "openfake": .15, "ntire2026": .10},
    "weighting": "per domain: equal labels, then equal categories, then equal sources, then 1/images per author; capped at 8x the median",
    "linearProbe": {"epochs": 1, "lr": 1e-3},
    "fineTune": {"epochs": 6, "baseLr": 3e-5, "layerDecay": .8, "headLr": 3e-4, "weightDecay": .05, "warmupSteps": 300, "schedule": "cosine"},
    "drawsPerEpoch": 24000,
    "batch": 32,
    "labelSmoothing": .03,
    "augmentation": {
        "width": "no-upscale choice of native/320/640/960/1080",
        "encode": "previews: 25% unchanged, 35% WebP q55-95, 30% JPEG q55-95, 10% JPEG then WebP; calibration originals always downscaled to <=1080 wide and re-encoded",
        "geometry": "short edge 256 (70%) or 224-320 (30%), random 224 crop; no flips (text orientation)",
        "classIndependent": True,
    },
    "selection": "epoch with the highest select partial AUROC at 1% FPR; threshold caps every select real category at 1% FPR",
}


def bucket(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 100


def archive_rows(development_authors: set[str]) -> list[dict]:
    rows = []
    for line in (ARCHIVE / "manifest.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # the collector may be mid-append
        if r["basis"] == "realorai-verified":
            continue
        if r["split"].startswith("test"):
            split = "final"
        elif r["group"] in development_authors or bucket("select:" + r["group"]) >= PLAN["selectPercent"]:
            split = "fit"
        else:
            split = "select"
        rows.append({"id": "archive:" + r["id"], "domain": "archive", "split": split, "label": r["label"],
                     "category": r["category"], "source": r["sourceKey"], "sourceGroup": "archive:" + r["group"],
                     "root": str(ARCHIVE), "path": str(ARCHIVE / r["preview"]["path"]), "sourceHash": r["preview"]["sha256"],
                     "originalFile": False, "finalSplit": r["split"]})
    return rows


def old_reddit_rows() -> list[dict]:
    eligible = {r["id"]: r["eligibleForWeakSupervision"] for r in load(ROOT / "artifacts/reddit-adaptation-2026-10-14/label-audit.json")["rows"]}
    rows = []
    for r in load(ROOT / "artifacts/reddit-adaptation-2026-10-14/image-map.json")["rows"]:
        if r["role"] != "preview" or r["usageSplit"] not in ("development", "tuning") or not eligible[r["id"]]:
            continue
        rows.append({"id": "reddit:" + r["id"], "domain": "reddit-old", "split": "fit" if r["usageSplit"] == "development" else "select",
                     "label": r["label"], "category": r["category"], "source": r["source"], "sourceGroup": "reddit:" + r["sourceGroup"],
                     "root": str(OLD_REDDIT), "path": str(OLD_REDDIT / r["preview"]["path"]), "sourceHash": r["sourceHash"], "originalFile": False})
    return rows


def calibration_rows() -> list[dict]:
    rows = []
    for r in load(ROOT / "artifacts/ferretnet-improvement-2026-10-11/manifest.json")["rows"]:
        if r["experimentSplit"] not in ("development", "tuning"):
            continue
        relative = Path(r["file"])
        assert relative.parts[0] == "calibration"
        rows.append({"id": "calibration:" + r["file"], "domain": "calibration", "split": "fit" if r["experimentSplit"] == "development" else "select",
                     "label": r["label"], "category": r["group"], "source": "calibration-" + r["label"], "sourceGroup": "calibration:" + r["sourceGroup"],
                     "root": str(CALIBRATION), "path": str(CALIBRATION / Path(*relative.parts[1:])), "sourceHash": r["sourceHash"], "originalFile": True})
    return rows


def external_rows() -> list[dict]:
    manifest = EXTERNAL / "manifest.jsonl"
    if not manifest.exists():
        return []
    rows = []
    for line in manifest.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # the external collector may be mid-append
        rows.append({"id": r["id"], "domain": r["dataset"], "split": "select" if bucket("select:" + r["group"]) < PLAN["selectPercent"] else "fit",
                     "label": r["label"], "category": r["category"], "source": r["source"], "sourceGroup": r["group"],
                     "root": str(EXTERNAL), "path": str(EXTERNAL / r["path"]), "sourceHash": r["sha256"], "originalFile": True})
    return rows


def all_rows() -> list[dict]:
    development = {r["group"] for r in load(FROZEN_SPLITS)["rows"] if r["split"] == "development"}
    rows = archive_rows(development) + old_reddit_rows() + calibration_rows() + external_rows()
    # Unseen-source communities are held out whole, so their authors' posts elsewhere leave training.
    final_groups = {r["sourceGroup"] for r in rows if r["split"] == "final"}
    rows = [r for r in rows if r["split"] == "final" or r["sourceGroup"] not in final_groups]
    fit = {r["sourceGroup"] for r in rows if r["split"] == "fit"}
    held = {r["sourceGroup"] for r in rows if r["split"] != "fit"}
    if fit & held:
        raise ValueError(f"{len(fit & held)} author groups cross the fit/select/final boundary")
    return rows


def sample_weights(rows: list[dict]) -> np.ndarray:
    weights = np.zeros(len(rows))
    by_domain = defaultdict(list)
    for i, r in enumerate(rows):
        by_domain[r["domain"]].append(i)
    for domain, ids in by_domain.items():
        domain_weights = np.zeros(len(ids))
        for label in ("real", "ai"):
            members = [j for j, i in enumerate(ids) if rows[i]["label"] == label]
            if not members:
                raise ValueError(f"{domain} has no {label} rows")
            categories = Counter(rows[ids[j]]["category"] for j in members)
            sources = {c: Counter(rows[ids[j]]["source"] for j in members if rows[ids[j]]["category"] == c) for c in categories}
            groups = Counter((rows[ids[j]]["source"], rows[ids[j]]["sourceGroup"]) for j in members)
            groups_per_source = Counter(source for source, _ in groups)
            raw = np.array([1 / (len(categories) * len(sources[rows[ids[j]]["category"]]) * groups_per_source[rows[ids[j]]["source"]]
                                 * groups[(rows[ids[j]]["source"], rows[ids[j]]["sourceGroup"])]) for j in members])
            raw = np.minimum(raw, 8 * np.median(raw))
            domain_weights[members] = .5 * raw / raw.sum()
        weights[ids] = PLAN["domainMix"][domain] * domain_weights
    return weights / weights.sum() * len(weights)


def reencode(image: Image.Image, kind: str) -> Image.Image:
    stream = io.BytesIO()
    if kind == "webp":
        image.save(stream, "WEBP", quality=random.randint(55, 95), method=2)
    else:
        image.save(stream, "JPEG", quality=random.randint(55, 95))
    with Image.open(io.BytesIO(stream.getvalue())) as encoded:
        return encoded.convert("RGB")


def shrink(image: Image.Image, width: int) -> Image.Image:
    if width >= image.width:
        return image
    return image.resize((width, max(1, round(image.height * width / image.width))), Image.Resampling.LANCZOS)


class TrainingImages(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        image = decode_image(safe_data(r))[0]
        if r["originalFile"]:
            image = reencode(shrink(image, 1080), "webp")
        image = shrink(image, random.choice([image.width, 320, 640, 960, 1080]))
        choice = random.random()
        if r["originalFile"] or choice >= .25:
            if choice < .6:
                image = reencode(image, "webp")
            elif choice < .9:
                image = reencode(image, "jpeg")
            else:
                image = reencode(reencode(image, "jpeg"), "webp")
        short = 256 if random.random() < .7 else random.randint(224, 320)
        image = TF.resize(image, short, interpolation=InterpolationMode.BILINEAR, antialias=True)
        top, left = random.randint(0, image.height - 224), random.randint(0, image.width - 224)
        pixels = TF.normalize(TF.to_tensor(TF.crop(image, top, left, 224, 224)), MEAN, STD)
        return pixels, torch.tensor(float(r["label"] == "ai"))


class ScoringImages(Dataset):
    """Inference transform of the deployed backend: Resize(256), CenterCrop(224)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = decode_image(safe_data(self.rows[index]))[0]
        image = TF.center_crop(TF.resize(image, 256, interpolation=InterpolationMode.BILINEAR, antialias=True), 224)
        return TF.normalize(TF.to_tensor(image), MEAN, STD)


def worker_seed(worker: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def score(model, device, rows: list[dict], workers: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(ScoringImages(rows), batch_size=64, num_workers=workers)
    return np.concatenate([model(pixels.to(device)).float().flatten().cpu().numpy() for pixels in loader])


def metrics(logits: np.ndarray, rows: list[dict], cut: float | None = None) -> dict:
    keyed = [{**r, "category": f"{r['domain']}:{r['category']}"} for r in rows]
    if cut is None:
        cut = cutoff_at_category_fpr(logits, [r["label"] == "ai" for r in keyed], [r["category"] for r in keyed])
    domains = {d: summarize(logits[[i for i, r in enumerate(keyed) if r["domain"] == d]], [r for r in keyed if r["domain"] == d], cut)
               for d in sorted({r["domain"] for r in keyed})}
    return {"cutLogit": cut, "pooled": summarize(logits, keyed, cut), "domains": domains}


def parameter_groups(model) -> list[dict]:
    tune = PLAN["fineTune"]
    depth = len(model.blocks)
    groups = [{"params": list(model.head.parameters()), "lr": tune["headLr"], "weight_decay": 0.0},
              {"params": list(model.norm.parameters()), "lr": tune["baseLr"], "weight_decay": 0.0}]
    for i, block in enumerate(model.blocks):
        lr = tune["baseLr"] * tune["layerDecay"] ** (depth - 1 - i)
        decay = [p for n, p in block.named_parameters() if p.ndim > 1]
        flat = [p for n, p in block.named_parameters() if p.ndim <= 1]
        groups += [{"params": decay, "lr": lr, "weight_decay": tune["weightDecay"]}, {"params": flat, "lr": lr, "weight_decay": 0.0}]
    embed = [model.cls_token, model.pos_embed, *model.patch_embed.parameters()]
    groups.append({"params": embed, "lr": tune["baseLr"] * tune["layerDecay"] ** depth, "weight_decay": 0.0})
    for group in groups:
        group["initialLr"] = group["lr"]
    return groups


def run_epochs(model, device, optimizer, loader, epochs: int, schedule_steps: int | None, label: str, status: dict, smoke: bool):
    smoothing = PLAN["labelSmoothing"]
    step_count = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for step, (pixels, labels) in enumerate(loader):
            if schedule_steps:
                global_step = status.setdefault("globalStep", 0)
                warmup = PLAN["fineTune"]["warmupSteps"]
                factor = global_step / warmup if global_step < warmup else .5 * (1 + math.cos(math.pi * (global_step - warmup) / max(1, schedule_steps - warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = group["initialLr"] * factor
                status["globalStep"] = global_step + 1
            targets = labels.to(device) * (1 - smoothing) + smoothing / 2
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(pixels.to(device)).flatten(), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step_count += 1
            if (step + 1) % 50 == 0:
                status.update({"phase": label, "epoch": epoch + 1, "step": step + 1, "steps": len(loader), "meanLoss": float(np.mean(losses[-200:])),
                               "seconds": round(time.time() - status["started"])})
                write(OUT / "status.json", status)
                print(json.dumps(status), flush=True)
            if smoke and step >= 5:
                break
        yield epoch + 1, float(np.mean(losses))


def save(model, name: str) -> tuple[str, Path]:
    folder = OUT / "models"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{name}.partial.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temporary))
    sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
    target = folder / f"{name}-{sha[:12]}.safetensors"
    temporary.replace(target)
    return sha, target


def train(args):
    random.seed(PLAN["seed"])
    np.random.seed(PLAN["seed"])
    torch.manual_seed(PLAN["seed"])
    rows = all_rows()
    fit = [r for r in rows if r["split"] == "fit"]
    select = [r for r in rows if r["split"] == "select"]
    if args.smoke:
        rng = random.Random(0)
        fit, select = rng.sample(fit, min(512, len(fit))), rng.sample(select, min(256, len(select)))
    counts = {split: dict(Counter(f"{r['domain']}/{r['label']}" for r in rows if r["split"] == split)) for split in ("fit", "select", "final")}
    plan = {**PLAN, "fitHash": digest([r["id"] for r in fit]), "selectHash": digest([r["id"] for r in select]), "counts": counts, "smoke": args.smoke}
    write(OUT / ("smoke-plan.json" if args.smoke else "plan.json"), plan)
    print(json.dumps(counts), flush=True)

    backbone = CommunityBackbone()
    model, device = backbone.model, backbone.device
    draws = 64 * 8 if args.smoke else min(PLAN["drawsPerEpoch"], 4 * len(fit))
    sampler = WeightedRandomSampler(torch.tensor(sample_weights(fit), dtype=torch.float64), draws, replacement=True,
                                    generator=torch.Generator().manual_seed(PLAN["seed"]))
    loader = DataLoader(TrainingImages(fit), batch_size=PLAN["batch"], sampler=sampler, num_workers=args.workers,
                        worker_init_fn=worker_seed, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    status = {"started": time.time()}
    history = []

    model.requires_grad_(False)
    model.head.requires_grad_(True)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    probe = torch.optim.AdamW(model.head.parameters(), lr=PLAN["linearProbe"]["lr"], weight_decay=0.0)
    for epoch, loss in run_epochs(model, device, probe, loader, PLAN["linearProbe"]["epochs"], None, "linear-probe", status, args.smoke):
        history.append({"phase": "linear-probe", "epoch": epoch, "meanLoss": loss})

    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameter_groups(model))
    epochs = 1 if args.smoke else PLAN["fineTune"]["epochs"]
    best = None
    for epoch, loss in run_epochs(model, device, optimizer, loader, epochs, epochs * len(loader), "fine-tune", status, args.smoke):
        logits = score(model, device, select, args.workers)
        primary = np.array([r["domain"] in SELECT_DOMAINS for r in select])
        result = metrics(logits[primary], [r for r, keep in zip(select, primary) if keep])
        if (~primary).any():
            result["external"] = metrics(logits[~primary], [r for r, keep in zip(select, primary) if not keep], result["cutLogit"])
        entry = {"phase": "fine-tune", "epoch": epoch, "meanLoss": loss, "selectPartialAuRoc01": result["pooled"]["partialAuRoc01"],
                 "selectAuroc": result["pooled"]["auroc"], "selectRecallAtCut": result["pooled"]["recall"]}
        history.append(entry)
        print("SELECT", json.dumps(entry), flush=True)
        if best is None or entry["selectPartialAuRoc01"] > best["selectPartialAuRoc01"]:
            sha, path = save(model, "smoke" if args.smoke else f"{args.name}-epoch{epoch}")
            best = {**entry, "weightsSha256": sha, "weightsPath": str(path.relative_to(ROOT)), "select": result,
                    "selectPredictions": [{"id": r["id"], "logit": float(v)} for r, v in zip(select, logits)]}
        write(OUT / ("smoke-training.json" if args.smoke else "training.json"), {"planHash": digest(plan), "history": history, "best": {k: v for k, v in best.items() if k != "selectPredictions"}})
    if args.smoke:
        return
    write(OUT / "select-predictions.json", {"weightsSha256": best["weightsSha256"], "rows": best["selectPredictions"]})
    profile = {"schemaVersion": 1, "kind": "reddit-student", "model": f"reddit-{args.name}", "pipeline": "community-vit-s224-full-finetune-v2",
               "checkpointSha256": best["weightsSha256"], "sourceCheckpointSha256": CHECKPOINT_SHA, "checkpointPath": best["weightsPath"],
               "logitOffset": best["select"]["cutLogit"], "threshold": .5, "trainingHash": digest(plan)}
    profile["modelHash"] = digest(profile)
    write(OUT / "student-profile.json", profile)
    write(OUT / "status.json", {**status, "phase": "complete", "seconds": time.time() - status["started"]})
    print("COMPLETE", json.dumps({k: v for k, v in best.items() if k not in ("select", "selectPredictions")}), flush=True)


def final(args):
    """Score the frozen archive test splits once with the selected checkpoint and threshold.

    With --fresh-after, only final rows absent from that earlier final-test file are scored.
    """
    target = OUT / ("final-fresh-test.json" if args.fresh_after else "final-test.json")
    if target.exists():
        raise SystemExit(f"{target} already exists; the final test is scored once.")
    already = {r["id"] for r in load(ROOT / args.fresh_after)["rows"]} if args.fresh_after else set()
    profile = load(OUT / "student-profile.json")
    backbone = CommunityBackbone()
    path = ROOT / profile["checkpointPath"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile["checkpointSha256"]:
        raise ValueError("Checkpoint changed after selection")
    backbone.model.load_state_dict(load_file(str(path)), strict=True)
    rows = [r for r in all_rows() if r["split"] == "final" and r["id"] not in already]
    logits = score(backbone.model, backbone.device, rows, args.workers)
    cut = profile["logitOffset"]
    result = {split: metrics(logits[[i for i, r in enumerate(rows) if r["finalSplit"] == split]], [r for r in rows if r["finalSplit"] == split], cut)
              for split in sorted({r["finalSplit"] for r in rows})}
    write(target, {"profileHash": profile["modelHash"], "excludedScoredFile": args.fresh_after, "results": result, "rows": [{"id": r["id"], "logit": float(v)} for r, v in zip(rows, logits)]})
    print(json.dumps({s: {k: v["pooled"][k] for k in ("recall", "fpr", "precision", "auroc", "partialAuRoc01")} for s, v in result.items()}, indent=1))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="a few steps on a small sample to check the pipeline")
    parser.add_argument("--final", action="store_true", help="score the frozen archive test splits once")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(OUT.relative_to(ROOT)), help="artifact folder, relative to the project")
    parser.add_argument("--name", default="student-v2", help="checkpoint prefix; the profile model id is reddit-<name>")
    parser.add_argument("--draws", type=int, default=PLAN["drawsPerEpoch"], help="training images drawn per epoch")
    parser.add_argument("--fresh-after", help="with --final: skip rows already scored in this final-test JSON")
    parser.add_argument("--mix", help='JSON domain sampling weights, e.g. \'{"archive": .7, "reddit-old": .1, ...}\'')
    args = parser.parse_args()
    OUT = ROOT / args.out
    PLAN["drawsPerEpoch"] = args.draws
    if args.mix:
        mix = json.loads(args.mix)
        if set(mix) != set(PLAN["domainMix"]) or abs(sum(mix.values()) - 1) > 1e-9:
            raise SystemExit(f"--mix needs exactly {sorted(PLAN['domainMix'])} summing to 1")
        PLAN["domainMix"] = mix
    final(args) if args.final else train(args)


if __name__ == "__main__":
    main()
