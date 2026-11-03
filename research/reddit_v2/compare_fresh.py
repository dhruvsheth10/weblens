"""Score Sherlock 1.0 and Sherlock 2 on archive test rows that Sherlock 1.0 never scored.

    backend/.venv/bin/python -m research.reddit_v2.compare_fresh
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from safetensors.torch import load_file

from backend.community import CommunityBackbone
from research.reddit_adaptation.common import load, write
from research.reddit_v2.train import ROOT, all_rows, metrics, score

PRIOR = ROOT / "artifacts/reddit-student-v2-2026-10-16/final-test.json"
OUT = ROOT / "artifacts/sherlock-2-2026-10-17/fresh-comparison.json"
PROFILES = {
    "sherlock-1.0": ROOT / "backend/models/sherlock-profile.json",
    "sherlock-2": ROOT / "backend/models/sherlock-2-profile.json",
}


def checked_rows() -> list[dict]:
    seen = {row["id"] for row in load(PRIOR)["rows"]}
    return [row for row in all_rows() if row["split"] == "final" and row["id"] not in seen]


def logits_for(profile_path, rows: list[dict]) -> tuple[dict, np.ndarray]:
    profile = load(profile_path)
    path = ROOT / profile["checkpointPath"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile["checkpointSha256"]:
        raise ValueError(f"checkpoint changed: {profile['model']}")
    backbone = CommunityBackbone()
    backbone.model.load_state_dict(load_file(str(path)), strict=True)
    return profile, score(backbone.model, backbone.device, rows, 6)


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists")
    rows = checked_rows()
    results = {}
    stored = []
    for name, profile_path in PROFILES.items():
        profile, logits = logits_for(profile_path, rows)
        by_split = {}
        for split in sorted({row["finalSplit"] for row in rows}):
            mask = np.array([row["finalSplit"] == split for row in rows])
            subset = [row for row, keep in zip(rows, mask) if keep]
            by_split[split] = metrics(logits[mask], subset, profile["logitOffset"])
        results[name] = {"profileHash": profile["modelHash"], "cutLogit": profile["logitOffset"], "splits": by_split}
        stored.append((name, logits))
        for split, result in by_split.items():
            pooled = result["pooled"]
            print(name, split, {key: round(pooled[key], 3) for key in ("auroc", "partialAuRoc01", "recall", "fpr", "precision")}, "FP", pooled["fp"], flush=True)
    write(OUT, {"rows": [{"id": row["id"], **{name: float(values[index]) for name, values in stored}} for index, row in enumerate(rows)], "results": results})


if __name__ == "__main__":
    main()
