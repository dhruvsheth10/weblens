"""Score reddit-student-last2-v1 at its packaged cutoff on the rows v2's final test used.

    backend/.venv/bin/python -m research.reddit_v2.compare_v1
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from safetensors.torch import load_file

from backend.community import CommunityBackbone
from research.reddit_adaptation.common import load, write
from research.reddit_v2.train import OUT, ROOT, all_rows, metrics, score


def main():
    final = load(OUT / "final-test.json")
    profile = load(ROOT / "artifacts/reddit-adaptation-2026-10-14/student-profile.json")
    path = ROOT / profile["checkpointPath"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile["checkpointSha256"]:
        raise ValueError("v1 checkpoint changed")
    backbone = CommunityBackbone()
    backbone.model.load_state_dict(load_file(str(path)), strict=True)
    by_id = {r["id"]: r for r in all_rows() if r["split"] == "final"}
    rows = [by_id[p["id"]] for p in final["rows"]]
    v2 = np.array([p["logit"] for p in final["rows"]])
    v1 = score(backbone.model, backbone.device, rows, 6) - profile["logitOffset"]
    v2_cut = load(OUT / "student-profile.json")["logitOffset"]
    result = {}
    for split in sorted({r["finalSplit"] for r in rows}):
        mask = np.array([r["finalSplit"] == split for r in rows])
        subset = [r for r, keep in zip(rows, mask) if keep]
        result[split] = {"v1": metrics(v1[mask], subset, 0.0), "v2": metrics(v2[mask], subset, v2_cut)}
    write(OUT / "final-test-v1-comparison.json", {"v1ProfileHash": profile["modelHash"], "results": result})
    for split, models in result.items():
        for name, m in models.items():
            pooled = m["pooled"]
            fps = {c.split(":", 1)[1]: v["fp"] for c, v in pooled["categories"].items() if v["realN"] and v["fp"]}
            print(split, name, {k: round(pooled[k], 3) for k in ("auroc", "partialAuRoc01", "recall", "fpr", "precision")}, "FP", pooled["fp"], fps)


if __name__ == "__main__":
    main()
