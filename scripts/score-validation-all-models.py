"""Score every packaged detector on the human-authorized validation folder.

The 2026-10-16 plan stays frozen. This run records a new authorization, reuses the
checksummed manifest, and does not fit a threshold or train anything. Completed
Community + Ferret and Reddit-student jobs are reused rather than repeated.

    backend/.venv/bin/python scripts/score-validation-all-models.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.community import CommunityEngine, CommunityFerretEngine
from backend.engine import ImageEngine
from backend.ferretnet import THRESHOLD as FERRET_THRESHOLD
from backend.ferretnet import FerretNetEngine
from backend.hybrid import THRESHOLD as HYBRID_THRESHOLD
from backend.hybrid import HybridEngine
from backend.mirror import MirrorCommunityEngine, MirrorEngine
from backend.previews import predict_image
from backend.reddit_research import RedditStudentEngine
from benchmark_mcp.core import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/heldout-validation-2026-10-16"
OUT = ROOT / "artifacts/validation-all-models-2026-10-17"
SOURCE = Path("/Users/dhruvsheth/Downloads/benchmark/validation")
VARIANTS = ("original", "reddit-webp640", "reddit-webp960")
AUTHORIZATION = "can you run every single model you have stored on the validation testset now."
REUSED = {
    "community-ferret-preview-v1": "community-ferret",
    "reddit-student-last2-v1": "reddit-student",
}
# User stopped FiSeR on 2026-10-17 after ~5 s/image. Leave it out of this comparison.
SKIPPED = {"fiser-dear-r-v1"}


def checksum(path: Path) -> str:
    digest_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_hash.update(chunk)
    return digest_hash.hexdigest()


def load_manifest() -> list[dict]:
    manifest = read_json(FROZEN / "manifest.json")
    if digest(manifest) != read_json(FROZEN / "manifest-checksum.json")["sha256"]:
        raise ValueError("Frozen validation manifest checksum changed")
    if Path(manifest["source"]).resolve() != SOURCE.resolve():
        raise ValueError("Manifest source is not the validation folder")
    return manifest["rows"]


def packaged():
    profiles = {}
    for path in sorted((ROOT / "backend/models").glob("*-profile.json")):
        profile = json.loads(path.read_text())
        if profile.get("model") and profile.get("threshold") is not None and profile.get("modelHash"):
            profiles[profile["model"]] = profile
    profiles["ferretnet-b-v1"] = {"model": "ferretnet-b-v1", "threshold": FERRET_THRESHOLD, "kind": "ferretnet"}
    legacy = json.loads((ROOT / "backend/models/profile.json").read_text())
    profiles[legacy["model"]] = legacy
    return profiles


def make_engine(model: str, profiles: dict):
    profile = profiles[model]
    if model == "ferretnet-b-v1":
        return FerretNetEngine(), FERRET_THRESHOLD
    if model == "ferretnet-opensynthid-v1":
        return HybridEngine(), HYBRID_THRESHOLD
    if model == "fiser-dear-r-v1":
        engine = ImageEngine()
        return engine, engine.profile["threshold"]
    if model == "community-preview-v1":
        return CommunityEngine(profile), profile["threshold"]
    if model == "community-ferret-preview-v1":
        return CommunityFerretEngine(profile=profile), profile["threshold"]
    if model == "mirror-h-v1":
        return MirrorEngine(profile), profile["threshold"]
    if model == "mirror-community-v1":
        return MirrorCommunityEngine(profile=profile), profile["threshold"]
    if profile.get("kind") == "reddit-student":
        return RedditStudentEngine(profile), profile["threshold"]
    raise ValueError(f"No packaged engine for {model}")


def score_model(model: str, profiles: dict, rows: list[dict]) -> None:
    folder = OUT / "scores" / model
    folder.mkdir(parents=True, exist_ok=True)
    pending = [variant for variant in VARIANTS if not (folder / f"{variant}.complete").exists()]
    if not pending:
        print(f"SKIP {model}", flush=True)
        return
    engine, cutoff = make_engine(model, profiles)
    for variant in pending:
        path = folder / f"{variant}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                if line:
                    done.add(json.loads(line)["file"])
        failures = []
        started = time.perf_counter()
        with path.open("a") as stream:
            for index, row in enumerate(rows, 1):
                if row["file"] in done:
                    continue
                image = SOURCE / row["file"]
                try:
                    if not image.resolve().is_relative_to(SOURCE.resolve()):
                        raise ValueError("path escaped validation root")
                    data = image.read_bytes()
                    if hashlib.sha256(data).hexdigest() != row["sourceHash"]:
                        raise ValueError("image bytes changed")
                    prediction = predict_image(engine, data, variant)
                    record = {"file": row["file"], "label": row["label"], "group": row["group"], "sourceHash": row["sourceHash"],
                              "score": prediction["score"], "logit": prediction["logit"], "model": prediction["model"],
                              "modelHash": prediction["modelHash"], "pipeline": prediction["pipeline"], "ms": prediction.get("elapsedMs")}
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                except Exception as error:  # noqa: BLE001 - one bad file must not stop the comparison
                    failures.append({"file": row["file"], "error": str(error)[:200]})
                if index % 100 == 0:
                    print(f"{model} {variant} {index}/{len(rows)}", flush=True)
        (folder / f"{variant}.failures.json").write_text(json.dumps(failures, indent=1))
        (folder / f"{variant}.complete").write_text(json.dumps({"cutoff": cutoff, "seconds": round(time.perf_counter() - started, 1), "failures": len(failures)}))
        print(f"DONE {model} {variant} failures {len(failures)}", flush=True)


def import_reused(rows: list[dict]) -> None:
    runs = read_json(FROZEN / "runs.json")
    for model, profile_name in REUSED.items():
        folder = OUT / "scores" / model
        folder.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            marker = folder / f"{variant}.complete"
            if marker.exists():
                continue
            run = next(item for item in reversed(runs) if item["profile"] == profile_name and item["variant"] == variant)
            status = read_json(Path(run["artifacts"]) / "status.json")
            if status.get("status") != "complete":
                raise ValueError(f"reused run incomplete: {model} {variant}")
            summary = read_json(Path(run["artifacts"]) / "summary.json")
            lines = []
            for raw in (Path(run["artifacts"]) / "scores.jsonl").read_text().splitlines():
                row = json.loads(raw)
                lines.append(json.dumps({"file": row["file"], "label": row["label"], "group": row["group"], "sourceHash": row["sourceHash"],
                                         "score": row["score"], "logit": row["logit"], "model": row["model"], "modelHash": row["modelHash"],
                                         "pipeline": row["pipeline"], "ms": row.get("timingMs")}))
            (folder / f"{variant}.jsonl").write_text("\n".join(lines) + "\n")
            marker.write_text(json.dumps({"cutoff": summary["threshold"], "reusedRun": run["runId"], "failures": summary["failed"]}))
            print(f"REUSED {model} {variant} n {len(lines)}", flush=True)


def evaluate() -> None:
    review = read_json(FROZEN / "reviewed-data-audit.json")
    scopes = {"full-folder": None, "new-unique-bytes": set(review["exactNewUniqueFiles"])}
    records = []
    for model_dir in sorted((OUT / "scores").iterdir()):
        if model_dir.name in SKIPPED or not model_dir.is_dir():
            continue
        if any(not (model_dir / f"{variant}.complete").exists() for variant in VARIANTS):
            print(f"SKIP incomplete {model_dir.name}", flush=True)
            continue
        for variant in VARIANTS:
            rows = [json.loads(line) for line in (model_dir / f"{variant}.jsonl").read_text().splitlines() if line]
            meta = json.loads((model_dir / f"{variant}.complete").read_text())
            for scope, chosen in scopes.items():
                selected = rows if chosen is None else [row for row in rows if row["file"] in chosen]
                labels = np.array([row["label"] == "ai" for row in selected])
                scores = np.array([row["score"] for row in selected], dtype=float)
                flags = scores >= meta["cutoff"]
                true_positive = int((labels & flags).sum())
                false_positive = int((~labels & flags).sum())
                true_negative = int((~labels & ~flags).sum())
                false_negative = int((labels & ~flags).sum())
                recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
                false_rate = false_positive / (false_positive + true_negative) if false_positive + true_negative else None
                both = len(set(labels)) == 2
                records.append({"model": rows[0]["model"] if rows else model_dir.name, "variant": variant, "scope": scope, "n": len(selected),
                                "ai": int(labels.sum()), "real": int((~labels).sum()), "cutoff": meta["cutoff"], "tp": true_positive, "fp": false_positive,
                                "tn": true_negative, "fn": false_negative, "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
                                "recall": recall, "fpr": false_rate, "auroc": float(roc_auc_score(labels, scores)) if both else None,
                                "partialAuRoc01": float(roc_auc_score(labels, scores, max_fpr=0.01)) if both else None,
                                "averagePrecision": float(average_precision_score(labels, scores)) if labels.any() else None,
                                "medianMs": float(np.median([row["ms"] for row in selected if row.get("ms") is not None])) if any(row.get("ms") is not None for row in selected) else None})
    (OUT / "results.json").write_text(json.dumps({
        "authorization": AUTHORIZATION,
        "skipped": sorted(SKIPPED),
        "skippedReason": "FiSeR was stopped on 2026-10-17 at about 5 seconds per image, before a full variant finished.",
        "records": records,
    }, indent=1))
    print(json.dumps(records, indent=1)[:4000])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    profiles = packaged()
    (OUT / "plan.json").write_text(json.dumps({"authorization": AUTHORIZATION, "authorizationDate": "2026-10-17", "source": str(SOURCE),
                                              "manifestSha256": checksum(FROZEN / "manifest.json"), "models": {name: profile.get("modelHash") for name, profile in profiles.items()},
                                              "thresholds": {name: profile["threshold"] for name, profile in profiles.items()}, "variants": list(VARIANTS),
                                              "note": "Packaged thresholds only. No fitting. Community + Ferret and the Reddit student reuse the 2026-10-16 complete runs."}, indent=1))
    import_reused(rows)
    for model in profiles:
        if model in REUSED or model in SKIPPED:
            continue
        score_model(model, profiles, rows)
    evaluate()


if __name__ == "__main__":
    main()
