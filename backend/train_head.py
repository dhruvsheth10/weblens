"""Build a portable FiSeR head and a measured ensemble operating point."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .assets import DATA, DATA_REPO, DATA_REV, DEAR_SHA, FISER_SHA, MODELS, ROOT, setup_dataset, setup_models, sha256
from .engine import ImageEngine, LAYER, MODEL_ID, PIPELINE, combine, decode_image


def metrics(rows: list[dict], threshold: float) -> dict:
    real = [row for row in rows if row["label"] == 0]
    ai = [row for row in rows if row["label"] == 1]
    return {"images": len(rows), "realImages": len(real), "aiImages": len(ai),
            "falsePositiveRate": sum(row["score"] >= threshold for row in real) / len(real) if real else None,
            "aiRecall": sum(row["score"] >= threshold for row in ai) / len(ai) if ai else None,
            "auroc": float(roc_auc_score([row["label"] for row in rows], [row["score"] for row in rows])) if real and ai else None}


def train() -> None:
    setup_models()
    source_rows = setup_dataset()
    rows, excluded = [], []
    for row in source_rows:
        try:
            decode_image((DATA / row["local"]).read_bytes())
            rows.append(row)
        except ValueError as error:
            excluded.append({"file": row["local"], "split": row["split"], "reason": str(error)})
    print(f"Eligible references: {len(rows)}; excluded by the application's image limits: {len(excluded)}", flush=True)
    engine = ImageEngine(with_head=False)
    print(f"Encoding FiSeR references on {engine.device}…", flush=True)
    cache_path = DATA / "features.npz"
    source_key = hashlib.sha256((FISER_SHA + PIPELINE + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
    features = None
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        if str(cache["sourceKey"].item()) == source_key:
            features = cache["features"]
    if features is None:
        batches = []
        for offset in range(0, len(rows), 4):
            images = [decode_image((DATA / row["local"]).read_bytes())[0] for row in rows[offset:offset + 4]]
            batches.append(engine.encode(images))
            if offset % 40 == 0:
                print(f"FiSeR features: {min(offset + 4, len(rows))}/{len(rows)}", flush=True)
        features = np.concatenate(batches)
        np.savez(cache_path, features=features, sourceKey=source_key)
    training = np.array([row["split"] == "train" for row in rows])
    labels = np.array([row["label"] for row in rows])
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=20261010)
    classifier.fit(features[training], labels[training])
    head = {"kind": "linear", "layer": LAYER, "weight": classifier.coef_[0].tolist(),
            "bias": float(classifier.intercept_[0]), "labels": ["natural", "synthetic"],
            "trainingImages": int(training.sum()), "dataset": DATA_REPO, "datasetRevision": DATA_REV,
            "sourceKey": source_key, "estimator": "LogisticRegression(C=1.0)"}
    head_path = MODELS / "fiser-head.json"
    head_path.write_text(json.dumps(head, indent=2) + "\n")
    raw_fiser = classifier.decision_function(features)
    cache_file = DATA / "dear-logits.json"
    dear_cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    scored = []
    selected = [index for index, row in enumerate(rows) if row["split"] != "train"]
    for count, index in enumerate(selected, 1):
        row = rows[index]
        key = row["sha256"] + ":" + DEAR_SHA + ":" + PIPELINE
        if key not in dear_cache:
            image = decode_image((DATA / row["local"]).read_bytes())[0]
            dear_cache[key] = engine.dear_logit(image)[0]
            cache_file.write_text(json.dumps(dear_cache))
        scored.append({**row, "fiserLogit": float(raw_fiser[index]), "dearLogit": dear_cache[key]})
        if count % 10 == 0:
            print(f"DEAR-r calibration / validation: {count}/{len(selected)}", flush=True)
    calibration_rows = [row for row in scored if row["split"] == "calibration"]
    calibration = {}
    for name, field in (("fiser", "fiserLogit"), ("dear_r", "dearLogit")):
        values = np.array([row[field] for row in calibration_rows], dtype=np.float64)
        center, scale = float(values.mean()), float(values.std())
        if scale < 1e-8:
            raise RuntimeError(f"{name} returned constant calibration logits")
        probe = LogisticRegression(C=1.0, max_iter=2000, random_state=20261010)
        # Regularize on a comparable scale so arbitrary logit units cannot dominate fusion.
        probe.fit(((values - center) / scale).reshape(-1, 1), [row["label"] for row in calibration_rows])
        slope = float(probe.coef_[0, 0]) / scale
        if slope <= 0:
            raise RuntimeError(f"{name} has reversed/absent signal on calibration data. Refusing to invent a positive score.")
        calibration[name] = {"a": slope, "b": float(probe.intercept_[0]) - slope * center,
                             "fitInputMean": center, "fitInputStd": scale}
    for row in scored:
        row["score"], row["logit"], row["componentScores"] = combine(row["fiserLogit"], row["dearLogit"], calibration)
    # Original-image empirical 1% FPR constraint; never tune on validation rows.
    real_scores = sorted((row["score"] for row in calibration_rows if row["label"] == 0), reverse=True)
    threshold = max(.5, float(np.nextafter(real_scores[int(.01 * len(real_scores))], 1.0)))
    if threshold >= 1:
        raise RuntimeError("No deployable calibration threshold exists")
    artifacts = {str(path.relative_to(ROOT)): sha256(path) for path in (
        MODELS / "fiser/model.safetensors", MODELS / "fiser/config.json", MODELS / "fiser/preprocessor_config.json",
        MODELS / "dear_r.pth", head_path,
        ROOT / "vendor/dear/nn_classifier/resnet.py", ROOT / "vendor/dear/detector/corvi_mask_gated_detector.py", ROOT / "engine.py")}
    model_hash = hashlib.sha256(json.dumps({"artifacts": artifacts, "pipeline": PIPELINE, "calibration": calibration}, sort_keys=True).encode()).hexdigest()
    profile = {"model": MODEL_ID, "pipeline": PIPELINE, "modelHash": model_hash, "threshold": threshold,
               "artifacts": artifacts, "calibration": calibration, "fusion": "equal mean of separately Platt-scaled scores",
               "targetEmpiricalFpr": .01, "calibrationImages": len(calibration_rows), "calibrationDevice": engine.device.type, "validationIsUntouched": True}
    (MODELS / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    validation = [row for row in scored if row["split"] == "validation"]
    report = {"profile": profile, "trainingImages": int(training.sum()),
              "calibration": metrics(calibration_rows, threshold), "validation": metrics(validation, threshold),
              "validationByGenerator": {source: metrics([row for row in validation if row["generator"] in {"Real", source}], threshold)
                                        for source in sorted({row["generator"] for row in validation if row["label"] == 1})},
              "excluded": excluded,
              "limitations": "Small public dataset; not a representative Reddit accuracy claim. Original images only. Eligibility exclusions are recorded.",
              "rows": scored}
    (ROOT / "validation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    frontend = ROOT.parent / "src/detection/profile.json"
    frontend.write_text(json.dumps({"modelHash": model_hash, "threshold": threshold}, indent=2) + "\n")
    print(json.dumps({"threshold": threshold, "validation": report["validation"], "byGenerator": report["validationByGenerator"]}, indent=2), flush=True)


if __name__ == "__main__":
    train()
