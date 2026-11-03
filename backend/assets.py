"""Pinned public assets. Setup downloads them; inference never uses the network."""
from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
FISER_REPO = "heyongxin233/FiSeR-DINOv3-ViT-L16"
FISER_REV = "0c7bac7680d29bc2e1c5f338caa0ea949703a567"
FISER_SHA = "3bfda6c3040fe22f8fe82588bd4c8506572a6844a04c15bb5c8feebbbd9e11a5"
DEAR_REPO = "k-aisi-anti-deepfake/dear-checkpoints"
DEAR_REV = "5b57350b0ee75553109b3844f3c8a9341fc7e707"
DEAR_SHA = "430fde11debe1850ab24af43945b4d68f2bb3ee52a837d84cf525eb603ccde97"
DEAR_CODE_REV = "5e0dc665eee24b632c03be09a27a05319abc9e7f"
FERRETNET_REPO = "xigua7105/FerretNet"
FERRETNET_REV = "e92796c1a2fb07ccc57ecd7e718e6dce067be5fa"
FERRETNET_SHA = "fe755d78370bb6547070329553572405b4ecebd23382c9a6cbb11c4ab85a82c2"
OPENSYNTHID_REPO = "fyxme/opensynthid-detect-0.1"
OPENSYNTHID_REV = "e1384859752c8311b8a168125ac16a3eb240d7f5"
OPENSYNTHID_SHA = "8e910c0e32f86c69761d85ff884dbe0c37e25feac501454080b1ab63fde46496"
DATA_REPO = "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
DATA_REV = "651a325f0ece2744c21d6c68280d2aaac227177b"
COMMUNITY_REPO = 'OwensLab/commfor-model-224'
COMMUNITY_REV = '26afc31e6b40c312c3fd42c05a758be62446215b'
COMMUNITY_SHA = 'a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, expected: str | None = None) -> None:
    if path.exists() and (expected is None or sha256(path) == expected):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    for attempt in range(3):
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
            if expected and sha256(temporary) != expected:
                raise RuntimeError(f"Checksum mismatch for {path.name}")
            temporary.replace(path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def hf_url(repo: str, revision: str, file: str, dataset: bool = False) -> str:
    prefix = "datasets/" if dataset else ""
    return f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{quote(file, safe='/')}"


def setup_models() -> None:
    assets = [
        (hf_url(COMMUNITY_REPO, COMMUNITY_REV, 'model.safetensors'), MODELS/'community224.safetensors', COMMUNITY_SHA),
        (hf_url(COMMUNITY_REPO, COMMUNITY_REV, 'README.md'), MODELS/'licenses/community/README.md', None),
        (hf_url(OPENSYNTHID_REPO, OPENSYNTHID_REV, "model.pt"), MODELS / "opensynthid.pt", OPENSYNTHID_SHA),
        (hf_url(OPENSYNTHID_REPO, OPENSYNTHID_REV, "README.md"), MODELS / "licenses/opensynthid/README.md", None),
        (hf_url(FISER_REPO, FISER_REV, "model.safetensors"), MODELS / "fiser/model.safetensors", FISER_SHA),
        (hf_url(DEAR_REPO, DEAR_REV, "dear_r/model_best.pth"), MODELS / "dear_r.pth", DEAR_SHA),
        (f"https://raw.githubusercontent.com/{FERRETNET_REPO}/{FERRETNET_REV}/checkpoints/4cls_ckpt/ferretnet-b-median-3.pth",
         MODELS / "ferretnet_b.pth", FERRETNET_SHA),
    ]
    for file in ("config.json", "preprocessor_config.json", "README.md"):
        assets.append((hf_url(FISER_REPO, FISER_REV, file), MODELS / "fiser" / file, None))
    for file in ("README.md", "LICENSE", "NOTICE"):
        assets.append((hf_url(DEAR_REPO, DEAR_REV, file), MODELS / "licenses/dear-r" / file, None))
    assets.append((f"https://raw.githubusercontent.com/{FERRETNET_REPO}/{FERRETNET_REV}/LICENSE",
                   MODELS / "licenses/ferretnet/LICENSE", None))
    for file in ("dear/nn_classifier/resnet.py", "dear/detector/corvi_mask_gated_detector.py", "LICENSE"):
        url = f"https://raw.githubusercontent.com/anti-fake/dear/{DEAR_CODE_REV}/{file}"
        assets.append((url, ROOT / "vendor" / file, None))
    for folder in ("vendor/dear", "vendor/dear/detector", "vendor/dear/nn_classifier"):
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").touch()
    print("Verifying FerretNet-B (4 MiB), Community Forensics (83 MiB), OpenSynthID (82 MiB), FiSeR (578 MiB), and DEAR-r (90 MiB)…", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, *asset): asset[1] for asset in assets}
        for future in as_completed(futures):
            future.result()
            print(f"Ready: {futures[future].relative_to(ROOT)}", flush=True)


def setup_dataset() -> list[dict]:
    manifest_path = DATA / "manifest.json"
    if manifest_path.exists():
        rows = json.loads(manifest_path.read_text())
        if all((DATA / row["local"]).is_file() and sha256(DATA / row["local"]) == row["sha256"] for row in rows):
            return rows
    with urlopen(f"https://huggingface.co/api/datasets/{DATA_REPO}/revision/{DATA_REV}?blobs=true", timeout=120) as response:
        files = json.load(response)["siblings"]
    by_stem = {str(Path(entry["rfilename"]).with_suffix("")): entry for entry in files if entry["rfilename"].startswith("images/")}
    indices = list(range(1, 3001))
    random.Random(20261010).shuffle(indices)
    partitions = {"train": indices[:48], "calibration": indices[48:68], "validation": indices[68:88]}
    generators = ("FLUX.2", "GPT Image 2", "Nano Banana 2", "Qwen Image 2.1")
    rows = []
    for split, ids in partitions.items():
        for style in ("Hybrid Images", "Structure Images", "Texture Images"):
            for source in ("Real", *generators):
                selected = ids if source == "Real" else ids[generators.index(source)::4]
                for index in selected:
                    stem = f"images/{source}/{style}/{index:04d}"
                    entry = by_stem.get(stem)
                    if entry is None:
                        raise RuntimeError(f"Missing labeled dataset entry: {stem}")
                    file = entry["rfilename"]
                    rows.append({"remote": file, "local": file, "label": int(source != "Real"),
                                 "generator": source, "style": style, "split": split,
                                 "sourceGroup": str(index), "sha256": entry["lfs"]["sha256"]})
    print(f"Preparing {len(rows)} labeled references with disjoint train/calibration/validation source groups…", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    download(hf_url(DATA_REPO, DATA_REV, "README.md", True), DATA / "README.md")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, hf_url(DATA_REPO, DATA_REV, row["remote"], True), DATA / row["local"], row["sha256"]) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 40 == 0 or index == len(futures):
                print(f"References: {index}/{len(futures)}", flush=True)
    hashes: dict[str, str] = {}
    for row in rows:
        previous = hashes.get(row["sha256"])
        if previous and previous != row["split"]:
            raise RuntimeError("Duplicate bytes cross the dataset split boundary")
        hashes[row["sha256"]] = row["split"]
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    setup_models()
    setup_dataset()
