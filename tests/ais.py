#!/usr/bin/env python3


import csv
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

REPO = "ThreeLiu/Treasure"
OUT_DIR = os.path.expanduser("~/Desktop/random_artworks")   # same folder as the artwork script
N = 100                                                     # images per subfolder
WORKERS = 8

# subfolder name -> folder under fake/ in the dataset
TARGETS = {
    "gpt": "gpt-image-1.5",
    "qwen": "Qwen-Image",
    "flux": "FLUX.2",
}
EXTS = (".jpg", ".jpeg", ".png", ".webp")
MIN_BYTES = 10_000          # skip tiny/broken files

api = HfApi()
_listing_cache = {}


def list_images(generator):
    """All image paths in fake/<generator> (cached). Placeholder folders return []."""
    if generator not in _listing_cache:
        items = api.list_repo_tree(REPO, path_in_repo=f"fake/{generator}",
                                   recursive=True, repo_type="dataset")
        _listing_cache[generator] = [i.path for i in items if i.path.lower().endswith(EXTS)]
    return _listing_cache[generator]


def fetch(repo_path, dest):
    """Download one file and move it to dest. Returns True on success."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = hf_hub_download(REPO, repo_path, repo_type="dataset", local_dir=tmp)
            if os.path.getsize(p) < MIN_BYTES:
                return False
            shutil.move(p, dest)
        return True
    except Exception as e:
        print(f"   skipped {repo_path}: {type(e).__name__}")
        return False


def download_batch(jobs):
    """jobs = [(repo_path, dest)] -> list of bools."""
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(lambda j: fetch(*j), jobs))


def do_labeled(sub, generator):
    folder = os.path.join(OUT_DIR, sub)
    os.makedirs(folder, exist_ok=True)
    files = list_images(generator)
    if not files:
        print(f"[{sub}] no images found for fake/{generator}")
        return
    random.shuffle(files)
    got, i = 0, 0
    while got < N and i < len(files):
        batch = files[i:i + (N - got)]
        i += len(batch)
        jobs = [(f, os.path.join(folder, f"{sub}_{os.path.basename(f)}")) for f in batch]
        got += sum(download_batch(jobs))
        print(f"[{sub}] {got}/{N}")
    print(f"[{sub}] DONE: {got} images from {generator}")


def do_misc():
    """Random images from every OTHER included generator. Source goes in manifest.csv, not filenames."""
    folder = os.path.join(OUT_DIR, "misc")
    os.makedirs(folder, exist_ok=True)
    skip = set(TARGETS.values())
    gens = [os.path.basename(f.path) for f in api.list_repo_tree(REPO, path_in_repo="fake", repo_type="dataset")
            if os.path.basename(f.path) not in skip and not f.path.endswith(".md")]
    random.shuffle(gens)
    usable = []
    for g in gens:                       # keep only folders that actually contain images
        if list_images(g):
            usable.append(g)
            #usable.append(g)
    if not usable:
        print("[misc] no usable generator folders found")
        return

    rows, seen, got, attempts = [], set(), 0, 0
    while got < N and attempts < N * 10:
        jobs = []
        for _ in range(N - got):
            g = random.choice(usable)
            f = random.choice(list_images(g))
            attempts += 1
            if f in seen:
                continue
            seen.add(f)
            dest = os.path.join(folder, f"misc_{len(seen):04d}{os.path.splitext(f)[1].lower()}")
            jobs.append((f, dest, g))
        results = download_batch([(f, d) for f, d, _ in jobs])
        for ok, (f, d, g) in zip(results, jobs):
            if ok:
                got += 1
                rows.append([os.path.basename(d), g, f])
        print(f"[misc] {got}/{N}")

    with open(os.path.join(folder, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "generator", "dataset_path"])
        w.writerows(rows)
    print(f"[misc] DONE: {got} images (generator list in misc/manifest.csv)")


def preflight():
    try:
        next(iter(api.list_repo_tree(REPO, path_in_repo="fake", repo_type="dataset")), None)
    except Exception as e:
        print("\nCouldn't access the dataset. It's gated, so you need to:")
        print("  1. Log in to huggingface.co and request access at https://huggingface.co/datasets/ThreeLiu/Treasure")
        print("  2. Run `hf auth login` (or set HF_TOKEN) in this terminal")
        print(f"\n({type(e).__name__}: {e})")
        sys.exit(1)


if __name__ == "__main__":
    random.seed()
    os.makedirs(OUT_DIR, exist_ok=True)
    preflight()
    for sub, gen in TARGETS.items():
        do_labeled(sub, gen)
    do_misc()
    print(f"\nAll done. Files are in {OUT_DIR}")