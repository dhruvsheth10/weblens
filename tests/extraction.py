#!/usr/bin/env python3
"""
Downloads 100 RANDOM artworks from each of 5 free museum APIs (no keys, no sign-in)
into a single folder. Standard library only, so no pip installs needed.

Run:  python3 random_artworks.py
"""

import json
import math
import os
import random
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PER_SOURCE = 100                                   # images per source
OUT_DIR = os.path.expanduser("~/Desktop/random_artworks")
HIGH_RES = False                                   # True = bigger files, slower
UA = "random-artwork-downloader/1.0 (personal use)"
MAX_ATTEMPTS_FACTOR = 12                           # give up after 12x tries per source


# ---------- helpers ----------

def get_json(url, params=None, retries=3):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def download(url, path, retries=2):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                if "image" not in r.headers.get("Content-Type", ""):
                    return False
                data = r.read()
            if len(data) < 5000:
                return False
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return True
        except Exception:
            time.sleep(1.5 * (i + 1))
    return False


# ---------- sources (each yields random candidates lazily) ----------

def met():
    """The Met: shuffle every object ID, keep public-domain paintings/drawings/prints."""
    data = get_json("https://collectionapi.metmuseum.org/public/collection/v1/objects")
    ids = data["objectIDs"]
    random.shuffle(ids)
    for oid in ids:
        o = get_json(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}", retries=1)
        time.sleep(0.05)
        if not o or not o.get("isPublicDomain"):
            continue
        if not re.search(r"paint|draw|print", o.get("classification", "") or "", re.I):
            continue
        url = o.get("primaryImage") if HIGH_RES else o.get("primaryImageSmall")
        if url:
            yield {"id": oid, "url": url}


def artic():
    """Art Institute of Chicago: random IDs, keep public-domain works that have an image."""
    size = "1686," if HIGH_RES else "843,"
    while True:
        ids = random.sample(range(1, 300000), 100)
        data = get_json(
            "https://api.artic.edu/api/v1/artworks",
            {"ids": ",".join(map(str, ids)), "limit": 100,
             "fields": "id,image_id,is_public_domain"},
        )
        for a in (data or {}).get("data", []):
            if a.get("is_public_domain") and a.get("image_id"):
                yield {"id": a["id"],
                       "url": f"https://www.artic.edu/iiif/2/{a['image_id']}/full/{size}/0/default.jpg"}


def cleveland():
    """Cleveland Museum of Art: random offset into the CC0 + has-image set."""
    base = "https://openaccess-api.clevelandart.org/api/artworks/"
    q = {"has_image": 1, "cc0": 1, "limit": 1}
    info = get_json(base, q)
    total = info["info"]["total"]
    for skip in random.sample(range(total), min(total, PER_SOURCE * MAX_ATTEMPTS_FACTOR)):
        d = get_json(base, {**q, "skip": skip}, retries=1)
        for a in (d or {}).get("data", []):
            imgs = a.get("images") or {}
            key = "print" if HIGH_RES else "web"
            url = (imgs.get(key) or imgs.get("web") or {}).get("url")
            if url:
                yield {"id": a["id"], "url": url}


def vam():
    """Victoria & Albert Museum: random pages of paintings that have images."""
    base = "https://api.vam.ac.uk/v2/objects/search"
    q = {"q": "painting", "images": 1, "page_size": 100}
    first = get_json(base, {**q, "page": 1})
    pages = min(math.ceil(first["info"]["record_count"] / 100), 100)
    order = list(range(1, pages + 1))
    random.shuffle(order)
    size = "full" if HIGH_RES else "!1000,1000"
    for p in order:
        d = get_json(base, {**q, "page": p}, retries=1)
        recs = [r for r in (d or {}).get("records", []) if r.get("_primaryImageId")]
        random.shuffle(recs)
        for r in recs[:8]:  # a few per page so picks are spread out
            yield {"id": r["systemNumber"],
                   "url": f"https://framemark.vam.ac.uk/collections/{r['_primaryImageId']}/full/{size}/0/default.jpg"}


def wellcome():
    """Wellcome Collection: random pages of public-domain / CC0 illustrations & artwork."""
    base = "https://api.wellcomecollection.org/catalogue/v2/images"
    q = {"query": "painting", "locations.license": "pdm,cc-0", "pageSize": 100}
    first = get_json(base, {**q, "page": 1})
    pages = min(first.get("totalPages", 1), 100)
    order = list(range(1, pages + 1))
    random.shuffle(order)
    for p in order:
        d = get_json(base, {**q, "page": p}, retries=1)
        recs = (d or {}).get("results", [])
        random.shuffle(recs)
        for r in recs[:8]:
            thumb = (r.get("thumbnail") or {}).get("url")
            if thumb:
                size = "full" if HIGH_RES else "!1000,1000"
                url = re.sub(r"/full/[^/]+/", f"/full/{size}/", thumb)
                yield {"id": r["id"], "url": url}


SOURCES = [("met", met), ("artic", artic), ("cleveland", cleveland), ("vam", vam), ("wellcome", wellcome)]


# ---------- runner ----------

def run(name, gen_fn):
    got = tried = 0
    seen = set()
    try:
        for item in gen_fn():
            if got >= PER_SOURCE or tried >= PER_SOURCE * MAX_ATTEMPTS_FACTOR:
                break
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            tried += 1
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(item["id"]))
            path = os.path.join(OUT_DIR, f"{name}_{safe_id}.jpg")
            if os.path.exists(path) or download(item["url"], path):
                got += 1
                print(f"[{name}] {got}/{PER_SOURCE}")
    except Exception as e:
        print(f"[{name}] stopped early: {e}")
    print(f"[{name}] DONE: {got} images")
    return got


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
        totals = list(ex.map(lambda s: run(*s), SOURCES))
    print(f"\nFinished: {sum(totals)} images saved to {OUT_DIR}")