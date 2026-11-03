#!/usr/bin/env python3
"""
Downloads images from r/aiArt using Reddrix (https://github.com/Coding-Doctor-Omar/reddrix)
and sorts them into folders by post flair (the flair is where people say which AI they used).

Setup (Reddrix needs Python 3.10+, so use your Python 3.14, not /usr/bin/python3):
    python3 -m pip install reddrix

Step 1 - see which flairs exist, without downloading anything:
    python3 reddit_aiart.py --survey

Step 2 - download for real:
    python3 reddit_aiart.py

Output: ~/Desktop/random_artworks/reddit_aiart/<flair>/<postid>_<n>.<ext>
        plus manifest.csv (folder, original flair, title, score, post URL, file) in the same folder.
"""

import asyncio
import csv
import html
import os
import re
import sys
import urllib.request
from collections import Counter

from reddrix import Reddrix

# ---------------- settings ----------------
SUBREDDIT_URL = "https://www.reddit.com/r/aiArt"
OUT_DIR = os.path.expanduser("~/Desktop/random_artworks/reddit_aiart")
MAX_POSTS = 1200             # how many posts to scan (not all will be images with flairs)
SORT = "default"            # "default", "new" or "hot"
TIME_FRAME = "past_year"     # "all_time", "past_year", "past_month", "past_week", "today", "past_hour"
MIN_SCORE = 3             # skip posts with fewer upvotes (weeds out junk)
ALLOW_NSFW = True
MAIN_FLAIRS = ["chatgpt", "flux", "qwen", "gemini"]   # each gets its own folder
MISC_FOLDER = "miscellaneous"                         # every OTHER flaired AI model goes here
SKIP_FLAIRS = [             # never downloaded: unflaired, placeholder, and non-art flairs
    "unflaired", "other_please_edit_or_your_post_may_be_deleted",
    "video", "politics", "music", "text", "announcement", "other_discussion", "gemini_notebook",
    "discussion", "question", "help", "news", "meme", "meta", "tutorial", "tool",
]
MERGE_FLAIRS = {            # combine variants of the same model (keys are cleaned names)
    "google_gemini": "gemini",
    "google_nano_banana": "gemini",
    "google_gemini_at_imastudio": "gemini",
    "krea_2": "krea",
    "krea2": "krea",
    "krea_2_turbo": "krea",
    "stable_diffusion_adobe_firefly": "adobe_firefly",
    "other_google_flow": "google_flow",
    "google_gemini_amp_chatgpt": "mixed",      # flairs naming 2+ models
    "midjourney_to_chatgpt": "mixed",
}
CONCURRENT_DOWNLOADS = 6
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SURVEY_ONLY = "--survey" in sys.argv
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
# ------------------------------------------


def clean_flair(raw):
    """':emoji: Midjourney v7' -> 'midjourney_v7'. Empty -> 'unflaired'."""
    if not raw:
        return "unflaired"
    s = re.sub(r":[A-Za-z0-9_+-]+:", "", raw)          # drop :emoji_codes:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    s = re.sub(r"^image_", "", s) or "unflaired"      # r/aiArt prefixes most flairs with "image_"
    return MERGE_FLAIRS.get(s, s)


def ext_for(url):
    path = url.split("?")[0].lower()
    for e in IMAGE_EXTS:
        if path.endswith(e):
            return ".jpg" if e == ".jpeg" else e
    return ".jpg"


def download_sync(url, dest):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            if "image" not in r.headers.get("Content-Type", ""):
                return False
            data = r.read()
        if len(data) < 3000:
            return False
        with open(dest + ".part", "wb") as f:
            f.write(data)
        os.replace(dest + ".part", dest)
        return True
    except Exception:
        return False


async def grab(sem, url, dest):
    if os.path.exists(dest):
        return True
    async with sem:
        return await asyncio.to_thread(download_sync, url, dest)


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    seen_flairs = Counter()      # every post scanned
    saved = Counter()            # images saved per flair
    pending = []                 # (task, flair, post, dest)
    scanned = 0

    async with Reddrix(
        mode="standard",
        subreddit_urls=[SUBREDDIT_URL],
        sort=SORT,
        time_frame=TIME_FRAME,
        include_comments=False,
        max_results=MAX_POSTS,
        max_posts_per_collection=MAX_POSTS,
        verbose=True,
    ) as rx:
        async for item in rx.get_items():
            if item.get("submissionType") != "post":
                continue
            scanned += 1
            flair = clean_flair(item.get("postFlair"))
            seen_flairs[flair] += 1

            if SURVEY_ONLY:
                continue
            if item.get("postType") not in ("image", "gallery"):
                continue
            if item.get("isNSFW") and not ALLOW_NSFW:
                continue
            if (item.get("score") or 0) < MIN_SCORE:
                continue
            if flair in SKIP_FLAIRS:
                continue
            if flair not in MAIN_FLAIRS:
                flair = MISC_FOLDER

            urls = list(item.get("galleryImagesUrls") or [])
            if not urls and item.get("imageUrl"):
                urls = [item["imageUrl"]]
            folder = os.path.join(OUT_DIR, flair)
            os.makedirs(folder, exist_ok=True)
            for n, u in enumerate(urls, 1):
                u = html.unescape(u)
                dest = os.path.join(folder, f"{item['submissionId']}_{n}{ext_for(u)}")
                pending.append((asyncio.create_task(grab(sem, u, dest)), flair, item, dest))

            if scanned % 25 == 0:
                print(f"scanned {scanned} posts, queued {len(pending)} images...")

    rows = []
    for task, flair, item, dest in pending:
        if await task:
            saved[flair] += 1
            rows.append([flair, clean_flair(item.get("postFlair")), item.get("title"), item.get("score"), item.get("submissionUrl"),
                         os.path.relpath(dest, OUT_DIR)])

    print(f"\nScanned {scanned} posts. Flairs seen (all posts):")
    for f, c in seen_flairs.most_common():
        print(f"  {f:<30} {c}")

    if SURVEY_ONLY:
        print("\nSurvey only, nothing downloaded. Edit MAIN_FLAIRS / SKIP_FLAIRS / MERGE_FLAIRS, then run without --survey.")
        return

    if rows:
        with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["folder", "original_flair", "title", "score", "post_url", "file"])
            w.writerows(rows)
    print(f"\nSaved {sum(saved.values())} images:")
    for f, c in saved.most_common():
        print(f"  {f:<30} {c}")
    print(f"\nFiles are in {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())