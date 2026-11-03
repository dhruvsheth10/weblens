"""Review Sherlock 1.0's held-out test errors with the AI or Not API.

    python3.14 research/reviewers/aiornot.py            # query (cached, resumable)
    python3.14 research/reviewers/aiornot.py --report   # compile the report only

Keys are read from aiornot.key and aiornot2.key in the project root and used in turn.
Only the ai_generated report is requested. Results are cached by image sha256 in
artifacts/aiornot-review-2026-10-17/results.jsonl, so reruns never pay twice.

Review set, drawn from v2's scored final-test rows (no model or threshold is fitted here):
  false-flag    every real-labeled image Sherlock flags
  missed-ai     the 60 AI-labeled images Sherlock scores lowest
  control-real  50 random real-labeled images Sherlock leaves unflagged
  control-ai    50 random AI-labeled images Sherlock flags
The controls measure AI or Not's own agreement with the source labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "artifacts/aiornot-review-2026-10-17"
V2 = PROJECT / "artifacts/reddit-student-v2-2026-10-16"
ARCHIVE = PROJECT / "datasets/reddit/archive-2026-10-16"
ENDPOINT = "https://api.aiornot.com/v2/image/sync?only=ai_generated"
KEY_FILES = [PROJECT / "aiornot.key", PROJECT / "aiornot2.key"]


def review_rows() -> list[dict]:
    manifest = {}
    for line in (ARCHIVE / "manifest.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        manifest["archive:" + r["id"]] = r
    cut = json.loads((V2 / "student-profile.json").read_text())["logitOffset"]
    scored = []
    for p in json.loads((V2 / "final-test.json").read_text())["rows"]:
        r = manifest[p["id"]]
        scored.append({"id": p["id"], "logit": p["logit"], "flagged": p["logit"] >= cut, "label": r["label"], "category": r["category"],
                       "subreddit": r["subreddit"], "split": r["split"], "permalink": r["permalink"],
                       "path": str(ARCHIVE / r["preview"]["path"]), "sha256": r["preview"]["sha256"]})
    rng = random.Random(20261017)
    picks = [{**r, "bucket": "false-flag"} for r in scored if r["label"] == "real" and r["flagged"]]
    missed = sorted((r for r in scored if r["label"] == "ai" and not r["flagged"]), key=lambda r: r["logit"])[:60]
    picks += [{**r, "bucket": "missed-ai"} for r in missed]
    picks += [{**r, "bucket": "control-real"} for r in rng.sample([r for r in scored if r["label"] == "real" and not r["flagged"]], 50)]
    picks += [{**r, "bucket": "control-ai"} for r in rng.sample([r for r in scored if r["label"] == "ai" and r["flagged"]], 50)]
    return picks


def multipart(data: bytes, name: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class Client:
    def __init__(self):
        self.keys = [path.read_text().strip() for path in KEY_FILES if path.exists()]
        if not self.keys:
            raise SystemExit("No AI or Not key files found")
        self.dead: set[int] = set()
        self.turn = 0
        self.lock = threading.Lock()

    def next_key(self) -> int | None:
        with self.lock:
            live = [i for i in range(len(self.keys)) if i not in self.dead]
            if not live:
                return None
            self.turn += 1
            return live[self.turn % len(live)]

    def check(self, row: dict) -> dict | None:
        data = Path(row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"changed image {row['id']}")
        body, content_type = multipart(data, Path(row["path"]).name)
        last = "keys exhausted"
        for attempt in range(4):
            key = self.next_key()
            if key is None:
                return None
            request = urllib.request.Request(ENDPOINT, data=body, method="POST",
                                             headers={"Authorization": f"Bearer {self.keys[key]}", "Content-Type": content_type})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read())
                report = result["report"]["ai_generated"]
                return {"sha256": row["sha256"], "key": key, "verdict": report["verdict"], "aiConfidence": report["ai"]["confidence"],
                        "generator": report.get("generator"), "requestId": result.get("id")}
            except urllib.error.HTTPError as error:
                detail = error.read()[:300].decode(errors="replace")
                if error.code in (401, 402, 403):
                    print(f"key {key + 1} unusable ({error.code}): {detail}", flush=True)
                    with self.lock:
                        self.dead.add(key)
                elif error.code == 429:
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"{row['id']}: HTTP {error.code} {detail}", flush=True)
                    return {"sha256": row["sha256"], "key": key, "error": f"HTTP {error.code}: {detail}"}
            except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as error:
                time.sleep(5 * (attempt + 1))
                last = str(error)
        return {"sha256": row["sha256"], "error": f"failed after retries: {last}"}


def query(rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "results.jsonl"
    cached = {json.loads(l)["sha256"] for l in cache_path.read_text().splitlines()} if cache_path.exists() else set()
    todo = [r for r in rows if r["sha256"] not in cached]
    print(f"{len(rows)} review rows, {len(todo)} to query", flush=True)
    client, lock = Client(), threading.Lock()

    def work(row):
        result = client.check(row)
        if result is None:
            return False
        if "error" not in result:
            with lock, cache_path.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
        return True

    with ThreadPoolExecutor(len(client.keys)) as pool:
        done = 0
        for ok in pool.map(work, todo):
            done += 1
            if not ok:
                print("all keys exhausted; stopping", flush=True)
                break
            if done % 25 == 0:
                print(f"queried {done}/{len(todo)}", flush=True)


def report(rows: list[dict]):
    cache_path = OUT / "results.jsonl"
    results = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            results[r["sha256"]] = r
    joined = [{**row, **{k: v for k, v in results[row["sha256"]].items() if k != "sha256"}} for row in rows if row["sha256"] in results]
    buckets = {}
    for name in ("control-real", "control-ai", "false-flag", "missed-ai"):
        members = [r for r in joined if r["bucket"] == name]
        buckets[name] = {"reviewed": len(members), "planned": sum(r["bucket"] == name for r in rows),
                         "verdicts": dict(Counter(r["verdict"] for r in members)),
                         "byCategory": {c: dict(Counter(r["verdict"] for r in members if r["category"] == c)) for c in sorted({r["category"] for r in members})}}
    (OUT / "review.json").write_text(json.dumps({"buckets": buckets, "rows": joined}, indent=1))
    lines = ["# AI or Not review of Sherlock 1.0 held-out test errors (2026-10-17)", "",
             "Rows come from v2's one-time final test (archive test + test-unseen-source previews). Labels are source-derived.",
             "AI or Not verdicts are a second opinion, not ground truth; the control rows measure how often it agrees with the source labels.", "",
             "| Bucket | Reviewed | AI or Not says AI | says human | unknown |", "|---|---|---|---|---|"]
    for name, b in buckets.items():
        v = b["verdicts"]
        lines.append(f"| {name} | {b['reviewed']}/{b['planned']} | {v.get('ai', 0)} | {v.get('human', 0)} | {v.get('unknown', 0)} |")
    lines += ["", "## Per row (false flags and misses)", "", "| Bucket | Category | Subreddit | Sherlock logit | AI or Not | AI conf. | Post |", "|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in joined if r["bucket"] in ("false-flag", "missed-ai")), key=lambda r: (r["bucket"], -r["logit"])):
        lines.append(f"| {r['bucket']} | {r['category']} | {r['subreddit']} | {r['logit']:.2f} | {r['verdict']} | {r['aiConfidence']:.2f} | [link]({r['permalink']}) |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: (b["reviewed"], b["verdicts"]) for k, b in buckets.items()}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int, help="query only the first N review rows (for a smoke test)")
    args = parser.parse_args()
    rows = review_rows()
    if not args.report:
        query(rows[: args.limit] if args.limit else rows)
    report(rows)


if __name__ == "__main__":
    main()
