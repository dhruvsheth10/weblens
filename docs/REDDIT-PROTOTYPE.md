> **Current behavior (0.1.4):** Sherlock 2 is the default, with its packaged 0.5 cutoff and displayed-image acquisition. Pair collection is disabled throughout the extension and production backend. Saved files are preserved. Collection instructions below describe the earlier research prototype and are no longer available in the running app.

# Reddit prototype: fetching, recording and review

The prototype uses **Community Preview + FerretNet** at **0.5**, with its frozen semantic uncertain cutoff **0.03761442731955501**. This update changes acquisition, loading and collection, not weights, preprocessing, head coefficients or detector scores. All previous model choices remain available. New installs use the combination; existing version-2 model/source/cutoff choices are preserved.

## Try it

1. Run `npm run dev`. Restart an older backend to load the recording routes. The backend loads and warms the combination before becoming ready; optional OpenSynthID and FiSeR + DEAR-r load on selection.
2. Run `npm run build`, then load `extension-dist` using Chrome's **Load unpacked**. After rebuilding an already loaded extension, reload it and the Reddit tab.
3. In extension Settings choose **Community Preview + FerretNet**, reset the operating point to **0.500**, and select **Preview first · original for uncertain signals**. Enable opened-post scanning; enable feed scanning to scan while scrolling. Both start disabled on new installs.
4. Only visible, loaded still images on supported Reddit image hosts with both dimensions at least 128 pixels are eligible. GIFs, video and external image hosts are outside this prototype's coverage.

## Visible scan feedback

Reddit always displays a **Reddit Lens** status panel with Start/Pause, Settings, and Retry after failures. It distinguishes paused, feed-disabled, connecting, scanning, detector-offline and running states. Running checks local backend readiness. Fetching/analyzing/saving stages and elapsed time appear during requests. The toolbar badge shows OFF, POST, SCAN, ERR or ON for that tab; the popup reads the same page status without requesting broad tabs permissions.

Each successfully scanned eligible image gets a border and an on-image badge: **red** for an AI signal, **amber** for uncertainty and **green** for scanned with no strong AI signal. In-progress images show a spinner; failed scans show an error rather than a negative label. Percentages are raw detector scores, not AI probabilities, and green does not establish real origin. Duplicate gallery DOM elements do not stack badges or inflate unique-URL counts. Markings follow scrolling and overflow clipping, stay below a visible Reddit header, and disappear when scanning is paused. Background tabs do not start new scans; an already started request can finish.

For everyday browsing, keep **Preview first** and pair saving **off**. Select **Original first** to inspect original Reddit images more thoroughly. Pair saving is for building and reviewing a test collection; it uses extra data/disk space and does not alter the normal badge decision. The popup and Settings explain the currently selected source policy.

## Acquisition policy

Preview-first scores the actual displayed bytes. A strong preview flag is retained immediately; an unmarked preview needs no original request. An uncertain preview triggers a same-image original request, whose result becomes final. A failed original fetch/decode keeps the preview result and an explicit fallback reason. This can miss AI whose preview score is below the uncertain band. **Original first** provides a more thorough scan; **Displayed only** minimizes acquisition.

The resolver reads `shreddit-post` content links, gallery anchors, hidden/lazy original image sources, known gallery JSON attributes and open shadow roots. Candidates must match the displayed media ID. Modern `title-v0-ID.ext` and plain `ID.ext` preview paths have a best-effort `i.redd.it/ID.ext` fallback. Explicit originals take precedence, including when their extensions differ. Caption variants require an explicit matching original; external previews are not rewritten. An inferred URL does not guarantee that an original exists.

Fetching omits credentials/referrers, permits only `i.redd.it`, `preview.redd.it` and `external-preview.redd.it`, checks redirects/MIME, caps each stream at 20 MB and times out after 20 seconds. Failed scanning displays **Scan unavailable**. Badges remain experimental; percentages are raw scores, not calibrated AI probabilities. Normal detection adds no JPEG/WebP encoding. The model still performs its required 224/256-pixel feature/crop processing.

Chrome needs image bytes for inference. Fetch may reuse its HTTP cache, but the extension cannot reliably recover original encoded bytes from a cross-origin page image. Preview-first reduces original requests, not a guaranteed percentage of bandwidth. URLs with `auto=webp` can deliver JPEG: provenance uses the actual response MIME and bytes read.

Cache identity includes model/profile, policy, operating point, displayed/original URLs and recording session/post. Threshold changes requeue because they change escalation. Concurrent cache/history updates are serialized. Recording failures preserve a valid prediction and remain retryable.

## Collect and review development data

Turn on **Save image pairs locally** in extension Settings or the popup. This is opt-in. Normal scans keep blobs in memory and send them only to the local Python service. Recording additionally saves exact fetched bytes under:

```text
artifacts/reddit-recordings/
  records.sqlite3
  objects/<sha256>.<actual-image-extension>
```

Each session stops at **1,000 captures or 2 GiB**. After a rejected capture, the extension clears a stopped/capped session so future scans stop fetching record-only originals. Saved images remain reviewable. Pause scanning or turn off recording to stop collection. Recording collects unmarked images too; extra original scores do not change the browsing decision. It increases traffic, inference work, disk usage and time to final result.

Observations include original/preview checksums, dimensions, URLs, post permalink, subreddit when available, both model outputs, chosen output, profile identity, policy, operating point and fetch attempts. Identical pairs within a session/post deduplicate. Missing originals remain explicit. There is no cloud upload or external inference API.

Open **Recordings**, refresh and select the session. Every capture starts **unknown**. Assign real/AI only after checking evidence; save a category and creator/source/generator record or evidence URL. Subreddit membership and model scores are not ground truth. Export manifest downloads captures and review fields. This is a review export, not automatic training or automatic MCP registration; unknown captures must be excluded from supervised training and accuracy estimates.

The remaining data target is verified photos, screenshots, memes and artwork from varied sources, plus current AI originals and generator records; roughly 200–500 real examples per category if practical. Keep gallery frames, originals/previews and related edits grouped by source/post when splitting. Keep this development collection separate from the held-out validation folder. Training stays unchanged until verified development data is available.

## Diagnostics and acceptance checks

Settings shows recent scan-request failures, original resolution, HTTP 403s and visibility-to-final timings. **Export fetch diagnostics** downloads the last 500 events. Repeated DOM elements can share cached results. For analysis, retain the first event for each URL under the same model/policy/cutoff/session and report cold startup separately. Timings include queue, network and local inference; recording adds collection time.

Check at least 100 visible image posts across photo, art/meme and declared-AI communities. Report successful original downloads separately from inferred URLs. `not-needed` means preview-first avoided a resolved original; `unresolved` means none was found; `failed` preserves an original attempt; `same-as-displayed` means the displayed input was already an original. Track unsupported or never-eligible posts separately.

The October 12 functional probe and frozen cascade simulation are saved in `artifacts/reddit-prototype-2026-10-12`. Public browsing checks acquisition/timing, not accuracy: these posts lack independently verified labels. Simulated WebP results do not establish Reddit-native precision/recall. The user's validation directory remains sealed for final evaluation.
