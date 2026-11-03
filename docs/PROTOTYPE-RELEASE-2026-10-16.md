# Reddit Lens 0.1.1 — prototype snapshot

The Chrome extension is built in `extension-dist`. The local Python backend and
image lab are started with `npm run dev`. This remains a Chrome extension plus
local helper; the extension package does not contain model weights or standalone
browser inference.

## Start testing

1. Keep `npm run dev` running in this project.
2. In `chrome://extensions`, reload the existing Reddit Lens extension. For a new
   installation, enable Developer mode and load `extension-dist` unpacked.
3. Reload Reddit tabs to replace their old content scripts.
4. Enable **Run Reddit Lens** and **Scan the Reddit feed** in the popup.
5. Keep pair saving off for normal browsing. Enable it only when collecting a
   local development dataset, and review its unknown labels before training.

Community Preview + FerretNet remains the default at 0.5 with preview-first
acquisition. The experimental Reddit Preview Student remains available and selects
displayed previews. Existing user choices are preserved. A green badge means the
image was scanned without a strong AI signal; it does not establish real origin.

## Changes in this snapshot

- Badge layout skips off-screen images before style or geometry reads. A browser
  regression with 1,000 connected images and one visible image recorded one image
  geometry read and zero off-screen image geometry reads. This is a focused DOM
  regression, not an overall scrolling FPS claim.
- Late preview/final results cannot mark a replacement image. Source changes and
  image loads invalidate the old request. Only loaded images enter scanning.
- Previously removed image elements are observed again when Reddit reinserts them.
- If original analysis succeeds but downloading its recording preview fails, the
  inference result is retained and the recording failure is visible. Background
  result caching excludes recording failures. The cache namespace was advanced to
  avoid reusing older silently incomplete recording results.
- Readiness checks verify the selected model's hash, preprocessing pipeline and
  default thresholds against the local backend, including lazily loaded models.
  The launcher detects older backends missing those identities.
- Packaging rejects extension/backend profile mismatches before replacing the
  existing extension directory.
- The development browser smoke command checks the selected model and preserves
  uploaded bytes instead of re-encoding them through a canvas.

No model weights, preprocessing, learned parameters or operating points changed.
Consequently this release makes no new accuracy or precision claim. Legacy model
implementations and benchmark profiles are preserved.

## Verification

40 frontend tests, 56 backend tests and 16 MCP tests passed. Type checking, Vite
build and extension bundling passed. Real Chromium checks covered stale preview
rejection, rescanning replacement/reinserted images, off-screen layout work, all
six model identities through the extension service worker, and actual inference
through the web image lab. The popup was visually checked. Test browsers were
closed; the local helper was left running.

Artifacts and logs: `artifacts/prototype-release-2026-10-16/`.
The browser lifecycle test uses synthetic DOM and mock transport, never model
labels. This does not complete the outstanding live 100-post Reddit acceptance
test. Validation, calibration audit and Reddit holdout pixels were not accessed.

## Accuracy limits and feasible next improvements

The [October 14 experiment](../artifacts/reddit-adaptation-2026-10-14/report.md)
remains the latest accuracy evidence. On actual Reddit tuning previews, the student
reached 98.6% precision and 22.4% recall, with one false flag. Its cutoff was selected
on those same images, whose origin labels are source-derived; this is not held-out
feed precision. It detected only 2/22 difficult Reddit AI examples and no AI images
in the small Park external sample. The current data/training recipe is a limitation;
these results do not establish an inherent ceiling for local detection.

| Next option | Feasibility | Expected benefit and limit |
|---|---|---|
| Review existing provenance and collect creator-verified modern photoreal, AI meme, screenshot and text-heavy examples | Practical now; the main cost is obtaining reliable labels | Best-supported route toward improved generalization. Visual plausibility alone is not a label. |
| Train the small model on the expanded, grouped development set | Demonstrated locally: the pilot's two training epochs took about six minutes | Directly targets accuracy; requires a separately versioned experiment and evaluation. The 200 new pairs and 504 public references are prepared but were not used in the first pilot. |
| Port the selected small model to a browser runtime or packaged native helper | Feasible engineering investigation; runtime parity and latency still need measurement | Improves installation/runtime usability. Does not itself increase detection accuracy. Current 22 ms student timing is Python/MPS, not Chrome. |
| Improve a large teacher, then distill clean originals into preview inputs | More substantial data, training and evaluation work | Could improve compression robustness; distilling the present teacher would also transmit its errors. |
| More threshold sweeps or OR ensembles on the existing tuning data | Easy, but poor next priority | Primarily changes the recall/false-flag tradeoff. Existing fusion experiments did not solve hard cases. |

Freeze the present models for private prototype testing, collect representative
failures with provenance, then return to training. Keep reserved evaluation data
for the authorized final check after selecting the next model and operating point.
