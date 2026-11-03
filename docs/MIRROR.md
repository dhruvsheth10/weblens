# MIRROR local research integration

The default remains Community Preview + FerretNet at its frozen 0.5 cutoff.
Settings also offers `mirror-h-v1` and `mirror-community-v1`. Both load on their
first detection request; neither adds large-model startup cost to normal browsing.
The old standalone FerretNet-B and FiSeR + DEAR-r cards are retired. Their source,
weights, and benchmark identities remain intact. Existing recordings remain readable.
Stored retired selections migrate to the default **and reset the old model's cutoff**.

## Offline loading and exact input path

MIRROR's full local phase-2 checkpoint already contains the backbone, LoRA adapters,
memory bank and classifier. No Hugging Face authentication or runtime download is used.
The loader verifies SHA-256 before opening it with `weights_only=True`; only the
checkpoint's known `argparse.Namespace` type is allowlisted. All tensors load strictly.
The Phase-1 checkpoint is not needed in addition to the full checkpoint.

DINOv3 H+: 32 layers, hidden size 1280, 20 heads, gated SiLU intermediate size 5120,
four register tokens, LoRA rank 8 / alpha 16 on query/key/value, float32 eager attention.
Frozen memory key/value projections are cached after the first scan. A direct comparison against the original forward pass produced exactly identical logits on three fixed Reddit images; warm latency improved modestly. The cache resets on device/dtype changes.

The four register tokens remain in the memory-bank input, matching upstream.

Preprocessing follows upstream evaluation at commit
`adb928ec56c4ffd77c53ccb908bea29bb559922c`:

1. Shared safe decode, EXIF orientation and alpha over white.
2. If the shorter edge exceeds 512, resize it to 512 using bicubic interpolation and
   rounded aspect-preserving dimensions. Smaller images are not enlarged.
3. Center crop to 224 × 224; very small inputs are padded by the torchvision crop.
4. For PNG inputs, apply JPEG quality 96 after cropping, matching upstream evaluation.
   Detect codec from bytes, never a filename. JPEG/WebP stress variants are already
   lossy and do not receive the PNG branch.
5. Convert pixels to float32 [0,1]; the model applies ImageNet normalization once.
6. AI is class 1: score = sigmoid(logit_1 − logit_0), equivalent to two-class softmax.

No subreddit, flair, comments, post text, filename or collection labels enter the
network. Scores are detection signals, not calibrated probabilities or watermarks.
The center crop can miss content near the image edges.

The combination takes the maximum of the MIRROR and frozen Community + Ferret logits.
Both are centered at cutoff 0.5. It is an OR, so it can increase recall **and** false
positives. No component threshold is fitted or changed in this integration.
MIRROR has no calibrated uncertain band; selecting it changes preview-first to
original-first. Displayed-only remains available for testing real feed bytes.

## Reproducible checks and benchmarks

- `backend/.venv/bin/python scripts/check-mirror-parity.py`: actual CPU/MPS agreement,
  component/blend agreement and memory accounting, two explicit development images.
- `backend/.venv/bin/python scripts/prepare-mirror-reddit.py`: deterministic grouped
  development/tuning/holdout partition and a fixed category-balanced tuning sample.
  Author, post, exact hashes and dHash distance ≤4 connect related images. Conflicting
  groups and detected ambiguous answers are quarantined. Metadata labels are weak
  supervision and need manual review before training.
- `benchmark_mcp/.venv/bin/python scripts/run-mirror-mcp.py`: fresh stdio MCP instance,
  inspect scope, register immutable profile, start/poll sequential MPS benchmarks.
  Scores development+tuning only for original, JPEG75 and WebP640, plus actual Reddit
  tuning previews. A fresh server is needed if the desktop MCP was started before
  the new adapter was installed. No configuration weakening or validation access.

The benchmark runs the full combination once per image and retains each component's
logit and timing. MIRROR-only and old-combination accuracy are reconstructed exactly
from those frozen component logits on the identical successfully scored image set.
This avoids redundant expensive inference; component timings are measured within
the combination process, rather than independent isolated throughput runs.

Results, raw MCP run IDs, parity evidence, frozen Reddit manifest and screenshots:
`artifacts/mirror-evaluation-2026-10-13/`.

## Distribution

Keep the weights local. Upstream advertises MIT in its README, but the linked LICENSE
file was absent at the pinned revision; DINOv3 terms also apply. This integration
makes no claim that the 3.45 GB checkpoint can be redistributed in a Chrome extension.
