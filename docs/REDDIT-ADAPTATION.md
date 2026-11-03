# Reddit adaptation research — 2026-10-14

This experiment addresses the Reddit domain gap using local development data.
The fine-tuned student is available as an explicitly experimental Settings choice.
The released/default profiles are preserved; no public deployment gate is claimed.

## Trying the student

Run `npm run dev`, open Settings and choose **Reddit Preview Student · experimental**.
Selecting it switches the image-source policy to displayed previews, the input domain
used for its operating point. An explicitly selected original-first policy remains
available, but its performance differs. Reload the unpacked extension from
`extension-dist` to see the new choice there. The popup now builds its choices from
the same registry as the image lab, removing the old stale hardcoded list.

The student loads lazily from its verified local 83 MiB checkpoint. It uses local
Python inference, not in-extension inference. The displayed percentage is a detector
signal after a Reddit-tuning-selected offset, not a calibrated probability of origin.
Its conservative cutoff still misses most hard cases. Community + Ferret remains
the default, and all original model artifacts remain available to the benchmark.

## Data boundaries

The frozen Reddit grouping is read from
`artifacts/mirror-evaluation-2026-10-13/reddit-splits.json`.
Only development (3,693) and tuning (783) previews and their existing originals
(780 development, 146 tuning) are staged for the benchmark. Holdout/quarantine
pixels are excluded. The user's calibration validation directory is never inspected.
The calibration-development set may train the student; calibration tuning is a
transfer check after the model and Reddit operating points are fixed. Audit is excluded.

The MCP's external-dataset protocol labels transport rows `tuning`. The separate
`usageSplit` and dataset names retain the true development/tuning meaning. Fusion
fits only Reddit development; thresholds use tuning. Do not treat transport labels
as permission to fit the tuning subset.

`image-map.json` preserves actual source groups and roles. Staging uses hardlinks to
immutable object bytes; inference never modifies them. All staged assets are hashed.

## Label and collection work

`datasets/reddit/labels.py` rejects mixed/qualified bot answers. The evidence audit
excludes ambiguous hard-case galleries from student/fusion training. It does not
change existing tuning labels or use model disagreements to relabel examples.
The review queue contains prior disagreements plus stratified random agreements.
An evidence audit cannot establish photographic origin from visual appearance alone.

`datasets/reddit/collect.py` supports a separate output collection, bounded duration,
listing limits and frozen-manifest exclusions. Previously failed original downloads
no longer crash resumption. Old posts/assets and reserved tuning/holdout authors
are excluded from the new development supplement. Source claims remain weak labels.

The separate original backfill recovers up to 40 development originals per real
photo/art/meme/screenshot category and 40 AI-art originals. Selection is deterministic
and independent of scores. Its manifest is separate from the frozen collection.
No images are posted and no paid generation or cloud training is started.

## Fixed model experiments

1. Score all frozen Reddit development/tuning previews and available originals with
   MIRROR + Community + Ferret, retaining all three logits.
2. Fit two- or three-logit regularized fusion. Select regularization using five-fold
   source-group development OOF partial AUROC at 1% FPR. Clip component logits at ±20.
3. Fit a new Community linear head using development images only, with the existing
   four-view calibration-development feature cache verified before reuse.
4. Start that small model from the fitted raw head and fine-tune its last two blocks,
   final norm and head for two fixed epochs, 4,096 sampled views each. Sampling balances
   dataset/class and caps rare-source weights. Both classes receive the same random
   JPEG/WebP/resize/crop distributions. There are no teacher-generated labels.
5. Select a global operating point on Reddit tuning with empirical FPR ≤1% in every
   observed real category. Use a 1e-4 logit margin against numerical rounding. Report
   frozen-cut results as well as constrained results. No tuning gradients.

The student pilot is partial backbone fine-tuning, not a claim of exhaustive full
fine-tuning or optimal hyperparameters. `student-plan.json` is saved before fitting;
an initial untrained sampling plan was archived after its extreme weights were found
and capped. No tuning results informed that pre-fit correction.

## What results must report

- Recall, precision, F1, AUROC, partial AUROC at low FPR, per-category false flags.
- Precision reweighted to assumed AI prevalences of 1%, 5%, 10%.
- Sample counts, intervals and the limits of correlated source labels.
- Hard-case performance separately; AI memes/screenshots remain a known category gap.
- Existing-original results separately because original coverage is source-confounded.
- Development OOF vs fitted development vs threshold-selection tuning distinctions.
- Local calibration transfer with thresholds frozen from Reddit.

An empirical low FPR on a small tuning category is not a confidence guarantee. The
public-badge deployment gate also requires defensible labels, representative coverage,
and upper uncertainty bounds. The current dataset may be too small to pass that gate.

## Commands

```bash
backend/.venv/bin/python scripts/prepare-reddit-adaptation.py
backend/.venv/bin/python scripts/audit-reddit-labels.py
benchmark_mcp/.venv/bin/python scripts/run-reddit-adaptation-mcp.py
backend/.venv/bin/python scripts/run-reddit-adaptation-training.py
```

The training launcher waits for benchmark GPU work before starting its own GPU work.
It runs the fusion analysis and student pilot. After all baseline runs finish,
`scripts/enable-reddit-research-benchmarks.py` installs the explicit research adapters
for versioned fusion and fine-tuned student profiles. Do not edit the fingerprinted
benchmark adapter files while a worker is active.

Artifacts are under `artifacts/reddit-adaptation-2026-10-14/`.
The preserved source collection and the new paired-original supplement are under
`datasets/reddit/` in their explicitly named directories.

Completed measurements are in
[`report.md`](../artifacts/reddit-adaptation-2026-10-14/report.md).
To verify and rebuild the report after all required runs have completed:

```bash
benchmark_mcp/.venv/bin/python scripts/run-reddit-candidate-mcp.py
backend/.venv/bin/python scripts/check-reddit-candidate-parity.py
backend/.venv/bin/python scripts/report-reddit-adaptation.py
```

`verification.json` records the passing frontend/backend/MCP checks, build and
isolated-browser smoke test. The latter exercised uploads in both the web lab
and the extension's own origin, plus the popup's model/source synchronization.
The screenshots `student-result.png` and `student-popup.png` show the tested UI.
This is not a new labeled accuracy evaluation or a live 100-post Reddit test.

## Modern public reference supplement

When all five probed Reddit listing sources remained blocked, public original-image
recovery still succeeded. In addition, `scripts/collect-modern-development-references.py`
prepares a separate bounded reference set from the publisher's pinned
MLLM-Generated-Image-Detection dataset revision. It reuses only the 288 existing
**train** references and downloads 216 additional files from 24 new source IDs,
grouped across three styles and Real/GPT Image 2/Nano Banana 2 folders. Existing
reference calibration/validation source IDs are excluded. Actual filenames and LFS
checksums come from a saved publisher file-index snapshot; no filename convention
is assumed after resolution. No paid generation requests are used.

The resulting 504 references are **not used in this first frozen student pilot**.
They are prepared for a separately versioned follow-up training experiment, avoiding
an unrecorded mid-experiment dataset change. Publisher labels are evidence of origin,
not independent ground-truth verification. The broad style labels do not establish
that photoreal AI, memes and screenshots are all adequately represented.

`scripts/audit-modern-development-references.py` verifies object hashes and excludes
known reserved/evaluation hash collisions or contradictory duplicate labels from
`eligible-development.json`. It reads reserved hash metadata, never reserved pixels.
The original phase-1 teacher and production models are preserved.

For an individual explicitly named development image, use:

```bash
backend/.venv/bin/python scripts/score-reddit-candidate.py \
  --profile artifacts/reddit-adaptation-2026-10-14/student-profile.json \
  /absolute/path/to/an/authorized/development-image.jpg
```

The command refuses known reserved calibration-audit and Reddit-holdout paths,
validation directories and symlinked inputs. It accepts image files individually,
not a recursive dataset directory. It loads a pinned numerical profile and verifies
the candidate weights. File paths are output metadata, never classifier inputs.
