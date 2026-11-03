# Local benchmark MCP

The server is installed in Codex as **`redditscanner_benchmarks`**. It uses stdio, runs locally, requires no API key, and launches model workers in the existing backend environment. The MCP SDK lives in its own virtual environment so it does not change Torch/FastAPI dependencies.

New Codex sessions can discover the tools after loading MCP configuration. The current conversation's tool inventory may require a fresh session or app reload. This does not require restarting the image lab or detector backend.

## Scope

The installed configuration pins the split manifest checksum and refuses changed boundaries. It is pinned to:

- Dataset: `/Users/dhruvsheth/Downloads/benchmark/calibration`
- Fixed partition manifest: `/Users/dhruvsheth/redditscanner/artifacts/ferretnet-improvement-2026-10-11/manifest.json`
- Outputs: `/Users/dhruvsheth/redditscanner/artifacts/mcp-benchmarks/`

Default runs use **research = development + tuning** (1,931 images). `development` (1,448) and `tuning` (483) are also supported. The reserved calibration audit and all validation requests are rejected. There is no tool that unlocks them or discovers another dataset. Symlinks, traversal, changed source checksums, and arbitrary profile code are rejected.

The manifest excludes two exact duplicate entries and the 18 images rejected by the original baseline, keeping comparisons on the same successfully decoded collection. Metrics exclude run-time failures and always report coverage and failure counts. A run where every image fails is marked failed and has no accuracy summary.

## Workflow for an agent

1. Call `benchmark_info` and `list_models`.
2. For the released detector, use `version="ferretnet-b-v1"`. The older ensemble is `fiser-dear-r-v1` and will load its large weights only in its worker.
3. The frozen local combination is available as `version="ferretnet-opensynthid-v1"`, with combined threshold 0.5. It preserves Ferret's 0.8304235935211183 component cut and OpenSynthID's 0.9013915360638743 cut through maximum threshold-centered logits. Its score is not a calibrated probability or verified watermark. For a modified FerretNet head/crop aggregation, write a new profile JSON inside this project, calculate its checksum, and call `register_model(profile_path="/absolute/path/profile.json")`. Registration saves an immutable snapshot and returns a SHA version.
4. Call `start_benchmark(version=..., partition="tuning", threshold=null, variant="original")`. It returns a run ID; it does not wait for all inference to finish. A threshold override is optional and recorded explicitly.
5. Poll `get_run(run_id=...)` at sensible intervals. On completion it includes precision, recall, F1, false-positive rate, AUROC, average precision, per-category flag rates, failures, startup time, first-image time, warm/median/p95 latency, and artifact location.
6. Use `compare_runs(run_ids=[baseline_id,candidate_id])`. Runs must match image hashes, successful files, partition, image variant, and device. Different model/profile or implementation hashes are recorded in each summary. No threshold tuning or winner selection happens automatically.

`list_runs(limit=20)` retrieves persisted runs. `cancel_run(run_id=...)` stops a worker and preserves partial scores and logs. Normal server shutdown cancels workers it owns. After an abrupt process kill, a leftover worker may finish independently; consult its status/log before starting more work.

One server accepts one active worker. Workers using the same output directory also hold a cross-process lock for their entire run, preventing two Codex sessions from distorting each other's inference timing. This cannot stop unrelated applications from using the GPU.

## Versioned profile format

Community Forensics numeric heads and the frozen Community + Ferret combination are also supported. Register `backend/models/community-profile.json` or `backend/models/community-ferret-profile.json` with `register_model`, then use the returned SHA as `version`. `list_models` includes registered profiles. These are `kind="semantic-head"` / `kind="semantic-hybrid"`, require the pinned authors' 224 checkpoint, a finite 384-dimensional head, and checksummed decision policy. The combination additionally checks its semantic profile identity and fixed component cuts. See [training and implementation](SEMANTIC-PREVIEW.md).

Deployment stress variants are `reddit-webp640` and `reddit-webp960`: width-limited, no-upscale, aspect-preserving Lanczos followed by a single WebP quality-80 pass for both labels. They are simulations, not a verified byte-for-byte Reddit encoder reproduction. Original source dimensions remain in the result; transformation metadata records the processed dimensions. Original-byte inference remains the default. Comparisons still require matching variants and successful image sets.

This adapter currently supports the released **FerretNet-B** checkpoint with fixed crop aggregations or a trained linear head on frozen CNN features. It supports center256, native 5/9 crops, resized 5 crops, native512, mean/std pooling, and native/resized multiscale combinations. The profile contains numbers, not executable Python.

Example fixed profile before adding `modelHash`:

```json
{
  "schemaVersion": 1,
  "model": "ferretnet-my-native5-v1",
  "pipeline": "ferretnet-b-native5-mean-v1",
  "checkpointSha256": "fe755d78370bb6547070329553572405b4ecebd23382c9a6cbb11c4ab85a82c2",
  "kind": "fixed",
  "representation": "native5_mean",
  "threshold": 0.83
}
```

For `kind="probe"`, supply `head.weight` (384/768/1,152 numerical coefficients according to representation) and `head.bias`. Fold feature standardization into the coefficients before exporting; the runtime is a dot product plus sigmoid. Only pixels are inputs. Metadata such as generator names and filenames are never classifier inputs.

Calculate `modelHash` as SHA-256 of canonical JSON excluding `modelHash` itself: sorted keys, separators `(',', ':')`, UTF-8, no NaN/Infinity. In Python, use `benchmark_mcp.core.digest`. Any change, including threshold or metadata, gets a different immutable version. Editing the original registered file does not alter existing runs.

The preserved `artifacts/ferretnet-improvement-2026-10-11/experimental-profile.json` is a working probe example. Its frozen calibration audit **failed**; it is a research artifact, not a recommended replacement.

## External evaluation datasets

An explicitly configured registry makes downloaded external datasets available without granting access to the user's validation folder. The installed configuration uses `LENS_BENCHMARK_EXTERNAL_CONFIG=/Users/dhruvsheth/redditscanner/artifacts/external-benchmarks-2026-10-11/registry.json`. You can also pass `--external-config /absolute/project/path/registry.json` to the server.

Each registry entry contains a project-local `root` directory named `calibration`, a `manifest`, and `manifestSha256`. The manifest lists relative image files, verified hashes, labels, groups, and `experimentSplit="tuning"`. This partition name is used for transport compatibility: these external evaluation images are **never used to fit thresholds**. Symlinks, traversal, paths outside the project, and validation paths are refused.

Call `benchmark_info` to see the registered names, then `start_benchmark(version="ferretnet-opensynthid-v1", dataset="park", partition="tuning")`. The default `dataset="local"` still uses the user's original checksum-pinned development/tuning boundary. The external names currently are `park`, `sensifake`, and `synthetic-real`. Request/summary artifacts identify the external dataset; comparison requires matching image sets and device/runtime. No dataset discovery or automatic threshold selection occurs.

The frozen evaluation plan, provenance snapshots, preparation/deduplication manifest, actual stdio client, and results are in `artifacts/external-benchmarks-2026-10-11/`.

Changes to the compatible Python detector implementation are fingerprinted in requests and summaries; a change during a run fails it. A different architecture or fine-tuned checkpoint needs a new explicit adapter. The server intentionally does not execute a Python file supplied by a tool caller.

## Reinstall and configuration

Run `npm run setup:mcp` after `npm run setup` on a new machine. The SDK is pinned in `benchmark_mcp/requirements.txt`.

The installed user config entry in `~/.codex/config.toml` is equivalent to:

```toml
[mcp_servers.redditscanner_benchmarks]
command = "/Users/dhruvsheth/redditscanner/benchmark_mcp/.venv/bin/python"
args = ["/Users/dhruvsheth/redditscanner/benchmark_mcp/server.py"]

[mcp_servers.redditscanner_benchmarks.env]
LENS_BENCHMARK_CALIBRATION_DIR = "/Users/dhruvsheth/Downloads/benchmark/calibration"
LENS_BENCHMARK_SPLIT_MANIFEST = "/Users/dhruvsheth/redditscanner/artifacts/ferretnet-improvement-2026-10-11/manifest.json"
LENS_BENCHMARK_SPLIT_MANIFEST_SHA = "f6c0b73a767dd3fee4e44f87fb1d40e850daaf4608234cc45bd87806bf952d69"
```

Paths must be adjusted on another machine. The server requires an explicit directory named `calibration` and never searches its parent. If a future human-authorized dataset has no partition manifest, configure its calibration root explicitly and omit the manifest; the only allowed partition then is `calibration`. Validation remains unsupported.

This follows [Codex's stdio MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) and the [official Python MCP SDK](https://py.sdk.modelcontextprotocol.io/run/).

## Verification

`npm run test:mcp` uses standard unittest plus a real MCP client/server stdio connection. It initializes the protocol, lists all eight tools, benchmarks actual FerretNet weights, registers/runs a versioned profile, compares compatible runs, rejects mismatches/validation, cancels a job, and checks persisted history. Additional tests cover source hashes, immutable snapshots, bad profiles, symlinks/traversal, launch failure, total decode failure, and concurrency.

`npm run test:backend` additionally checks numerical head forward passes, checksum agreement, tied-score metrics, crop rounding/padding, decode parity, and MPO handling. The configured server was also exercised on both original FerretNet and the frozen experimental profile over the 483-image tuning partition; see `artifacts/ferretnet-improvement-2026-10-11/mcp-configured-check.json`.

Outputs for every run: `request.json`, `status.json`, `worker.log`, streaming `scores.jsonl`, `summary.json`, `failures.json`, `all-results.csv`, `false-positives.csv`, and `false-negatives.csv`. Torch/model import and initialization are measured separately from per-image decode/inference. This startup timer excludes the interpreter launch and initial NumPy import. File loading/hashing, network transfer, browser overhead, and Chrome/WASM execution are not included in per-image timing. Python/Torch/Pillow/platform/thread metadata is saved; comparisons reject differing recorded runtime environments. Keep system load consistent when comparing latency.

## MIRROR research profiles

The pinned `mirror` and `mirror-hybrid` numerical profiles are now supported:
`backend/models/mirror-profile.json` and `backend/models/mirror-community-profile.json`.
Register the JSON and use its returned SHA version. Checkpoint, upstream revision,
preprocessing and the 0.5 component cuts are immutable. The large weights load only
inside a requested worker. Legacy builtin IDs remain supported.

If this desktop session's MCP process predates the adapter, restart that server or
use `benchmark_mcp/.venv/bin/python scripts/run-mirror-mcp.py`, which connects through
a fresh stdio MCP instance with the same pinned calibration boundaries. It also
registers the fixed project-local Reddit tuning sample; its source-derived labels
are reported as weak-label research evidence, not independent ground truth.
See [MIRROR](MIRROR.md) for preprocessing, parity and resource details.

## Reddit adaptation candidates

The `reddit-fusion` and `reddit-student` numerical profile families accept pinned
local JSON profiles. Registration verifies their canonical identities; the student
also pins its local safetensors checksum. They run through the same isolated worker,
runtime fingerprint, dataset safeguards and artifact format as other detectors.
No arbitrary Python implementation paths are accepted. See
[Reddit adaptation](REDDIT-ADAPTATION.md) for development/tuning roles, the reproducible
candidate runner and the separate parity check. The optional student picker uses
the same profile as the MCP candidate; this does not change the builtin defaults.

## Explicitly authorized validation check

The user authorized a fixed validation evaluation on 2026-10-16. The ordinary MCP
scope remains locked. Its dedicated runner reuses immutable profile registration,
the unchanged inference worker, progress and comparisons while recording the actual
validation root and human authorization in requests. See
[validation scope, overlap audit and reproduction](VALIDATION-2026-10-16.md).
Those results do not unlock additional validation tuning or training.
