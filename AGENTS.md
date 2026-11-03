# Reddit Scanner

This project is a local image detector and a Chrome extension research prototype.

## Benchmark boundaries

- `/Users/dhruvsheth/Downloads/benchmark/calibration` is authorized for ongoing development. Access to the sibling `validation` directory requires explicit human authorization; do not run scripts that implicitly process it.
- On 2026-10-16 the human explicitly authorized one fixed validation evaluation of Community + Ferret and the Reddit student at their packaged cutoffs, on originals and WebP 640/960. Its plan and results are in `artifacts/heldout-validation-2026-10-16/`; see `docs/VALIDATION-2026-10-16.md`. That data is now unblinded and has substantial prior calibration overlap. Do not use it for training, threshold selection, or additional model/configuration runs without further human instruction, and do not present the full folder as independent validation. The fixed authorized runner may resume incomplete jobs in that plan.
- Calibration is partitioned by `artifacts/ferretnet-improvement-2026-10-11/manifest.json`. Use development for training, tuning for parameter/threshold selection, and keep related source groups together. The audit portion was unblinded once and failed the deployment gate; do not tune against it or present it as independent validation.
- Prefer the `redditscanner_benchmarks` local MCP for repeatable runs. Inspect `benchmark_info`, register a checksummed numerical profile, start a run, poll `get_run`, and compare only matching datasets/devices. See `docs/BENCHMARK-MCP.md`.
- Keep the original FerretNet and FiSeR + DEAR-r available. The experimental resize5 head is not approved as the default. Score changes require versioned model/profile identities and numerical parity checks.
- `backend/engine.py` is part of the legacy model fingerprint. Changing it without regenerating the profile breaks that detector's checksum contract.

## Commands

- `npm test`, `npm run test:backend`, `npm run build`
- `npm run setup:mcp`, `npm run test:mcp`
- `npm run dev` starts the local backend and image lab.

The production UI still uses local Python inference. Model size alone does not establish Chrome/WASM performance or detection accuracy.
