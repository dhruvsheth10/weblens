# Human-authorized validation evaluation

On 2026-10-16 the user explicitly requested the validation version of the benchmark.
The permitted source is `/Users/dhruvsheth/Downloads/benchmark/validation`.
The fixed plan is recorded before inference in
`artifacts/heldout-validation-2026-10-16/plan.json`.

The plan evaluates the packaged Community Preview + FerretNet and Reddit Preview
Student profiles at their existing 0.5 operating points, using original bytes plus
the existing 640/960-pixel WebP simulations. No threshold fitting or training is
part of this evaluation. Repeated transformed inputs are robustness checks.

The routine benchmark MCP remains unable to start validation jobs. The dedicated
runner registers immutable profiles through a fresh MCP session, records the human
authorization in each request, launches the existing unchanged benchmark worker,
then uses MCP `get_run` and `compare_runs`. Worker requests explicitly identify the
validation root and partition; the data is not renamed or staged as calibration.

## Reproduction and artifacts

`scripts/run-authorized-validation.py` can resume unfinished jobs from this fixed
plan. It does not fit models or rerun completed jobs. Do not change the model,
thresholds, manifest or scope to produce new validation experiments without a new
human instruction.

After all six jobs complete:

```bash
backend/.venv/bin/python scripts/report-authorized-validation.py
```

Artifacts are under `artifacts/heldout-validation-2026-10-16/`:

- `manifest.json` and its checksum pin the actual files and labels.
- `data-audit.json` retains the initial metadata diagnostics.
- `reviewed-data-audit.json` corrects broad catalog-prefix grouping and freezes the
  exact-new and stricter sensitivity subsets without using predictions.
- `runs.json` links each model/input combination to its MCP worker directory.
- `mcp-comparisons.json` checks matched files, variants, devices and runtimes.
- `report.md` and `results.json` contain the final scores and interpretation.

## Independence issue

The validation folder has 2,099 supported image files: 1,250 AI and 849 real.
1,226 entries exactly match images already present in the calibration experiment
manifest, including 730 development/training entries. One hash is duplicated within
validation. The full folder therefore cannot serve as an independent holdout.

Removing known exact overlap and internal duplicates leaves 872 unique new byte
files: only 25 AI and 847 real. On original-byte inference, one AI file exceeds
20 MB and one real file is too small, leaving 24 AI and 846 real predictions. These
coverage failures are retained and are not counted as negative detections.

The initial source-prefix heuristic also matched 100 distinct museum filenames
such as `artic_<catalog ID>`. A shared museum prefix does not identify a Reddit post
or one artwork. Those broad matches are discarded. A separate stricter sensitivity
subset excludes one ambiguous whole-filename match; its selection uses metadata
only. No image is relabeled based on a detector's output.

These checks do not establish complete independence of similar content, correct
origin labels, or absence from a released model's pretraining. The small new AI
sample limits recall conclusions. The data is now unblinded; further development
against its results would require fresh independent evaluation data.
