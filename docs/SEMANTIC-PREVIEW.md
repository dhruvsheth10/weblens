# Community Preview and Reddit input handling

The small detector uses the authors' `OwensLab/commfor-model-224` checkpoint (revision `26afc31e6b40c312c3fd42c05a758be62446215b`, SHA-256 `a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0`). This is the actual 224 model, not a repackaged 384 checkpoint or an older ONNX export. It has about 22 million parameters and occupies 86,678,644 bytes in safetensors.

## Fitted detector

The backbone stays frozen. A regularized linear head is fitted to all 1,448 development sources and four identical transformations per source: original, JPEG quality 75, width-limited 640-pixel WebP quality 80, and width-limited 960-pixel WebP quality 80. Resizing preserves aspect ratio, never upscales, and uses Lanczos for these WebP simulations. The simulations approximate a deployment stress case; Reddit's exact encoder/quality settings are not established here. Original input inference does not add a JPEG/WebP pass.

Each view is processed using PIL's validated EXIF/alpha decoding, resize shortest edge to 256, a 224 center crop, ImageNet normalization, and the frozen ViT-S feature vector. The original authors' 224 architecture is restored strictly from safetensors, with no network or custom downloaded code during inference. The representation has 384 numbers. Standardization is folded into the exported head so runtime needs only a dot product and a sigmoid.

Training uses inverse source-group, category and class weighting. All transformed views of a related source stay together in five grouped development folds. Feature standardization and weighting are computed inside each training fold. `C=0.05` and the transformation/cutoff policy were recorded before any new tuning/external predictions. The high cutoff is the maximum of grouped out-of-fold real-score cuts at 0.5% overall and 5% within each real category, using each source's worst score over its four views. Categories smaller than 20 allow no observed development false positives. The lower uncertain cutoff uses 2% overall. These policies target empirical development behavior, not a proven population FPR.

The final classifier is fitted on development only. The high raw-logit cutoff is 5.837027415830514; the uncertain raw-logit cutoff is 2.5949999243699367. Subtracting the high cutoff from the head bias makes the displayed high score threshold exactly 0.5. The uncertain displayed cutoff is 0.03761442731955501. This transformed sigmoid score is not a calibrated probability of AI origin. The numeric artifact is checksum-versioned; editing any coefficient, cutoff or metadata changes its identity.

`Community Preview + FerretNet` adds the released Ferret checkpoint at its unchanged 0.8304235935211183 cutoff. Its high decision is the OR of the two fixed component decisions, using the maximum of their centered logits. Its uncertain decision comes only from the semantic score; a below-threshold Ferret score cannot create uncertainty. Both branches share one validated image decode and transformation. The backend reuses already-loaded component models. This OR cannot correct Ferret's existing false positives and may add new ones.

The standalone semantic head and combination are selectable alongside all previous models. All production choices still run in the local Python backend. New installs now default to Community + Ferret for the explicitly authorized prototype; existing selected models and operating points are preserved. This does not establish a validated deployment gate. New head data and optional ONNX exports are described in the experiment report under `artifacts/semantic-preview-2026-10-11/`.

## Reddit fetch policy

New installs use preview-first acquisition, while scanning remains paused. Actual displayed bytes are scored first; only the semantic uncertain band triggers an original request. Strong preview flags are retained. Original-first with preview fallback and displayed-only remain selectable. The resolver handles modern `title-v0-ID.ext` slugs, same-ID `content-href` URLs and gallery sources, including hidden original image elements and open shadow roots. Unsupported originals remain explicit, and failed original fetch/decode retains the analyzed preview.

Source URLs, actual MIME, bytes, statuses, policy and timings are exported in diagnostics. Cache identity includes the operating point because it affects escalation. Recording is separate opt-in: it collects actual paired bytes for unknown-label review and never changes a normal browsing decision. See [setup, collection limits, review and diagnostics](REDDIT-PROTOTYPE.md).

## Benchmark and exports

The existing MCP supports both new `reddit-webp640` and `reddit-webp960` variants and checksum-registered `semantic-head` / `semantic-hybrid` numeric profiles. These do not accept executable model code. Existing audit/validation restrictions remain enforced. All evaluations use frozen heads/cuts. A second width is evaluated after freezing; it is not used to change a parameter. New category results and input failures must be reported, not hidden behind aggregate accuracy.

FP32 and INT8 ONNX exports contain the actual fitted semantic head. INT8 is an experimental artifact, not silently substituted into production. `export.py` verifies FP32 numerical parity and records INT8 decision changes. The local browser probe checks actual development tensors in Chrome with ONNX Runtime Web/WASM. This establishes execution and tensor-level parity for that sample; it does not establish browser image-preprocessing parity, extension behavior on Reddit, or population accuracy.

## Pending broader data

The user is assembling a separate development collection. It must stay separate from the untouched validation folder. Current real-photo/meme groups are too small, and real museum art versus AI Reddit content can create source/content shortcuts. Before teacher fine-tuning or distillation, collect varied verified photos, screenshots, memes and art, plus current generated files with generator/provenance records. Keep original source files, captions and related versions grouped together. Apply degradation identically to both labels.

Pre-2022 publication dates alone are not proof of non-AI origin: GANs and other synthetic media predate that date. Teacher pseudo-labels are also not ground truth; distillation can copy the teacher's errors. A larger teacher/student pipeline will be implemented and evaluated when the agreed data is available, rather than shipping untrained adapters or claiming a validation result that has not happened.
