# Detector review — 2026-10-10

## Actual inference path

The extension observes Reddit `<img>` elements, waits for at least 25% visibility, filters to post images on three Reddit image hosts with both dimensions at least 128 pixels, and submits one request at a time per tab. Feed scanning must be enabled separately. The background worker caches up to 200 URL/model/pipeline results for the browser session. An offscreen page fetches the image and a web worker posts its original bytes to the Python backend on localhost. The backend serializes inference across clients. No model runs inside Chrome in this version.

The backend decodes the image, applies EXIF orientation and RGB conversion, and composites transparent pixels onto white. Original-mode decoding does not JPEG-recompress or downsample the image. FiSeR separately converts a copy to its expected normalized 224×224 tensor, extracts the normalized 1024-dimensional CLS vector at layer 18, and applies this project's linear classifier. That head was trained on 287 reference images; it is not the paper's feature-bank evaluation.

DEAR-r uses ImageNet normalization and native-resolution pixels. It evaluates up to 16 deterministic 768-pixel windows and averages their logits. Windows overlap for many ordinary image sizes; exceptionally large images may have gaps. Its channel gate masks features after the backbone computation, so the word “pruning” does not mean the expensive convolutions were removed from this implementation.

Separate fitted sigmoid mappings calibrate each model's logit. Their two scores receive equal weight. The default flag threshold is 0.7749065220844583. This score is a model signal, not a verified probability that an image is AI-generated. The application neither reads generator provenance as its classifier nor consults ChatGPT online.

## Timing measurements

Host: Apple M1 Pro, 16 GB RAM, PyTorch MPS. Fresh Python processes, already-downloaded weights; not a controlled cold-disk benchmark. Profiling overhead, OS caching, GPU kernel caching and memory pressure affect these figures. Existing dev backend was resident but idle during measurements.

| Measurement | Before | After |
| --- | ---: | ---: |
| Python/library imports | 5.20 s | 6.93 s |
| Model initialization and verification | 4.21 s | 3.19 s |
| Imports + initialization | 9.41 s | 10.12 s |
| 405×400 image, repeated warmed pass, both models | 0.77 s | 0.39 s |
| 1086×1448 image, repeated warmed pass, both models | 5.30 s | 5.25 s |

These are illustrative samples, not a claim that the whole application starts faster by a measured percentage. Initialization work decreased, but library imports dominated and varied enough to hide that gain. The large image's four DEAR-r windows still took about five seconds. The small-image difference also includes caching effects; DEAR-r itself was not optimized.

An isolated five-image FiSeR comparison took 0.483 s for 24 blocks versus 0.298 s for 18 blocks, with identical feature arrays. This only measures FiSeR, not the ensemble.

Weights total 700,672,906 bytes (about 668 MiB). FiSeR's BF16 disk weights are expanded to FP32 at runtime. Model tensors and intermediate activations therefore cost substantially more memory than the download size. Native-resolution DEAR-r activations are especially expensive because the initial convolution/pooling strides are both one.

## Changes made

- Run original-image analysis by default; JPEG quality 75 and max-edge 512 transformations remain optional robustness tests. Previously, the lab ran all three complete ensemble analyses by default. The extension already requested only the original variant.
- Replace the per-image “Loading local model” label with accurate analysis wording. Backend models are loaded once per process, before the dev launcher opens the lab.
- Discard FiSeR blocks 19–24 before GPU transfer and omit their unnecessary computation. Keep the original layer-18 extraction and preprocessing.
- Hash each large weight file once per engine initialization, reusing that verified digest for the profile check. Integrity verification still runs at every startup.
- Log model initialization duration through Uvicorn's active logger.
- Update implementation checksums, backend/frontend profile identities, and the report's provenance. Model weights, head, calibration and threshold are unchanged.

Recomputed FiSeR logits for all 120 saved validation images. Maximum logit difference versus saved values: 2.5583e-7; maximum combined-score difference: 2.3763e-8; flag changes: zero. DEAR-r was unchanged; this comparison reused its saved logits. Direct before/after inference on one real and one GPT image also returned identical component logits. Unit/API tests and production extension build passed. This is performance/parity verification, not new independent accuracy validation.

Evidence is in `artifacts/performance-review/`: baseline/optimized JSON and profiling logs, the isolated truncation comparison, and validation parity results.

## Accuracy evidence

The existing untouched validation split has 60 real and 60 AI images. At the default threshold it flagged 0/60 real images and detected 44/60 AI images (73.3% recall).

| Generator | Detected |
| --- | ---: |
| GPT Image 2 | 9/15 (60.0%) |
| Nano Banana 2 | 7/15 (46.7%) |
| FLUX.2 | 14/15 (93.3%) |
| Qwen Image 2.1 | 14/15 (93.3%) |

These are small, related samples from one public dataset, with only 20 source-group IDs in validation. They do not establish a 1% false-positive rate on Reddit. Artwork, screenshots, memes, Reddit compression and unfamiliar generators need independent evaluation. Detection on a few personal ChatGPT examples can be good while overall recall remains incomplete.

## Extension suitability and priorities

Useful as a research prototype backed by a local service. Not yet suitable as a self-contained, lightweight Chrome extension that instantly scans a fast feed. Chrome currently receives no model weights or browser inference runtime; users need Python and a running backend. Multi-second large-image latency can make the global inference queue fall behind scrolling. Only supported still images on allowed hosts are eligible, not every visual asset on Reddit.

For the current desktop workflow, leave `npm run backend` running separately between lab sessions. `npm run dev` reuses it, eliminating repeated Python imports/model loading while that process survives. First-use kernels and new image shapes can still incur overhead.

Next experimental priorities:

1. Measure component-only accuracy and latency on independent Reddit-like images. Removing DEAR-r would be the biggest obvious latency opportunity, but needs its own calibration, threshold and validation; do not silently reuse the ensemble threshold.
2. For an all-local browser product, evaluate a smaller detector, or a distilled model, exported to ONNX and measured with ONNX Runtime Web/WebGPU on ordinary laptops. Quantization and reduced precision need score-parity and accuracy checks; export is not automatically a speedup.
3. If keeping this ensemble, consider a calibrated cascade or fewer native windows, with explicit validation of accuracy changes. The current DEAR path is the bottleneck.
4. Prioritize images that remain visible, bound the global queue, handle cancellation/stale work, and measure CPU/GPU/battery cost during actual scrolling. Existing client cancellation does not cancel an inference already executing in Python.
5. Check distribution rights before shipping: DEAR-r weights are released under CC BY-NC 4.0 with additional inherited restrictions; FiSeR uses the DINOv3 license.

Sources: [FiSeR model card](https://huggingface.co/heyongxin233/FiSeR-DINOv3-ViT-L16), [DEAR checkpoint card](https://huggingface.co/k-aisi-anti-deepfake/dear-checkpoints), [ONNX Runtime WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html), and the repository's source/profile/validation report.
