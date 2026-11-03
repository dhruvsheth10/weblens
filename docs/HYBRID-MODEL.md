# FerretNet + OpenSynthID

Select **FerretNet + OpenSynthID** in Image lab → Settings or the extension popup. Both older options remain available. The backend lazily loads the additional model when selected, sharing the already-loaded Ferret network. Run `npm run setup` on a fresh installation to download and verify the pinned weights. Inference uses local files and no remote detection service. The current Chrome extension still connects to the local Python backend; this change does not embed browser model inference.

The frozen model identity is `ferretnet-opensynthid-v1`, pipeline `ferret256-opensynthid512-max-centered-logits-v1`. Its numerical profile is in `backend/models/hybrid-profile.json` and mirrored in the frontend. That profile, both checkpoint checksums, and the preprocessing implementation are versioned in benchmark requests. The additional checkpoint is about 86 MB; setup sources it from fyxme's OpenSynthID v0.1 revision `e1384859752c8311b8a168125ac16a3eb240d7f5`.

Both networks run on every image so component outputs remain available and the signal slider works consistently. Ferret's native 256-pixel crop is unchanged. OpenSynthID uses the original six-channel 512-pixel preprocessing: RGB, wavelet residual, FFT magnitude and carrier mask. It does not JPEG-reencode ordinary inputs. OpenCV follows the upstream decoding policy for original images; the explicit robustness variants use the application's transformed image. This is an independent classifier, not Google's official watermark decoder.

For component logits `f` and `o`, the combined logit is `max(f - logit(0.8304235935211183), o - logit(0.9013915360638743))`. The displayed score is its sigmoid. At the default 0.5 threshold this equals flagging either component at its frozen cut. The score is not a probability of AI origin. The image lab displays each component's original score/cut and the API explicitly returns `watermarkVerified=false`.

This conservative OpenSynthID cut was selected on development data before the external evaluations. No Kaggle results are used to recalibrate it. Use the benchmark MCP with `version="ferretnet-opensynthid-v1"` to reproduce evaluations. The old calibration audit and user validation remain locked.

OpenSynthID's model card declares Apache-2.0; source attribution is in `backend/models/licenses/opensynthid/NOTICE`. No spectral reverse-SynthID code or research-only codebook is included in the application.
