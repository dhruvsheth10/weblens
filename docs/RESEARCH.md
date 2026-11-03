# Current detection research

The active local engine is FiSeR + DEAR-r, selected explicitly for this prototype. Model loading and successful API responses establish operational correctness, not universal AI image detection accuracy.

## Primary sources

- [FiSeR paper](https://arxiv.org/abs/2606.00606), [authors' code](https://github.com/heyongxin233/FiSeR), and [released encoder](https://huggingface.co/heyongxin233/FiSeR-DINOv3-ViT-L16). The released model needs a classification head; this project trains a portable linear head from labeled references.
- [DEAR paper](https://arxiv.org/abs/2606.10309), [authors' implementation](https://github.com/anti-fake/dear), and [checkpoint release](https://huggingface.co/k-aisi-anti-deepfake/dear-checkpoints). The DEAR-r weights are research/noncommercial. Native-resolution window sampling in this project is a laptop memory adaptation, not the paper's full-frame protocol.
- [MLLM-generated image benchmark](https://arxiv.org/abs/2608.01258), [released data](https://huggingface.co/datasets/zr-zhang/MLLM-Generated-Image-Detection-Dataset). Our deterministic subset includes Nano Banana 2, GPT Image 2, FLUX.2, Qwen Image 2.1 and real images. Files, source groups, exclusions and split membership are recorded locally.
- [SynthID-Image](https://arxiv.org/abs/2510.09263). A paired encoder/decoder verifies deliberately embedded watermarks. No local image decoder is included here. The public [SynthID repository](https://github.com/google-deepmind/synthid-text) is for text. The [SynthID portal](https://synthid.com/) is a manual verification service, with terms prohibiting frontend use as a bulk automated API.

## Remaining evidence needed

Test independently sourced real photos, artwork, memes, game/UI screenshots and actual Reddit previews. Add fresh generations from the exact Gemini version being used. Compare the two components and their ensemble at the same false-positive target, including compression and resizing. The generated validation report covers a small public subset and originals only; it does not establish general performance on Reddit or future generators.
