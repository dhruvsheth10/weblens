# WebLens

Chrome extension that marks images it thinks were made by AI. Scoring runs on your own computer. The image stays there.

Reddit is the site it actually scans right now. I started there because I could collect a lot of posts and test on people the model had never seen. There's also an image lab in the local page: drop a file in, get a score, and optionally run the same check on compressed and resized copies. Other sites are the next step, not something it does yet.

## Install

Node and Python 3.12 or newer.

```sh
npm install
npm run setup
npm run dev
```

`npm run setup` downloads the model weights. They are large and they are not in this repo. Then open http://127.0.0.1:5173

The page talks to a FastAPI server on http://127.0.0.1:8000. `npm run backend` starts just that server. `LENS_DEVICE=cpu npm run dev` forces CPU. Otherwise it uses CUDA if it's there, then Apple Metal, then CPU.

## Extension

```sh
npm run build
```

Load the `extension-dist` folder from `chrome://extensions` with developer mode on. The server has to be running. If it isn't, the badge says the detector is offline.

On a Reddit image you get a label and a raw score. "AI signal detected" means it crossed the cutoff. A low score does not mean the picture is real.

## Sherlock 2

This is the default model. It's a ViT-S/16. I started from the public Community Forensics weights and fine-tuned the whole network in PyTorch on Reddit images.

I collected 229,044 images. I did not hand-label every file. The label comes from the community and the post date: older posts, communities that ban AI images, and communities that are for AI art.

The test set does not share authors with training, and 12 communities were held out completely. A score on someone the model already trained on mostly means it remembers them.

On 24,708 posts from authors it never saw:

- 58% of the AI images were caught
- 31 of 20,699 real images were flagged (0.15%)

The cutoff is strict on purpose. I'd rather miss an AI image than put a flag on a real drawing. Art and illustration are still the cases it gets wrong. Treat a flag as a reason to look closer.

Other models are still in Settings. Sherlock 2 is what the app loads unless you change it.

## What's in here

- `src/` — extension and the lab UI (TypeScript and React)
- `backend/` — the Python server
- `extension/` — manifest and the files the build copies
- `docs/` — longer notes on training and the other detectors
- `video/` — Remotion project for the demo video. Recordings and the rendered mp4 are not committed
- `tests/` — frontend tests

## Tests

```sh
npm test
npm run test:backend
npm run build
```

## Weights and data

Model files, the Reddit dump, and local benchmark folders are gitignored. Setup fetches the pinned weights and checks their hashes. Some of the optional older models have their own licenses, including noncommercial terms. Read `THIRD_PARTY_NOTICES.md` before you reuse any of them. Sherlock 2 is a research model. The data it was trained on has terms too.
