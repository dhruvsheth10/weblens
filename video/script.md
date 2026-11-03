# WebLens demo video: final script

About 1:55 at a relaxed pace (262 words). Solo project, no competition named anywhere, so the video works for any submission.

## Narration

| # | Time | Narration | On screen |
|---|---|---|---|
| 1 | 0:00–0:09 | Scroll online for a minute and you'll probably pass an AI-generated image. Most of them come with no label at all. | Animated wall of image tiles drifting up. One slows, sharpens, and gets a quiet question mark. |
| 2 | 0:09–0:20 | I'm Dhruv, and I built WebLens, a Chrome extension that checks the images you see for signs of AI, right on your own computer. | Your name, then the WebLens mark and title animating in. |
| 3 | 0:20–0:26 | It's for anyone who wants a second look before they trust or share a picture. | Short labels: readers, moderators, people sharing news. |
| 4 | 0:26–0:38 | Turn it on and browse like normal. Each image gets a small label with its score, and when the model finds a strong AI signal, WebLens flags it. | Feed clip, cropped and zoomed: green "no strong AI signal" labels on real photos, then purple "AI signal detected" on r/aiArt. Push in on the flagged throne post. |
| 5 | 0:38–0:46 | You can also drop any picture into the image lab and see exactly how it scores. | Image lab clip: strawberry at 11.4%, lion at 62.0%. |
| 6 | 0:46–0:55 | Everything runs locally, on a small server on your own laptop. No account, no API key, and nothing ever leaves your device. | Terminal starting the server, then the lab opening on localhost. Animated diagram: image to laptop to score, with a lock. |
| 7 | 0:55–1:15 | The detector is Sherlock 2, a model I trained on over two hundred thousand Reddit images, labeled by where and when they were posted. I tested it only on people and communities it never saw in training, so it couldn't memorize its way to a good score. | Model switching clip selecting Sherlock 2. Counter rolling to 229,044. Grid where the held-out authors and twelve communities slide aside. |
| — | 1:15–1:18 | (music only) | Stats beat, below. |
| 8 | 1:18–1:34 | On almost twenty-five thousand unseen posts, it caught 58 percent of AI images and wrongly flagged just 0.15 percent of real ones. I tuned it to stay quiet unless it's confident, because accusing a real artist does real harm. | Stats beat continues, then a brief glimpse of the Benchmarks tab. |
| 9 | 1:34–1:49 | Art is still its hardest case, so a flag means look closer. I trained it in Python with PyTorch, run it on a local FastAPI server, and built the extension in TypeScript and React. | Tool stack as a clean diagram: Python + PyTorch (training), FastAPI (local server), TypeScript + React (extension and lab). |
| 10 | 1:49–1:55 | WebLens started on Reddit. Next, it's coming to the rest of the web. | The Reddit card shrinks into a row of generic site cards that light up one by one. |
| 11 | 1:55–1:58 | WebLens. Look closer. | End card: WebLens mark, tagline, "Built by Dhruv". |

Other closing lines if "Look closer" doesn't fit: "WebLens. Know what you're looking at." or "WebLens. A second opinion on every image."

## Stats beat (on screen, about 3 s of music plus scene 8)

All numbers come from the fresh Reddit test (authors held out, posts collected after the previous model).

- 229,044 Reddit images collected
- 24,708 test posts from authors it never saw, plus 8,185 from 12 communities it never saw
- 58% of AI images caught
- 0.15% of real images flagged (31 of 20,699)
- About 30 ms per image on a laptop

## Clips received (Oct 23)

| Clip | Use | Notes |
|---|---|---|
| Feed (4:41) | Scene 4 | Crop to the post column. Skip political posts and celebrity faces. Left sidebar shows your Gmail address, so it gets cropped out. |
| Opened post, clear | Scene 4 | Lightning strike. Good. |
| Opened post, flagged | Scene 4 | Throne, 44.2%. Good. |
| Image lab | Scene 5 | Strawberry and lion. Good. |
| Benchmarks | Scene 8 | Brief glimpse only. Its table uses the validation folder, which overlaps training data, so the stats beat uses the fresh test numbers. |
| Model switching | Scene 7 | Good. |
| npm run dev | Scene 6 | Crop to the terminal pane only (hides the aiornot.key file names and the English assignment). Cover the `reddit-lens@0.1.4` line. |
| Opening localhost | Scene 6 | Good, apart from the name. |

Branding: keep the current clips and cover "Reddit Lens" in editing. Patch the sidebar logo and the status panel title with a matching WebLens label, and crop Settings to the model list so the "Reddit scanning" section stays out of frame.
