"""Grab labeled frames from a render into review sheets: python frames.py out/draft.mp4 name t1 t2 ..."""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

video, name, *times = sys.argv[1:]
tiles = []
for t in times:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", t, "-i", video, "-frames:v", "1", "-vf", "scale=640:-1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    path = Path(f"/tmp/frame-{t}.png")
    path.write_bytes(raw)
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 60, 22], fill="red")
    draw.text((5, 5), f"{t}s", fill="white")
    tiles.append(image)
w, h = tiles[0].size
cols = 3
sheet = Image.new("RGB", (w * cols, h * ((len(tiles) + cols - 1) // cols)), "black")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
sheet.save(Path(__file__).with_name("review") / f"{name}.jpg", quality=85)
