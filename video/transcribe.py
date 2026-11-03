import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("small.en", device="cpu", compute_type="int8")
out = {}
for path in sys.argv[1:]:
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=False, beam_size=5)
    words = []
    for segment in segments:
        for word in segment.words:
            words.append({"w": word.word.strip(), "s": round(word.start, 2), "e": round(word.end, 2)})
    out[Path(path).name] = words
    print(Path(path).name)
    print(" ".join(f"{w['w']}@{w['s']}" for w in words))
Path(__file__).with_name("transcript.json").write_text(json.dumps(out, indent=1))
