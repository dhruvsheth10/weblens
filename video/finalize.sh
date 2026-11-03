#!/usr/bin/env bash
# Brings the Remotion render to about -14 LUFS for YouTube and fades the last 2 s of audio to silence.
set -euo pipefail
cd "$(dirname "$0")/out"
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 weblens-demo.mp4)
FADE_START=$(python3 -c "print(round($DURATION - 2, 3))")
ffmpeg -v error -y -i weblens-demo.mp4 -c:v copy \
  -af "volume=11.4dB,alimiter=limit=0.84:attack=5:release=60:level=false,afade=t=out:st=$FADE_START:d=2:curve=qsin,aresample=48000" \
  -c:a aac -b:a 320k -movflags +faststart WebLens-demo-final.mp4
ffmpeg -v info -i WebLens-demo-final.mp4 -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+(I|Peak):"
cp WebLens-demo-final.mp4 ~/Downloads/WebLens-demo.mp4
