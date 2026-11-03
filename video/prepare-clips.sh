#!/usr/bin/env bash
# Cuts and crops the raw screen recordings into H.264 segments used by the Remotion project.
# Crops keep the old app name, the Google sign-in email, and editor file names out of frame.
set -euo pipefail
cd "$(dirname "$0")"
SRC=~/Downloads
OUT=public/clips
STILLS=public/stills
mkdir -p "$OUT" "$STILLS"

FEED="$SRC/BetterCapture Oct 23 2026.mp4"
CLEAR="$SRC/BetterCapture_2026-10-23-15.22.38.mp4"
FLAGGED="$SRC/BetterCapture_2026-10-23-15.25.20.mp4"
LAB="$SRC/BetterCapture_2026-10-23-15.27.26.mp4"
BENCH="$SRC/BetterCapture_2026-10-23-15.28.21.mp4"
MODELS="$SRC/BetterCapture_2026-10-23-15.28.39.mp4"
TERMINAL="$SRC/BetterCapture_2026-10-23-15.29.05.mp4"

clip() { # name source start end filter
  ffmpeg -v error -y -ss "$3" -to "$4" -i "$2" -an -vf "$5,fps=30,format=yuv420p" \
    -c:v libx264 -preset slow -crf 16 -movflags +faststart "$OUT/$1.mp4"
}
still() { # name source time filter
  ffmpeg -v error -y -ss "$3" -i "$2" -frames:v 1 -vf "$4" "$STILLS/$1.png"
}

FEED_CROP="crop=1395:860:345:100"
clip feed-real "$FEED" 218.5 225 "$FEED_CROP"
clip feed-ai "$FEED" 178.6 183 "$FEED_CROP"
clip post-clear "$CLEAR" 5 9.8 "crop=1750:984:830:330,scale=1920:1080"
clip post-flagged "$FLAGGED" 2.2 6.6 "crop=1750:984:830:380,scale=1920:1080"
clip lab-real "$LAB" 4 8.5 "crop=2690:1513:600:180,scale=1920:1080"
clip lab-ai "$LAB" 11.5 17.5 "crop=2690:1513:600:180,scale=1920:1080"
clip models "$MODELS" 2.5 8 "crop=1580:889:1700:380,scale=1920:1080"
clip terminal "$TERMINAL" 8.3 10.75 "crop=1488:250:715:1000"
clip bench "$BENCH" 0 9 "crop=2690:1513:600:180,scale=1920:1080"

still art-miss "$FEED" 158 "crop=776:597:559:347"
still t-pinup "$FEED" 180.5 "crop=394:532:750:105"
still t-miko "$FEED" 180.5 "crop=544:319:675:881"
still t-street "$FEED" 221 "crop=394:243:750:188"
still t-autumn "$FEED" 221 "crop=328:515:788:647"
still t-gold "$FEED" 233.5 "crop=768:449:563:188"
still t-drawing "$FEED" 233.5 "crop=468:319:713:881"
still t-crab "$FEED" 265 "crop=318:318:788:638"
still t-lion "$LAB" 14.5 "crop=726:725:1232:608"
still t-throne "$FLAGGED" 3 "crop=1401:810:1012:590"
still t-lightning "$CLEAR" 6 "crop=1451:810:979:557"
still t-strawberry "$LAB" 7 "crop=726:725:1232:608"
ls -la "$OUT" "$STILLS"
