#!/usr/bin/env bash
set -euo pipefail
# Safe generator: writes filter to a temporary file and calls ffmpeg with -filter_complex_script
# Usage: ./generate_sync_clip_safe.sh out.mkv [duration] [fps] [freq]

OUT=${1:-out_sync_test_safe.mkv}
DURATION=${2:-5}
FPS=${3:-25}
FREQ=${4:-1000}

TMPDIR=$(mktemp -d)
FILTER_FILE="$TMPDIR/filter.filt"

# write a single-line filter with no shell-embedded quotes to avoid parsing/linewrap issues
printf '%s\n' "[2:v]format=rgba[flash];" "[0:v][flash]overlay=enable=lt(mod(t\,1)\,1/${FPS})[v];" "[1:a]volume=if(lt(mod(t\,1)\,1/${FPS})\,20\,0)[a]" > "$FILTER_FILE"

echo "Using filter file: $FILTER_FILE"

ffmpeg -y \
  -f lavfi -i "color=black:s=1280x720:rate=${FPS}:d=${DURATION}" \
  -f lavfi -i "sine=frequency=${FREQ}:sample_rate=44100:duration=${DURATION}" \
  -f lavfi -i "color=white:s=1280x720:rate=${FPS}:d=${DURATION}" \
  -filter_complex_script "$FILTER_FILE" \
  -map "[v]" -map "[a]" -r ${FPS} -c:v libx264 -pix_fmt yuv420p -c:a pcm_s16le "$OUT"

WAV="${OUT%.*}.wav"
PNG="${OUT%.*}_waveform.png"
ffmpeg -y -i "$OUT" -vn -ac 1 -ar 44100 "$WAV"
ffmpeg -y -i "$WAV" -lavfi "aformat=channel_layouts=mono,showwavespic=s=1280x256:colors=white" -frames:v 1 "$PNG"

echo "Created: $OUT, $WAV, $PNG"
rm -rf "$TMPDIR"
