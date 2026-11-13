#!/usr/bin/env bash
# Generate an H.264 MP4 test clip. Orin Nano has no NVENC, so this uses x264enc (CPU);
# playback in the pipeline still decodes on NVDEC via nvv4l2decoder.
# usage: [PATTERN=ball] scripts/make_test_video.sh [out.mp4] [seconds] [WxH]
set -euo pipefail
OUT=${1:-media/test_1080p.mp4}
SECS=${2:-20}
RES=${3:-1920x1080}
PATTERN=${PATTERN:-smpte}  # e.g. PATTERN=ball for content that changes every frame
mkdir -p "$(dirname "$OUT")"
gst-launch-1.0 -q videotestsrc pattern=$PATTERN num-buffers=$((SECS * 30)) ! \
  "video/x-raw,format=I420,width=${RES%x*},height=${RES#*x},framerate=30/1" ! \
  x264enc speed-preset=ultrafast tune=zerolatency bitrate=8000 key-int-max=30 ! \
  video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location="$OUT"
# Force 4:2:0 + High profile: without it x264enc picks High 4:4:4, which NVDEC rejects.
echo "wrote $OUT"
