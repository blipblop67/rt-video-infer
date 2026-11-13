#!/usr/bin/env bash
# Record a live camera demo for the README. rtinfer classifies the camera while a tee in the same
# GStreamer pipeline records those exact frames (matched later by PTS); render_demo.py then
# overlays the predictions and writes assets/<name>.mp4 and assets/<name>.gif.
# usage: scripts/record_demo.sh [/dev/video0] [seconds] [name]
#   env: ENGINE=models/ResNet50_fp16_p1-4-8.engine  SIZE=1280x720 (camera MJPEG mode)
set -euo pipefail
cd "$(dirname "$0")/.."
DEV=${1:-/dev/video0}
SECS=${2:-10}
NAME=${3:-demo_usb_camera}
ENGINE=${ENGINE:-models/ResNet50_fp16_p1-4-8.engine}
SIZE=${SIZE:-1280x720}
LABELS=/usr/src/tensorrt/data/resnet50/class_labels.txt
RAW=media/${NAME}_raw.mp4
mkdir -p media reports assets

# Camera MJPEG -> NVJPG decode -> tee: (1) VIC downscale + x264 recording, (2) rtinfer's branch
# (rtinfer appends "! RGBA 224x224 caps ! appsink" after the final nvvidconv).
PIPE="v4l2src device=$DEV ! image/jpeg,width=${SIZE%x*},height=${SIZE#*x},framerate=30/1 ! \
nvv4l2decoder mjpeg=1 ! tee name=t \
t. ! queue ! nvvidconv ! video/x-raw,format=I420,width=960,height=540 ! \
x264enc speed-preset=ultrafast tune=zerolatency bitrate=6000 key-int-max=30 ! h264parse ! \
mp4mux ! filesink location=$RAW \
t. ! queue ! nvvidconv"

echo "recording ${SECS}s from $DEV ..."
./build/rtinfer --engine "$ENGINE" --source "pipeline:$PIPE" --labels "$LABELS" \
  --frames $((SECS * 30)) --warmup 0 --print-every 30 \
  --pred-csv "reports/$NAME.pred.csv" --csv "reports/$NAME.csv" 2>&1 | grep -vE 'NvMM|Opening|^\s*$'

python3 -I scripts/render_demo.py --video "$RAW" --pred "reports/$NAME.pred.csv" \
  --batches "reports/$NAME.csv" --labels "$LABELS" --out "assets/$NAME"
