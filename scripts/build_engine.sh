#!/usr/bin/env bash
# Build a TensorRT engine from ONNX and dump trtexec's per-layer profile.
# usage: scripts/build_engine.sh [model.onnx] [fp16|fp16io|fp32] [out.engine]
set -euo pipefail
cd "$(dirname "$0")/.."

ONNX=${1:-/usr/src/tensorrt/data/resnet50/ResNet50.onnx}
PREC=${2:-fp16}
OUT=${3:-models/$(basename "${ONNX%.*}")_${PREC}.engine}
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
mkdir -p models reports

flags=()
[[ $PREC == fp16 ]] && flags+=(--fp16)
# FP16 I/O: the preprocess kernel writes half directly, so TensorRT skips its FP32->FP16
# input reformat node. rtinfer reads the I/O types from the engine, so both variants work.
[[ $PREC == fp16io ]] && flags+=(--fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw)

# Engines are tied to the GPU + TensorRT version: rebuild after a JetPack upgrade.
"$TRTEXEC" --onnx="$ONNX" "${flags[@]}" --saveEngine="$OUT" \
  --memPoolSize=workspace:1024 --skipInference 2>&1 | tail -3

# Isolated inference benchmark (no video, no preprocessing): the upper bound for the pipeline.
"$TRTEXEC" --loadEngine="$OUT" --useCudaGraph --noDataTransfers --warmUp=500 --duration=5 \
  --dumpProfile --separateProfileRun --exportProfile="reports/$(basename "$OUT").layers.json" \
  2>&1 | grep -E 'Throughput|Latency:|GPU Compute Time:' || true
echo "engine: $OUT"
