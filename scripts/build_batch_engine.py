#!/usr/bin/env python3
"""Build a dynamic-batch TensorRT engine from an ONNX model exported with batch fixed to 1.

The ResNet50 sample hard-codes batch 1 in its Reshape nodes (e.g. [1, 2048]). After making the
input batch dynamic we rewrite those to [-1, ...]; reshapes fed by
constants (bias broadcast) keep their leading 1. -1 (infer) is used rather than 0 (copy) because
the ONNX importer sets zero_is_placeholder=False on some reshapes, making 0 a literal size.

With --profiles 1,4,8 the engine gets one optimization profile per range ([1,1], [2,4], [5,8]),
each tuned at its upper bound, so small batches don't run tactics chosen for batch 8.

usage: scripts/build_batch_engine.py [--onnx M.onnx] [--max-batch 8 [--opt-batch N] | --profiles 1,4,8]
                                     [--fp16io] [-o out]
"""
import argparse
import os

import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default="/usr/src/tensorrt/data/resnet50/ResNet50.onnx")
ap.add_argument("--max-batch", type=int, default=8)
ap.add_argument("--opt-batch", type=int, default=0, help="tactic tuning batch (default: max)")
ap.add_argument("--profiles", help="comma-separated profile upper bounds, e.g. 1,4,8")
ap.add_argument("--fp16io", action="store_true", help="fp16 input/output tensors")
ap.add_argument("-o", "--output")
a = ap.parse_args()
bounds = sorted(int(x) for x in a.profiles.split(",")) if a.profiles else None
if bounds:
    a.max_batch = bounds[-1]
opt = a.opt_batch or a.max_batch
suffix = "p" + "-".join(map(str, bounds)) if bounds else "b%d" % a.max_batch
out = a.output or "models/%s_fp16%s_%s.engine" % (
    os.path.splitext(os.path.basename(a.onnx))[0], "io" if a.fp16io else "", suffix)

log = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(log)
net = builder.create_network(0)
parser = trt.OnnxParser(net, log)
if not parser.parse_from_file(a.onnx):
    raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

inp = net.get_input(0)
c, h, w = inp.shape[1:]
inp.shape = (-1, c, h, w)

for i in range(net.num_layers):
    layer = net.get_layer(i)
    if layer.type != trt.LayerType.SHUFFLE:
        continue
    layer.__class__ = trt.IShuffleLayer
    src = layer.get_input(0)
    dims = layer.reshape_dims
    if len(src.shape) and src.shape[0] == -1 and len(dims) and dims[0] == 1:
        layer.reshape_dims = (-1,) + tuple(dims)[1:]
        print("patched %-28s %s -> %s" % (layer.name, tuple(dims), layer.reshape_dims))

outp = net.get_output(0)
if outp.shape[0] != -1:
    raise SystemExit("output batch is still fixed: %s" % (outp.shape,))
if a.fp16io:
    inp.dtype = trt.float16
    outp.dtype = trt.float16

cfg = builder.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
ranges = [(lo + 1, hi, hi) for lo, hi in zip([0] + bounds[:-1], bounds)] if bounds else \
         [(1, opt, a.max_batch)]
for lo, op, hi in ranges:
    prof = builder.create_optimization_profile()
    prof.set_shape(inp.name, (lo, c, h, w), (op, c, h, w), (hi, c, h, w))
    cfg.add_optimization_profile(prof)
    print("profile: batch %d..%d (tuned at %d)" % (lo, hi, op))

blob = builder.build_serialized_network(net, cfg)
if blob is None:
    raise SystemExit("engine build failed")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "wb") as f:
    f.write(blob)
print("engine: %s  input %s [1..%d]x%dx%dx%d  output %s %s" %
      (out, inp.name, a.max_batch, c, h, w, outp.name, outp.shape))
