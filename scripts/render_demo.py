#!/usr/bin/env python3
"""Overlay rtinfer predictions on a recorded clip and write an MP4 + a README-friendly GIF.

Frames are matched to predictions by timestamp: the recording and rtinfer's appsink branch come
from the same tee, so each video frame's offset from the first frame equals its prediction's PTS
offset from the first prediction.

usage: scripts/render_demo.py --video raw.mp4 --pred X.pred.csv --batches X.csv --labels L.txt --out assets/name
"""
import argparse
import bisect
import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--pred", required=True, help="rtinfer --pred-csv output (top-3 per frame)")
ap.add_argument("--batches", help="rtinfer --csv output (per-batch timings)")
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True, help="output path without extension")
ap.add_argument("--caption", default="ResNet50 FP16 · TensorRT 10 · Jetson Orin Nano Super")
ap.add_argument("--gif-width", type=int, default=480)
ap.add_argument("--gif-fps", type=int, default=10)
a = ap.parse_args()

labels = [l.strip() for l in open(a.labels)]
preds = sorted((int(r["pts_ns"]), r) for r in csv.DictReader(open(a.pred)) if r["stream"] == "0")
if not preds:
    raise SystemExit("no predictions in " + a.pred)
pts0 = preds[0][0]
pred_ms = [(p - pts0) / 1e6 for p, _ in preds]
timing = {}
if a.batches:
    for r in csv.DictReader(open(a.batches)):
        timing[r["batch_index"]] = (float(r["infer_ms"]), float(r["e2e latency_ms"]))


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/" + name, size)
    except OSError:
        return ImageFont.load_default()


cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
s = H / 540  # layout is designed at 960x540 and scaled
f_big, f_small, f_cap = font(int(26 * s), True), font(int(20 * s)), font(int(17 * s))

writer = cv2.VideoWriter(
    "appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc speed-preset=medium bitrate=2500 ! "
    "video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=%s.mp4" % a.out,
    cv2.CAP_GSTREAMER, 0, fps, (W, H))
if not writer.isOpened():
    raise SystemExit("could not open the GStreamer video writer")

gif_frames, gif_step, n = [], max(1, round(fps / a.gif_fps)), 0
recent = []  # pts of recent predictions, for a measured FPS readout
while True:
    t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    ok, frame = cap.read()
    if not ok:
        break
    i = bisect.bisect_right(pred_ms, t_ms + 1.0) - 1  # latest prediction at or before this frame
    if i < 0:
        continue
    _, r = preds[i]
    if not recent or recent[-1] != pred_ms[i]:
        recent = [x for x in recent if x > pred_ms[i] - 1000] + [pred_ms[i]]

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    # top-3 panel
    px, py, pw, row = int(16 * s), int(16 * s), int(420 * s), int(40 * s)
    d.rounded_rectangle([px, py, px + pw, py + int(16 * s) + 3 * row], radius=int(10 * s), fill=(0, 0, 0, 150))
    for k, (ck, sk) in enumerate([("class", "score"), ("class2", "score2"), ("class3", "score3")]):
        c, sc = int(r[ck]), float(r[sk])
        y = py + int(10 * s) + k * row
        bar = int((pw - int(24 * s)) * max(0.0, min(1.0, sc)))
        d.rounded_rectangle([px + int(12 * s), y + int(28 * s), px + int(12 * s) + max(bar, 2), y + int(34 * s)],
                            radius=int(3 * s), fill=(118, 185, 0, 255) if k == 0 else (200, 200, 200, 200))
        name = labels[c] if 0 <= c < len(labels) else str(c)
        d.text((px + int(12 * s), y), "%s" % name, font=f_big if k == 0 else f_small,
               fill=(255, 255, 255, 255 if k == 0 else 210))
        d.text((px + pw - int(12 * s), y), "%.0f%%" % (100 * sc), font=f_big if k == 0 else f_small,
               fill=(255, 255, 255, 255 if k == 0 else 210), anchor="ra")
    # bottom caption + live numbers
    bh = int(34 * s)
    d.rectangle([0, H - bh, W, H], fill=(0, 0, 0, 150))
    d.text((int(14 * s), H - bh // 2), a.caption, font=f_cap, fill=(255, 255, 255, 230), anchor="lm")
    inf, lat = timing.get(r["batch_index"], (None, None))
    stats = ["%.0f FPS" % len(recent)] if pred_ms[i] >= 1000 else []  # needs a full second of history
    if inf is not None:
        stats.insert(0, "infer %.1f ms" % inf)
    if lat is not None and lat >= 0:
        stats.insert(1, "camera→result %.0f ms" % lat)
    d.text((W - int(14 * s), H - bh // 2), "  ·  ".join(stats), font=f_cap, fill=(118, 185, 0, 255), anchor="rm")

    out = Image.alpha_composite(img, ov).convert("RGB")
    writer.write(cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR))
    if n % gif_step == 0:
        g = out.resize((a.gif_width, round(a.gif_width * H / W)), Image.LANCZOS)
        gif_frames.append(g.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE))  # Pillow 9.0 names
    n += 1
writer.release()

gif_frames[0].save(a.out + ".gif", save_all=True, append_images=gif_frames[1:], loop=0,
                   duration=round(1000 * gif_step / fps), optimize=True)
print("wrote %s.mp4 (%d frames) and %s.gif (%d frames)" % (a.out, n, a.out, len(gif_frames)))
