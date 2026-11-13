# Raw benchmark data

Unedited output from the runs behind the tables in the main [README](../README.md#benchmarks).
To recompute every summary number from these files:

```bash
python3 scripts/summarize_benchmarks.py benchmarks
```

Each run has three files:

| File | Written by | Contents |
|---|---|---|
| `<run>.json` | `rtinfer --json` | FPS, frames, drops, and mean/p50/p90/p99/max for every stage |
| `<run>.csv` | `rtinfer --csv` | one row per batch: batch size, top class and score, and every stage's time in ms |
| `<run>.tegrastats.log` | `scripts/run_profiled.sh` | `tegrastats` sampled every 200 ms during the run (GPU %, CPU, RAM, temperatures, power rails), with timestamps |

## Setup

Jetson Orin Nano Super Developer Kit · L4T R36.5 (JetPack 6) · `nvpmodel` MAXN_SUPER · CUDA 12.6 ·
TensorRT 10.3 · GStreamer 1.20. The model is ResNet50 FP16 at 224×224, from the TensorRT sample
`ResNet50.onnx`.

## Folders and the commands that produced them

| Folder | README table | Command |
|---|---|---|
| `single_stream/` | Single stream | `scripts/bench_all.sh models/ResNet50_fp16.engine media/test_1080p.mp4` |
| `multi_stream_realtime/` | Multi-stream, real time | `BATCHED=models/ResNet50_fp16_p1-4-8.engine scripts/bench_streams.sh rt media/test_1080p.mp4` |
| `multi_stream_max/` | Multi-stream, max throughput | `BATCHED=models/ResNet50_fp16_p1-4-8.engine scripts/bench_streams.sh max media/ball_360p.mp4` |
| `demos/demo_8x1080p.*` | (demo run) | `scripts/run_profiled.sh demo_8x1080p --engine models/ResNet50_fp16_p1-4-8.engine --source file:media/test_1080p.mp4 --streams 8 --frames 0` |
| `demos/demo_usb_camera.*` | README GIF | `scripts/record_demo.sh /dev/video0 12 demo_usb_camera` (`.pred.csv` has the top 3 classes per frame) |

The test clips come from `scripts/make_test_video.sh` (`media/test_1080p.mp4`: 20 s SMPTE bars;
`media/ball_360p.mp4`: `PATTERN=ball`, 640×360). Engines come from `scripts/build_engine.sh` and
`scripts/build_batch_engine.py --profiles 1,4,8`. Engines are tied to the GPU and TensorRT
version, so rebuild them on your own device.

## Caveats

- **The desktop (GNOME) was running.** It keeps the GPU 6–40% busy at idle. Comparisons within
  one folder were run back to back and hold up; absolute numbers move by up to 25% between
  sessions, and would likely be better and steadier without the desktop.
- **Each configuration was run once,** for 600–4,500 frames, so the data doesn't show
  run-to-run spread.
- **Power is VDD_IN,** the whole board including the desktop, not the pipeline alone.
- **In `multi_stream_max/`, high "drop" counts are expected.** Decode outruns inference and
  appsink keeps only the newest frame. Throughput is the metric there.
- **Some older data is no longer here.** The "single-profile" column in the README's
  max-throughput table, and the first `ResNet50_fp16_b8` real-time runs, were overwritten by
  later runs with the same names, so their raw files no longer exist. Every other README number
  is reproduced by `summarize_benchmarks.py` from the files in this folder.
