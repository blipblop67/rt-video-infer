#!/usr/bin/env python3
"""Rebuild the README's benchmark numbers from the raw run outputs in benchmarks/.

For every <run>.json (written by rtinfer --json) prints FPS, drops, mean batch, per-frame
inference, process mean/p99 and e2e latency, plus GPU %, CPU % and VDD_IN power averaged from the
matching <run>.tegrastats.log, and the wall-clock span of the run from the tegrastats timestamps.

usage: scripts/summarize_benchmarks.py [benchmarks/]
"""
import glob
import json
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "benchmarks"


def tegrastats(path):
    if not os.path.exists(path):
        return None
    gpu, cpu, pw, stamps = [], [], [], []
    for line in open(path):
        if m := re.search(r"GR3D_FREQ (\d+)%", line):
            gpu.append(int(m.group(1)))
        if m := re.search(r"CPU \[([^\]]+)\]", line):
            cores = [int(c.split("%")[0]) for c in m.group(1).split(",") if "%" in c]
            cpu.append(sum(cores) / len(cores))
        if m := re.search(r"VDD_IN (\d+)mW", line):
            pw.append(int(m.group(1)))
        if m := re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)", line):
            stamps.append(m.group(1))
    avg = lambda v: sum(v) / len(v) if v else float("nan")
    return {"gpu": avg(gpu), "cpu": avg(cpu), "w": avg(pw) / 1000, "n": len(gpu),
            "span": f"{stamps[0]} .. {stamps[-1][11:]}" if stamps else "-"}


for d in sorted(glob.glob(os.path.join(root, "*", ""))):
    runs = sorted(glob.glob(os.path.join(d, "*.json")))
    if not runs:
        continue
    print(f"\n## {os.path.basename(os.path.dirname(d))}")
    print(f"{'run':<44} {'FPS':>6} {'drop':>5} {'batch':>5} {'inf/f':>6} {'proc':>6} {'p99':>6} "
          f"{'lat50':>6} {'GPU%':>5} {'CPU%':>5} {'W':>5}  recorded")
    for path in runs:
        j = json.load(open(path))
        s = j["stages_ms"]
        g = lambda k, f="mean": s.get(k, {}).get(f, float("nan"))
        mb = j["frames"] / max(j.get("batches", j["frames"]), 1)
        t = tegrastats(path[:-5] + ".tegrastats.log") or {}
        print(f"{os.path.basename(path)[:-5]:<44} {j['fps']:6.1f} {j['dropped']:5d} {mb:5.2f} "
              f"{g('infer') / mb:6.3f} {g('process(total)'):6.3f} {g('process(total)', 'p99'):6.3f} "
              f"{g('e2e latency', 'p50'):6.1f} {t.get('gpu', float('nan')):5.0f} "
              f"{t.get('cpu', float('nan')):5.0f} {t.get('w', float('nan')):5.1f}  {t.get('span', '-')}")
