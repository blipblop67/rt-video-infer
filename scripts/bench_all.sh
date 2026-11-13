#!/usr/bin/env bash
# Run the memory-mode x pacing matrix through run_profiled.sh and print one comparison table.
# usage: scripts/bench_all.sh [engine] [clip.mp4]
set -euo pipefail
cd "$(dirname "$0")/.."
ENGINE=${1:-models/ResNet50_fp16.engine}
CLIP=${2:-media/test_1080p.mp4}
TAG=$(basename "${ENGINE%.engine}")

declare -A MODES=([cpu]="--mem cpu" [nvmm]="--mem nvmm" [nvmm-nocache]="--mem nvmm --no-egl-cache")
runs=()
for mode in cpu nvmm nvmm-nocache; do
  for pace in rt max; do
    name="bench_${TAG}_${mode}_${pace}"
    extra=()
    [[ $pace == max ]] && extra+=(--no-sync)
    # shellcheck disable=SC2086
    scripts/run_profiled.sh "$name" --engine "$ENGINE" --source "file:$CLIP" ${MODES[$mode]} \
      --frames 0 --print-every 0 "${extra[@]}" 2>/dev/null | grep '^tegrastats' > "reports/$name.ts" || true
    runs+=("$name")
  done
done

python3 -I - "${runs[@]}" <<'EOF'
import json, re, sys
print(f"{'run':<44} {'FPS':>6} {'drop':>5} {'map':>6} {'pre':>6} {'infer':>6} {'proc':>6} {'p99':>6} {'GPU%':>5} {'W':>5}")
for name in sys.argv[1:]:
    j = json.load(open(f"reports/{name}.json")); s = j["stages_ms"]
    ts = open(f"reports/{name}.ts").read()
    gpu = re.search(r"GPU (\d+)%", ts); w = re.search(r"VDD_IN ([\d.]+) W", ts)
    g = lambda k, f="mean": s.get(k, {}).get(f, float("nan"))
    print(f"{name.removeprefix('bench_'):<44} {j['fps']:6.1f} {j['dropped']:5d} {g('map'):6.3f} "
          f"{g('preprocess'):6.3f} {g('infer'):6.3f} {g('process(total)'):6.3f} "
          f"{g('process(total)','p99'):6.3f} {gpu.group(1) if gpu else '-':>5} {w.group(1) if w else '-':>5}")
print("(ms = mean unless noted; p99 = process p99)")
EOF
