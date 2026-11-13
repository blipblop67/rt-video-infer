#!/usr/bin/env bash
# Stream-count scaling: N copies of a clip, batched (dynamic engine) vs sequential (batch-1 engine).
# usage: scripts/bench_streams.sh <rt|max> [clip.mp4] [counts...]
#   rt  = each stream paced at its frame rate (real-time cameras); measures latency, drops, power
#   max = --no-sync, as fast as decode allows; measures aggregate throughput
# env: BATCHED=models/ResNet50_fp16_b8.engine SEQUENTIAL=models/ResNet50_fp16.engine EXTRA="..."
set -euo pipefail
cd "$(dirname "$0")/.."
PACE=${1:?rt or max}; shift
CLIP=${1:-media/test_1080p.mp4}; shift || true
COUNTS=("$@"); [[ ${#COUNTS[@]} -eq 0 ]] && COUNTS=(1 2 4 8)
BATCHED=${BATCHED:-models/ResNet50_fp16_b8.engine}
SEQUENTIAL=${SEQUENTIAL:-models/ResNet50_fp16.engine}
TAG=$(basename "${CLIP%.*}")

runs=()
for n in "${COUNTS[@]}"; do
  for kind in batched sequential; do
    engine=$BATCHED; [[ $kind == sequential ]] && engine=$SEQUENTIAL
    name="streams_${TAG}_${PACE}_${n}x_${kind}"
    args=(--engine "$engine" --source "file:$CLIP" --streams "$n" --print-every 0 --frames 0)
    [[ $PACE == max ]] && args+=(--no-sync)
    # shellcheck disable=SC2086
    scripts/run_profiled.sh "$name" "${args[@]}" ${EXTRA:-} 2>/dev/null | grep '^tegrastats' \
      > "reports/$name.ts" || true
    runs+=("$name")
  done
done

python3 -I - "${runs[@]}" <<'EOF'
import json, re, sys
hdr = f"{'run':<40} {'FPS':>6} {'drop%':>6} {'batch':>5} {'infer/f':>7} {'proc':>6} {'p99':>6} {'lat50':>6} {'lat99':>6} {'GPU%':>5} {'W':>5}"
print(hdr)
for name in sys.argv[1:]:
    j = json.load(open(f"reports/{name}.json")); s = j["stages_ms"]
    ts = open(f"reports/{name}.ts").read()
    gpu = re.search(r"GPU (\d+)%", ts); w = re.search(r"VDD_IN ([\d.]+) W", ts)
    g = lambda k, f="mean": s.get(k, {}).get(f, float("nan"))
    mb = j["frames"] / max(j["batches"], 1)
    drop = 100.0 * j["dropped"] / max(j["frames_at_appsink"], 1)
    print(f"{name.split('_', 2)[2]:<40} {j['fps']:6.1f} {drop:6.1f} {mb:5.2f} {g('infer')/mb:7.3f} "
          f"{g('process(total)'):6.2f} {g('process(total)','p99'):6.2f} {g('e2e latency','p50'):6.1f} "
          f"{g('e2e latency','p99'):6.1f} {gpu.group(1) if gpu else '-':>5} {w.group(1) if w else '-':>5}")
print("ms per batch unless noted; infer/f = inference per frame; lat = e2e latency (rt only)")
EOF
