#!/usr/bin/env bash
# Run rtinfer with tegrastats logging alongside, then summarize GPU/CPU/RAM/power.
# usage: scripts/run_profiled.sh <run-name> [rtinfer args...]
#   e.g. scripts/run_profiled.sh nvmm_file --engine models/ResNet50_fp16.engine --source file:media/test_1080p.mp4
# Set NSYS=1 to also capture an Nsight Systems timeline (CUDA + NVTX + OS runtime).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?run name}; shift
mkdir -p reports
TS_LOG=reports/$NAME.tegrastats.log
rm -f "$TS_LOG"

tegrastats --interval 200 --logfile "$TS_LOG" &
TS_PID=$!
trap 'kill $TS_PID 2>/dev/null || true' EXIT

CMD=(./build/rtinfer --csv "reports/$NAME.csv" --json "reports/$NAME.json" "$@")
if [[ ${NSYS:-0} == 1 ]]; then
  nsys profile -t cuda,nvtx,osrt --force-overwrite=true -o "reports/$NAME" "${CMD[@]}"
else
  "${CMD[@]}"
fi

kill $TS_PID 2>/dev/null || true
wait $TS_PID 2>/dev/null || true
trap - EXIT

# tegrastats line: ... RAM used/totalMB ... CPU [a%@f,b%@f,...] GR3D_FREQ g% ... VDD_IN cur/avg mW
awk '
  { for (i = 1; i <= NF; i++) {
      if ($i == "RAM") { split($(i+1), r, "/"); ram += r[1] }
      if ($i == "GR3D_FREQ") { g = $(i+1); sub("%.*", "", g); gpu += g }
      if ($i == "VDD_IN") { split($(i+1), p, "/"); sub("mW", "", p[1]); pw += p[1] }
      if ($i == "CPU") { c = $(i+1); gsub(/[\[\]]/, "", c); m = split(c, cores, ",");
        s = 0; for (j = 1; j <= m; j++) { split(cores[j], x, "%"); s += x[1] } cpu += s / m }
    }
    n++ }
  END { if (n) printf "tegrastats (%d samples): GPU %.0f%%  CPU avg %.0f%%  RAM %.0f MB  VDD_IN %.1f W\n",
                       n, gpu/n, cpu/n, ram/n, pw/n/1000 }' "$TS_LOG"
