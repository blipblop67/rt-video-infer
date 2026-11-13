#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <nvtx3/nvToolsExt.h>

// Per-batch stage timings (a batch is one frame from each stream that had one ready).
// CPU stages use steady_clock, GPU stages use CUDA events. With one stream, batch == frame.
enum Stage {
  kWait,        // CPU: gathering the batch (idle time waiting for frames)
  kMap,         // CPU: GstBuffer map (+ EGLImage map / CUDA register in NVMM mode), all frames
  kUpload,      // GPU: host->device copy (CPU mode only)
  kPreprocess,  // GPU: RGBA -> normalized NCHW kernel
  kInfer,       // GPU: TensorRT enqueueV3
  kDownload,    // GPU: device->host copy of the output
  kPostprocess, // CPU: argmax
  kProcess,     // CPU: wall time from sample in hand to result ready (map..post+release)
  kLatency,     // capture timestamp -> result ready, worst frame in the batch (sync mode only)
  kNumStages
};

struct FrameRecord {
  uint64_t index = 0;
  int batch = 1;       // frames in this record
  int top_class = -1;  // first frame of the batch
  float top_score = 0;
  std::array<double, kNumStages> ms{};  // negative = not measured
};

class Profiler {
 public:
  explicit Profiler(uint64_t warmup) : warmup_(warmup) {}

  // arrived_now = frames that reached any appsink so far; used to report drops after warmup.
  void add(const FrameRecord& r, uint64_t arrived_now);
  void finish(uint64_t arrived_now);

  uint64_t measured_frames() const { return frames_; }
  void print_summary(const std::string& title) const;
  bool write_csv(const std::string& path) const;
  bool write_json(const std::string& path, const std::string& title) const;

 private:
  struct Stat { double mean = 0, p50 = 0, p90 = 0, p99 = 0, max = 0; size_t n = 0; };
  Stat stat(int stage) const;
  double fps() const;

  uint64_t warmup_;
  uint64_t seen_ = 0, frames_ = 0;
  std::vector<FrameRecord> records_;
  std::chrono::steady_clock::time_point t0_, t1_;
  uint64_t arrived_start_ = 0, arrived_end_ = 0;
};

// RAII NVTX range: shows up as a named span in Nsight Systems timelines.
struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};

inline double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
