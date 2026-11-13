#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

#include <cuda.h>
#include <gst/gst.h>

struct NvBufSurface;

// kNvmm: nvvidconv (VIC) writes RGBA into an NVMM surface; CUDA reads it in place via EGL.
// kCpu:  nvvidconv copies RGBA into system memory; we upload it with cudaMemcpy2D.
enum class MemMode { kNvmm, kCpu };

struct SourceConfig {
  std::string spec;  // test | file:PATH | image:PATH | csi:N | usb:/dev/videoN | mjpeg:/dev/videoN[@WxH] | rtsp:URL | pipeline:GST
  int width = 224;   // VIC scales to the network input size
  int height = 224;
  MemMode mem = MemMode::kNvmm;
  bool sync = true;  // appsink sync: true = real-time pacing, false = as fast as possible
  bool cache_egl = true;  // kNvmm: keep each pool surface EGL-mapped + CUDA-registered
  // Called from the streaming thread when a frame or EOS reaches appsink (wakes a gatherer).
  std::function<void()> on_ready;
};

// One RGBA frame borrowed from the pipeline. Exactly one of host/dev is set.
struct FrameView {
  GstSample* sample = nullptr;
  GstBuffer* buffer = nullptr;
  GstMapInfo map{};
  const uint8_t* host = nullptr;  // kCpu
  const uint8_t* dev = nullptr;   // kNvmm
  int pitch = 0;
  NvBufSurface* surf = nullptr;
  CUgraphicsResource egl_res = nullptr;
  bool cached = false;  // egl_res is owned by the cache, not this frame
};

class GstSource {
 public:
  ~GstSource();
  bool open(const SourceConfig& cfg);
  void close();

  // Blocks up to timeout_ms; nullptr on timeout, EOS or error (check eos()/error()).
  GstSample* pull(int timeout_ms);
  bool acquire(GstSample* sample, FrameView& out);
  void release(FrameView& f);

  // Capture-to-now latency of a buffer in ms (pipeline clock), or -1 if unknown.
  double latency_ms(GstSample* sample) const;

  bool eos() const { return eos_; }
  bool error() const { return error_; }
  uint64_t frames_arrived() const { return arrived_.load(); }
  const std::string& pipeline_desc() const { return desc_; }

 private:
  void poll_bus();
  bool register_surface(NvBufSurface* surf, CUgraphicsResource& res, const uint8_t*& dev,
                        int& pitch);
  void clear_egl_cache();
  static GstPadProbeReturn count_probe(GstPad*, GstPadProbeInfo*, gpointer self);
  static GstFlowReturn on_new_sample(GstElement*, gpointer self);
  static void on_eos(GstElement*, gpointer self);

  SourceConfig cfg_;
  std::string desc_;
  GstElement* pipeline_ = nullptr;
  GstElement* sink_ = nullptr;
  GstBus* bus_ = nullptr;
  bool eos_ = false, error_ = false;
  std::atomic<uint64_t> arrived_{0};

  // nvvidconv recycles a small, fixed pool of NVMM buffers, so registering each surface once
  // replaces a per-frame EGL map + CUDA register/unregister with a hash lookup.
  struct EglEntry {
    int fd;
    CUgraphicsResource res;
    const uint8_t* dev;
    int pitch;
  };
  std::unordered_map<NvBufSurface*, EglEntry> egl_cache_;
};
