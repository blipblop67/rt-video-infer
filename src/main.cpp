// Real-time video inference: GStreamer (NVDEC/VIC) -> CUDA preprocess -> TensorRT, with
// per-stage profiling (CUDA events + steady_clock), NVTX ranges and CSV/JSON reports.
// N sources are gathered into one batch per iteration (one frame from each stream that has one).
#include <algorithm>
#include <array>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gst/gst.h>

#include "gst_source.h"
#include "preprocess.h"
#include "profiler.h"
#include "trt_engine.h"

#define CUDA_CHECK(x)                                                              \
  do {                                                                             \
    cudaError_t e_ = (x);                                                          \
    if (e_ != cudaSuccess) {                                                       \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_sigint(int) { g_stop = 1; }

struct Args {
  std::string engine;
  SourceConfig src;  // settings shared by every stream; spec is filled per stream
  std::vector<std::string> specs;
  int streams = 0;            // replicate a single --source N times
  int batch_timeout_ms = 0;   // 0 = wait until every live stream has a frame
  uint64_t frames = 600;
  uint64_t warmup = 30;
  std::string csv, json, labels, pred_csv;
  NormParams norm{{0.45f, 0.45f, 0.45f}, {0.225f, 0.225f, 0.225f}};  // TRT ResNet50 sample
  int print_every = 30;
};

void usage(const char* argv0) {
  printf(
      "usage: %s --engine model.engine [options]\n"
      "  --source SPEC     test | file:clip.mp4 | image:pic.jpg | csi:0 | usb:/dev/video0 |\n"
      "                    mjpeg:/dev/video0[@1280x720] (hardware JPEG decode) |\n"
      "                    rtsp://... | pipeline:'<gst elements ending in nvvidconv>'  (test)\n"
      "                    repeat --source for several streams\n"
      "  --streams N       run N copies of a single --source                        (1)\n"
      "  --batch-timeout MS  after the first frame, wait at most MS for the other streams;\n"
      "                    0 = wait for every live stream                           (0)\n"
      "  --mem nvmm|cpu    zero-copy NVMM+EGL interop, or system memory + H2D copy   (nvmm)\n"
      "  --no-sync         don't pace to the clock: run as fast as frames decode\n"
      "  --no-egl-cache    nvmm: re-register every frame with CUDA instead of once per surface\n"
      "  --frames N        frames to measure after warmup, 0 = until EOS/Ctrl-C     (600)\n"
      "  --warmup N        batches excluded from statistics                        (30)\n"
      "  --labels FILE     one class name per line\n"
      "  --mean r,g,b --std r,g,b   input normalization          (0.45 / 0.225 x3)\n"
      "  --csv FILE        per-batch stage timings\n"
      "  --json FILE       summary statistics\n"
      "  --pred-csv FILE   per-frame predictions: batch, stream, pts, class, score\n"
      "  --print-every N   log predictions every N batches, 0 = never            (30)\n"
      "Engines with a dynamic batch run exactly the frames gathered; a fixed-batch engine\n"
      "smaller than the stream count is run several times per batch.\n",
      argv0);
}

bool parse_triplet(const char* s, float out[3]) {
  return sscanf(s, "%f,%f,%f", &out[0], &out[1], &out[2]) == 3;
}

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    const char* v = nullptr;
    if (k == "--engine" && (v = next())) a.engine = v;
    else if (k == "--source" && (v = next())) a.specs.push_back(v);
    else if (k == "--streams" && (v = next())) a.streams = atoi(v);
    else if (k == "--batch-timeout" && (v = next())) a.batch_timeout_ms = atoi(v);
    else if (k == "--mem" && (v = next())) {
      if (!strcmp(v, "cpu")) a.src.mem = MemMode::kCpu;
      else if (!strcmp(v, "nvmm")) a.src.mem = MemMode::kNvmm;
      else return false;
    } else if (k == "--no-sync") a.src.sync = false;
    else if (k == "--no-egl-cache") a.src.cache_egl = false;
    else if (k == "--frames" && (v = next())) a.frames = strtoull(v, nullptr, 10);
    else if (k == "--warmup" && (v = next())) a.warmup = strtoull(v, nullptr, 10);
    else if (k == "--labels" && (v = next())) a.labels = v;
    else if (k == "--mean" && (v = next())) { if (!parse_triplet(v, a.norm.mean)) return false; }
    else if (k == "--std" && (v = next())) { if (!parse_triplet(v, a.norm.std)) return false; }
    else if (k == "--csv" && (v = next())) a.csv = v;
    else if (k == "--json" && (v = next())) a.json = v;
    else if (k == "--pred-csv" && (v = next())) a.pred_csv = v;
    else if (k == "--print-every" && (v = next())) a.print_every = atoi(v);
    else return false;
  }
  if (a.specs.empty()) a.specs.push_back("test");
  if (a.streams > 1) {
    if (a.specs.size() != 1) return false;
    a.specs.assign(a.streams, a.specs[0]);
  }
  return !a.engine.empty();
}

std::vector<std::string> load_labels(const std::string& path) {
  std::vector<std::string> out;
  std::ifstream f(path);
  for (std::string line; std::getline(f, line);) out.push_back(line);
  return out;
}

// Wakes the gatherer whenever any appsink receives a frame or EOS. A generation counter (instead
// of holding the mutex while pulling) keeps appsink's own locks out of our critical section.
struct ReadySignal {
  std::mutex mu;
  std::condition_variable cv;
  uint64_t gen = 0;
  void notify() {
    {
      std::lock_guard<std::mutex> l(mu);
      ++gen;
    }
    cv.notify_one();
  }
};

using Sources = std::vector<std::unique_ptr<GstSource>>;

// Collect at most one sample per stream. Returns when every live stream delivered, or
// timeout_ms after the first frame arrived (timeout_ms > 0), or on stop/all-EOS.
int gather(Sources& srcs, std::vector<GstSample*>& got, int timeout_ms, ReadySignal& ready) {
  using clk = std::chrono::steady_clock;
  int n = 0;
  clk::time_point first{};
  clk::time_point last_progress = clk::now();
  for (;;) {
    uint64_t gen;
    {
      std::lock_guard<std::mutex> l(ready.mu);
      gen = ready.gen;
    }
    int live = 0;
    for (size_t i = 0; i < srcs.size(); ++i) {
      if (got[i] || srcs[i]->eos() || srcs[i]->error()) continue;
      if ((got[i] = srcs[i]->pull(0))) {
        if (n++ == 0) first = clk::now();
      } else if (!srcs[i]->eos() && !srcs[i]->error()) {
        ++live;
      }
    }
    if (live == 0 || g_stop) return n;
    const auto now = clk::now();
    if (n > 0 && timeout_ms > 0 && now >= first + std::chrono::milliseconds(timeout_ms)) return n;
    if (n == 0 && now - last_progress > std::chrono::seconds(2)) {
      fprintf(stderr, "no frame from %d live stream(s) for 2 s, still waiting...\n", live);
      last_progress = now;
    }

    std::unique_lock<std::mutex> lk(ready.mu);
    const auto deadline = (n > 0 && timeout_ms > 0) ? first + std::chrono::milliseconds(timeout_ms)
                                                   : now + std::chrono::milliseconds(100);
    ready.cv.wait_until(lk, deadline, [&] { return ready.gen != gen || g_stop; });
  }
}

float read_out(const void* buf, size_t i, bool half) {
  return half ? __half2float(static_cast<const __half*>(buf)[i]) : static_cast<const float*>(buf)[i];
}

}  // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  Args args;
  if (!parse_args(argc, argv, args)) {
    usage(argv[0]);
    return 2;
  }
  std::signal(SIGINT, on_sigint);

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));  // create the primary context; driver-API EGL calls reuse it

  TrtEngine engine;
  if (!engine.load(args.engine)) return 1;
  printf("engine: %s  input %s %dx%d %s batch %s%s  output %s [%zu] %s\n", args.engine.c_str(),
         engine.input_name().c_str(), engine.input_w(), engine.input_h(),
         engine.input_is_half() ? "fp16" : "fp32", engine.profiles_desc().c_str(),
         engine.dynamic_batch() ? " (profiles)" : " (fixed)", engine.output_name().c_str(),
         engine.output_count(),
         engine.output_is_half() ? "fp16" : "fp32");
  const auto labels = args.labels.empty() ? std::vector<std::string>{} : load_labels(args.labels);

  const int W = engine.input_w(), H = engine.input_h();
  const int nstreams = static_cast<int>(args.specs.size());
  const int chunk = engine.max_batch();  // frames per enqueue
  const int nchunks = (nstreams + chunk - 1) / chunk;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  enum { E_START, E_PRE, E_INFER, E_OUT, E_END, E_COUNT };
  std::vector<std::array<cudaEvent_t, E_COUNT>> ev(nchunks);
  for (auto& c : ev)
    for (auto& e : c) CUDA_CHECK(cudaEventCreate(&e));

  uint8_t* d_staging = nullptr;  // CPU mode: RGBA upload target, one slot per batch row
  const size_t rgba_bytes = size_t(W) * H * 4;
  if (args.src.mem == MemMode::kCpu) CUDA_CHECK(cudaMalloc(&d_staging, rgba_bytes * chunk));
  void* h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_out, engine.output_bytes(nstreams)));

  ReadySignal ready;
  Sources sources;
  for (const auto& spec : args.specs) {
    SourceConfig cfg = args.src;
    cfg.spec = spec;
    cfg.width = W;
    cfg.height = H;
    cfg.on_ready = [&ready] { ready.notify(); };
    sources.push_back(std::make_unique<GstSource>());
    if (!sources.back()->open(cfg)) return 1;
    printf("stream %zu: %s\n", sources.size() - 1, sources.back()->pipeline_desc().c_str());
  }
  auto arrived_total = [&] {
    uint64_t a = 0;
    for (auto& s : sources) a += s->frames_arrived();
    return a;
  };

  FILE* pred = args.pred_csv.empty() ? nullptr : fopen(args.pred_csv.c_str(), "w");
  if (pred) fprintf(pred, "batch_index,stream,pts_ns,class,score,class2,score2,class3,score3\n");

  std::string title = nstreams > 1 && std::all_of(args.specs.begin(), args.specs.end(),
                                                  [&](auto& s) { return s == args.specs[0]; })
                          ? std::to_string(nstreams) + "x " + args.specs[0]
                          : (nstreams > 1 ? std::to_string(nstreams) + " streams" : args.specs[0]);
  title += std::string(" | mem=") +
           (args.src.mem == MemMode::kCpu ? "cpu" : args.src.cache_egl ? "nvmm" : "nvmm-nocache") +
           (args.src.sync ? " | sync" : " | no-sync") + " | batch " + engine.profiles_desc() +
           (engine.dynamic_batch() ? "" : " (fixed)");

  Profiler prof(args.warmup);
  std::vector<uint64_t> processed(nstreams, 0);
  uint64_t idx = 0;
  bool ok = true;

  struct Item {
    int stream;
    GstSample* sample;
    FrameView f;
    int cls[3];  // top-3, best first
    float score[3];
  };
  std::vector<GstSample*> got(nstreams);
  std::vector<Item> items;
  items.reserve(nstreams);

  while (ok && !g_stop && (args.frames == 0 || prof.measured_frames() < args.frames)) {
    FrameRecord rec;
    rec.ms.fill(-1);
    rec.index = idx;

    auto t = std::chrono::steady_clock::now();
    std::fill(got.begin(), got.end(), nullptr);
    int n;
    {
      NvtxRange r("gather");
      n = gather(sources, got, args.batch_timeout_ms, ready);
    }
    rec.ms[kWait] = ms_since(t);
    if (std::any_of(sources.begin(), sources.end(), [](auto& s) { return s->error(); })) ok = false;
    if (n == 0) {
      for (auto* s : got) if (s) gst_sample_unref(s);
      break;  // all streams ended, error, or Ctrl-C
    }

    NvtxRange batch_range("batch");
    const auto t_proc = std::chrono::steady_clock::now();
    items.clear();
    for (int i = 0; i < nstreams; ++i)
      if (got[i]) items.push_back(Item{i, got[i], {}, {-1, -1, -1}, {0, 0, 0}});
    {
      NvtxRange r("map");
      t = std::chrono::steady_clock::now();
      for (auto& it : items) {
        if (!sources[it.stream]->acquire(it.sample, it.f)) {
          ok = false;
          break;
        }
      }
      rec.ms[kMap] = ms_since(t);
    }

    const int nch = ok ? (n + chunk - 1) / chunk : 0;
    for (int c = 0; c < nch; ++c) {
      const int base = c * chunk, m = std::min(chunk, n - base);
      auto& e = ev[c];
      CUDA_CHECK(cudaEventRecord(e[E_START], stream));
      if (args.src.mem == MemMode::kCpu)
        for (int j = 0; j < m; ++j)
          CUDA_CHECK(cudaMemcpy2DAsync(d_staging + j * rgba_bytes, W * 4, items[base + j].f.host,
                                       items[base + j].f.pitch, W * 4, H, cudaMemcpyHostToDevice,
                                       stream));
      CUDA_CHECK(cudaEventRecord(e[E_PRE], stream));
      for (int j = 0; j < m; ++j) {
        const auto& f = items[base + j].f;
        const bool cpu = args.src.mem == MemMode::kCpu;
        launch_rgba_to_nchw(cpu ? d_staging + j * rgba_bytes : f.dev, cpu ? W * 4 : f.pitch, W, H,
                            engine.input_slot(j), engine.input_is_half(), args.norm, stream);
      }
      CUDA_CHECK(cudaEventRecord(e[E_INFER], stream));
      if (!engine.infer(m, stream)) {
        fprintf(stderr, "enqueueV3 failed\n");
        ok = false;
        break;
      }
      CUDA_CHECK(cudaEventRecord(e[E_OUT], stream));
      CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(h_out) + engine.output_bytes(base),
                                 engine.output_buffer(), engine.output_bytes(m),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaEventRecord(e[E_END], stream));
    }
    {
      NvtxRange r("gpu sync");
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaGetLastError());

    if (ok) {
      NvtxRange r("postprocess");
      t = std::chrono::steady_clock::now();
      const size_t k = engine.output_count();
      for (int j = 0; j < n; ++j) {
        Item& it = items[j];
        std::fill(std::begin(it.cls), std::end(it.cls), -1);
        std::fill(std::begin(it.score), std::end(it.score), -1e30f);
        for (size_t i = 0; i < k; ++i) {  // insertion into a 3-slot top list
          float v = read_out(h_out, j * k + i, engine.output_is_half());
          int c = int(i);
          for (int r = 0; r < 3; ++r)
            if (v > it.score[r]) std::swap(v, it.score[r]), std::swap(c, it.cls[r]);
        }
      }
      rec.ms[kPostprocess] = ms_since(t);
      rec.top_class = items[0].cls[0];
      rec.top_score = items[0].score[0];
    }
    if (args.src.sync && ok) {
      double worst = -1;
      for (auto& it : items) worst = std::max(worst, sources[it.stream]->latency_ms(it.sample));
      rec.ms[kLatency] = worst;
    }
    if (pred && ok)
      for (auto& it : items) {
        const GstBuffer* b = gst_sample_get_buffer(it.sample);
        fprintf(pred, "%lu,%d,%ld,%d,%.6f,%d,%.6f,%d,%.6f\n", (unsigned long)idx, it.stream,
                GST_BUFFER_PTS_IS_VALID(b) ? (long)GST_BUFFER_PTS(b) : -1L, it.cls[0], it.score[0],
                it.cls[1], it.score[1], it.cls[2], it.score[2]);
      }
    for (auto& it : items) {
      if (it.f.sample) sources[it.stream]->release(it.f);  // also unrefs the sample
      else gst_sample_unref(it.sample);
      ++processed[it.stream];
    }
    if (!ok) break;
    rec.ms[kProcess] = ms_since(t_proc);
    rec.batch = n;

    double up = 0, pre = 0, inf = 0, down = 0;
    for (int c = 0; c < nch; ++c) {
      float g;
      if (args.src.mem == MemMode::kCpu) cudaEventElapsedTime(&g, ev[c][E_START], ev[c][E_PRE]), up += g;
      cudaEventElapsedTime(&g, ev[c][E_PRE], ev[c][E_INFER]), pre += g;
      cudaEventElapsedTime(&g, ev[c][E_INFER], ev[c][E_OUT]), inf += g;
      cudaEventElapsedTime(&g, ev[c][E_OUT], ev[c][E_END]), down += g;
    }
    if (args.src.mem == MemMode::kCpu) rec.ms[kUpload] = up;
    rec.ms[kPreprocess] = pre;
    rec.ms[kInfer] = inf;
    rec.ms[kDownload] = down;

    prof.add(rec, arrived_total());
    if (args.print_every > 0 && idx % args.print_every == 0) {
      printf("batch %5lu [%d]  process %.2f ms  latency %.1f ms ", (unsigned long)idx, n,
             rec.ms[kProcess], rec.ms[kLatency]);
      for (int j = 0; j < std::min(n, 4); ++j) {
        const auto& it = items[j];
        const char* name = it.cls[0] < int(labels.size()) ? labels[it.cls[0]].c_str() : "";
        printf("| s%d %d %s %.3f ", it.stream, it.cls[0], name, it.score[0]);
      }
      printf("%s\n", n > 4 ? "| ..." : "");
    }
    ++idx;
  }
  prof.finish(arrived_total());
  if (pred) fclose(pred);

  bool src_error = false;
  for (auto& s : sources) src_error |= s->error();
  prof.print_summary(title);
  if (nstreams > 1)
    for (int i = 0; i < nstreams; ++i)
      printf("  stream %d: %lu processed / %lu arrived (whole run)\n", i,
             (unsigned long)processed[i], (unsigned long)sources[i]->frames_arrived());
  for (auto& s : sources) s->close();

  if (!args.csv.empty() && prof.write_csv(args.csv)) printf("wrote %s\n", args.csv.c_str());
  if (!args.json.empty() && prof.write_json(args.json, title))
    printf("wrote %s\n", args.json.c_str());
  if (pred) printf("wrote %s\n", args.pred_csv.c_str());

  for (auto& c : ev)
    for (auto& e : c) cudaEventDestroy(e);
  cudaStreamDestroy(stream);
  if (d_staging) cudaFree(d_staging);
  cudaFreeHost(h_out);
  return ok && !src_error ? 0 : 1;
}
