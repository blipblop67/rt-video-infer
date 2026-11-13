#include "profiler.h"

#include <algorithm>
#include <cstdio>

namespace {
const char* kNames[kNumStages] = {"wait",  "map",         "upload(H2D)",  "preprocess", "infer",
                                  "download(D2H)", "postprocess", "process(total)", "e2e latency"};
}

void Profiler::add(const FrameRecord& r, uint64_t arrived_now) {
  if (++seen_ <= warmup_) {
    if (seen_ == warmup_) {
      t0_ = std::chrono::steady_clock::now();
      arrived_start_ = arrived_now;
    }
    return;
  }
  if (records_.empty() && warmup_ == 0) {  // count the first batch's own frames as arrivals
    t0_ = std::chrono::steady_clock::now();
    arrived_start_ = arrived_now >= uint64_t(r.batch) ? arrived_now - r.batch : 0;
  }
  records_.push_back(r);
  frames_ += r.batch;
}

void Profiler::finish(uint64_t arrived_now) {
  t1_ = std::chrono::steady_clock::now();
  arrived_end_ = arrived_now;
}

Profiler::Stat Profiler::stat(int stage) const {
  std::vector<double> v;
  v.reserve(records_.size());
  for (const auto& r : records_)
    if (r.ms[stage] >= 0) v.push_back(r.ms[stage]);
  Stat s;
  s.n = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) { return v[std::min(v.size() - 1, size_t(p * (v.size() - 1) + 0.5))]; };
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  s.p50 = pct(0.50);
  s.p90 = pct(0.90);
  s.p99 = pct(0.99);
  s.max = v.back();
  return s;
}

double Profiler::fps() const {
  const double sec = std::chrono::duration<double>(t1_ - t0_).count();
  return sec > 0 ? frames_ / sec : 0;
}

void Profiler::print_summary(const std::string& title) const {
  const uint64_t arrived = arrived_end_ - arrived_start_;
  const uint64_t dropped = arrived > frames_ ? arrived - frames_ : 0;
  printf("\n=== %s ===\n", title.c_str());
  printf("frames measured: %lu in %zu batches (mean batch %.2f, warmup %lu batches excluded)  "
         "throughput: %.1f FPS\n",
         (unsigned long)frames_, records_.size(),
         records_.empty() ? 0.0 : double(frames_) / records_.size(), (unsigned long)warmup_, fps());
  printf("frames at appsink: %lu  dropped (newer frame replaced stale): %lu (%.1f%%)\n",
         (unsigned long)arrived, (unsigned long)dropped, arrived ? 100.0 * dropped / arrived : 0.0);
  printf("%-16s %8s %8s %8s %8s %8s   (ms per batch)\n", "stage", "mean", "p50", "p90", "p99",
         "max");
  for (int s = 0; s < kNumStages; ++s) {
    const Stat st = stat(s);
    if (!st.n) continue;
    printf("%-16s %8.3f %8.3f %8.3f %8.3f %8.3f\n", kNames[s], st.mean, st.p50, st.p90, st.p99,
           st.max);
  }
  // Process time bounds the achievable frame rate when the GPU pipeline is serialized.
  const Stat p = stat(kProcess), inf = stat(kInfer);
  const double per_batch = records_.empty() ? 1.0 : double(frames_) / records_.size();
  if (inf.n) printf("infer per frame: %.3f ms\n", inf.mean / per_batch);
  if (p.n)
    printf("max sustainable rate (frames per batch * 1000/mean process): %.1f FPS\n",
           per_batch * 1000.0 / p.mean);
}

bool Profiler::write_csv(const std::string& path) const {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return false;
  fprintf(f, "batch_index,batch_size,class,score");
  for (int s = 0; s < kNumStages; ++s) fprintf(f, ",%s_ms", kNames[s]);
  fprintf(f, "\n");
  for (const auto& r : records_) {
    fprintf(f, "%lu,%d,%d,%.5f", (unsigned long)r.index, r.batch, r.top_class, r.top_score);
    for (int s = 0; s < kNumStages; ++s) fprintf(f, ",%.4f", r.ms[s]);
    fprintf(f, "\n");
  }
  fclose(f);
  return true;
}

bool Profiler::write_json(const std::string& path, const std::string& title) const {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return false;
  const uint64_t arrived = arrived_end_ - arrived_start_;
  fprintf(f, "{\n  \"run\": \"%s\",\n  \"frames\": %lu,\n  \"batches\": %zu,\n  \"fps\": %.3f,\n",
          title.c_str(), (unsigned long)frames_, records_.size(), fps());
  fprintf(f, "  \"frames_at_appsink\": %lu,\n  \"dropped\": %lu,\n  \"stages_ms\": {\n",
          (unsigned long)arrived, (unsigned long)(arrived > frames_ ? arrived - frames_ : 0));
  bool first = true;
  for (int s = 0; s < kNumStages; ++s) {
    const Stat st = stat(s);
    if (!st.n) continue;
    fprintf(f, "%s    \"%s\": {\"mean\": %.4f, \"p50\": %.4f, \"p90\": %.4f, \"p99\": %.4f, "
               "\"max\": %.4f}",
            first ? "" : ",\n", kNames[s], st.mean, st.p50, st.p90, st.p99, st.max);
    first = false;
  }
  fprintf(f, "\n  }\n}\n");
  fclose(f);
  return true;
}
