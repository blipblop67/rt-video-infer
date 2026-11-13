#include "trt_engine.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
  }
} g_logger;

size_t volume(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
  return v;
}

size_t elem_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    default: return 0;  // unsupported for this demo
  }
}

}  // namespace

TrtEngine::~TrtEngine() {
  profiles_.clear();
  engine_.reset();
  runtime_.reset();
  if (d_in_) cudaFree(d_in_);
  if (d_out_) cudaFree(d_out_);
  if (d_act_) cudaFree(d_act_);
}

bool TrtEngine::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open engine %s\n", path.c_str());
    return false;
  }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  runtime_.reset(nvinfer1::createInferRuntime(g_logger));
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) return false;

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      in_name_ = name;
    else
      out_name_ = name;
  }
  if (engine_->getNbIOTensors() != 2 || in_name_.empty() || out_name_.empty()) {
    fprintf(stderr, "expected exactly one input and one output tensor\n");
    return false;
  }

  in_dims_ = engine_->getTensorShape(in_name_.c_str());
  if (in_dims_.nbDims != 4 || in_dims_.d[1] != 3) {
    fprintf(stderr, "expected NCHW input with 3 channels\n");
    return false;
  }
  dynamic_ = in_dims_.d[0] == -1;
  in_h_ = static_cast<int>(in_dims_.d[2]);
  in_w_ = static_cast<int>(in_dims_.d[3]);

  const auto in_t = engine_->getTensorDataType(in_name_.c_str());
  const auto out_t = engine_->getTensorDataType(out_name_.c_str());
  if (!elem_size(in_t) || !elem_size(out_t)) {
    fprintf(stderr, "only FP32/FP16 I/O tensors are supported\n");
    return false;
  }
  in_half_ = in_t == nvinfer1::DataType::kHALF;
  out_half_ = out_t == nvinfer1::DataType::kHALF;

  if (!dynamic_) {
    Profile p;
    p.min = p.max = max_batch_ = static_cast<int>(in_dims_.d[0]);
    p.ctx.reset(engine_->createExecutionContext());
    if (!p.ctx) return false;
    profiles_.push_back(std::move(p));
  } else {
    // One context per profile; contexts run one at a time on one stream, so they can share the
    // largest profile's activation memory instead of each allocating its own.
    const int n = engine_->getNbOptimizationProfiles();
    int64_t act_bytes = 0;
    for (int i = 0; i < n; ++i) act_bytes = std::max(act_bytes, engine_->getDeviceMemorySizeForProfileV2(i));
    if (cudaMalloc(&d_act_, act_bytes) != cudaSuccess) return false;
    cudaStream_t s;
    cudaStreamCreate(&s);
    for (int i = 0; i < n; ++i) {
      Profile p;
      using Sel = nvinfer1::OptProfileSelector;
      p.min = static_cast<int>(engine_->getProfileShape(in_name_.c_str(), i, Sel::kMIN).d[0]);
      p.max = static_cast<int>(engine_->getProfileShape(in_name_.c_str(), i, Sel::kMAX).d[0]);
      p.ctx.reset(engine_->createExecutionContext(
          nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
      if (!p.ctx) return false;
      p.ctx->setDeviceMemoryV2(d_act_, act_bytes);
      if (!p.ctx->setOptimizationProfileAsync(i, s)) return false;
      max_batch_ = std::max(max_batch_, p.max);
      profiles_.push_back(std::move(p));
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
  }

  for (const auto& p : profiles_) ranges_.push_back({p.min, p.max});

  // Resolve the output shape at max batch to size the buffers once.
  Profile& top = profiles_[profile_for(max_batch_)];
  nvinfer1::Dims full = in_dims_;
  full.d[0] = max_batch_;
  if (dynamic_) top.ctx->setInputShape(in_name_.c_str(), full);
  top.last_rows = max_batch_;
  const nvinfer1::Dims out_dims = top.ctx->getTensorShape(out_name_.c_str());
  in_item_bytes_ = volume(full) / max_batch_ * elem_size(in_t);
  out_item_count_ = volume(out_dims) / max_batch_;
  out_item_bytes_ = out_item_count_ * elem_size(out_t);

  if (cudaMalloc(&d_in_, in_item_bytes_ * max_batch_) != cudaSuccess) return false;
  if (cudaMalloc(&d_out_, out_item_bytes_ * max_batch_) != cudaSuccess) return false;
  for (auto& p : profiles_) {
    p.ctx->setTensorAddress(in_name_.c_str(), d_in_);
    p.ctx->setTensorAddress(out_name_.c_str(), d_out_);
  }
  return true;
}

int TrtEngine::profile_for(int batch) const { return pick_profile(ranges_, batch); }

int TrtEngine::rows_for(int batch) const {
  if (!dynamic_) return max_batch_;
  const int p = profile_for(batch);
  return p < 0 ? -1 : rows_for_profile(ranges_[p], batch);
}

std::string TrtEngine::profiles_desc() const {
  std::string s;
  for (const auto& p : profiles_)
    s += (s.empty() ? "" : ",") + (p.min == p.max ? std::to_string(p.min)
                                                  : std::to_string(p.min) + ".." + std::to_string(p.max));
  return s;
}

bool TrtEngine::infer(int batch, cudaStream_t stream) {
  const int pi = profile_for(batch);
  if (batch < 1 || pi < 0) return false;
  Profile& p = profiles_[pi];
  const int rows = rows_for(batch);
  if (dynamic_ && rows != p.last_rows) {
    nvinfer1::Dims d = in_dims_;
    d.d[0] = rows;
    if (!p.ctx->setInputShape(in_name_.c_str(), d)) return false;
    p.last_rows = rows;
  }
  return p.ctx->enqueueV3(stream);
}
