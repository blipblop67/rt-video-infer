#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "batch_plan.h"

// Single-input / single-output TensorRT 10 engine wrapper (enqueueV3, named tensors).
// Batch is dim 0. A static engine always runs its fixed batch (unused rows are ignored).
// A dynamic engine may carry several optimization profiles (e.g. [1], [2..4], [5..8]); each gets
// its own execution context, all sharing one activation buffer, and every call runs on the
// profile with the smallest upper bound that fits the batch.
class TrtEngine {
 public:
  ~TrtEngine();
  bool load(const std::string& engine_path);
  bool infer(int batch, cudaStream_t stream);

  int max_batch() const { return max_batch_; }
  bool dynamic_batch() const { return dynamic_; }
  int profile_for(int batch) const;
  int rows_for(int batch) const;  // rows actually computed for a batch of this size
  std::string profiles_desc() const;

  int input_w() const { return in_w_; }
  int input_h() const { return in_h_; }
  bool input_is_half() const { return in_half_; }
  void* input_slot(int i) const { return static_cast<char*>(d_in_) + i * in_item_bytes_; }

  bool output_is_half() const { return out_half_; }
  size_t output_count() const { return out_item_count_; }  // per batch item
  size_t output_bytes(int rows) const { return rows * out_item_bytes_; }
  void* output_buffer() const { return d_out_; }
  const std::string& input_name() const { return in_name_; }
  const std::string& output_name() const { return out_name_; }

 private:
  struct Profile {
    int min = 1, max = 1;
    int last_rows = -1;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  };

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::vector<Profile> profiles_;
  std::vector<BatchRange> ranges_;  // profiles_[i].min/max, for pick_profile
  std::string in_name_, out_name_;
  nvinfer1::Dims in_dims_{};
  int in_w_ = 0, in_h_ = 0;
  int max_batch_ = 1;
  bool dynamic_ = false;
  bool in_half_ = false, out_half_ = false;
  size_t in_item_bytes_ = 0, out_item_count_ = 0, out_item_bytes_ = 0;
  void* d_in_ = nullptr;
  void* d_out_ = nullptr;
  void* d_act_ = nullptr;  // activation memory shared by all contexts (they run serially)
};
