#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct NormParams {
  float mean[3];
  float std[3];
};

// RGBA8 (pitch-linear, any pitch) -> planar NCHW, normalized: (x/255 - mean) / std.
// dst_is_half selects FP16 vs FP32 output to match the engine's input tensor type.
void launch_rgba_to_nchw(const uint8_t* src, int src_pitch, int width, int height,
                         void* dst, bool dst_is_half, const NormParams& norm,
                         cudaStream_t stream);
