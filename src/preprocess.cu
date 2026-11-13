#include "preprocess.h"
#include <cuda_fp16.h>

namespace {

template <typename T> __device__ __forceinline__ T cvt(float v);
template <> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template <> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }

template <typename T>
__global__ void rgba_to_nchw(const uint8_t* __restrict__ src, int pitch, int w, int h,
                             T* __restrict__ dst, float3 scale, float3 bias) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const uchar4 px = *reinterpret_cast<const uchar4*>(src + y * pitch + x * 4);
  const int plane = w * h;
  const int i = y * w + x;
  // (x/255 - mean)/std folded into one FMA per channel
  dst[i]             = cvt<T>(px.x * scale.x + bias.x);
  dst[plane + i]     = cvt<T>(px.y * scale.y + bias.y);
  dst[2 * plane + i] = cvt<T>(px.z * scale.z + bias.z);
}

}  // namespace

void launch_rgba_to_nchw(const uint8_t* src, int src_pitch, int width, int height,
                         void* dst, bool dst_is_half, const NormParams& n,
                         cudaStream_t stream) {
  const float3 scale = make_float3(1.f / (255.f * n.std[0]), 1.f / (255.f * n.std[1]),
                                   1.f / (255.f * n.std[2]));
  const float3 bias = make_float3(-n.mean[0] / n.std[0], -n.mean[1] / n.std[1],
                                  -n.mean[2] / n.std[2]);
  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  if (dst_is_half)
    rgba_to_nchw<__half><<<grid, block, 0, stream>>>(src, src_pitch, width, height,
                                                     static_cast<__half*>(dst), scale, bias);
  else
    rgba_to_nchw<float><<<grid, block, 0, stream>>>(src, src_pitch, width, height,
                                                     static_cast<float*>(dst), scale, bias);
}
