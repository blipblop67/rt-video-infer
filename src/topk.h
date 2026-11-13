#pragma once
#include <cstddef>
#include <utility>

// Top-K (class, score) over n scores, best first, in one pass with K-slot insertion. get(i)
// returns score i. Slots stay (-1, -1e30) when n < K. Ties keep the lower class index first.
template <int K, typename Get>
void top_k(size_t n, Get get, int (&cls)[K], float (&score)[K]) {
  for (int r = 0; r < K; ++r) cls[r] = -1, score[r] = -1e30f;
  for (size_t i = 0; i < n; ++i) {
    float v = get(i);
    int c = static_cast<int>(i);
    for (int r = 0; r < K; ++r)
      if (v > score[r]) std::swap(v, score[r]), std::swap(c, cls[r]);
  }
}
