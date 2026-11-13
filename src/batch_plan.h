#pragma once
#include <algorithm>
#include <vector>

// Choosing a TensorRT optimization profile for a gathered batch (pure, unit tested).
struct BatchRange {
  int min, max;
};

// The profile with the smallest upper bound that fits `batch`, or -1 if none does.
inline int pick_profile(const std::vector<BatchRange>& profiles, int batch) {
  int best = -1;
  for (int i = 0; i < int(profiles.size()); ++i)
    if (batch >= 1 && profiles[i].max >= batch &&
        (best < 0 || profiles[i].max < profiles[best].max))
      best = i;
  return best;
}

// Rows the engine actually computes: a profile that starts above `batch` is padded up to its min.
inline int rows_for_profile(const BatchRange& p, int batch) { return std::max(batch, p.min); }
