#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

// Summary statistics for a stage's samples. Percentiles use nearest rank on the sorted samples:
// index round(p * (n - 1)), so p50 of an even-sized set is the upper of the two middle values.
struct Summary {
  double mean = 0, p50 = 0, p90 = 0, p99 = 0, max = 0;
  size_t n = 0;
};

inline Summary summarize(std::vector<double> v) {
  Summary s;
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
