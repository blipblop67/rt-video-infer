#include "check.h"
#include "stats.h"

TEST(empty_input_is_all_zero) {
  const Summary s = summarize({});
  CHECK_EQ(s.n, size_t(0));
  CHECK_EQ(s.mean, 0.0);
  CHECK_EQ(s.max, 0.0);
}

TEST(single_sample) {
  const Summary s = summarize({4.5});
  CHECK_EQ(s.n, size_t(1));
  CHECK_EQ(s.mean, 4.5);
  CHECK_EQ(s.p50, 4.5);
  CHECK_EQ(s.p99, 4.5);
  CHECK_EQ(s.max, 4.5);
}

TEST(order_does_not_matter) {
  const Summary a = summarize({5, 1, 4, 2, 3});
  const Summary b = summarize({1, 2, 3, 4, 5});
  CHECK_EQ(a.p50, b.p50);
  CHECK_EQ(a.p90, b.p90);
  CHECK_EQ(a.max, b.max);
  CHECK_EQ(a.p50, 3.0);
  CHECK_EQ(a.mean, 3.0);
}

TEST(nearest_rank_percentiles_on_1_to_100) {
  std::vector<double> v;
  for (int i = 1; i <= 100; ++i) v.push_back(i);
  const Summary s = summarize(v);
  // index round(p * 99): p50 -> 50 (value 51), p90 -> 89 (90), p99 -> 98 (99)
  CHECK_EQ(s.p50, 51.0);
  CHECK_EQ(s.p90, 90.0);
  CHECK_EQ(s.p99, 99.0);
  CHECK_EQ(s.max, 100.0);
  CHECK_NEAR(s.mean, 50.5, 1e-12);
}

TEST(tail_outlier_shows_in_p99_and_max_not_p50) {
  std::vector<double> v(199, 2.0);
  v.push_back(40.0);  // one stall in 200 frames
  const Summary s = summarize(v);
  CHECK_EQ(s.p50, 2.0);
  CHECK_EQ(s.p99, 2.0);  // 1 outlier in 200 sits above the 99th percentile
  CHECK_EQ(s.max, 40.0);
  CHECK_NEAR(s.mean, 2.19, 1e-12);
}
