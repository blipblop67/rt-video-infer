#include "batch_plan.h"
#include "check.h"

namespace {
// The engine built by: scripts/build_batch_engine.py --profiles 1,4,8
const std::vector<BatchRange> kP148 = {{1, 1}, {2, 4}, {5, 8}};
}  // namespace

TEST(each_batch_size_gets_the_tightest_profile) {
  CHECK_EQ(pick_profile(kP148, 1), 0);
  for (int b = 2; b <= 4; ++b) CHECK_EQ(pick_profile(kP148, b), 1);
  for (int b = 5; b <= 8; ++b) CHECK_EQ(pick_profile(kP148, b), 2);
}

TEST(batch_beyond_every_profile_is_rejected) {
  CHECK_EQ(pick_profile(kP148, 9), -1);
  CHECK_EQ(pick_profile(kP148, 0), -1);
  CHECK_EQ(pick_profile({}, 1), -1);
}

TEST(profile_order_in_engine_does_not_matter) {
  const std::vector<BatchRange> shuffled = {{5, 8}, {1, 1}, {2, 4}};
  CHECK_EQ(pick_profile(shuffled, 1), 1);
  CHECK_EQ(pick_profile(shuffled, 3), 2);
  CHECK_EQ(pick_profile(shuffled, 7), 0);
}

TEST(single_wide_profile) {
  const std::vector<BatchRange> b8 = {{1, 8}};
  CHECK_EQ(pick_profile(b8, 1), 0);
  CHECK_EQ(pick_profile(b8, 8), 0);
  CHECK_EQ(rows_for_profile(b8[0], 3), 3);  // runs exactly the gathered frames
}

TEST(gap_below_a_profile_pads_up_to_its_min) {
  // An engine with only a [4..8] profile: a batch of 2 must run 4 rows (2 are ignored).
  const std::vector<BatchRange> p = {{4, 8}};
  const int i = pick_profile(p, 2);
  CHECK_EQ(i, 0);
  CHECK_EQ(rows_for_profile(p[i], 2), 4);
  CHECK_EQ(rows_for_profile(p[i], 6), 6);
}
