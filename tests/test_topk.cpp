#include <vector>

#include "check.h"
#include "topk.h"

namespace {
void top3(const std::vector<float>& v, int (&c)[3], float (&s)[3]) {
  top_k(v.size(), [&](size_t i) { return v[i]; }, c, s);
}
}  // namespace

TEST(top3_best_first) {
  int c[3];
  float s[3];
  top3({0.1f, 0.6f, 0.05f, 0.2f, 0.05f}, c, s);
  CHECK_EQ(c[0], 1);
  CHECK_EQ(c[1], 3);
  CHECK_EQ(c[2], 0);
  CHECK_EQ(s[0], 0.6f);
  CHECK_EQ(s[1], 0.2f);
  CHECK_EQ(s[2], 0.1f);
}

TEST(top3_with_best_at_the_end) {
  int c[3];
  float s[3];
  top3({1, 2, 3, 4, 5, 6}, c, s);  // ascending input: every element displaces the slots
  CHECK_EQ(c[0], 5);
  CHECK_EQ(c[1], 4);
  CHECK_EQ(c[2], 3);
}

TEST(ties_keep_lower_index_first) {
  int c[3];
  float s[3];
  top3({0.5f, 0.5f, 0.5f, 0.5f}, c, s);
  CHECK_EQ(c[0], 0);
  CHECK_EQ(c[1], 1);
  CHECK_EQ(c[2], 2);
}

TEST(fewer_scores_than_k_leaves_empty_slots) {
  int c[3];
  float s[3];
  top3({0.3f, 0.7f}, c, s);
  CHECK_EQ(c[0], 1);
  CHECK_EQ(c[1], 0);
  CHECK_EQ(c[2], -1);
}

TEST(negative_scores_such_as_logits) {
  int c[3];
  float s[3];
  top3({-5.f, -1.f, -3.f, -0.5f}, c, s);
  CHECK_EQ(c[0], 3);
  CHECK_EQ(c[1], 1);
  CHECK_EQ(c[2], 2);
}

TEST(k1_is_argmax) {
  int c[1];
  float s[1];
  const std::vector<float> v{0.2f, 0.9f, 0.4f};
  top_k(v.size(), [&](size_t i) { return v[i]; }, c, s);
  CHECK_EQ(c[0], 1);
}
