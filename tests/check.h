#pragma once
// Minimal test harness (no external dependency): TEST(name) registers a case, CHECK/CHECK_EQ
// record failures with file:line, and test_main.cpp runs everything and sets the exit code.
#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace check {

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}
inline int& failures() {
  static int n = 0;
  return n;
}
struct Register {
  Register(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

template <typename A, typename B>
void eq(const A& a, const B& b, const char* ea, const char* eb, const char* file, int line) {
  if (a == b) return;
  std::ostringstream os;
  os << ea << " == " << eb << "\n      got: " << a << "\n expected: " << b;
  fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, os.str().c_str());
  ++failures();
}

}  // namespace check

#define CHECK_CAT2(a, b) a##b
#define CHECK_CAT(a, b) CHECK_CAT2(a, b)
#define TEST(name)                                                                \
  static void name();                                                             \
  static check::Register CHECK_CAT(reg_, name)(#name, name);                      \
  static void name()

#define CHECK(expr)                                                               \
  do {                                                                            \
    if (!(expr)) {                                                                \
      fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);           \
      ++check::failures();                                                        \
    }                                                                             \
  } while (0)
#define CHECK_EQ(a, b) check::eq((a), (b), #a, #b, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(double(a) - double(b)) <= (tol))
