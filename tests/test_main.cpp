#include "check.h"

int main() {
  int failed_cases = 0;
  for (const auto& c : check::registry()) {
    const int before = check::failures();
    c.fn();
    const bool ok = check::failures() == before;
    failed_cases += !ok;
    printf("%s %s\n", ok ? "[ OK ]" : "[FAIL]", c.name);
  }
  printf("%zu tests, %d failed\n", check::registry().size(), failed_cases);
  return failed_cases ? 1 : 0;
}
