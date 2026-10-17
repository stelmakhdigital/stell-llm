// tests/parallel: точка входа тестового харнесса.
#include "test_harness.hpp"

int main() {
    testfw::run_all("test_parallel");
    return testfw::failures() == 0 ? 0 : 1;
}
