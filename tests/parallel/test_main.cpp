#include "test_harness.hpp"
#include <cstring>

int main(int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : nullptr;
    testfw::run_all("test_parallel", filter);
    return testfw::failures() == 0 ? 0 : 1;
}
