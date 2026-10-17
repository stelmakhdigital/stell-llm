// Минимальный харнесс для тестов core (без внешних зависимостей).
#pragma once

#include <cstdio>
#include <cstdlib>

#define STELL_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__,    \
                         #cond);                                              \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define STELL_CHECK_MSG(cond, msg)                                            \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "[FAIL] %s:%d: %s (%s)\n", __FILE__,         \
                         __LINE__, #cond, (msg));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define STELL_TEST_MAIN(fn)                                                   \
    int main() { fn(); std::puts("[PASS]"); return 0; }
