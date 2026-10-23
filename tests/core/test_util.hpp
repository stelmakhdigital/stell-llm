// Минимальный харнесс для тестов core (без внешних зависимостей).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace stell {
namespace test {
// %s varargs требует char*: у макроса msg бывает и литерал, и склеенный
// std::string — UB при передаче std::string в varargs (на части тулчейнов —
// ошибка сборки). Конверсия — через перегрузку, без typeid-игр.
inline const char* msg_c_str(const std::string& s) { return s.c_str(); }
inline const char* msg_c_str(const char* s) { return s; }
} // namespace test
} // namespace stell

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
                         __LINE__, #cond, stell::test::msg_c_str(msg));       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define STELL_TEST_MAIN(fn)                                                   \
    int main() { fn(); std::puts("[PASS]"); return 0; }
