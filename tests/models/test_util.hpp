// Минимальный каркас тестов модуля models (без внешних зависимостей):
// CHECK_* макросы + run() с единым перехватом и кодом возврата.
#pragma once

#include <cstdio>
#include <string>
#include <exception>
#include <functional>
#include <string>

namespace stell_test {

struct TestFailure : std::exception {
    std::string msg;
    std::string where;
    TestFailure(std::string m, std::string w) : msg(std::move(m)), where(std::move(w)) {}
};

inline void check(bool cond, const std::string& what, const std::string& where) {
    if (!cond) throw TestFailure(what, where);
}

inline int run(const std::string& name, const std::function<void()>& body) {
    try {
        body();
        std::printf("[ OK  ] %s\n", name.c_str());
        return 0;
    } catch (const TestFailure& e) {
        std::printf("[FAIL ] %s: %s (%s)\n", name.c_str(), e.msg.c_str(), e.where.c_str());
    } catch (const std::exception& e) {
        std::printf("[FAIL ] %s: unexpected exception: %s\n", name.c_str(), e.what());
    }
    return 1;
}

}  // namespace stell_test

#define CHECK(cond) \
    stell_test::check(!!(cond), #cond, std::string(__FILE__) + ":" + std::to_string(__LINE__))

#define CHECK_MSG(cond, msg) \
    stell_test::check(!!(cond), (msg), std::string(__FILE__) + ":" + std::to_string(__LINE__))

#define CHECK_THROW(stmt, Ex, msg)                                                  \
    do {                                                                            \
        bool thrown = false;                                                        \
        const std::string _where = std::string(__FILE__) + ":" + std::to_string(__LINE__); \
        try {                                                                       \
            stmt;                                                                   \
        } catch (const Ex&) {                                                       \
            thrown = true;                                                          \
        } catch (const std::exception& e) {                                         \
            throw stell_test::TestFailure(                                          \
                std::string("wrong exception type: ") + e.what(), _where);          \
        }                                                                           \
        stell_test::check(thrown, msg, _where);                                     \
    } while (0)

#define TEST_MAIN(name)                                       \
    int main() {                                              \
        return stell_test::run(name, [] { body(); });         \
    }
