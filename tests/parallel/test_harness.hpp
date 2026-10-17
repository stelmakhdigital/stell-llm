// tests/parallel: минимальный тестовый харнесс (без внешних зависимостей —
// локальная CPU-конфигурация собирается без сторонних библиотек, ADR-07).
#pragma once

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <atomic>

namespace testfw {

struct Case {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> v;
    return v;
}
inline int& failures() {
    static int n = 0;
    return n;
}
inline int& checks() {
    static int n = 0;
    return n;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline void run_all(const char* suite, const char* filter = nullptr) {
    std::printf("[%s]%s\n", suite, filter ? " (фильтр)" : "");
    for (const auto& c : registry()) {
        if (filter && std::strstr(c.name, filter) == nullptr) continue;
        const int before = failures();
        try {
            c.fn();
        } catch (const std::exception& e) {
            std::printf("  FAIL %-40s exception: %s\n", c.name, e.what());
            ++failures();
        }
        if (failures() == before)
            std::printf("  OK   %s\n", c.name);
        else
            std::printf("  FAIL %s\n", c.name);
    }
    std::printf("  checks=%d failures=%d\n", checks(), failures());
}

} // namespace testfw

// Потокобезопасный флаг для многопоточных тестов (CHECK в потоках не
// используется — счётчики харнесса не атомарные).
struct SafeFlag {
    std::atomic<bool> v{true};
    void fail() { v.store(false); }
    bool get() const { return v.load(); }
};

#define TEST(name)                                                               \
    static void test_##name##_impl();                                            \
    static testfw::Registrar reg_##name{#name, &test_##name##_impl};             \
    static void test_##name##_impl()

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++testfw::checks();                                                      \
        if (!(cond)) {                                                           \
            std::printf("    CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++testfw::failures();                                                \
        }                                                                        \
    } while (0)

#define CHECK_THROWS_AS(expr, exc_type)                                          \
    do {                                                                         \
        ++testfw::checks();                                                      \
        bool threw = false;                                                      \
        try { expr; }                                                            \
        catch (const exc_type&) { threw = true; }                                \
        catch (...) {}                                                           \
        if (!threw) {                                                            \
            std::printf("    expected %s: %s (%s:%d)\n", #exc_type, #expr, __FILE__, __LINE__); \
            ++testfw::failures();                                                \
        }                                                                        \
    } while (0)
