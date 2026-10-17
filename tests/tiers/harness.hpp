// Минимальный тестовый харнесс для tests/tiers (без внешних зависимостей).
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <stell/tiers/types.hpp>

namespace ttest {

inline std::vector<std::pair<std::string, std::function<void()>>>& items() {
    static std::vector<std::pair<std::string, std::function<void()>>> v;
    return v;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        items().emplace_back(name, std::move(fn));
    }
};

inline int run_all() {
    int failed = 0;
    for (const auto& kv : items()) {
        try {
            kv.second();
            std::printf("[ OK ] %s\n", kv.first.c_str());
        } catch (const std::exception& e) {
            ++failed;
            std::printf("[FAIL] %s: %s\n", kv.first.c_str(), e.what());
        } catch (...) {
            ++failed;
            std::printf("[FAIL] %s: unknown exception\n", kv.first.c_str());
        }
    }
    std::printf("итог: passed=%zu failed=%d\n", items().size() - (unsigned)failed, failed);
    return failed == 0 ? 0 : 1;
}

} // namespace ttest

#define T_CASE(name)                                                          \
    static void name();                                                       \
    static ttest::Registrar r_##name(#name, &name);                           \
    static void name()

#define T_REQUIRE(cond, msg)                                                  \
    do {                                                                      \
        if (!(cond))                                                          \
            throw std::runtime_error(std::string(__FILE__) + ":" +            \
                                     std::to_string(__LINE__) + " " + (msg)); \
    } while (0)

#define T_MAIN int main() { setvbuf(stdout, nullptr, _IONBF, 0); return ttest::run_all(); }

// --- общие фикстуры tiers ----------------------------------------------------
namespace fixture {

inline stell::tiers::TensorGroup make_group(std::uint32_t id, const std::string& name,
                                            stell::tiers::Role role, stell::tiers::DType dtype,
                                            std::size_t slot_bytes,
                                            std::vector<std::int64_t> shape = {1}) {
    stell::tiers::TensorGroup g;
    g.id = stell::tiers::GroupId{id};
    g.name = name;
    g.role = role;
    g.dtype = dtype;
    g.shape = std::move(shape);
    g.slot_bytes = slot_bytes;
    return g;
}

// Ожидание resident-ности группы (для детерминизма prefetch-тестов).
template <class Mgr>
void wait_resident(Mgr& mgr, stell::tiers::GroupId id, int timeout_ms = 5000) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    while (!mgr.resident_in_vram(id)) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0)
                .count() > timeout_ms)
            throw std::runtime_error("wait_resident: timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace fixture

using fixture::make_group;
using fixture::wait_resident;
