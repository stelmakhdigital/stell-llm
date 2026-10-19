// tests/parallel: GPU-тест NCCL-backend (ADR-02/ADR-10 §2) — задача 3 раунда 2,
// ретрай-протокол P2P — [R3] раунда 3 (planner round 2 [Q1]).
//
// Собирается ТОЛЬКО при STELL_CUDA=ON + libnccl (сервер, cuda 12.8 + libnccl).
//
// Протокол [R3] (docs/design/server-env.md, src/parallel/include/stell/parallel/p2p_retry.hpp):
//   КОРРЕКТНОСТЬ — пара GPU0 (V100) + GPU4 (RTX 2060) по умолчанию
//   (STELL_GPU_TEST_DEVICES "0,4"; переопределяется env STELL_GPU_DEVICES,
//   напр. "0,1,2,3"): побитово == CPU-эталону (эмуляторные golden не трогаются).
//   При неудаче/зависании попытки 1 (зависание P2P-коллективов при
//   ncclCommInitRank COMPLETE — типичный признак вмешательства co-tenancy
//   в direct-pointer P2P) — АВТОМАТИЧЕСКАЯ попытка 2 с NCCL_P2P_DISABLE=1:
//   корректность обязана подтверждаться и без P2P (ring через host/shm).
//   Ретрай исполняется самим тестом: бинарник перезапускает себя дочерним
//   процессом под watchdog'ом (зависание NCCL-коллектива не перехватывается
//   внутри процесса). Третьей попытки нет. Логика решений — чистый заголовок
//   p2p_retry.hpp, юнит-тест на CPU: tests/parallel/test_p2p_retry.cpp.
//   Ручной вариант для worker'а (если ретрай подавлен или нужен отдельный шаг):
//     NCCL_P2P_DISABLE=1 STELL_GPU_DEVICES=0,4 ctest -R parallel_gpu --output-on-failure
//   LATENCY-БЕНЧ (V100↔V100 по NVLink, для целей PLAN.md §5) — отдельная фаза,
//   ТОЛЬКО при наличии окна (после подтверждения корректности; пропуск —
//   env STELL_GPU_BENCH=0), с co-tenancy-пометкой; замеры на RTX 2060
//   с целями §5 не сравниваются (server-env: «не пригодна для §5»).
//
// Честный PENDING (checks=0 failures=0) при недоступности карт СОХРАНЁН:
// недоступность/OOM/сбой бутстрапа → PENDING, тест зелёный, проверок нет.
// Честный FAIL: если прогон дошёл до проверок, а корректность НЕ подтверждена
// (битовое равенство нарушено) — тест красный (маскировать pending'ом
// запрещено: «корректность обязана подтверждаться»).
//
// Запуск: ctest -R parallel_gpu  (или бинарник напрямую).
#include "test_harness.hpp"
#include "stell/parallel/comm.hpp"
#include "stell/parallel/nccl_communicator.hpp"
#include "stell/parallel/p2p_retry.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;   // для execve: окружение дочернего процесса

using namespace stell::parallel;

namespace {

#ifndef STELL_GPU_TEST_DEVICES
#define STELL_GPU_TEST_DEVICES "0,4"   // протокол [Q1]: GPU0 (V100) + GPU4 (RTX 2060)
#endif

std::vector<int> parse_devices(const char* s) {
    std::vector<int> out;
    int v = 0;
    bool any = false;
    for (const char* p = s; *p; ++p) {
        if (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); any = true; }
        else if (any) { out.push_back(v); v = 0; any = false; }
    }
    if (any) out.push_back(v);
    return out;
}

void check_cuda(cudaError_t r, const char* what) {
    if (r != cudaSuccess) {
        std::printf("CUDA-FAIL: %s: %s\n", what, cudaGetErrorString(r));
        throw CommError(CommError::Code::NcclFailure, what);
    }
}

// Исход фазы для родителя-оркестратора (значение файла результата дочернего
// процесса; см. p2p_retry.hpp). Перезаписывается тестами.
std::string g_phase_outcome = "pending:тесты не выполнены";

// Была ли в фазе исключительная ситуация (отличать от битового несовпадения):
// true (CommError/прочее) → честный PENDING; false при !ok — прогон дошёл
// до проверок, корректность не подтверждена → честный FAIL.
std::atomic<bool> g_threw{false};

// Честная co-tenancy: занятость каждой карты перед прогоном.
// Не бросает: недоступность карты — честный pending (см. ниже), не FAIL.
void report_cotenance(const std::vector<int>& devices) {
    int prev = -1;
    cudaGetDevice(&prev);
    for (int d : devices) {
        if (cudaSetDevice(d) != cudaSuccess) {
            std::printf("COTENANCE: GPU %d: cudaSetDevice не удался — карта недоступна, "
                        "честный pending\n", d);
            g_phase_outcome = "pending:карта недоступна (cudaSetDevice)";
            continue;
        }
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
            std::printf("COTENANCE: GPU %d: свободно %.2f GB / %.2f GB "
                        "(чужой процесс strata держит остальное)\n", d,
                        double(free_b) / 1e9, double(total_b) / 1e9);
        else
            std::printf("COTENANCE: GPU %d: cudaMemGetInfo не удался — честный pending\n", d);
    }
    cudaSetDevice(prev);
}

// Один ранок: создаёт комм, исполняет коллективы, пишет результаты.
struct RankOut {
    bool ok = true;
    std::vector<double> latency_us;      // по размерам
};

void rank_body(const ParallelConfig& cfg, const std::vector<std::size_t>& sizes,
               int iters_per_size, RankOut& out) {
    auto comm = make_communicator(cfg);
    out.latency_us.assign(sizes.size(), 0.0);
    const rank_t world = comm->world_size();
    const std::size_t maxn = *std::max_element(sizes.begin(), sizes.end());
    // NCCL работает над DEVICE-буферами: выделяем в свободной VRAM (co-tenancy).
    float* dev_buf = nullptr;
    float* dev_recv = nullptr;
    std::vector<float> host(maxn), recv(maxn * std::size_t(world));
    check_cuda(cudaMalloc(&dev_buf, maxn * 4), "cudaMalloc(allreduce buf)");
    check_cuda(cudaMalloc(&dev_recv, maxn * std::size_t(world) * 4),
               "cudaMalloc(allgather recv)");
    auto cleanup = [&] {
        cudaFree(dev_buf);
        cudaFree(dev_recv);
    };
    try {
        // allgather: корректность (конкатенация send-паттернов по ранкам).
        {
            for (std::size_t i = 0; i < maxn; ++i)
                host[i] = 100.0f * float(cfg.rank) + float(i);
            check_cuda(cudaMemcpy(dev_buf, host.data(), maxn * 4, cudaMemcpyHostToDevice),
                       "H2D allgather");
            comm->allgather(dev_buf, dev_recv, maxn, CommDType::F32);
            check_cuda(cudaMemcpy(recv.data(), dev_recv,
                                  maxn * std::size_t(world) * 4, cudaMemcpyDeviceToHost),
                       "D2H allgather");
            for (rank_t s = 0; s < world; ++s)
                for (std::size_t i = 0; i < maxn; ++i)
                    if (!(recv[std::size_t(s) * maxn + i] == 100.0f * float(s) + float(i)))
                        out.ok = false;
        }
        // allreduce: корректность + latency (медиана по итерациям).
        for (std::size_t si = 0; si < sizes.size(); ++si) {
            const std::size_t n = sizes[si];
            for (std::size_t i = 0; i < n; ++i) host[i] = 1000.0f * float(cfg.rank) + float(i);
            check_cuda(cudaMemcpy(dev_buf, host.data(), n * 4, cudaMemcpyHostToDevice),
                       "H2D allreduce");
            comm->allreduce(dev_buf, n, CommDType::F32);   // warmup + корректность
            check_cuda(cudaMemcpy(host.data(), dev_buf, n * 4, cudaMemcpyDeviceToHost),
                       "D2H allreduce");
            for (std::size_t i = 0; i < n; ++i) {
                float expect = 0.f;
                for (rank_t s = 0; s < world; ++s) expect += 1000.0f * float(s) + float(i);
                if (!(host[i] == expect)) out.ok = false;   // суммы целые < 2^22 — точны
            }
            std::vector<double> samples;
            samples.reserve(std::size_t(iters_per_size));
            for (int it = 0; it < iters_per_size; ++it) {
                const auto t0 = std::chrono::steady_clock::now();
                comm->allreduce(dev_buf, n, CommDType::F32);
                const auto t1 = std::chrono::steady_clock::now();
                samples.push_back(
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(samples.begin(), samples.end());
            out.latency_us[si] = samples[samples.size() / 2];
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}
} // namespace

TEST(parallel_gpu_nccl_allreduce_allgather_latency) {
    // Список устройств: env STELL_GPU_DEVICES (напр. "0,1,2,3") перекрывает
    // дефолт STELL_GPU_TEST_DEVICES. Фаза bench [R3]: только при наличии окна
    // (запускается оркестратором после подтверждения корректности или в
    // TP1-режиме); замеры co-tenancy помечены и с целями §5 не сравниваются.
    const char* env = std::getenv("STELL_GPU_DEVICES");
    const std::vector<int> devices =
        parse_devices(env ? env : STELL_GPU_TEST_DEVICES);
    std::printf("NCCL: версия %d.%d.%d, устройств в прогоне: %zu (%s)\n",
                NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, devices.size(), STELL_GPU_TEST_DEVICES);
    report_cotenance(devices);

    const std::vector<std::size_t> sizes = {1, 64, 256, 1024, 4096, 16384, 65536, 262144};
    const int iters = 100;

    ParallelConfig cfg;
    cfg.mode = CommMode::Nccl;
    cfg.pp = 1;
    cfg.tp = rank_t(devices.size());
    cfg.ep = 1;
    cfg.device_ids = devices;
    cfg.timeout_ms = 30000;

    SafeFlag ok;
    std::vector<RankOut> outs(devices.size());
    std::vector<std::thread> th;
    for (rank_t r = 0; r < rank_t(devices.size()); ++r)
        th.emplace_back([&, r] {
            ParallelConfig rc = cfg;
            rc.rank = r;
            try {
                rank_body(rc, sizes, iters, outs[std::size_t(r)]);
            } catch (const CommError& e) {
                std::printf("PENDING: CommError на ранке %d: %s\n", int(r), e.what());
                g_threw.store(true);
                ok.fail();
            } catch (...) {
                std::printf("PENDING: неизвестная ошибка на ранке %d\n", int(r));
                g_threw.store(true);
                ok.fail();
            }
        });
    for (auto& t : th) t.join();

    if (!ok.get()) {
        if (g_threw.load())
            g_phase_outcome = "pending:NCCL-прогон не удался (co-tenancy/OOM/бутстрап)";
        else
            g_phase_outcome = "fail:корректность allreduce/allgather не подтверждена";
        std::printf("PENDING: NCCL-прогон не удался (co-tenancy/OOM/бутстрап) — "
                    "см. вывод выше; GPU-результат не выдаётся за проверку\n");
        return;
    }
    std::printf("NCCL-BENCH allreduce F32, медиана µs по %d итерациям "
                "(co-tenancy: замер под нагрузкой strata, с целями §5 НЕ сравнивать "
                "если среди карт есть RTX 2060 / нет NVLink):\n", iters);
    for (std::size_t si = 0; si < sizes.size(); ++si) {
        std::printf("NCCL-BENCH: size=%6zu floats (%6zu KB):", sizes[si],
                    sizes[si] * 4 / 1024);
        for (std::size_t r = 0; r < outs.size(); ++r)
            std::printf(" rank%zu=%.1fµs", std::size_t(r), outs[r].latency_us[si]);
        std::printf("\n");
    }
    g_phase_outcome = kOutcomePass;
    CHECK(ok.get());
}

// --- [R6] GPU-контракт expert_alltoall (token_elems) + kv_stream -------------

// CPU-эмуляторный контракт этих операций прогнан в tests/parallel (tok_width_*,
// sub_*); здесь — NCCL-исполнение на паре [Q1] (GPU0 V100 + GPU4 RTX 2060
// по умолчанию) с ретраем [R3] при зависании P2P (см. шапку файла).
TEST(parallel_gpu_nccl_expert_alltoall_kv_stream) {
    const char* env = std::getenv("STELL_GPU_DEVICES");
    const std::vector<int> devices = parse_devices(env ? env : STELL_GPU_TEST_DEVICES);
    if (devices.size() < 2) {
        std::printf("PENDING: нужно ≥2 устройств (STELL_GPU_DEVICES)\n");
        g_phase_outcome = kOutcomeFewDevices;
        return;
    }
    constexpr int DM = 8, T = 4;                     // токен = d_model векторов
    SafeFlag ok;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < rank_t(devices.size()); ++r)
        th.emplace_back([&, r] {
            ParallelConfig cfg;
            cfg.mode = CommMode::Nccl;
            cfg.pp = 1;
            cfg.tp = 1;
            cfg.ep = rank_t(devices.size());         // EP: владелец = expert_id % ep
            cfg.device_ids = devices;
            cfg.rank = r;
            cfg.group_id = 1;                        // отдельный бутстрап от bench-теста
            cfg.timeout_ms = 30000;
            try {
                auto comm = make_communicator(cfg);
                check_cuda(cudaSetDevice(devices[std::size_t(r)]), "setdev");
                // --- expert_alltoall: токен = вектор d_model ---
                float* send_t = nullptr;
                TokenMeta* send_m = nullptr;
                const std::size_t cap = std::size_t(T) * std::size_t(devices.size());
                float* recv_t = nullptr;
                TokenMeta* recv_m = nullptr;
                std::vector<float> hsend(std::size_t(T) * DM);
                std::vector<TokenMeta> hsm(T);
                for (int i = 0; i < T; ++i) {
                    hsm[std::size_t(i)] = TokenMeta{std::int32_t((r + 1) % int(cfg.ep)),
                                                    std::int32_t(r * 16 + i)};
                    for (int dd = 0; dd < DM; ++dd)
                        hsend[std::size_t(i) * DM + dd] =
                            float(r) * 1000.f + float(i) * 10.f + float(dd);
                }
                check_cuda(cudaMalloc((void**)&send_t, hsend.size() * sizeof(float)), "malloc");
                check_cuda(cudaMalloc((void**)&send_m, hsm.size() * sizeof(TokenMeta)), "malloc");
                check_cuda(cudaMalloc((void**)&recv_t, cap * DM * sizeof(float)), "malloc");
                check_cuda(cudaMalloc((void**)&recv_m, cap * sizeof(TokenMeta)), "malloc");
                check_cuda(cudaMemcpy(send_t, hsend.data(), hsend.size() * sizeof(float),
                                      cudaMemcpyHostToDevice), "H2D");
                check_cuda(cudaMemcpy(send_m, hsm.data(), hsm.size() * sizeof(TokenMeta),
                                      cudaMemcpyHostToDevice), "H2D");
                const auto t0 = std::chrono::steady_clock::now();
                comm->expert_alltoall(send_t, send_m, recv_t, recv_m, std::size_t(T),
                                      std::size_t(DM), CommDType::F32);
                const double us =
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count();
                std::vector<float> hrecv(cap * DM);
                std::vector<TokenMeta> hrm(cap);
                check_cuda(cudaMemcpy(hrecv.data(), recv_t, hrecv.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost), "D2H");
                check_cuda(cudaMemcpy(hrm.data(), recv_m, hrm.size() * sizeof(TokenMeta),
                                      cudaMemcpyDeviceToHost), "D2H");
                // С одним expert_id=(src+1)%ep все токены ранка src уезжают
                // на следующий ранок: входящих = T (от предыдущего ранка);
                // recv_meta.expert_id = владелец (= мой ранок); хвост — заполнители.
                const rank_t src_expect = rank_t((int(r) + int(cfg.ep) - 1) % int(cfg.ep));
                std::size_t got = 0;
                for (std::size_t i = 0; i < cap; ++i) {
                    if (hrm[i].expert_id < 0) continue;
                    ++got;
                    const rank_t src = rank_t(hrm[i].slot_id) / 16;
                    const int k = int(hrm[i].slot_id % 16);
                    if (!(src == src_expect)) ok.fail();
                    if (!(hrm[i].expert_id == std::int32_t(r))) ok.fail();
                    for (int dd = 0; dd < DM; ++dd)
                        if (!(hrecv[i * DM + dd] == float(src) * 1000.f + float(k) * 10.f +
                              float(dd)))
                            ok.fail();
                }
                if (got != std::size_t(T)) ok.fail();
                std::printf("NCCL alltoall: rank%d: входящих=%zu, %.1f µs\n", int(r), got, us);
                cudaFree(send_t); cudaFree(send_m); cudaFree(recv_t); cudaFree(recv_m);

                // --- kv_stream: 3 блока r0→r1 по кольцу, побайтово ---
                if (r == 0) {
                    for (int b = 0; b < 3; ++b) {
                        std::vector<std::uint8_t> blk(64);
                        for (std::size_t i = 0; i < blk.size(); ++i)
                            blk[i] = std::uint8_t(b * 31 + i);
                        comm->kv_stream_push(blk.data(), blk.size());
                    }
                } else {
                    for (int b = 0; b < 3; ++b) {
                        std::vector<std::uint8_t> blk(64, 0);
                        bool got_blk = false;
                        for (int spin = 0; spin < 5000 && !got_blk; ++spin)
                            got_blk = comm->kv_stream_pop(blk.data(), blk.size());
                        if (!got_blk) { ok.fail(); continue; }
                        for (std::size_t i = 0; i < blk.size(); ++i)
                            if (blk[i] != std::uint8_t(b * 31 + i)) ok.fail();
                    }
                    std::vector<std::uint8_t> empty(64);
                    if (comm->kv_stream_pop(empty.data(), empty.size())) ok.fail();
                }
            } catch (const CommError& e) {
                std::printf("PENDING: CommError (rank %d): %s\n", int(r), e.what());
                g_threw.store(true);
                ok.fail();
            } catch (...) {
                std::printf("PENDING: неизвестная ошибка (rank %d)\n", int(r));
                g_threw.store(true);
                ok.fail();
            }
        });
    for (auto& t : th) t.join();
    if (!ok.get()) {
        if (g_threw.load())
            g_phase_outcome = "pending:GPU-контракт expert_alltoall/kv_stream: "
                              "co-tenancy/OOM/бутстрап (проверки не проведены)";
        else
            g_phase_outcome = "fail:корректность expert_alltoall/kv_stream не подтверждена";
        std::printf("PENDING: GPU-контракт expert_alltoall/kv_stream не подтверждён — "
                    "см. вывод выше\n");
        return;
    }
    g_phase_outcome = kOutcomePass;
    CHECK(ok.get());
}

// --- оркестратор [R3]: дочерние процессы под watchdog'ом ----------------------

namespace {

constexpr int kDefaultWatchdogS = 240;   // зависание NCCL-коллектива — только извне

const char* phase_filter(const char* phase) {
    // bench-фаза — только латентности; correctness — контракт alltoall/kv_stream.
    return std::strcmp(phase, "bench") == 0 ? "parallel_gpu_nccl_allreduce"
                                            : "parallel_gpu_nccl_expert_alltoall";
}

int run_child_phase(const char* phase, const char* out_file,
                    bool disable_p2p, int watchdog_s, const char* tag) {
    const pid_t pid = fork();
    if (pid < 0) {
        std::printf("PENDING: fork не удался (%s) — честный pending\n", tag);
        return -1;
    }
    if (pid == 0) {
        // Дочерний процесс: свежий образ (CUDA/NCCL-состояние родителя не
        // наследуется — exec), окружение + метки фазы и файла результата.
        std::vector<std::string> env_store;
        for (char** e = environ; e && *e; ++e) {
            std::string kv(*e);
            // NCCL_P2P_DISABLE попытки 2 — заменить унаследованный.
            if (disable_p2p && kv.rfind("NCCL_P2P_DISABLE=", 0) == 0) continue;
            env_store.push_back(std::move(kv));
        }
        env_store.push_back(std::string("STELL_GPU_TEST_PHASE=") + phase);
        env_store.push_back(std::string("STELL_GPU_TEST_OUT=") + out_file);
        if (disable_p2p) env_store.push_back("NCCL_P2P_DISABLE=1");
        std::vector<char*> envp;
        for (auto& kv : env_store) envp.push_back(kv.data());
        envp.push_back(nullptr);
        char arg0[] = "/proc/self/exe";
        char* argv[] = {arg0, nullptr};
        execve("/proc/self/exe", argv, envp.data());
        _exit(127);   // exec не удался
    }
    // Watchdog: зависший NCCL-коллектив завершается только извне (SIGKILL).
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(watchdog_s);
    int status = 0;
    bool timed_out = false;
    for (;;) {
        const pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) break;
        if (std::chrono::steady_clock::now() > deadline) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            timed_out = true;
            break;
        }
        usleep(200 * 1000);
    }
    if (timed_out) {
        std::printf("WATCHDOG: попытка %s убита по тайм-ауту %dс (NCCL-коллектив завис?)\n",
                    tag, watchdog_s);
        return -2;   // зависание
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -3;
}

std::string read_outcome(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) return "";   // крах/убитый процесс — пустой исход (ретрай-класс)
    std::string s;
    char buf[256];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

int orchestrate() {
    const char* env_devices = std::getenv("STELL_GPU_DEVICES");
    const std::vector<int> devices =
        parse_devices(env_devices ? env_devices : STELL_GPU_TEST_DEVICES);
    std::string devices_str = "[";
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (i) devices_str += ',';
        devices_str += std::to_string(devices[i]);
    }
    devices_str += ']';
    int watchdog_s = kDefaultWatchdogS;
    if (const char* w = std::getenv("STELL_GPU_TEST_TIMEOUT_S")) watchdog_s = atoi(w);

    char out_path[] = "/tmp/stell_gpu_test_out.XXXXXX";
    const int out_fd = mkstemp(out_path);
    if (out_fd < 0) {
        std::printf("PENDING: mkstemp не удался — честный pending\n");
        return 0;
    }
    close(out_fd);
    const std::string out_file(out_path);

    // --- Фаза 1: корректность (до 2 попыток, вторая — с NCCL_P2P_DISABLE=1) --
    std::string outcome;
    int attempt = 1;
    for (;;) {
        if (attempt == 1)
            std::printf("GPU-VALIDATION: фаза корректности, попытка 1 (карты %s: "
                        "протокол [Q1] — пара GPU0 V100 + GPU4 RTX 2060; "
                        "переопределить: STELL_GPU_DEVICES)\n",
                        devices_str.c_str());
        else
            std::printf("GPU-VALIDATION: попытка 2 с NCCL_P2P_DISABLE=1 "
                        "(ring через host/shm — корректность обязана подтверждаться "
                        "и без P2P)\n");
        std::fflush(stdout);
        std::remove(out_file.c_str());   // исход предыдущей попытки не выдавать за текущую
        const int rc = run_child_phase("correctness", out_file.c_str(),
                                       std::string(p2p_disable_env(attempt)) == "1",
                                       watchdog_s, "корректности");
        if (rc == -1) {   // fork не удался — повтор бесполезен, честный pending
            outcome = "pending:fork не удался";
            break;
        }
        outcome = read_outcome(out_file.c_str());   // rc == -2 → файл пуст (зависание)
        // Решение протокола: pass → стоп; неудача попытки 1 → ретрай без P2P;
        // неудача попытки 2 (или зависание) → GiveUp.
        const GpuRetryVerdict v = p2p_retry_decide(attempt, outcome);
        if (v == GpuRetryVerdict::Pass) break;
        if (v == GpuRetryVerdict::RetryWithP2pDisabled) { attempt = 2; continue; }
        break;   // GiveUpPending: попытки исчерпаны
    }

    // --- Вердикт фазы корректности -----------------------------------------
    if (outcome_is_fail(outcome)) {
        std::printf("GPU-VALIDATION FAIL: корректность NCCL-контракта НЕ подтверждена "
                    "(побитовое равенство CPU-эталону нарушено) — см. вывод попыток выше\n");
        std::remove(out_file.c_str());
        return 1;   // честный FAIL: маскировать pending'ом запрещено
    }
    if (!bench_allowed_after(outcome)) {
        std::printf("GPU-VALIDATION PENDING: %s "
                    "(checks=0 failures=0 — проверки корректности не проводились; "
                    "worker повторит в окне; co-tenancy: см. server-env)\n",
                    outcome.c_str());
        std::remove(out_file.c_str());
        return 0;
    }
    if (outcome != kOutcomePass)
        std::printf("GPU-VALIDATION: корректность неприменима (меньше 2 карт) — "
                    "переходим к bench-фазе (TP1-режим scaling-бенчмарка [R8])\n");
    else if (attempt == 2)
        std::printf("GPU-VALIDATION: корректность подтверждена БЕЗ P2P "
                    "(NCCL_P2P_DISABLE=1, ring через host/shm) — co-tenancy "
                    "вмешалась в direct-pointer P2P; пометить в отчёте\n");
    else
        std::printf("GPU-VALIDATION: корректность подтверждена (побитово == CPU-эталону)\n");

    // --- Фаза 2: latency-бенч — только при наличии окна ([R3]б) ------------
    if (const char* b = std::getenv("STELL_GPU_BENCH"); b && std::strcmp(b, "0") == 0) {
        std::printf("GPU-VALIDATION: bench-фаза пропущена (STELL_GPU_BENCH=0) — "
                    "latency-замер только в чистом окне V100↔V100 по NVLink, "
                    "co-tenancy-пометка обязательна\n");
        std::remove(out_file.c_str());
        return 0;
    }
    std::fflush(stdout);
    std::remove(out_file.c_str());
    const int rc = run_child_phase("bench", out_file.c_str(), /*disable_p2p=*/false,
                                   watchdog_s, "bench");
    const std::string bench_outcome =
        (rc == -2) ? "pending:bench-прогон завис (убит watchdog'ом)" : read_outcome(out_file.c_str());
    if (rc == -2) {
        std::printf("GPU-VALIDATION: latency-бенч не проведён (зависание) — "
                    "co-tenancy: замер отложен до чистого окна V100↔V100; "
                    "корректность выше подтверждена\n");
    } else if (bench_outcome.empty()) {
        std::printf("GPU-VALIDATION: latency-бенч не проведён (дочерний процесс "
                    "завершился аварийно, exit=%d) — замер отложен до чистого окна "
                    "V100↔V100; корректность выше подтверждена\n", rc);
    } else if (!outcome_is_pass(bench_outcome)) {
        std::printf("GPU-VALIDATION: latency-бенч не проведён (%s) — "
                    "co-tenancy: замер отложен до чистого окна V100↔V100; "
                    "корректность выше подтверждена\n", bench_outcome.c_str());
    } else {
        std::printf("GPU-VALIDATION OK: корректность + латентности (см. NCCL-BENCH выше; "
                    "co-tenancy-пометка обязательна)\n");
    }
    std::remove(out_file.c_str());
    return 0;
}

} // namespace

int main() {
    // Дочерний режим: исполнить одну фазу и записать исход в файл результата.
    if (const char* phase = std::getenv("STELL_GPU_TEST_PHASE")) {
        testfw::run_all("test_parallel_gpu", phase_filter(phase));
        const char* out = std::getenv("STELL_GPU_TEST_OUT");
        if (out) {
            FILE* f = std::fopen(out, "w");
            if (f) {
                std::fwrite(g_phase_outcome.data(), 1, g_phase_outcome.size(), f);
                std::fclose(f);
            }
        }
        // pending (checks=0 failures=0) → 0; fail (CHECK не прошёл) → 1.
        return testfw::failures() == 0 ? 0 : 1;
    }
    return orchestrate();
}
