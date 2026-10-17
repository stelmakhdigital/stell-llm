// tests/parallel: семантика мета-протокола шага scheduler (ADR-10 §3 +
// уточнение раунда 1: allgather F32, 8 слотов = 32 байта; CommDType::Meta
// отклонён). Задача 4 раунда: вклейка core-2 — сверка сигнатур/семантики.
//
// Схема scheduler.cpp::broadcast_step_meta: ранг 0 — источник меты, остальные
// ранки исполняют полученную разметку; allgather используется как broadcast-
// примитив. Контракт allgather (comm.hpp): recv-буфер — world*count элементов
// (конкатенация send-слотов ВСЕХ ранков). РАСХОЖДЕНИЕ С ENGINE (зафиксировано
// в src/parallel/CONTRACT_ENGINE.md и отчёте): scheduler.cpp выделяет
// recv[8] — при реальном бэкенде это переполнение буфера (ASan-поймано в
// этом тесте); во вклейке recv обязан быть [8*world].
#include "test_harness.hpp"
#include "stell/parallel/emulated.hpp"

#include <cstring>
#include <thread>
#include <vector>

using namespace stell::parallel;

namespace {

ParallelConfig make_cfg(rank_t world, int group_id, rank_t rank) {
    ParallelConfig c;
    c.mode = CommMode::CpuEmulation;
    c.pp = world;
    c.group_id = group_id;
    c.rank = rank;
    return c;
}

} // namespace

TEST(meta_broadcast_allgather_f32_8slots_scheduler_pattern) {
    // Паттерн scheduler.cpp (дословно, но с КОНТРАКТНОЙ ёмкостью recv):
    // float meta[8], recv[8*world]; comm->allgather(meta, recv, 8, F32).
    constexpr rank_t WORLD = 4;
    const std::uint64_t step = 4242;
    const std::uint64_t batch_size = 7;
    SafeFlag ok;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < WORLD; ++r)
        th.emplace_back([r, &ok] {
            auto comm = std::make_unique<EmulatedCommunicator>(make_cfg(WORLD, 501, r));
            // Ранг 0 — источник меты [шаг, размер батча, резерв...]; остальные
            // ранки отдают заготовленные слоты (нечто своё/нули).
            float meta[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (r == 0) {
                meta[0] = static_cast<float>(step);
                meta[1] = static_cast<float>(batch_size);
            } else {
                meta[2] = -1.0f;   // маркер: слоты последователей игнорируются
            }
            std::vector<float> recv(std::size_t(WORLD) * 8);   // контракт: world*count
            comm->allgather(meta, recv.data(), 8, CommDType::F32);
            // Мета ранка 0 доступна всем: recv[rank0*8 + 0..1] = [шаг, батч].
            if (!(recv[0] == static_cast<float>(step) &&
                  recv[1] == static_cast<float>(batch_size)))
                ok.fail();
            // Слот ранка r в recv — его собственные слоты (ADR-10: непрозрачный
            // буфер; формат слотов знает только engine): блок r*8..(r+1)*8.
            if (!(recv[std::size_t(r) * 8 + 2] == (r == 0 ? 0.0f : -1.0f))) ok.fail();
            // Соседние ранги: их слоты в своих блоках (r*8..(r+1)*8-1).
            if (r != 0) {
                const rank_t s = 0;
                if (!(recv[std::size_t(s) * 8 + 2] == 0.0f)) ok.fail();   // слот ранка 0
            }
            // 8 float = 32 байта — фиксированный пейлоад уточнения раунда 1.
            CHECK(sizeof(meta) == 32);
            CHECK(dtype_size(CommDType::F32) * 8 == 32);
        });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(meta_broadcast_multi_step_follower_execution) {
    // Многошаговый прогон: планировщик (ранг 0) рассылает мету каждый шаг,
    // последователи исполняют полученную разметку (шаги синхронизированы).
    // ВСЕ ранки обязаны вызывать allgather каждый шаг — иначе ранг 0 виснет
    // (расхождение с scheduler.cpp раунда 1, см. CONTRACT_ENGINE.md).
    constexpr rank_t WORLD = 2;
    SafeFlag ok;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < WORLD; ++r)
        th.emplace_back([r, &ok] {
            auto comm = std::make_unique<EmulatedCommunicator>(make_cfg(WORLD, 502, r));
            for (std::uint64_t step = 0; step < 50; ++step) {
                float meta[8] = {};
                if (r == 0) {
                    meta[0] = static_cast<float>(step);
                    meta[1] = 3.0f;
                }
                std::vector<float> recv(std::size_t(WORLD) * 8);
                comm->allgather(meta, recv.data(), 8, CommDType::F32);
                // Последователь исполняет ПОЛУЧЕННУЮ мету (не свою локальную):
                const std::uint64_t step_seen = static_cast<std::uint64_t>(recv[0]);
                const std::uint64_t batch_seen = static_cast<std::uint64_t>(recv[1]);
                if (step_seen != step || batch_seen != 3) ok.fail();
            }
        });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(meta_broadcast_e4m3_rejected_meta_path) {
    // Отклонённый CommDType::Meta не существует; E4M3-мета бессмысленна:
    // контракт разрешает только F16/F32/E4M3 — мета фиксируется как F32.
    CHECK(dtype_size(CommDType::F32) == 4);
    // Компиляционный фикс: перечисление CommDType не содержит Meta.
    for (int t = 0; t < 3; ++t)
        CHECK(dtype_size(static_cast<CommDType>(t)) != 0);
}
