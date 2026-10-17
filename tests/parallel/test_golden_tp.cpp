// tests/parallel: golden-инвариант TP2-логиты == TP1 на детерминированных
// входах (главный тест корректности шардирования без GPU, ADR-10 §4).
//
// Модель-фикстура: dense «attention» без softmax — логиты = сумма по головам
// w[h] · x[h]. TP1 считает все головы последовательно (по возрастанию h);
// TP2 делит головы по ранкам и сводит частичные суммы allreduce-ом в порядке
// ранков 0→1. Поскольку эмулятор редуцирует строго в порядке ранков, порядок
// сложения float сохраняется → побитовое совпадение.
//
// Семантика редукции зафиксирована (ADR-10 §2/§4): порядок ранков 0..N-1,
// аккумулятор F32, dtype буфера — как в операции.
#include "test_harness.hpp"
#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/emulated.hpp"
#include "stell/parallel/plan.hpp"

#include <cmath>
#include <thread>
#include <vector>

using namespace stell::parallel;

namespace {

constexpr int N_HEADS = 8;   // кратно 1 и 2 — деление по головам обязано
constexpr int N_LOGITS = 4;

// Детерминированные «веса голов» и «входы» (фикстура ADR-09 §4 — одна модель,
// одни входы; значения dyadic → точно представимы в half).
float head_weight(int h) { return 0.25f * float((h % 5) + 1); }
float head_input(int h, int l) { return 0.125f * float((h + l) % 7) + 1.0f; }

// TP1: эталон — все головы подряд по возрастанию, аккумулятор F32.
void reference_tp1(float* logits) {
    for (int l = 0; l < N_LOGITS; ++l) {
        float acc = 0.f;
        for (int h = 0; h < N_HEADS; ++h)
            acc += head_weight(h) * head_input(h, l);   // порядок h ↑ зафиксирован
        logits[l] = acc;
    }
}

template <class Fn>
std::vector<std::thread> run_tp(rank_t tp, int group_id, Fn&& fn) {
    ParallelConfig c;
    c.mode = CommMode::CpuEmulation;
    c.tp = tp;
    c.group_id = group_id;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < tp; ++r) {
        ParallelConfig rc = c;
        rc.rank = r;
        th.emplace_back(fn, std::make_unique<EmulatedCommunicator>(rc));
    }
    return th;
}

} // namespace

TEST(golden_tp2_logits_equal_tp1_f32_bitwise) {
    float ref[N_LOGITS];
    reference_tp1(ref);

    // --- TP1 (один ранок) через тот же ICommunicator ---
    {
        SafeFlag ok;
        auto th = run_tp(1, 201, [&](std::unique_ptr<ICommunicator> c) {
            float partial[N_LOGITS] = {0};
            if (c->rank() == 0)
                for (int l = 0; l < N_LOGITS; ++l) {
                    float acc = 0.f;
                    for (int h = 0; h < N_HEADS; ++h)
                        acc += head_weight(h) * head_input(h, l);
                    partial[l] = acc;
                }
            c->allreduce(partial, N_LOGITS, CommDType::F32);
            for (int l = 0; l < N_LOGITS; ++l)
                if (partial[l] != ref[l]) ok.fail();
        });
        for (auto& t : th) t.join();
        CHECK(ok.get());
    }

    // --- TP2: головы 0..3 на ранке 0, 4..7 на ранке 1 (по головам, ADR-10 §3);
    // частичные суммы сводятся allreduce-ом в порядке ранков 0→1 →
    // порядок сложения float совпадает с эталоном → побитовое равенство.
    {
        SafeFlag ok;
        auto th = run_tp(2, 202, [&](std::unique_ptr<ICommunicator> c) {
            const ShardRange heads = shard_range(N_HEADS, c->world_size(), c->rank());
            float partial[N_LOGITS] = {0};
            for (int l = 0; l < N_LOGITS; ++l) {
                float acc = 0.f;
                for (int h = heads.begin; h < heads.end; ++h)
                    acc += head_weight(h) * head_input(h, l);
                partial[l] = acc;
            }
            c->allreduce(partial, N_LOGITS, CommDType::F32);
            for (int l = 0; l < N_LOGITS; ++l)
                if (partial[l] != ref[l]) ok.fail();    // побитовое равенство
        });
        for (auto& t : th) t.join();
        CHECK(ok.get());
    }
}

TEST(golden_tp2_logits_equal_tp1_f16) {
    // F16-путь: входы dyadic; редукция в F32-аккумуляторе, результат в F16.
    float ref[N_LOGITS];
    reference_tp1(ref);
    SafeFlag ok;
    auto th = run_tp(2, 203, [&](std::unique_ptr<ICommunicator> c) {
        const ShardRange heads = shard_range(N_HEADS, c->world_size(), c->rank());
        std::uint16_t partial[N_LOGITS];
        for (int l = 0; l < N_LOGITS; ++l) {
            float acc = 0.f;
            for (int h = heads.begin; h < heads.end; ++h)
                acc += head_weight(h) * head_input(h, l);
            partial[l] = float_to_half(acc);
        }
        c->allreduce(partial, N_LOGITS, CommDType::F16);
        // Эталон: TP1 в тех же условиях (F16-запись частичных сумм, редукция
        // в F32) — порядок сложения тот же → совпадение с F32-эталоном.
        for (int l = 0; l < N_LOGITS; ++l) {
            float acc = 0.f;
            for (int h = 0; h < N_HEADS; ++h)
                acc += head_weight(h) * head_input(h, l);
            if (!((half_to_float(partial[l]) == acc))) ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(golden_tp4_logits_equal_tp1) {
    // TP4: 8 голов → по 2 на ранок; allreduce порядок 0→1→2→3 = эталонный h ↑.
    float ref[N_LOGITS];
    reference_tp1(ref);
    SafeFlag ok;
    auto th = run_tp(4, 204, [&](std::unique_ptr<ICommunicator> c) {
        const ShardRange heads = shard_range(N_HEADS, c->world_size(), c->rank());
        float partial[N_LOGITS] = {0};
        for (int l = 0; l < N_LOGITS; ++l) {
            float acc = 0.f;
            for (int h = heads.begin; h < heads.end; ++h)
                acc += head_weight(h) * head_input(h, l);
            partial[l] = acc;
        }
        c->allreduce(partial, N_LOGITS, CommDType::F32);
        for (int l = 0; l < N_LOGITS; ++l)
            if (partial[l] != ref[l]) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(golden_allgather_tp2_matches_tp1_full_tensor) {
    // Сбор выхода attention-голов (ADR-10 §1): allgather по ранкам —
    // конкатенация shard-ов = полный тензор TP1 (по возрастанию ранков).
    SafeFlag ok;
    auto th = run_tp(2, 205, [&](std::unique_ptr<ICommunicator> c) {
        const ShardRange heads = shard_range(N_HEADS, c->world_size(), c->rank());
        const int n_local = N_HEADS / 2;
        float send[N_LOGITS * n_local];
        for (int l = 0; l < N_LOGITS; ++l)
            for (int h = 0; h < n_local; ++h)
                send[l * n_local + h] = head_weight(heads.begin + h) * head_input(heads.begin + h, l);
        float recv[N_LOGITS * N_HEADS];
        c->allgather(send, recv, std::size_t(N_LOGITS * n_local), CommDType::F32);
        if (c->rank() == 0) {
            // Ожидаемая конкатенация по порядку ранков: буфер ранка s занимает
            // позиции s*N_LOGITS*n_local + l*n_local + hl
            // (голова глобальная = s*n_local + hl).
            const int n_local = N_HEADS / 2;
            for (rank_t s = 0; s < 2; ++s)
                for (int l = 0; l < N_LOGITS; ++l)
                    for (int hl = 0; hl < n_local; ++hl) {
                        const std::size_t idx = std::size_t(s) * (N_LOGITS * n_local) +
                                               std::size_t(l) * n_local + std::size_t(hl);
                        const int head = int(s) * n_local + hl;
                        if (recv[idx] != head_weight(head) * head_input(head, l)) ok.fail();
                    }
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}
