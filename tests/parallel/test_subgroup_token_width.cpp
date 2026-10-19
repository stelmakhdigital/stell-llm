// tests/parallel: уточнения контракта ADR-10 (round 2, R4/R5) — ширина токена
// expert_alltoall (token_elems) и подгруппы subgroup(ranks) с честной ошибкой
// BufferTooSmall (без молчаливого сокращения пересылки).
#include "test_harness.hpp"
#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/emulated.hpp"
#include "stell/parallel/plan.hpp"

#include <atomic>
#include <thread>
#include <vector>

using namespace stell::parallel;

namespace {

// Ранок-раннер: fn(comm) в потоке ранка rank (как в test_emulated_collectives).
// CHECK внутри потоков не используется — ошибки собираются в флаги после join.
template <class Fn>
std::vector<std::thread> run_group(rank_t pp, rank_t tp, rank_t ep, int group_id, Fn&& fn) {
    ParallelConfig c;
    c.mode = CommMode::CpuEmulation;
    c.pp = pp; c.tp = tp; c.ep = ep;
    c.group_id = group_id;
    c.timeout_ms = 3000;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < c.world(); ++r) {
        ParallelConfig rc = c;
        rc.rank = r;
        th.emplace_back(fn, std::make_unique<EmulatedCommunicator>(rc));
    }
    return th;
}

// Подгруппа TP/EP-этапа при pp>1: стадия pp = ранки [pp*(tp*ep) .. +tp*ep).
std::vector<rank_t> stage_ranks(rank_t pp_rank, rank_t tp, rank_t ep) {
    std::vector<rank_t> out;
    const rank_t base = pp_rank * tp * ep;
    for (rank_t i = 0; i < tp * ep; ++i) out.push_back(rank_t(base + i));
    return out;
}

constexpr float tok_value(rank_t src, int i, int dd) {
    return float(src * 1000 + i * 10 + dd);
}

// Коды ошибок из потоков (счётчики харнесса не потокобезопасны).
std::atomic<bool>& flag(const char* name) {
    static std::atomic<bool> dup{false}, out{false}, foreign{false}, bts{false};
    if (name[0] == 'd') return dup;         // "dup"
    if (name[0] == 'o') return out;         // "out"
    if (name[0] == 'f') return foreign;     // "foreign"
    return bts;                             // "bts"
}

} // namespace

// --- [R4] token_elems = d_model > 1: маршрутизация ЦЕЛЫХ токенов --------------

TEST(tok_width_expert_alltoall_dmodel_ep2) {
    // world=2 (ep=2, tp=1): токен — вектор d_model=8; побитовая доставка вектора,
    // детерминированный порядок (источник ↑, индекс ↑), заполнители.
    constexpr int DM = 8, T = 4;
    constexpr rank_t EP = 2;
    SafeFlag ok;
    auto th = run_group(1, 1, EP, 4501, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        std::vector<float> send(std::size_t(T) * DM);
        std::vector<TokenMeta> smeta(T);
        for (int i = 0; i < T; ++i) {
            const std::int32_t eid = std::int32_t((my * 3 + i) % int(EP));
            smeta[std::size_t(i)] = TokenMeta{eid, std::int32_t(my * 16 + i)};
            for (int dd = 0; dd < DM; ++dd)
                send[std::size_t(i) * DM + dd] = tok_value(my, i, dd);
        }
        const std::size_t cap = std::size_t(T) * std::size_t(EP);           // токенов
        std::vector<float> recv(cap * DM);
        std::vector<TokenMeta> rmeta(cap);
        c->expert_alltoall(send.data(), smeta.data(), recv.data(), rmeta.data(),
                           std::size_t(T), std::size_t(DM), CommDType::F32);
        // Ожидаемое число входящих токенов (владельцы моих экспертов).
        std::size_t expect = 0;
        for (rank_t s = 0; s < EP; ++s)
            for (int i = 0; i < T; ++i)
                if (rank_t((s * 3 + i) % int(EP)) == my) ++expect;
        std::size_t got = 0;
        for (std::size_t i = 0; i < cap; ++i) {
            if (rmeta[i].expert_id < 0) {
                if (!(rmeta[i].slot_id == -1)) ok.fail();                   // filler-мета
                for (int dd = 0; dd < DM; ++dd)
                    if (!(recv[i * DM + dd] == 0.f)) ok.fail();             // filler-нули
                continue;
            }
            ++got;
            // Мета восстанавливается детерминированно из (eid, slot).
            bool found = false;
            for (rank_t s = 0; s < EP && !found; ++s)
                for (int k = 0; k < T && !found; ++k) {
                    const std::int32_t eid = std::int32_t((s * 3 + k) % int(EP));
                    const std::int32_t slot = std::int32_t(s * 16 + k);
                    if (rmeta[i].expert_id == eid && rmeta[i].slot_id == slot) {
                        found = true;
                        if (rank_t(eid) % EP != my) ok.fail();              // чужой токен!
                        for (int dd = 0; dd < DM; ++dd)
                            if (!(recv[i * DM + dd] == tok_value(s, k, dd)))
                                ok.fail();                                  // целый вектор
                    }
                }
            if (!found) ok.fail();
        }
        if (!(got == expect)) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

// --- [R5] subgroup: TP-этап при pp>1 ------------------------------------------

TEST(sub_tp_stage_allreduce_pp2_tp2) {
    // pp=2, tp=2: allreduce внутри подгруппы TP-этапа — сумма ТОЛЬКО по ранкам
    // группы; повторный subgroup() с тем же ranks — тот же объект (владение у
    // родителя); родительская группа после подгруппы не повреждена.
    SafeFlag ok;
    auto th = run_group(2, 2, 1, 4502, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        const rank_t stage = my / 2;                       // tp*ep = 2
        ICommunicator* sub = c->subgroup(stage_ranks(stage, 2, 1));
        if (!(sub->world_size() == 2)) ok.fail();
        if (!(sub->rank() == int(my) % 2)) ok.fail();      // перенумерация: позиция в ranks
        float v[4];
        for (int i = 0; i < 4; ++i) v[i] = float(my) + 1.f + float(i);
        sub->allreduce(v, 4, CommDType::F32);
        // Сумма только по своей стадии: {0,1} → (0+1+i)+(1+1+i) = 3+2i;
        // {2,3} → (2+1+i)+(3+1+i) = 7+2i.
        const float stage_base = (stage == 0) ? 3.f : 7.f;
        for (int i = 0; i < 4; ++i)
            if (!(v[i] == stage_base + 2.f * float(i))) ok.fail();
        // Повторный вызов с тем же ranks — ТОТ ЖЕ под-комм.
        ICommunicator* sub2 = c->subgroup(stage_ranks(stage, 2, 1));
        if (!(sub2 == sub)) ok.fail();
        // Родительская группа жива: полный allreduce по всем четырём ранкам.
        float w[2] = {float(my), float(my) * 2.f};
        c->allreduce(w, 2, CommDType::F32);
        for (int i = 0; i < 2; ++i)
            if (!(w[i] == 6.f * float(i + 1))) ok.fail();  // 0+1+2+3 = 6
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(sub_tp_stage_allgather_pp2_tp2) {
    // allgather внутри подгруппы: конкатенация только по членам стадии.
    SafeFlag ok;
    auto th = run_group(2, 2, 1, 4503, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        const rank_t stage = my / 2;
        ICommunicator* sub = c->subgroup(stage_ranks(stage, 2, 1));
        float send[2], recv[4];
        for (int i = 0; i < 2; ++i) send[i] = 100.f * float(my) + float(i);
        sub->allgather(send, recv, 2, CommDType::F32);
        const rank_t base = stage * 2;
        for (int k = 0; k < 2; ++k)
            for (int i = 0; i < 2; ++i)
                if (!(recv[k * 2 + i] == 100.f * float(base + k) + float(i))) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(sub_expert_alltoall_fanout_inside_stage) {
    // pp=2, tp=2, ep=1: подгруппа = TP-этап; expert_alltoall с token_elems=4 —
    // fan-out на оба tp-ранка стадии; входящий поток == ёмкость (worst-case).
    constexpr int S = 3, DM = 4;
    SafeFlag ok;
    auto th = run_group(2, 2, 1, 4504, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        const rank_t stage = my / 2;
        ICommunicator* sub = c->subgroup(stage_ranks(stage, 2, 1));
        std::vector<float> send(std::size_t(S) * DM);
        std::vector<TokenMeta> smeta(S);
        for (int i = 0; i < S; ++i) {
            smeta[std::size_t(i)] = TokenMeta{0, std::int32_t(my * 16 + i)};  // ep=1 → владелец 0
            for (int dd = 0; dd < DM; ++dd)
                send[std::size_t(i) * DM + dd] = tok_value(my, i, dd);
        }
        const std::size_t cap = std::size_t(S) * 2;                           // токенов
        std::vector<float> recv(cap * DM);
        std::vector<TokenMeta> rmeta(cap);
        sub->expert_alltoall(send.data(), smeta.data(), recv.data(), rmeta.data(),
                             std::size_t(S), std::size_t(DM), CommDType::F32);
        // Входящие: от обоих членов стадии по S токенов = 2S = cap, ровно;
        // порядок (источник ↑, индекс ↑).
        std::size_t idx = 0;
        for (rank_t s = 0; s < 2; ++s) {
            const rank_t src = stage * 2 + s;
            for (int i = 0; i < S; ++i, ++idx) {
                if (!(rmeta[idx].expert_id == 0)) ok.fail();
                if (!(rmeta[idx].slot_id == std::int32_t(src * 16 + i))) ok.fail();
                for (int dd = 0; dd < DM; ++dd)
                    if (!(recv[idx * DM + dd] == tok_value(src, i, dd))) ok.fail();
            }
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

// --- [R5] BufferTooSmall: маршрут вне подгруппы, молчаливого сокращения нет ---

TEST(sub_route_outside_buffer_too_small) {
    // world=4 (tp=2, ep=2, pp=1); подгруппа {0,1} = EP-группа 0. Токен ранка 0
    // с expert_id=1 (владелец — EP-группа 1: ранки {2,3}) уходит ЗА пределы
    // подгруппы → CommError(BufferTooSmall) у ВСЕХ членов; чужие ранки (2,3)
    // коллективом не затрагиваются и продолжают работать в родителе.
    SafeFlag ok;
    auto th = run_group(1, 2, 2, 4505, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        if (my >= 2) {
            // Не-члены подгруппу не создают; их СВОЯ подгруппа {2,3} (EP-группа 1)
            // и родительская группа живут независимо: сумма только по {2,3} = 5.
            ICommunicator* other = c->subgroup({2, 3});
            float w[1] = {float(my)};
            other->allreduce(w, 1, CommDType::F32);
            if (!(w[0] == 5.f)) ok.fail();
            return;
        }
        ICommunicator* sub = c->subgroup({0, 1});
        TokenMeta sm[2];
        float st[2] = {0.f};
        // Ранок 0 шлёт токены чужой EP-группы (expert_id=1 → ранки 2,3).
        sm[0] = TokenMeta{my == 0 ? 1 : 0, std::int32_t(my)};
        sm[1] = TokenMeta{my == 0 ? 1 : 0, std::int32_t(my + 8)};
        std::vector<TokenMeta> rm(4);
        std::vector<float> rt(4);
        try {
            sub->expert_alltoall(st, sm, rt.data(), rm.data(), 2, 1, CommDType::F32);
            ok.fail();                                         // ошибки быть обязано
        } catch (const CommError& e) {
            if (e.code() == CommError::Code::BufferTooSmall)
                flag("bts").store(true);
            else
                ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(flag("bts").load());
    CHECK(ok.get());
}

// --- subgroup: валидация ranks + вложенность -----------------------------------

TEST(sub_invalid_and_nested) {
    SafeFlag ok;
    auto th = run_group(2, 2, 1, 4506, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        // Повторяющиеся ранки → RankMismatch.
        try {
            c->subgroup({0, 0});
            ok.fail();
        } catch (const CommError& e) {
            if (e.code() == CommError::Code::RankMismatch) flag("dup").store(true);
            else ok.fail();
        }
        // Ранок вне world → RankMismatch.
        try {
            c->subgroup({5});
            ok.fail();
        } catch (const CommError& e) {
            if (e.code() == CommError::Code::RankMismatch) flag("out").store(true);
            else ok.fail();
        }
        // Чужая подгруппа (собственный ранок не входит) → RankMismatch.
        try {
            const rank_t other_stage = (my < 2) ? 1 : 0;
            c->subgroup(stage_ranks(other_stage, 2, 1));
            ok.fail();
        } catch (const CommError& e) {
            if (e.code() == CommError::Code::RankMismatch) flag("foreign").store(true);
            else ok.fail();
        }
        // Вложенная подгруппа: из стадии — одноранковая {my}; allreduce возвращает
        // собственное значение.
        const rank_t stage = my / 2;
        ICommunicator* sub = c->subgroup(stage_ranks(stage, 2, 1));
        std::vector<rank_t> single{my};
        ICommunicator* one = sub->subgroup(single);
        if (!(one->world_size() == 1 && one->rank() == 0)) ok.fail();
        float v[1] = {float(my) + 5.f};
        one->allreduce(v, 1, CommDType::F32);
        if (!(v[0] == float(my) + 5.f)) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(flag("dup").load());
    CHECK(flag("out").load());
    CHECK(flag("foreign").load());
    CHECK(ok.get());
}
