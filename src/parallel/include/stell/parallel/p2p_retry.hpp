// stell-llm — parallel: протокол ретрая GPU-валидации NCCL при зависании P2P.
//
// Источник: planner round 2 [Q1] + [R3] round 3 + [R1] round 4
// (T-20261017200354), протокол docs/design/server-env.md
// («NCCL-транспорт: ограничение среды»):
//   попытка 1 — дефолтное окружение (P2P как есть);
//   при неудаче (зависание коллективов при ncclCommInitRank COMPLETE —
//   типичный признак вмешательства co-tenancy в direct-pointer P2P) —
//   попытка 2 с NCCL_P2P_DISABLE=1: NCCL сам выбирает SHM-транспорт
//   для intra-node P2P (ring через host/shm); корректность ОБЯЗАНА
//   подтверждаться и без P2P;
//   при неудаче попытки 2 (shm-attach errno 115/11 — round 3) —
//   попытка 3 с NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1: вынуждает NCCL
//   на NET/sockets-транспорт между картами (штатная альтернатива SHM
//   в NCCL 2.26.2);
//   ровно три попытки — транспорт больше не меняется, исход попытки 3
//   окончателен (pending или fail).
//
// Заголовок БЕЗ CUDA/NCCL-зависимостей: логика юнит-тестируется на CPU
// (tests/parallel/test_p2p_retry.cpp), исполнение — в test_nccl_gpu.cpp
// (оркестратор: дочерние процессы под watchdog'ом).
#pragma once

#include <string>

namespace stell::parallel {

// --- исходы попытки (значение файла результата дочернего процесса) -----------

// Проверки корректности пройдены (битовое равенство CPU-эталону).
inline constexpr const char* kOutcomePass = "pass";

// Карт меньше двух: тест корректности неприменим (нужны ≥2 ранка), но
// bench-фаза (latency allreduce) при этом может исполняться (TP1).
inline constexpr const char* kOutcomeFewDevices =
    "pending:нужно ≥2 устройств (STELL_GPU_DEVICES)";

// Исход = «pass»?
inline bool outcome_is_pass(const std::string& outcome) {
    return outcome == kOutcomePass;
}

// Префиксы честного pending (проверки не проводились: checks=0 failures=0)
// и честного fail (прогон дошёл до проверок и НЕ подтвердил корректность —
// маскировать pending'ом запрещено: «корректность обязана подтверждаться»).
inline bool outcome_is_pending(const std::string& outcome) {
    return outcome.rfind("pending:", 0) == 0;
}
inline bool outcome_is_fail(const std::string& outcome) {
    return outcome.rfind("fail:", 0) == 0;
}

// --- решение о следующей попытке ---------------------------------------------

enum class GpuRetryVerdict {
    Pass,                 // попытка подтвердила корректность — ретрай не нужен
    RetryWithP2pDisabled, // повторить с NCCL_P2P_DISABLE=1 (попытка 2, SHM-транспорт)
    RetryWithShmDisabled, // повторить с NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1
                          //   (попытка 3, NET/sockets-транспорт)
    GiveUpPending,        // попытки исчерпаны — честный pending/fail как есть
};

inline constexpr int kGpuRetryMaxAttempts = 3;

// Решение после попытки номер `attempt` (1-базированно) с исходом `outcome`.
// Ретрай — после ЛЮБОЙ неудачи попытки 1 (зависание = отсутствие файла
// результата при убитом watchdog'ом дочернем процессе; OOM; сбой бутстрапа;
// неподтверждённая корректность): NCCL_P2P_DISABLE меняет транспорт на
// ring через host, что снимает именно класс зависаний direct-pointer P2P.
// Ретрай после неудачи попытки 2: SHM-транспорт тоже сломался средой
// (round 3: shm-attach errno 115/11) — попытка 3 вынуждает NET/sockets.
inline GpuRetryVerdict p2p_retry_decide(int attempt, const std::string& outcome) {
    if (outcome_is_pass(outcome)) return GpuRetryVerdict::Pass;
    if (attempt < 1) attempt = 1;
    if (attempt >= 2) {
        if (attempt == 2) return GpuRetryVerdict::RetryWithShmDisabled;
        return GpuRetryVerdict::GiveUpPending;
    }
    return GpuRetryVerdict::RetryWithP2pDisabled;
}

// Значение NCCL_P2P_DISABLE для попытки `attempt` (пусто — не выставлять).
// Попытка 2+: "1" (ring через host/shm); попытка 1 — наследуем окружение.
inline const char* p2p_disable_env(int attempt) {
    return attempt >= 2 ? "1" : "";
}

// Значение NCCL_SHM_DISABLE для попытки `attempt` (пусто — не выставлять).
// Только попытка 3: "1" — запрещает и SHM-транспорт, NCCL вынуждается на
// NET/sockets между картами; попытка 2 оставляет SHM как транспорт по умолчанию.
inline const char* shm_disable_env(int attempt) {
    return attempt >= 3 ? "1" : "";
}

// Разрешён ли bench-фазе (latency allreduce) прогон при данном исходе
// фазы корректности: да — если корректность подтверждена, либо если она
// неприменима (меньше 2 карт — TP1-случай scaling-бенчмарка [R8]).
inline bool bench_allowed_after(const std::string& correctness_outcome) {
    return outcome_is_pass(correctness_outcome) ||
           correctness_outcome == kOutcomeFewDevices;
}

} // namespace stell::parallel
