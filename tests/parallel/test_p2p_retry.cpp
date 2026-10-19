// tests/parallel: ретрай-протокол P2P GPU-валидации ([R3] round 3, planner
// round 2 [Q1]; третья попытка — [R1] round 4) — ЧИСТАЯ CPU-логика из
// stell/parallel/p2p_retry.hpp.
//
// Исполнение самого GPU-теста — на сервере (test_nccl_gpu.cpp, оркестратор
// с дочерними процессами под watchdog'ом); здесь проверяются решения
// протокола: ровно три попытки (1 дефолт; 2 с NCCL_P2P_DISABLE=1; 3 с
// NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1), bench-фаза разрешена после pass
// и после «<2 карт», но не после неудачной корректности.
#include "test_harness.hpp"

#include "stell/parallel/p2p_retry.hpp"

#include <string>

using namespace stell::parallel;

TEST(p2p_retry_pass_on_first_attempt_no_retry) {
    CHECK(p2p_retry_decide(1, kOutcomePass) == GpuRetryVerdict::Pass);
    CHECK(p2p_retry_decide(2, kOutcomePass) == GpuRetryVerdict::Pass);
    CHECK(p2p_retry_decide(3, kOutcomePass) == GpuRetryVerdict::Pass);
}

TEST(p2p_retry_hang_or_pending_retries_once_with_p2p_disabled) {
    // Зависание/сбой попытки 1 (любой pending-исход) → попытка 2 без P2P.
    CHECK(p2p_retry_decide(1, "pending:OOM") ==
          GpuRetryVerdict::RetryWithP2pDisabled);
    CHECK(p2p_retry_decide(1, "pending:nccl bootstrap timeout") ==
          GpuRetryVerdict::RetryWithP2pDisabled);
    // «Зависание» — дочерний процесс убит watchdog'ом, файл результата
    // отсутствует → пустой исход; тоже ретрай.
    CHECK(p2p_retry_decide(1, "") == GpuRetryVerdict::RetryWithP2pDisabled);
    // Попытка 2 с NCCL_P2P_DISABLE=1 (SHM остаётся транспортом по умолчанию).
    CHECK(std::string(p2p_disable_env(2)) == "1");
    CHECK(std::string(p2p_disable_env(1)).empty());
    CHECK(std::string(shm_disable_env(2)).empty());
    // Неудача попытки 2 (round 3: shm-attach errno 115/11) → попытка 3
    // без P2P и без SHM (NET/sockets-транспорт).
    CHECK(p2p_retry_decide(2, "pending:nccl bootstrap timeout") ==
          GpuRetryVerdict::RetryWithShmDisabled);
    CHECK(p2p_retry_decide(2, "") == GpuRetryVerdict::RetryWithShmDisabled);
    CHECK(std::string(shm_disable_env(3)) == "1");
    CHECK(std::string(p2p_disable_env(3)) == "1");
    // Четвёртой попытки нет — транспорт больше не меняется, исход попытки 3
    // окончателен (pending или fail).
    CHECK(p2p_retry_decide(3, "pending:nccl bootstrap timeout") ==
          GpuRetryVerdict::GiveUpPending);
    CHECK(p2p_retry_decide(4, "pending:nccl bootstrap timeout") ==
          GpuRetryVerdict::GiveUpPending);
}

TEST(p2p_retry_confirmed_bitmatch_gives_fail_not_pending) {
    // Прогон дошёл до проверок и НЕ подтвердил корректность (fail:) —
    // ретрай есть (внезапный транспортный эффект), но после попытки 3
    // вердикт GiveUp, и оркестратор обязан выдать ЧЕСТНЫЙ FAIL, не pending.
    CHECK(p2p_retry_decide(1, "fail:битовое равенство не подтверждено") ==
          GpuRetryVerdict::RetryWithP2pDisabled);
    CHECK(p2p_retry_decide(2, "fail:битовое равенство не подтверждено") ==
          GpuRetryVerdict::RetryWithShmDisabled);
    CHECK(p2p_retry_decide(3, "fail:битовое равенство не подтверждено") ==
          GpuRetryVerdict::GiveUpPending);
    CHECK(outcome_is_fail("fail:битовое равенство не подтверждено"));
    CHECK(!outcome_is_pending("fail:битовое равенство не подтверждено"));
}

TEST(p2p_retry_max_attempts_is_three) {
    // Протокол [R1] round 4: РОВНО три попытки (дефолт / без P2P / без P2P+SHM).
    CHECK(kGpuRetryMaxAttempts == 3);
    // Каждая неудача попытки < 3 даёт следующий шаг цепочки транспорта;
    // неудача попытки 3 — терминальный вердикт GiveUp (не ретрай).
    CHECK(p2p_retry_decide(kGpuRetryMaxAttempts, "pending:x") ==
          GpuRetryVerdict::GiveUpPending);
    CHECK(p2p_retry_decide(kGpuRetryMaxAttempts, "") ==
          GpuRetryVerdict::GiveUpPending);
    // Согласованность bench_allowed_after с тремя попытками: bench разрешён
    // только по исходу самой корректности (pass или «<2 карт»), номер
    // попытки на допуск bench не влияет и не проверяется отдельно.
    CHECK(bench_allowed_after(kOutcomePass));
    CHECK(bench_allowed_after(kOutcomeFewDevices));
    CHECK(!bench_allowed_after("pending:попытки исчерпаны"));
    CHECK(!bench_allowed_after("fail:попытки исчерпаны"));
}

TEST(p2p_retry_outcome_classification) {
    CHECK(outcome_is_pass(kOutcomePass));
    CHECK(!outcome_is_pass(kOutcomeFewDevices));
    CHECK(outcome_is_pending(kOutcomeFewDevices));
    CHECK(outcome_is_pending("pending:нужно ≥2 устройств"));
    CHECK(!outcome_is_pending(kOutcomePass));
    CHECK(!outcome_is_fail(kOutcomePass));
}

TEST(p2p_retry_bench_allowed_after_pass_or_few_devices_only) {
    CHECK(bench_allowed_after(kOutcomePass));
    // <2 карт: корректность неприменима, но bench (TP1) идти может —
    // иначе scaling-бенчмарк TP1/TP2/TP4 [R8] невозможен на одной карте.
    CHECK(bench_allowed_after(kOutcomeFewDevices));
    // Неудачная корректность (доступные карты не подтвердили контракт) —
    // bench не запускаем: результаты латентностей на сломанном контуре
    // недостоверны.
    CHECK(!bench_allowed_after("pending:OOM"));
    CHECK(!bench_allowed_after("fail:битовое равенство"));
    CHECK(!bench_allowed_after(""));
}
