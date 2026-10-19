# [R2] round 5 — per-group bench на реальном layout (роль tiers, Поток B)

> Статус: **СЕРВЕРНЫЙ ПРОГОН — worker после merge** (push на сервер запрещён).
> Этот каталог ждёт логи per-group прогона; ниже — точные команды worker'а,
> пометки протокола и локальный смоук плумбинга (НЕ серверный замер).

## Задача [R2] round 5 (planner [Q3]/[Q8])

`bench_tiers_pause` в режиме **pergroup** принимает реальный stell-tier
layout v1 — слоты групп **переменной ёмкости** (`TierLayoutReader`: план,
роли, ёмкости — из индекса; карта слоёв из имён: префикс `l<N>.` → слой N+1,
глобальные группы `embed`/`norm`/`head` → слой 0; допущение зафиксировано —
в проде карту слоёв знает engine, ADR-08 §5):

- **пауза слоя = Σ пауз групп слоя** (pin каждой группы слоя, ожидание
  готовности; загрузки групп — в очереди, батчи до k в полёте — deep-prefetch
  [R2b] round 4); wall-пауза fan-in печатается справочно;
- **hit-rate prefetch считается по группам** (`TierManager::
  group_prefetch_stats()` — расширение на concrete-классе; разрез по ролям
  expert/attn/kv из плана, ADR-09);
- режим I/O (`STELL_TIER_ODIRECT`) и глубина k (`STELL_TIER_PREFETCH_DEPTH`)
  — те же env, что в uniform-режиме (матрица round 4).

## Команды worker'а (сервер, после merge)

```bash
export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
cmake -S . -B build/cpu -G Ninja && cmake --build build/cpu
# layout tiny-qwen2moe должен существовать (конвертер models):
#   /tmp/stell_stream_real_out/tiny-qwen2moe.stelltier (65 групп, 18.8 МиБ)
PER_GROUP=1 ./src/tiers/bench_p95_nvme.sh build/cpu ~/tiers_bench
# (или с явным путём: REAL_LAYOUT=/tmp/stell_stream_real_out/tiny-qwen2moe.stelltier
#  PER_GROUP=1 ./src/tiers/bench_p95_nvme.sh build/cpu ~/tiers_bench)
```

Ячейки per-group (обе, логи `~/tiers_bench/pergroup_io*_k4.log` → этот
каталог, названия сохранены):

1. `pergroup_ioodirect_k4.log` — **целевой профиль {O_DIRECT + k=4}**;
   O_DIRECT — штатный честный cold по протоколу server-env round 5 ([Q8]);
   активность подтверждается строкой `bench io mode: ... odirect_active=true`
   в логе (если false — FS не поддерживает O_DIRECT: честный откат на
   buffered, ячейка помечается откатом, НЕ выдается за O_DIRECT-замер).
2. `pergroup_iobuffered_k4.log` — сравнительная buffered-ячейка (k=4);
   **тёплая категория** (fadvise DONTNEED best-effort под давлением page
   cache co-tenancy), помечена в логе.

## Обязательные пометки отчёта worker'а

- **co-tenancy**: NVMe/GPU общие с серверной нагрузкой (strata держит
  21–30 ГБ VRAM — НЕ трогать); `nvidia-smi`-снимок попадает в лог ячейки
  автоматически (строка `co-tenancy:`).
- **«НЕ ВЕРДИКТ ПО ЦЕЛИ §4 (p95 ≤ 5 мс)»** — вердикт только по чистому
  окну; цель не отменяется. Замер под co-tenancy — точки регрессии.
- **Сравнение замеров только внутри категории I/O** (server-env round 5,
  [Q8]): O_DIRECT = cold, buffered + fadvise = warm; cross-категорийное
  сравнение (odirect vs buffered ячейки) в вердикты не идёт — допустимо
  только как побитовая сверка корректности чтения (тест round 4).
- **Честная пометка профиля**: целевой «4 МиБ-слоты» относится к
  uniform-матрице round 4; в per-group режиме VRAM-слот = max slot_bytes
  ГРУПП layout (у tiny-qwen2moe — embed ≈ 9.3–9.7 МиБ: F16 [151646, 32]) —
  фактические числа печатаются строкой `pergroup layout:` в логе каждой
  ячейки. Профиль {O_DIRECT + k=4} применяется как есть. Если носитель
  layout — tmpfs/ramfs, ячейки помечаются «носитель RAM-disk — НЕ NVMe»
  (скрипт делает это автоматически).
- Профиль остатка: сырые Σ-паузы слоёв (`raw layer sums (ms)`) и per-group
  таблица (`pergroup groups (gid:hits/misses)`) — в логах ячеек
  (BENCH_DEBUG=1 включён скриптом).

## Локальный смоук (НЕ серверный замер)

- `smoke-local-pergroup.log` — прогон per-group bench на синтетическом
  per-group layout (8 групп, переменные слоты 4/8/12 КиБ, 3 «слоя»),
  O_DIRECT+k4 и buffered ячейки: **плумбинг смоук, локальный диск, НЕ
  серверный NVMe-замер; числа не сравниваются с целями §5**.
- Синтетика в тестах: ctest `tiers/test_bench_per_group` — per-group
  hit-rate побитово == ожидаемому паттерну (prefetch: все группы 1.0000;
  cold: все 0.0000 — детерминизм: префетч слоя материализуется до паузы);
  p50/p95 считаются; uniform-режим (6 аргументов) — прежний вывод.

## Чек-лист приёмки worker'а

- [ ] обе ячейки существуют, `bench io mode:` в каждой;
- [ ] odirect-ячейка: `odirect_active=true` (иначе честная пометка отката);
- [ ] co-tenancy-строка (nvidia-smi) в обеих ячейках;
- [ ] пометка «НЕ вердикт §4» перенесена в summary отчёта;
- [ ] сравнение только внутри категории I/O; cross-категорийные числа —
      рядом, без вердикта;
- [ ] логи закоммичены в этот каталог.
