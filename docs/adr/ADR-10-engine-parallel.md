# ADR-10: Контракт engine ↔ parallel — комм-паттерны TP/EP/PP/KV-streaming

- **Статус:** Принято (интерфейсный контракт, зафиксирован до старта потоков A и C)
- **Стороны:** `core` (engine) ↔ `parallel` (NCCL-обёртка и комм-паттерны)
- **Правило изменения:** только через ADR и согласование с worker.
- **Связано:** ADR-02 (NCCL), ADR-08 (слоты), ADR-09 (описание модели)

## Контекст

Поток C строит TP (dense), EP-шардирование экспертов и PP поверх NCCL по
NVLink; поток A — планировщик, который должен исполнять слои модели на
нескольких ранках, не зная деталей коммуникаций. Нужен единый контракт:
какие паттерны, над каким интерфейсом, как engine делит слоты/сессии при
TP/EP, и как всё это тестируется на CPU без NCCL и GPU.

## Решение

### 1. Комм-паттерны (зафиксированный набор)

| Паттерн | Что делает | Где используется |
|---|---|---|
| `AllReduce` | суммирование по всем ранкам | TP: residual/LN, logits, градиентоподобные суммы |
| `AllGather` | сбор полного тензора со всех ранков | TP: сбор выхода attention-heads / KV-стримов |
| `ExpertAllToAll` | рассылка токенов к владельцам экспертов и обратно | EP: gather/scatter MoE-слоя |
| `ChainSend/ChainRecv` | последовательная передача по цепочке ранков | PP: передача активаций/градиентов; дренаж |
| `KvStream` | потоковая передача KV-блоков между узлами | длинный контекст (поток A + C) |

`AllReduce`/`AllGather` — NCCL-коллективы; `ExpertAllToAll` — комбинация
alltoall-семантики поверх NCCL (send/recv по парам ранков) + MoE gather/scatter
ядра (kernels). `KvStream` — поверх ChainSend/ChainRecv с кольцевой очередью.

### 2. Интерфейс поверх NCCL (заголовок `stell/parallel/comm.hpp`)

```cpp
namespace stell::parallel {

using rank_t = std::int32_t;

// Тип данных комм — только те, что реально гоняются по NVLink (ADR-04):
enum class CommDType { F16, F32, E4M3 };

class ICommunicator {
public:
    virtual ~ICommunicator() = default;
    virtual rank_t world_size() const = 0;
    virtual rank_t rank() const = 0;

    // Все операции синхронны по семантике (блокируют поток до завершения
    // коллективной операции) и обязаны запускаться на переданном CUDA-stream
    // (CPU-эмуляция игнорирует stream). Асинхронные хендлы — по мере надобности
    // отдельным ADR, пока потребность не доказана.
    virtual void allreduce(void* buf, std::size_t count, CommDType t) = 0;
    virtual void allgather(void* send, void* recv, std::size_t count, CommDType t) = 0;

    // EP: токены (индексы экспертов + эмбеддинги) уезжают к владельцам экспертов
    // и возвращаются после FFN. Семантика: send_tokens[i] уходит на ранок
    // owner(expert_id[i]); recv_tokens возвращают локальные токены других ранков.
    virtual void expert_alltoall(
        void* send_tokens, void* send_meta,   // meta: expert_id, слот (ADR-08)
        void* recv_tokens, void* recv_meta,
        std::size_t max_tokens, CommDType t) = 0;

    // PP: point-to-point по цепочке (peer = rank±1).
    virtual void chain_send(void* buf, std::size_t count, CommDType t, rank_t peer) = 0;
    virtual void chain_recv(void* buf, std::size_t count, CommDType t, rank_t peer) = 0;

    // KV-streaming: неблокирующий кольцевой канал KV-блоков (ADR-04 dtype).
    virtual void kv_stream_push(const void* kv_block, std::size_t bytes) = 0;
    virtual bool  kv_stream_pop(void* dst, std::size_t bytes) = 0;
};

// Фабрика: NCCL-реализация (CUDA) либо CPU-эмулятор (§4).
std::unique_ptr<ICommunicator> make_communicator(
    const ParallelConfig& cfg);   // { tp, pp, ep, device_ids } — из конфига engine

} // namespace stell::parallel
```

Ошибки — исключениями (единая политика с ADR-08):
`CommError::Code { NcclFailure, RankMismatch, BufferTooSmall, Timeout }`.

### 3. Как engine делит слоты и сессии при TP/EP

Контракт распределения (одинаково на всех ранках, без переговоров):

- **Сессии/планировщик:** планировщик continuous batching живёт на **ранке 0**
  и рассылает решение о шаге батча; остальные ранки исполняют ту же логику по
  полученной разметке (broadcast через `allgather` мета-структуры шага —
  маленькая, фиксированная). Слоты сессий (ADR-08) **одинаковы на всех ранках**:
  slot_id один и тот же, содержимое слота шардировано по ранкам (§ ниже).
- **TP (dense):** головы attention и колонки FFN делятся по ранкам равномерно;
  тензорные группы с ролью `attn` размечаются уже по shard-формам в описании
  модели (ADR-09: поле `shape` группы на ранке после шардирования);
  `slot_bytes` слота — на ранок, т.е. VRAM-слот каждого ранка вмещает свою долю.
  KV-кеш делится по KV-головам (n_kv_heads % tp == 0 — обязательная проверка
  при разметке).
- **EP (MoE):** эксперты закрепляются за ранками статически на план
  (`expert_id % ep == rank`); роутер вычисляется на ранке 0, токены
  отправляются владельцам через `expert_alltoall`. Тензорные группы с ролью
  `expert` размечаются только для локальных экспертов ранка.
- **PP (dense > VRAM):** слои делятся на цепочку ранков; слот сессии
  на ранке содержит только свой диапазон слоёв; передача активаций —
  `chain_send/chain_recv` на границе слоёв; KV-кеш — на ранке-владельце слоя.
- Комбинации (PP×TP, PP×EP) — декартово произведение вышеизложенного;
  порядок: сначала PP (слои), затем TP/EP (внутри слоёв), ранок = pp_rank * tp_ep + tp_ep_rank.

### 4. CPU-эмуляция комм-паттернов (обязательный CPU-путь)

`make_communicator({mode: "cpu-emulation"})` возвращает `EmulatedCommunicator`:

- ранки = потоки одного процесса (или отдельные процессы на shared-memory
  сегментах — фиксируется реализацией parallel, контракт — только интерфейс);
- `allreduce/allgather` — барьерные коллективы на бэкетах; `expert_alltoall` —
  обмен через общие очереди; `chain_send/recv` — очереди; `KvStream` —
  SPSC-очередь;
- **никаких NCCL/GPU/CUDA** — юнит-тесты parallel и CPU-конфиг CI (ADR-07)
  гоняют интерфейс без сервера;
- семантика тайм-аутов: эмулятор обязан поддерживать те же ошибки
  (`Timeout`) — тесты на deadlock-свободность планировщика.

Дополнительная симметрия контракта: golden-логиты dense-модели на
CPU-эмуляторе TP2 обязаны совпасть с TP1 (одна модель, одни фикстуры
ADR-09 §4) — это главный тест корректности шардирования без GPU.

## Альтернативы

- **Асинхронные коллективы (хендлы) по всему интерфейсу** — усложняет engine
  без доказанной нужды; выбраны синхронные операции на CUDA-stream (NCCL
  синхронен по семантике, но асинхронен по исполнению на stream).
- **Свой allreduce по NVLink (как в 1Cat-ядрах)** — откат к ADR-02: NCCL
  закрывает коллективы; свои ядра оставлены для MoE gather/scatter в kernels.
- **Динамическое переподшардирование слотов на лету** — противоречит фиксированным
  пулам (ADR-05); переснятие графа + новый план — единственный путь.

## Последствия

- Потоки A и C пишут независимо: A — против `ICommunicator` и правил §3,
  C — реализует NCCL-версию и CPU-эмулятор.
- Разметка модели (ADR-09) обязана хранить **shard-формы** для TP/EP —
  это вход `set_plan` (ADR-08), т.е. tiers работает с уже подшардированными группами.
- Серверные прогоны TP2/TP4 — только с учётом совмещения (чужой процесс держит
  ~21–30 GB VRAM на V100); CPU-эмуляция — локально.
