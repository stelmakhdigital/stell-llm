/* stell/engine/capi.h — контракт engine ↔ serve (ADR-11 §1, выписан дословно).
   Единственная точка стыка: стабильный C ABI, никаких C++-типов в границе.
   Служба (serve, поток E) связывается через ctypes/cffi; события и запросы —
   JSON-строки (поля только добавляются — мажорный ABI не меняется). */
#ifndef STELL_ENGINE_CAPI_H
#define STELL_ENGINE_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct stell_engine stell_engine_t;      /* непрозрачный handle */

/* Lifecycle: cfg — JSON-строка (модель, tier-границы, parallel, профили).
   Round 1 CPU-контур: {"runner":"demo", "engine":{"max_slots":16,
   "batch_step_max":16}, "parallel":{"tp":1,"pp":1,"ep":1}}.
   Возвращает 0 при успехе, -1 — ошибка (msg_out — текст, free() обязателен). */
int  stell_engine_create(const char* cfg_json, stell_engine_t** out);
void stell_engine_destroy(stell_engine_t* e);

/* Постановка запроса (ADR-11 §2). Возвращает id сессии (>0), 0 — ошибка
   (msg_out — текст, free() обязателен). Формат request_json:
     {"sampling":{"max_tokens":256,"temperature":1.0,"top_p":0.95,"top_k":20,
                  "seed":0},
      "prompt_tokens":[..], "eos_token":2}                                  */
uint64_t stell_submit(stell_engine_t* e, const char* request_json, char** msg_out);

/* Неблокирующее выкачивание событий сессии (ADR-11 §3). Число событий,
   0 — событий нет, <0 — сессия закрыта/ошибка. out_json — массив событий
   [{"seq":N,"type":"token","token_id":X}, ...]; free() обязателен. */
int  stell_poll(stell_engine_t* e, uint64_t session_id, char** out_json);

/* Отмена: сессия выбрасывается из батча на ближайшем шаге (ADR-11 §3).
   Возврат — по конвенции модуля (ADR-11 §1, как engine_create/submit):
   0 — успех; -1 — сессия не найдена или уже закрыта. */
int  stell_cancel(stell_engine_t* e, uint64_t session_id);

/* Round-1 расширение (однопоточный CPU-контур; ADR-11 §1: «поля только
   добавляются», новый символ — ABI-совместимо): один шаг декода батча.
   В прод-контуре шаги ведёт внутренний поток engine; serve вызывает
   stell_step, пока нет фоновой нити (фиксируется в отчёте потока core). */
void stell_step(stell_engine_t* e);

/* Метрики/health — JSON (ADR-11 §4): снапшот, потокобезопасно;
   free() обязателен.
   Round 3 [R5] (planner Q5): секция "logits" — строгий формат,
   БЕЗ payload логитов (массивы в /metrics не выдаются никогда):
     "logits":{"requests_total":N,"errors_total":N,
               "last_n_vocab":V,"last_positions":"last"|"all"|null}
   requests_total — число УСПЕШНЫХ вызовов stell_logits (rc==0; ошибочный
   вызов не даёт payload — тот же словарь, что и в serve-секции:
   requests_total + errors_total = полное число вызовов);
   last_n_vocab / last_positions — параметры последнего УСПЕШНОГО вызова,
   до первого успеха — null. */
int  stell_metrics(stell_engine_t* e, char** out_json);

/* Round-2 расширение (ADR-11 §1: «поля/символы только добавляются» —
   ABI-совместимо, прецедент stell_step): логиты финального слоя (ДО softmax)
   на готовых token id — quality-гейты (PPL) считают logprob по позициям.
   request_json: {"prompt_tokens":[...], "positions":"last"|"all"}
     — "positions" по умолчанию "last";
     — "last" — логиты последней позиции промпта;
     — "all" — по всем позициям промпта (teacher-forcing за один вызов).
   out_json: {"logits":[[float,...],...], "n_positions":P, "n_vocab":V}
     — P = 1 ("last") или число токенов промпта ("all"); V = vocab_size.
   Возврат: 0 — успех; -1 — ошибка (out_json = {"error":"..."}; runner:"demo"
   логиты не поддерживает — нужен runner:"ops"). Строки malloc — free() на
   вызывающем (конвенция ADR-11 §1 для malloc-строк). */
int  stell_logits(stell_engine_t* e, const char* request_json, char** out_json);

/* round 8 (срез 4): спекулятивный декод на реальном runner'е (ADR-11 §1 —
   ABI-совместимое расширение, новый символ). Драфтер — модель-плагин из
   конфига "spec" (spec.draft_model); верификация — FP32-логиты цели;
   последовательность на выходе побитово == plain greedy (спекуляция без
   потери качества). Требует runner:"ops" и секцию "spec".
   request_json: {"prompt_tokens":[...], "n_steps":K} (n_steps — 16 по
   умолчанию; K — сколько токенов декодировать в этом вызове).
   out_json: {"tokens":[...], "accepted":N, "proposed":M, "rounds":R,
              "emitted":E, "wall_s":S, "tok_s":T, "enabled":bool}
     — tok_s — честная скорость спек-декода (на CPU-контуре верификация
     последовательна: ускорения может не быть — арбитр калибровки решает).
   Возврат: 0 — успех; -1 — ошибка (out_json = {"error":"..."}). */
int  stell_spec_step(stell_engine_t* e, const char* request_json, char** out_json);

#ifdef __cplusplus
}
#endif

#endif /* STELL_ENGINE_CAPI_H */
