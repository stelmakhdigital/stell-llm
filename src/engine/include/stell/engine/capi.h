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

/* Отмена: сессия выбрасывается из батча на ближайшем шаге (ADR-11 §3). */
int  stell_cancel(stell_engine_t* e, uint64_t session_id);

/* Round-1 расширение (однопоточный CPU-контур; ADR-11 §1: «поля только
   добавляются», новый символ — ABI-совместимо): один шаг декода батча.
   В прод-контуре шаги ведёт внутренний поток engine; serve вызывает
   stell_step, пока нет фоновой нити (фиксируется в отчёте потока core). */
void stell_step(stell_engine_t* e);

/* Метрики/health — JSON (ADR-11 §4): снапшот, потокобезопасно;
   free() обязателен. */
int  stell_metrics(stell_engine_t* e, char** out_json);

#ifdef __cplusplus
}
#endif

#endif /* STELL_ENGINE_CAPI_H */
