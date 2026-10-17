/*
 * capi.h — зеркало контракта ADR-11 (engine ↔ serve), C ABI.
 *
 * ВНИМАНИЕ: авторитетная копия заголовка живёт в src/engine (поток A).
 * Этот файл — зеркальная копия для mock-движка и Python-биндинга, чтобы
 * serve не зависел от чужого каталога. Контракт менять нельзя — только
 * через ADR и согласование с worker.
 *
 * События и запросы — JSON-строки через границу ABI (ADR-11 §1..§3).
 * Владение памятью: строки, возвращённые через char** (msg_out, out_json),
 * выделены malloc() и освобождаются вызывающей стороной. Для удобства
 * mock экспортирует необязательный помощник stell_free() (расширение ABI
 * только в сторону добавления; реальный engine может его не предоставлять —
 * биндинг тогда использует free() из libc того же процесса).
 */
#ifndef STELL_SERVE_MOCK_CAPI_H
#define STELL_SERVE_MOCK_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct stell_engine stell_engine_t; /* непрозрачный handle */

/* Lifecycle: cfg — JSON-строка (конфиг: модель, tier-границы, parallel, профили).
 * Возвращает 0 при успехе, <0 при ошибке (out не тронут). */
int  stell_engine_create(const char* cfg_json, stell_engine_t** out);
void stell_engine_destroy(stell_engine_t* e);

/* Постановка запроса в планировщик (ADR-11 §2). Возвращает id сессии (>0);
 * 0 — ошибка (msg_out — текст ошибки). msg_out всегда выделяется
 * (на успехе — пустая строка) и освобождается вызывающим. */
uint64_t stell_submit(stell_engine_t* e, const char* request_json, char** msg_out);

/* Неблокирующее выкачивание событий сессии (ADR-11 §3).
 * Возвращает число событий (>0), 0 — событий нет, <0 — сессия закрыта/ошибка.
 * out_json — JSON-массив событий [{"seq":1,"type":"token",...}, ...]. */
int  stell_poll(stell_engine_t* e, uint64_t session_id, char** out_json);

/* Отмена: сессия выбрасывается из батча; следующий poll вернёт
 * finish с reason="cancelled". 0 — ок, <0 — сессия неизвестна. */
int  stell_cancel(stell_engine_t* e, uint64_t session_id);

/* Метрики/health — JSON-снапшот (ADR-11 §4). 0 — ок, <0 — ошибка. */
int  stell_metrics(stell_engine_t* e, char** out_json);

/* (расширение mock) Освободить строку, возвращённую этим движком. */
void stell_free(char* p);

#ifdef __cplusplus
}
#endif

#endif /* STELL_SERVE_MOCK_CAPI_H */
