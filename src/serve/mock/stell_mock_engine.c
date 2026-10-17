/*
 * stell_mock_engine.c — CPU-fallback mock-движок с контрактом ADR-11.
 *
 * Зачем mock на C, а не Python-fake за интерфейсом: serve связывается с
 * движком строго через ctypes/dlopen поверх C ABI (ADR-11 §1). Python-fake
 * за интерфейсом не проверяет самую хрупкую часть контракта — проход
 * JSON-строк через границу C ABI, владение malloc-памятью (char**), коды
 * возврата submit/poll. Маленькая C-библиотека (~450 строк) проверяет ровно
 * это; сборка — cmake target `stell_mock_core` (src/serve/CMakeLists.txt).
 *
 * Поведение mock (детерминированное, для контрактных тестов):
 *   cfg_json:  {"model":"...","mock":{"slots_total":16,"ttft_ms":1.0,
 *               "tpot_ms":14.0,"prefill_tok_s":250.0,"token_text":" hi",
 *               "tool_name":"get_weather","tool_arguments":"{...}",
 *               "fail_create":bool,"fail_submit":bool}}
 *   request:   {"session":{"max_tokens":N,...},"tools":[...],
 *               "messages":[...],"stream":bool,"mock":{...}}
 *   события:   token xN → tool_call? → finish(stop|length|tool_call|
 *               cancelled, usage); после finish poll возвращает -1.
 *   tool_call эмитится, если в request непуст "tools", либо принудительно
 *   через request.mock.tool_call=true.
 */
#include "capi.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* ===================== tiny JSON parser (subset) ===================== */

typedef enum { JN_NULL, JN_BOOL, JN_NUM, JN_STR, JN_ARR, JN_OBJ } jtype;

typedef struct jnode {
    jtype type;
    int boolean;
    double number;
    char *str;              /* JN_STR (owned) */
    struct jnode **items;   /* JN_ARR элементы / JN_OBJ значения (owned) */
    char **keys;            /* JN_OBJ ключи (owned) */
    size_t count;
} jnode;

typedef struct jcur { const char *s; size_t i; int ok; } jcur;

static void jskip_ws(jcur *c) {
    while (c->s[c->i] == ' ' || c->s[c->i] == '\t' ||
           c->s[c->i] == '\n' || c->s[c->i] == '\r')
        c->i++;
}

static jnode *jnew(jtype t) {
    jnode *n = (jnode *)calloc(1, sizeof(jnode));
    if (n) n->type = t;
    return n;
}

static void jfree(jnode *n) {
    if (!n) return;
    if (n->type == JN_STR) free(n->str);
    for (size_t k = 0; k < n->count; k++) {
        jfree(n->items[k]);
        if (n->keys) free(n->keys[k]);
    }
    free(n->items);
    free(n->keys);
    free(n);
}

/* строка: открывающая кавычка уже под курсором; возвращает malloc-строку */
static char *jparse_str_raw(jcur *c) {
    const char *s = c->s;
    size_t i = c->i + 1; /* за открывающей кавычкой */
    size_t cap = 16, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) { c->ok = 0; return NULL; }
    for (;;) {
        char ch = s[i];
        if (ch == '\0') { c->ok = 0; free(out); return NULL; }
        if (ch == '"') break;
        char mapped = 0;
        int advance = 1;
        if (ch == '\\') {
            char e = s[i + 1];
            if (e == '\0') { c->ok = 0; free(out); return NULL; }
            switch (e) {
                case '"': case '\\': case '/': mapped = e; advance = 2; break;
                case 'n': mapped = '\n'; advance = 2; break;
                case 't': mapped = '\t'; advance = 2; break;
                case 'r': mapped = '\r'; advance = 2; break;
                case 'b': mapped = '\b'; advance = 2; break;
                case 'f': mapped = '\f'; advance = 2; break;
                case 'u': mapped = '?'; advance = 6; break; /* \uXXXX не нужен mock */
                default: mapped = e; advance = 2; break;
            }
        } else {
            mapped = ch;
        }
        if (len + 2 > cap) {
            cap *= 2;
            char *no = (char *)realloc(out, cap);
            if (!no) { c->ok = 0; free(out); return NULL; }
            out = no;
        }
        out[len++] = mapped;
        i += advance;
    }
    out[len] = '\0';
    c->i = i + 1;
    return out;
}

static jnode *jparse_value(jcur *c);

static jnode *jparse_arr(jcur *c) {
    c->i++; /* [ */
    jnode *arr = jnew(JN_ARR);
    if (!arr) { c->ok = 0; return NULL; }
    size_t cap = 4;
    arr->items = (jnode **)malloc(cap * sizeof(jnode *));
    if (!arr->items) { jfree(arr); c->ok = 0; return NULL; }
    jskip_ws(c);
    if (c->s[c->i] == ']') { c->i++; return arr; }
    for (;;) {
        jnode *v = jparse_value(c);
        if (!c->ok || !v) { jfree(arr); c->ok = 0; return NULL; }
        if (arr->count == cap) {
            cap *= 2;
            jnode **ni = (jnode **)realloc(arr->items, cap * sizeof(jnode *));
            if (!ni) { jfree(arr); c->ok = 0; return NULL; }
            arr->items = ni;
        }
        arr->items[arr->count++] = v;
        jskip_ws(c);
        if (c->s[c->i] == ',') { c->i++; jskip_ws(c); continue; }
        if (c->s[c->i] == ']') { c->i++; return arr; }
        jfree(arr); c->ok = 0; return NULL;
    }
}

static jnode *jparse_obj(jcur *c) {
    c->i++; /* { */
    jnode *obj = jnew(JN_OBJ);
    if (!obj) { c->ok = 0; return NULL; }
    size_t cap = 4;
    obj->items = (jnode **)malloc(cap * sizeof(jnode *));
    obj->keys = (char **)malloc(cap * sizeof(char *));
    if (!obj->items || !obj->keys) { jfree(obj); c->ok = 0; return NULL; }
    jskip_ws(c);
    if (c->s[c->i] == '}') { c->i++; return obj; }
    for (;;) {
        jskip_ws(c);
        if (c->s[c->i] != '"') { jfree(obj); c->ok = 0; return NULL; }
        char *key = jparse_str_raw(c);
        if (!key) { jfree(obj); c->ok = 0; return NULL; }
        jskip_ws(c);
        if (c->s[c->i] != ':') { free(key); jfree(obj); c->ok = 0; return NULL; }
        c->i++;
        jnode *v = jparse_value(c);
        if (!c->ok || !v) { free(key); jfree(obj); c->ok = 0; return NULL; }
        if (obj->count == cap) {
            cap *= 2;
            jnode **ni = (jnode **)realloc(obj->items, cap * sizeof(jnode *));
            char **nk = (char **)realloc(obj->keys, cap * sizeof(char *));
            if (!ni || !nk) { jfree(obj); free(key); c->ok = 0; return NULL; }
            obj->items = ni; obj->keys = nk;
        }
        obj->keys[obj->count] = key;
        obj->items[obj->count] = v;
        obj->count++;
        jskip_ws(c);
        if (c->s[c->i] == ',') { c->i++; continue; }
        if (c->s[c->i] == '}') { c->i++; return obj; }
        jfree(obj); c->ok = 0; return NULL;
    }
}

static jnode *jparse_value(jcur *c) {
    jskip_ws(c);
    char ch = c->s[c->i];
    if (ch == '{') return jparse_obj(c);
    if (ch == '[') return jparse_arr(c);
    if (ch == '"') {
        jnode *n = jnew(JN_STR);
        if (!n) { c->ok = 0; return NULL; }
        n->str = jparse_str_raw(c);
        if (!n->str) { jfree(n); c->ok = 0; return NULL; }
        return n;
    }
    if (ch == 't' && strncmp(&c->s[c->i], "true", 4) == 0) {
        c->i += 4;
        jnode *n = jnew(JN_BOOL);
        if (!n) { c->ok = 0; return NULL; }
        n->boolean = 1;
        return n;
    }
    if (ch == 'f' && strncmp(&c->s[c->i], "false", 5) == 0) {
        c->i += 5;
        jnode *n = jnew(JN_BOOL);
        if (!n) { c->ok = 0; return NULL; }
        n->boolean = 0;
        return n;
    }
    if (ch == 'n' && strncmp(&c->s[c->i], "null", 4) == 0) {
        c->i += 4;
        return jnew(JN_NULL);
    }
    /* число */
    jnode *n = jnew(JN_NUM);
    if (!n) { c->ok = 0; return NULL; }
    char buf[64];
    size_t k = 0;
    if (ch == '-') { buf[k++] = c->s[c->i++]; }
    while (k < 63 &&
           ((c->s[c->i] >= '0' && c->s[c->i] <= '9') ||
            strchr(".eE", c->s[c->i]))) {
        buf[k++] = c->s[c->i++];
    }
    if (k == 0 || (k == 1 && buf[0] == '-')) { jfree(n); c->ok = 0; return NULL; }
    buf[k] = '\0';
    n->number = strtod(buf, NULL);
    return n;
}

static jnode *json_parse(const char *s) {
    jcur c = { s ? s : "", 0, 1 };
    jnode *v = jparse_value(&c);
    if (!c.ok) { jfree(v); return NULL; }
    return v;
}

static jnode *jobj_get(const jnode *obj, const char *key) {
    if (!obj || obj->type != JN_OBJ) return NULL;
    for (size_t k = 0; k < obj->count; k++)
        if (strcmp(obj->keys[k], key) == 0) return obj->items[k];
    return NULL;
}

static long jarr_len(const jnode *arr) {
    if (!arr || arr->type != JN_ARR) return -1;
    return (long)arr->count;
}

static int jnum(const jnode *n, int def) {
    return (n && n->type == JN_NUM) ? (int)n->number : def;
}

static const char *jstr(const jnode *n) {
    return (n && n->type == JN_STR) ? n->str : NULL;
}

/* ===================== строковый буфер для JSON-вывода ===================== */

typedef struct sbuf { char *p; size_t len, cap; } sbuf;

static int sb_init(sbuf *b) {
    b->cap = 512; b->len = 0;
    b->p = (char *)malloc(b->cap);
    if (!b->p) return -1;
    b->p[0] = '\0';
    return 0;
}

static int sb_grow(sbuf *b, size_t need) {
    if (b->len + need + 1 <= b->cap) return 0;
    size_t cap = b->cap;
    while (cap < b->len + need + 1) cap *= 2;
    char *np = (char *)realloc(b->p, cap);
    if (!np) return -1;
    b->p = np; b->cap = cap;
    return 0;
}

static void sb_putc(sbuf *b, char c) {
    if (sb_grow(b, 1)) return;
    b->p[b->len++] = c;
    b->p[b->len] = '\0';
}

static int sb_puts(sbuf *b, const char *s) {
    size_t n = strlen(s);
    if (sb_grow(b, n)) return -1;
    memcpy(b->p + b->len, s, n + 1);
    b->len += n;
    return 0;
}

static int sb_printf(sbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (needed < 0) return -1;
    if (sb_grow(b, (size_t)needed)) return -1;
    va_start(ap, fmt);
    vsnprintf(b->p + b->len, (size_t)needed + 1, fmt, ap);
    va_end(ap);
    b->len += (size_t)needed;
    return 0;
}

/* экранирование строки как JSON string value */
static int sb_json_str(sbuf *b, const char *s) {
    sb_putc(b, '"');
    for (const char *c = s; *c; c++) {
        if (*c == '"' || *c == '\\') { sb_printf(b, "\\%c", *c); continue; }
        if (*c == '\n') { sb_puts(b, "\\n"); continue; }
        if ((unsigned char)*c < 0x20) { sb_printf(b, "\\u%04x", *c); continue; }
        sb_putc(b, *c);
    }
    sb_putc(b, '"');
    return 0;
}

/* ===================== mock engine ===================== */

typedef struct mock_session {
    uint64_t id;
    int phase;                /* 0 = активна, 2 = закрыта (finish отдан) */
    int tokens_left;
    int max_tokens;
    int prompt_tokens;
    int completion_tokens;
    int tool_pending;
    int cancelled;
    uint64_t seq;             /* следующий seq события */
    char token_text[128];
    char tool_id[64];
    char tool_name[128];
    char tool_args[1024];
    struct mock_session *next;
} mock_session;

struct stell_engine {
    mock_session *sessions;
    uint64_t next_id;
    int slots_total;
    double ttft_ms, tpot_ms, prefill_tok_s;
    char token_text[128];
    char model[128];
    char version[32];
    char tool_name[128];
    char tool_args[1024];
    int fail_create;
    int fail_submit;
    int cancelled_total;      /* расширение метрик (additive-поле, ADR-11 §1: поля только добавляются) */
};

static void copy_str(char *dst, size_t cap, const char *src) {
    if (!src) return;
    strncpy(dst, src, cap - 1);
    dst[cap - 1] = '\0';
}

static int mock_cfg_apply(struct stell_engine *eng, const jnode *cfg) {
    const char *m = jstr(jobj_get(cfg, "model"));
    copy_str(eng->model, sizeof(eng->model), m ? m : "mock-model");
    const jnode *mock = jobj_get(cfg, "mock");
    if (!mock) return 0;
    const jnode *v;
    if ((v = jobj_get(mock, "slots_total"))) eng->slots_total = jnum(v, eng->slots_total);
    if ((v = jobj_get(mock, "ttft_ms"))) eng->ttft_ms = v->number;
    if ((v = jobj_get(mock, "tpot_ms"))) eng->tpot_ms = v->number;
    if ((v = jobj_get(mock, "prefill_tok_s"))) eng->prefill_tok_s = v->number;
    if ((v = jobj_get(mock, "token_text")) && jstr(v))
        copy_str(eng->token_text, sizeof(eng->token_text), jstr(v));
    if ((v = jobj_get(mock, "tool_name")) && jstr(v))
        copy_str(eng->tool_name, sizeof(eng->tool_name), jstr(v));
    if ((v = jobj_get(mock, "tool_arguments")) && jstr(v))
        copy_str(eng->tool_args, sizeof(eng->tool_args), jstr(v));
    if ((v = jobj_get(mock, "fail_create"))) eng->fail_create = v->boolean;
    if ((v = jobj_get(mock, "fail_submit"))) eng->fail_submit = v->boolean;
    return 0;
}

int stell_engine_create(const char *cfg_json, stell_engine_t **out) {
    if (!out) return -1;
    *out = NULL;
    jnode *cfg = json_parse(cfg_json);
    if (!cfg) return -1;
    struct stell_engine *eng =
        (struct stell_engine *)calloc(1, sizeof(struct stell_engine));
    if (!eng) { jfree(cfg); return -1; }
    eng->slots_total = 16;
    eng->ttft_ms = 1.0;
    eng->tpot_ms = 14.0;
    eng->prefill_tok_s = 250.0;
    copy_str(eng->token_text, sizeof(eng->token_text), " hi");
    copy_str(eng->tool_name, sizeof(eng->tool_name), "get_weather");
    copy_str(eng->tool_args, sizeof(eng->tool_args), "{\"city\":\"Moscow\"}");
    copy_str(eng->version, sizeof(eng->version), "0.1.0-mock");
    mock_cfg_apply(eng, cfg);
    jfree(cfg);
    if (eng->fail_create) { free(eng); return -1; }
    *out = eng;
    return 0;
}

void stell_engine_destroy(stell_engine_t *e) {
    struct stell_engine *eng = (struct stell_engine *)e;
    if (!eng) return;
    mock_session *s = eng->sessions;
    while (s) { mock_session *n = s->next; free(s); s = n; }
    free(eng);
}

static int sessions_active(struct stell_engine *eng) {
    int n = 0;
    for (mock_session *s = eng->sessions; s; s = s->next)
        if (s->phase != 2) n++;
    return n;
}

static mock_session *session_find(struct stell_engine *eng, uint64_t id) {
    for (mock_session *s = eng->sessions; s; s = s->next)
        if (s->id == id) return s;
    return NULL;
}

uint64_t stell_submit(stell_engine_t *e, const char *request_json, char **msg_out) {
    struct stell_engine *eng = (struct stell_engine *)e;
    if (msg_out) *msg_out = (char *)calloc(2, 1); /* "" по умолчанию */
    if (!msg_out || !*msg_out) return 0;
    if (eng->fail_submit) {
        free(*msg_out);
        *msg_out = strdup("mock: fail_submit requested");
        return 0;
    }
    jnode *req = json_parse(request_json);
    if (!req) {
        free(*msg_out);
        *msg_out = strdup("mock: unparsable request_json");
        return 0;
    }
    if (sessions_active(eng) >= eng->slots_total) {
        free(*msg_out);
        *msg_out = strdup("slot_overflow");
        jfree(req);
        return 0;
    }
    const jnode *session = jobj_get(req, "session");
    const jnode *messages = jobj_get(req, "messages");
    const jnode *tools = jobj_get(req, "tools");
    long n_msgs = jarr_len(messages) > 0 ? jarr_len(messages) : 0;
    long n_tools = jarr_len(tools) > 0 ? jarr_len(tools) : 0;
    int max_tokens = 16;
    if (session) max_tokens = jnum(jobj_get(session, "max_tokens"), 16);
    if (max_tokens < 1) max_tokens = 1;

    mock_session *s = (mock_session *)calloc(1, sizeof(mock_session));
    if (!s) { jfree(req); return 0; }
    s->id = ++eng->next_id;
    s->phase = 0;
    s->max_tokens = max_tokens;
    s->tokens_left = max_tokens;
    s->prompt_tokens = (int)(n_msgs * 4 + 2);
    s->seq = 1;
    s->tool_pending = (n_tools > 0) ? 1 : 0;
    copy_str(s->token_text, sizeof(s->token_text), eng->token_text);
    copy_str(s->tool_name, sizeof(s->tool_name), eng->tool_name);
    copy_str(s->tool_args, sizeof(s->tool_args), eng->tool_args);
    snprintf(s->tool_id, sizeof(s->tool_id), "call_%llu", (unsigned long long)s->id);

    /* переопределения на уровне запроса (request.mock.*) */
    const jnode *rm = jobj_get(req, "mock");
    if (rm) {
        const jnode *v;
        if ((v = jobj_get(rm, "tool_call"))) s->tool_pending = v->boolean;
        if ((v = jobj_get(rm, "tool_name")) && jstr(v))
            copy_str(s->tool_name, sizeof(s->tool_name), jstr(v));
        if ((v = jobj_get(rm, "tool_arguments")) && jstr(v))
            copy_str(s->tool_args, sizeof(s->tool_args), jstr(v));
        if ((v = jobj_get(rm, "token_text")) && jstr(v))
            copy_str(s->token_text, sizeof(s->token_text), jstr(v));
        if ((v = jobj_get(rm, "token_count"))) {
            int n = jnum(v, s->tokens_left);
            if (n >= 0 && n < s->tokens_left) s->tokens_left = n; /* естественный stop */
        }
    }
    s->next = eng->sessions;
    eng->sessions = s;
    jfree(req);
    return s->id;
}

int stell_poll(stell_engine_t *e, uint64_t session_id, char **out_json) {
    struct stell_engine *eng = (struct stell_engine *)e;
    mock_session *s = session_find(eng, session_id);
    if (!s || s->phase == 2) return -1;

    sbuf b;
    if (sb_init(&b)) return -1;
    sb_putc(&b, '[');
    int first = 1;
    int nevents = 0;

    if (s->cancelled) {
        /* finish cancelled: сессия выброшена из батча (ADR-11 §3) */
        sb_printf(&b, "{\"seq\":%llu,\"type\":\"finish\",\"reason\":\"cancelled\","
                      "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}",
                  (unsigned long long)s->seq++, s->prompt_tokens, s->completion_tokens);
        nevents = 1;
    } else {
        while (s->tokens_left > 0) {
            sb_printf(&b, "%s{\"seq\":%llu,\"type\":\"token\",\"text\":",
                      first ? "" : ",", (unsigned long long)s->seq++);
            sb_json_str(&b, s->token_text);
            sb_printf(&b, ",\"token_id\":%d}", 1000 + (int)s->seq);
            first = 0;
            s->tokens_left--;
            s->completion_tokens++;
            nevents++;
        }
        const char *reason;
        if (s->tool_pending) {
            sb_printf(&b, "%s{\"seq\":%llu,\"type\":\"tool_call\",\"id\":\"%s\","
                          "\"name\":\"%s\",\"arguments\":",
                      first ? "" : ",", (unsigned long long)s->seq++,
                      s->tool_id, s->tool_name);
            sb_json_str(&b, s->tool_args);
            sb_putc(&b, '}');
            reason = "tool_call";
            nevents++;
        } else {
            reason = (s->completion_tokens >= s->max_tokens) ? "length" : "stop";
        }
        sb_printf(&b, "%s{\"seq\":%llu,\"type\":\"finish\",\"reason\":\"%s\","
                      "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}",
                  first ? "" : ",", (unsigned long long)s->seq++, reason,
                  s->prompt_tokens, s->completion_tokens);
        nevents++;
    }
    sb_puts(&b, "]");
    s->phase = 2; /* finish отдан — дальнейшие poll = закрытая сессия */
    *out_json = b.p;
    return nevents;
}

int stell_cancel(stell_engine_t *e, uint64_t session_id) {
    struct stell_engine *eng = (struct stell_engine *)e;
    mock_session *s = session_find(eng, session_id);
    if (!s || s->phase == 2) return -1;
    s->cancelled = 1;
    eng->cancelled_total++;
    return 0;
}

int stell_metrics(stell_engine_t *e, char **out_json) {
    struct stell_engine *eng = (struct stell_engine *)e;
    sbuf b;
    if (sb_init(&b)) return -1;
    int active = sessions_active(eng);
    double decode_tok_s = eng->tpot_ms > 0 ? 1000.0 / eng->tpot_ms : 0.0;
    sb_printf(&b,
        "{\"engine\":{\"state\":\"serving\",\"cancelled_total\":%d,\"sessions_active\":%d,"
        "\"slots_busy\":%d,\"slots_total\":%d,\"batch_size_max\":%d},"
        "\"latency\":{\"ttft_ms\":{\"p50\":%.1f,\"p95\":%.1f,\"p99\":%.1f},"
        "\"tpot_ms\":{\"p50\":%.1f,\"p95\":%.1f,\"p99\":%.1f},"
        "\"tok_s\":{\"decode\":%.1f,\"prefill\":%.1f,\"spec_decode\":0.0}},"
        "\"tiers\":{\"prefetch\":{\"hit_rate\":0.63},"
        "\"layer_pause_ms\":{\"p50\":2.1,\"p95\":4.4,\"p99\":6.0},"
        "\"tier_bytes\":[8589934592,0,1099511627776]},"
        "\"parallel\":{\"tp\":1,\"ep\":1,\"nccl\":false},"
        "\"features\":{\"spec_draft\":false,\"kv_quant\":\"F16\"},"
        "\"build\":{\"version\":\"%s\",\"cuda\":\"none (CPU mock)\",\"sm\":\"cpu\"},"
        "\"model\":\"%s\"}",
        eng->cancelled_total, active, active, eng->slots_total, eng->slots_total,
        eng->ttft_ms, eng->ttft_ms * 1.3, eng->ttft_ms * 1.6,
        eng->tpot_ms, eng->tpot_ms * 1.3, eng->tpot_ms * 1.6,
        decode_tok_s, eng->prefill_tok_s,
        eng->version, eng->model);
    *out_json = b.p;
    return 0;
}

void stell_free(char *p) { free(p); }
