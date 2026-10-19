/* Публикация C ABI (round 2, ADR-11 §1): smoke-тест dlopen libstell_core.so.
   Символы экспортированы (visibility default); serve подключит через
   ctypes/dlopen. Тест НЕ линкует stell_engine (проверяется именно загрузка .so
   и жизненный цикл через границу ABI). */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>
#include "test_util.hpp"

typedef struct stell_engine stell_engine_t;
extern int  stell_engine_create(const char* cfg_json, stell_engine_t** out);
extern void stell_engine_destroy(stell_engine_t* e);
extern uint64_t stell_submit(stell_engine_t* e, const char* request_json, char** msg_out);
extern int  stell_poll(stell_engine_t* e, uint64_t session_id, char** out_json);
extern int  stell_cancel(stell_engine_t* e, uint64_t session_id);
extern void stell_step(stell_engine_t* e);
extern int  stell_metrics(stell_engine_t* e, char** out_json);

static void run_steps(void* lib) {
    int (*poll_fn)(stell_engine_t*, uint64_t, char**) =
        (int (*)(stell_engine_t*, uint64_t, char**))dlsym(lib, "stell_poll");
    void (*step_fn)(stell_engine_t*) = (void (*)(stell_engine_t*))dlsym(lib, "stell_step");
    uint64_t (*submit_fn)(stell_engine_t*, const char*, char**) =
        (uint64_t (*)(stell_engine_t*, const char*, char**))dlsym(lib, "stell_submit");
    void (*destroy_fn)(stell_engine_t*) = (void (*)(stell_engine_t*))dlsym(lib, "stell_engine_destroy");
    int (*cancel_fn)(stell_engine_t*, uint64_t) =
        (int (*)(stell_engine_t*, uint64_t))dlsym(lib, "stell_cancel");
    int (*metrics_fn)(stell_engine_t*, char**) =
        (int (*)(stell_engine_t*, char**))dlsym(lib, "stell_metrics");

    STELL_CHECK(poll_fn && step_fn && submit_fn && destroy_fn && cancel_fn && metrics_fn);

    stell_engine_t* e = NULL;
    int (*create_fn)(const char*, stell_engine_t**) =
        (int (*)(const char*, stell_engine_t**))dlsym(lib, "stell_engine_create");
    STELL_CHECK(create_fn != NULL);
    STELL_CHECK(create_fn("{}", &e) == 0);
    STELL_CHECK(e != NULL);

    char* msg = NULL;
    uint64_t id = submit_fn(e, "{\"prompt_tokens\":[1,2,3],"
                             "\"sampling\":{\"max_tokens\":2,\"temperature\":0},"
                             "\"eos_token\":-1}", &msg);
    if (msg) free(msg);
    STELL_CHECK(id != 0);

    for (int i = 0; i < 2; ++i) step_fn(e);
    char* out = NULL;
    const int n = poll_fn(e, id, &out);
    STELL_CHECK(n >= 0);
    STELL_CHECK(out != NULL);
    STELL_CHECK(strstr(out, "\"type\"") != NULL);
    STELL_CHECK(strstr(out, "\"text\"") != NULL || strstr(out, "token") != NULL);
    free(out);

    char* m = NULL;
    STELL_CHECK(metrics_fn(e, &m) == 0);
    STELL_CHECK(strstr(m, "sessions_active") != NULL);
    free(m);

    STELL_CHECK(cancel_fn(e, id) == -1);   // сессия завершена — cancel → -1
                                           // (round 2: 0=успех, -1=не найдена/закрыта)
    destroy_fn(e);
}

int main() {
    const char* lib_path = STELL_CORE_LIB;
    void* lib = dlopen(lib_path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "[FAIL] dlopen(%s): %s\n", lib_path, dlerror());
        return 1;
    }
    run_steps(lib);
    dlclose(lib);
    puts("[PASS]");
    return 0;
}
