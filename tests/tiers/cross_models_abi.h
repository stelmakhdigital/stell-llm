// C-ABI стука кросс-модульного теста tiers↔models (ADR-12).
//
// ВЫНУЖДЕННОСТЬ отдельного ABI: публичные заголовки models тянут локальное
// зеркало типов ADR-08 (src/models/include/stell/models/tiers_compat.hpp),
// несовместимое в одной единице трансляции с реальным
// stell/tiers/types.hpp (enum class Role/DType декларированы по-разному —
// ODR-нарушение, проверено компиляцией). Поэтому models-сторона живёт в
// отдельной TU (cross_models_shim.cpp) и наружу выставляет плоский C-фасад.
// При вклейке (ADR-09: «заменить зеркало на реальный types.hpp») шим можно
// будет слить с тестом в одну TU.
#ifndef CROSS_MODELS_ABI_H
#define CROSS_MODELS_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Плоская запись плана (коды — ADR-12 §2.2: role 0=attn,1=expert,2=kv;
   dtype 0=F16,1=F32,2=E4M3,3=NVFP4,4=W4A16). */
struct CMGroupInfo {
    uint32_t id;
    uint8_t  role;
    uint8_t  dtype;
    uint64_t slot_bytes;
    uint32_t shape_rank;
    uint32_t shape[4];
    char     name[32];
};

/* Строит синтетический safetensors + конвертирует его в stell-tier
   конвертером models (StellTierConverter, ADR-12 §1/§4) для двух фикстур
   реестра: which 0 = mini_dense.json (F16), 1 = mini_moe.json (NVFP4).
   Пишет пути layout'ов в dense_path/moe_path (caped). Возвращает 0 при
   успехе, -1 при ошибке (детали — cm_last_error()). */
int cm_build(const char* fixtures_dir, const char* out_dir,
             char* dense_path, int dense_cap, char* moe_path, int moe_cap);

/* Число некешевых групп плана (порядок = порядок записей индекса layout'а). */
int cm_plan_size(int which);

/* Метаданные группы плана (idx в порядке индекса). 0 при успехе, -1 — нет. */
int cm_plan_group(int which, int idx, struct CMGroupInfo* out);

/* Хранимый header_crc32 файла (байты 0x2C..0x2F little-endian). */
uint32_t cm_stored_header_crc(const char* path);

/* Чтение слота группы читателем models (stelltier_read_header_and_index +
   fread; проверяет header CRC/выравнивания/монотонность). 0 при успехе. */
int cm_read_slot(const char* path, uint32_t group_id, uint8_t* dst, size_t cap);

/* Последнее сообщение об ошибке шима (для честного отчёта теста). */
const char* cm_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* CROSS_MODELS_ABI_H */
