// Минимальный JSON-парсер engine (round 1): объекты/массивы/строки/числа/
// bool/null. Полного стандарта нет: escapes строк не поддержаны (serve
// поставляет токены как числа; текстовые поля — ADR-11 §3 — идут из событий
// наружу, не внутрь). Заменяется nlohmann/json при вклейке моделей (ADR-09 §1
// выбрал nlohmann — реестр принесёт зависимость).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stell::engine {

struct Json {
    enum class T { Null, Bool, Number, String, Array, Object };
    T t = T::Null;
    double num = 0.0;
    bool boolean = false;
    std::string str;
    std::vector<Json> arr;
    // порядок ключей сохранён (детерминизм вывода метрик)
    std::vector<std::pair<std::string, Json>> obj;

    const Json* find(const std::string& key) const {
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    // число как integer (для токен-id, лимитов)
    std::int64_t as_int(std::int64_t dflt = 0) const {
        return t == T::Number ? static_cast<std::int64_t>(num) : dflt;
    }
    std::string as_string(const std::string& dflt = "") const {
        return t == T::String ? str : dflt;
    }
};

// Парсинг: бросает std::runtime_error с позиционной диагностикой.
Json json_parse(const std::string& src);

// Сериализация минимального Json (для stell_metrics / stell_poll; round 1).
std::string json_dump(const Json& v);

} // namespace stell::engine
