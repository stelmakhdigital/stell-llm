#include "stell/engine/ngram_drafter.hpp"

#include <algorithm>

namespace stell::engine {

namespace {

// FNV-1a (32-битные токены → байты), длина окна подмешана в ключ —
// окна разных порядков не сталкиваются в одной корзине.
std::uint64_t window_key(const std::vector<std::int32_t>& tokens,
                         std::size_t begin, std::size_t length) {
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t byte) {
        h ^= byte;
        h *= 1099511628211ull;
    };
    for (std::size_t i = begin; i < begin + length; ++i) {
        const std::uint32_t t = static_cast<std::uint32_t>(tokens[i]);
        mix(t & 0xFFull);
        mix((t >> 8) & 0xFFull);
        mix((t >> 16) & 0xFFull);
        mix((t >> 24) & 0xFFull);
    }
    mix(static_cast<std::uint64_t>(length));
    return h;
}

} // namespace

NgramDrafter::NgramDrafter(std::size_t max_order, std::size_t max_context)
    : max_order_(std::max<std::size_t>(max_order, 1)),
      max_context_(std::max<std::size_t>(max_context, max_order_ + 1)) {}

void NgramDrafter::append_token(std::int32_t token) {
    // Окно не больше max_order_ и не длиннее имеющегося текста.
    const std::size_t n = text_.size();
    const std::size_t max_l = std::min(max_order_, n);
    for (std::size_t l = 1; l <= max_l; ++l) {
        // Окно — последние l токенов (индексы n-l..n-1); прогноз — text_[i+1]
        // записывается для позиции i = n-1.
        const std::uint64_t key = window_key(text_, n - l, l);
        index_[key].push_back(static_cast<std::uint32_t>(n - 1));
    }
    text_.push_back(token);
}

void NgramDrafter::ensure_indexed(
    const std::vector<std::int32_t>& context) {
    // Монотонный рост (нормальный случай декодера): доклеиваем дельту.
    if (context.size() >= text_.size() &&
        std::equal(text_.begin(), text_.end(), context.begin())) {
        for (std::size_t i = text_.size(); i < context.size(); ++i)
            append_token(context[i]);
        // Честное окно: лимит размера — старые токены выпадают (перестройка).
        if (text_.size() > max_context_) {
            text_.assign(context.end() -
                             static_cast<std::ptrdiff_t>(max_context_ / 2),
                         context.end());
            index_.clear();
            const std::vector<std::int32_t> rebuilt = text_;
            text_.clear();
            text_.push_back(rebuilt.front());
            for (std::size_t i = 1; i < rebuilt.size(); ++i)
                append_token(rebuilt[i]);
        }
        return;
    }
    // Не-монотонный контекст (новая сессия/другой слот): полная перестройка.
    index_.clear();
    text_.clear();
    if (!context.empty()) {
        text_.push_back(context.front());
        for (std::size_t i = 1; i < context.size(); ++i)
            append_token(context[i]);
    }
}

SpeculativeDecoder::DrafterFn NgramDrafter::fn() {
    return [this](const std::vector<std::int32_t>& context,
                  std::size_t n_draft) {
        if (context.empty() || n_draft == 0) return std::vector<std::int32_t>{};
        ensure_indexed(context);
        if (text_.size() < 2) return std::vector<std::int32_t>{};

        std::vector<std::int32_t> out;
        out.reserve(n_draft);
        // Виртуальное продолжение: суффиксы, захватывающие уже испущенные
        // в ЭТОМ вызове кандидаты, в индексе отсутствуют — честный фолбэк
        // на короткое L (кандидаты не подтверждены — не индексируются).
        std::vector<std::int32_t> full = context;   // контекст + драфты
        for (std::size_t produced = 0; produced < n_draft; ++produced) {
            const std::size_t len = full.size();
            std::int32_t next = -1;
            for (std::size_t l = std::min(max_order_, len); l >= 1; --l) {
                const std::uint64_t key =
                    window_key(full, len - l, l);   // хвост текущего текста
                const auto it = index_.find(key);
                if (it == index_.end() || it->second.empty()) continue;
                // Самая свежая позиция, у которой ЕСТЬ следующий токен
                // (детерминизм: обход от конца списка).
                bool found = false;
                for (auto rit = it->second.rbegin(); rit != it->second.rend();
                     ++rit) {
                    if (*rit + 1 < text_.size()) {
                        next = text_[*rit + 1];
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (next < 0) break;   // совпадений нет — пустой остаток (честно)
            out.push_back(next);
            full.push_back(next);
        }
        return out;
    };
}

} // namespace stell::engine
