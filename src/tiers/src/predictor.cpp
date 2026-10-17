// Router-lookahead prefetch-предиктор: марковская модель 1-го порядка.
#include <stell/tiers/predictor.hpp>

#include <algorithm>

namespace stell::tiers {

MarkovPredictor::MarkovPredictor(std::size_t history_cap)
    : history_cap_(history_cap ? history_cap : 1) {}

void MarkovPredictor::observe(GroupId prev, GroupId next) {
    if (prev == next) return; // самопереходы не информативны для lookahead
    if (observed_ >= history_cap_) {
        // Простое кольцо: при переполнении гасим половину старых счётчиков.
        for (auto& kv : counts_)
            for (auto& nv : kv.second)
                nv.second /= 2;
        observed_ /= 2;
    }
    ++counts_[prev.value][next.value];
    ++observed_;
}

std::vector<GroupId> MarkovPredictor::predict(GroupId last, std::size_t k) const {
    auto it = counts_.find(last.value);
    if (it == counts_.end() || k == 0) return {};
    std::vector<std::pair<GroupId, std::uint64_t>> ranked;
    ranked.reserve(it->second.size());
    for (const auto& kv : it->second) ranked.push_back({GroupId{kv.first}, kv.second});
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::vector<GroupId> out;
    for (std::size_t i = 0; i < ranked.size() && i < k; ++i)
        out.push_back(ranked[i].first);
    return out;
}

} // namespace stell::tiers
