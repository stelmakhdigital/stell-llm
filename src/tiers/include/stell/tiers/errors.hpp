// ADR-08 §3: единственный механизм ошибок tiers — исключения.
#pragma once

#include <stdexcept>
#include <string>

namespace stell::tiers {

class TierError : public std::runtime_error {  // базовая; код — для логов/дашборда
public:
    enum class Code { PlanMismatch, SlotExhausted, IoFailure, NotPinned, Timeout };
    TierError(Code c, const std::string& what)
        : std::runtime_error(what), code_(c) {}
    Code code() const noexcept { return code_; }

private:
    Code code_;
};

} // namespace stell::tiers
