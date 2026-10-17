// Драфтер-ядра (спекулятивный контур engine): детерминированные выборки по
// логитам. CPU-референс; CUDA-версия — драфтер-ядро (sm70 drafter).
#pragma once

#include <cstddef>
#include <cstdint>

namespace stell::kernels {

// argmax (FP32). Требует n > 0.
std::int32_t argmax_f32(const float* logits, std::size_t n);

// Top-k выбор кандидатов драфта по логитам (softmax внутри):
//   out_ids    — [k] индексы логитов по убыванию вероятности
//   out_probs  — [k] вероятности (softmax по всему vocab, ограниченная top-k)
// Требует k <= n.
void draft_topk_f32(const float* logits, std::size_t n, std::uint32_t k,
                    std::int32_t* out_ids, float* out_probs);

} // namespace stell::kernels
