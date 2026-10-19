// Профильный вариант лоадера NVFP4-W4A16 (ADR-03: квантованное хранение —
// вариант лоадера/конвертера, не отдельный входной формат). Читает
// safetensors или GGUF (F16/F32) и квантуется в целевой dtype группы
// на read(); упаковочные профили — weights.hpp / quant.hpp (блок 32).
#include "tensor_file_formats.hpp"
#include "quant.hpp"

#include <cstring>
#include <functional>

namespace stell::models {
namespace {

class QuantizingSource : public FileWeightsSource {
public:
    QuantizingSource(std::string path, ModelDescription desc)
        : FileWeightsSource(std::move(path), std::move(desc)) {
        const std::string& p = this->desc().dtype_weights;
        if (p != "NVFP4" && p != "W4A16")
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "nvfp4w4a16 loader: профиль '" + p +
                                  "' не является квантованным (ADR-03)");
    }

protected:
    TensorFileMap parse_header() override {
        const WeightsFormat fmt = detect_format(path());
        switch (fmt) {
            case WeightsFormat::Safetensors: return parse_safetensors_header(file());
            case WeightsFormat::GGUF: return parse_gguf_header(file());
            default:
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "nvfp4w4a16 loader: входной формат должен быть "
                                  "safetensors/GGUF (ADR-03)");
        }
    }

    void do_fill(const stell::tiers::TensorGroup& g, const TensorInfo& t,
                 void* dst, std::size_t dst_bytes) override {
        check_shape(g, t);
        // Плоские группы (F16/F32) профиля — fill_plain (round 2: профильный
        // лоадер обслуживает ОПИСАНИЕ целиком; round-1 бросал UnsupportedQuant
        // на неквантованные группы, что не позволяло загрузить модель одним
        // лоадером — исправлено, см. отчёт раунда).
        if (g.dtype != stell::tiers::DType::NVFP4 &&
            g.dtype != stell::tiers::DType::W4A16) {
            if (!compatible_plain(g.dtype, t))
                throw LoaderError(
                    LoaderError::Code::UnsupportedQuant,
                    "nvfp4w4a16 loader: группа '" + g.name + "' (dtype " +
                        dtype_name(g.dtype) + "): источник '" + t.dtype_str +
                        "' не совместим (плоский путь)");
            fill_plain(g, t, dst, dst_bytes);
            return;
        }
        // Источник → f32: F16/F32-тензора напрямую; GGUF Q4_0/Q8_0 — dequant
        // (round 2: квантованные GGUF-блоки подхвачены NVFP4/W4A16-профилем).
        std::size_t n = numel(g.shape);
        std::vector<float> src(n);
        if (is_quantized_ggml(t.dtype_str)) {
            gguf_dequant_tensor(file(), t, src);
        } else {
            if (!t.dtype || (*t.dtype != stell::tiers::DType::F16 &&
                             *t.dtype != stell::tiers::DType::F32))
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "nvfp4w4a16 loader: источник должен быть F16/F32/BF16, получено '" +
                                      t.dtype_str + "'");
            if (*t.dtype == stell::tiers::DType::F16) {
                std::vector<std::uint8_t> raw(t.bytes);
                read_tensor_payload(t, raw.data());
                const bool bf16 = t.bf16;  // round 4: BF16-флаг (dtype маппится в F16)
                for (std::size_t i = 0; i < n; ++i) {
                    std::uint16_t h = static_cast<std::uint16_t>(raw[2 * i]) |
                                      (static_cast<std::uint16_t>(raw[2 * i + 1]) << 8);
                    src[i] = bf16 ? bf16_bits_to_f32(h) : f16_bits_to_f32(h);
                }
            } else {
                read_tensor_payload(t, src.data());
            }
        }
        // Квантование в payload группы; хвост слота (выравнивание) = 0.
        std::uint8_t* out = static_cast<std::uint8_t*>(dst);
        if (g.dtype == stell::tiers::DType::NVFP4)
            quantize_nvfp4(src.data(), n, out);
        else
            quantize_w4a16(src.data(), n, out);
        const std::size_t payload = payload_bytes(g.dtype, n);
        std::memset(out + payload, 0, dst_bytes - payload);
    }

private:
    static std::size_t numel(const std::vector<std::int64_t>& shape) {
        std::size_t n = 1;
        for (auto d : shape) n *= static_cast<std::size_t>(d);
        return n;
    }
    static std::size_t payload_bytes(stell::tiers::DType dt, std::size_t n) {
        if (dt == stell::tiers::DType::NVFP4) return (n + 1) / 2 + (n + 31) / 32;
        if (dt == stell::tiers::DType::W4A16) return (n + 1) / 2 + 2 * ((n + 31) / 32);
        return 0;
    }

    static const char* dtype_name(stell::tiers::DType d) {
        switch (d) {
            case stell::tiers::DType::F16: return "F16";
            case stell::tiers::DType::F32: return "F32";
            case stell::tiers::DType::E4M3: return "E4M3";
            case stell::tiers::DType::NVFP4: return "NVFP4";
            case stell::tiers::DType::W4A16: return "W4A16";
        }
        return "?";
    }
};

}  // namespace

class NVFP4W4A16Loader : public IWeightsLoader {
public:
    WeightsFormat format() const override { return WeightsFormat::NVFP4W4A16; }
    std::unique_ptr<IWeightsSource> open(const std::string& uri,
                                         const ModelDescription& desc) override {
        auto src = std::make_unique<QuantizingSource>(uri, desc);
        src->resolve_all();
        return src;
    }
};

IWeightsLoader* make_nvfp4w4a16_loader() { return new NVFP4W4A16Loader(); }

}  // namespace stell::models
