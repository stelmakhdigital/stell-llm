#include "stell/models/description.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace stell::models {
namespace {

std::size_t numel(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) {
        if (d <= 0) throw DescriptionError("shape entries must be positive");
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

std::size_t align_up(std::size_t v, std::size_t a) {
    return (v + a - 1) / a * a;
}

std::string ctx(const std::string& source, const std::string& what) {
    return source.empty() ? what : (source + ": " + what);
}

[[noreturn]] void fail(const std::string& source, const std::string& msg) {
    throw DescriptionError(ctx(source, msg));
}

// Замена токена "{key}": имя группы — с ведущими нулями до 3 цифр
// ("l012.e003"), source_pattern — сырые десятичные значения.
std::string expand_tokens(const std::string& text,
                          const std::vector<std::pair<std::string, std::uint32_t>>& values,
                          bool pad3) {
    std::string out = text;
    for (const auto& [key, v] : values) {
        char token[32], value[16];
        std::snprintf(token, sizeof token, "{%s}", key.c_str());
        if (pad3) std::snprintf(value, sizeof value, "%03u", v);
        else std::snprintf(value, sizeof value, "%u", v);
        std::string::size_type pos = 0;
        while ((pos = out.find(token)) != std::string::npos)
            out.replace(pos, std::strlen(token), value);
    }
    return out;
}

OpSpec parse_op(const JsonValue& v, const std::string& source, std::size_t idx) {
    std::string where = std::to_string(idx);
    try {
        OpSpec op;
        op.op = v.get_string("op");
        if (op.op.empty()) throw DescriptionError("empty op name");
        for (const auto& io : v.get_array("io").array_v) {
            if (io.type != JsonValue::Type::String) throw DescriptionError("io entries must be strings");
            op.io.push_back(io.string_v);
        }
        const JsonValue* params = v.find("params");
        if (params) {
            if (params->type != JsonValue::Type::Object) throw DescriptionError("params must be an object");
            for (const auto& kv : params->object_v) op.params.push_back(kv);
        }
        return op;
    } catch (const JsonError& e) {
        throw DescriptionError(ctx(source, "ops[" + where + "]: " + e.what()));
    }
}

TensorGroupTemplate parse_group_template(const JsonValue& v, const std::string& source,
                                         std::size_t idx) {
    std::string where = std::to_string(idx);
    try {
        TensorGroupTemplate t;
        t.name = v.get_string("name");
        t.role = v.get_string("role");
        t.dtype = v.get_string("dtype");
        t.source_pattern = v.get_string("source_pattern");

        for (const auto& dim : v.get_array("shape").array_v) {
            if (dim.type != JsonValue::Type::Int || dim.int_v <= 0)
                throw DescriptionError("shape entries must be positive ints");
            t.shape.push_back(dim.int_v);
        }

        const JsonValue* kvb = v.find("kv_block");
        if (kvb) {
            if (kvb->type != JsonValue::Type::Bool) throw DescriptionError("kv_block must be bool");
            t.kv_block = kvb->bool_v;
        }

        // source_slice: шаблон с токенами ("{expert}") или целочисленный литерал
        // (round 2, R9: стековные экспертные тензоры GGUF-mixtral).
        if (const JsonValue* slice = v.find("source_slice")) {
            if (slice->type == JsonValue::Type::String) {
                t.source_slice = slice->string_v;
            } else if (slice->type == JsonValue::Type::Int) {
                t.source_slice = std::to_string(slice->int_v);
            } else {
                throw DescriptionError("source_slice must be a string or int");
            }
        }

        const JsonValue* card = v.find("cardinality");
        if (card) {
            if (card->type != JsonValue::Type::Object) throw DescriptionError("cardinality must be an object");
            for (const auto& kv : card->object_v) {
                if (kv.second.type != JsonValue::Type::Int || kv.second.int_v <= 0)
                    throw DescriptionError("cardinality values must be positive ints");
                t.cardinality.emplace_back(kv.first, static_cast<std::uint32_t>(kv.second.int_v));
            }
        }
        return t;
    } catch (const JsonError& e) {
        throw DescriptionError(ctx(source, "tensor_groups[" + where + "]: " + e.what()));
    }
}

void validate_template(const TensorGroupTemplate& t, const ModelDescription& d,
                       const std::string& source) {
    try {
        parse_role(t.role);
    } catch (const DescriptionError&) {
        fail(source, "tensor_groups: unknown role '" + t.role + "'");
    }
    try {
        parse_dtype(t.dtype);
    } catch (const DescriptionError&) {
        fail(source, "tensor_groups: unknown dtype '" + t.dtype + "'");
    }
    // Каждый ключ cardinality обязан встречаться токеном в name и в
    // source_pattern ИЛИ в source_slice (привязка разметки к лоадеру, ADR-09
    // §1; round 2, R9: ключ экспертов стековного GGUF-тензора живёт в
    // source_slice, а не в имени тензора).
    for (const auto& [key, n] : t.cardinality) {
        std::string token = "{" + key + "}";
        if (t.name.find(token) == std::string::npos)
            fail(source, "tensor_groups '" + t.name + "': cardinality key '" + key +
                             "' absent in name template");
        if (t.source_pattern.find(token) == std::string::npos &&
            t.source_slice.find(token) == std::string::npos)
            fail(source, "tensor_groups '" + t.name + "': cardinality key '" + key +
                             "' absent in source_pattern/source_slice");
    }
    // KV-группы не имеют весов в файле — их pattern привязан к кешу, не к лоадеру.
    if (t.kv_block && t.role != "kv")
        fail(source, "tensor_groups '" + t.name + "': kv_block=true допустим только для role=kv");
    if (d.is_moe && !t.cardinality.empty()) {
        for (const auto& [key, n] : t.cardinality) {
            if (key == "layer" && n != d.n_layers)
                fail(source, "tensor_groups '" + t.name + "': cardinality layer=" +
                                 std::to_string(n) + " != n_layers=" + std::to_string(d.n_layers));
            if (key == "expert" && d.moe && n != d.moe->n_experts)
                fail(source, "tensor_groups '" + t.name + "': cardinality expert=" +
                                 std::to_string(n) + " != moe.n_experts=" +
                                 std::to_string(d.moe->n_experts));
        }
    }
}

void expand_groups(ModelDescription& d, const std::string& source) {
    std::uint32_t id = 0;
    for (const auto& t : d.group_templates) {
        validate_template(t, d, source);
        const std::size_t n_dims = t.cardinality.size();
        std::size_t total = 1;
        for (const auto& [key, n] : t.cardinality) total *= n;

        for (std::size_t iter = 0; iter < total; ++iter) {
            std::vector<std::pair<std::string, std::uint32_t>> instance(n_dims);
            // Смешанная система счисления: последний ключ cardinality меняется быстрее всего.
            std::size_t rem = iter;
            for (std::size_t k = n_dims; k-- > 0;) {
                instance[k] = {t.cardinality[k].first,
                               static_cast<std::uint32_t>(rem % t.cardinality[k].second)};
                rem /= t.cardinality[k].second;
            }
            std::string name = expand_tokens(t.name, instance, /*pad3=*/true);
            std::string pattern = expand_tokens(t.source_pattern, instance, /*pad3=*/false);
            stell::tiers::TensorGroup g;
            g.id.value = id++;
            g.name = name;
            g.role = parse_role(t.role);
            g.dtype = parse_dtype(t.dtype);
            g.shape = t.shape;
            g.slot_bytes = slot_bytes_for(t.dtype, t.shape);
            d.groups.push_back(std::move(g));
            d.group_source_patterns.push_back(std::move(pattern));
            // Срез стекового тензора (round 2, R9): токены заменяются сырыми
            // значениями; пустой шаблон → −1 (тензор целиком).
            if (t.source_slice.empty()) {
                d.group_source_slices.push_back(-1);
            } else {
                const std::string slice = expand_tokens(t.source_slice, instance, false);
                try {
                    d.group_source_slices.push_back(std::stoi(slice));
                } catch (const std::exception&) {
                    fail(source, "tensor_groups '" + t.name + "': source_slice '" +
                                     slice + "' is not an integer");
                }
            }
        }
    }
}

}  // namespace

stell::tiers::Role parse_role(const std::string& role) {
    if (role == "attn") return stell::tiers::Role::Attn;
    if (role == "expert") return stell::tiers::Role::Expert;
    if (role == "kv") return stell::tiers::Role::Kv;
    throw DescriptionError("unknown role '" + role + "'");
}

stell::tiers::DType parse_dtype(const std::string& dtype) {
    if (dtype == "F16") return stell::tiers::DType::F16;
    if (dtype == "F32") return stell::tiers::DType::F32;
    if (dtype == "E4M3") return stell::tiers::DType::E4M3;
    if (dtype == "NVFP4") return stell::tiers::DType::NVFP4;
    if (dtype == "W4A16") return stell::tiers::DType::W4A16;
    throw DescriptionError("unknown dtype '" + dtype + "'");
}

std::size_t payload_bytes_for(const std::string& dtype, const std::vector<std::int64_t>& shape) {
    const std::size_t n = numel(shape);
    if (dtype == "F16") return n * 2;
    if (dtype == "F32") return n * 4;
    if (dtype == "E4M3") return n;
    if (dtype == "NVFP4") return (n + 1) / 2 + (n + 31) / 32;         // нибблы + E4M3 scale/блок
    if (dtype == "W4A16") return (n + 1) / 2 + 2 * ((n + 31) / 32);   // нибблы + F16 scale/блок
    throw DescriptionError("unknown dtype '" + dtype + "'");
}

std::size_t slot_bytes_for(const std::string& dtype, const std::vector<std::int64_t>& shape) {
    return align_up(payload_bytes_for(dtype, shape), 4096);
}

std::uint32_t dtype_profile_code(const std::string& dtype_weights) {
    if (dtype_weights == "F16") return 1;
    if (dtype_weights == "NVFP4") return 2;
    if (dtype_weights == "W4A16") return 3;
    throw DescriptionError("unknown dtype_weights profile '" + dtype_weights + "'");
}

ModelDescription parse_model_description(const std::string& json_text,
                                         const std::string& source_name) {
    JsonValue root;
    try {
        root = json_parse(json_text);
    } catch (const JsonError& e) {
        throw DescriptionError(ctx(source_name, e.what()));
    }
    if (root.type != JsonValue::Type::Object)
        fail(source_name, "top-level JSON value must be an object");

    ModelDescription d;
    try {
        d.schema = root.get_string("schema");
        {
            // Мажор версии схемы обязателен: "1.x" совместим, "2.x" — нет (ADR-09 §1).
            std::size_t dot = d.schema.find('.');
            std::string major = dot == std::string::npos ? d.schema : d.schema.substr(0, dot);
            if (major != "1")
                fail(source_name, "unsupported schema version '" + d.schema + "' (expected 1.x)");
        }
        d.name = root.get_string("name");
        std::string klass = root.get_string("class");
        if (klass == "moe") d.is_moe = true;
        else if (klass != "dense") fail(source_name, "class must be \"dense\" or \"moe\"");
        d.dtype_weights = root.get_string("dtype_weights");
        dtype_profile_code(d.dtype_weights);  // валидация профиля (ADR-03)
        d.vocab_size = root.get_uint("vocab_size");
        d.d_model = root.get_uint("d_model");
        d.n_layers = static_cast<std::uint32_t>(root.get_uint("n_layers"));
        d.n_heads = static_cast<std::uint32_t>(root.get_uint("n_heads"));
        d.d_head = static_cast<std::uint32_t>(root.get_uint("d_head"));

        const JsonValue& kv = root.get_object("kv");
        d.kv.dtype = kv.get_string("dtype");
        d.kv.n_kv_heads = static_cast<std::uint32_t>(kv.get_uint("n_kv_heads"));
        d.kv.d_kv = static_cast<std::uint32_t>(kv.get_uint("d_kv"));
        const JsonValue* quant = kv.find("quant");
        if (quant && quant->type == JsonValue::Type::String) d.kv.quant = quant->string_v;
        else if (quant && quant->type != JsonValue::Type::Null)
            fail(source_name, "kv.quant must be null or string");

        if (d.is_moe) {
            const JsonValue& moe = root.get_object("moe");
            MoeSpec m;
            m.n_experts = static_cast<std::uint32_t>(moe.get_uint("n_experts"));
            m.n_experts_active = static_cast<std::uint32_t>(moe.get_uint("n_experts_active"));
            m.expert_d_ff = static_cast<std::uint32_t>(moe.get_uint("expert_d_ff"));
            m.router = moe.get_string("router");
            if (m.n_experts == 0 || m.n_experts_active == 0 || m.expert_d_ff == 0)
                fail(source_name, "moe.n_experts/n_experts_active/expert_d_ff must be positive");
            if (m.n_experts_active > m.n_experts)
                fail(source_name, "moe.n_experts_active > n_experts");
            d.moe = m;
        }

        // [R2] round 7 (ADR-09 round 7): опциональное top-уровневое поле
        // router_renorm (bool; отсутствие — дефолт true, прежние описания
        // читаются без изменений). class=dense роутера не имеет — явный
        // false недопустим (честный отказ, не тихое игнорирование поля).
        if (const JsonValue* rr = root.find("router_renorm"); rr) {
            if (rr->type != JsonValue::Type::Bool)
                fail(source_name, "router_renorm must be a bool");
            if (!rr->bool_v && !d.is_moe)
                fail(source_name,
                     "router_renorm=false при class=dense: роутера нет — "
                     "поле неприменимо (честный отказ)");
            if (d.is_moe) d.moe->router_renorm = rr->bool_v;
        }

        for (const auto& op : root.get_array("ops").array_v) {
            if (op.type != JsonValue::Type::Object) fail(source_name, "ops entries must be objects");
            d.ops.push_back(parse_op(op, source_name, d.ops.size()));
        }
        if (d.ops.empty()) fail(source_name, "ops must be non-empty");

        for (const auto& g : root.get_array("tensor_groups").array_v) {
            if (g.type != JsonValue::Type::Object) fail(source_name, "tensor_groups entries must be objects");
            d.group_templates.push_back(parse_group_template(g, source_name, d.group_templates.size()));
        }
        if (d.group_templates.empty()) fail(source_name, "tensor_groups must be non-empty");
    } catch (const JsonError& e) {
        throw DescriptionError(ctx(source_name, e.what()));
    }

    expand_groups(d, source_name);
    return d;
}

}  // namespace stell::models
