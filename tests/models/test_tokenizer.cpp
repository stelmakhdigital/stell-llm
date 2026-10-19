// ============================================================================
// Тесты byte-level BPE токенизатора (round 2, задача R10; ADR-11 §3 / план
// Q7: реализация ITokenizer в models, инжект в Scheduler::set_tokenizer).
//
// Фикстуры (сгенерированы ЗАРАНЕЕ в env эталонного Python-стека — env hf
// репозитория, см. golden/generate_tokenizer_fixture.py — и закоммичены;
// сеть не использовалась):
//   fixtures/tokenizer/tokenizer.json      — BPE-модель (алфавит 256 байт-
//                                             символов + merges; формат
//                                             tokenizer.json);
//   fixtures/tokenizer/tokenizer_ref.json  — эталонные id фиксированного
//                                             набора ASCII-строк от
//                                             эталонная реализация —
//                                             см. meta в tokenizer_ref.json).
// Проверки:
//   - расхождение encode с эталоном на этих строках = 0;
//   - идемпотентность encode→decode = 100% на том же наборе;
//   - decode одного id == текст токена, конкатенация — вся строка;
//   - виртуальный вызов через stell::engine::ITokenizer (инжект в engine);
//   - честные ошибки загрузки (нет файла, не-BYTE pretokenizer, added tokens).
// ============================================================================
#include "stell/models/tokenizer.hpp"
#include "stell/models/json.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"

using namespace stell::models;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

void body() {
    const std::string tok_path =
        std::string(STELL_FIXTURES_DIR) + "/tokenizer/tokenizer.json";
    const std::string ref_path =
        std::string(STELL_FIXTURES_DIR) + "/tokenizer/tokenizer_ref.json";

    const ByteLevelBpeTokenizer tok = ByteLevelBpeTokenizer::load(tok_path);
    CHECK(tok.vocab_size() == 350);  // 256 алфавит + 94 merges (fixture)
    CHECK(tok.n_merges() == 94);

    const JsonValue ref = json_parse(read_file(ref_path));
    CHECK(ref.get_string("kind") == "tokenizer_ref");
    const std::vector<JsonValue>& strings = ref.get_array("strings").array_v;
    const std::vector<JsonValue>& ids_ref = ref.get_array("ids").array_v;
    CHECK(strings.size() == ids_ref.size());
    CHECK(!strings.empty());

    // 1) Расхождение с эталонным токенизатором (meta в tokenizer_ref.json) = 0
    //    и 2) идемпотентность encode→decode = 100%.
    std::size_t n_mismatch = 0, n_nonidempotent = 0, n_tokens_total = 0;
    for (std::size_t i = 0; i < strings.size(); ++i) {
        const std::string& s = strings[i].string_v;
        const std::vector<JsonValue>& want = ids_ref[i].array_v;

        const std::vector<std::int32_t> got = tok.encode(s);
        n_tokens_total += got.size();
        // Расхождение id = 0.
        if (got.size() != want.size()) {
            ++n_mismatch;
        } else {
            for (std::size_t k = 0; k < got.size(); ++k)
                if (got[k] != want[k].int_v) ++n_mismatch;
        }
        // Идемпотентность: decode(encode(s)) == s.
        if (tok.decode(got) != s) ++n_nonidempotent;
    }
    CHECK_MSG(n_mismatch == 0, "расхождение id с эталоном должно быть 0 (R10)");
    CHECK_MSG(n_nonidempotent == 0,
              "идемпотентность encode→decode = 100% (R10)");
    CHECK(n_tokens_total > 0);

    // 3) Decode одного id: текст токена; пустой на неизвестном id.
    CHECK(tok.decode(-1).empty());
    CHECK(tok.decode(100000).empty());
    {
        const std::vector<std::int32_t> ids = tok.encode("hello world");
        std::string joined;
        for (std::int32_t id : ids) joined += tok.decode(id);
        CHECK(joined == "hello world");
    }

    // 4) Виртуальный вызов через интерфейс engine (инжект Scheduler::
    //    set_tokenizer): ITokenizer* → те же результаты.
    {
        const stell::engine::ITokenizer* iface = &tok;
        const std::vector<std::int32_t> ids = iface->encode("stell-llm");
        CHECK(iface->decode(ids) == "stell-llm");
        CHECK(iface->decode(ids.at(0)) == tok.decode(ids.at(0)));
    }

    // 5) Честные ошибки загрузки (тишина запрещена).
    auto expect_load_error = [&](const std::string& json_text,
                                 const char* what) {
        const std::string p = "/tmp/stell_test_tokenizer_bad.json";
        {
            std::ofstream f(p, std::ios::binary);
            f << json_text;
        }
        try {
            ByteLevelBpeTokenizer::load(p);
            CHECK_MSG(false, what);
        } catch (const TokenizerError&) {
            CHECK(true);
        }
    };
    // Отсутствует pretokenizer.
    expect_load_error(R"({"model":{"vocab":{"a":0},"merges":[]}})",
                      "missing pretokenizer rejected");
    // Не-BYTE pretokenizer.
    expect_load_error(
        R"({"pre_tokenizer":{"type":"Whitespace"},"model":{"vocab":{"a":0},"merges":[]},"decoder":{"type":"ByteLevel"}})",
        "non-BYTE pretokenizer rejected");
    // ByteLevel use_regex=true (регэксп-сплиты вне subset).
    expect_load_error(
        R"({"pre_tokenizer":{"type":"ByteLevel","use_regex":true},"model":{"vocab":{"a":0},"merges":[]},"decoder":{"type":"ByteLevel"}})",
        "use_regex=true rejected");
    // Added/special tokens.
    expect_load_error(
        R"({"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":0}],"model":{"vocab":{"a":0},"merges":[]}})",
        "added tokens rejected");
    // unk_token.
    expect_load_error(
        R"({"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},"model":{"vocab":{"a":0},"merges":[],"unk_token":"<unk>"}})",
        "unk_token rejected");
    // Некорректный merge.
    expect_load_error(
        R"({"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},"model":{"vocab":{"a":0},"merges":[["a"]]}})",
        "bad merge entry rejected");
    // Файл отсутствует → TokenizerError (не крэш).
    try {
        ByteLevelBpeTokenizer::load("/tmp/stell_no_such_tokenizer_dir_x/tokenizer.json");
        CHECK_MSG(false, "missing file rejected");
    } catch (const TokenizerError&) {
        CHECK(true);
    }

    std::printf("tokenizer: %d строк fixture, %d токенов, ref-mismatch=0, "
                "non-idempotent=0\n",
                static_cast<int>(strings.size()),
                static_cast<int>(n_tokens_total));
}

TEST_MAIN("test_models_tokenizer")
