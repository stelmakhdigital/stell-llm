// KV-streaming для длинного контекста: кольцевой буфер + бюджет локальных
// блоков с вытеснением в поток (ADR-04 dtype, ADR-10 KvStream-семантика).
#include "test_util.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "stell/engine/kv_stream.hpp"

using namespace stell::engine::kv;

void run() {
    // --- кольцевая очередь: FIFO, mismatch, pending --------------------------
    KvRingBuffer ring;
    STELL_CHECK(ring.pending() == 0);
    ring.push("abc", 3);
    ring.push("de", 2);
    STELL_CHECK(ring.pending() == 2);

    char buf[16]{};
    STELL_CHECK(ring.pop(buf, 3));
    STELL_CHECK(std::string(buf, 3) == "abc");
    STELL_CHECK(ring.pending() == 1);

    // несовпадение размера — контракт нарушен вызывающим, false (ADR-10 §2)
    STELL_CHECK(!ring.pop(buf, 5));
    STELL_CHECK(ring.pending() == 1);         // блок не потерян

    STELL_CHECK(ring.pop(buf, 2));
    STELL_CHECK(std::string(buf, 2) == "de");
    STELL_CHECK(!ring.pop(buf, 2));           // пусто
    STELL_CHECK(ring.pending() == 0);

    // --- KvStreamer: бюджет локальных блоков + вытеснение в поток ------------
    KvRingBuffer sink;
    KvStreamer streamer(sink, 8 /*block_bytes*/, 2 /*budget*/);

    std::vector<std::string> blocks = {"block-00", "block-01", "block-02",
                                      "block-03", "block-04"};
    for (std::size_t i = 0; i < blocks.size(); ++i)
        streamer.push_block(0 /*slot*/, static_cast<std::uint32_t>(i), blocks[i].data());

    // локально осталось 2 блока (последние), в поток уехали 3
    STELL_CHECK(streamer.local_blocks() == 2);
    STELL_CHECK(streamer.streamed_blocks() == 3);
    STELL_CHECK(sink.pending() == 3);

    // выгрузка из потока — в порядке отправки (FIFO)
    char out[8]{};
    for (int i = 0; i < 3; ++i) {
        STELL_CHECK(streamer.pull_block(out));
        std::string got(out, 8);
        STELL_CHECK_MSG(got == blocks[i], "KV-блоки выгружаются в порядке seq");
    }
    STELL_CHECK(!streamer.pull_block(out));   // поток пуст
    STELL_CHECK(sink.pending() == 0);
}

STELL_TEST_MAIN(run)
