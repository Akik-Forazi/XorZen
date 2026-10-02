#include "xorzen/tokenizer.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::filesystem::path resolve_tokenizer_path(int argc, char** argv) {
    if (argc > 1) return argv[1];

    const std::vector<std::filesystem::path> candidates = {
        ".xorzen_python_READ-ONLY/xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        "../.xorzen_python_READ-ONLY/xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        "../../.xorzen_python_READ-ONLY/xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        "../../xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        "../xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        "xorzen/tokenizer/pretrained/zero_bpe_10k.json"
    };
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) return path;
    }
    throw std::runtime_error("could not locate zero_bpe_10k.json; pass its path as argv[1]");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto tokenizer_path = resolve_tokenizer_path(argc, argv);
        const auto start = std::chrono::steady_clock::now();
        auto tokenizer = xorzen::BEBPETokenizer::from_tokenizer_json(tokenizer_path);
        auto encoding = tokenizer.encode("Hello world! XORZEN BEBPE tokenizer smoke.", true);
        auto decoded = tokenizer.decode(encoding.input_ids, true);
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        if (tokenizer.vocab_size() <= 0 || encoding.input_ids.empty() ||
            decoded.find("Hello") == std::string::npos) {
            std::cerr << "BEBPE tokenizer smoke failed\n";
            return 2;
        }

        std::cout << "vocab=" << tokenizer.vocab_size()
                  << " tokens=" << encoding.input_ids.size()
                  << " ms=" << elapsed
                  << " text=" << decoded << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tokenizer smoke error: " << e.what() << std::endl;
        return 1;
    }
}
