#include <xorzen/data_pipeline.h>
#include <xorzen/tokenizer.h>
#include <xorzen/types.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_failures = 0;

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << __FILE__ << ":" << __LINE__ << " EXPECT_TRUE failed: " #cond "\n"; \
        ++g_failures; \
    } \
} while (0)

#define EXPECT_EQ(lhs, rhs) do { \
    const auto& _lhs = (lhs); \
    const auto& _rhs = (rhs); \
    if (!(_lhs == _rhs)) { \
        std::cerr << __FILE__ << ":" << __LINE__ << " EXPECT_EQ failed: " #lhs " == " #rhs "\n"; \
        ++g_failures; \
    } \
} while (0)

static fs::path locate_tokenizer() {
    const std::vector<fs::path> candidates = {
        fs::current_path() / ".xorzen_python_READ-ONLY/xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        fs::current_path() / "../.xorzen_python_READ-ONLY/xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        fs::current_path() / "../../.xorzen_python_READ-ONLY/xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        fs::current_path() / "xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        fs::current_path() / "../xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        fs::current_path() / "../../xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        fs::current_path() / "xorzen.cpp/xorzen/tokenizer/pretrained/zero_bpe_10k.json",
        fs::current_path() / "../xorzen.cpp/xorzen/tokenizer/pretrained/zero_bpe_10k.json",
    };
    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) return fs::weakly_canonical(candidate);
    }
    throw std::runtime_error("could not locate zero_bpe_10k.json");
}

static fs::path make_temp_dir(const std::string& name) {
    auto dir = fs::temp_directory_path() / ("xorzen_cpp_tests_" + name);
    fs::create_directories(dir);
    return dir;
}

static void test_tokenizer_roundtrip() {
    auto tokenizer_path = locate_tokenizer();
    auto tokenizer = xorzen::BEBPETokenizer::from_tokenizer_json(tokenizer_path);
    auto encoding = tokenizer.encode("Hello from xorzen.", true);
    EXPECT_TRUE(!encoding.input_ids.empty());
    EXPECT_TRUE(tokenizer.vocab_size() > 0);
    auto decoded = tokenizer.decode(encoding.input_ids);
    EXPECT_TRUE(!decoded.empty());
}

static void test_data_converter_and_mmap() {
    auto tokenizer_path = locate_tokenizer();
    auto tokenizer = xorzen::BEBPETokenizer::from_tokenizer_json(tokenizer_path);

    auto temp_dir = make_temp_dir("converter");
    auto txt_path = temp_dir / "sample.txt";
    auto bin_path = temp_dir / "sample.bin";

    {
        std::ofstream out(txt_path);
        out << "Hello xorzen cpp tests.\n";
        out << "This is another line for the dataset.\n";
        out << "Tokenization should work.\n";
    }

    xorzen::DataConverter converter(tokenizer);
    auto stats = converter.txt_to_bin(txt_path, bin_path, 8, 8, true);
    EXPECT_TRUE(stats.lines_read > 0);
    EXPECT_TRUE(stats.tokens_written > 0);
    EXPECT_TRUE(fs::exists(bin_path));
    EXPECT_TRUE(fs::exists(bin_path.string() + ".meta.json"));

    xorzen::MMapDataset dataset(bin_path, 8);
    auto size = dataset.size();
    EXPECT_TRUE(size.has_value());
    EXPECT_TRUE(size.value() > 0);

    auto sample = dataset.get(0);
    EXPECT_EQ(sample.data.dim(), 1);
    EXPECT_EQ(sample.target.dim(), 1);
    EXPECT_EQ(sample.data.size(0), 8);
    EXPECT_EQ(sample.target.size(0), 8);
}

static void test_text_dataset() {
    auto tokenizer_path = locate_tokenizer();
    auto tokenizer = xorzen::BEBPETokenizer::from_tokenizer_json(tokenizer_path);

    auto temp_dir = make_temp_dir("text");
    auto txt_path = temp_dir / "sample.txt";
    {
        std::ofstream out(txt_path);
        out << "Text dataset smoke test for xorzen.\n";
        out << "Sequence windows should line up.\n";
    }

    xorzen::TextDataPipelineConfig cfg;
    cfg.sequence_length = 8;
    cfg.stride = 8;
    xorzen::BEBPETextDataset dataset({txt_path}, tokenizer, cfg);
    auto size = dataset.size();
    EXPECT_TRUE(size.has_value());
    EXPECT_TRUE(size.value() > 0);

    auto sample = dataset.get(0);
    EXPECT_EQ(sample.data.dim(), 1);
    EXPECT_EQ(sample.target.dim(), 1);
    EXPECT_EQ(sample.data.size(0), 8);
    EXPECT_EQ(sample.target.size(0), 8);
}

static void test_total_loss_accessor() {
    xorzen::ModelOutput output;
    output.loss = torch::tensor(3.5f);
    output.routing_loss = torch::tensor(1.25f);
    output.load_balance_loss = torch::tensor(0.75f);
    output.cot_consistency_loss = torch::tensor(0.5f);

    auto total = output.total_loss();
    EXPECT_TRUE(torch::isclose(total, torch::tensor(3.5f)).item<bool>());
}

int main() {
    try {
        test_tokenizer_roundtrip();
        test_text_dataset();
        test_data_converter_and_mmap();
        test_total_loss_accessor();
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << "\n";
        return 1;
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }

    std::cout << "xorzen cpp tests passed\n";
    return 0;
}
