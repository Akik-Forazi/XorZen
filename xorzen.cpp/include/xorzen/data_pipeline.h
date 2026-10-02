#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <memory>
#include <vector>

#include "xorzen/tokenizer.h"

namespace xorzen::optimized {
class MMapRegion;
}

namespace xorzen {

struct TextDataPipelineConfig {
    int64_t sequence_length = 128;
    int64_t stride = 0;
    bool add_special_tokens = true;
    bool recursive = true;
};

std::vector<std::filesystem::path> discover_text_files(const std::filesystem::path& root,
                                                       bool recursive = true);

class BEBPETextDataset : public torch::data::datasets::Dataset<BEBPETextDataset> {
public:
    BEBPETextDataset(std::vector<std::filesystem::path> paths,
                     BEBPETokenizer tokenizer,
                     TextDataPipelineConfig config = {});

    torch::data::Example<> get(size_t index) override;
    std::optional<size_t> size() const override;

    size_t token_count() const { return tokens_.size(); }
    const TextDataPipelineConfig& config() const { return config_; }

private:
    void append_file(const std::filesystem::path& path);
    size_t start_for_index(size_t index) const;

    BEBPETokenizer tokenizer_;
    TextDataPipelineConfig config_;
    std::vector<int64_t> tokens_;
    size_t sample_count_ = 0;
};

class DataConverter {
public:
    explicit DataConverter(BEBPETokenizer tokenizer);

    struct ConversionStats {
        size_t lines_read = 0;
        size_t tokens_written = 0;
        size_t sequences_created = 0;
        size_t file_size_bytes = 0;
    };

    ConversionStats txt_to_bin(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        int64_t max_length = 0,
        int64_t stride = 0,
        bool add_special_tokens = true
    );

private:
    BEBPETokenizer tokenizer_;
};

class MMapDataset : public torch::data::datasets::Dataset<MMapDataset> {
public:
    MMapDataset(const std::filesystem::path& bin_path, int64_t seq_len);
    torch::data::Example<> get(size_t index) override;
    std::optional<size_t> size() const override;

private:
    int64_t seq_len_;
    int64_t record_len_;
    int64_t num_sequences_;
    size_t file_size_;
    size_t element_size_;
    bool is_uint16_;
    std::shared_ptr<optimized::MMapRegion> mmap_region_;
    const void* mmap_ptr_;
    std::vector<std::byte> owned_bytes_;
};

} // namespace xorzen
