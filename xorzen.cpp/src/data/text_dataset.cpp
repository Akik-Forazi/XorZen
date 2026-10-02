#include "xorzen/data_pipeline.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace xorzen {
namespace {

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open text dataset file: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool is_text_like(const std::filesystem::path& path) {
    const auto ext = path.extension().string();
    return ext == ".txt" || ext == ".md" || ext == ".jsonl" || ext == ".csv" ||
           ext == ".tsv" || ext == ".text";
}

} // namespace

std::vector<std::filesystem::path> discover_text_files(const std::filesystem::path& root,
                                                       bool recursive) {
    std::vector<std::filesystem::path> files;
    if (!std::filesystem::exists(root)) return files;
    if (std::filesystem::is_regular_file(root)) {
        if (is_text_like(root)) files.push_back(root);
        return files;
    }

    if (recursive) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.is_regular_file() && is_text_like(entry.path())) files.push_back(entry.path());
        }
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_regular_file() && is_text_like(entry.path())) files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

BEBPETextDataset::BEBPETextDataset(std::vector<std::filesystem::path> paths,
                                   BEBPETokenizer tokenizer,
                                   TextDataPipelineConfig config)
    : tokenizer_(std::move(tokenizer)), config_(config) {
    if (config_.sequence_length <= 0) throw std::invalid_argument("sequence_length must be positive");
    if (config_.stride <= 0) config_.stride = config_.sequence_length;
    if (paths.empty()) throw std::invalid_argument("BEBPETextDataset needs at least one text file");

    for (const auto& path : paths) append_file(path);
    const auto required = static_cast<size_t>(config_.sequence_length + 1);
    if (tokens_.size() < required) {
        throw std::runtime_error("text dataset has fewer tokens than one training sequence");
    }
    sample_count_ = 1 + (tokens_.size() - required) / static_cast<size_t>(config_.stride);
}

torch::data::Example<> BEBPETextDataset::get(size_t index) {
    if (index >= sample_count_) throw std::out_of_range("BEBPETextDataset index out of range");
    const auto start = start_for_index(index);
    const auto seq = static_cast<size_t>(config_.sequence_length);
    const auto* input_ptr = tokens_.data() + static_cast<std::ptrdiff_t>(start);
    const auto* target_ptr = tokens_.data() + static_cast<std::ptrdiff_t>(start + 1);
    auto input = torch::from_blob(const_cast<int64_t*>(input_ptr),
                                  {static_cast<long>(seq)},
                                  torch::TensorOptions().dtype(torch::kLong)).clone();
    auto target = torch::from_blob(const_cast<int64_t*>(target_ptr),
                                   {static_cast<long>(seq)},
                                   torch::TensorOptions().dtype(torch::kLong)).clone();
    return {std::move(input), std::move(target)};
}

std::optional<size_t> BEBPETextDataset::size() const {
    return sample_count_;
}

void BEBPETextDataset::append_file(const std::filesystem::path& path) {
    auto text = read_text_file(path);
    auto encoding = tokenizer_.encode(text, config_.add_special_tokens);
    tokens_.insert(tokens_.end(), encoding.input_ids.begin(), encoding.input_ids.end());
}

size_t BEBPETextDataset::start_for_index(size_t index) const {
    return index * static_cast<size_t>(config_.stride);
}

} // namespace xorzen
