#include "xorzen/data_pipeline.h"
#include "xorzen/optimized/expert_mmap.h"

#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace xorzen {

namespace {

std::filesystem::path meta_path_for(const std::filesystem::path& bin_path) {
    auto meta_path = bin_path;
    meta_path.replace_extension(".meta.json");
    return meta_path;
}

std::string read_all(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open dataset metadata: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string json_string_value(const std::string& text,
                              const std::string& key,
                              const std::string& fallback) {
    const auto needle = "\"" + key + "\"";
    auto pos = text.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    pos = text.find('"', pos);
    if (pos == std::string::npos) return fallback;
    auto end = text.find('"', pos + 1);
    if (end == std::string::npos) return fallback;
    return text.substr(pos + 1, end - pos - 1);
}

int64_t json_i64_value(const std::string& text, const std::string& key, int64_t fallback) {
    const auto needle = "\"" + key + "\"";
    auto pos = text.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    auto end = text.find_first_of(",}\r\n", pos);
    try {
        return std::stoll(text.substr(pos, end == std::string::npos ? end : end - pos));
    } catch (...) {
        return fallback;
    }
}

} // namespace

MMapDataset::MMapDataset(const std::filesystem::path& bin_path, int64_t seq_len)
    : seq_len_(seq_len),
      record_len_(seq_len + 1),
      num_sequences_(0),
      file_size_(0),
      element_size_(sizeof(uint16_t)),
      is_uint16_(true),
      mmap_ptr_(nullptr) {
    if (seq_len_ <= 0) {
        throw std::invalid_argument("MMapDataset sequence length must be positive");
    }

    auto meta_path = meta_path_for(bin_path);
    if (std::filesystem::exists(meta_path)) {
        const auto meta = read_all(meta_path);
        const auto dtype = json_string_value(meta, "dtype", "uint16");
        is_uint16_ = (dtype == "uint16");
        element_size_ = is_uint16_ ? sizeof(uint16_t) : sizeof(uint32_t);
        record_len_ = json_i64_value(meta, "record_length", seq_len_ + 1);
        if (record_len_ < seq_len_ + 1) {
            record_len_ = seq_len_ + 1;
        }
    } else {
        element_size_ = sizeof(uint16_t);
        record_len_ = seq_len_ + 1;
    }

    file_size_ = std::filesystem::file_size(bin_path);
    if (file_size_ == 0) {
        throw std::runtime_error("dataset file is empty: " + bin_path.string());
    }

#ifdef XORZEN_ENABLE_MMAP
    auto region = std::make_shared<optimized::MMapRegion>();
    if (!region->open(bin_path, true)) {
        throw std::runtime_error("failed to mmap dataset: " + bin_path.string());
    }
    mmap_ptr_ = region->data();
    mmap_region_ = std::move(region);
#else
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open dataset file: " + bin_path.string());
    }
    owned_bytes_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    mmap_ptr_ = owned_bytes_.data();
#endif

    const auto total_tokens = file_size_ / element_size_;
    if (total_tokens < static_cast<size_t>(record_len_)) {
        throw std::runtime_error("dataset does not contain a full training record: " + bin_path.string());
    }
    num_sequences_ = static_cast<int64_t>(total_tokens / static_cast<size_t>(record_len_));
}

torch::data::Example<> MMapDataset::get(size_t index) {
    if (index >= static_cast<size_t>(num_sequences_)) {
        throw std::out_of_range("MMapDataset index out of range");
    }

    const auto* base = static_cast<const std::byte*>(mmap_ptr_) +
                       index * static_cast<size_t>(record_len_) * element_size_;
    std::vector<int64_t> input;
    std::vector<int64_t> target;
    input.reserve(static_cast<size_t>(seq_len_));
    target.reserve(static_cast<size_t>(seq_len_));

    if (is_uint16_) {
        const auto* tokens = reinterpret_cast<const uint16_t*>(base);
        for (int64_t i = 0; i < seq_len_; ++i) {
            input.push_back(static_cast<int64_t>(tokens[i]));
            target.push_back(static_cast<int64_t>(tokens[i + 1]));
        }
    } else {
        const auto* tokens = reinterpret_cast<const uint32_t*>(base);
        for (int64_t i = 0; i < seq_len_; ++i) {
            input.push_back(static_cast<int64_t>(tokens[i]));
            target.push_back(static_cast<int64_t>(tokens[i + 1]));
        }
    }

    return {torch::tensor(std::move(input), torch::kLong),
            torch::tensor(std::move(target), torch::kLong)};
}

std::optional<size_t> MMapDataset::size() const {
    return static_cast<size_t>(num_sequences_);
}

} // namespace xorzen
