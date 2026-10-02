#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace xorzen {

struct TokenizerEncoding {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
};

struct TokenizerMetadata {
    std::string name;
    int64_t vocab_size = 0;
    std::string version = "1.0.0";
    std::string description;
    std::unordered_map<std::string, int64_t> special_tokens;
    std::string author = "Akik faraji";
    std::string created_at;
};

struct BEBPETokenizerConfig {
    bool add_prefix_space = false;
    bool byte_fallback = true;
    
    // Core Special Tokens
    int64_t pad_token_id = 0;
    int64_t unk_token_id = 1;
    int64_t bos_token_id = 2;
    int64_t eos_token_id = 3;
    int64_t mask_token_id = 4;
    
    std::string pad_token = "<pad>";
    std::string unk_token = "<unk>";
    std::string bos_token = "<s>";
    std::string eos_token = "</s>";
    std::string mask_token = "<mask>";

    // Enhanced Special Tokens (IDs often 5-150+)
    std::unordered_map<std::string, int64_t> added_tokens;
    
    // Template Processing
    std::string bos_template = "<s>";
    std::string eos_template = "</s>";
};

class BEBPETokenizer {
public:
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const noexcept;
    };

    BEBPETokenizer();

    static BEBPETokenizer from_tokenizer_json(const std::filesystem::path& path);
    static BEBPETokenizer from_vocab_and_merges(const std::filesystem::path& vocab_path,
                                                const std::filesystem::path& merges_path);

    TokenizerEncoding encode(std::string_view text,
                             bool add_special_tokens = true,
                             int64_t max_length = 0,
                             bool padding = false,
                             bool truncation = false) const;
    std::vector<TokenizerEncoding> batch_encode(const std::vector<std::string>& texts,
                                                bool add_special_tokens = true,
                                                int64_t max_length = 0,
                                                bool padding = true,
                                                bool truncation = true) const;
    std::string decode(const std::vector<int64_t>& token_ids,
                       bool skip_special_tokens = true,
                       bool clean_up_tokenization_spaces = false) const;

    int64_t vocab_size() const;
    bool empty() const;
    std::optional<int64_t> token_to_id(std::string_view token) const;
    std::string id_to_token(int64_t id) const;
    
    const BEBPETokenizerConfig& config() const { return config_; }
    const TokenizerMetadata& metadata() const { return metadata_; }

private:
    void load_json_text(const std::string& text);
    void load_metadata_json(const std::filesystem::path& path);
    void rebuild_byte_tables();
    std::vector<std::string> bytes_to_symbols(std::string_view text) const;
    std::vector<std::string> apply_bpe(std::vector<std::string> symbols) const;
    int32_t pair_rank(const std::string& a, const std::string& b) const;
    bool is_special_id(int64_t id) const;
    void infer_special_ids();

    BEBPETokenizerConfig config_;
    TokenizerMetadata metadata_;
    std::unordered_map<std::string, int64_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merge_rank_;
    std::vector<std::string> byte_encoder_;
    std::unordered_map<std::string, uint8_t> byte_decoder_;
};

} // namespace xorzen
