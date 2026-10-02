#include "xorzen/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace xorzen {
namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open tokenizer file: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

uint32_t parse_hex4(std::string_view s, size_t pos) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        char c = s[pos + i];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(10 + c - 'a');
        else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(10 + c - 'A');
        else throw std::runtime_error("invalid JSON unicode escape");
    }
    return value;
}

void skip_ws(std::string_view s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
}

std::string parse_json_string(std::string_view s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '"') throw std::runtime_error("expected JSON string");
    ++pos;
    std::string out;
    while (pos < s.size()) {
        char c = s[pos++];
        if (c == '"') return out;
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (pos >= s.size()) throw std::runtime_error("bad JSON escape");
        char e = s[pos++];
        switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (pos + 4 > s.size()) throw std::runtime_error("short JSON unicode escape");
                uint32_t cp = parse_hex4(s, pos);
                pos += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && pos + 6 <= s.size() && s[pos] == '\\' && s[pos + 1] == 'u') {
                    pos += 2;
                    uint32_t low = parse_hex4(s, pos);
                    pos += 4;
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                }
                append_utf8(out, cp);
                break;
            }
            default: throw std::runtime_error("unsupported JSON escape");
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

int64_t parse_json_int(std::string_view s, size_t& pos) {
    skip_ws(s, pos);
    bool neg = false;
    if (pos < s.size() && s[pos] == '-') {
        neg = true;
        ++pos;
    }
    int64_t value = 0;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
        value = value * 10 + (s[pos++] - '0');
    }
    return neg ? -value : value;
}

size_t find_key(std::string_view s, std::string_view key, size_t start = 0) {
    std::string needle = "\"" + std::string(key) + "\"";
    return s.find(needle, start);
}

size_t matching_bracket(std::string_view s, size_t open_pos, char open, char close) {
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (size_t i = open_pos; i < s.size(); ++i) {
        char c = s[i];
        if (in_string) {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == open) ++depth;
        else if (c == close && --depth == 0) return i;
    }
    throw std::runtime_error("unmatched JSON bracket");
}

void parse_vocab_object(std::string_view s,
                        std::unordered_map<std::string, int64_t>& token_to_id,
                        std::vector<std::string>& id_to_token) {
    size_t key = find_key(s, "vocab");
    if (key == std::string_view::npos) throw std::runtime_error("tokenizer JSON missing model.vocab");
    size_t open = s.find('{', key);
    if (open == std::string_view::npos) throw std::runtime_error("tokenizer JSON has invalid vocab");
    size_t close = matching_bracket(s, open, '{', '}');
    size_t pos = open + 1;
    while (pos < close) {
        skip_ws(s, pos);
        if (pos >= close || s[pos] == '}') break;
        std::string token = parse_json_string(s, pos);
        skip_ws(s, pos);
        if (pos >= close || s[pos++] != ':') throw std::runtime_error("invalid vocab entry");
        int64_t id = parse_json_int(s, pos);
        token_to_id[token] = id;
        if (id >= 0) {
            if (static_cast<size_t>(id) >= id_to_token.size()) id_to_token.resize(static_cast<size_t>(id) + 1);
            id_to_token[static_cast<size_t>(id)] = token;
        }
        skip_ws(s, pos);
        if (pos < close && s[pos] == ',') ++pos;
    }
}

std::pair<std::string, std::string> split_merge_string(const std::string& merge) {
    size_t sep = merge.find(' ');
    if (sep == std::string::npos) throw std::runtime_error("invalid BPE merge string");
    return {merge.substr(0, sep), merge.substr(sep + 1)};
}

void parse_merges(std::string_view s,
                  std::unordered_map<std::pair<std::string, std::string>, int32_t, BEBPETokenizer::PairHash>& ranks) {
    size_t key = find_key(s, "merges");
    if (key == std::string_view::npos) return;
    size_t open = s.find('[', key);
    if (open == std::string_view::npos) return;
    size_t close = matching_bracket(s, open, '[', ']');
    size_t pos = open + 1;
    int32_t rank = 0;
    while (pos < close) {
        skip_ws(s, pos);
        if (pos >= close || s[pos] == ']') break;
        std::pair<std::string, std::string> pair;
        if (s[pos] == '"') {
            pair = split_merge_string(parse_json_string(s, pos));
        } else if (s[pos] == '[') {
            ++pos;
            pair.first = parse_json_string(s, pos);
            skip_ws(s, pos);
            if (s[pos] == ',') ++pos;
            pair.second = parse_json_string(s, pos);
            skip_ws(s, pos);
            if (s[pos] != ']') throw std::runtime_error("invalid merge pair array");
            ++pos;
        } else {
            throw std::runtime_error("invalid merges array");
        }
        ranks[pair] = rank++;
        skip_ws(s, pos);
        if (pos < close && s[pos] == ',') ++pos;
    }
}

void parse_added_tokens(std::string_view s, BEBPETokenizerConfig& config,
                        const std::unordered_map<std::string, int64_t>& vocab) {
    auto assign_if_present = [&](const std::string& token, int64_t& id, std::string& target) {
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            id = it->second;
            target = token;
        }
    };
    assign_if_present("<pad>", config.pad_token_id, config.pad_token);
    assign_if_present("[PAD]", config.pad_token_id, config.pad_token);
    assign_if_present("<unk>", config.unk_token_id, config.unk_token);
    assign_if_present("[UNK]", config.unk_token_id, config.unk_token);
    assign_if_present("<s>", config.bos_token_id, config.bos_token);
    assign_if_present("[BOS]", config.bos_token_id, config.bos_token);
    assign_if_present("</s>", config.eos_token_id, config.eos_token);
    assign_if_present("[EOS]", config.eos_token_id, config.eos_token);

    size_t key = find_key(s, "add_prefix_space");
    if (key != std::string_view::npos) {
        size_t colon = s.find(':', key);
        if (colon != std::string_view::npos) {
            size_t pos = colon + 1;
            skip_ws(s, pos);
            config.add_prefix_space = s.substr(pos, 4) == "true";
        }
    }
}

std::vector<uint32_t> utf8_codepoints(std::string_view s) {
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < s.size();) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        if (c < 0x80) {
            cps.push_back(c);
            ++i;
        } else if ((c >> 5) == 0x6 && i + 1 < s.size()) {
            cps.push_back(((c & 0x1F) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3F));
            i += 2;
        } else if ((c >> 4) == 0xE && i + 2 < s.size()) {
            cps.push_back(((c & 0x0F) << 12) |
                          ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                          (static_cast<uint8_t>(s[i + 2]) & 0x3F));
            i += 3;
        } else if ((c >> 3) == 0x1E && i + 3 < s.size()) {
            cps.push_back(((c & 0x07) << 18) |
                          ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                          ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) |
                          (static_cast<uint8_t>(s[i + 3]) & 0x3F));
            i += 4;
        } else {
            cps.push_back(c);
            ++i;
        }
    }
    return cps;
}

std::string clean_spaces(std::string text) {
    std::string out;
    bool last_space = false;
    for (char c : text) {
        bool space = std::isspace(static_cast<unsigned char>(c));
        if (space) {
            if (!last_space) out.push_back(' ');
        } else {
            out.push_back(c);
        }
        last_space = space;
    }
    return out;
}

} // namespace

size_t BEBPETokenizer::PairHash::operator()(const std::pair<std::string, std::string>& p) const noexcept {
    size_t h1 = std::hash<std::string>{}(p.first);
    size_t h2 = std::hash<std::string>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
}

BEBPETokenizer::BEBPETokenizer() {
    rebuild_byte_tables();
}

BEBPETokenizer BEBPETokenizer::from_tokenizer_json(const std::filesystem::path& path) {
    BEBPETokenizer tokenizer;
    tokenizer.load_json_text(read_file(path));
    return tokenizer;
}

BEBPETokenizer BEBPETokenizer::from_vocab_and_merges(const std::filesystem::path& vocab_path,
                                                     const std::filesystem::path& merges_path) {
    BEBPETokenizer tokenizer;
    parse_vocab_object(read_file(vocab_path), tokenizer.token_to_id_, tokenizer.id_to_token_);

    std::ifstream in(merges_path);
    if (!in) throw std::runtime_error("failed to open merges file: " + merges_path.string());
    std::string line;
    int32_t rank = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        tokenizer.merge_rank_[split_merge_string(line)] = rank++;
    }
    tokenizer.infer_special_ids();
    tokenizer.rebuild_byte_tables();
    return tokenizer;
}

void BEBPETokenizer::load_json_text(const std::string& text) {
    token_to_id_.clear();
    id_to_token_.clear();
    merge_rank_.clear();
    parse_vocab_object(text, token_to_id_, id_to_token_);
    parse_merges(text, merge_rank_);
    parse_added_tokens(text, config_, token_to_id_);
    infer_special_ids();
    rebuild_byte_tables();
}

void BEBPETokenizer::rebuild_byte_tables() {
    byte_encoder_.assign(256, {});
    byte_decoder_.clear();

    std::vector<int> bs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    std::unordered_set<int> visible(bs.begin(), bs.end());
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!visible.count(b)) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        std::string encoded;
        append_utf8(encoded, static_cast<uint32_t>(cs[i]));
        byte_encoder_[static_cast<size_t>(bs[i])] = encoded;
        byte_decoder_[encoded] = static_cast<uint8_t>(bs[i]);
    }
}

TokenizerEncoding BEBPETokenizer::encode(std::string_view text,
                                          bool add_special_tokens,
                                          int64_t max_length,
                                          bool padding,
                                          bool truncation) const {
    auto symbols = bytes_to_symbols(text);
    auto pieces = apply_bpe(std::move(symbols));

    TokenizerEncoding out;
    if (add_special_tokens) out.input_ids.push_back(config_.bos_token_id);
    for (const auto& piece : pieces) {
        auto it = token_to_id_.find(piece);
        out.input_ids.push_back(it == token_to_id_.end() ? config_.unk_token_id : it->second);
    }
    if (add_special_tokens) out.input_ids.push_back(config_.eos_token_id);

    if (max_length > 0 && static_cast<int64_t>(out.input_ids.size()) > max_length && truncation) {
        out.input_ids.resize(static_cast<size_t>(max_length));
        if (add_special_tokens && !out.input_ids.empty()) out.input_ids.back() = config_.eos_token_id;
    }

    out.attention_mask.assign(out.input_ids.size(), 1);
    if (max_length > 0 && padding && static_cast<int64_t>(out.input_ids.size()) < max_length) {
        size_t old = out.input_ids.size();
        out.input_ids.resize(static_cast<size_t>(max_length), config_.pad_token_id);
        out.attention_mask.resize(static_cast<size_t>(max_length), 0);
        std::fill(out.attention_mask.begin(), out.attention_mask.begin() + static_cast<std::ptrdiff_t>(old), 1);
    }
    return out;
}

std::vector<TokenizerEncoding> BEBPETokenizer::batch_encode(const std::vector<std::string>& texts,
                                                            bool add_special_tokens,
                                                            int64_t max_length,
                                                            bool padding,
                                                            bool truncation) const {
    std::vector<TokenizerEncoding> results;
    results.reserve(texts.size());
    int64_t batch_max = max_length;
    if (padding && max_length == 0) {
        for (const auto& text : texts) {
            auto tmp = encode(text, add_special_tokens, 0, false, false);
            batch_max = std::max<int64_t>(batch_max, static_cast<int64_t>(tmp.input_ids.size()));
            results.push_back(std::move(tmp));
        }
        for (auto& r : results) {
            size_t old = r.input_ids.size();
            r.input_ids.resize(static_cast<size_t>(batch_max), config_.pad_token_id);
            r.attention_mask.resize(static_cast<size_t>(batch_max), 0);
            std::fill(r.attention_mask.begin(), r.attention_mask.begin() + static_cast<std::ptrdiff_t>(old), 1);
        }
        return results;
    }
    for (const auto& text : texts) {
        results.push_back(encode(text, add_special_tokens, max_length, padding, truncation));
    }
    return results;
}

std::string BEBPETokenizer::decode(const std::vector<int64_t>& token_ids,
                                   bool skip_special_tokens,
                                   bool clean_up_tokenization_spaces) const {
    std::string bytes;
    for (int64_t id : token_ids) {
        if (skip_special_tokens && is_special_id(id)) continue;
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        const auto& token = id_to_token_[static_cast<size_t>(id)];
        for (uint32_t cp : utf8_codepoints(token)) {
            std::string one;
            append_utf8(one, cp);
            auto it = byte_decoder_.find(one);
            if (it != byte_decoder_.end()) bytes.push_back(static_cast<char>(it->second));
            else bytes += one;
        }
    }
    return clean_up_tokenization_spaces ? clean_spaces(bytes) : bytes;
}

int64_t BEBPETokenizer::vocab_size() const {
    return static_cast<int64_t>(id_to_token_.size());
}

bool BEBPETokenizer::empty() const {
    return token_to_id_.empty();
}

std::optional<int64_t> BEBPETokenizer::token_to_id(std::string_view token) const {
    auto it = token_to_id_.find(std::string(token));
    if (it == token_to_id_.end()) return std::nullopt;
    return it->second;
}

std::string BEBPETokenizer::id_to_token(int64_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return {};
    return id_to_token_[static_cast<size_t>(id)];
}

std::vector<std::string> BEBPETokenizer::bytes_to_symbols(std::string_view text) const {
    std::string input(text);
    if (config_.add_prefix_space && !input.empty() && input.front() != ' ') {
        input.insert(input.begin(), ' ');
    }
    std::vector<std::string> symbols;
    symbols.reserve(input.size());
    for (unsigned char b : input) {
        std::string raw(1, static_cast<char>(b));
        if (token_to_id_.find(raw) != token_to_id_.end()) symbols.push_back(raw);
        else symbols.push_back(byte_encoder_[b]);
    }
    return symbols;
}

std::vector<std::string> BEBPETokenizer::apply_bpe(std::vector<std::string> symbols) const {
    if (symbols.size() < 2 || merge_rank_.empty()) return symbols;

    struct SymbolNode {
        std::string text;
        int32_t prev;
        int32_t next;
    };

    std::vector<SymbolNode> nodes(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i) {
        nodes[i] = { std::move(symbols[i]), static_cast<int32_t>(i) - 1, static_cast<int32_t>(i) + 1 };
    }
    nodes.back().next = -1;

    while (true) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        int32_t best_node_idx = -1;

        int32_t curr = 0;
        while (curr != -1 && nodes[curr].next != -1) {
            int32_t next_idx = nodes[curr].next;
            int32_t rank = pair_rank(nodes[curr].text, nodes[next_idx].text);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_node_idx = curr;
            }
            curr = next_idx;
        }

        if (best_node_idx == -1) break;

        std::string target_a = nodes[best_node_idx].text;
        std::string target_b = nodes[nodes[best_node_idx].next].text;

        int32_t i = 0;
        while (i != -1 && nodes[i].next != -1) {
            int32_t next_idx = nodes[i].next;
            if (nodes[i].text == target_a && nodes[next_idx].text == target_b) {
                nodes[i].text += nodes[next_idx].text;
                int32_t next_next_idx = nodes[next_idx].next;
                nodes[i].next = next_next_idx;
                if (next_next_idx != -1) {
                    nodes[next_next_idx].prev = i;
                }
                i = next_next_idx;
            } else {
                i = next_idx;
            }
        }
    }

    std::vector<std::string> result;
    result.reserve(symbols.size());
    int32_t curr = 0;
    while (curr != -1) {
        result.push_back(std::move(nodes[curr].text));
        curr = nodes[curr].next;
    }
    return result;
}

int32_t BEBPETokenizer::pair_rank(const std::string& a, const std::string& b) const {
    auto it = merge_rank_.find({a, b});
    if (it == merge_rank_.end()) return -1;
    return it->second;
}

bool BEBPETokenizer::is_special_id(int64_t id) const {
    return id == config_.pad_token_id || id == config_.unk_token_id ||
           id == config_.bos_token_id || id == config_.eos_token_id;
}

void BEBPETokenizer::infer_special_ids() {
    auto set_if = [&](const std::string& token, int64_t& id) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) id = it->second;
    };
    set_if(config_.pad_token, config_.pad_token_id);
    set_if(config_.unk_token, config_.unk_token_id);
    set_if(config_.bos_token, config_.bos_token_id);
    set_if(config_.eos_token, config_.eos_token_id);
}

} // namespace xorzen
