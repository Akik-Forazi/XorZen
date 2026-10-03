// ============================================================
//  xorzen.cpp — src/storage/project.cpp
//  .xorzen folder-based project format implementation
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/project.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace xorzen {

// ============================================================
//  Internal helpers
// ============================================================

static std::string now_iso() {
    auto t  = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&tt), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

static std::string session_stamp() {
    auto t  = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&tt), "%Y_%m_%d_%H%M%S");
    return oss.str();
}

static std::string jstr(const std::string& v) {
    std::string out; out.reserve(v.size() + 2); out += '"';
    for (char c : v) {
        if      (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else                { out += c; }
    }
    out += '"'; return out;
}

static std::string indent(int n) { return std::string(n * 2, ' '); }

// Save a tensor to a binary file (raw float32 blob + header)
static void save_tensor_bin(const std::filesystem::path& path,
                            const torch::Tensor& t) {
    std::filesystem::create_directories(path.parent_path());
    auto ct = t.to(torch::kCPU).to(torch::kFloat32).contiguous();
    std::ofstream f(path, std::ios::binary);
    // Header: magic(4) + ndim(4) + shape(ndim*8) + data
    const uint32_t MAGIC = 0x585A4B56; // "XZKV"
    f.write(reinterpret_cast<const char*>(&MAGIC), 4);
    uint32_t ndim = static_cast<uint32_t>(ct.dim());
    f.write(reinterpret_cast<const char*>(&ndim), 4);
    for (int64_t i = 0; i < ct.dim(); ++i) {
        int64_t sz = ct.size(i);
        f.write(reinterpret_cast<const char*>(&sz), 8);
    }
    int64_t nbytes = ct.numel() * sizeof(float);
    f.write(reinterpret_cast<const char*>(ct.data_ptr<float>()), nbytes);
}

static torch::Tensor load_tensor_bin(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open tensor file: " + path.string());
    uint32_t magic = 0; f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x585A4B56)
        throw std::runtime_error("Bad magic in: " + path.string());
    uint32_t ndim = 0; f.read(reinterpret_cast<char*>(&ndim), 4);
    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i)
        f.read(reinterpret_cast<char*>(&shape[i]), 8);
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    auto t = torch::empty(shape, torch::kFloat32);
    f.read(reinterpret_cast<char*>(t.data_ptr<float>()), numel * sizeof(float));
    return t;
}

// ============================================================
//  ProjectManifest serialization (minimal hand-written JSON)
// ============================================================

std::string ProjectManifest::to_json() const {
    std::ostringstream o;
    o << "{\n";
    o << indent(1) << "\"xorzen_format\": \"4.0\",\n";
    o << indent(1) << "\"name\": " << jstr(name) << ",\n";
    o << indent(1) << "\"author\": " << jstr(author) << ",\n";
    o << indent(1) << "\"email\": " << jstr(email) << ",\n";
    o << indent(1) << "\"version\": " << jstr(version) << ",\n";
    o << indent(1) << "\"created_at\": " << jstr(created_at) << ",\n";
    o << indent(1) << "\"license\": " << jstr(license) << ",\n";
    o << indent(1) << "\"signature\": " << jstr(signature) << ",\n";
    // Model config
    o << indent(1) << "\"model_config\": {\n";
    o << indent(2) << "\"vocab_size\": " << model_config.vocab_size << ",\n";
    o << indent(2) << "\"hidden_size\": " << model_config.hidden_size << ",\n";
    o << indent(2) << "\"num_layers\": " << model_config.num_layers << ",\n";
    o << indent(2) << "\"num_attention_heads\": " << model_config.num_attention_heads << ",\n";
    o << indent(2) << "\"num_experts\": " << model_config.num_experts << ",\n";
    o << indent(2) << "\"top_k_experts\": " << model_config.top_k_experts << ",\n";
    o << indent(2) << "\"context_length\": " << model_config.context_length << ",\n";
    o << indent(2) << "\"cot_dim\": " << model_config.cot_dim << ",\n";
    o << indent(2) << "\"cot_components\": " << model_config.cot_components << "\n";
    o << indent(1) << "},\n";
    // Expert domains
    o << indent(1) << "\"expert_domains\": {\n";
    bool first_domain = true;
    for (const auto& [name, info] : expert_domains) {
        if (!first_domain) o << ",\n";
        first_domain = false;
        o << indent(2) << jstr(name) << ": {\n";
        o << indent(3) << "\"confidence\": " << info.confidence << ",\n";
        o << indent(3) << "\"last_updated\": " << jstr(info.last_updated) << ",\n";
        o << indent(3) << "\"training_tokens\": " << info.training_tokens << ",\n";
        o << indent(3) << "\"needs_improvement\": " << (info.needs_improvement ? "true" : "false") << ",\n";
        o << indent(3) << "\"expert_ids\": [";
        for (size_t i = 0; i < info.expert_ids.size(); ++i) {
            if (i) o << ", ";
            o << info.expert_ids[i];
        }
        o << "]\n";
        o << indent(2) << "}";
    }
    o << "\n" << indent(1) << "},\n";
    // Statistics
    o << indent(1) << "\"total_tokens_processed\": " << total_tokens_processed << ",\n";
    o << indent(1) << "\"total_learning_steps\": " << total_learning_steps << ",\n";
    o << indent(1) << "\"best_eval_loss\": " << best_eval_loss << "\n";
    o << "}\n";
    return o.str();
}

void ProjectManifest::save(const std::filesystem::path& path) const {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot write manifest: " + path.string());
    f << to_json();
}

static int64_t parse_i64(const std::string& s) {
    try { return std::stoll(s); } catch (...) { return 0; }
}
static double parse_f64(const std::string& s) {
    try { return std::stod(s); } catch (...) { return 0.0; }
}
// Ultra-minimal JSON value extractor (handles string, number, bool)
static std::string extract_json_value(const std::string& json,
                                      const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    ++pos;
    while (pos < json.size() && std::isspace(json[pos])) ++pos;
    if (json[pos] == '"') {
        auto end = json.find('"', pos + 1);
        while (end != std::string::npos && json[end - 1] == '\\')
            end = json.find('"', end + 1);
        return json.substr(pos + 1, end - pos - 1);
    }
    // Number or bool
    auto end = json.find_first_of(",}\n", pos);
    auto val = json.substr(pos, end - pos);
    // trim
    while (!val.empty() && std::isspace(val.back())) val.pop_back();
    return val;
}

ProjectManifest ProjectManifest::from_json(const std::string& json) {
    ProjectManifest m;
    m.name            = extract_json_value(json, "name");
    m.author          = extract_json_value(json, "author");
    m.email           = extract_json_value(json, "email");
    m.version         = extract_json_value(json, "version");
    m.created_at      = extract_json_value(json, "created_at");
    m.license         = extract_json_value(json, "license");
    m.signature       = extract_json_value(json, "signature");
    m.total_tokens_processed = parse_i64(extract_json_value(json, "total_tokens_processed"));
    m.total_learning_steps   = parse_i64(extract_json_value(json, "total_learning_steps"));
    m.best_eval_loss         = parse_f64(extract_json_value(json, "best_eval_loss"));
    // Model config fields
    m.model_config.vocab_size          = static_cast<int64_t>(parse_i64(extract_json_value(json, "vocab_size")));
    m.model_config.hidden_size         = static_cast<int64_t>(parse_i64(extract_json_value(json, "hidden_size")));
    m.model_config.num_layers          = static_cast<int64_t>(parse_i64(extract_json_value(json, "num_layers")));
    m.model_config.num_attention_heads = static_cast<int64_t>(parse_i64(extract_json_value(json, "num_attention_heads")));
    m.model_config.num_experts         = static_cast<int64_t>(parse_i64(extract_json_value(json, "num_experts")));
    m.model_config.top_k_experts       = static_cast<int64_t>(parse_i64(extract_json_value(json, "top_k_experts")));
    m.model_config.context_length      = static_cast<int64_t>(parse_i64(extract_json_value(json, "context_length")));
    m.model_config.cot_dim             = static_cast<int64_t>(parse_i64(extract_json_value(json, "cot_dim")));
    m.model_config.cot_components      = static_cast<int64_t>(parse_i64(extract_json_value(json, "cot_components")));
    m.model_config.normalize();
    return m;
}

ProjectManifest ProjectManifest::load(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Manifest not found: " + path.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    return from_json(json);
}

// ============================================================
//  XorzenProject — construction
// ============================================================

XorzenProject::XorzenProject(std::filesystem::path root, ProjectManifest manifest)
    : root_(std::move(root)), manifest_(std::move(manifest)) {}

XorzenProject XorzenProject::create(const std::filesystem::path& root,
                                    const std::string& name,
                                    const ModelConfig& cfg) {
    if (std::filesystem::exists(root))
        throw std::runtime_error("Project already exists: " + root.string());

    ProjectManifest m;
    m.name         = name;
    m.version      = "4.0";
    m.created_at   = now_iso();
    m.model_config = cfg;
    m.model_config.normalize();

    XorzenProject proj(root, std::move(m));
    proj.ensure_layout();
    proj.save_manifest();
    write_config_toml(root / CONFIG_FILE, cfg);

    std::cout << "[XorzenProject] Created: " << root << "\n";
    return proj;
}

XorzenProject XorzenProject::open(const std::filesystem::path& root) {
    auto mpath = root / MANIFEST_FILE;
    if (!std::filesystem::exists(mpath))
        throw std::runtime_error("Not a .xorzen project: " + root.string());
    auto m = ProjectManifest::load(mpath);
    XorzenProject proj(root, std::move(m));
    proj.ensure_layout();
    std::cout << "[XorzenProject] Opened: " << proj.manifest_.name
              << " (v" << proj.manifest_.version << ")\n";
    return proj;
}

void XorzenProject::ensure_layout() const {
    for (const char* d : {
        BASE_DIR, "base/layers",
        EXPERTS_DIR, ADAPTERS_DIR,
        MEMORY_DIR, KV_CACHE_DIR, EPISODIC_DIR, SEMANTIC_DIR,
        ROUTER_DIR
    }) {
        std::filesystem::create_directories(root_ / d);
    }
}

// ============================================================
//  Base model save / load
// ============================================================

void XorzenProject::save_base_model(torch::nn::Module& model) const {
    auto arch_path = root_ / BASE_DIR / "model.pt";
    torch::serialize::OutputArchive ar;
    model.save(ar);
    ar.save_to(arch_path.string());
    std::cout << "[XorzenProject] Base model saved → " << arch_path.filename() << "\n";
}

void XorzenProject::load_base_model(torch::nn::Module& model) const {
    auto arch_path = root_ / BASE_DIR / "model.pt";
    if (!std::filesystem::exists(arch_path))
        throw std::runtime_error("No base model at: " + arch_path.string());
    torch::serialize::InputArchive ar;
    ar.load_from(arch_path.string());
    model.load(ar);
}

void XorzenProject::save_lm_head(const torch::Tensor& weight) const {
    save_tensor_bin(root_ / BASE_DIR / "lm_head.bin", weight);
}

torch::Tensor XorzenProject::load_lm_head() const {
    return load_tensor_bin(root_ / BASE_DIR / "lm_head.bin");
}

// ============================================================
//  Expert domain management
// ============================================================

void XorzenProject::save_expert(const std::string& domain,
                                const std::string& expert_name,
                                const torch::Tensor& weights) const {
    auto dir = root_ / EXPERTS_DIR / domain;
    std::filesystem::create_directories(dir);
    save_tensor_bin(dir / (expert_name + ".bin"), weights);
}

torch::Tensor XorzenProject::load_expert(const std::string& domain,
                                         const std::string& expert_name) const {
    auto p = root_ / EXPERTS_DIR / domain / (expert_name + ".bin");
    return load_tensor_bin(p);
}

std::vector<std::string> XorzenProject::list_domains() const {
    std::vector<std::string> result;
    auto d = root_ / EXPERTS_DIR;
    for (auto& e : std::filesystem::directory_iterator(d)) {
        if (e.is_directory()) result.push_back(e.path().filename().string());
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::filesystem::path>
XorzenProject::list_experts(const std::string& domain) const {
    std::vector<std::filesystem::path> result;
    auto d = root_ / EXPERTS_DIR / domain;
    if (!std::filesystem::exists(d)) return result;
    for (auto& e : std::filesystem::directory_iterator(d)) {
        if (e.path().extension() == ".bin") result.push_back(e.path());
    }
    std::sort(result.begin(), result.end());
    return result;
}

void XorzenProject::register_domain(const ExpertDomainInfo& info) {
    manifest_.expert_domains[info.name] = info;
    // Write domain metadata.json
    auto meta_path = root_ / EXPERTS_DIR / info.name / "metadata.json";
    std::filesystem::create_directories(meta_path.parent_path());
    std::ofstream f(meta_path);
    f << "{\n"
      << "  \"domain\": " << jstr(info.name) << ",\n"
      << "  \"confidence\": " << info.confidence << ",\n"
      << "  \"last_updated\": " << jstr(info.last_updated) << ",\n"
      << "  \"training_tokens\": " << info.training_tokens << ",\n"
      << "  \"expert_ids\": [";
    for (size_t i = 0; i < info.expert_ids.size(); ++i) {
        if (i) f << ", ";
        f << info.expert_ids[i];
    }
    f << "]\n}\n";
    save_manifest();
}

void XorzenProject::update_domain_confidence(const std::string& domain,
                                             double confidence,
                                             int64_t tokens_trained) {
    auto it = manifest_.expert_domains.find(domain);
    if (it == manifest_.expert_domains.end()) {
        ExpertDomainInfo info;
        info.name = domain;
        manifest_.expert_domains[domain] = info;
        it = manifest_.expert_domains.find(domain);
    }
    it->second.confidence      = confidence;
    it->second.training_tokens += tokens_trained;
    it->second.last_updated    = now_iso();
    it->second.needs_improvement = (confidence < 0.6);
    save_manifest();
}

// ============================================================
//  LoRA Adapter sessions
// ============================================================

std::string XorzenProject::save_adapter_session(
    const std::unordered_map<std::string, torch::Tensor>& deltas,
    const AdapterSessionInfo& info) {
    std::string sid = info.session_id.empty() ? session_stamp() : info.session_id;
    auto session_dir = root_ / ADAPTERS_DIR / sid;
    std::filesystem::create_directories(session_dir);

    for (const auto& [key, tensor] : deltas) {
        // Sanitize key for filesystem
        std::string fname = key;
        for (char& c : fname)
            if (c == '/' || c == '\\' || c == ':') c = '_';
        save_tensor_bin(session_dir / (fname + ".bin"), tensor);
    }

    // Write stats
    std::ofstream f(session_dir / "stats.json");
    f << "{\n"
      << "  \"session_id\": " << jstr(sid) << ",\n"
      << "  \"topic\": " << jstr(info.topic) << ",\n"
      << "  \"steps_trained\": " << info.steps_trained << ",\n"
      << "  \"final_loss\": " << info.final_loss << ",\n"
      << "  \"created_at\": " << jstr(now_iso()) << "\n"
      << "}\n";

    manifest_.adapter_sessions.push_back({sid, now_iso(), info.topic,
                                          info.steps_trained, info.final_loss,
                                          info.updated_layers});
    ++manifest_.total_learning_steps;
    save_manifest();
    std::cout << "[XorzenProject] Adapter session saved: " << sid << "\n";
    return sid;
}

std::unordered_map<std::string, torch::Tensor>
XorzenProject::load_adapter_session(const std::string& session_id) const {
    auto session_dir = root_ / ADAPTERS_DIR / session_id;
    std::unordered_map<std::string, torch::Tensor> result;
    if (!std::filesystem::exists(session_dir)) return result;
    for (auto& entry : std::filesystem::directory_iterator(session_dir)) {
        if (entry.path().extension() != ".bin") continue;
        std::string key = entry.path().stem().string();
        result[key] = load_tensor_bin(entry.path());
    }
    return result;
}

std::unordered_map<std::string, torch::Tensor>
XorzenProject::load_merged_adapters(double ema_decay) const {
    // Load all sessions newest-first and EMA-merge
    std::unordered_map<std::string, torch::Tensor> merged;
    auto sessions = list_adapter_sessions();
    // newest first
    for (auto& s : sessions) {
        auto deltas = load_adapter_session(s.session_id);
        for (auto& [key, tensor] : deltas) {
            auto it = merged.find(key);
            if (it == merged.end()) {
                merged[key] = tensor.clone();
            } else {
                // EMA: current = decay * current + (1-decay) * new
                it->second = ema_decay * it->second + (1.0 - ema_decay) * tensor;
            }
        }
    }
    return merged;
}

std::vector<AdapterSessionInfo> XorzenProject::list_adapter_sessions() const {
    auto sessions = manifest_.adapter_sessions;
    // Sort newest first by session_id (which is timestamp-based)
    std::sort(sessions.begin(), sessions.end(),
        [](const auto& a, const auto& b) { return a.session_id > b.session_id; });
    return sessions;
}

// ============================================================
//  Persistent CoT state
// ============================================================

void XorzenProject::save_cot_state(const torch::Tensor& cot_vector) const {
    save_tensor_bin(root_ / COT_STATE_FILE, cot_vector);
}

torch::Tensor XorzenProject::load_cot_state(int64_t expected_dim) const {
    auto p = root_ / COT_STATE_FILE;
    if (!std::filesystem::exists(p)) {
        return torch::zeros({1, 1, expected_dim});
    }
    return load_tensor_bin(p);
}

bool XorzenProject::has_cot_state() const {
    return std::filesystem::exists(root_ / COT_STATE_FILE);
}

// ============================================================
//  KV memory paths
// ============================================================

std::filesystem::path XorzenProject::kv_session_dir(const std::string& session_id) const {
    return root_ / KV_CACHE_DIR / session_id;
}

void XorzenProject::ensure_kv_session(const std::string& session_id) const {
    std::filesystem::create_directories(kv_session_dir(session_id));
}

// ============================================================
//  Manifest save / info print
// ============================================================

void XorzenProject::save_manifest() const {
    manifest_.save(root_ / MANIFEST_FILE);
}

void XorzenProject::print_info() const {
    std::cout << "\n" << std::string(64, '=') << "\n"
              << "  .xorzen Project: " << manifest_.name << "\n"
              << "  Author   : " << manifest_.author << "\n"
              << "  Version  : " << manifest_.version << "\n"
              << "  Created  : " << manifest_.created_at << "\n"
              << std::string(64, '-') << "\n"
              << "  Model    : H=" << manifest_.model_config.hidden_size
              << " L=" << manifest_.model_config.num_layers
              << " E=" << manifest_.model_config.num_experts << "\n"
              << "  Domains  : " << manifest_.expert_domains.size() << "\n"
              << "  Adapters : " << manifest_.adapter_sessions.size() << " sessions\n"
              << "  Tokens   : " << manifest_.total_tokens_processed << "\n"
              << "  License  : " << manifest_.license << "\n"
              << std::string(64, '=') << "\n\n";
}

// ============================================================
//  Helpers
// ============================================================

std::string sha256_file(const std::filesystem::path& path) {
    // Lightweight FNV-1a hash as placeholder (real SHA256 needs OpenSSL)
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return "UNSIGNED";
    uint64_t hash = 14695981039346656037ULL;
    char buf[4096];
    while (f.read(buf, sizeof(buf))) {
        for (std::streamsize i = 0; i < f.gcount(); ++i)
            hash = (hash ^ static_cast<uint8_t>(buf[i])) * 1099511628211ULL;
    }
    for (std::streamsize i = 0; i < f.gcount(); ++i)
        hash = (hash ^ static_cast<uint8_t>(buf[i])) * 1099511628211ULL;
    std::ostringstream oss;
    oss << "fnv1a:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return oss.str();
}

void write_config_toml(const std::filesystem::path& path,
                       const ModelConfig& cfg) {
    std::ofstream f(path);
    f << "# XORZEN v4.0 Configuration\n"
      << "# FRAZIYM TECH & AI — Akik Faraji\n\n"
      << "[model]\n"
      << "vocab_size          = " << cfg.vocab_size          << "\n"
      << "hidden_size         = " << cfg.hidden_size          << "\n"
      << "num_layers          = " << cfg.num_layers           << "\n"
      << "num_attention_heads = " << cfg.num_attention_heads  << "\n"
      << "context_length      = " << cfg.context_length       << "\n"
      << "num_experts         = " << cfg.num_experts          << "\n"
      << "top_k_experts       = " << cfg.top_k_experts        << "\n"
      << "cot_dim             = " << cfg.cot_dim              << "\n"
      << "cot_components      = " << cfg.cot_components       << "\n"
      << "dropout             = " << cfg.dropout              << "\n\n"
      << "[memory]\n"
      << "# Disk-sharded KV cache settings\n"
      << "kv_block_tokens     = 256\n"
      << "ram_blocks          = 32\n"
      << "kv_svd_rank         = 64\n"
      << "use_int8_disk       = true\n\n"
      << "[learning]\n"
      << "# Progressive learning settings\n"
      << "lora_rank           = 16\n"
      << "lora_alpha          = 32.0\n"
      << "ewc_lambda          = 0.4\n"
      << "confidence_threshold = 0.3\n"
      << "auto_learn          = false\n";
}

} // namespace xorzen
