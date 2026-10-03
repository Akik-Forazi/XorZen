// ============================================================
//  xorzen.cpp — include/xorzen/project.h
//  .xorzen folder-based model project format
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <fstream>
#include <functional>

#include "xorzen/types.h"

namespace xorzen {

// ============================================================
//  ProjectManifest — manifest.json root descriptor
// ============================================================
struct ExpertDomainInfo {
    std::string             name;          // e.g. "mathematics"
    std::vector<int64_t>    expert_ids;    // which expert indices
    double                  confidence     = 0.0;
    std::string             last_updated;
    int64_t                 training_tokens = 0;
    bool                    needs_improvement = false;
    // subtopics: subtopic_name -> expert_ids
    std::unordered_map<std::string, std::vector<int64_t>> subtopics;
};

struct AdapterSessionInfo {
    std::string session_id;       // e.g. "2026_05_19_143022"
    std::string created_at;
    std::string topic;
    int64_t     steps_trained  = 0;
    double      final_loss     = 0.0;
    std::vector<int64_t> updated_layers;
};

struct ProjectManifest {
    // Identity
    std::string name;
    std::string author        = "FRAZIYM TECH & AI";
    std::string email         = "akikfaraji@gmail.com";
    std::string version       = "4.0";
    std::string created_at;
    std::string signature;    // sha256 of base weights

    // Architecture
    ModelConfig model_config;

    // Expert domain registry
    std::unordered_map<std::string, ExpertDomainInfo> expert_domains;

    // Adapter sessions
    std::vector<AdapterSessionInfo> adapter_sessions;

    // Statistics
    int64_t total_tokens_processed = 0;
    int64_t total_learning_steps   = 0;
    double  best_eval_loss         = 1e30;

    // License
    std::string license = "FRAZIYM-PROPRIETARY";

    // Serialize / deserialize
    void save(const std::filesystem::path& manifest_path) const;
    static ProjectManifest load(const std::filesystem::path& manifest_path);

    std::string to_json() const;
    static ProjectManifest from_json(const std::string& json);
};

// ============================================================
//  XorzenProject — manages a full .xorzen folder
// ============================================================
class XorzenProject {
public:
    // Layout constants
    static constexpr const char* MANIFEST_FILE     = "manifest.json";
    static constexpr const char* CONFIG_FILE        = "config.toml";
    static constexpr const char* BASE_DIR           = "base";
    static constexpr const char* EXPERTS_DIR        = "experts";
    static constexpr const char* ADAPTERS_DIR       = "adapters";
    static constexpr const char* MEMORY_DIR         = "memory";
    static constexpr const char* ROUTER_DIR         = "router";
    static constexpr const char* KV_CACHE_DIR       = "memory/kv_cache";
    static constexpr const char* EPISODIC_DIR       = "memory/episodic";
    static constexpr const char* SEMANTIC_DIR       = "memory/semantic";
    static constexpr const char* COT_STATE_FILE     = "memory/cot_state.bin";

    // ---- Construction ----

    // Create a brand-new .xorzen project folder
    static XorzenProject create(const std::filesystem::path& root,
                                const std::string& name,
                                const ModelConfig& cfg);

    // Open an existing .xorzen project
    static XorzenProject open(const std::filesystem::path& root);

    // ---- Base Model ----

    // Save the full model state to base/
    void save_base_model(torch::nn::Module& model) const;

    // Load base model weights into an existing module
    void load_base_model(torch::nn::Module& model) const;

    // Save only the LM head
    void save_lm_head(const torch::Tensor& weight) const;
    torch::Tensor load_lm_head() const;

    // ---- Expert Domains ----

    // Save a single expert tensor under experts/<domain>/<name>.bin
    void save_expert(const std::string& domain,
                     const std::string& expert_name,
                     const torch::Tensor& weights) const;

    // Load a single expert tensor
    torch::Tensor load_expert(const std::string& domain,
                              const std::string& expert_name) const;

    // List all expert domains
    std::vector<std::string> list_domains() const;

    // List expert files within a domain
    std::vector<std::filesystem::path> list_experts(const std::string& domain) const;

    // Register a new domain in the manifest
    void register_domain(const ExpertDomainInfo& info);

    // Update domain confidence after learning
    void update_domain_confidence(const std::string& domain,
                                  double confidence,
                                  int64_t tokens_trained);

    // ---- LoRA Adapters ----

    // Save a LoRA adapter session
    std::string save_adapter_session(
        const std::unordered_map<std::string, torch::Tensor>& deltas,
        const AdapterSessionInfo& info);

    // Load all adapter sessions and merge into parameter dict
    std::unordered_map<std::string, torch::Tensor>
    load_merged_adapters(double ema_decay = 0.9) const;

    // Load a specific session's adapters
    std::unordered_map<std::string, torch::Tensor>
    load_adapter_session(const std::string& session_id) const;

    // List adapter sessions (newest first)
    std::vector<AdapterSessionInfo> list_adapter_sessions() const;

    // ---- Persistent CoT State ----
    void save_cot_state(const torch::Tensor& cot_vector) const;
    torch::Tensor load_cot_state(int64_t expected_dim) const;
    bool has_cot_state() const;

    // ---- Memory (KV Cache) ----
    std::filesystem::path kv_session_dir(const std::string& session_id) const;
    void ensure_kv_session(const std::string& session_id) const;

    // ---- Manifest / Config ----
    const ProjectManifest& manifest() const { return manifest_; }
    ProjectManifest& manifest()             { return manifest_; }
    void save_manifest() const;

    const std::filesystem::path& root() const { return root_; }
    std::filesystem::path path(const char* rel) const { return root_ / rel; }

    // Print a human-readable summary
    void print_info() const;

private:
    explicit XorzenProject(std::filesystem::path root, ProjectManifest manifest);

    void ensure_layout() const;

    std::filesystem::path root_;
    ProjectManifest       manifest_;
};

// ============================================================
//  Helpers
// ============================================================

// Compute sha256 of a file (for manifest signing)
std::string sha256_file(const std::filesystem::path& path);

// Write a minimal TOML-like config.toml (human-readable)
void write_config_toml(const std::filesystem::path& path,
                       const ModelConfig& cfg);

} // namespace xorzen
