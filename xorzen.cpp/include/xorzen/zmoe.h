#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/zmoe.h
//  ShardedExpertFabric — Disk-Based MoE with LRU cache
//  Ported from xorzen/model/zmoe.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
// Key innovation: 192 experts × 32M params = 6GB total on disk
//   Only 24 hot experts cached in RAM (768MB)
//   On-demand loading with LRU eviction
//   Makes 277M model train on 16GB RAM!
// ============================================================

#include <torch/torch.h>
#include "xorzen/types.h"

#include <filesystem>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xorzen {

// ============================================================
//  ExpertStats
// ============================================================
struct ExpertStats {
    int64_t expert_id        = 0;
    int64_t activation_count = 0;
    double  total_weight     = 0.0;
    double  avg_weight       = 0.0;
    int64_t cache_hits       = 0;
    int64_t cache_misses     = 0;
    double  load_time_ms     = 0.0;

    void update(double weight, bool was_cached, double load_time_ms_in = 0.0);
    double hit_rate() const;
};

// ============================================================
//  MoE forward output
// ============================================================
struct MoEOutput {
    torch::Tensor output;          // [N, H]
    torch::Tensor load_balance_loss; // scalar
    double routing_entropy = 0.0;
    double cache_hit_rate  = 0.0;
    int64_t experts_used   = 0;
};

// ============================================================
//  ExpertFFN — SwiGLU expert (single building block)
// ============================================================
struct ExpertFFNImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t intermediate_dim;
    int64_t expert_id = -1;

    torch::nn::Linear gate_proj{nullptr};
    torch::nn::Linear up_proj{nullptr};
    torch::nn::Linear down_proj{nullptr};
    torch::nn::Dropout dropout_{nullptr};

    ExpertFFNImpl(int64_t hidden_dim, int64_t intermediate_dim,
                  double dropout = 0.0, bool bias = false);

    // [N, H] → [N, H] via SwiGLU
    torch::Tensor forward(const torch::Tensor& x);

    int64_t num_params() const;
};
TORCH_MODULE(ExpertFFN);

// ============================================================
//  LRUExpertCache — thread-safe LRU cache for ExpertFFN
// ============================================================
class LRUExpertCache {
public:
    explicit LRUExpertCache(int64_t capacity = 24);

    // Returns nullptr on miss, pointer on hit (moves to MRU position)
    std::shared_ptr<ExpertFFNImpl> get(int64_t expert_id);

    // Insert / refresh expert in cache
    void put(int64_t expert_id, std::shared_ptr<ExpertFFNImpl> expert);

    void clear();

    struct Stats {
        int64_t capacity;
        int64_t current_size;
        int64_t hits;
        int64_t misses;
        int64_t evictions;
        double  hit_rate;
    };
    Stats get_stats() const;

private:
    int64_t capacity_;
    // Order: front = LRU, back = MRU
    std::list<std::pair<int64_t, std::shared_ptr<ExpertFFNImpl>>> list_;
    std::unordered_map<int64_t, decltype(list_)::iterator> map_;
    mutable std::mutex mutex_;
    int64_t hits_ = 0, misses_ = 0, evictions_ = 0;
};

// ============================================================
//  ExpertDiskManager — save/load per-expert .pt files
// ============================================================
class ExpertDiskManager {
public:
    ExpertDiskManager(const std::filesystem::path& shard_dir,
                      int64_t num_experts,
                      int64_t hidden_dim,
                      int64_t intermediate_dim);

    // Load expert from disk (or create fresh if missing)
    std::shared_ptr<ExpertFFNImpl> load_expert(int64_t expert_id,
                                               torch::Device device);

    // Save an expert's state dict to disk
    void save_expert(int64_t expert_id,
                     const std::shared_ptr<ExpertFFNImpl>& expert);

    // Initialize all N experts on disk (noop if already exist, unless force)
    void initialize_all(bool force = false);

    bool expert_exists(int64_t expert_id) const;
    double total_size_mb() const;
    std::filesystem::path expert_path(int64_t expert_id) const;
    int64_t experts_on_disk() const;

private:
    std::filesystem::path shard_dir_;
    int64_t num_experts_;
    int64_t hidden_dim_;
    int64_t intermediate_dim_;

    std::shared_ptr<ExpertFFNImpl> create_new_expert(int64_t id) const;
    void write_manifest() const;
    void read_manifest();
    std::unordered_map<int64_t, bool> manifest_;
    mutable std::mutex manifest_mutex_;
};

// ============================================================
//  ShardedExpertFabric — the full MoE with disk sharding
// ============================================================
class ShardedExpertFabric : public torch::nn::Module {
public:
    explicit ShardedExpertFabric(const ModelConfig& cfg,
                                 bool test_mode = false);

    // Forward: route tokens to experts, aggregate
    // hidden_states: [N, H]
    // expert_indices: [N, top_k]
    // expert_weights: [N, top_k]
    // Returns MoEOutput
    MoEOutput forward(const torch::Tensor& hidden_states,
                      const torch::Tensor& expert_indices,
                      const torch::Tensor& expert_weights);

    void   print_statistics() const;
    void   clear_cache();
    void   save_all_experts();
    void   set_device(torch::Device dev) { device_ = dev; }

    const std::unordered_map<int64_t, ExpertStats>& expert_stats() const {
        return expert_stats_;
    }
    LRUExpertCache::Stats cache_stats() const { return cache_.get_stats(); }

private:
    ModelConfig   config_;
    bool          test_mode_;
    int64_t       num_experts_;
    int64_t       top_k_;
    int64_t       hidden_dim_;
    int64_t       intermediate_dim_;
    torch::Device device_{torch::kCPU};

    LRUExpertCache  cache_;
    std::unique_ptr<ExpertDiskManager> disk_mgr_;

    // Dummy expert for test_mode (registered as submodule for gradients)
    ExpertFFN dummy_expert_{nullptr};

    std::unordered_map<int64_t, ExpertStats> expert_stats_;

    torch::Tensor compute_load_balance_loss(
        const torch::Tensor& expert_indices,
        const torch::Tensor& expert_weights) const;

    double compute_routing_entropy(const torch::Tensor& expert_weights) const;
};

} // namespace xorzen
