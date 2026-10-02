#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <list>
#include <mutex>
#include <unordered_map>
#include "xorzen/types.h"

namespace xorzen {

struct ExpertStats {
    int64_t expert_id = -1;
    int64_t activation_count = 0;
    int64_t cache_hits = 0;
    int64_t cache_misses = 0;
    double avg_weight = 0.0;
    double load_time_ms = 0.0;

    void update(double weight, bool was_cached, double load_ms);
};

struct MoEStats {
    int64_t experts_used = 0;
    int64_t cache_hits = 0;
    int64_t cache_misses = 0;
    double cache_hit_rate = 0.0;
    double total_load_time_ms = 0.0;
    torch::Tensor load_balance_loss;
    double routing_entropy = 0.0;
};

struct ExpertFFNImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t intermediate_dim;
    int64_t expert_id = -1;
    torch::nn::Linear gate_proj{nullptr};
    torch::nn::Linear up_proj{nullptr};
    torch::nn::Linear down_proj{nullptr};
    torch::nn::Dropout dropout{nullptr};

    ExpertFFNImpl(int64_t hidden_dim, int64_t intermediate_dim, double dropout = 0.0, bool bias = false);
    torch::Tensor forward(const torch::Tensor& x);
    int64_t num_parameters() const;
};
TORCH_MODULE(ExpertFFN);

class LRUExpertCache {
public:
    explicit LRUExpertCache(size_t capacity = 24);
    ExpertFFN get(int64_t expert_id);
    void put(int64_t expert_id, ExpertFFN expert);
    void clear();
    std::unordered_map<std::string, double> stats() const;

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::list<int64_t> order_;
    struct Entry {
        ExpertFFN expert{nullptr};
        std::list<int64_t>::iterator it;
    };
    std::unordered_map<int64_t, Entry> cache_;
    mutable int64_t hits_ = 0;
    mutable int64_t misses_ = 0;
    int64_t evictions_ = 0;
};

class ExpertDiskManager {
public:
    ExpertDiskManager(std::filesystem::path shard_dir,
                      int64_t num_experts,
                      int64_t hidden_dim,
                      int64_t intermediate_dim,
                      double dropout = 0.0);
    std::filesystem::path expert_path(int64_t expert_id) const;
    std::filesystem::path expert_mmap_path(int64_t expert_id) const;
    bool expert_exists(int64_t expert_id) const;
    void save_expert(int64_t expert_id, ExpertFFN expert);
    ExpertFFN load_expert(int64_t expert_id, torch::Device device);
    void initialize_all(bool force = false);

private:
    ExpertFFN create_expert(int64_t expert_id) const;
    std::filesystem::path shard_dir_;
    int64_t num_experts_;
    int64_t hidden_dim_;
    int64_t intermediate_dim_;
    double dropout_;
};

struct ShardedExpertFabricImpl : torch::nn::Module {
    ModelConfig config;
    bool test_mode;
    int64_t num_experts;
    int64_t top_k;
    int64_t hidden_dim;
    int64_t intermediate_dim;
    double load_balance_weight;
    std::unique_ptr<ExpertDiskManager> disk_manager;
    std::shared_ptr<LRUExpertCache> cache;
    std::unordered_map<int64_t, ExpertStats> expert_stats;
    ExpertFFN dummy_expert{nullptr};

    explicit ShardedExpertFabricImpl(ModelConfig config, bool test_mode = false);
    std::tuple<torch::Tensor, MoEStats> forward(const torch::Tensor& hidden_states,
                                                const torch::Tensor& expert_indices,
                                                const torch::Tensor& expert_weights,
                                                const torch::Tensor& attention_mask = {});
    torch::Tensor compute_load_balance_loss(const torch::Tensor& expert_indices,
                                            const torch::Tensor& expert_weights) const;
    double routing_entropy(const torch::Tensor& expert_weights) const;
    std::unordered_map<std::string, double> cache_statistics() const;
};
TORCH_MODULE(ShardedExpertFabric);

} // namespace xorzen
