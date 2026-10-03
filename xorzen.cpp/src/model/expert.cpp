#include "xorzen/expert.h"
#include "xorzen/ops.h"
#include "xorzen/optimized/expert_mmap.h"
#include "xorzen/optimized/thread_pool.h"
#include "xorzen/optimized/ggml_bridge.h"
#include "xorzen/optimized/simd_ops.h"

#include <chrono>
#include <set>
#include <mutex>
#include <future>

namespace xorzen {

void ExpertStats::update(double weight, bool was_cached, double load_ms) {
    activation_count++;
    avg_weight += (weight - avg_weight) / static_cast<double>(activation_count);
    if (was_cached) cache_hits++;
    else cache_misses++;
    load_time_ms += load_ms;
}

ExpertFFNImpl::ExpertFFNImpl(int64_t hidden, int64_t intermediate, double dropout_p, bool bias)
    : hidden_dim(hidden), intermediate_dim(intermediate) {
    gate_proj = register_module("gate_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, intermediate_dim).bias(bias)));
    up_proj = register_module("up_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, intermediate_dim).bias(bias)));
    down_proj = register_module("down_proj", torch::nn::Linear(torch::nn::LinearOptions(intermediate_dim, hidden_dim).bias(bias)));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    torch::nn::init::normal_(gate_proj->weight, 0.0, 0.02);
    torch::nn::init::normal_(up_proj->weight, 0.0, 0.02);
    torch::nn::init::normal_(down_proj->weight, 0.0, 0.02);
}

torch::Tensor ExpertFFNImpl::forward(const torch::Tensor& x) {
    return dropout->forward(down_proj->forward(
        optimized::fused_swiglu_simd(gate_proj->forward(x), up_proj->forward(x))
    ));
}

int64_t ExpertFFNImpl::num_parameters() const {
    int64_t total = 0;
    for (const auto& p : parameters()) total += p.numel();
    return total;
}

LRUExpertCache::LRUExpertCache(size_t capacity) : capacity_(std::max<size_t>(1, capacity)) {}

ExpertFFN LRUExpertCache::get(int64_t expert_id) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = cache_.find(expert_id);
    if (it == cache_.end()) {
        misses_++;
        return nullptr;
    }
    order_.splice(order_.end(), order_, it->second.it);
    hits_++;
    return it->second.expert;
}

void LRUExpertCache::put(int64_t expert_id, ExpertFFN expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = cache_.find(expert_id);
    if (it != cache_.end()) {
        it->second.expert = expert;
        order_.splice(order_.end(), order_, it->second.it);
        return;
    }
    order_.push_back(expert_id);
    cache_[expert_id] = Entry{expert, std::prev(order_.end())};
    while (cache_.size() > capacity_) {
        auto oldest = order_.front();
        order_.pop_front();
        cache_.erase(oldest);
        evictions_++;
    }
}

void LRUExpertCache::clear() {
    std::lock_guard<std::mutex> guard(mutex_);
    cache_.clear();
    order_.clear();
}

std::unordered_map<std::string, double> LRUExpertCache::stats() const {
    std::lock_guard<std::mutex> guard(mutex_);
    const double requests = static_cast<double>(hits_ + misses_);
    return {
        {"capacity", static_cast<double>(capacity_)},
        {"current_size", static_cast<double>(cache_.size())},
        {"hits", static_cast<double>(hits_)},
        {"misses", static_cast<double>(misses_)},
        {"evictions", static_cast<double>(evictions_)},
        {"hit_rate", requests > 0.0 ? static_cast<double>(hits_) / requests : 0.0},
    };
}

ExpertDiskManager::ExpertDiskManager(std::filesystem::path shard_dir,
                                     int64_t num_experts,
                                     int64_t hidden_dim,
                                     int64_t intermediate_dim,
                                     double dropout)
    : shard_dir_(std::move(shard_dir)),
      num_experts_(num_experts),
      hidden_dim_(hidden_dim),
      intermediate_dim_(intermediate_dim),
      dropout_(dropout) {
    std::filesystem::create_directories(shard_dir_);
}

std::filesystem::path ExpertDiskManager::expert_path(int64_t expert_id) const {
    char name[64];
    std::snprintf(name, sizeof(name), "expert_%04lld.pt", static_cast<long long>(expert_id));
    return shard_dir_ / name;
}

std::filesystem::path ExpertDiskManager::expert_mmap_path(int64_t expert_id) const {
    char name[64];
    std::snprintf(name, sizeof(name), "expert_%04lld.xorzen_expert", static_cast<long long>(expert_id));
    return shard_dir_ / name;
}

bool ExpertDiskManager::expert_exists(int64_t expert_id) const {
    return std::filesystem::exists(expert_mmap_path(expert_id)) || std::filesystem::exists(expert_path(expert_id));
}

ExpertFFN ExpertDiskManager::create_expert(int64_t expert_id) const {
    auto expert = ExpertFFN(hidden_dim_, intermediate_dim_, dropout_);
    expert->expert_id = expert_id;
    return expert;
}

void ExpertDiskManager::save_expert(int64_t expert_id, ExpertFFN expert) {
    auto mpath = expert_mmap_path(expert_id);
    std::vector<torch::Tensor> weights = {
        expert->gate_proj->weight,
        expert->up_proj->weight,
        expert->down_proj->weight
    };

#ifdef XORZEN_ENABLE_MMAP
    optimized::save_expert_mmap_file(mpath, weights, hidden_dim_, intermediate_dim_);
#else
    torch::serialize::OutputArchive archive;
    expert->save(archive);
    archive.write("expert_id", torch::tensor(expert_id));
    archive.save_to(expert_path(expert_id).string());
#endif
}

ExpertFFN ExpertDiskManager::load_expert(int64_t expert_id, torch::Device device) {
    auto expert = create_expert(expert_id);
    
#ifdef XORZEN_ENABLE_MMAP
    auto mpath = expert_mmap_path(expert_id);
    if (std::filesystem::exists(mpath)) {
        auto region = std::make_shared<optimized::MMapRegion>();
        if (region->open(mpath)) {
            auto [gate, up, down] = optimized::load_expert_mmap_file(*region, device);
            expert->gate_proj->weight.copy_(gate);
            expert->up_proj->weight.copy_(up);
            expert->down_proj->weight.copy_(down);
            return expert;
        }
    }
#endif

    auto path = expert_path(expert_id);
    if (!std::filesystem::exists(path)) {
        save_expert(expert_id, expert);
    } else {
        torch::serialize::InputArchive archive;
        archive.load_from(path.string(), device);
        expert->load(archive);
    }
    expert->to(device);
    return expert;
}

void ExpertDiskManager::initialize_all(bool force) {
    for (int64_t i = 0; i < num_experts_; ++i) {
        if (force || !expert_exists(i)) save_expert(i, create_expert(i));
    }
}

ShardedExpertFabricImpl::ShardedExpertFabricImpl(ModelConfig cfg, bool test)
    : config(std::move(cfg)), test_mode(test) {
    config.normalize();
    num_experts = config.expert_count;
    top_k = config.top_k_experts;
    hidden_dim = config.hidden_size;
    intermediate_dim = static_cast<int64_t>(config.hidden_size * config.expert_hidden_multiplier);
    load_balance_weight = config.load_balancing_weight;
    cache = std::make_shared<LRUExpertCache>(static_cast<size_t>(config.max_expert_cache));
    for (int64_t i = 0; i < num_experts; ++i) expert_stats[i] = ExpertStats{i};
    if (test_mode || !config.use_disk_cache) {
        dummy_expert = register_module("dummy_expert", ExpertFFN(hidden_dim, intermediate_dim, config.dropout));
    } else {
        disk_manager = std::make_unique<ExpertDiskManager>(config.expert_shard_dir, num_experts,
                                                           hidden_dim, intermediate_dim, config.dropout);
    }
}

std::tuple<torch::Tensor, MoEStats> ShardedExpertFabricImpl::forward(const torch::Tensor& hidden_states,
                                                                     const torch::Tensor& expert_indices,
                                                                     const torch::Tensor& expert_weights,
                                                                     const torch::Tensor& attention_mask) {
    const int64_t N = hidden_states.size(0);
    auto output = torch::zeros_like(hidden_states);
    MoEStats stats;

    if (test_mode || !config.use_disk_cache) {
        auto sum_weights = expert_weights.sum(-1, true);
        output = dummy_expert->forward(hidden_states) * sum_weights;
        stats.experts_used = 1;
        stats.load_balance_loss = compute_load_balance_loss(expert_indices, expert_weights);
        return {output, stats};
    }

    std::mutex output_mutex;
    std::mutex stats_mutex;
    std::set<int64_t> used;
    std::mutex used_mutex;

    auto& pool = optimized::global_thread_pool();

    for (int64_t k = 0; k < top_k; ++k) {
        auto slot_indices = expert_indices.select(1, k);
        auto slot_weights = expert_weights.select(1, k);
        
        std::vector<int64_t> unique_experts;
        {
            std::set<int64_t> unique_set;
            auto slot_indices_cpu = slot_indices.to(torch::kCPU).contiguous();
            auto slot_indices_acc = slot_indices_cpu.accessor<int64_t, 1>();
            for (int64_t n = 0; n < slot_indices_cpu.size(0); ++n) {
                unique_set.insert(slot_indices_acc[n]);
            }
            unique_experts.assign(unique_set.begin(), unique_set.end());
        }

        std::vector<std::future<void>> futures;
        for (int64_t expert_id : unique_experts) {
            futures.push_back(pool.submit([&, expert_id, slot_indices, slot_weights]() {
                auto mask = slot_indices == expert_id;
                if (!mask.any().item<bool>()) return;
                
                auto token_indices = torch::nonzero(mask).squeeze(1);
                auto token_hidden = hidden_states.index_select(0, token_indices);
                auto token_weights = slot_weights.index_select(0, token_indices).unsqueeze(1);

                auto t0 = std::chrono::high_resolution_clock::now();
                auto expert = cache->get(expert_id);
                bool was_cached = !expert.is_empty();
                
                if (expert.is_empty()) {
                    expert = disk_manager->load_expert(expert_id, hidden_states.device());
                    cache->put(expert_id, expert);
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    stats.cache_misses++;
                } else {
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    stats.cache_hits++;
                }
                
                auto t1 = std::chrono::high_resolution_clock::now();
                double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                
                {
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    stats.total_load_time_ms += load_ms;
                }
                
                {
                    std::lock_guard<std::mutex> lock(used_mutex);
                    used.insert(expert_id);
                }

                auto expert_output = expert->forward(token_hidden) * token_weights;
                
                {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    auto prior = output.index_select(0, token_indices);
                    output.index_put_({token_indices}, prior + expert_output);
                }

                {
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    expert_stats[expert_id].update(token_weights.mean().item<double>(), was_cached, load_ms);
                }
            }));
        }
        
        for (auto& f : futures) f.get();
    }

    auto total_weight = expert_weights.sum(1, true);
    output = output / (total_weight + 1e-8);
    if (attention_mask.defined()) output = output * attention_mask.unsqueeze(-1).to(output.dtype());
    stats.experts_used = static_cast<int64_t>(used.size());
    auto requests = stats.cache_hits + stats.cache_misses;
    stats.cache_hit_rate = requests > 0 ? static_cast<double>(stats.cache_hits) / static_cast<double>(requests) : 0.0;
    stats.load_balance_loss = compute_load_balance_loss(expert_indices, expert_weights);
    stats.routing_entropy = routing_entropy(expert_weights);
    return {output, stats};
}

torch::Tensor ShardedExpertFabricImpl::compute_load_balance_loss(const torch::Tensor& expert_indices,
                                                                 const torch::Tensor& expert_weights) const {
    const int64_t N = expert_indices.size(0);
    auto usage = torch::zeros({num_experts}, expert_weights.options());
    for (int64_t k = 0; k < top_k; ++k) {
        usage.scatter_add_(0, expert_indices.select(1, k), expert_weights.select(1, k));
    }
    auto probs = usage / (static_cast<double>(N * top_k) + 1e-8);
    auto target = 1.0 / static_cast<double>(num_experts);
    return load_balance_weight * (probs - target).pow(2).sum();
}

double ShardedExpertFabricImpl::routing_entropy(const torch::Tensor& expert_weights) const {
    auto avg = expert_weights.mean(0);
    auto probs = avg / (avg.sum() + 1e-8);
    return (-(probs * torch::log(probs + 1e-8)).sum()).item<double>();
}

std::unordered_map<std::string, double> ShardedExpertFabricImpl::cache_statistics() const {
    return cache->stats();
}

} // namespace xorzen
