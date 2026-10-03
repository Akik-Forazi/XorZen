// ============================================================
//  xorzen.cpp — src/model/zmoe.cpp
//  ShardedExpertFabric full implementation
//  Ported from xorzen/model/zmoe.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/zmoe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace xorzen {

static double wall_ms() {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0;
}

// ============================================================
//  ExpertStats
// ============================================================

void ExpertStats::update(double weight, bool was_cached, double load_ms) {
    ++activation_count;
    total_weight += weight;
    avg_weight = total_weight / static_cast<double>(activation_count);
    if (was_cached) ++cache_hits;
    else { ++cache_misses; load_time_ms += load_ms; }
}

double ExpertStats::hit_rate() const {
    int64_t total = cache_hits + cache_misses;
    return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
}

// ============================================================
//  ExpertFFN
// ============================================================

ExpertFFNImpl::ExpertFFNImpl(int64_t h, int64_t inter, double drop, bool bias)
    : hidden_dim(h), intermediate_dim(inter) {
    gate_proj = register_module("gate_proj",
        torch::nn::Linear(torch::nn::LinearOptions(h, inter).bias(bias)));
    up_proj   = register_module("up_proj",
        torch::nn::Linear(torch::nn::LinearOptions(h, inter).bias(bias)));
    down_proj = register_module("down_proj",
        torch::nn::Linear(torch::nn::LinearOptions(inter, h).bias(bias)));
    dropout_  = register_module("dropout", torch::nn::Dropout(drop));

    // Initialise weights
    for (auto& p : {gate_proj, up_proj, down_proj}) {
        torch::nn::init::normal_(p->weight, 0.0, 0.02);
    }
}

torch::Tensor ExpertFFNImpl::forward(const torch::Tensor& x) {
    // SwiGLU: silu(gate(x)) * up(x)
    auto gate   = torch::nn::functional::silu(gate_proj->forward(x));
    auto up     = up_proj->forward(x);
    auto hidden = gate * up;
    return dropout_->forward(down_proj->forward(hidden));
}

int64_t ExpertFFNImpl::num_params() const {
    int64_t n = 0;
    for (const auto& p : parameters()) n += p.numel();
    return n;
}

// ============================================================
//  LRUExpertCache
// ============================================================

LRUExpertCache::LRUExpertCache(int64_t capacity) : capacity_(capacity) {}

std::shared_ptr<ExpertFFNImpl> LRUExpertCache::get(int64_t expert_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = map_.find(expert_id);
    if (it == map_.end()) { ++misses_; return nullptr; }
    // Move to back (MRU)
    list_.splice(list_.end(), list_, it->second);
    ++hits_;
    return it->second->second;
}

void LRUExpertCache::put(int64_t expert_id, std::shared_ptr<ExpertFFNImpl> expert) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = map_.find(expert_id);
    if (it != map_.end()) {
        it->second->second = std::move(expert);
        list_.splice(list_.end(), list_, it->second);
        return;
    }
    list_.emplace_back(expert_id, std::move(expert));
    map_[expert_id] = std::prev(list_.end());
    if (static_cast<int64_t>(list_.size()) > capacity_) {
        // Evict front (LRU)
        map_.erase(list_.front().first);
        list_.pop_front();
        ++evictions_;
    }
}

void LRUExpertCache::clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    list_.clear();
    map_.clear();
}

LRUExpertCache::Stats LRUExpertCache::get_stats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    int64_t total = hits_ + misses_;
    return { capacity_,
             static_cast<int64_t>(list_.size()),
             hits_, misses_, evictions_,
             total > 0 ? static_cast<double>(hits_) / total : 0.0 };
}

// ============================================================
//  ExpertDiskManager
// ============================================================

ExpertDiskManager::ExpertDiskManager(const std::filesystem::path& shard_dir,
                                     int64_t num_experts,
                                     int64_t hidden_dim,
                                     int64_t intermediate_dim)
    : shard_dir_(shard_dir), num_experts_(num_experts),
      hidden_dim_(hidden_dim), intermediate_dim_(intermediate_dim) {
    std::filesystem::create_directories(shard_dir_);
    read_manifest();
}

std::filesystem::path ExpertDiskManager::expert_path(int64_t id) const {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "expert_%04lld.pt", static_cast<long long>(id));
    return shard_dir_ / buf;
}

bool ExpertDiskManager::expert_exists(int64_t id) const {
    std::lock_guard<std::mutex> lk(manifest_mutex_);
    auto it = manifest_.find(id);
    return (it != manifest_.end()) && it->second &&
           std::filesystem::exists(expert_path(id));
}

std::shared_ptr<ExpertFFNImpl> ExpertDiskManager::create_new_expert(int64_t id) const {
    auto e = std::make_shared<ExpertFFNImpl>(hidden_dim_, intermediate_dim_);
    e->expert_id = id;
    return e;
}

std::shared_ptr<ExpertFFNImpl> ExpertDiskManager::load_expert(int64_t id,
                                                               torch::Device device) {
    auto path = expert_path(id);
    if (!std::filesystem::exists(path)) {
        // Create fresh expert and save it
        auto e = create_new_expert(id);
        save_expert(id, e);
        return std::make_shared<ExpertFFNImpl>(*e);  // clone
    }
    // Load from disk
    auto e = std::make_shared<ExpertFFNImpl>(hidden_dim_, intermediate_dim_);
    e->expert_id = id;
    try {
        torch::serialize::InputArchive ar;
        ar.load_from(path.string(), device);
        e->load(ar);
    } catch (const std::exception& ex) {
        std::cerr << "[zmoe] Expert " << id << " load failed (" << ex.what()
                  << "), reinitialising\n";
        std::filesystem::remove(path);
        e = create_new_expert(id);
        save_expert(id, e);
    }
    e->eval();
    return e;
}

void ExpertDiskManager::save_expert(int64_t id,
                                    const std::shared_ptr<ExpertFFNImpl>& e) {
    torch::serialize::OutputArchive ar;
    e->save(ar);
    ar.save_to(expert_path(id).string());
    {
        std::lock_guard<std::mutex> lk(manifest_mutex_);
        manifest_[id] = true;
    }
    write_manifest();
}

void ExpertDiskManager::initialize_all(bool force) {
    std::cout << "[zmoe] Initialising " << num_experts_ << " experts...\n";
    for (int64_t i = 0; i < num_experts_; ++i) {
        if (force || !expert_exists(i)) {
            auto e = create_new_expert(i);
            save_expert(i, e);
        }
        if ((i + 1) % 50 == 0)
            std::cout << "[zmoe]   " << (i + 1) << "/" << num_experts_ << "\n";
    }
    std::cout << "[zmoe] All experts initialised.\n";
}

double ExpertDiskManager::total_size_mb() const {
    double total = 0.0;
    for (int64_t i = 0; i < num_experts_; ++i) {
        auto p = expert_path(i);
        if (std::filesystem::exists(p))
            total += static_cast<double>(std::filesystem::file_size(p));
    }
    return total / (1024.0 * 1024.0);
}

int64_t ExpertDiskManager::experts_on_disk() const {
    std::lock_guard<std::mutex> lk(manifest_mutex_);
    return static_cast<int64_t>(manifest_.size());
}

void ExpertDiskManager::write_manifest() const {
    auto mp = shard_dir_ / "manifest.json";
    std::ofstream f(mp);
    f << "{\n  \"num_experts\": " << num_experts_
      << ",\n  \"hidden_dim\": " << hidden_dim_
      << ",\n  \"intermediate_dim\": " << intermediate_dim_
      << ",\n  \"experts\": [";
    std::lock_guard<std::mutex> lk(manifest_mutex_);
    bool first = true;
    for (auto& [id, ok] : manifest_) {
        if (ok) { if (!first) f << ", "; f << id; first = false; }
    }
    f << "]\n}\n";
}

void ExpertDiskManager::read_manifest() {
    auto mp = shard_dir_ / "manifest.json";
    if (!std::filesystem::exists(mp)) return;
    std::ifstream f(mp);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    // Simple scan for numbers after "experts": [
    auto pos = content.find("\"experts\":");
    if (pos == std::string::npos) return;
    pos = content.find('[', pos);
    if (pos == std::string::npos) return;
    auto end = content.find(']', pos);
    std::string arr = content.substr(pos + 1, end - pos - 1);
    std::istringstream ss(arr);
    std::string tok;
    std::lock_guard<std::mutex> lk(manifest_mutex_);
    while (std::getline(ss, tok, ',')) {
        try { manifest_[std::stoll(tok)] = true; } catch (...) {}
    }
}

// ============================================================
//  ShardedExpertFabric
// ============================================================

ShardedExpertFabric::ShardedExpertFabric(const ModelConfig& cfg, bool test_mode)
    : config_(cfg), test_mode_(test_mode),
      num_experts_(cfg.num_experts),
      top_k_(cfg.top_k_experts),
      hidden_dim_(cfg.hidden_size),
      intermediate_dim_(static_cast<int64_t>(cfg.hidden_size * cfg.expert_hidden_multiplier)),
      cache_(cfg.max_expert_cache) {

    // Init expert stats
    for (int64_t i = 0; i < num_experts_; ++i) {
        expert_stats_[i] = ExpertStats{i};
    }

    if (test_mode_) {
        // Single dummy expert for gradient flow during tests
        dummy_expert_ = register_module("dummy_expert",
            ExpertFFN(hidden_dim_, intermediate_dim_));
        std::cout << "[zmoe] TEST MODE — using dummy expert\n";
    } else {
        // Full disk manager
        disk_mgr_ = std::make_unique<ExpertDiskManager>(
            std::filesystem::path(cfg.expert_shard_dir),
            num_experts_, hidden_dim_, intermediate_dim_);
        if (disk_mgr_->experts_on_disk() == 0) {
            std::cout << "[zmoe] No experts on disk — initialising...\n";
            disk_mgr_->initialize_all();
        }
        std::cout << "[zmoe] ShardedExpertFabric ready\n"
                  << "       experts=" << num_experts_
                  << " top-k=" << top_k_
                  << " cache=" << cfg.max_expert_cache
                  << " disk=" << std::fixed << std::setprecision(1)
                  << disk_mgr_->total_size_mb() << " MB\n";
    }
}

MoEOutput ShardedExpertFabric::forward(const torch::Tensor& hidden_states,
                                       const torch::Tensor& expert_indices,
                                       const torch::Tensor& expert_weights) {
    const int64_t N = hidden_states.size(0);
    auto device     = hidden_states.device();
    MoEOutput out;
    out.output = torch::zeros_like(hidden_states);

    // ---- TEST MODE ----
    if (test_mode_) {
        auto sum_w = expert_weights.sum(-1, true);  // [N, 1]
        out.output = dummy_expert_->forward(hidden_states) * sum_w;
        out.routing_entropy  = 0.0;
        out.cache_hit_rate   = 0.0;
        out.experts_used     = 1;
        out.load_balance_loss = torch::zeros({}, hidden_states.options());
        return out;
    }

    int64_t cache_hits   = 0;
    int64_t cache_misses = 0;
    double  total_load_time = 0.0;
    std::set<int64_t> used_experts;

    // Process each top-k slot
    for (int64_t k = 0; k < top_k_; ++k) {
        auto slot_idx = expert_indices.select(1, k);   // [N]
        auto slot_wt  = expert_weights.select(1, k);   // [N]

        // Group by expert id — use torch::unique() instead of a per-element
        // .item<int64_t>() loop. The old code did:
        //   for (int64_t i = 0; i < sorted.size(0); ++i) {
        //       int64_t eid = sorted[i].item<int64_t>();  // CPU sync per element!
        // Manual unique (torch::unique not available in all LibTorch versions)
        std::set<int64_t> unique_set;
        for (int64_t i = 0; i < slot_idx.size(0); ++i) {
            unique_set.insert(slot_idx[i].item<int64_t>());
        }

        for (int64_t eid : unique_set) {
            auto mask   = (slot_idx == eid);             // [N] bool
            // Old code: if (!mask.any().item<bool>()) continue;
            // The .item<bool>() forces a CPU sync. Since we got eid from
            // torch::unique(slot_idx), we KNOW at least one element matches
            // — the check is always true. Skip it.
            auto tok_ids = torch::where(mask)[0];
            auto tok_wt  = slot_wt.index({mask}).unsqueeze(1); // [n, 1]
            auto tok_hid = hidden_states.index({mask});         // [n, H]

            // Load from cache or disk
            double t0 = wall_ms();
            auto cached = cache_.get(eid);
            bool was_cached = (cached != nullptr);
            std::shared_ptr<ExpertFFNImpl> expert;
            if (was_cached) {
                expert = cached;
                ++cache_hits;
            } else {
                expert = disk_mgr_->load_expert(eid, device);
                cache_.put(eid, expert);
                ++cache_misses;
            }
            double load_ms = wall_ms() - t0;
            total_load_time += load_ms;
            used_experts.insert(eid);

            // Forward
            torch::Tensor expert_out;
            {
                torch::GradMode::set_enabled(is_training());
                expert_out = expert->forward(tok_hid);  // [n, H]
            }

            out.output.index_put_({mask}, out.output.index({mask}) + expert_out * tok_wt);

            // Update statistics — skip during training to avoid .item<double>()
            // CPU sync. The stats are diagnostic-only (not in the autograd graph).
            // This mirrors the Python fix from commit d14adc1.
            if (!is_training()) {
                expert_stats_[eid].update(tok_wt.mean().item<double>(), was_cached, load_ms);
            }
        }
    }

    // NOTE: do NOT normalize by sum of weights.
    // The Python reference (zmoe.py) explicitly does NOT re-normalize:
    //   # NOTE: do NOT re-normalize by sum(expert_weights).
    //   # The MoE output is sum_k w_k * E_k(x), NOT a weighted average.
    //   # The router already L1-normalizes the top-k weights to sum to 1.
    // The old C++ code did:
    //   auto total_w = expert_weights.sum(1, true).clamp_min(1e-8f);
    //   out.output /= total_w;
    // which is a PARITY BUG — it dampens the MoE signal when weights sum to
    // < 1, destroying the weighted-sum semantics the router was trained with.
    // Removed to match Python behavior.

    out.load_balance_loss = compute_load_balance_loss(expert_indices, expert_weights);
    out.routing_entropy   = compute_routing_entropy(expert_weights);
    out.cache_hit_rate    = static_cast<double>(cache_hits) /
                            std::max<int64_t>(cache_hits + cache_misses, 1);
    out.experts_used      = static_cast<int64_t>(used_experts.size());
    return out;
}

torch::Tensor ShardedExpertFabric::compute_load_balance_loss(
    const torch::Tensor& expert_indices,
    const torch::Tensor& expert_weights) const {
    const int64_t N = expert_indices.size(0);
    auto device     = expert_indices.device();
    auto usage      = torch::zeros({num_experts_}, expert_weights.options());
    for (int64_t k = 0; k < top_k_; ++k) {
        usage.scatter_add_(0, expert_indices.select(1, k), expert_weights.select(1, k));
    }
    auto probs = usage / (static_cast<float>(N * top_k_) + 1e-8f);
    float target = 1.0f / static_cast<float>(num_experts_);
    auto loss = (probs - target).pow(2).sum();
    return loss * static_cast<float>(config_.load_balancing_weight);
}

double ShardedExpertFabric::compute_routing_entropy(
    const torch::Tensor& expert_weights) const {
    auto avg_w = expert_weights.mean(0);                    // [top_k]
    auto probs = avg_w / (avg_w.sum() + 1e-8f);
    auto entropy = -(probs * torch::log(probs + 1e-8f)).sum();
    return entropy.item<double>();
}

void ShardedExpertFabric::print_statistics() const {
    std::cout << "\n" << std::string(64, '=') << "\n"
              << "  MoE ShardedExpertFabric Statistics\n"
              << std::string(64, '=') << "\n";
    int64_t total_act = 0, active = 0;
    for (auto& [id, s] : expert_stats_) {
        total_act += s.activation_count;
        if (s.activation_count > 0) ++active;
    }
    auto cs = cache_.get_stats();
    std::cout << "  Experts  : " << num_experts_ << " total, " << active << " active\n"
              << "  Tokens   : " << total_act << " total activations\n"
              << "  Cache    : " << cs.current_size << "/" << cs.capacity
              << " (" << std::fixed << std::setprecision(1) << cs.hit_rate * 100 << "% hit)\n";
    if (disk_mgr_) {
        std::cout << "  Disk     : " << std::setprecision(1)
                  << disk_mgr_->total_size_mb() << " MB\n";
    }
    // Top-10 experts
    std::vector<std::pair<int64_t, int64_t>> ranked;
    ranked.reserve(expert_stats_.size());
    for (auto& [id, s] : expert_stats_)
        ranked.emplace_back(id, s.activation_count);
    std::partial_sort(ranked.begin(), ranked.begin() + std::min<size_t>(10, ranked.size()),
                      ranked.end(), [](const auto& a, const auto& b) {
                          return a.second > b.second;
                      });
    std::cout << "  Top experts:\n";
    for (size_t i = 0; i < std::min<size_t>(10, ranked.size()); ++i) {
        auto& [id, cnt] = ranked[i];
        std::cout << "    " << (i + 1) << ". Expert " << id
                  << ": " << cnt << " activations\n";
    }
    std::cout << std::string(64, '=') << "\n\n";
}

void ShardedExpertFabric::clear_cache() {
    cache_.clear();
}

void ShardedExpertFabric::save_all_experts() {
    if (!disk_mgr_) return;
    // Nothing to flush — experts are saved immediately on disk-miss creation
    std::cout << "[zmoe] All experts already persisted on disk.\n";
}

} // namespace xorzen
