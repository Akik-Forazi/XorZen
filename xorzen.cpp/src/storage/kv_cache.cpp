// ============================================================
//  xorzen.cpp — src/storage/kv_cache.cpp
//  Disk-Sharded Infinite Context KV Cache
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/kv_cache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace xorzen {

// ============================================================
//  KVBlock
// ============================================================
int64_t KVBlock::size_bytes() const {
    int64_t n = 0;
    if (k_compressed.defined()) n += k_compressed.numel() * (is_quantized ? 1 : 2);
    if (v_compressed.defined()) n += v_compressed.numel() * (is_quantized ? 1 : 2);
    if (k_basis.defined())      n += k_basis.numel() * 2;
    if (v_basis.defined())      n += v_basis.numel() * 2;
    if (summary.defined())      n += summary.numel() * 4;
    return n;
}

// ============================================================
//  DiskKVIndex
// ============================================================
DiskKVIndex::DiskKVIndex(int64_t num_layers, int64_t head_dim)
    : num_layers_(num_layers), head_dim_(head_dim) {}

void DiskKVIndex::add(int64_t block_id, const torch::Tensor& summary) {
    std::lock_guard<std::mutex> lk(mutex_);
    // Flatten: [L, H] → [L*H]
    auto flat = summary.reshape({-1}).to(torch::kFloat32);
    block_ids_.push_back(block_id);
    if (!summaries_.defined()) {
        summaries_ = flat.unsqueeze(0);
    } else {
        summaries_ = torch::cat({summaries_, flat.unsqueeze(0)}, 0);
    }
    dirty_ = true;
}

void DiskKVIndex::remove(int64_t block_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = std::find(block_ids_.begin(), block_ids_.end(), block_id);
    if (it == block_ids_.end()) return;
    int64_t idx = std::distance(block_ids_.begin(), it);
    block_ids_.erase(it);
    // Remove row from summaries
    if (summaries_.size(0) == 1) {
        summaries_ = torch::Tensor{};
    } else {
        auto before = summaries_.slice(0, 0, idx);
        auto after  = summaries_.slice(0, idx + 1);
        summaries_  = torch::cat({before, after}, 0);
    }
    dirty_ = true;
}

void DiskKVIndex::rebuild_if_needed() const {
    if (!dirty_ || !summaries_.defined()) return;
    // L2-normalise rows for cosine similarity
    summaries_norm_ = torch::nn::functional::normalize(summaries_,
        torch::nn::functional::NormalizeFuncOptions().dim(1));
    dirty_ = false;
}

std::vector<int64_t> DiskKVIndex::query(const torch::Tensor& query_key,
                                        int64_t top_k) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (block_ids_.empty() || !summaries_.defined()) return {};
    rebuild_if_needed();

    // query_key: [L, H] → flatten & normalise
    auto q = torch::nn::functional::normalize(
        query_key.reshape({-1}).to(torch::kFloat32).unsqueeze(0),
        torch::nn::functional::NormalizeFuncOptions().dim(1)); // [1, L*H]

    // Cosine scores: [1, N] via matmul with normalised summaries
    auto scores = torch::mm(q, summaries_norm_.t()).squeeze(0); // [N]

    top_k = std::min(top_k, static_cast<int64_t>(block_ids_.size()));
    auto [vals, idxs] = torch::topk(scores, top_k);
    auto idxs_v = idxs.to(torch::kCPU);

    std::vector<int64_t> result;
    result.reserve(top_k);
    for (int64_t i = 0; i < top_k; ++i)
        result.push_back(block_ids_[idxs_v[i].item<int64_t>()]);
    return result;
}

size_t DiskKVIndex::size() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return block_ids_.size();
}

void DiskKVIndex::clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    block_ids_.clear();
    summaries_ = torch::Tensor{};
    summaries_norm_ = torch::Tensor{};
    dirty_ = true;
}

// ============================================================
//  DiskKVStore
// ============================================================
DiskKVStore::DiskKVStore(const std::filesystem::path& dir,
                         const KVCacheConfig& cfg)
    : dir_(dir), cfg_(cfg) {
    std::filesystem::create_directories(dir_);
}

std::filesystem::path DiskKVStore::block_path(int64_t id) const {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "kv_%08lld.bin", static_cast<long long>(id));
    return dir_ / buf;
}

int64_t DiskKVStore::save_block(const KVBlock& blk) {
    auto path = block_path(blk.block_id);
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot write KV block: " + path.string());

    // Header
    const uint32_t MAGIC = 0x584B5643; // "XKVC"
    f.write(reinterpret_cast<const char*>(&MAGIC), 4);
    f.write(reinterpret_cast<const char*>(&blk.block_id),   8);
    f.write(reinterpret_cast<const char*>(&blk.num_tokens),  8);
    f.write(reinterpret_cast<const char*>(&blk.num_layers),  8);
    uint8_t quant = blk.is_quantized ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&quant), 1);

    // Helper lambda: write tensor
    auto write_tensor = [&](const torch::Tensor& t) {
        auto ct = t.to(torch::kCPU).contiguous();
        uint32_t ndim = ct.dim();
        f.write(reinterpret_cast<const char*>(&ndim), 4);
        for (int i = 0; i < ct.dim(); ++i) {
            int64_t s = ct.size(i);
            f.write(reinterpret_cast<const char*>(&s), 8);
        }
        int64_t nb = ct.numel() * ct.element_size();
        f.write(reinterpret_cast<const char*>(ct.data_ptr()), nb);
    };

    write_tensor(blk.summary);
    write_tensor(blk.k_compressed);
    write_tensor(blk.v_compressed);
    write_tensor(blk.k_basis);
    write_tensor(blk.v_basis);

    int64_t bytes = static_cast<int64_t>(f.tellp());
    return bytes;
}

KVBlock DiskKVStore::load_block(int64_t id) const {
    auto path = block_path(id);
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("KV block not found: " + path.string());

    uint32_t magic; f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x584B5643) throw std::runtime_error("Bad KV block magic");

    KVBlock blk;
    f.read(reinterpret_cast<char*>(&blk.block_id),  8);
    f.read(reinterpret_cast<char*>(&blk.num_tokens), 8);
    f.read(reinterpret_cast<char*>(&blk.num_layers), 8);
    uint8_t quant; f.read(reinterpret_cast<char*>(&quant), 1);
    blk.is_quantized = (quant != 0);

    auto read_tensor = [&]() {
        uint32_t ndim; f.read(reinterpret_cast<char*>(&ndim), 4);
        std::vector<int64_t> shape(ndim);
        for (uint32_t i = 0; i < ndim; ++i)
            f.read(reinterpret_cast<char*>(&shape[i]), 8);
        int64_t numel = 1;
        for (auto s : shape) numel *= s;
        // Detect dtype from quantization flag: int8 for compressed, float32 for basis/summary
        // We store summary and basis as float32, compressed as int8 if quantized
        // Simplified: all float32 for now (INT8 quantization stored as float for portability)
        auto t = torch::empty(shape, torch::kFloat32);
        int64_t nb = numel * sizeof(float);
        f.read(reinterpret_cast<char*>(t.data_ptr<float>()), nb);
        return t;
    };

    blk.summary      = read_tensor();
    blk.k_compressed = read_tensor();
    blk.v_compressed = read_tensor();
    blk.k_basis      = read_tensor();
    blk.v_basis      = read_tensor();
    return blk;
}

void DiskKVStore::delete_block(int64_t id) {
    auto path = block_path(id);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

bool DiskKVStore::block_exists(int64_t id) const {
    return std::filesystem::exists(block_path(id));
}

int64_t DiskKVStore::num_blocks() const {
    int64_t n = 0;
    for (auto& e : std::filesystem::directory_iterator(dir_))
        if (e.path().extension() == ".bin") ++n;
    return n;
}

double DiskKVStore::total_size_mb() const {
    double total = 0;
    for (auto& e : std::filesystem::directory_iterator(dir_))
        if (e.path().extension() == ".bin")
            total += static_cast<double>(e.file_size());
    return total / (1024.0 * 1024.0);
}

// ============================================================
//  DiskShardedKVCache
// ============================================================

DiskShardedKVCache::DiskShardedKVCache(const KVCacheConfig& cfg)
    : cfg_(cfg),
      disk_store_(cfg.cache_dir / cfg.session_id, cfg),
      disk_index_(cfg.num_layers, cfg.head_dim) {

    // Initialise empty active window per layer
    active_k_.resize(cfg.num_layers);
    active_v_.resize(cfg.num_layers);

    if (cfg.async_compress) {
        compress_running_ = true;
        compress_thread_  = std::thread(&DiskShardedKVCache::compress_worker, this);
    }

    std::cout << "[KVCache] Disk-sharded KV cache initialised\n"
              << "         session=" << cfg.session_id
              << " active=" << cfg.active_tokens
              << " ram_blocks=" << cfg.ram_blocks
              << " svd_rank=" << cfg.svd_rank << "\n"
              << "         bytes/token ≈ " << std::fixed << std::setprecision(1)
              << cfg.bytes_per_token_compressed() << " B\n";
}

DiskShardedKVCache::~DiskShardedKVCache() {
    compress_running_ = false;
    if (compress_thread_.joinable()) compress_thread_.join();
}

void DiskShardedKVCache::append(const torch::Tensor& k_new,
                                const torch::Tensor& v_new) {
    // k_new: [B, L_new, H_kv, D]  — take B=0 for simplicity (single sequence)
    // Reshape to [L_new, H_kv, D] (batch dim removed)
    auto k = k_new.squeeze(0).to(torch::kFloat32).contiguous(); // [L_new, H_kv, D]
    auto v = v_new.squeeze(0).to(torch::kFloat32).contiguous();

    // k is [L_new, num_kv_heads, head_dim] but we want per-layer:
    // Actually external format is [B, seq, layers, H_kv, D] — but for simpler
    // interface, we treat k as [seq_len, num_layers, H_kv, D] internally.
    // Here we do: k shape is [seq, H_kv, D], replicated across layers is the
    // caller's job. Cache accepts [num_layers, seq, H_kv, D] tensors.
    // For the simple port: treat first dim as layer for now.
    int64_t L_new = k.size(0);

    {
        std::lock_guard<std::mutex> lk(active_mutex_);
        for (int64_t l = 0; l < cfg_.num_layers; ++l) {
            if (!active_k_[l].defined()) {
                active_k_[l] = k.unsqueeze(0);  // [1, H, D] as placeholder
                active_v_[l] = v.unsqueeze(0);
            } else {
                active_k_[l] = torch::cat({active_k_[l], k.unsqueeze(0)}, 0);
                active_v_[l] = torch::cat({active_v_[l], v.unsqueeze(0)}, 0);
            }
        }
        active_tokens_ += L_new;
    }

    // If active window exceeds limit, push oldest block to RAM
    if (active_tokens_ > cfg_.active_tokens) {
        evict_active_to_ram();
    }
}

std::pair<torch::Tensor, torch::Tensor>
DiskShardedKVCache::retrieve(const torch::Tensor& query_key) {
    // query_key: [B, L_q, H_kv, D] → use mean over L_q and B
    auto q_mean = query_key.mean(torch::IntArrayRef{0, 1}).to(torch::kFloat32); // [H_kv, D]

    // Build a fake [num_layers, head_dim] summary by repeating
    auto q_summary = q_mean.unsqueeze(0).expand({cfg_.num_layers, cfg_.head_dim});

    // Query disk index for top-K blocks
    auto top_block_ids = disk_index_.query(q_summary, cfg_.retrieval_top_k);

    // Collect retrieved K, V tensors
    std::vector<torch::Tensor> k_parts, v_parts;

    // 1. Disk retrieved blocks
    for (int64_t bid : top_block_ids) {
        try {
            auto blk = disk_store_.load_block(bid);
            auto [k_blk, v_blk] = decompress_block(blk);
            // k_blk: [L, T, H, D] → mean over layers → [T, H, D]
            k_parts.push_back(k_blk.mean(0));
            v_parts.push_back(v_blk.mean(0));
            ++retrieval_hits_;
        } catch (...) {
            ++retrieval_misses_;
        }
    }

    // 2. Active window (always included)
    {
        std::lock_guard<std::mutex> lk(active_mutex_);
        if (!active_k_.empty() && active_k_[0].defined()) {
            k_parts.push_back(active_k_[0]);
            v_parts.push_back(active_v_[0]);
        }
    }

    if (k_parts.empty()) {
        // Nothing in cache yet
        auto empty = torch::zeros({1, 0, cfg_.num_kv_heads, cfg_.head_dim});
        return {empty, empty};
    }

    auto k_ctx = torch::cat(k_parts, 0).unsqueeze(0); // [1, T_ctx, H, D]
    auto v_ctx = torch::cat(v_parts, 0).unsqueeze(0);
    return {k_ctx, v_ctx};
}

KVBlock DiskShardedKVCache::compress_block(int64_t block_id,
                                           const torch::Tensor& k,
                                           const torch::Tensor& v) const {
    // k, v: [T, H_kv, D]
    KVBlock blk;
    blk.block_id  = block_id;
    blk.num_tokens = k.size(0);
    blk.num_layers = 1;  // simplified: one block = one layer slice

    // Mean summary for retrieval
    blk.summary = k.mean(0).mean(0).unsqueeze(0); // [1, D]

    // SVD compression: reshape [T, H*D], take top-R singular vectors
    auto k_flat = k.reshape({k.size(0), -1}).to(torch::kFloat32); // [T, H*D]
    auto v_flat = v.reshape({v.size(0), -1}).to(torch::kFloat32);

    int64_t rank = std::min(cfg_.svd_rank, std::min(k_flat.size(0), k_flat.size(1)));

    // SVD: k_flat = U @ S @ Vt  →  compressed = U[:, :R] * S[:R]
    auto [Uk, Sk, Vkt] = at::linalg_svd(k_flat, /*full_matrices=*/false);
    auto [Uv, Sv, Vvt] = at::linalg_svd(v_flat, false);

    blk.k_compressed = (Uk.slice(1, 0, rank) * Sk.slice(0, 0, rank).unsqueeze(0))
                           .to(torch::kFloat32); // [T, R]
    blk.v_compressed = (Uv.slice(1, 0, rank) * Sv.slice(0, 0, rank).unsqueeze(0))
                           .to(torch::kFloat32);
    blk.k_basis = Vkt.slice(0, 0, rank).to(torch::kFloat32); // [R, H*D]
    blk.v_basis = Vvt.slice(0, 0, rank).to(torch::kFloat32);
    blk.is_quantized = false;
    return blk;
}

std::pair<torch::Tensor, torch::Tensor>
DiskShardedKVCache::decompress_block(const KVBlock& blk) const {
    // Reconstruct: k_flat = k_compressed @ k_basis  →  [T, H*D]
    auto k_flat = torch::mm(blk.k_compressed, blk.k_basis); // [T, H*D]
    auto v_flat = torch::mm(blk.v_compressed, blk.v_basis);

    // Reshape back to [1, T, H_kv, D] with a layer dim
    auto shape = std::vector<int64_t>{1, blk.num_tokens, cfg_.num_kv_heads, cfg_.head_dim};
    auto k = k_flat.reshape(shape);
    auto v = v_flat.reshape(shape);
    return {k, v};
}

void DiskShardedKVCache::evict_active_to_ram() {
    std::lock_guard<std::mutex> lk(active_mutex_);
    if (!active_k_[0].defined() || active_k_[0].size(0) < cfg_.block_tokens) return;

    // Take first block_tokens from active window
    int64_t T = cfg_.block_tokens;
    auto k_blk = active_k_[0].slice(0, 0, T).clone(); // [T, H, D]
    auto v_blk = active_v_[0].slice(0, 0, T).clone();

    // Trim active window
    for (int64_t l = 0; l < cfg_.num_layers; ++l) {
        if (!active_k_[l].defined()) continue;
        active_k_[l] = active_k_[l].slice(0, T);
        active_v_[l] = active_v_[l].slice(0, T);
    }
    active_tokens_ -= T;

    // Compress and push to RAM
    int64_t bid = next_block_id_++;
    auto blk    = compress_block(bid, k_blk, v_blk);

    // Add to RAM pool
    {
        std::lock_guard<std::mutex> rlk(ram_mutex_);
        ram_list_.push_back({bid, k_blk.unsqueeze(0), v_blk.unsqueeze(0)});
        ram_map_[bid] = std::prev(ram_list_.end());

        // If RAM pool full, evict LRU to disk
        if (static_cast<int64_t>(ram_list_.size()) > cfg_.ram_blocks) {
            evict_ram_to_disk();
        }
    }
}

void DiskShardedKVCache::evict_ram_to_disk() {
    // Must be called with ram_mutex_ held
    if (ram_list_.empty()) return;
    auto& front = ram_list_.front();
    int64_t bid = front.block_id;

    // Compress and save
    auto k = front.k.squeeze(0); // [T, H, D]
    auto v = front.v.squeeze(0);
    auto blk = compress_block(bid, k, v);

    if (cfg_.async_compress) {
        std::lock_guard<std::mutex> clk(compress_mutex_);
        compress_queue_.push_back(std::move(blk));
    } else {
        disk_store_.save_block(blk);
        disk_index_.add(bid, blk.summary);
    }

    ram_map_.erase(bid);
    ram_list_.pop_front();
}

void DiskShardedKVCache::compress_worker() {
    while (compress_running_) {
        std::vector<KVBlock> batch;
        {
            std::lock_guard<std::mutex> lk(compress_mutex_);
            batch.swap(compress_queue_);
        }
        for (auto& blk : batch) {
            try {
                disk_store_.save_block(blk);
                disk_index_.add(blk.block_id, blk.summary);
            } catch (const std::exception& e) {
                std::cerr << "[KVCache] compress_worker error: " << e.what() << "\n";
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void DiskShardedKVCache::flush_to_ram() {
    evict_active_to_ram();
}

void DiskShardedKVCache::flush_to_disk(bool async) {
    std::lock_guard<std::mutex> rlk(ram_mutex_);
    while (!ram_list_.empty()) {
        evict_ram_to_disk();
    }
}

int64_t DiskShardedKVCache::active_token_count() const {
    std::lock_guard<std::mutex> lk(active_mutex_);
    return active_tokens_;
}

int64_t DiskShardedKVCache::total_token_count() const {
    std::lock_guard<std::mutex> lk(active_mutex_);
    std::lock_guard<std::mutex> rlk(ram_mutex_);
    return active_tokens_ +
           static_cast<int64_t>(ram_list_.size()) * cfg_.block_tokens +
           disk_index_.size() * cfg_.block_tokens;
}

void DiskShardedKVCache::reset(const std::string& new_session_id) {
    std::lock_guard<std::mutex> lk(active_mutex_);
    for (auto& t : active_k_) t = torch::Tensor{};
    for (auto& t : active_v_) t = torch::Tensor{};
    active_tokens_ = 0;
    {
        std::lock_guard<std::mutex> rlk(ram_mutex_);
        ram_list_.clear(); ram_map_.clear();
    }
    disk_index_.clear();
    next_block_id_ = 0;
    std::cout << "[KVCache] Cache reset.\n";
}

DiskShardedKVCache::Stats DiskShardedKVCache::statistics() const {
    std::lock_guard<std::mutex> lk(active_mutex_);
    std::lock_guard<std::mutex> rlk(ram_mutex_);
    return {
        active_tokens_,
        static_cast<int64_t>(ram_list_.size()),
        static_cast<int64_t>(disk_index_.size()),
        disk_store_.total_size_mb(),
        retrieval_hits_.load(),
        retrieval_misses_.load()
    };
}

void DiskShardedKVCache::save_state() const {
    const_cast<DiskShardedKVCache*>(this)->flush_to_disk(false); // sync flush before saving
    std::cout << "[KVCache] State saved to disk.\n";
}

void DiskShardedKVCache::load_state() {
    // Re-scan disk for existing blocks and rebuild index
    auto dir = cfg_.cache_dir / cfg_.session_id;
    if (!std::filesystem::exists(dir)) return;
    int64_t reloaded = 0;
    for (auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".bin") continue;
        // Extract block_id from filename: kv_XXXXXXXX.bin
        std::string stem = entry.path().stem().string();
        if (stem.size() > 3) {
            try {
                int64_t bid = std::stoll(stem.substr(3));
                auto blk = disk_store_.load_block(bid);
                disk_index_.add(bid, blk.summary);
                next_block_id_ = std::max(next_block_id_.load(), bid + 1);
                ++reloaded;
            } catch (...) {}
        }
    }
    std::cout << "[KVCache] Reloaded " << reloaded << " blocks from disk.\n";
}

} // namespace xorzen
