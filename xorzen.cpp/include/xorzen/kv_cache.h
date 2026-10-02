#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/kv_cache.h
//  Disk-Sharded Infinite Context KV Cache
//  4-tier memory: Active(CPU) → RAM → SSD → HDD
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
// KV Cache Memory Math (per token):
//   Uncompressed: 2 × layers × kv_heads × head_dim × 2 bytes
//   SVD compressed (rank-R): × (R / head_dim)
//   INT8 quantized: × 0.25
//   → ~12 KB/token with 24L/8H/128D model, INT8+SVD-64
//   → 1TB SSD ≈ 83M tokens ≈ 60M words of context
// ============================================================

#include <torch/torch.h>
#include <filesystem>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <functional>
#include <optional>
#include <atomic>

namespace xorzen {

// ============================================================
//  KVBlock — one compressed KV block for N tokens
// ============================================================
struct KVBlock {
    int64_t     block_id  = -1;
    int64_t     num_tokens = 0;
    int64_t     num_layers = 0;

    // Summary vector for retrieval scoring: mean-pooled keys [L, H]
    torch::Tensor summary;     // [num_layers, head_dim]

    // Compressed K and V: [num_layers, num_tokens, svd_rank]
    torch::Tensor k_compressed;
    torch::Tensor v_compressed;

    // SVD basis matrices (U): [num_layers, head_dim, svd_rank]
    // Stored per-block so we can reconstruct with (K_comp @ U^T)
    torch::Tensor k_basis;
    torch::Tensor v_basis;

    bool is_quantized = false;  // If true, k_compressed/v_compressed are INT8

    int64_t size_bytes() const;
};

// ============================================================
//  KVCacheConfig
// ============================================================
struct KVCacheConfig {
    // Architecture
    int64_t num_layers   = 24;
    int64_t num_kv_heads = 8;
    int64_t head_dim     = 128;

    // Tier 0 (active window, always in RAM)
    int64_t active_tokens = 512;

    // Tier 1 (RAM KV pool)
    int64_t ram_blocks    = 16;    // Number of full KV blocks in RAM
    int64_t block_tokens  = 256;   // Tokens per block

    // Tier 2 (SSD, compressed)
    int64_t svd_rank      = 64;    // SVD compression rank
    bool    use_int8      = true;  // Quantise compressed KV to INT8
    int64_t max_ssd_blocks = 4096; // Max blocks on SSD (≈1B tokens for 256-token blocks)

    // Retrieval
    int64_t retrieval_top_k = 64;  // Top-K blocks to retrieve from SSD per query

    // Async compression
    bool async_compress = true;   // Compress to disk on background thread

    // Storage
    std::filesystem::path cache_dir = "memory/kv_cache";
    std::string session_id          = "default";

    // Computed helpers
    int64_t total_kv_dim() const {
        return num_layers * num_kv_heads * head_dim;
    }
    double bytes_per_token_uncompressed() const {
        return 2.0 * num_layers * num_kv_heads * head_dim * 2.0; // FP16
    }
    double bytes_per_token_compressed() const {
        double base = 2.0 * num_layers * svd_rank * (use_int8 ? 1.0 : 2.0);
        return base;
    }
};

// ============================================================
//  DiskKVIndex — fast block retrieval index (pure C++, no FAISS)
//  Uses mean-pooled key similarity for top-K block selection
// ============================================================
class DiskKVIndex {
public:
    explicit DiskKVIndex(int64_t num_layers, int64_t head_dim);

    // Register a block's summary (mean-pooled K) in the index
    void add(int64_t block_id, const torch::Tensor& summary);  // [L, H]

    // Remove a block from the index
    void remove(int64_t block_id);

    // Query top-K blocks by cosine similarity to query key
    // query: [L, H]  →  returns block_ids sorted by score descending
    std::vector<int64_t> query(const torch::Tensor& query_key,
                               int64_t top_k = 64) const;

    size_t size() const;
    void clear();

private:
    int64_t num_layers_;
    int64_t head_dim_;
    mutable std::mutex mutex_;

    // Flat storage: block_ids and their [L*H] flattened summary vectors
    std::vector<int64_t>   block_ids_;
    torch::Tensor          summaries_;  // [N_indexed, L*H]
    mutable bool           dirty_ = true; // Needs rebuild before query
    void rebuild_if_needed() const;

    // Working copy for rebuild
    mutable torch::Tensor summaries_norm_;
};

// ============================================================
//  DiskKVStore — manages KVBlocks on disk
// ============================================================
class DiskKVStore {
public:
    DiskKVStore(const std::filesystem::path& store_dir,
                const KVCacheConfig& cfg);

    // Save a block to disk (returns bytes written)
    int64_t save_block(const KVBlock& block);

    // Load a block from disk
    KVBlock load_block(int64_t block_id) const;

    // Delete a block from disk
    void delete_block(int64_t block_id);

    bool block_exists(int64_t block_id) const;
    int64_t num_blocks() const;
    double total_size_mb() const;

    std::filesystem::path block_path(int64_t block_id) const;

private:
    std::filesystem::path dir_;
    KVCacheConfig cfg_;
};

// ============================================================
//  DiskShardedKVCache — the main 4-tier cache manager
// ============================================================
class DiskShardedKVCache {
public:
    // KV tensor format used externally: [B, L, H_kv, D]  (B=batch, L=seq, H_kv=heads, D=head_dim)
    // The cache internally works per-layer as [T, H_kv, D]

    explicit DiskShardedKVCache(const KVCacheConfig& cfg);
    ~DiskShardedKVCache();

    // Append new KV pairs for the current sequence
    // k_new, v_new: [B, L, num_kv_heads, head_dim]
    void append(const torch::Tensor& k_new, const torch::Tensor& v_new);

    // Retrieve relevant KV context for a query
    // query_key: [B, L_q, num_kv_heads, head_dim]
    // Returns: {k_context, v_context} each [B, L_ctx, num_kv_heads, head_dim]
    // L_ctx = active window + retrieved disk tokens
    std::pair<torch::Tensor, torch::Tensor>
    retrieve(const torch::Tensor& query_key);

    // Save current active window to a RAM block (call periodically)
    void flush_to_ram();

    // Save RAM blocks that exceed capacity to disk (async optional)
    void flush_to_disk(bool async = true);

    // Get active window size in tokens
    int64_t active_token_count() const;
    int64_t total_token_count() const;

    // Reset cache (new session)
    void reset(const std::string& new_session_id = "");

    // Statistics
    struct Stats {
        int64_t active_tokens;
        int64_t ram_blocks;
        int64_t disk_blocks;
        double  total_size_mb;
        int64_t retrieval_hits;
        int64_t retrieval_misses;
    };
    Stats statistics() const;

    // Save/load cache state to disk (for cross-session persistence)
    void save_state() const;
    void load_state();

private:
    KVCacheConfig cfg_;

    // Tier 0: active window (full precision, in RAM)
    std::vector<torch::Tensor> active_k_; // per layer [T_active, H, D]
    std::vector<torch::Tensor> active_v_;
    int64_t active_tokens_ = 0;
    mutable std::mutex active_mutex_;

    // Tier 1: RAM block pool (LRU, compressed)
    struct RAMBlock {
        int64_t block_id;
        torch::Tensor k;  // [L, T_block, H, D]
        torch::Tensor v;
    };
    std::list<RAMBlock>          ram_list_;  // front=LRU, back=MRU
    std::unordered_map<int64_t, std::list<RAMBlock>::iterator> ram_map_;
    mutable std::mutex ram_mutex_;

    // Tier 2: disk store + retrieval index
    DiskKVStore  disk_store_;
    DiskKVIndex  disk_index_;
    mutable std::mutex disk_mutex_;

    // Block counter
    std::atomic<int64_t> next_block_id_{0};

    // Async compression thread
    std::thread  compress_thread_;
    bool         compress_running_ = false;
    std::vector<KVBlock> compress_queue_;
    std::mutex   compress_mutex_;
    void compress_worker();

    // Statistics
    mutable std::atomic<int64_t> retrieval_hits_{0};
    mutable std::atomic<int64_t> retrieval_misses_{0};

    // Helpers
    KVBlock compress_block(int64_t block_id,
                           const torch::Tensor& k,  // [L, T, H, D]
                           const torch::Tensor& v) const;

    // Reconstruct full KV from compressed block: [L, T, H, D]
    std::pair<torch::Tensor, torch::Tensor>
    decompress_block(const KVBlock& blk) const;

    void evict_ram_to_disk();
    void evict_active_to_ram();
};

} // namespace xorzen
