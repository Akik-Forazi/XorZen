#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <list>
#include <mutex>
#include <unordered_map>
#include "xorzen/types.h"

namespace xorzen {
namespace anime {

struct ChunkMemoryEntry {
    torch::Tensor key;   // [key_dim]
    torch::Tensor value; // [val_dim]
};

class LRUChunkCache {
public:
    explicit LRUChunkCache(size_t capacity = 256);
    std::shared_ptr<ChunkMemoryEntry> get(int64_t chunk_id);
    void put(int64_t chunk_id, std::shared_ptr<ChunkMemoryEntry> entry);
    void clear();

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::list<int64_t> order_;
    struct Entry {
        std::shared_ptr<ChunkMemoryEntry> data;
        std::list<int64_t>::iterator it;
    };
    std::unordered_map<int64_t, Entry> cache_;
};

class ChunkDiskManager {
public:
    ChunkDiskManager(std::filesystem::path shard_dir, int64_t key_dim, int64_t val_dim);
    std::filesystem::path chunk_path(int64_t chunk_id) const;
    void save_chunk(int64_t chunk_id, const torch::Tensor& key, const torch::Tensor& value);
    std::shared_ptr<ChunkMemoryEntry> load_chunk(int64_t chunk_id, torch::Device device);

private:
    std::filesystem::path shard_dir_;
    int64_t key_dim_;
    int64_t val_dim_;
};

struct ChunkMemoryBankImpl : torch::nn::Module {
    ModelConfig config;
    int64_t num_slots;
    int64_t key_dim;
    int64_t val_dim;
    int64_t top_k;

    torch::nn::Sequential chunk_encoder{nullptr};
    torch::nn::Linear to_key{nullptr};
    torch::nn::Linear to_value{nullptr};
    torch::nn::MultiheadAttention memory_attn{nullptr};

    std::unique_ptr<ChunkDiskManager> disk_manager;
    std::unique_ptr<LRUChunkCache> lru_cache;

    explicit ChunkMemoryBankImpl(ModelConfig config);

    // Compress chunk and persist
    void write_chunk(const torch::Tensor& chunk_tokens, int64_t chunk_id);

    // Retrieve relevant memories for current context
    torch::Tensor read_relevant(const torch::Tensor& query, int64_t top_k_override = -1);

private:
    void build_encoder();
};
TORCH_MODULE(ChunkMemoryBank);

} // namespace anime
} // namespace xorzen
