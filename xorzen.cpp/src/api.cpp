#include "xorzen/api.h"
#include "xorzen/model.h"
#include "xorzen/tokenizer.h"
#include "xorzen/optimized/simd_ops.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace {

struct XorzenHandle {
    std::mutex mutex;
    xorzen::XorzenModel model;
    explicit XorzenHandle(xorzen::XorzenModel m) : model(std::move(m)) {}
};

struct TokenizerHandle {
    std::mutex mutex;
    xorzen::BEBPETokenizer tokenizer;
    explicit TokenizerHandle(xorzen::BEBPETokenizer t) : tokenizer(std::move(t)) {}
};

xorzen::ModelConfig default_api_config() {
    xorzen::ModelConfig cfg;
    cfg.normalize();
    cfg.use_disk_cache = false;
    cfg.expert_count = cfg.num_experts = 16;
    cfg.max_expert_cache = 8;
    return cfg;
}

}

extern "C" {

void* xorzen_load_model(const char* checkpoint_path) {
    try {
        auto cfg = default_api_config();
        auto model = xorzen::XorzenModel(cfg, true);
        if (checkpoint_path && std::strlen(checkpoint_path) > 0) {
            model->load_checkpoint(checkpoint_path);
        }
        return new XorzenHandle(model);
    } catch (...) {
        return nullptr;
    }
}

int xorzen_forward(void* raw, const int* input_ids, int batch_size, int seq_len, float* logits) {
    if (!raw || !input_ids || !logits || batch_size <= 0 || seq_len <= 0) return -1;
    try {
        auto* handle = static_cast<XorzenHandle*>(raw);
        std::lock_guard<std::mutex> guard(handle->mutex);
        auto input = torch::from_blob(const_cast<int*>(input_ids), {batch_size, seq_len}, torch::kInt32).to(torch::kLong).clone();
        torch::NoGradGuard no_grad;
        auto out = handle->model->forward(input);
        auto cpu_logits = out.logits.to(torch::kCPU).contiguous();
        std::memcpy(logits, cpu_logits.data_ptr<float>(), static_cast<size_t>(cpu_logits.numel()) * sizeof(float));
        return 0;
    } catch (...) {
        return -2;
    }
}

int xorzen_train_step(void* raw, const int* input_ids, const int* labels,
                      int batch_size, int seq_len, float* loss_out) {
    if (!raw || !input_ids || !labels || !loss_out) return -1;
    try {
        auto* handle = static_cast<XorzenHandle*>(raw);
        std::lock_guard<std::mutex> guard(handle->mutex);
        auto input = torch::from_blob(const_cast<int*>(input_ids), {batch_size, seq_len}, torch::kInt32).to(torch::kLong).clone();
        auto y = torch::from_blob(const_cast<int*>(labels), {batch_size, seq_len}, torch::kInt32).to(torch::kLong).clone();
        handle->model->train();
        auto out = handle->model->forward(input, {}, {}, y);
        *loss_out = out.loss.item<float>();
        return 0;
    } catch (...) {
        return -2;
    }
}

int xorzen_generate(void* raw, const int* prompt, int prompt_len,
                    int max_new_tokens, int* output, int* output_len) {
    if (!raw || !prompt || !output || !output_len || prompt_len <= 0) return -1;
    try {
        auto* handle = static_cast<XorzenHandle*>(raw);
        std::lock_guard<std::mutex> guard(handle->mutex);
        xorzen::GenerationConfig cfg;
        cfg.max_length = max_new_tokens;
        auto p = torch::from_blob(const_cast<int*>(prompt), {prompt_len}, torch::kInt32).to(torch::kLong).clone();
        auto generated = handle->model->generate(p, cfg).squeeze(0).to(torch::kCPU).contiguous();
        *output_len = static_cast<int>(generated.numel());
        auto acc = generated.accessor<int64_t, 1>();
        for (int i = 0; i < *output_len; ++i) output[i] = static_cast<int>(acc[i]);
        return 0;
    } catch (...) {
        return -2;
    }
}

int xorzen_save_checkpoint(void* raw, const char* path) {
    if (!raw || !path) return -1;
    try {
        auto* handle = static_cast<XorzenHandle*>(raw);
        std::lock_guard<std::mutex> guard(handle->mutex);
        handle->model->save_checkpoint(path);
        return 0;
    } catch (...) {
        return -2;
    }
}

int xorzen_get_metrics(void* raw, float* metrics, int max_metrics) {
    if (!raw || !metrics || max_metrics <= 0) return -1;
    auto* handle = static_cast<XorzenHandle*>(raw);
    std::lock_guard<std::mutex> guard(handle->mutex);
    if (max_metrics > 0) metrics[0] = static_cast<float>(handle->model->step_count);
    if (max_metrics > 1) metrics[1] = static_cast<float>(handle->model->total_tokens_processed);
    if (max_metrics > 2) metrics[2] = static_cast<float>(handle->model->count_parameters());
    return std::min(max_metrics, 3);
}

void xorzen_free_model(void* raw) {
    delete static_cast<XorzenHandle*>(raw);
}

void* xorzen_load_tokenizer(const char* tokenizer_json_path) {
    if (!tokenizer_json_path) return nullptr;
    try {
        auto tokenizer = xorzen::BEBPETokenizer::from_tokenizer_json(tokenizer_json_path);
        return new TokenizerHandle(std::move(tokenizer));
    } catch (...) {
        return nullptr;
    }
}

int xorzen_tokenizer_encode(void* raw, const char* text,
                            int* output_ids, int max_tokens,
                            int add_special_tokens) {
    if (!raw || !text || !output_ids || max_tokens <= 0) return -1;
    try {
        auto* handle = static_cast<TokenizerHandle*>(raw);
        std::lock_guard<std::mutex> guard(handle->mutex);
        auto enc = handle->tokenizer.encode(text, add_special_tokens != 0,
                                            max_tokens, false, true);
        int count = std::min<int>(max_tokens, static_cast<int>(enc.input_ids.size()));
        for (int i = 0; i < count; ++i) output_ids[i] = static_cast<int>(enc.input_ids[static_cast<size_t>(i)]);
        return count;
    } catch (...) {
        return -2;
    }
}

int xorzen_tokenizer_decode(void* raw, const int* token_ids,
                            int token_count, char* output_text,
                            int max_chars, int skip_special_tokens) {
    if (!raw || !token_ids || !output_text || token_count < 0 || max_chars <= 0) return -1;
    try {
        auto* handle = static_cast<TokenizerHandle*>(raw);
        std::lock_guard<std::mutex> guard(handle->mutex);
        std::vector<int64_t> ids(static_cast<size_t>(token_count));
        for (int i = 0; i < token_count; ++i) ids[static_cast<size_t>(i)] = token_ids[i];
        auto text = handle->tokenizer.decode(ids, skip_special_tokens != 0);
        int n = std::min<int>(max_chars - 1, static_cast<int>(text.size()));
        std::memcpy(output_text, text.data(), static_cast<size_t>(n));
        output_text[n] = '\0';
        return n;
    } catch (...) {
        return -2;
    }
}

void xorzen_free_tokenizer(void* raw) {
    delete static_cast<TokenizerHandle*>(raw);
}

int xorzen_rmsnorm(const float* x, const float* weight, int batch, int seq, int dim, float* output) {
    if (!x || !weight || !output) return -1;
    try {
        auto x_tensor = torch::from_blob(const_cast<float*>(x), {batch, seq, dim}, torch::kFloat32);
        auto w_tensor = torch::from_blob(const_cast<float*>(weight), {dim}, torch::kFloat32);
        
        auto out_tensor = xorzen::optimized::rmsnorm_simd(x_tensor, w_tensor);
        
        std::memcpy(output, out_tensor.data_ptr<float>(), out_tensor.nbytes());
        return 0;
    } catch (...) {
        return -2;
    }
}

}
