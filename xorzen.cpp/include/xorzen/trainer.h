#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/trainer.h
//  Production-grade C++ training loop
//  Ported from xorzen/training/trainer.py v2.0
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================

#include <torch/torch.h>
#include "xorzen/model.h"
#include "xorzen/checkpoint.h"
#include "xorzen/data_pipeline.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace xorzen {

// ============================================================
//  TrainerConfig — all hyper-parameters for one training run
// ============================================================
struct TrainerConfig {
    // Optimiser
    double learning_rate          = 3e-4;
    double weight_decay           = 0.1;
    double adam_beta1             = 0.9;
    double adam_beta2             = 0.95;
    double adam_eps               = 1e-8;

    // Schedule
    int64_t warmup_steps          = 100;
    std::string lr_schedule       = "cosine";  // "cosine" | "linear" | "constant"
    double min_lr_ratio           = 0.1;       // minimum lr = learning_rate * ratio

    // Training length
    int64_t max_epochs            = 3;
    int64_t max_steps             = -1;        // -1 = unlimited
    int64_t batch_size            = 8;
    int64_t gradient_accumulation = 1;

    // Regularisation
    double  max_grad_norm         = 1.0;
    bool    gradient_clipping     = true;

    // Data
    int64_t num_workers           = 0;

    // Logging
    int64_t log_every_n_steps     = 10;
    int64_t eval_every_n_steps    = 500;
    int64_t save_every_n_steps    = 1000;

    // Checkpointing
    std::filesystem::path checkpoint_dir = "checkpoints";
    int64_t max_checkpoints       = 5;
    bool    save_best             = true;

    // CoT
    bool enable_cot               = false;

    // Device
    std::string device            = "auto";  // "auto" | "cpu" | "cuda"
};

// ============================================================
//  Trainer metrics snapshot
// ============================================================
struct TrainerMetrics {
    int64_t step        = 0;
    int64_t epoch       = 0;
    double  train_loss  = 0.0;
    double  eval_loss   = 1e30;
    double  lm_loss     = 0.0;
    double  routing_loss = 0.0;
    double  cot_loss    = 0.0;
    double  grad_norm   = 0.0;
    double  lr          = 0.0;
    double  tokens_per_sec = 0.0;
};

// ============================================================
//  XorzenTrainer
// ============================================================
class XorzenTrainer {
public:
    using DataLoaderT = torch::data::StatelessDataLoader<
        BEBPETextDataset,
        torch::data::samplers::SequentialSampler>;
    using CallbackFn  = std::function<void(const TrainerMetrics&)>;

    XorzenTrainer(XorzenModel           model,
                  TrainerConfig         config,
                  BEBPETokenizer*       tokenizer  = nullptr,
                  const std::string&    train_path = "",
                  const std::string&    eval_path  = "");

    // Main entry point
    std::vector<TrainerMetrics> train();

    // Evaluate on eval dataset, returns avg loss
    double evaluate();

    // Register a callback invoked after every logged step
    void add_callback(CallbackFn fn);

    // Resume from a checkpoint directory/file
    void resume_from(const std::filesystem::path& path);

    XorzenModel&          model()   { return model_; }
    const TrainingState&  state()   const { return state_; }
    CheckpointManager&    ckpt_mgr() { return *ckpt_mgr_; }

private:
    XorzenModel       model_;
    TrainerConfig     config_;
    BEBPETokenizer*   tokenizer_;
    torch::Device     device_;
    TrainingState     state_;

    std::unique_ptr<torch::optim::AdamW>  optimizer_;
    std::unique_ptr<CheckpointManager>    ckpt_mgr_;

    // Data
    std::unique_ptr<BEBPETextDataset>  train_dataset_;
    std::unique_ptr<BEBPETextDataset>  eval_dataset_;

    std::vector<CallbackFn>    callbacks_;
    std::vector<TrainerMetrics> metric_history_;

    // Build internals
    void setup_device();
    void setup_optimizer();
    void setup_datasets(const std::string& train_path,
                        const std::string& eval_path);
    void setup_checkpoint_manager();

    // Single forward+backward step; returns (loss, lm_loss, routing_loss)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    training_step(const torch::Tensor& input_ids,
                  const torch::Tensor& labels);

    // LR schedule value at global step s
    double compute_lr(int64_t s) const;
    void   apply_lr(double lr);
    double current_lr() const;

    void print_setup_info() const;
    void print_epoch_summary(int64_t epoch, double avg_loss) const;
    void log_metrics(const TrainerMetrics& m);
};

} // namespace xorzen
