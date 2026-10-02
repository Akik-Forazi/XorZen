// ============================================================
//  xorzen.cpp — src/training/trainer.cpp
//  Full production training loop
//  Ported from xorzen/training/trainer.py v2.0
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/trainer.h"
#include "xorzen/ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace xorzen {

// ============================================================
//  Utility: wall-clock in seconds
// ============================================================
static double wall_now() {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0;
}

// ============================================================
//  Constructor
// ============================================================
XorzenTrainer::XorzenTrainer(XorzenModel           model,
                             TrainerConfig         config,
                             BEBPETokenizer*       tokenizer,
                             const std::string&    train_path,
                             const std::string&    eval_path)
    : model_(std::move(model)),
      config_(std::move(config)),
      tokenizer_(tokenizer),
      device_(torch::kCPU) {
    setup_device();
    model_->to(device_);
    if (config_.enable_cot) model_->enable_cot(true);
    setup_optimizer();
    setup_checkpoint_manager();
    if (!train_path.empty() || !eval_path.empty()) {
        setup_datasets(train_path, eval_path);
    }
    print_setup_info();
}

// ============================================================
//  Device setup
// ============================================================
void XorzenTrainer::setup_device() {
    if (config_.device == "cuda" && torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        std::cout << "[Trainer] Using CUDA\n";
    } else {
        device_ = torch::kCPU;
        std::cout << "[Trainer] Using CPU\n";
        // Use all available logical threads
        torch::set_num_threads(static_cast<int>(
            std::max(1u, std::thread::hardware_concurrency())));
    }
}

// ============================================================
//  Optimizer — AdamW with decoupled weight decay
// ============================================================
void XorzenTrainer::setup_optimizer() {
    torch::optim::AdamWOptions opts(config_.learning_rate);
    opts.betas({config_.adam_beta1, config_.adam_beta2});
    opts.eps(config_.adam_eps);
    opts.weight_decay(config_.weight_decay);
    optimizer_ = std::make_unique<torch::optim::AdamW>(
        model_->parameters(), opts);
}

// ============================================================
//  Datasets (optional — only when paths are provided)
// ============================================================
void XorzenTrainer::setup_datasets(const std::string& train_path,
                                   const std::string& eval_path) {
    if (!tokenizer_) {
        std::cerr << "[Trainer] WARNING: no tokenizer provided, skipping dataset setup\n";
        return;
    }
    TextDataPipelineConfig dp_cfg;
    dp_cfg.sequence_length = model_->config.context_length;

    if (!train_path.empty() && std::filesystem::exists(train_path)) {
        auto files = discover_text_files(train_path, true);
        if (!files.empty()) {
            train_dataset_ = std::make_unique<BEBPETextDataset>(
                files, *tokenizer_, dp_cfg);
            std::cout << "[Trainer] Train tokens: "
                      << train_dataset_->token_count() << "\n";
        }
    }
    if (!eval_path.empty() && std::filesystem::exists(eval_path)) {
        auto files = discover_text_files(eval_path, true);
        if (!files.empty()) {
            eval_dataset_ = std::make_unique<BEBPETextDataset>(
                files, *tokenizer_, dp_cfg);
        }
    }
}

// ============================================================
//  Checkpoint manager
// ============================================================
void XorzenTrainer::setup_checkpoint_manager() {
    CheckpointManager::Options opts;
    opts.max_checkpoints = config_.max_checkpoints;
    opts.save_best       = config_.save_best;
    opts.best_metric     = "eval_loss";
    opts.best_mode_min   = true;
    opts.save_optimizer  = true;
    ckpt_mgr_ = std::make_unique<CheckpointManager>(config_.checkpoint_dir, opts);
}

// ============================================================
//  LR schedule helpers
// ============================================================
double XorzenTrainer::compute_lr(int64_t s) const {
    const double base_lr  = config_.learning_rate;
    const double min_lr   = base_lr * config_.min_lr_ratio;
    const int64_t warmup  = config_.warmup_steps;
    // Effective total steps
    int64_t total = config_.max_steps > 0
        ? config_.max_steps
        : config_.max_epochs * 10000; // fallback when no dataset length known

    if (s < warmup) {
        // Linear warm-up
        return base_lr * (static_cast<double>(s + 1) /
                          static_cast<double>(std::max<int64_t>(warmup, 1)));
    }
    double progress = static_cast<double>(s - warmup) /
                      static_cast<double>(std::max<int64_t>(total - warmup, 1));
    progress = std::min(1.0, progress);

    if (config_.lr_schedule == "cosine") {
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + std::cos(M_PI * progress));
    } else if (config_.lr_schedule == "linear") {
        return base_lr + progress * (min_lr - base_lr);
    }
    return base_lr; // constant
}

void XorzenTrainer::apply_lr(double lr) {
    for (auto& pg : optimizer_->param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr);
    }
}

double XorzenTrainer::current_lr() const {
    return static_cast<const torch::optim::AdamWOptions&>(
        optimizer_->param_groups()[0].options()).lr();
}

// ============================================================
//  Single training step
// ============================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
XorzenTrainer::training_step(const torch::Tensor& input_ids,
                             const torch::Tensor& labels) {
    auto out = model_->forward(input_ids, {}, {}, labels);
    auto loss = out.loss;
    return {loss, out.lm_loss.defined() ? out.lm_loss : loss,
                  out.routing_loss.defined() ? out.routing_loss
                                             : torch::zeros({}, device_)};
}

// ============================================================
//  evaluate()
// ============================================================
double XorzenTrainer::evaluate() {
    if (!eval_dataset_) return state_.best_eval_loss;
    model_->eval();
    torch::NoGradGuard no_grad;

    auto loader = torch::data::make_data_loader(
        *eval_dataset_,
        torch::data::DataLoaderOptions()
            .batch_size(config_.batch_size)
            .workers(static_cast<int>(config_.num_workers)));

    double total_loss = 0.0;
    int64_t n_batches = 0;
    for (auto& batch : *loader) {
        if (batch.empty()) continue;
        auto input_ids = batch[0].data.to(torch::kLong).to(device_);
        auto labels    = batch[0].target.to(torch::kLong).to(device_);
        auto out = model_->forward(input_ids, {}, {}, labels);
        if (out.lm_loss.defined()) total_loss += out.lm_loss.item<double>();
        ++n_batches;
    }
    model_->train();
    return n_batches > 0 ? total_loss / static_cast<double>(n_batches) : 1e30;
}

// ============================================================
//  add_callback
// ============================================================
void XorzenTrainer::add_callback(CallbackFn fn) {
    callbacks_.push_back(std::move(fn));
}

// ============================================================
//  resume_from
// ============================================================
void XorzenTrainer::resume_from(const std::filesystem::path& path) {
    auto [epoch, step] = ckpt_mgr_->load(path, model_.get(), optimizer_.get());
    state_.epoch       = epoch;
    state_.global_step = step;
    std::cout << "[Trainer] Resumed: epoch=" << epoch << " step=" << step << "\n";
}

// ============================================================
//  train()  — main loop
// ============================================================
std::vector<TrainerMetrics> XorzenTrainer::train() {
    model_->train();
    state_.start_time = wall_now();
    std::cout << "\n[Trainer] Starting training for "
              << config_.max_epochs << " epochs\n";

    // If no dataset, run a quick smoke check and return
    if (!train_dataset_) {
        std::cerr << "[Trainer] No training dataset provided — nothing to train.\n";
        return metric_history_;
    }

    for (int64_t epoch = state_.epoch; epoch < config_.max_epochs; ++epoch) {
        state_.epoch = epoch;
        state_.reset_epoch();
        double epoch_loss = 0.0;
        int64_t epoch_steps = 0;
        const int64_t grad_accum = std::max<int64_t>(1, config_.gradient_accumulation);

        auto loader = torch::data::make_data_loader(
            *train_dataset_,
            torch::data::DataLoaderOptions()
                .batch_size(config_.batch_size)
                .workers(static_cast<int>(config_.num_workers))
                .enforce_ordering(false));

        bool stop = false;
        int64_t pending_accum = 0;
        auto optimizer_step = [&]() {
            double grad_norm = 0.0;
            if (config_.gradient_clipping) {
                grad_norm = torch::nn::utils::clip_grad_norm_(
                    model_->parameters(), config_.max_grad_norm);
            }
            optimizer_->step();
            optimizer_->zero_grad(true);
            state_.grad_norm = grad_norm;
        };
        for (auto& batch : *loader) {
            if (batch.empty()) continue;

            auto input_ids = batch[0].data.to(torch::kLong).to(device_);
            auto labels    = batch[0].target.to(torch::kLong).to(device_);

            // Apply LR schedule
            double lr = compute_lr(state_.global_step);
            apply_lr(lr);

            // Forward + backward
            model_->train();
            auto [loss, lm_loss, routing_loss] = training_step(input_ids, labels);

            // Scale for gradient accumulation
            auto scaled_loss = loss / static_cast<double>(grad_accum);
            scaled_loss.backward();
            ++pending_accum;

            const bool should_stop_after_batch =
                config_.max_steps > 0 && state_.global_step + 1 >= config_.max_steps;
            bool do_step = pending_accum >= grad_accum || should_stop_after_batch;
            if (do_step) {
                optimizer_step();
                pending_accum = 0;
            }

            double loss_val = loss.item<double>();
            epoch_loss += loss_val;
            ++epoch_steps;
            ++state_.steps_in_epoch;
            ++state_.global_step;
            ++model_->step_count;
            model_->total_tokens_processed +=
                input_ids.size(0) * input_ids.size(1);
            state_.train_loss = epoch_loss / static_cast<double>(epoch_steps);

            // Logging
            if (state_.global_step % config_.log_every_n_steps == 0) {
                TrainerMetrics m;
                m.step         = state_.global_step;
                m.epoch        = epoch;
                m.train_loss   = loss_val;
                m.lm_loss      = lm_loss.defined() ? lm_loss.item<double>() : loss_val;
                m.routing_loss = routing_loss.defined() ? routing_loss.item<double>() : 0.0;
                m.grad_norm    = state_.grad_norm;
                m.lr           = lr;
                m.tokens_per_sec = static_cast<double>(
                    model_->total_tokens_processed) / std::max(1.0, wall_now() - state_.start_time);
                log_metrics(m);
                metric_history_.push_back(m);
                for (auto& cb : callbacks_) cb(m);
            }

            // Periodic evaluation
            if (config_.eval_every_n_steps > 0 &&
                state_.global_step % config_.eval_every_n_steps == 0) {
                double eval_loss = evaluate();
                state_.eval_loss = eval_loss;
                if (eval_loss < state_.best_eval_loss) {
                    state_.best_eval_loss = eval_loss;
                    ckpt_mgr_->save(*model_, state_,
                        {{"eval_loss", eval_loss}, {"train_loss", state_.train_loss}},
                        optimizer_.get(), /*is_best=*/true);
                }
            }

            // Periodic save
            if (config_.save_every_n_steps > 0 &&
                state_.global_step % config_.save_every_n_steps == 0) {
                ckpt_mgr_->save(*model_, state_,
                    {{"train_loss", state_.train_loss}},
                    optimizer_.get());
            }

            // Max steps check
            if (config_.max_steps > 0 && state_.global_step >= config_.max_steps) {
                stop = true;
                break;
            }
        } // batch loop

        if (pending_accum > 0) {
            optimizer_step();
            pending_accum = 0;
        }

        // End-of-epoch evaluation
        if (eval_dataset_) {
            double eval_loss = evaluate();
            state_.eval_loss = eval_loss;
            bool is_best = eval_loss < state_.best_eval_loss;
            if (is_best) state_.best_eval_loss = eval_loss;
            ckpt_mgr_->save(*model_, state_,
                {{"eval_loss", eval_loss}, {"train_loss", state_.train_loss}},
                optimizer_.get(), is_best);
        } else {
            // Save checkpoint at epoch boundary
            ckpt_mgr_->save(*model_, state_,
                {{"train_loss", state_.train_loss}},
                optimizer_.get());
        }

        print_epoch_summary(epoch, epoch_steps > 0
            ? epoch_loss / static_cast<double>(epoch_steps) : 0.0);

        if (stop || state_.should_stop) {
            std::cout << "[Trainer] Stopping at step " << state_.global_step << "\n";
            break;
        }
    } // epoch loop

    double total_time = wall_now() - state_.start_time;
    std::cout << "\n[Trainer] Training complete."
              << " Total time: " << std::fixed << std::setprecision(2)
              << total_time / 60.0 << " min"
              << " | Best eval loss: "
              << (state_.best_eval_loss < 1e29 ? state_.best_eval_loss : -1.0)
              << "\n";
    return metric_history_;
}

// ============================================================
//  Logging / printing
// ============================================================
void XorzenTrainer::log_metrics(const TrainerMetrics& m) {
    std::cout << "[Trainer] ep=" << m.epoch
              << " step=" << m.step
              << " loss=" << std::fixed << std::setprecision(4) << m.train_loss
              << " lm=" << m.lm_loss
              << " route=" << m.routing_loss
              << " gnorm=" << std::setprecision(3) << m.grad_norm
              << " lr=" << std::scientific << std::setprecision(2) << m.lr
              << " tok/s=" << std::fixed << std::setprecision(0) << m.tokens_per_sec
              << "\n";
}

void XorzenTrainer::print_setup_info() const {
    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  XORZEN.CPP Trainer — FRAZIYM TECH & AI\n"
              << std::string(72, '=') << "\n";
    int64_t total = 0, trainable = 0;
    for (const auto& p : model_->parameters()) {
        total += p.numel();
        if (p.requires_grad()) trainable += p.numel();
    }
    std::cout << "  Parameters : " << total
              << " (trainable=" << trainable << ")\n"
              << "  LR         : " << config_.learning_rate << "\n"
              << "  Schedule   : " << config_.lr_schedule << "\n"
              << "  Warmup     : " << config_.warmup_steps << " steps\n"
              << "  Max epochs : " << config_.max_epochs << "\n"
              << "  Batch size : " << config_.batch_size << "\n"
              << "  Grad accum : " << config_.gradient_accumulation << "\n"
              << "  CoT        : " << (config_.enable_cot ? "ON" : "off") << "\n"
              << "  Device     : " << (device_ == torch::kCUDA ? "CUDA" : "CPU") << "\n"
              << std::string(72, '=') << "\n\n";
}

void XorzenTrainer::print_epoch_summary(int64_t epoch, double avg_loss) const {
    double epoch_time = wall_now() - state_.epoch_start_time;
    std::cout << "[Trainer] --- Epoch " << epoch << " done"
              << " | avg_loss=" << std::fixed << std::setprecision(4) << avg_loss
              << " | eval_loss=" << state_.eval_loss
              << " | time=" << std::setprecision(1) << epoch_time << "s"
              << " | steps/s=" << std::setprecision(2) << state_.steps_per_second()
              << " ---\n";
}

} // namespace xorzen
