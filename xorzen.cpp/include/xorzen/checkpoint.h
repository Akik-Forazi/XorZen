#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/checkpoint.h  (stdlib-only JSON)
//  Checkpoint management: save/load/versioning/best tracking
//  Ported from xorzen/training/checkpoint.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================

#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <fstream>
#include <sstream>
#include <ctime>
#include <functional>

namespace xorzen {

// ============================================================
//  TrainingState — live metrics carried through the loop
// ============================================================
struct TrainingState {
    int64_t epoch          = 0;
    int64_t global_step    = 0;
    int64_t steps_in_epoch = 0;

    double train_loss     = 0.0;
    double eval_loss      = 1e30;
    double best_eval_loss = 1e30;
    double grad_norm      = 0.0;

    double start_time       = 0.0;
    double epoch_start_time = 0.0;
    bool   should_stop      = false;

    double elapsed_seconds() const;
    void reset_epoch();
    double steps_per_second() const;
};

// ============================================================
//  CheckpointManager
// ============================================================
class CheckpointManager {
public:
    static constexpr const char* CHECKPOINT_VERSION = "1.0";
    static constexpr const char* XORZEN_VERSION     = "0.2.5";

    struct Options {
        int64_t     max_checkpoints = 5;
        bool        save_best       = true;
        std::string best_metric     = "eval_loss";
        bool        best_mode_min   = true;   // true=min, false=max
        bool        save_optimizer  = true;
    };

    explicit CheckpointManager(const std::filesystem::path& dir,
                               Options opts = Options{});

    // Save model (+ optional optimizer) checkpoint.
    // Returns the path of the saved .pt file.
    std::filesystem::path save(
        torch::nn::Module&   model,
        const TrainingState& state,
        const std::unordered_map<std::string, double>& metrics = {},
        torch::optim::Optimizer* optimizer = nullptr,
        bool is_best       = false,
        const std::string& notes = "");

    // Load from an explicit path. Returns epoch and step metadata.
    // Optionally restores model + optimizer state.
    std::pair<int64_t, int64_t> load(
        const std::filesystem::path& path,
        torch::nn::Module*       model = nullptr,
        torch::optim::Optimizer* opt   = nullptr);

    // Convenience: load the most-recently-modified checkpoint
    std::pair<int64_t, int64_t> load_latest(
        torch::nn::Module*       model = nullptr,
        torch::optim::Optimizer* opt   = nullptr);

    // Convenience: load best_model.pt
    std::pair<int64_t, int64_t> load_best(
        torch::nn::Module*       model = nullptr,
        torch::optim::Optimizer* opt   = nullptr);

    std::vector<std::filesystem::path> list_checkpoints() const;
    void delete_checkpoint(const std::filesystem::path& path);
    void delete_all(bool keep_best = true);

    std::unordered_map<std::string, double> statistics() const;
    bool has_best() const;
    const std::filesystem::path& dir() const { return dir_; }
    double best_value() const { return best_value_; }

private:
    std::filesystem::path dir_;
    Options opts_;
    double best_value_;
    std::optional<std::filesystem::path> best_path_;

    void  cleanup_old();
    void  copy_as_best(const std::filesystem::path& src);
    bool  is_better(double new_val, double old_val) const;
    void  write_meta(const std::filesystem::path& ckpt_path,
                     const TrainingState& state,
                     const std::unordered_map<std::string, double>& metrics,
                     bool is_best,
                     const std::string& notes) const;
    void  save_best_info() const;
    void  load_best_info();
    static std::string now_iso();
};

} // namespace xorzen
