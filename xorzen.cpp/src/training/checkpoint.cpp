// ============================================================
//  xorzen.cpp — src/training/checkpoint.cpp
//  CheckpointManager full implementation
//  Ported from xorzen/training/checkpoint.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/checkpoint.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ---- internal helper: JSON string escaping ----
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if      (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else if (c == '\r') { out += "\\r"; }
        else if (c == '\t') { out += "\\t"; }
        else                 { out += c; }
    }
    return out;
}

namespace xorzen {

// ============================================================
//  TrainingState helpers
// ============================================================

double TrainingState::elapsed_seconds() const {
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return static_cast<double>(now) - start_time;
}

void TrainingState::reset_epoch() {
    steps_in_epoch = 0;
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    epoch_start_time = static_cast<double>(now);
}

double TrainingState::steps_per_second() const {
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    double secs = static_cast<double>(now) - epoch_start_time + 1e-8;
    return static_cast<double>(steps_in_epoch) / secs;
}

// ============================================================
//  Utility
// ============================================================

std::string CheckpointManager::now_iso() {
    auto now  = std::chrono::system_clock::now();
    auto tt   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&tt), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

// Minimal JSON writer — only what we need to write meta files
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

// ============================================================
//  CheckpointManager constructor
// ============================================================

CheckpointManager::CheckpointManager(const std::filesystem::path& dir,
                                     Options opts)
    : dir_(dir), opts_(std::move(opts)),
      best_value_(opts_.best_mode_min ? 1e30 : -1e30) {
    std::filesystem::create_directories(dir_);
    load_best_info();
}

// ============================================================
//  save
// ============================================================

std::filesystem::path CheckpointManager::save(
    torch::nn::Module&   model,
    const TrainingState& state,
    const std::unordered_map<std::string, double>& metrics,
    torch::optim::Optimizer* optimizer,
    bool is_best,
    const std::string& notes) {

    // Build timestamped filename
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "checkpoint_epoch_%lld_step_%lld.pt",
        static_cast<long long>(state.epoch),
        static_cast<long long>(state.global_step));
    auto ckpt_path = dir_ / buf;

    // Build the serialisation archive
    torch::serialize::OutputArchive archive;
    model.save(archive);
    // Save metadata tensors
    archive.write("epoch",       torch::tensor(state.epoch));
    archive.write("global_step", torch::tensor(state.global_step));
    archive.write("train_loss",  torch::tensor(state.train_loss));
    archive.write("eval_loss",   torch::tensor(state.eval_loss));
    archive.save_to(ckpt_path.string());

    // Save optimiser state separately if requested
    if (opts_.save_optimizer && optimizer) {
        std::filesystem::path opt_path = ckpt_path;
        opt_path.replace_extension(".opt.pt");
        torch::serialize::OutputArchive opt_archive;
        optimizer->save(opt_archive);
        opt_archive.save_to(opt_path.string());
    }

    // Write human-readable meta JSON
    write_meta(ckpt_path, state, metrics, is_best, notes);

    std::cout << "[CheckpointManager] Saved: " << ckpt_path.filename() << "\n";

    // Best model tracking
    if (opts_.save_best) {
        auto it = metrics.find(opts_.best_metric);
        if (it != metrics.end()) {
            if (is_better(it->second, best_value_)) {
                best_value_ = it->second;
                copy_as_best(ckpt_path);
                save_best_info();
            }
        }
    }

    cleanup_old();
    return ckpt_path;
}

// ============================================================
//  load
// ============================================================

std::pair<int64_t, int64_t> CheckpointManager::load(
    const std::filesystem::path& path,
    torch::nn::Module*       model,
    torch::optim::Optimizer* opt) {

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Checkpoint not found: " + path.string());
    }
    std::cout << "[CheckpointManager] Loading: " << path.filename() << "\n";

    torch::serialize::InputArchive archive;
    archive.load_from(path.string());

    if (model) {
        model->load(archive);
    }

    // Read back scalars
    torch::Tensor epoch_t, step_t;
    archive.read("epoch",       epoch_t);
    archive.read("global_step", step_t);
    int64_t epoch = epoch_t.item<int64_t>();
    int64_t step  = step_t.item<int64_t>();

    // Optionally restore optimiser
    if (opt) {
        auto opt_path = std::filesystem::path(path).replace_extension(".opt.pt");
        if (std::filesystem::exists(opt_path)) {
            torch::serialize::InputArchive opt_archive;
            opt_archive.load_from(opt_path.string());
            opt->load(opt_archive);
        }
    }

    std::cout << "[CheckpointManager] Resumed from epoch=" << epoch
              << " step=" << step << "\n";
    return {epoch, step};
}

std::pair<int64_t, int64_t> CheckpointManager::load_latest(
    torch::nn::Module*       model,
    torch::optim::Optimizer* opt) {
    auto ckpts = list_checkpoints();
    if (ckpts.empty())
        throw std::runtime_error("No checkpoints found in: " + dir_.string());
    // Pick newest by last-write-time
    auto latest = *std::max_element(ckpts.begin(), ckpts.end(),
        [](const auto& a, const auto& b) {
            return std::filesystem::last_write_time(a) <
                   std::filesystem::last_write_time(b);
        });
    return load(latest, model, opt);
}

std::pair<int64_t, int64_t> CheckpointManager::load_best(
    torch::nn::Module*       model,
    torch::optim::Optimizer* opt) {
    auto best = dir_ / "best_model.pt";
    if (!std::filesystem::exists(best))
        throw std::runtime_error("No best checkpoint in: " + dir_.string());
    return load(best, model, opt);
}

// ============================================================
//  List / delete
// ============================================================

std::vector<std::filesystem::path> CheckpointManager::list_checkpoints() const {
    std::vector<std::filesystem::path> result;
    for (auto& entry : std::filesystem::directory_iterator(dir_)) {
        if (!entry.is_regular_file()) continue;
        auto name = entry.path().filename().string();
        if (name.rfind("checkpoint_", 0) == 0 &&
            entry.path().extension() == ".pt") {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

void CheckpointManager::delete_checkpoint(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    // Also remove optional sidecar files
    auto meta = path; meta.replace_extension(".meta.json");
    std::filesystem::remove(meta, ec);
    auto opt  = path; opt.replace_extension(".opt.pt");
    std::filesystem::remove(opt, ec);
}

void CheckpointManager::delete_all(bool keep_best) {
    for (auto& p : list_checkpoints()) {
        if (keep_best && p.filename() == "best_model.pt") continue;
        delete_checkpoint(p);
    }
}

// ============================================================
//  Statistics
// ============================================================

std::unordered_map<std::string, double> CheckpointManager::statistics() const {
    auto ckpts = list_checkpoints();
    double total_size = 0.0;
    for (auto& p : ckpts)
        total_size += static_cast<double>(std::filesystem::file_size(p));
    return {
        {"num_checkpoints",  static_cast<double>(ckpts.size())},
        {"total_size_mb",    total_size / (1024.0 * 1024.0)},
        {"max_checkpoints",  static_cast<double>(opts_.max_checkpoints)},
        {"has_best",         static_cast<double>(has_best() ? 1 : 0)},
        {"best_value",       best_value_},
    };
}

bool CheckpointManager::has_best() const {
    return std::filesystem::exists(dir_ / "best_model.pt");
}

// ============================================================
//  Private helpers
// ============================================================

void CheckpointManager::cleanup_old() {
    if (opts_.max_checkpoints <= 0) return;
    auto ckpts = list_checkpoints();
    // Sort oldest first by write-time
    std::sort(ckpts.begin(), ckpts.end(),
        [](const auto& a, const auto& b) {
            return std::filesystem::last_write_time(a) <
                   std::filesystem::last_write_time(b);
        });
    while (static_cast<int64_t>(ckpts.size()) > opts_.max_checkpoints) {
        auto oldest = ckpts.front();
        ckpts.erase(ckpts.begin());
        // Don't delete the best model if it slipped into the regular list
        if (oldest.filename() == "best_model.pt") continue;
        delete_checkpoint(oldest);
    }
}

void CheckpointManager::copy_as_best(const std::filesystem::path& src) {
    auto dst = dir_ / "best_model.pt";
    std::filesystem::copy_file(src, dst,
        std::filesystem::copy_options::overwrite_existing);
    // Also copy opt file if present
    auto src_opt = std::filesystem::path(src).replace_extension(".opt.pt");
    if (std::filesystem::exists(src_opt)) {
        auto dst_opt = dir_ / "best_model.opt.pt";
        std::filesystem::copy_file(src_opt, dst_opt,
            std::filesystem::copy_options::overwrite_existing);
    }
    best_path_ = dst;
    std::cout << "[CheckpointManager] New best checkpoint (metric=" << opts_.best_metric
              << " val=" << best_value_ << ")\n";
}

bool CheckpointManager::is_better(double new_val, double old_val) const {
    return opts_.best_mode_min ? (new_val < old_val) : (new_val > old_val);
}

void CheckpointManager::write_meta(const std::filesystem::path& ckpt_path,
                                   const TrainingState& state,
                                   const std::unordered_map<std::string, double>& metrics,
                                   bool is_best,
                                   const std::string& notes) const {
    auto meta_path = std::filesystem::path(ckpt_path).replace_extension(".meta.json");
    std::ofstream f(meta_path);
    if (!f.is_open()) return;
    f << "{\n";
    f << "  \"checkpoint_version\": \"" << CHECKPOINT_VERSION << "\",\n";
    f << "  \"xorzen_version\": \""      << XORZEN_VERSION     << "\",\n";
    f << "  \"created_at\": \""          << now_iso()          << "\",\n";
    f << "  \"epoch\": "                 << state.epoch        << ",\n";
    f << "  \"global_step\": "           << state.global_step  << ",\n";
    f << "  \"train_loss\": "            << state.train_loss   << ",\n";
    f << "  \"eval_loss\": "             << state.eval_loss    << ",\n";
    f << "  \"best_eval_loss\": "        << state.best_eval_loss << ",\n";
    f << "  \"is_best\": "               << (is_best ? "true" : "false") << ",\n";
    f << "  \"notes\": \""               << json_escape(notes) << "\",\n";
    f << "  \"metrics\": {";
    bool first = true;
    for (auto& [k, v] : metrics) {
        if (!first) f << ",";
        f << "\n    \"" << json_escape(k) << "\": " << v;
        first = false;
    }
    f << "\n  }\n}\n";
}

void CheckpointManager::save_best_info() const {
    auto info_path = dir_ / "best_checkpoint_info.json";
    std::ofstream f(info_path);
    if (!f.is_open()) return;
    f << "{\n";
    f << "  \"best_value\": " << best_value_ << ",\n";
    f << "  \"best_metric\": \"" << json_escape(opts_.best_metric) << "\",\n";
    f << "  \"best_mode_min\": " << (opts_.best_mode_min ? "true" : "false") << ",\n";
    f << "  \"updated_at\": \"" << now_iso() << "\"\n";
    f << "}\n";
}

void CheckpointManager::load_best_info() {
    auto info_path = dir_ / "best_checkpoint_info.json";
    if (!std::filesystem::exists(info_path)) return;
    std::ifstream f(info_path);
    // Minimal parser: just look for "best_value"
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find("\"best_value\":");
        if (pos != std::string::npos) {
            try { best_value_ = std::stod(line.substr(pos + 13)); } catch (...) {}
        }
    }
    if (std::filesystem::exists(dir_ / "best_model.pt"))
        best_path_ = dir_ / "best_model.pt";
}

} // namespace xorzen
