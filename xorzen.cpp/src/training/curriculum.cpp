// ============================================================
//  xorzen.cpp — src/training/curriculum.cpp
//  CurriculumTrainer: file-by-file progressive training
//  Ported from xorzen/training/curriculum.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/curriculum.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace xorzen {

// ============================================================
//  BookDataset
// ============================================================

BookDataset::BookDataset(const std::vector<int64_t>& token_ids,
                         int64_t seq_len,
                         int64_t max_tokens) {
    // Hard cap
    int64_t n = std::min(static_cast<int64_t>(token_ids.size()), max_tokens);
    for (int64_t i = 0; i + seq_len + 1 <= n; i += seq_len) {
        std::vector<int64_t> src(token_ids.begin() + i,
                                 token_ids.begin() + i + seq_len);
        std::vector<int64_t> tgt(token_ids.begin() + i + 1,
                                 token_ids.begin() + i + seq_len + 1);
        if (static_cast<int64_t>(src.size()) == seq_len)
            sequences_.emplace_back(std::move(src), std::move(tgt));
    }
}

std::pair<BookDataset::Slice, BookDataset::Slice>
BookDataset::split(double train_frac) const {
    int64_t n = static_cast<int64_t>(sequences_.size());
    int64_t split = std::max(int64_t{1}, static_cast<int64_t>(n * train_frac));
    return {Slice{sequences_, 0, split},
            Slice{sequences_, split, n}};
}

// Slice helpers
std::pair<torch::Tensor, torch::Tensor>
BookDataset::Slice::get_batch(const std::vector<int64_t>& indices) const {
    int64_t B = static_cast<int64_t>(indices.size());
    int64_t L = static_cast<int64_t>(seqs_[start_ + indices[0]].first.size());
    auto src_t = torch::zeros({B, L}, torch::kInt64);
    auto tgt_t = torch::zeros({B, L}, torch::kInt64);
    for (int64_t b = 0; b < B; ++b) {
        auto& [src, tgt] = seqs_[start_ + indices[b]];
        for (int64_t i = 0; i < L; ++i) {
            src_t[b][i] = src[i];
            tgt_t[b][i] = tgt[i];
        }
    }
    return {src_t, tgt_t};
}

std::pair<torch::Tensor, torch::Tensor>
BookDataset::Slice::random_batch(int64_t batch_size,
                                 std::mt19937& rng) const {
    int64_t sz = size();
    if (sz == 0) return {torch::zeros({0}), torch::zeros({0})};
    int64_t B = std::min(batch_size, sz);
    std::vector<int64_t> idx(sz);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(B);
    return get_batch(idx);
}

// ============================================================
//  CurriculumTrainer
// ============================================================

CurriculumTrainer::CurriculumTrainer(torch::nn::Module& model,
                                     TokenizerFn tokenize_fn,
                                     const std::filesystem::path& data_dir,
                                     const std::filesystem::path& output_dir,
                                     const CurriculumConfig& cfg)
    : model_(model), tokenize_(std::move(tokenize_fn)),
      data_dir_(data_dir), output_dir_(output_dir), cfg_(cfg),
      rng_(std::random_device{}()) {
    std::filesystem::create_directories(output_dir_);
    log_path_   = output_dir_ / "curriculum_log.jsonl";
    state_path_ = output_dir_ / "curriculum_state.json";

    // Build AdamW optimizer
    optimizer_ = std::make_unique<torch::optim::AdamW>(
        model_.parameters(),
        torch::optim::AdamWOptions(cfg_.learning_rate)
            .weight_decay(cfg_.weight_decay));
}

// ---- public entry point ----

void CurriculumTrainer::train() {
    // Collect files
    std::vector<std::filesystem::path> files;
    for (auto& e : std::filesystem::directory_iterator(data_dir_)) {
        if (e.path().extension() == ".txt")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("No .txt files found in " + data_dir_.string());

    std::cout << "[Curriculum] Found " << files.size() << " books in " << data_dir_ << "\n";

    // Resume
    int64_t start_idx = 0;
    if (cfg_.resume) start_idx = try_resume();

    if (cfg_.shuffle_files) std::shuffle(files.begin(), files.end(), rng_);

    for (int64_t i = 0; i < static_cast<int64_t>(files.size()); ++i) {
        if (i < start_idx) continue;

        std::cout << "[Curriculum] [" << (i+1) << "/" << files.size() << "] "
                  << files[i].filename() << "\n";

        auto result = train_one_book(files[i], i, 0);
        log_result(result);

        if (!std::isnan(result.eval_loss) &&
            result.eval_loss > cfg_.revisit_loss_threshold &&
            result.revisits_done < cfg_.max_revisits) {
            revisit_queue_.push_back({files[i], cfg_.max_revisits - result.revisits_done});
            std::cout << "[Curriculum]   → Queued for revisit (eval_loss="
                      << std::fixed << std::setprecision(3) << result.eval_loss << ")\n";
        }

        ++state_.books_completed;

        if ((i + 1) % cfg_.save_every_n_books == 0)
            save_checkpoint("book_" + std::to_string(i + 1));
    }

    // Revisit pass
    if (!revisit_queue_.empty()) {
        std::cout << "[Curriculum] Revisit pass: " << revisit_queue_.size() << " books\n";
        for (auto& [path, left] : revisit_queue_) {
            if (std::filesystem::exists(path)) {
                auto result = train_one_book(path, -1, cfg_.max_revisits - left + 1);
                log_result(result);
            }
        }
    }

    save_checkpoint("final");
    write_summary();
    std::cout << "[Curriculum] Training complete.\n";
}

// ---- train one book ----

CurriculumTrainer::BookResult
CurriculumTrainer::train_one_book(const std::filesystem::path& path,
                                  int64_t /*book_idx*/, int revisit_num) {
    auto t0 = std::chrono::steady_clock::now();

    // Read text
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[Curriculum] Cannot read " << path << "\n";
        return empty_result(path, revisit_num);
    }
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    // Tokenise
    auto token_ids = tokenize_(text);
    if (static_cast<int64_t>(token_ids.size()) < cfg_.max_seq_len * 2) {
        std::cout << "[Curriculum]   Skipping (too short: " << token_ids.size() << " tokens)\n";
        return empty_result(path, revisit_num);
    }

    // Build dataset
    BookDataset dataset(token_ids, cfg_.max_seq_len, cfg_.max_tokens_per_book);
    auto [train_ds, eval_ds] = dataset.split(cfg_.train_split);
    if (train_ds.size() == 0) return empty_result(path, revisit_num);

    double train_loss = run_train_steps(train_ds);
    double eval_loss  = run_eval(eval_ds);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[Curriculum]   train=" << std::fixed << std::setprecision(4) << train_loss
              << "  eval=" << eval_loss
              << "  seqs=" << dataset.size()
              << "  t=" << std::setprecision(1) << elapsed << "s\n";

    // Parse author / title from filename (Author___Title.txt)
    std::string stem = path.stem().string();
    std::string author = "Unknown", title = stem;
    auto sep = stem.find("___");
    if (sep != std::string::npos) {
        author = stem.substr(0, sep);
        title  = stem.substr(sep + 3);
    }

    return {path.filename().string(), author, title,
            train_loss, eval_loss,
            dataset.size(), static_cast<int64_t>(token_ids.size()),
            elapsed, revisit_num, now_iso()};
}

// ---- train steps ----

double CurriculumTrainer::run_train_steps(const BookDataset::Slice& ds) {
    model_.train();
    double total_loss = 0.0;
    for (int64_t step = 0; step < cfg_.steps_per_book; ++step) {
        // LR warmup
        double lr = cfg_.learning_rate;
        if (step < cfg_.warmup_steps)
            lr *= static_cast<double>(step + 1) / cfg_.warmup_steps;
        for (auto& pg : optimizer_->param_groups())
            pg.options().set_lr(lr);

        auto [src, tgt] = ds.random_batch(cfg_.batch_size, rng_);
        optimizer_->zero_grad();
        auto loss = forward_loss(src, tgt);
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model_.parameters(), cfg_.grad_clip);
        optimizer_->step();

        total_loss += loss.item<double>();
        ++state_.total_steps;

        if ((step + 1) % cfg_.log_every_n_steps == 0) {
            std::cout << "[Curriculum]     step " << (step+1) << "/" << cfg_.steps_per_book
                      << "  loss=" << std::fixed << std::setprecision(4)
                      << total_loss / (step + 1) << "\n";
        }
    }
    return total_loss / cfg_.steps_per_book;
}

// ---- eval ----

double CurriculumTrainer::run_eval(const BookDataset::Slice& ds) {
    if (ds.size() == 0) return std::numeric_limits<double>::quiet_NaN();
    torch::NoGradGuard ng;
    model_.eval();
    double total_loss = 0.0;
    int64_t n_batches = 0;
    int64_t batch_size = cfg_.batch_size * 2;
    int64_t sz = ds.size();
    for (int64_t start = 0; start < sz; start += batch_size) {
        std::vector<int64_t> idx;
        for (int64_t j = start; j < std::min(start + batch_size, sz); ++j) idx.push_back(j);
        auto [src, tgt] = ds.get_batch(idx);
        total_loss += forward_loss(src, tgt).item<double>();
        ++n_batches;
    }
    return total_loss / std::max(n_batches, int64_t{1});
}

// ---- forward loss ----

torch::Tensor CurriculumTrainer::forward_loss(const torch::Tensor& src,
                                              const torch::Tensor& tgt) {
    // Call model — handle both returning struct and plain tensor
    auto output = model_.forward({src});
    // Try to get logits attribute from output (ModelOutput struct)
    torch::Tensor logits;
    if (output.isTensor()) {
        logits = output.toTensor();
    } else if (output.isGenericDict()) {
        logits = output.toGenericDict().at("logits").toTensor();
    } else {
        // Tuple: first element is logits
        logits = output.toTuple()->elements()[0].toTensor();
    }

    // logits: [B, T, V]
    auto B = logits.size(0), T = logits.size(1), V = logits.size(2);
    return torch::nn::functional::cross_entropy(
        logits.reshape({B * T, V}),
        tgt.reshape({B * T}),
        torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));
}

// ---- checkpoint / logging ----

void CurriculumTrainer::save_checkpoint(const std::string& tag) {
    auto path = output_dir_ / ("checkpoint_" + tag + ".pt");
    torch::serialize::OutputArchive ar;
    model_.save(ar);
    ar.save_to(path.string());

    // Save optimizer state separately
    torch::save(*optimizer_, (output_dir_ / ("opt_" + tag + ".pt")).string());

    // Save state JSON
    std::ofstream f(state_path_);
    f << "{\n"
      << "  \"books_completed\": " << state_.books_completed << ",\n"
      << "  \"total_steps\": " << state_.total_steps << "\n"
      << "}\n";
    std::cout << "[Curriculum]   Checkpoint saved → " << path.filename() << "\n";
}

int64_t CurriculumTrainer::try_resume() {
    // Find latest checkpoint
    std::vector<std::filesystem::path> ckpts;
    for (auto& e : std::filesystem::directory_iterator(output_dir_)) {
        if (e.path().filename().string().find("checkpoint_") == 0 &&
            e.path().extension() == ".pt")
            ckpts.push_back(e.path());
    }
    if (ckpts.empty()) return 0;
    std::sort(ckpts.begin(), ckpts.end());
    auto latest = ckpts.back();
    std::cout << "[Curriculum] Resuming from " << latest.filename() << "\n";
    torch::serialize::InputArchive ar;
    ar.load_from(latest.string());
    model_.load(ar);
    // Restore state
    std::ifstream sf(state_path_);
    if (sf.is_open()) {
        std::string content((std::istreambuf_iterator<char>(sf)),
                             std::istreambuf_iterator<char>());
        auto bpos = content.find("\"books_completed\":");
        if (bpos != std::string::npos) {
            state_.books_completed = std::stoll(content.substr(bpos + 18));
        }
    }
    return state_.books_completed;
}

void CurriculumTrainer::log_result(const BookResult& r) {
    state_.book_results.push_back(r);
    std::ofstream f(log_path_, std::ios::app);
    f << "{\"file\":" << "\"" << r.file << "\""
      << ",\"train_loss\":" << r.train_loss
      << ",\"eval_loss\":" << r.eval_loss
      << ",\"n_sequences\":" << r.n_sequences
      << ",\"elapsed_s\":" << r.elapsed_s
      << ",\"timestamp\":\"" << r.timestamp << "\""
      << "}\n";
}

void CurriculumTrainer::write_summary() {
    auto& results = state_.book_results;
    std::vector<const BookResult*> valid;
    for (auto& r : results)
        if (!std::isnan(r.eval_loss)) valid.push_back(&r);
    if (valid.empty()) return;

    double avg = 0;
    for (auto* r : valid) avg += r->eval_loss;
    avg /= valid.size();

    auto best  = *std::min_element(valid.begin(), valid.end(),
        [](const BookResult* a, const BookResult* b){ return a->eval_loss < b->eval_loss; });
    auto worst = *std::max_element(valid.begin(), valid.end(),
        [](const BookResult* a, const BookResult* b){ return a->eval_loss < b->eval_loss; });

    double total_time = 0;
    for (auto* r : valid) total_time += r->elapsed_s;

    std::ofstream f(output_dir_ / "curriculum_summary.json");
    f << std::fixed << std::setprecision(4);
    f << "{\n"
      << "  \"books_trained\": " << valid.size() << ",\n"
      << "  \"total_steps\": " << state_.total_steps << ",\n"
      << "  \"avg_eval_loss\": " << avg << ",\n"
      << "  \"best_book\": \"" << best->file << "\",\n"
      << "  \"best_eval_loss\": " << best->eval_loss << ",\n"
      << "  \"worst_book\": \"" << worst->file << "\",\n"
      << "  \"worst_eval_loss\": " << worst->eval_loss << ",\n"
      << "  \"total_time_minutes\": " << (total_time / 60.0) << "\n"
      << "}\n";

    std::cout << "[Curriculum] Summary: avg_eval=" << avg
              << " best=" << best->eval_loss << " worst=" << worst->eval_loss << "\n";
}

CurriculumTrainer::BookResult
CurriculumTrainer::empty_result(const std::filesystem::path& path, int revisit_num) {
    return {path.filename().string(), "Unknown", path.stem().string(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            0, 0, 0.0, revisit_num, now_iso()};
}

std::string CurriculumTrainer::now_iso() {
    auto t  = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&tt), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

} // namespace xorzen
