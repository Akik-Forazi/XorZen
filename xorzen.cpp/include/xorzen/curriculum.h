#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/curriculum.h
//  CurriculumTrainer: file-by-file progressive training
//  Ported from xorzen/training/curriculum.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================

#include <torch/torch.h>
#include <filesystem>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace xorzen {

// ============================================================
//  CurriculumConfig
// ============================================================
struct CurriculumConfig {
    // Data
    bool    shuffle_files            = true;
    double  train_split              = 0.90;
    int64_t max_seq_len              = 512;
    int64_t max_tokens_per_book      = 50'000;

    // Training per book
    int64_t steps_per_book           = 64;
    int64_t batch_size               = 4;
    double  learning_rate            = 3e-4;
    double  weight_decay             = 0.01;
    double  grad_clip                = 1.0;
    int64_t warmup_steps             = 4;

    // Revisit logic
    double  revisit_loss_threshold   = 3.5;
    int     max_revisits             = 2;

    // Checkpointing
    int64_t save_every_n_books       = 10;
    bool    resume                   = true;

    // Logging
    int64_t log_every_n_steps        = 16;
};

// ============================================================
//  BookDataset — tokenises one text file into sequences
// ============================================================
class BookDataset {
public:
    using Sequence = std::pair<std::vector<int64_t>, std::vector<int64_t>>;

    struct Slice {
        const std::vector<Sequence>& seqs_;
        int64_t start_, end_;
        int64_t size() const { return end_ - start_; }

        std::pair<torch::Tensor, torch::Tensor>
        get_batch(const std::vector<int64_t>& indices) const;

        std::pair<torch::Tensor, torch::Tensor>
        random_batch(int64_t batch_size, std::mt19937& rng) const;
    };

    BookDataset(const std::vector<int64_t>& token_ids,
                int64_t seq_len,
                int64_t max_tokens);

    int64_t size() const { return static_cast<int64_t>(sequences_.size()); }

    // Returns (train_slice, eval_slice)
    std::pair<Slice, Slice> split(double train_frac) const;

private:
    std::vector<Sequence> sequences_;
};

// ============================================================
//  CurriculumTrainer
// ============================================================
using TokenizerFn = std::function<std::vector<int64_t>(const std::string&)>;

class CurriculumTrainer {
public:
    struct BookResult {
        std::string file, author, title;
        double train_loss = std::numeric_limits<double>::quiet_NaN();
        double eval_loss  = std::numeric_limits<double>::quiet_NaN();
        int64_t n_sequences = 0;
        int64_t n_tokens    = 0;
        double  elapsed_s   = 0.0;
        int     revisits_done = 0;
        std::string timestamp;
    };

    struct State {
        int64_t books_completed = 0;
        int64_t total_steps     = 0;
        std::vector<BookResult> book_results;
    };

    CurriculumTrainer(torch::nn::Module& model,
                      TokenizerFn tokenize_fn,
                      const std::filesystem::path& data_dir,
                      const std::filesystem::path& output_dir,
                      const CurriculumConfig& cfg = {});

    void train();

private:
    torch::nn::Module& model_;
    TokenizerFn tokenize_;
    std::filesystem::path data_dir_;
    std::filesystem::path output_dir_;
    CurriculumConfig cfg_;
    std::mt19937 rng_;
    std::unique_ptr<torch::optim::AdamW> optimizer_;

    std::filesystem::path log_path_;
    std::filesystem::path state_path_;
    State state_;
    std::vector<std::pair<std::filesystem::path, int>> revisit_queue_;

    BookResult train_one_book(const std::filesystem::path& path,
                              int64_t book_idx, int revisit_num);
    double run_train_steps(const BookDataset::Slice& ds);
    double run_eval(const BookDataset::Slice& ds);
    torch::Tensor forward_loss(const torch::Tensor& src, const torch::Tensor& tgt);

    void    save_checkpoint(const std::string& tag);
    int64_t try_resume();
    void    log_result(const BookResult& r);
    void    write_summary();

    static BookResult empty_result(const std::filesystem::path& path, int revisit_num);
    static std::string now_iso();
};

} // namespace xorzen
