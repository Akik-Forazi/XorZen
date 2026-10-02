// ============================================================
//  xorzen.cpp — src/optimized/thread_pool.cpp
//  High-performance thread pool for parallel expert dispatch
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
#include "xorzen/optimized/thread_pool.h"

namespace xorzen {
namespace optimized {

ThreadPool::ThreadPool(size_t n_threads) {
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 4; // Fallback
    }

    for (size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::parallel_for(size_t n, std::function<void(size_t)> f) {
    if (n == 0) return;
    if (n == 1 || workers_.empty()) {
        f(0);
        return;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        futures.push_back(submit(f, i));
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    done_condition_.wait(lock, [this]() {
        return tasks_.empty() && active_tasks_ == 0;
    });
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this]() {
                return stop_ || !tasks_.empty();
            });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
            active_tasks_++;
        }

        task();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            active_tasks_--;
        }
        done_condition_.notify_all();
    }
}

// Global thread pool implementation
static std::unique_ptr<ThreadPool> g_thread_pool = nullptr;
static std::mutex g_tp_mutex;
static size_t g_num_threads = 0;

ThreadPool& global_thread_pool() {
    std::lock_guard<std::mutex> lock(g_tp_mutex);
    if (!g_thread_pool) {
        g_thread_pool = std::make_unique<ThreadPool>(g_num_threads);
    }
    return *g_thread_pool;
}

void set_num_threads(size_t n) {
    std::lock_guard<std::mutex> lock(g_tp_mutex);
    if (g_thread_pool) {
        // Pool already exists, we could recreate it or warn
        // For now, let's just ignore if it's the same, or warn if different
        return;
    }
    g_num_threads = n;
}

size_t get_num_threads() {
    return global_thread_pool().num_threads();
}

} // namespace optimized
} // namespace xorzen
