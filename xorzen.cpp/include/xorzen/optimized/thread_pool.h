#pragma once

#include <functional>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

namespace xorzen {
namespace optimized {

/**
 * @brief High-performance thread pool for parallel expert dispatch
 * 
 * Based on GGML threading but simplified for expert execution.
 * Uses work-stealing queue for load balancing.
 */
class ThreadPool {
public:
    /**
     * @brief Create thread pool with n_threads workers
     * If n_threads = 0, uses std::thread::hardware_concurrency()
     */
    explicit ThreadPool(size_t n_threads = 0);
    
    ~ThreadPool();
    
    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
    /**
     * @brief Submit a task to the pool
     * Returns a future that will hold the result
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool: submit on stopped pool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }
    
    /**
     * @brief Parallel for loop: execute f(i) for i in [0, n)
     * Blocks until all iterations complete
     */
    void parallel_for(size_t n, std::function<void(size_t)> f);
    
    /**
     * @brief Get number of worker threads
     */
    size_t num_threads() const { return workers_.size(); }
    
    /**
     * @brief Wait for all queued tasks to complete
     */
    void wait_all();
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};
    std::condition_variable done_condition_;
    
    void worker_thread();
};

/**
 * @brief Global thread pool (initialized once, reused everywhere)
 */
ThreadPool& global_thread_pool();

/**
 * @brief Set number of threads for global pool (call before first use)
 */
void set_num_threads(size_t n);

/**
 * @brief Get current number of threads
 */
size_t get_num_threads();

} // namespace optimized
} // namespace xorzen
