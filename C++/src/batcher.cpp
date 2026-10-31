// batcher.cpp
#include "batcher.h"

#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

namespace gk {

NetService::NetService(Evaluator* inner, int max_states, double window_ms, int n_threads)
    : inner_(inner),
      max_states_(max_states > 0 ? max_states : 256),
      window_ms_(window_ms),
      n_threads_(n_threads > 0 ? n_threads : 1) {
    queue_.clear();
}

NetService::~NetService() { stop(); }

void NetService::start() {
    if (!ths_.empty()) return;
    quit_.store(false);
    for (int i = 0; i < n_threads_; ++i) ths_.emplace_back([this] { loop_worker(); });
}

void NetService::stop() {
    if (ths_.empty()) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_.store(true);
    }
    cv_.notify_all();
    for (auto& t : ths_) t.join();
    ths_.clear();
    // 把还挂在队列里的请求就地处理掉，避免调用方永久等待
    std::deque<Req*> left;
    {
        std::lock_guard<std::mutex> lk(mu_);
        left.swap(queue_);
    }
    for (Req* r : left) {
        inner_->evaluate_batch(r->states, r->n, r->policies, r->values);
        {
            std::lock_guard<std::mutex> lk(mu_);
            r->done = true;
        }
    }
    if (!left.empty()) done_cv_.notify_all();
}

void NetService::evaluate_batch(const float* states, int batch, float* policies, float* values) {
    if (batch <= 0) return;
    const auto t_req = std::chrono::steady_clock::now();
    Req req{states, batch, policies, values, false};
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(&req);
        {
            std::lock_guard<std::mutex> sl(stat_mu_);
            stats_.requests++;
        }
    }
    cv_.notify_one();
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&req] { return req.done; });
    const double waited = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t_req).count();
    {
        std::lock_guard<std::mutex> sl(stat_mu_);
        stats_.wait_ms += waited;
    }
}

NetService::Stats NetService::stats() const {
    std::lock_guard<std::mutex> sl(stat_mu_);
    return stats_;
}

void NetService::loop_worker() {
    using clk = std::chrono::steady_clock;
    // 每个评估线程各自一条 CUDA stream：一个线程在等自己那次前向完成时，
    // 另一个线程可以继续往 GPU 里塞活（默认 stream 是共享的，会互相阻塞）。
    std::unique_ptr<at::cuda::CUDAStreamGuard> stream_guard;
    if (torch::cuda::is_available()) {
        const auto dev = c10::cuda::current_device();
        stream_guard = std::make_unique<at::cuda::CUDAStreamGuard>(
            at::cuda::getStreamFromPool(/*isHighPriority=*/false, dev));
    }
    std::vector<float> st((size_t)max_states_ * 3 * N_CELLS);
    std::vector<float> pol((size_t)max_states_ * N_CELLS);
    std::vector<float> val((size_t)max_states_);

    while (true) {
        std::vector<Req*> work;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return quit_.load() || !queue_.empty(); });
            if (quit_.load() && queue_.empty()) return;
            work.push_back(queue_.front());
            queue_.pop_front();
            int states = work.front()->n;
            // 合并窗口：等一小会儿，让别的对局把请求也挂上来（凑大 batch）
            const auto deadline =
                clk::now() + std::chrono::microseconds((int64_t)(window_ms_ * 1000.0));
            while (states < max_states_) {
                if (!queue_.empty()) {
                    Req* r = queue_.front();
                    queue_.pop_front();
                    if (states + r->n <= max_states_) {
                        work.push_back(r);
                        states += r->n;
                        continue;
                    }
                    // 放不下就留到下一批（放回队首）
                    queue_.push_front(r);
                    break;
                }
                if (clk::now() >= deadline) break;
                // 注意：不能用 cv_.wait_until 等这 0.4ms —— Windows 的等待粒度约 15.6ms，
                // 会让每个 batch 周期凭空多出十几毫秒（实测吞吐掉到 1/5）。改成让出时间片自旋。
                lk.unlock();
                std::this_thread::yield();
                lk.lock();
                if (quit_.load()) break;
            }
        }

        // 合并成一次前向
        int total = 0;
        for (Req* r : work) total += r->n;
        size_t off = 0;
        for (Req* r : work) {
            std::memcpy(st.data() + off, r->states, (size_t)r->n * 3 * N_CELLS * sizeof(float));
            off += (size_t)r->n * 3 * N_CELLS;
        }
        const auto t_fwd = std::chrono::steady_clock::now();
        inner_->evaluate_batch(st.data(), total, pol.data(), val.data());
        const double fwd_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_fwd).count();
        // 拆回各请求
        // 注意：策略缓冲的下标是「状态数 × N_CELLS」，价值缓冲的下标是「状态数」，
        // 两者必须用不同的游标。早期版本用同一个 soff 累加，导致第二个请求起就去
        // val.data() + n0*361 处读价值 —— 越界读（ASan: heap-buffer-overflow），
        // 表现为多线程自对弈偶发 0xC0000005，且搜索用的 value 是垃圾值。
        size_t soff = 0, voff = 0;
        for (Req* r : work) {
            std::memcpy(r->policies, pol.data() + soff, (size_t)r->n * N_CELLS * sizeof(float));
            std::memcpy(r->values, val.data() + voff, (size_t)r->n * sizeof(float));
            soff += (size_t)r->n * N_CELLS;
            voff += (size_t)r->n;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (Req* r : work) r->done = true;
            std::lock_guard<std::mutex> sl(stat_mu_);
            stats_.batches++;
            stats_.states += (uint64_t)total;
            stats_.service_ms += fwd_ms;
            if (stats_.fwd_ms.size() < 200000) stats_.fwd_ms.push_back(fwd_ms);
        }
        done_cv_.notify_all();
    }
}

}  // namespace gk
