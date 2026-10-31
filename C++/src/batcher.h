// batcher.h — 跨对局共享的批量评估器
//
// 动机（NOTES.md:564 的结论）：Python 版每个自对弈进程都在自己的 16 叶批次上
// 同步等待 GPU，单次前向的**固定开销**（≈1.05ms）远大于算力本身，所以把多个
// 对局的叶子合并成一个大 batch 是唯一还能拿到 2× 以上的结构性改动。
//
// 与 NOTES.md:1229 否掉的「单局叶批量 16→64」不同：本类**不改变任何一局搜索树
// 的形状**（每局仍然是 n_sim=400、每批 16 叶），只是把同一时刻多个对局的请求
// 拼进一次前向。因此逐手行为与单局版等价，只是墙钟更快。
//
// 多线程：GPU 单次前向的固定开销很大，一条线程「提交→同步→取回」的串行会把
// GPU 空转掉。所以默认开 2 个评估线程，每个线程各自一条 CUDA stream（互不
// 阻塞地同步），实测 2 线程 b16 的总吞吐从 6.5k 状态/s 提到 9.8k。
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "board.h"
#include "mcts.h"

namespace gk {

class NetService : public Evaluator {
public:
    // n_threads: 评估线程数（各自独立 CUDA stream）；0/1 表示单线程
    NetService(Evaluator* inner, int max_states = 256, double window_ms = 0.4,
               int n_threads = 2);
    ~NetService() override;

    void start();   // 启动内部评估线程
    void stop();    // 停止并 join

    // 线程安全、阻塞：把 (states, n) 提交给评估线程，结果写回 policies/values
    void evaluate_batch(const float* states, int batch, float* policies, float* values) override;

    struct Stats {
        uint64_t requests = 0;   // 收到的 evaluate_batch 调用数
        uint64_t batches = 0;    // 实际前向次数
        uint64_t states = 0;     // 评估过的状态总数
        double wait_ms = 0.0;    // 调用方累计等待时间（含排队 + 前向）
        double service_ms = 0.0; // 评估线程累计花在 inner 前向里的时间
        std::vector<double> fwd_ms;  // 每次前向的耗时样本（用于看 p50/p95/max 分布）
    };
    Stats stats() const;

    double avg_batch() const {
        const Stats s = stats();
        return s.batches ? (double)s.states / (double)s.batches : 0.0;
    }

private:
    struct Req {
        const float* states;
        int n;
        float* policies;
        float* values;
        bool done;
    };

    void loop_worker();

    Evaluator* inner_;
    int max_states_;
    double window_ms_;
    int n_threads_;

    mutable std::mutex mu_;
    std::condition_variable cv_;       // 评估线程等待新请求
    std::condition_variable done_cv_;  // 调用方等待自己那批完成
    std::deque<Req*> queue_;
    std::atomic<bool> quit_{false};
    std::vector<std::thread> ths_;

    mutable std::mutex stat_mu_;
    Stats stats_;
};

}  // namespace gk
