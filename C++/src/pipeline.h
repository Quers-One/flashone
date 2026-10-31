// pipeline.h — 一代的完整流程：自对弈 → 回放池 → 训练 → 门控 → 落盘
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "dataset.h"
#include "net.h"
#include "selfplay.h"

namespace gk {

struct SelfPlayStats {
    int games = 0;
    int black_wins = 0;
    int white_wins = 0;
    int draws = 0;
    int resigned = 0;
    int truncated = 0;
    int samples = 0;
    long long plies = 0;
    long long evaluations = 0;
    double elapsed_ms = 0.0;
    double forward_ms = 0.0;   // 评估器累计墙钟（含等待合并）
};

// 并行自对弈：workers 个线程各持自己的 MCTS，共用 ev（一般是 NetService）。
SelfPlayStats collect_games(Evaluator* ev, const TrainConfig& cfg, int games, uint64_t seed,
                            std::vector<Sample>& out);

// 多进程自对弈（照 Python 版的做法）：把 weights_path 的权重交给 procs 个子进程，
// 每个子进程内部只有 1 局在做，于是每个进程都有独立的 CUDA 上下文 —— 这正是
// Python 版 16 进程能比「单进程多线程」更好地重叠 GPU 同步的原因。
// 子进程把样本写到 tmpdir/w<i>.bin，父进程读回来合并。
SelfPlayStats collect_games_mp(const TrainConfig& cfg, const std::string& weights_path, int games,
                               uint64_t seed, const std::string& tmpdir, std::vector<Sample>& out);

// 把 TrainConfig 翻译成搜索/对局参数
MCTSOptions make_mcts_options(const TrainConfig& cfg);
PlayOptions make_play_options(const TrainConfig& cfg);

struct LoopOptions {
    TrainConfig cfg;
    std::string dir;          // 输出目录（权重、回放池、日志）
    std::string init_weights; // 起始权重（GKAZ1 文件），空 = 随机初始化
    std::string resume_dir;   // 断点续训目录（空 = 不续训）
    int gens = 10;
};

int run_training_loop(const LoopOptions& opt);

}  // namespace gk
