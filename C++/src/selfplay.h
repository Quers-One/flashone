// selfplay.h — 一盘自对弈（对标 PYTHON/train.py:673 self_play_game）
//
// 语义对齐点（改这里要同步改 Python 侧的测试）：
//   * 温度退火按「该步之前的手数」：<12 用 1.0、<30 用 0.5、之后 0.0（贪心）
//   * 策略目标 = 根节点访问数归一化，与温度无关
//   * 认输：每步额外跑一次网络取价值，连续 resign_plies 步 < resign_threshold 就认输；
//     **认输那一步及其后的样本全部丢弃**（不是把 value 打成 0）
//   * value 标签 = 该步执子方视角的最终胜负（+1/0/-1）
//   * 落子后复用子树（set_root_move），与 Python 的 mcts.set_root_move 一致
#pragma once

#include <cstdint>
#include <vector>

#include "board.h"
#include "dataset.h"
#include "mcts.h"

namespace gk {

struct PlayOptions {
    // 搜索
    int n_sim = 400;
    double c_puct = 1.0;
    int max_children = 6;
    int root_max_children = 16;
    double heuristic_weight = 0.5;
    double policy_weight = 1.0;
    bool use_heuristics = true;

    // VCF 静态搜索（见 vcf.h）。默认关闭，理由见 config.h 实测注记。
    bool vcf = false;
    int vcf_max_ply = 6;
    int vcf_max_nodes = 128;
    bool guarantee_rush = false;

    // 温度
    double temp_open = 1.0;   // ply < temp_threshold1
    double temp_mid = 0.5;    // ply < temp_threshold2
    int temp_threshold1 = 12;
    int temp_threshold2 = 30;

    // 根噪声
    double noise_alpha = 0.15;
    double noise_eps = 0.25;

    // 随机开局（见 config.h 的说明）。0 = 关闭，行为与旧版完全一致。
    // 前 sp_open_plies 手用均匀随机合法点落子，**不产生样本**，之后才 set_root 开始正式搜索。
    int sp_open_plies = 0;

    // 认输（<= -1 表示关闭）
    double resign_threshold = -0.95;
    int resign_plies = 3;

    // 安全上限：0 = 不限制（Python 侧没有这个，只用来防死循环）
    int max_plies = 0;
};

struct GameStats {
    int plies = 0;          // 实际下了多少手（= 记录到的步数）
    int winner = 0;         // 0 平 / 1 黑 / 2 白
    bool resigned = false;
    bool truncated = false;
    double elapsed_ms = 0.0;
    int64_t evaluations = 0;
};

// 下一盘棋，把可用样本追加进 samples（不清空）。返回 winner。
int play_game(Evaluator* net, const PlayOptions& opt, std::vector<Sample>& samples,
              GameStats* stats = nullptr, uint64_t seed = 0);

}  // namespace gk
