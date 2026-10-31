// heuristics.h — 棋型规则先验（对标 PYTHON/heuristics.py）
//
// 设计原则与 Python 版一致：
//   * 出现「一步成五」-> 只保留成五点（网络先验被完全忽略）
//   * 否则出现「对方一步成五」-> 只保留堵点
//   * 其余情况：威胁分量级分离后开方压缩 -> 与网络概率按权重线性融合
//
// 与 Python 版的唯一差别是：所有中间缓冲由调用方复用（PriorBuf），
// 搜索里每次扩展不再重新分配 vector —— 这是纯 C++ 侧的优化，不影响数值。
#pragma once

#include <vector>

#include "board.h"

namespace gk {

// 威胁分级基准分：数值必须量级分离（1e6 / 1e5 / 1e4 / 1e3），否则搜索会把
// 模拟预算摊到「看起来不错」的点上而漏挡四连。详见 heuristics.py 顶部说明。
inline constexpr double TACTICAL_OPEN_FOUR = 1000000.0;   // 我能做活四（严格必胜）
inline constexpr double TACTICAL_FOUR = 100000.0;         // 我能做冲四
inline constexpr double TACTICAL_BLOCK_FOUR = 10000.0;    // 我能阻止对方四连
inline constexpr double TACTICAL_OPEN_THREE = 1000.0;     // 我能做活三
inline constexpr double TACTICAL_BLOCK_THREE = 500.0;     // 我能阻止对方活三
inline constexpr double TACTICAL_TWO = 100.0;             // 二连
inline constexpr double TACTICAL_ONE = 4.0;               // 单子

inline constexpr double FORCED_WIN_SCORE = 1e9;    // 必胜点象征分
inline constexpr double FORCED_BLOCK_SCORE = 1e8;  // 必挡点象征分

inline constexpr double DEFAULT_HEURISTIC_WEIGHT = 1.0;
inline constexpr double DEFAULT_POLICY_WEIGHT = 1.0;

double pattern_score(int count, bool open_ends);

// 一次扩展需要的所有中间缓冲（跨调用复用，避免热循环里 new/delete）
struct PriorBuf {
    std::vector<double> raw;
    std::vector<uint8_t> win;
    std::vector<uint8_t> block;
    std::vector<uint8_t> forced;   // 成五点 ∪ 必挡点
    // 强制程度分级，供宽度剪枝保护（借鉴 Rapfi 的 MovePicker 分阶段拾取）：
    //   3 = 我能一步成五
    //   2 = 对方能一步成五（必须堵）
    //   1 = 我能做出四 / 必须堵对方的四（不保证胜负，但漏掉就是战术事故）
    //   0 = 其余
    std::vector<uint8_t> tier;
    std::vector<float> priors;

    void resize(int n) {
        raw.resize(n);
        win.assign(n, 0);
        block.assign(n, 0);
        forced.assign(n, 0);
        tier.assign(n, 0);
        priors.resize(n);
    }
};

// 计算所有合法点的启发式分。legal 为扁平下标（y*SIZE+x）升序。
// 返回后 buf.raw / buf.win / buf.block 有效。
void move_scores(const Pos& pos, const std::vector<int>& legal, int me, PriorBuf& buf);

// 计算归一化先验。
//   net_probs: 网络在 legal 各点上的概率（长度 = legal.size()），可为 nullptr
//   forced:    输出，强制点标记（成五点 ∪ 必挡点），供宽度控制保护
void compute_prior(const Pos& pos, const std::vector<int>& legal, const float* net_probs,
                   double heuristic_weight, double policy_weight, bool use_heuristics,
                   PriorBuf& buf);

}  // namespace gk
