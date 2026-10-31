// heuristics.cpp
#include "heuristics.h"

#include <cmath>

namespace gk {

double pattern_score(int count, bool open_ends) {
    if (count >= 5) return FORCED_WIN_SCORE;
    if (count == 4) return open_ends ? TACTICAL_OPEN_FOUR : TACTICAL_FOUR;
    if (count == 3) return open_ends ? TACTICAL_OPEN_THREE : TACTICAL_BLOCK_THREE;
    if (count == 2) return TACTICAL_TWO * (open_ends ? 2.0 : 1.0);
    return TACTICAL_ONE;
}

void move_scores(const Pos& pos, const std::vector<int>& legal, int me, PriorBuf& buf) {
    const int n = static_cast<int>(legal.size());
    buf.resize(n);

    for (int k = 0; k < n; ++k) {
        const int idx = legal[k];

        // 0) 廉价预筛：8 邻域没有任何棋子 -> 最多形成「孤二」
        if (!pos.has_neighbor(idx)) {
            buf.raw[k] = TACTICAL_ONE;
            continue;
        }

        int c_me = 0, c_op = 0;
        bool o_me = false, o_op = false;
        pos.line_after_both(idx, me, c_me, o_me, c_op, o_op);
        if (c_me == 0) {  // 越界 / 已占用（合法点里不应出现）
            buf.raw[k] = 0.0;
            continue;
        }

        // 1) 我落这里立刻成五
        if (c_me >= 5) {
            buf.raw[k] = FORCED_WIN_SCORE;
            buf.win[k] = 1;
            buf.tier[k] = 3;
            continue;
        }
        // 2) 对方落这里立刻成五（我必须堵）
        if (c_op >= 5) {
            buf.raw[k] = FORCED_BLOCK_SCORE;
            buf.block[k] = 1;
            buf.tier[k] = 2;
            continue;
        }
        // 2b) 四的争夺：我落这里能做出四（含活四），或对方落这里能做出四。
        //     这类点不该因为宽度剪枝被剔除——漏一个冲四点往往直接输棋。
        //     注意：这里只做「保留保护」，不改变分数（分数仍由下面的棋型分级决定）。
        if (c_me == 4 || c_op == 4) buf.tier[k] = 1;
        // 3) 常规：我方威胁 + 对方威胁的防守价值
        const double off_me = pattern_score(c_me, o_me);
        const double def_op = pattern_score(c_op, o_op);
        buf.raw[k] = (off_me > def_op * 0.8 ? off_me : def_op * 0.8) +
                     0.1 * (off_me < def_op ? off_me : def_op);
    }
}

void compute_prior(const Pos& pos, const std::vector<int>& legal, const float* net_probs,
                   double heuristic_weight, double policy_weight, bool use_heuristics,
                   PriorBuf& buf) {
    const int n = static_cast<int>(legal.size());
    buf.resize(n);
    if (n == 0) return;

    if (!use_heuristics) {
        if (net_probs == nullptr) {
            const float u = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; ++i) buf.priors[i] = u;
            return;
        }
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += net_probs[i];
        if (s > 0) {
            for (int i = 0; i < n; ++i) buf.priors[i] = static_cast<float>(net_probs[i] / s);
        } else {
            const float u = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; ++i) buf.priors[i] = u;
        }
        return;
    }

    move_scores(pos, legal, pos.player, buf);

    bool any_win = false, any_block = false;
    for (int i = 0; i < n; ++i) {
        if (buf.win[i]) any_win = true;
        if (buf.block[i]) any_block = true;
        buf.forced[i] = static_cast<uint8_t>(buf.win[i] | buf.block[i]);
    }

    // ---- 强制点：完全忽略网络 ----
    if (any_win || any_block) {
        const uint8_t* sel = any_win ? buf.win.data() : buf.block.data();
        int cnt = 0;
        for (int i = 0; i < n; ++i) cnt += sel[i] ? 1 : 0;
        if (cnt > 0) {
            const float p = 1.0f / static_cast<float>(cnt);
            for (int i = 0; i < n; ++i) buf.priors[i] = sel[i] ? p : 0.0f;
            return;
        }
        // 强制点为空（理论不应发生）-> 落到常规路径
    }

    // ---- 常规：棋型分开方压缩后归一化 ----
    double tot = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = buf.raw[i] > 0.0 ? std::sqrt(buf.raw[i]) : 0.0;
        buf.raw[i] = v;
        tot += v;
    }

    if (net_probs == nullptr || policy_weight <= 0.0) {
        if (tot > 0.0) {
            for (int i = 0; i < n; ++i) buf.priors[i] = static_cast<float>(buf.raw[i] / tot);
        } else {
            const float u = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; ++i) buf.priors[i] = u;
        }
        return;
    }

    double ps = 0.0;
    for (int i = 0; i < n; ++i) ps += net_probs[i] > 0.0f ? net_probs[i] : 0.0;
    double mixed_tot = 0.0;
    static thread_local std::vector<double> mixed;
    mixed.resize(n);
    for (int i = 0; i < n; ++i) {
        const double h = tot > 0.0 ? buf.raw[i] / tot : 1.0 / n;
        const double p = ps > 0.0 ? (net_probs[i] > 0.0f ? net_probs[i] / ps : 0.0) : 1.0 / n;
        mixed[i] = heuristic_weight * h + policy_weight * p;
        mixed_tot += mixed[i];
    }
    if (mixed_tot > 0.0) {
        for (int i = 0; i < n; ++i) buf.priors[i] = static_cast<float>(mixed[i] / mixed_tot);
    } else {
        const float u = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) buf.priors[i] = u;
    }
}

}  // namespace gk
