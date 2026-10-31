// gate.cpp — 门控对局与判定
#include "gate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "board.h"

namespace gk {
namespace {
inline double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

MCTSOptions make_opts(const GateOptions& o, double weight_h, double weight_p, bool candidate) {
    MCTSOptions mo;
    mo.n_sim = (candidate && o.cand_sims > 0) ? o.cand_sims : o.n_sim;
    mo.c_puct = o.c_puct;
    mo.max_children = o.max_children;
    mo.root_max_children = o.root_max_children;
    mo.use_heuristics = true;
    mo.heuristic_weight = weight_h;
    mo.policy_weight = weight_p;
    mo.vcf = candidate ? o.cand_vcf : o.vcf;
    mo.vcf_root = candidate ? o.cand_vcf_root : o.vcf_root;
    mo.vcf_max_ply = o.vcf_max_ply;
    mo.vcf_max_nodes = o.vcf_max_nodes;
    mo.guarantee_rush = candidate ? o.cand_guarantee_rush : o.guarantee_rush;
    return mo;
}

// 【配方 3 起】配对开局：给定「对号」（第几对，每对两局），生成该对的固定开局落子。
// 第一子永远天元，第二子在对号决定的半径 2 邻域内选一格 —— 这样同一对的两局拿到完全
// 相同的盘面，只有执子方互换，颜色优势因此在两局间抵消。
// 落子数取偶数，保证摆完后仍轮到黑走。
std::vector<int> build_opening(uint64_t pair_seed, int plies) {
    std::vector<int> out;
    if (plies < 2) return out;
    const int n_plies = plies - (plies % 2);  // 偶数化
    uint64_t s = pair_seed ^ 0x9E3779B97F4A7C15ULL;
    if (s == 0) s = 0x2545F4914F6CDD1DULL;
    auto next_rand = [&s]() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    };
    const int c = SIZE / 2;
    out.push_back(c * SIZE + c);  // 天元
    if (n_plies >= 2) {
        std::vector<int> cand;
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const int x = c + dx, y = c + dy;
                if (x < 0 || y < 0 || x >= SIZE || y >= SIZE) continue;
                cand.push_back(y * SIZE + x);
            }
        }
        std::sort(cand.begin(), cand.end());
        out.push_back(cand[(size_t)(next_rand() % cand.size())]);
    }
    return out;
}
}  // namespace

void wilson_interval(int k, int n, double* lo, double* hi, double z) {
    if (n <= 0) {
        if (lo) *lo = 0.0;
        if (hi) *hi = 1.0;
        return;
    }
    const double p = (double)k / (double)n;
    const double d = 1.0 + z * z / n;
    const double c = p + z * z / (2.0 * n);
    const double half = z * std::sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n));
    if (lo) *lo = std::max(0.0, (c - half) / d);
    if (hi) *hi = std::min(1.0, (c + half) / d);
}

int play_gate_game(Evaluator* candidate, Evaluator* best, const GateOptions& opt, bool cand_black,
                   uint64_t seed, const std::vector<int>* opening) {
    MCTS m_cand(candidate, make_opts(opt, opt.heuristic_weight, opt.policy_weight, true), seed);
    MCTS m_best(best, make_opts(opt, opt.heuristic_weight, opt.policy_weight, false), seed + 7919);
    // 评估必须带根噪声，否则 20 局实际只有 2 个独立样本
    m_cand.root_noise_alpha = opt.noise_alpha;
    m_cand.root_noise_eps = opt.noise_eps;
    m_best.root_noise_alpha = opt.noise_alpha;
    m_best.root_noise_eps = opt.noise_eps;

    Pos board;
    m_cand.set_root(board);
    m_best.set_root(board);

    // 配对开局：黑、白、黑、白…… 依次摆上（落子数已偶数化，摆完仍轮到黑走）
    if (opening) {
        for (size_t i = 0; i < opening->size(); ++i) {
            const int mv = (*opening)[i];
            const int who = (i % 2 == 0) ? BLACK : WHITE;
            if (!board.place(mv % SIZE, mv / SIZE, who)) break;
            m_cand.set_root_move((int16_t)mv);
            m_best.set_root_move((int16_t)mv);
        }
    }

    std::vector<float> pol(N_CELLS);
    while (true) {
        if (board.stones >= N_CELLS) return 0;
        const bool cand_turn = (board.player == BLACK) == cand_black;
        MCTS& m = cand_turn ? m_cand : m_best;
        // get_best_action：温度 0 取访问数最大者（策略输出这里不用）
        const int mv = m.get_best_action(board);
        if (mv < 0) {
            // 兜底：n_sim <= batch_size(16) 时首次 flush 前所有模拟都落在根上，
            // 根的子访问数全 0（Python 同样是这个行为），此时取第一个合法点。
            std::vector<int> legal;
            board.legal_moves(legal);
            if (legal.empty()) return 0;
            const int x = legal[0] % SIZE, y = legal[0] / SIZE;
            const int mover = board.player;
            const bool win = board.place(x, y, mover);
            if (win) return mover;
            if (board.stones >= N_CELLS) return 0;
            m_cand.set_root_move((int16_t)legal[0]);
            m_best.set_root_move((int16_t)legal[0]);
            continue;
        }
        const int x = mv % SIZE, y = mv / SIZE;
        const int mover = board.player;
        const bool win = board.place(x, y, mover);
        if (win) return mover;
        if (board.stones >= N_CELLS) return 0;
        m_cand.set_root_move((int16_t)mv);
        m_best.set_root_move((int16_t)mv);
    }
}

GateResult run_gate(Evaluator* candidate, Evaluator* best, const GateOptions& opt_in,
                    uint64_t seed) {
    GateOptions opt = opt_in;
    if (opt.games % 2 != 0) opt.games += 1;  // 黑白各半

    GateResult r;
    r.games = opt.games;
    const double t0 = now_ms();
    for (int g = 0; g < opt.games; ++g) {
        const bool cand_black = (g % 2 == 0);  // 偶局候选执黑
        const int mover_color = cand_black ? BLACK : WHITE;
        // 配对开局：同一对（g/2）的两局用同一个开局，只有黑白互换
        std::vector<int> op = build_opening(seed + (uint64_t)(g / 2) * 7919ULL, opt.opening_plies);
        const int winner = play_gate_game(candidate, best, opt, cand_black,
                                          seed + (uint64_t)g * 104729ULL,
                                          op.empty() ? nullptr : &op);
        const int cand_color = mover_color;
        if (winner == 0) {
            ++r.draws;
        } else if (winner == cand_color) {
            ++r.wins;
        } else {
            ++r.losses;
        }
        if (cand_black) {
            ++r.games_black;
            if (winner == cand_color) ++r.wins_black;
        } else {
            ++r.games_white;
            if (winner == cand_color) ++r.wins_white;
        }
    }
    r.elapsed_ms = now_ms() - t0;

    r.win_rate = r.games > 0 ? (double)r.wins / (double)r.games : 0.0;
    wilson_interval(r.wins, r.games, &r.wilson_lo, &r.wilson_hi);
    r.pass_score = r.win_rate >= opt.threshold;
    r.pass_ci = (opt.ci_ll <= 0.0) || (r.wilson_lo > opt.ci_ll);
    r.accepted = r.pass_score && r.pass_ci;
    return r;
}

GateResult run_gate_threads(Evaluator* candidate, Evaluator* best, const GateOptions& opt_in,
                            uint64_t seed, int nthreads) {
    GateOptions opt = opt_in;
    if (opt.games % 2 != 0) opt.games += 1;  // 黑白各半

    GateResult r;
    r.games = opt.games;
    // 每局结果先按下标存好，再统一汇总：这样并行完成顺序不影响统计
    std::vector<int> winners((size_t)opt.games, 0);
    // 配对开局先按「对号」预生成：并发的两个 worker 各取自己那一局的副本，
    // 保证同一对的两局拿到逐格相同的开局，不受线程调度影响。
    std::vector<std::vector<int>> openings;
    if (opt.opening_plies > 0) {
        openings.resize((size_t)((opt.games + 1) / 2));
        for (size_t k = 0; k < openings.size(); ++k) {
            openings[k] = build_opening(seed + (uint64_t)k * 7919ULL, opt.opening_plies);
        }
    }
    std::atomic<int> next{0};
    const double t0 = now_ms();
    auto worker = [&]() {
        for (;;) {
            const int g = next.fetch_add(1);
            if (g >= opt.games) return;
            std::vector<int> op;
            if (!openings.empty()) op = openings[(size_t)(g / 2)];
            winners[(size_t)g] = play_gate_game(candidate, best, opt, g % 2 == 0,
                                                seed + (uint64_t)g * 104729ULL,
                                                op.empty() ? nullptr : &op);
        }
    };
    const int nt = std::max(1, std::min(nthreads, opt.games));
    std::vector<std::thread> ths;
    ths.reserve((size_t)(nt - 1));
    for (int i = 1; i < nt; ++i) ths.emplace_back(worker);
    worker();
    for (auto& t : ths) t.join();
    r.elapsed_ms = now_ms() - t0;

    for (int g = 0; g < opt.games; ++g) {
        const bool cand_black = (g % 2 == 0);
        const int cand_color = cand_black ? BLACK : WHITE;
        const int winner = winners[(size_t)g];
        // 逐局日志（CSV，前缀 GATE1 便于 grep 出来离线做颜色拆分）：
        // 局号, 配对号(同一对两局共用开局), 候选是否执黑, 胜者颜色(1黑/2白/0平), 候选视角(1胜/-1负/0平)
        printf("GATE1,%d,%d,%d,%d,%d\n", g, g / 2, cand_black ? 1 : 0, winner,
               winner == 0 ? 0 : (winner == cand_color ? 1 : -1));
        if (winner == 0) {
            ++r.draws;
        } else if (winner == cand_color) {
            ++r.wins;
        } else {
            ++r.losses;
        }
        if (cand_black) {
            ++r.games_black;
            if (winner == cand_color) ++r.wins_black;
        } else {
            ++r.games_white;
            if (winner == cand_color) ++r.wins_white;
        }
    }

    r.win_rate = r.games > 0 ? (double)r.wins / (double)r.games : 0.0;
    wilson_interval(r.wins, r.games, &r.wilson_lo, &r.wilson_hi);
    r.pass_score = r.win_rate >= opt.threshold;
    r.pass_ci = (opt.ci_ll <= 0.0) || (r.wilson_lo > opt.ci_ll);
    r.accepted = r.pass_score && r.pass_ci;
    return r;
}

}  // namespace gk
