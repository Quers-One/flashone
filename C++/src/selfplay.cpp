// selfplay.cpp — 一盘自对弈
#include "selfplay.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <random>

namespace gk {
namespace {

inline double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

struct Step {
    std::array<uint8_t, 3 * N_CELLS> state;
    std::vector<float> policy;
    int8_t player;
};

}  // namespace

int play_game(Evaluator* net, const PlayOptions& opt, std::vector<Sample>& samples,
              GameStats* stats, uint64_t seed) {
    const double t0 = now_ms();

    MCTSOptions mo;
    mo.n_sim = opt.n_sim;
    mo.c_puct = opt.c_puct;
    mo.max_children = opt.max_children;
    mo.root_max_children = opt.root_max_children;
    mo.use_heuristics = opt.use_heuristics;
    mo.heuristic_weight = opt.heuristic_weight;
    mo.policy_weight = opt.policy_weight;
    mo.vcf = opt.vcf;
    mo.vcf_max_ply = opt.vcf_max_ply;
    mo.vcf_max_nodes = opt.vcf_max_nodes;
    mo.guarantee_rush = opt.guarantee_rush;

    MCTS mcts(net, mo, seed ? seed : 0x9E3779B97F4A7C15ULL);
    mcts.heuristic_weight = opt.heuristic_weight;
    mcts.policy_weight = opt.policy_weight;

    Pos board;
    // 【2026-10-31 加】随机开局：先走 sp_open_plies 手「均匀随机的合法点」，**不记样本**。
    // 动机与边界见 config.h 的 sp_open_plies 注释：无禁手黑方先手优势约 89%，
    // 而每局都从空盘开始 ⇒ 黑方永远用 600 模拟搜出最强起手，白方样本几乎全是劣势局面。
    // 这几手没有搜索访问数（policy 目标是无意义的随机分布），所以绝不能进训练集。
    // 用 seed 派生，保证同 seed 可复现；sp_open_plies==0 时行为与旧版逐位一致。
    if (opt.sp_open_plies > 0) {
        std::mt19937_64 rng(seed ^ 0x5DEECE66DULL);
        std::vector<int> legal;
        for (int i = 0; i < opt.sp_open_plies; ++i) {
            legal.clear();
            board.legal_moves(legal);
            if (legal.empty()) break;
            const int mv = legal[(size_t)(rng() % (uint64_t)legal.size())];
            if (board.place(mv % SIZE, mv / SIZE, board.player)) break;  // 极小概率直接成五
        }
    }
    mcts.set_root(board);

    std::vector<Step> steps;
    std::vector<float> state_f(3 * N_CELLS);
    std::vector<float> policy(N_CELLS);
    std::vector<float> pv(N_CELLS);
    float resign_value = 0.0f;

    int resign_step = -1;
    int low_streak = 0;
    int winner = 0;
    bool truncated = false;
    int64_t evals = 0;

    while (true) {
        if (opt.max_plies > 0 && (int)steps.size() >= opt.max_plies) {
            truncated = true;
            break;
        }

        // 该步局面（先取状态，再搜）
        board.to_tensor(state_f.data());
        Step st;
        for (int i = 0; i < 3 * N_CELLS; ++i)
            st.state[(size_t)i] = (uint8_t)(state_f[(size_t)i] > 0.5f ? 1 : 0);
        st.player = (int8_t)board.player;

        const int ply = (int)steps.size();
        double temp;
        if (ply < opt.temp_threshold1) {
            temp = opt.temp_open;
        } else if (ply < opt.temp_threshold2) {
            temp = opt.temp_mid;
        } else {
            temp = 0.0;
        }

        const int mv = mcts.sample_move_with_policy(board, policy, temp, opt.n_sim,
                                                    opt.noise_alpha, opt.noise_eps);
        if (mv < 0) break;  // 无合法点：满盘

        st.policy = policy;
        steps.push_back(std::move(st));

        // 认输判定：用「当前局面」的 eval 模式价值（行棋方视角）
        if (opt.resign_threshold > -1.0) {
            net->evaluate_batch(state_f.data(), 1, pv.data(), &resign_value);
            ++evals;
            if ((double)resign_value < opt.resign_threshold) {
                ++low_streak;
            } else {
                low_streak = 0;
            }
            if (low_streak >= opt.resign_plies) {
                resign_step = (int)steps.size() - 1;
                break;
            }
        }

        const int x = mv % SIZE, y = mv / SIZE;
        const int mover = board.player;
        const bool win = board.place(x, y, mover);
        if (win) {
            winner = mover;
            break;
        }
        if (board.stones >= N_CELLS) {
            winner = 0;
            break;
        }
        mcts.set_root_move((int16_t)mv);
    }

    // 认输时把认输方判负
    if (resign_step >= 0) {
        const int resigned_player = steps[(size_t)resign_step].player;
        winner = 3 - resigned_player;
    }

    // 结算样本：只保留认输之前的步
    const int usable = (resign_step >= 0) ? resign_step : (int)steps.size();
    samples.reserve(samples.size() + (size_t)usable);
    for (int i = 0; i < usable; ++i) {
        const Step& st = steps[(size_t)i];
        Sample s;
        std::memcpy(s.state, st.state.data(), sizeof(s.state));
        std::memcpy(s.policy, st.policy.data(), sizeof(float) * N_CELLS);
        if (winner == 0) {
            s.value = 0.0f;
        } else {
            s.value = (winner == st.player) ? 1.0f : -1.0f;
        }
        samples.push_back(std::move(s));
    }

    if (stats) {
        stats->plies = (int)steps.size();
        stats->winner = winner;
        stats->resigned = (resign_step >= 0);
        stats->truncated = truncated;
        stats->elapsed_ms = now_ms() - t0;
        stats->evaluations = evals;
    }
    return winner;
}

}  // namespace gk
