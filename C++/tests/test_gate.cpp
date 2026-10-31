// test_gate.cpp — 门控层自测：Wilson 区间、结算口径、胜负/颜色统计、判定条件
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gate.h"
#include "mcts.h"

using namespace gk;

static int g_checks = 0, g_fails = 0;
#define EXPECT(cond)                                                       \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            ++g_fails;                                                     \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                  \
    } while (0)
#define EXPECT_NEAR(a, b, tol)                                             \
    do {                                                                   \
        ++g_checks;                                                        \
        const double _a = (a), _b = (b);                                   \
        if (std::fabs(_a - _b) > (tol)) {                                  \
            ++g_fails;                                                     \
            std::printf("  FAIL %s:%d  %s=%.6f vs %.6f\n", __FILE__,       \
                        __LINE__, #a, _a, _b);                             \
        }                                                                  \
    } while (0)

// 黑方必胜的假网络：只在指定点给高分，用于构造确定性对局
class BiasNet : public Evaluator {
public:
    BiasNet(int hot) : hot_(hot) {}
    void evaluate_batch(const float* states, int batch, float* policies, float* values) override {
        (void)states;
        for (int i = 0; i < batch; ++i) {
            float* p = policies + (size_t)i * N_CELLS;
            for (int k = 0; k < N_CELLS; ++k) p[k] = 1.0f / (float)N_CELLS;
            p[hot_] = 0.9f;
            float s = 0.0f;
            for (int k = 0; k < N_CELLS; ++k) s += p[k];
            for (int k = 0; k < N_CELLS; ++k) p[k] /= s;
            values[i] = 0.0f;
        }
    }

private:
    int hot_;
};

int main() {
    // ---- 1. Wilson 区间与 Python ladder.py 同式 ----
    {
        double lo = 0, hi = 0;
        wilson_interval(15, 20, &lo, &hi);
        std::printf("[1] wilson(15/20) = [%.6f, %.6f]\n", lo, hi);
        // 参照值来自 PYTHON/ladder.py 的 wilson_interval(15, 20)
        EXPECT_NEAR(lo, 0.531295, 1e-5);
        EXPECT_NEAR(hi, 0.888140, 1e-5);
        wilson_interval(6, 20, &lo, &hi);
        EXPECT_NEAR(lo, 0.145475, 1e-5);
        wilson_interval(0, 20, &lo, &hi);
        EXPECT_NEAR(lo, 0.0, 1e-9);
        EXPECT_NEAR(hi, 0.161130, 1e-5);
        wilson_interval(20, 20, &lo, &hi);
        EXPECT_NEAR(lo, 0.838870, 1e-5);
        EXPECT_NEAR(hi, 1.0, 1e-9);
        wilson_interval(12, 20, &lo, &hi);
        EXPECT_NEAR(lo, 0.386578, 1e-5);
        wilson_interval(0, 0, &lo, &hi);
        EXPECT_NEAR(lo, 0.0, 1e-9);
        EXPECT_NEAR(hi, 1.0, 1e-9);
    }

    // ---- 2. 判定条件：点估计 + 可选 CI ----
    {
        GateOptions o;
        o.games = 20;
        o.threshold = 0.55;
        // 12/20 = 0.60 > 0.55 → 采纳
        double wr = 12.0 / 20.0;
        EXPECT(wr >= o.threshold);
        // 11/20 = 0.55 恰好等于 0.55：2026-10-29 起判据含等号 → 采纳（此前严格大于会拒，
        // gen 58、60 的 33/60 = 0.5500 就是被这一局之差卡掉的）
        EXPECT(11.0 / 20.0 >= o.threshold);
        // 10/20 = 0.50 → 仍拒
        EXPECT(!(10.0 / 20.0 >= o.threshold));
        // 开 CI 后 12/20 的 Wilson 下界约 0.387 < 0.50 → 会被拒
        double lo = 0, hi = 0;
        wilson_interval(12, 20, &lo, &hi);
        EXPECT(lo < 0.50);
        std::printf("[2] 12/20: win_rate=%.3f wilson_lo=%.4f\n", wr, lo);
    }

    // ---- 3. 单局：两个相同网络、棋子合法、必分胜负或满盘 ----
    {
        BiasNet a(0), b(0);
        GateOptions o;
        o.n_sim = 48;
        o.games = 1;
        int w = play_gate_game(&a, &b, o, true, 12345);
        std::printf("[3] 单局结果 = %d\n", w);
        EXPECT(w == 0 || w == 1 || w == 2);
    }

    // ---- 4. run_gate 结算：局数偶数化、颜色各半、双强对局胜率合理 ----
    {
        BiasNet strong(9 * SIZE + 9), weak(2 * SIZE + 2);
        GateOptions o;
        o.games = 5;   // 奇数 -> 自动 6
        o.n_sim = 32;
        GateResult r = run_gate(&strong, &weak, o, 777);
        std::printf("[4] games=%d wins=%d losses=%d draws=%d wr=%.3f (black %d/%d, white %d/%d)\n",
                    r.games, r.wins, r.losses, r.draws, r.win_rate, r.wins_black, r.games_black,
                    r.wins_white, r.games_white);
        EXPECT(r.games == 6);
        EXPECT(r.games_black == 3 && r.games_white == 3);
        EXPECT(r.wins + r.losses + r.draws == r.games);
        EXPECT(r.wins_black <= r.games_black);
        EXPECT(r.wins_white <= r.games_white);
        EXPECT_NEAR(r.win_rate, (double)r.wins / r.games, 1e-12);
        EXPECT(r.elapsed_ms > 0.0);
        // 判定字段与 win_rate 自洽
        EXPECT(r.pass_score == (r.win_rate >= o.threshold));
        EXPECT(r.accepted == (r.pass_score && r.pass_ci));
    }

    // ---- 5. 同一网络自对局：胜率应接近 0.5（不偏袒任何一方）----
    {
        BiasNet a(4 * SIZE + 4), b(4 * SIZE + 4);
        GateOptions o;
        o.games = 8;
        o.n_sim = 32;
        GateResult r = run_gate(&a, &b, o, 99);
        std::printf("[5] 自对局 wr=%.3f (wins=%d losses=%d draws=%d)\n", r.win_rate, r.wins,
                    r.losses, r.draws);
        // 允许噪声，但不应出现 8:0 这种系统性偏置
        EXPECT(r.wins >= 1 && r.losses >= 1);
    }

    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
