// test_selfplay.cpp — 自对弈循环的行为自测（不依赖 libtorch，用假网络）
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "board.h"
#include "dataset.h"
#include "selfplay.h"

using namespace gk;

static int g_fail = 0;
static int g_check = 0;

#define EXPECT(cond)                                                                  \
    do {                                                                              \
        ++g_check;                                                                    \
        if (!(cond)) {                                                                \
            ++g_fail;                                                                 \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);             \
        }                                                                             \
    } while (0)

#define EXPECT_NEAR(a, b, tol)                                                        \
    do {                                                                              \
        ++g_check;                                                                    \
        const double va = (double)(a), vb = (double)(b);                               \
        if (!(std::fabs(va - vb) <= (tol))) {                                         \
            ++g_fail;                                                                 \
            std::printf("  FAIL %s:%d  |%g - %g| > %g\n", __FILE__, __LINE__, va, vb, \
                        (double)(tol));                                               \
        }                                                                             \
    } while (0)

// 均匀策略、固定价值的假网络。value 恒定 => 可精确控制认输形态。
struct DummyNet : Evaluator {
    float value = 0.0f;
    int calls = 0;
    void evaluate_batch(const float* states, int batch, float* policies, float* values) override {
        (void)states;
        calls += 1;
        const float p = 1.0f / (float)N_CELLS;
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < N_CELLS; ++j) policies[(size_t)i * N_CELLS + j] = p;
            values[i] = value;
        }
    }
};

static void check_samples_consistent(const std::vector<Sample>& samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        const Sample& s = samples[i];
        double sum = 0.0;
        for (int j = 0; j < N_CELLS; ++j) {
            EXPECT(s.policy[j] >= 0.0f);
            sum += s.policy[j];
        }
        if (std::fabs(sum - 1.0) > 1e-4) std::printf("  sample %zu policy sum=%g\n", i, sum);
        EXPECT_NEAR(sum, 1.0, 1e-4);
        // ch0/ch1 是黑白棋子，ch2 是行棋方；第 i 步应恰好有 i 个子
        int black = 0, white = 0, side = 0;
        for (int j = 0; j < N_CELLS; ++j) {
            black += s.state[j] ? 1 : 0;
            white += s.state[N_CELLS + j] ? 1 : 0;
            side += s.state[2 * N_CELLS + j] ? 1 : 0;
        }
        if (black + white != (int)i)
            std::printf("  [mismatch] i=%d black=%d white=%d stones=%d\n", (int)i, black, white,
                        black + white);
        EXPECT(black + white == (int)i);
        const bool black_to_move = ((i % 2) == 0);
        EXPECT(side == (black_to_move ? N_CELLS : 0));
        EXPECT(black == (black_to_move ? (int)i / 2 : (int)i / 2 + 1));
        EXPECT(s.value == 1.0f || s.value == -1.0f || s.value == 0.0f);
    }
}

int main() {
    // ---- 1. 不认输：跑满 max_plies，样本数 == 手数 ----
    {
        DummyNet net;
        net.value = 0.0f;  // 不触发认输
        PlayOptions opt;
        opt.n_sim = 48;
        opt.max_plies = 24;
        opt.resign_threshold = -2.0;  // 关闭认输
        opt.noise_alpha = 0.0;        // 关闭根噪声，让结果可复现
        std::vector<Sample> samples;
        GameStats st;
        const int winner = play_game(&net, opt, samples, &st, 12345);
        std::printf("[1] plies=%d winner=%d truncated=%d samples=%zu evals=%lld\n", st.plies,
                    winner, (int)st.truncated, samples.size(), (long long)st.evaluations);
        EXPECT(st.plies > 0);
        EXPECT(st.plies <= 24);
        EXPECT(samples.size() == (size_t)st.plies);
        if (st.truncated) EXPECT(winner == 0);
        check_samples_consistent(samples);
    }

    // ---- 2. 认输：连续 3 步价值 < -0.95，认输步及之后全部丢弃 ----
    {
        DummyNet net;
        net.value = -1.0f;
        PlayOptions opt;
        opt.n_sim = 48;
        opt.max_plies = 24;
        opt.resign_threshold = -0.95;
        opt.resign_plies = 3;
        opt.noise_alpha = 0.0;
        std::vector<Sample> samples;
        GameStats st;
        const int winner = play_game(&net, opt, samples, &st, 777);
        std::printf("[2] plies=%d winner=%d resigned=%d samples=%zu\n", st.plies, winner,
                    (int)st.resigned, samples.size());
        EXPECT(st.resigned);
        EXPECT(st.plies == 3);
        EXPECT(winner == 2);  // 第 3 步（下标 2）是黑方，认输 => 白胜
        EXPECT(samples.size() == 2);  // 认输步本身被剔除
        check_samples_consistent(samples);
    }

    // ---- 3. 温度 = 0（贪心）仍产出归一化策略 ----
    {
        DummyNet net;
        net.value = 0.0f;
        PlayOptions opt;
        opt.n_sim = 32;
        opt.max_plies = 12;
        opt.resign_threshold = -2.0;
        opt.noise_alpha = 0.0;
        opt.temp_threshold1 = 0;  // 从第一步就是贪心
        opt.temp_threshold2 = 0;
        std::vector<Sample> samples;
        GameStats st;
        play_game(&net, opt, samples, &st, 42);
        EXPECT(samples.size() == 12);
        check_samples_consistent(samples);
        // 注意：策略目标是「访问数归一化」，跟温度无关，所以贪心也不会是 one-hot；
        // 这里只验证最大值确实落在某个合法点上。
        int best = 0;
        for (int j = 1; j < N_CELLS; ++j)
            if (samples[0].policy[j] > samples[0].policy[best]) best = j;
        EXPECT(samples[0].policy[best] > 0.0f);
    }

    std::printf("\n%d checks, %d failures\n", g_check, g_fail);
    return g_fail == 0 ? 0 : 1;
}
