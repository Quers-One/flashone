// test_train.cpp — 训练步自测（CPU 小网络，不依赖 CUDA）
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "dataset.h"
#include "net.h"
#include "trainer.h"

using namespace gk;

static int g_fail = 0;
static int g_check = 0;

#define EXPECT(cond)                                                              \
    do {                                                                          \
        ++g_check;                                                                \
        if (!(cond)) {                                                            \
            ++g_fail;                                                             \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
        }                                                                         \
    } while (0)

// 造一条样本：按 seed 摆两三个子（保证 batch 内输入各不相同，否则 BN 方差为 0、
// 卷积层拿不到梯度），策略 one-hot 在 move、价值 label
static Sample make_sample(int seed, int move, float label) {
    Sample s;
    std::memset(&s, 0, sizeof(s));
    const int a = (seed * 7) % N_CELLS;
    int b = (seed * 13 + 5) % N_CELLS;
    if (b == a) b = (b + 1) % N_CELLS;
    s.state[a] = 1;                        // ch0: 黑
    s.state[N_CELLS + b] = 1;              // ch1: 白
    for (int i = 0; i < N_CELLS; ++i) s.state[2 * N_CELLS + i] = 1;  // ch2: 黑走
    s.policy[move] = 1.0f;
    s.value = label;
    return s;
}

int main() {
    const Arch arch{1, 16, 32};
    TorchNet net(arch, /*cuda=*/false, 20251025);

    TrainOptions topt;
    topt.batch_size = 32;
    topt.lr = 1e-2;
    topt.fresh_ratio = 0.7;
    Trainer tr(&net, topt, 7);

    // ---- 1. 记忆任务：同一局面恒为 (9,9)、恒为负 → loss 必须显著下降 ----
    std::vector<Sample> data;
    for (int i = 0; i < 128; ++i) data.push_back(make_sample(i, 9 * SIZE + 9, -1.0f));

    double first = -1.0, last = -1.0;
    double ce_first = -1.0, ce_last = -1.0, mse_last = -1.0;
    double gnorm_max = 0.0;
    int ok_steps = 0;
    for (int i = 0; i < 200; ++i) {
        TrainStepResult r = tr.train_on(data);
        if (!r.ok) continue;
        ++ok_steps;
        EXPECT(std::isfinite(r.loss));
        EXPECT(r.ce > 0.0);
        EXPECT(r.mse >= 0.0);
        EXPECT(std::isfinite(r.grad_norm));
        if (r.grad_norm > gnorm_max) gnorm_max = r.grad_norm;
        if (first < 0) { first = r.loss; ce_first = r.ce; }
        last = r.loss;
        ce_last = r.ce;
        mse_last = r.mse;
    }
    std::printf("[1] steps=%d loss %.4f -> %.4f (ce %.4f -> %.4f, mse_last %.5f) gnorm_max=%.3f\n",
                ok_steps, first, last, ce_first, ce_last, mse_last, gnorm_max);
    // 对照（_tmp/probe_learn.py，同一任务在 Python 版网络上跑）：loss 6.88 -> 4.81、
    // mse 40 步内就到 ~0、CE 卡在 ~4.77 —— 所以 CE 到不了 0 是这套「BN 训练模式 +
    // 常数目标」的真实行为，不是引擎 bug；这里只校验趋势与 value 头是否学到位。
    EXPECT(ok_steps == 200);
    EXPECT(last < first * 0.85);
    EXPECT(mse_last < 0.05);
    EXPECT(ce_last < ce_first);
    EXPECT(gnorm_max > 0.0);

    // ---- 1b. 裁剪确实生效：再裁一次，若前一次范数 > 5 则这次读到 <= 5 的值 ----
    {
        // 上一步刚做完 backward + 裁剪，梯度还在
        double now = torch::nn::utils::clip_grad_norm_(net.parameters(), 5.0);
        std::printf("[1b] 裁剪后梯度范数 = %.4f (裁剪前峰值 %.3f)\n", now, gnorm_max);
        EXPECT(now <= 5.0 + 1e-6);
        EXPECT(gnorm_max < 1e6);
    }

    // ---- 2. EMA 影子权重：与当前参数不同，且可以拷进另一个网络 ----
    std::vector<std::pair<std::string, torch::Tensor>> raw = net.named_tensors();
    TorchNet ema_net(arch, /*cuda=*/false, 20251025);
    net.ema_copy_to(ema_net);
    std::vector<std::pair<std::string, torch::Tensor>> ema = ema_net.named_tensors();
    double diff = 0.0;
    EXPECT(raw.size() == ema.size());
    for (size_t i = 0; i < raw.size() && i < ema.size(); ++i) {
        EXPECT(raw[i].first == ema[i].first);
        if (raw[i].second.sizes() != ema[i].second.sizes()) continue;
        diff += torch::sum(torch::abs(raw[i].second.to(torch::kCPU) - ema[i].second.to(torch::kCPU)))
                    .item<double>();
    }
    std::printf("[2] |raw - ema|_1 = %.4f\n", diff);
    EXPECT(diff > 1e-3);

    // ---- 3. ReplayBuffer 混合抽样：fresh = round(32*0.7) = 22, hist = 10 ----
    ReplayBuffer buf(100000, 60, 20.0);
    std::vector<Sample> gen1, gen2;
    for (int i = 0; i < 64; ++i) gen1.push_back(make_sample(i, 9 * SIZE + 9, 1.0f));
    for (int i = 0; i < 64; ++i) gen2.push_back(make_sample(i + 1000, 9 * SIZE + 10, -1.0f));
    buf.add_generation(0, std::move(gen1));
    buf.add_generation(1, std::move(gen2));

    std::vector<Sample> fresh;
    for (int i = 0; i < 64; ++i) fresh.push_back(make_sample(i + 2000, 9 * SIZE + 10, -1.0f));
    TrainStepResult r3 = tr.train_step(buf, fresh);
    std::printf("[3] ok=%d fresh=%d hist=%d loss=%.4f\n", (int)r3.ok, r3.fresh, r3.hist, r3.loss);
    EXPECT(r3.ok);
    EXPECT(r3.fresh == 22);
    EXPECT(r3.hist == 10);

    // ---- 4. 历史不足（只有一代、且当代样本也不多）→ 退回「全 fresh」，与 Python 一致 ----
    ReplayBuffer tiny(1000, 60, 20.0);
    std::vector<Sample> g;
    for (int i = 0; i < 8; ++i) g.push_back(make_sample(i, 9 * SIZE + 9, 1.0f));
    tiny.add_generation(0, std::move(g));
    std::vector<Sample> few;
    for (int i = 0; i < 4; ++i) few.push_back(make_sample(i, 9 * SIZE + 9, 1.0f));
    TrainStepResult r4 = tr.train_step(tiny, few);
    std::printf("[4] ok=%d fresh=%d hist=%d (期望 ok=1 fresh=32 hist=0)\n", (int)r4.ok, r4.fresh,
                r4.hist);
    EXPECT(r4.ok);
    EXPECT(r4.fresh == 32);
    EXPECT(r4.hist == 0);

    // ---- 5. lr 调整真的写进优化器 ----
    tr.set_lr(1e-4);
    EXPECT(std::fabs(tr.lr() - 1e-4) < 1e-12);
    TrainStepResult r5 = tr.train_on(data);
    EXPECT(r5.ok);

    std::printf("\n%d checks, %d failures\n", g_check, g_fail);
    return g_fail == 0 ? 0 : 1;
}
