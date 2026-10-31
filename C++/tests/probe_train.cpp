// probe_train.cpp — 诊断：训练步的分项耗时（增广 / 前向反向 / 裁剪 / Adam / EMA）
//
// 用法（在 C++\ 根目录跑）：build\probe_train.exe _tmp\v4_gen37.bin [batch] [steps]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "net.h"
#include "trainer.h"

using namespace gk;

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// 造 n 条互不相同的样本（batch 内全同会让 BN 方差为 0，梯度退化成 0，测出来的时间不具代表性）
static std::vector<Sample> make_samples(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<Sample> out((size_t)n);
    for (int i = 0; i < n; ++i) {
        Sample& s = out[(size_t)i];
        std::memset(&s, 0, sizeof(s));
        const int b1 = (int)(rng() % (uint32_t)N_CELLS);
        int w1 = (int)(rng() % (uint32_t)N_CELLS);
        if (w1 == b1) w1 = (w1 + 1) % N_CELLS;
        s.state[b1] = 1;
        s.state[N_CELLS + w1] = 1;
        s.state[2 * N_CELLS] = 1;
        s.policy[b1] = 1.0f;
        s.value = 0.0f;
    }
    return out;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "_tmp/v4_gen37.bin";
    const int batch = argc > 2 ? atoi(argv[2]) : 256;
    const int steps = argc > 3 ? atoi(argv[3]) : 20;
    const bool amp = argc > 4 ? atoi(argv[4]) != 0 : false;

    TorchNet net(Arch{6, 128, 64}, true, 1);
    Arch a2;
    std::vector<std::pair<std::string, torch::Tensor>> items;
    if (!load_named_tensors_file(path, a2, items)) {
        std::printf("加载失败: %s\n", path);
        return 1;
    }
    std::string err;
    if (!net.load_named_tensors(items, true, &err)) {
        std::printf("应用权重失败: %s\n", err.c_str());
        return 1;
    }

    TrainOptions to;
    to.batch_size = batch;
    to.lr = 1e-3;
    to.amp = amp;
    Trainer tr(&net, to, 12345);

    const std::vector<Sample> data = make_samples(batch, 7);

    // 前 3 步是预热（cuDNN 算法选择、显存池增长），不计入统计
    for (int i = 0; i < 3; ++i) tr.train_on(data);

    double s_data = 0, s_fwd = 0, s_clip = 0, s_step = 0, s_ema = 0, s_all = 0;
    double loss_first = 0, loss_last = 0;
    for (int i = 0; i < steps; ++i) {
        const double t = now_ms();
        TrainStepResult r = tr.train_on(data);
        s_all += now_ms() - t;
        s_data += r.ms_data;
        s_fwd += r.ms_fwd;
        s_clip += r.ms_clip;
        s_step += r.ms_step;
        s_ema += r.ms_ema;
        if (i == 0) loss_first = r.loss;
        loss_last = r.loss;
    }
    const double n = (double)steps;
    std::printf("batch=%d steps=%d amp=%d（不含 3 步预热）\n", batch, steps, (int)amp);
    std::printf("  总计      %7.2f ms/步\n", s_all / n);
    std::printf("  数据+增广 %7.2f ms  (%4.1f%%)\n", s_data / n, s_data / s_all * 100.0);
    std::printf("  前向+反向 %7.2f ms  (%4.1f%%)\n", s_fwd / n, s_fwd / s_all * 100.0);
    std::printf("  裁剪      %7.2f ms  (%4.1f%%)\n", s_clip / n, s_clip / s_all * 100.0);
    std::printf("  Adam      %7.2f ms  (%4.1f%%)\n", s_step / n, s_step / s_all * 100.0);
    std::printf("  EMA       %7.2f ms  (%4.1f%%)\n", s_ema / n, s_ema / s_all * 100.0);
    std::printf("  loss %.4f → %.4f（%d 步）\n", loss_first, loss_last, steps);
    return 0;
}
