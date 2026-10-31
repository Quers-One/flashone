// probe_eval.cpp — 诊断：评估路径延迟，以及 CPU 负载对单次前向延迟的影响
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "eval_torch.h"
#include "net.h"

using namespace gk;

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "../_tmp/v4_gen37.bin";
    int load_threads = argc > 2 ? atoi(argv[2]) : 0;
    std::vector<std::pair<std::string, torch::Tensor>> items;
    std::string err;
    Arch a2;
    if (!load_named_tensors_file(path, a2, items)) {
        std::printf("load failed\n");
        return 1;
    }
    // 用 checkpoint 自带的 arch 构造网络：否则测不了 6x64 这类更窄的网（形状不匹配）。
    TorchNet net(a2, true, 1);
    std::printf("arch %dx%dx%d\n", a2.blocks, a2.channels, a2.value_hidden);
    if (!net.load_named_tensors(items, true, &err)) {
        std::printf("apply failed: %s\n", err.c_str());
        return 1;
    }
    TorchEvaluator ev(&net);

    std::atomic<bool> stop{false};
    std::vector<std::thread> loaders;
    for (int i = 0; i < load_threads; ++i) {
        loaders.emplace_back([&stop, i]() {
            volatile double acc = 0;
            double x = 1.000001 + 0.001 * i;
            while (!stop.load(std::memory_order_relaxed)) {
                for (int k = 0; k < 20000; ++k) x = std::sqrt(x * 1.0000001 + 0.5);
                acc = x;
            }
        });
    }
    if (load_threads > 0) std::this_thread::sleep_for(std::chrono::milliseconds(300));

    const int sizes[] = {16, 32, 48, 64, 128, 256};
    for (int b : sizes) {
        std::vector<float> states((size_t)b * 3 * N_CELLS, 0.5f);
        std::vector<float> pol((size_t)b * N_CELLS, 0.f), val((size_t)b, 0.f);
        for (int k = 0; k < 3; ++k) ev.evaluate_batch(states.data(), b, pol.data(), val.data());
        const int iters = 30;
        const double t0 = now_ms();
        for (int k = 0; k < iters; ++k) ev.evaluate_batch(states.data(), b, pol.data(), val.data());
        const double dt = (now_ms() - t0) / iters;
        std::printf("load=%d b=%3d  %.2f ms/call  -> %.0f states/s\n", load_threads, b, dt,
                    (double)b * 1000.0 / dt);
    }
    stop.store(true);
    for (auto& t : loaders) t.join();
    return 0;
}
