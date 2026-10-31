// probe_multi.cpp — 诊断：K 个评估线程（各自独立 CUDA stream）能否提高总吞吐
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "eval_torch.h"
#include "net.h"

using namespace gk;

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "../_tmp/v4_gen37.bin";
    const bool use_streams = !(argc > 2 && std::string(argv[2]) == "nostream");
    TorchNet net(Arch{6, 128, 64}, true, 1);
    std::vector<std::pair<std::string, torch::Tensor>> items;
    Arch a2;
    if (!load_named_tensors_file(path, a2, items)) {
        std::printf("load failed\n");
        return 1;
    }
    std::string err;
    if (!net.load_named_tensors(items, true, &err)) return 1;

    // 预热（含 CUDA 初始化）
    {
        TorchEvaluator ev(&net);
        std::vector<float> s((size_t)256 * 3 * N_CELLS, 0.5f), p((size_t)256 * N_CELLS),
            v((size_t)256);
        ev.evaluate_batch(s.data(), 256, p.data(), v.data());
    }

    for (int b : {16, 32}) {
        for (int k : {1, 2, 3, 4}) {
            std::atomic<int64_t> calls{0}, states{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> ths;
            for (int i = 0; i < k; ++i) {
                ths.emplace_back([&, i]() {
                    at::cuda::CUDAStream stream =
                        use_streams ? at::cuda::getStreamFromPool() : at::cuda::getDefaultCUDAStream();
                    at::cuda::CUDAStreamGuard guard(stream);
                    std::vector<float> s((size_t)b * 3 * N_CELLS, 0.3f + 0.01f * i);
                    std::vector<float> p((size_t)b * N_CELLS), v((size_t)b);
                    while (!go.load(std::memory_order_relaxed)) std::this_thread::yield();
                    const double t_end = now_ms() + 3000.0;
                    int64_t n = 0;
                    while (now_ms() < t_end) {
                        net.infer_batch(s.data(), b, p.data(), v.data());
                        ++n;
                    }
                    calls += n;
                    states += n * b;
                });
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            go.store(true);
            for (auto& t : ths) t.join();
            const double secs = 3.0;
            std::printf("streams=%d threads=%d batch=%2d  %.0f states/s  (%.1f calls/s)\n",
                        use_streams ? 1 : 0, k, b, (double)states.load() / secs,
                        (double)calls.load() / secs);
        }
    }
    return 0;
}
