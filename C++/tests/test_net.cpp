// test_net.cpp — 网络层自测
//   1) 参数数 1,819,554（6x128x64）
//   2) 加载 Python 转换来的权重 + 与 Python 参考前向逐值比对
//   3) train_batch 的梯度/BN 运行统计更新
//   4) infer_batch 延迟（CPU/CUDA 各档 batch）
#include "net.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace gk;

static int g_fails = 0;
#define EXPECT(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++g_fails;                                       \
        }                                                    \
    } while (0)

static std::vector<float> read_floats(const std::string& path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("打不开 %s\n", path.c_str());
        std::exit(2);
    }
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(float)));
    if (!f) {
        std::printf("读不全 %s\n", path.c_str());
        std::exit(2);
    }
    return v;
}

static double now_ms() {
    using clk = std::chrono::steady_clock;
    static const auto t0 = clk::now();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static int run_all();

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run_all();
    } catch (const std::exception& e) {
        std::printf("异常: %s\n", e.what());
        return 3;
    } catch (...) {
        std::printf("未知异常\n");
        return 3;
    }
}

static int run_all() {
    {
        auto z = torch::zeros({}, torch::TensorOptions().dtype(torch::kInt64));
        std::printf("probe zeros: val=%lld dim=%d dtype=%d\n", (long long)z.item<int64_t>(),
                    (int)z.dim(), (int)z.scalar_type());
    }
    // ---- 1) 权重文件 ----
    Arch arch;
    std::vector<std::pair<std::string, torch::Tensor>> items;
    if (!load_named_tensors_file("_tmp/v4_gen37.bin", arch, items)) {
        std::printf("加载 _tmp/v4_gen37.bin 失败\n");
        return 2;
    }
    std::printf("权重: arch=(%d,%d,%d) 张量数=%zu\n", arch.blocks, arch.channels, arch.value_hidden,
                items.size());
    EXPECT(arch.blocks == 6 && arch.channels == 128 && arch.value_hidden == 64, "arch 不对");

    TorchNet net(arch, /*cuda=*/false, 7);
    int64_t np = 0;
    for (const auto& kv : net.named_parameters()) np += kv.value().numel();
    std::printf("参数数=%lld\n", (long long)np);
    EXPECT(np == 1819554, "参数数应为 1819554，实际 %lld", (long long)np);

    std::string err;
    EXPECT(net.load_named_tensors(items, /*strict=*/true, &err), "载入失败: %s", err.c_str());
    EXPECT(err.empty(), "err 非空: %s", err.c_str());

    // ---- 2) 与 Python 参考前向比对 ----
    const int B = 4;
    auto states = read_floats("_tmp/ref/states.bin", (size_t)B * 3 * N_CELLS);
    auto ref_logits = read_floats("_tmp/ref/logits.bin", (size_t)B * N_CELLS);
    auto ref_values = read_floats("_tmp/ref/values.bin", (size_t)B);

    std::vector<float> pol((size_t)B * N_CELLS), val((size_t)B);
    net.infer_batch(states.data(), B, pol.data(), val.data());

    double max_dp = 0, max_dv = 0;
    for (int b = 0; b < B; ++b) {
        double sum = 0;
        for (int i = 0; i < N_CELLS; ++i)
            sum += std::max(0.0, (double)std::exp(ref_logits[(size_t)b * N_CELLS + i]));
        for (int i = 0; i < N_CELLS; ++i) {
            double r = std::max(0.0, (double)std::exp(ref_logits[(size_t)b * N_CELLS + i])) / sum;
            double d = std::fabs(r - pol[(size_t)b * N_CELLS + i]);
            if (d > max_dp) max_dp = d;
        }
        double dv = std::fabs(ref_values[b] - val[b]);
        if (dv > max_dv) max_dv = dv;
    }
    std::printf("与 Python 参考比对: max|dp|=%.3e  max|dv|=%.3e\n", max_dp, max_dv);
    EXPECT(max_dp < 1e-4, "策略偏差过大: %.3e", max_dp);
    EXPECT(max_dv < 1e-5, "价值偏差过大: %.3e", max_dv);

    // ---- 3) 训练路径 ----
    auto x = torch::zeros({8, 3, SIZE, SIZE});
    {
        std::mt19937 rng(99);
        auto acc = x.accessor<float, 4>();
        for (int b = 0; b < 8; ++b)
            for (int i = 0; i < 20; ++i) {
                int p = (int)(rng() % N_CELLS);
                acc[b][(int)(rng() % 2)][p / SIZE][p % SIZE] = 1.0f;
            }
    }
    auto tp = torch::full({8, N_CELLS}, 1.0 / N_CELLS, torch::TensorOptions().dtype(torch::kFloat32));
    auto tv = torch::zeros({8}, torch::TensorOptions().dtype(torch::kFloat32));
    auto bufs = net.named_buffers();
    // named_buffers() 的键是内部名（"." 换成 "__"）
    auto* rm = bufs.find("stem_bn__running_mean");
    EXPECT(rm != nullptr, "找不到 stem_bn__running_mean");
    auto before = rm->clone();
    auto nbt0 = bufs.find("stem_bn__num_batches_tracked")->item<int64_t>();
    std::printf("BN num_batches_tracked 载入值 = %lld\n", (long long)nbt0);
    std::printf("step: 构造完成 x.dtype=%d tp.dtype=%d\n", (int)x.scalar_type(),
                (int)tp.scalar_type());
    std::fflush(stdout);
    auto losses = net.train_batch(x, tp, tv);
    std::printf("step: backward 完成\n");
    std::fflush(stdout);
    double grad_sum = 0;
    for (const auto& kv : net.named_parameters()) grad_sum += kv.value().grad().abs().sum().item<double>();
    std::printf("train_batch: ce=%.4f mse=%.6f |grad|=%.4f\n", losses.first, losses.second, grad_sum);
    EXPECT(std::isfinite(losses.first) && std::isfinite(losses.second), "loss 非有限");
    EXPECT(grad_sum > 0, "没有梯度");
    auto after = rm->clone();
    EXPECT(!torch::equal(before, after), "BN running_mean 没更新");
    auto nbt = bufs.find("stem_bn__num_batches_tracked");
    std::printf("nbt: dim=%d numel=%lld 载入=%lld 现在=%lld\n", (int)nbt->dim(),
                (long long)nbt->numel(), (long long)nbt0, (long long)nbt->item<int64_t>());
    EXPECT(nbt->item<int64_t>() == nbt0 + 1, "num_batches_tracked 应为 %lld，实际 %lld",
          (long long)(nbt0 + 1), (long long)nbt->item<int64_t>());

    // ---- 4) 延迟 ----
    std::printf("infer_batch 延迟: ");
    for (int bs : {1, 8, 16, 48, 128, 256}) {
        std::vector<float> st((size_t)bs * 3 * N_CELLS, 0.0f), po((size_t)bs * N_CELLS), va((size_t)bs);
        for (int i = 0; i < bs * 3 * N_CELLS; i += 37) st[(size_t)i] = 1.0f;
        net.infer_batch(st.data(), bs, po.data(), va.data());  // warmup
        const int reps = bs <= 16 ? 30 : 10;
        double t0 = now_ms();
        for (int r = 0; r < reps; ++r) net.infer_batch(st.data(), bs, po.data(), va.data());
        std::printf("b%d=%.2fms  ", bs, (now_ms() - t0) / reps);
    }
    std::printf("\n");

    if (torch::cuda::is_available()) {
        // BN 的 running stats 在训练步里被改过，先恢复成文件里的权重再比
        net.load_named_tensors(items, true, nullptr);
        TorchNet gnet(arch, true, 7);
        gnet.load_named_tensors(items, true, nullptr);
        std::printf("CUDA infer_batch: ");
        for (int bs : {1, 16, 48, 128, 256}) {
            std::vector<float> st((size_t)bs * 3 * N_CELLS, 0.0f), po((size_t)bs * N_CELLS),
                va((size_t)bs);
            for (int i = 0; i < bs * 3 * N_CELLS; i += 37) st[(size_t)i] = 1.0f;
            gnet.infer_batch(st.data(), bs, po.data(), va.data());
            const int reps = bs <= 16 ? 30 : 10;
            double t0 = now_ms();
            for (int r = 0; r < reps; ++r) gnet.infer_batch(st.data(), bs, po.data(), va.data());
            std::printf("b%d=%.2fms  ", bs, (now_ms() - t0) / reps);
        }
        std::printf("\n");
        // 与 CPU 结果比对
        std::vector<float> st((size_t)16 * 3 * N_CELLS, 0.0f), po((size_t)16 * N_CELLS), va(16);
        for (int i = 0; i < 16 * 3 * N_CELLS; i += 37) st[(size_t)i] = 1.0f;
        gnet.infer_batch(st.data(), 16, po.data(), va.data());
        std::vector<float> po2((size_t)16 * N_CELLS), va2(16);
        net.infer_batch(st.data(), 16, po2.data(), va2.data());
        double mmax = 0;
        for (size_t i = 0; i < po.size(); ++i) mmax = std::max(mmax, (double)std::fabs(po[i] - po2[i]));
        std::printf("CUDA vs CPU (b16): max|dp|=%.3e\n", mmax);
        // CUDA 默认开 TF32（conv 走 tensor core），差分 ~1e-4 量级是正常的
        EXPECT(mmax < 2e-3, "CUDA 与 CPU 结果不一致: %.3e", mmax);
    } else {
        std::printf("CUDA 不可用\n");
    }

    std::printf(g_fails ? "\n%d 项失败\n" : "\n全部通过\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
