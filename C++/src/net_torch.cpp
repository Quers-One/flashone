// net_torch.cpp — GomokuNet 的 libtorch 实现
#include "net.h"

#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gk {
namespace {

torch::Tensor conv3x3(const torch::Tensor& x, const torch::Tensor& w) {
    return at::conv2d(x, w, c10::optional<at::Tensor>(), at::IntArrayRef({1, 1}),
                      at::IntArrayRef({1, 1}), at::IntArrayRef({1, 1}), 1);
}

torch::Tensor conv1x1(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b) {
    return at::conv2d(x, w, b, at::IntArrayRef({1, 1}), at::IntArrayRef({0, 0}),
                      at::IntArrayRef({1, 1}), 1);
}

void init_conv(torch::Tensor& w) {
    torch::nn::init::kaiming_normal_(w, 0.0, torch::kFanOut, torch::kReLU);
}

// libtorch 的 register_parameter 不接受名字里的 '.'，内部用 "__" 代替；
// 对外的 named_tensors()/load_named_tensors() 仍用 Python 的点分名字。
std::string dot2us(const std::string& n) {
    std::string o;
    o.reserve(n.size() + 8);
    for (char c : n) {
        if (c == '.') {
            o += "__";
        } else {
            o += c;
        }
    }
    return o;
}

std::string us2dot(const std::string& n) {
    std::string o;
    o.reserve(n.size());
    for (size_t i = 0; i < n.size(); ++i) {
        if (n[i] == '_' && i + 1 < n.size() && n[i + 1] == '_') {
            o += '.';
            ++i;
        } else {
            o += n[i];
        }
    }
    return o;
}

void init_linear(torch::Tensor& w, torch::Tensor& b) {
    torch::nn::init::kaiming_normal_(w, 0.0, torch::kFanOut, torch::kReLU);
    torch::nn::init::zeros_(b);
}

// ---------------------------------------------------------------------------
// CUDA Graph 推理（HANDOFF §10A）
//
// 动机：batch 16–32 的前向约 2.4 ms，其中 0.4–0.7 ms 纯粹是发射 50–70 个
// kernel 的 CPU 开销，GPU 在这段时间空转；一手棋要 30–40 次前向。
// 把整张前向图捕获下来，重放用一次 cudaGraphLaunch 完成。
//
// 四条硬约束（不满足就静默算错，所以每条都落到代码里）：
//   1. 图烘的是**地址**：权重必须原地更新。load_named_tensors / EMA 都是
//      copy_（符合）；to_device() 会换张量缓冲 ⇒ 用版本号让下一次前向重建图。
//   2. 图要在**捕获时那条流**上重放，而 NetService 每个评估线程各有一条流
//      ⇒ 一个线程一份 GraphRunner（infer_batch 里 thread_local）。
//   3. 捕获期间不能有分配、CPU 分支、H2D：所以输入先拷进静态显存缓冲，
//      输出直接读图自己写的那块池内存，`.to(kCPU)` 留在图外。
//   4. 桶必须对得齐：**不能**用固定的几个大桶去套任意 batch。实测
//      `gk probe --workers 20` 的平均 batch 只有 18.3，把 17–18 的状态补齐到
//      24 张量算，等于让 GPU 白干 30%，高并发下吞吐从 55 手/s 掉到 37 手/s
//      （GPU 本来就是瓶颈，多算的部分直接扣掉）。所以改成按**实际出现的
//      batch 尺寸**惰性捕获，不做补齐；尺寸超出 [8,32] 或图数量超上限就走 eager。
constexpr int kGraphMinBatch = 8;
constexpr int kGraphMaxBatch = 64;
constexpr int kMaxGraphs = 14;  // 每线程最多几张图（每张约占 batch×128×361×4×几 的显存）

// 【2026-10-28 改】CUDA Graph **默认关闭**（原来默认开启）。
// 实测（20 线程 / 600 sims / eval-batch 24 / eval-threads 3 / run_v2 best.bin）：
//   开：47.73 手/s，11,790 状态/s（均批 18.0）
//   关：44.92 手/s，11,097 状态/s（均批 17.9）
// 只差 5.9% —— 每代省不回 21 秒。但它每代都要重建图：权重版本一变就 release_all()
// 再重新捕获，每张图占一个私有显存池；而 gk loop 在代边界每 4–5 代就 OOM 一次
// （free:0 / total:12.79GB；10-27 那天 8 次 0xC0000409，10-28 15:06 又挂在 gen 44）。
// 用 6% 的吞吐换掉这颗定时炸弹，值。
// 要开回来：GK_GRAPH=1。老开关 GK_NO_GRAPH=1 仍然优先（兼容旧脚本）。
bool graph_disabled_by_env() {
    static const bool disabled = [] {
        if (const char* e = std::getenv("GK_NO_GRAPH"); e && *e && std::strcmp(e, "0") != 0)
            return true;
        const char* g = std::getenv("GK_GRAPH");
        return !(g && *g && std::strcmp(g, "0") != 0);
    }();
    return disabled;
}

bool graph_debug() {
    static const bool on = [] {
        const char* e = std::getenv("GK_GRAPH_DEBUG");
        return e && *e && std::strcmp(e, "0") != 0;
    }();
    return on;
}

struct GraphBucket {
    int size = 0;  // 这张图固定的 batch 尺寸
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    torch::Tensor in;   // [B,3,19,19] 静态输入（图外拷进来）
    torch::Tensor pol;  // [B,361]   图内 softmax 的输出，池内存
    torch::Tensor val;  // [B]
};

// 进程级的存活 runner 表，只为了在退出前统一释放图。见 release_all_at_exit()。
std::mutex g_runners_mu;
std::vector<class GraphRunner*> g_runners;

// 每个线程一份：桶的静态缓冲、图、以及**捕获用的那条流**都跟着线程走。
// 为什么必须自己持有一条专用流：CUDA 不允许在默认（legacy）流上捕获图
// （"CUDA graphs must be captured on a non-default stream"），而 NetService
// 的评估线程若没套 stream guard 用的就是默认流。
//
// 销毁顺序的坑：主线程里那份 runner 是 thread_local，它的析构发生在 atexit
// 处理器之后、静态对象析构之间，那时 PyTorch 的 CUDA context 已经被拆掉，
// cudaGraphExecDestroy 会失败并打一串
// "CUDA-capable device(s) is/are busy or unavailable (function reset)"。
// 退出码仍是 0、输出不受影响，但看起来像真故障，会盖住真错误。所以构造时
// 把 runner 登记进全局表，第一次登记时挂一个 atexit（比库的静态析构先跑，
// 因为 atexit 是 LIFO 而库的注册早于 main），在那里把图全部释放掉。
class GraphRunner {
public:
    GraphRunner(TorchNet* net, int device, uint64_t version)
        : net_(net),
          device_(device),
          version_(version),
          stream_(at::cuda::getStreamFromPool(/*isHighPriority=*/false, device)) {
        {
            std::lock_guard<std::mutex> lk(g_runners_mu);
            g_runners.push_back(this);
        }
        std::call_once(atexit_once_, [] { std::atexit(&GraphRunner::release_all_at_exit); });
    }

    ~GraphRunner() {
        release_all();
        std::lock_guard<std::mutex> lk(g_runners_mu);
        g_runners.erase(std::remove(g_runners.begin(), g_runners.end(), this), g_runners.end());
    }

    // 把所有图（和它们的静态缓冲）放掉。必须在 CUDA context 还活着时调用。
    void release_all() {
        if (graph_debug() && n_ > 0)
            std::fprintf(stderr, "[graph] 释放 %d 张图（runner %p）\n", n_, (void*)this);
        for (int i = 0; i < n_; ++i) {
            if (graph_debug()) std::fprintf(stderr, "[graph]   -> reset b%d\n", buckets_[i].size);
            buckets_[i].graph.reset();
            if (graph_debug()) std::fprintf(stderr, "[graph]   <- reset b%d 完成\n", buckets_[i].size);
            buckets_[i].in = torch::Tensor();
            buckets_[i].pol = torch::Tensor();
            buckets_[i].val = torch::Tensor();
        }
        n_ = 0;
        if (graph_debug()) std::fprintf(stderr, "[graph] 释放完毕\n");
    }

    TorchNet* net() const { return net_; }
    int device() const { return device_; }
    uint64_t version() const { return version_; }

    // 拷输入 → 重放 → 拷回结果。三者都排在同一条流上，所以顺序天然成立。
    // 返回 false 表示这张图不可用（尺寸超出区间 / 图已满 / 捕获失败 / 自检不过），
    // 调用方整条走 eager。
    bool run(const float* states, int batch, float* policies, float* values) {
        if (disabled_) return false;
        if (batch < kGraphMinBatch || batch > kGraphMaxBatch) return false;
        // guard 必须罩住 capture：捕获一旦落在默认流上，PyTorch 会直接抛
        // "CUDA graphs must be captured on a non-default stream."
        at::cuda::CUDAStreamGuard guard(stream_);
        GraphBucket* bk = find(batch);
        if (!bk) {
            if (n_ >= kMaxGraphs) return false;  // 尺寸太分散，剩下的交给 eager
            bk = &buckets_[n_];
            if (!capture(*bk, batch, states)) return false;
            ++n_;
        }

        auto cpu_in = torch::from_blob(const_cast<float*>(states), {batch, 3, SIZE, SIZE},
                                       torch::TensorOptions().dtype(torch::kFloat32));
        bk->in.copy_(cpu_in);
        bk->graph->replay();
        auto pol_cpu = bk->pol.to(torch::kCPU);
        auto val_cpu = bk->val.to(torch::kCPU);
        std::memcpy(policies, pol_cpu.data_ptr<float>(), sizeof(float) * (size_t)batch * N_CELLS);
        std::memcpy(values, val_cpu.data_ptr<float>(), sizeof(float) * (size_t)batch);
        return true;
    }

private:
    GraphBucket* find(int batch) {
        for (int i = 0; i < n_; ++i)
            if (buckets_[i].size == batch) return &buckets_[i];
        return nullptr;
    }

    // probe 参数：捕获后自检用的样本（就是这次触发的输入）
    bool capture(GraphBucket& bk, int B, const float* probe) {
        try {
            torch::NoGradGuard ng;
            const auto opts =
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_);
            bk.size = B;
            bk.in = torch::zeros({B, 3, SIZE, SIZE}, opts);
            {
                auto cpu_in = torch::from_blob(const_cast<float*>(probe), {B, 3, SIZE, SIZE},
                                               torch::TensorOptions().dtype(torch::kFloat32));
                bk.in.copy_(cpu_in);
            }

            // 热身：走几遍前向让 cuDNN 选好算法、把 workspace 分配好。
            // 这段在捕获之前，分配进的是普通缓存池，不是图的私有池。
            for (int i = 0; i < 3; ++i) {
                auto out = net_->forward(bk.in, /*train_mode=*/false);
                (void)torch::softmax(out.first, 1);
            }
            stream_.synchronize();

            bk.graph = std::make_unique<at::cuda::CUDAGraph>();
            // ThreadLocal：捕获期间别的评估/训练线程照常往自己的流里塞 kernel，
            // 不会因为 Global 模式的全局检查而互相报错。
            bk.graph->capture_begin(at::cuda::graph_pool_handle(),
                                    cudaStreamCaptureModeThreadLocal);
            auto out = net_->forward(bk.in, /*train_mode=*/false);
            bk.pol = torch::softmax(out.first, 1);
            bk.val = out.second;
            bk.graph->capture_end();

            if (!verify(bk, B)) {
                disabled_ = true;
                bk.graph.reset();
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true))
                    std::printf("[graph] 捕获后自检不一致，推理回退 eager（本次运行不再尝试）\n");
                return false;
            }
            if (graph_debug()) std::printf("[graph] 捕获 b%d（流 %d）\n", B, (int)stream_.id());
            return true;
        } catch (const std::exception& e) {
            disabled_ = true;
            bk.graph.reset();
            static std::atomic<bool> warned2{false};
            if (!warned2.exchange(true))
                std::printf("[graph] 捕获失败，推理回退 eager：%s\n", e.what());
            return false;
        }
    }

    // 图坏了不会崩，只会悄悄算错（和 HANDOFF §8.2/§8.8 同一类坑），所以捕获完
    // 立刻用静态输入缓冲里那批数据比一次 replay 与 eager。
    bool verify(GraphBucket& bk, int B) {
        bk.graph->replay();
        auto pol_g = bk.pol.to(torch::kCPU);
        auto val_g = bk.val.to(torch::kCPU);

        torch::NoGradGuard ng;
        auto out = net_->forward(bk.in, /*train_mode=*/false);
        auto pol_e = torch::softmax(out.first, 1).to(torch::kCPU);
        auto val_e = out.second.to(torch::kCPU);

        const double dp = (pol_g - pol_e).abs().max().item<double>();
        const double dv = (val_g - val_e).abs().max().item<double>();
        if (graph_debug())
            std::printf("[graph] 自检 b%d: max|dp|=%.3e max|dv|=%.3e\n", B, dp, dv);
        return dp < 1e-4 && dv < 1e-4;
    }

    TorchNet* net_;
    int device_;
    uint64_t version_;
    at::cuda::CUDAStream stream_;  // 捕获与重放都用这一条
    bool disabled_ = false;
    int n_ = 0;
    GraphBucket buckets_[kMaxGraphs];

    static void release_all_at_exit() {
        std::lock_guard<std::mutex> lk(g_runners_mu);
        for (GraphRunner* r : g_runners) r->release_all();
        g_runners.clear();
    }

    static inline std::once_flag atexit_once_{};
};

}  // namespace

TorchNet::TorchNet(const Arch& arch, bool cuda, uint64_t seed)
    : arch_(arch), cuda_(cuda) {
    torch::manual_seed((int64_t)seed);
    // GPU 推理/训练全在 device 上，CPU 侧只剩小张量拷贝：把 intra-op 线程池压到 1，
    // 否则默认会拉起 24 个线程自旋，与 8 个自对弈线程抢核，单次前向从 5ms 抖动到 14ms。
    if (cuda_) {
        at::set_num_threads(1);
        try {
            at::set_num_interop_threads(1);
        } catch (...) {
        }
        // 【2026-10-28 试过 cudnn.benchmark，已回退】benchmark=true 时 cuDNN 会在
        // 「第一次见到某个形状」时同步枚举算法（cudnnFind*）；CUDA Graph 捕获期间
        // 这种同步是非法的，实测直接抛
        //   "[graph] 捕获失败，推理回退 eager：resource deadlock would occur"
        // 并且捕获失败后进程还会卡死。推理侧形状固定、且都走 CUDA Graph（图内 kernel
        // 由捕获时的启发式选定并固化），benchmark 对图路径本来就用不上，故不开启。
    }
    const int64_t C = arch.channels;
    auto opt = torch::TensorOptions().dtype(torch::kFloat32);
    auto iopt = torch::TensorOptions().dtype(torch::kInt64);

    stem_w = torch::empty({C, 3, 3, 3}, opt);
    conv1_w.resize((size_t)arch.blocks);
    conv2_w.resize((size_t)arch.blocks);
    bn1.resize((size_t)arch.blocks);
    bn2.resize((size_t)arch.blocks);
    for (int i = 0; i < arch.blocks; ++i) {
        conv1_w[(size_t)i] = torch::empty({C, C, 3, 3}, opt);
        conv2_w[(size_t)i] = torch::empty({C, C, 3, 3}, opt);
    }
    policy_conv_w = torch::empty({32, C, 3, 3}, opt);
    policy_out_w = torch::empty({1, 32, 1, 1}, opt);
    policy_out_b = torch::empty({1}, opt);
    value_conv_w = torch::empty({32, C, 1, 1}, opt);
    value_fc1_w = torch::empty({arch.value_hidden, 32}, opt);
    value_fc1_b = torch::empty({arch.value_hidden}, opt);
    value_fc2_w = torch::empty({1, arch.value_hidden}, opt);
    value_fc2_b = torch::empty({1}, opt);

    auto make_bn = [&](int64_t c) {
        BNParams b;
        b.weight = torch::ones({c}, opt);
        b.bias = torch::zeros({c}, opt);
        b.running_mean = torch::zeros({c}, opt);
        b.running_var = torch::ones({c}, opt);
        b.num_batches_tracked = torch::zeros({}, iopt);
        return b;
    };
    stem_bn = make_bn(C);
    for (int i = 0; i < arch.blocks; ++i) {
        bn1[(size_t)i] = make_bn(C);
        bn2[(size_t)i] = make_bn(C);
    }
    policy_bn = make_bn(32);
    value_bn = make_bn(32);

    register_all();
    if (std::getenv("GK_DEBUG_NBT")) {
        std::printf("[dbg] 建好: %lld\n", (long long)stem_bn.num_batches_tracked.item<int64_t>());
        if (auto* p = named_buffers().find("stem_bn__num_batches_tracked"))
            std::printf("[dbg] 注册后: %lld\n", (long long)p->item<int64_t>());
    }
    to_device(cuda);
    if (std::getenv("GK_DEBUG_NBT")) {
        std::printf("[dbg] to_device 后成员: %lld\n",
                    (long long)stem_bn.num_batches_tracked.item<int64_t>());
        if (auto* p = named_buffers().find("stem_bn__num_batches_tracked"))
            std::printf("[dbg] to_device 后注册: %lld\n", (long long)p->item<int64_t>());
    }

    // 初始化（对标 network.py:132-151）
    init_conv(stem_w);
    for (int i = 0; i < arch.blocks; ++i) {
        init_conv(conv1_w[(size_t)i]);
        init_conv(conv2_w[(size_t)i]);
    }
    init_conv(policy_conv_w);
    init_conv(value_conv_w);
    torch::nn::init::zeros_(policy_out_b);
    init_linear(value_fc1_w, value_fc1_b);
    // 输出层小方差：N(0, 0.01^2)
    {
        torch::NoGradGuard ng;
        policy_out_w.normal_(0.0, 0.01);
        value_fc2_w.normal_(0.0, 0.01);
        torch::nn::init::zeros_(policy_out_b);
        torch::nn::init::zeros_(value_fc2_b);
    }
}

void TorchNet::register_all() {
    auto P = [this](const std::string& n, torch::Tensor& t) { register_parameter(dot2us(n), t); };
    auto B = [this](const std::string& n, torch::Tensor& t) { register_buffer(dot2us(n), t); };
    P("stem.weight", stem_w);
    P("stem_bn.weight", stem_bn.weight);
    P("stem_bn.bias", stem_bn.bias);
    B("stem_bn.running_mean", stem_bn.running_mean);
    B("stem_bn.running_var", stem_bn.running_var);
    B("stem_bn.num_batches_tracked", stem_bn.num_batches_tracked);
    for (int i = 0; i < arch_.blocks; ++i) {
        const std::string p = "backbone." + std::to_string(i) + ".";
        P(p + "conv1.weight", conv1_w[(size_t)i]);
        P(p + "bn1.weight", bn1[(size_t)i].weight);
        P(p + "bn1.bias", bn1[(size_t)i].bias);
        B(p + "bn1.running_mean", bn1[(size_t)i].running_mean);
        B(p + "bn1.running_var", bn1[(size_t)i].running_var);
        B(p + "bn1.num_batches_tracked", bn1[(size_t)i].num_batches_tracked);
        P(p + "conv2.weight", conv2_w[(size_t)i]);
        P(p + "bn2.weight", bn2[(size_t)i].weight);
        P(p + "bn2.bias", bn2[(size_t)i].bias);
        B(p + "bn2.running_mean", bn2[(size_t)i].running_mean);
        B(p + "bn2.running_var", bn2[(size_t)i].running_var);
        B(p + "bn2.num_batches_tracked", bn2[(size_t)i].num_batches_tracked);
    }
    P("policy_conv.weight", policy_conv_w);
    P("policy_bn.weight", policy_bn.weight);
    P("policy_bn.bias", policy_bn.bias);
    B("policy_bn.running_mean", policy_bn.running_mean);
    B("policy_bn.running_var", policy_bn.running_var);
    B("policy_bn.num_batches_tracked", policy_bn.num_batches_tracked);
    P("policy_out.weight", policy_out_w);
    P("policy_out.bias", policy_out_b);
    P("value_conv.weight", value_conv_w);
    P("value_bn.weight", value_bn.weight);
    P("value_bn.bias", value_bn.bias);
    B("value_bn.running_mean", value_bn.running_mean);
    B("value_bn.running_var", value_bn.running_var);
    B("value_bn.num_batches_tracked", value_bn.num_batches_tracked);
    P("value_fc1.weight", value_fc1_w);
    P("value_fc1.bias", value_fc1_b);
    P("value_fc2.weight", value_fc2_w);
    P("value_fc2.bias", value_fc2_b);
}

void TorchNet::to_device(bool cuda) {
    cuda_ = cuda;
    this->to(cuda ? torch::kCUDA : torch::kCPU);
    // Module::to 会换掉参数/缓冲的张量对象 ⇒ 已经捕获的 CUDA Graph 烘的地址失效。
    // 推理侧的 GraphRunner 记着这个版本号，下次前向发现对不上就重建。
    device_version_.fetch_add(1, std::memory_order_release);
    fold_ready_.store(false, std::memory_order_release);
}

torch::Tensor TorchNet::bn(const torch::Tensor& x, BNParams& b, bool train_mode) {
    // 直接走 F.batch_norm，和 Python 的 nn.BatchNorm2d 是同一个 cuDNN 融合 kernel：
    //   * 训练模式用 batch 统计、按无偏方差更新 running_var、momentum=0.1
    //   * 推理模式用 running 统计
    // 手写 (x-mean)*rsqrt(var+eps)*w+b 的版本数值上等价，但要 6 个额外 kernel，
    // 而且会重复算一遍方差 —— 实测训练步因此慢了近一倍。
    auto out = torch::nn::functional::batch_norm(
        x, b.running_mean, b.running_var,
        torch::nn::functional::BatchNormFuncOptions()
            .weight(b.weight)
            .bias(b.bias)
            .training(train_mode)
            .momentum(0.1)
            .eps(1e-5));
    if (train_mode) {
        torch::NoGradGuard ng;
        b.num_batches_tracked.add_(1);
    }
    return out;
}

namespace {
bool bnfold_inference() {
    static const bool on = [] {
        const char* e = std::getenv("GK_BNFOLD");
        return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}
bool half_inference() {
    static const bool on = [] {
        const char* e = std::getenv("GK_HALF");
        return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}
}  // namespace

void TorchNet::fold_bn_inference() const {
    if (fold_ready_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(fold_mu_);
    if (fold_ready_.load(std::memory_order_relaxed)) return;
    torch::NoGradGuard ng;
    // BN(eval) 是逐通道仿射：y = (x-mean)*w/sqrt(var+eps) + b = x*scale + shift。
    // 折进前一层卷积：W' = W*scale，b' = b - mean*scale（这两个卷积本来都没有 bias）。
    // eps 与 bn() 里的 1e-5 保持一致。
    auto fold = [](const torch::Tensor& w, const BNParams& b, torch::Tensor& ow,
                   torch::Tensor& ob) {
        auto scale = b.weight / torch::sqrt(b.running_var + 1e-5);
        ow = (w * scale.view({-1, 1, 1, 1})).contiguous();
        ob = (b.bias - b.running_mean * scale).contiguous();
    };
    const size_t nb = (size_t)arch_.blocks;
    f_conv1_w_.resize(nb);
    f_conv2_w_.resize(nb);
    f_b1_.resize(nb);
    f_b2_.resize(nb);
    fold(stem_w, stem_bn, f_stem_w_, f_stem_b_);
    for (size_t i = 0; i < nb; ++i) {
        fold(conv1_w[i], bn1[i], f_conv1_w_[i], f_b1_[i]);
        fold(conv2_w[i], bn2[i], f_conv2_w_[i], f_b2_[i]);
    }
    fold(policy_conv_w, policy_bn, f_pol_w_, f_pol_b_);
    fold(value_conv_w, value_bn, f_val_w_, f_val_b_);

    if (half_inference() && cuda_) {
        // fp16 副本：只存推理用的折叠后权重。fp16 的 conv 在 cuDNN 里能走
        // tensor core（b≥32 时吞吐约 1.6–1.9×），b≤16 反而略慢，所以靠
        // 调用方用 GK_BATCH_MCTS 把批做大才有意义。
        auto h = [](const torch::Tensor& t) { return t.to(at::kHalf).contiguous(); };
        h_conv1_w_.resize(nb);
        h_conv2_w_.resize(nb);
        h_b1_.resize(nb);
        h_b2_.resize(nb);
        h_stem_w_ = h(f_stem_w_);
        h_stem_b_ = h(f_stem_b_);
        for (size_t i = 0; i < nb; ++i) {
            h_conv1_w_[i] = h(f_conv1_w_[i]);
            h_conv2_w_[i] = h(f_conv2_w_[i]);
            h_b1_[i] = h(f_b1_[i]);
            h_b2_[i] = h(f_b2_[i]);
        }
        h_pol_w_ = h(f_pol_w_);
        h_pol_b_ = h(f_pol_b_);
        h_polout_w_ = h(policy_out_w);
        h_polout_b_ = h(policy_out_b);
        h_val_w_ = h(f_val_w_);
        h_val_b_ = h(f_val_b_);
        h_fc1_w_ = h(value_fc1_w);
        h_fc1_b_ = h(value_fc1_b);
        h_fc2_w_ = h(value_fc2_w);
        h_fc2_b_ = h(value_fc2_b);
        half_ready_.store(true, std::memory_order_release);
    }
    fold_ready_.store(true, std::memory_order_release);
}

std::pair<torch::Tensor, torch::Tensor> TorchNet::forward(const torch::Tensor& x,
                                                          bool train_mode) {
    if (!train_mode && bnfold_inference()) fold_bn_inference();
    if (!train_mode && bnfold_inference() && half_ready_.load(std::memory_order_acquire)) {
        auto bias = [](const torch::Tensor& b) { return b.view({1, -1, 1, 1}); };
        auto xh = x.to(at::kHalf);
        auto s = torch::relu(conv3x3(xh, h_stem_w_) + bias(h_stem_b_));
        for (int i = 0; i < arch_.blocks; ++i) {
            auto h = torch::relu(conv3x3(s, h_conv1_w_[(size_t)i]) + bias(h_b1_[(size_t)i]));
            h = conv3x3(h, h_conv2_w_[(size_t)i]) + bias(h_b2_[(size_t)i]);
            s = torch::relu(s + h);
        }
        auto p = torch::relu(conv3x3(s, h_pol_w_) + bias(h_pol_b_));
        p = conv1x1(p, h_polout_w_, h_polout_b_).flatten(1).to(at::kFloat);  // [B,361]

        auto v = torch::relu(conv1x1(s, h_val_w_, torch::Tensor()) + bias(h_val_b_));
        v = v.mean({2, 3});  // [B,32]
        v = torch::relu(torch::matmul(v, h_fc1_w_.t()) + h_fc1_b_);
        v = torch::tanh(torch::matmul(v, h_fc2_w_.t()) + h_fc2_b_).squeeze(-1).to(at::kFloat);
        return {p, v};
    }
    if (!train_mode && bnfold_inference() && fold_ready_.load(std::memory_order_acquire)) {
        auto bias = [](const torch::Tensor& b) { return b.view({1, -1, 1, 1}); };
        auto s = torch::relu(conv3x3(x, f_stem_w_) + bias(f_stem_b_));
        for (int i = 0; i < arch_.blocks; ++i) {
            auto h = torch::relu(conv3x3(s, f_conv1_w_[(size_t)i]) + bias(f_b1_[(size_t)i]));
            h = conv3x3(h, f_conv2_w_[(size_t)i]) + bias(f_b2_[(size_t)i]);
            s = torch::relu(s + h);
        }
        auto p = torch::relu(conv3x3(s, f_pol_w_) + bias(f_pol_b_));
        p = conv1x1(p, policy_out_w, policy_out_b).flatten(1);  // [B,361]

        auto v = torch::relu(conv1x1(s, f_val_w_, torch::Tensor()) + bias(f_val_b_));
        v = v.mean({2, 3});  // [B,32]
        v = torch::relu(torch::matmul(v, value_fc1_w.t()) + value_fc1_b);
        v = torch::tanh(torch::matmul(v, value_fc2_w.t()) + value_fc2_b).squeeze(-1);
        return {p, v};
    }
    auto s = torch::relu(bn(conv3x3(x, stem_w), stem_bn, train_mode));
    for (int i = 0; i < arch_.blocks; ++i) {
        auto h = torch::relu(bn(conv3x3(s, conv1_w[(size_t)i]), bn1[(size_t)i], train_mode));
        h = bn(conv3x3(h, conv2_w[(size_t)i]), bn2[(size_t)i], train_mode);
        s = torch::relu(s + h);
    }
    auto p = torch::relu(bn(conv3x3(s, policy_conv_w), policy_bn, train_mode));
    p = conv1x1(p, policy_out_w, policy_out_b).flatten(1);  // [B,361]

    auto v = torch::relu(bn(conv1x1(s, value_conv_w, torch::Tensor()), value_bn, train_mode));
    v = v.mean({2, 3});  // [B,32] 空间平均
    v = torch::relu(torch::matmul(v, value_fc1_w.t()) + value_fc1_b);
    v = torch::tanh(torch::matmul(v, value_fc2_w.t()) + value_fc2_b).squeeze(-1);
    return {p, v};
}

torch::Tensor TorchNet::policy_logits(const torch::Tensor& x, bool train_mode) {
    return forward(x, train_mode).first;
}

torch::Tensor TorchNet::value_out(const torch::Tensor& x, bool train_mode) {
    return forward(x, train_mode).second;
}

void TorchNet::infer_batch(const float* states, int batch, float* policies, float* values) {
    if (batch <= 0) return;
    torch::NoGradGuard ng;
    if (device().is_cuda() && !graph_disabled_by_env()) {
        // 每条线程一份图：NetService 的每个评估线程各有自己的 CUDA stream，
        // 而 replay() 固定在捕获时那条流上启动；静态缓冲也必须线程私有。
        static thread_local std::unique_ptr<GraphRunner> runner;
        // device().index() 对 "cuda" 返回 -1（未指定序号），这里用当前设备。
        const int dev = c10::cuda::current_device();
        const uint64_t ver = device_version();
        if (!runner || runner->net() != this || runner->device() != dev ||
            runner->version() != ver) {
            runner = std::make_unique<GraphRunner>(this, dev, ver);
            if (graph_debug())
                std::printf("[graph] 线程新建 runner（net=%p dev=%d ver=%llu）\n", (void*)this, dev,
                            (unsigned long long)ver);
        }
        if (runner->run(states, batch, policies, values)) return;
    }
    auto cpu_in = torch::from_blob(const_cast<float*>(states), {batch, 3, SIZE, SIZE},
                                   torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor x;
    if (device().is_cuda()) {
        // 复用一个持久化的显存输入缓冲（每个线程各一份），避免每次前向都
        // 走一遍 caching allocator 的申请/释放 + 多做一次 .to(device()) 拷贝。
        // 实测这条路径能省掉每次前向 1ms 量级的固定开销。
        static thread_local torch::Tensor dev_in;
        if (!dev_in.defined() || dev_in.size(0) < batch ||
            dev_in.device() != device()) {
            dev_in = torch::empty({std::max(batch, 256), 3, SIZE, SIZE},
                                  torch::TensorOptions().dtype(torch::kFloat32).device(device()));
        }
        x = dev_in.narrow(0, 0, batch);
        x.copy_(cpu_in);
    } else {
        x = cpu_in;
    }
    auto out = forward(x, /*train_mode=*/false);
    auto pol = torch::softmax(out.first, 1).to(torch::kCPU);
    auto val = out.second.to(torch::kCPU);
    std::memcpy(policies, pol.data_ptr<float>(), sizeof(float) * (size_t)batch * N_CELLS);
    std::memcpy(values, val.data_ptr<float>(), sizeof(float) * (size_t)batch);
}

std::pair<double, double> TorchNet::train_batch(const torch::Tensor& x,
                                                const torch::Tensor& target_p,
                                                const torch::Tensor& target_v, bool amp) {
    this->zero_grad();
    // 训练会原地更新权重（优化器 step 在调用方），折叠缓存必须作废：
    // 下次推理再按新的 running stats / 权重重新折叠。
    fold_ready_.store(false, std::memory_order_release);
    // bf16 autocast：卷积/矩阵乘走 bf16，BN、softmax、loss 仍走 fp32（与 torch.amp 同策略）。
    // 实测 batch=256 前向+反向从 ~82 ms 降到 ~30 ms；权重和优化器状态保持 fp32。
    const bool prev_amp = at::autocast::is_autocast_enabled(at::kCUDA);    if (amp) {
        at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
        // 关掉权重转换缓存：优化器用 _foreach_* 就地更新参数时不一定会让 autcast 的
        // 缓存失效，缓存命中就会拿「上一步的 bf16 权重」做前向 —— 训练会缓慢劣化。
        at::autocast::set_autocast_cache_enabled(false);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
    }
    auto out = forward(x, /*train_mode=*/true);
    auto logits = out.first, value = out.second;
    auto tgt = target_p + 1e-8;
    tgt = tgt / tgt.sum(1, /*keepdim=*/true);
    auto ce = -(tgt * torch::log_softmax(logits, 1)).sum(1).mean();
    auto diff = value - target_v;
    auto mse = (diff * diff).mean();
    auto loss = ce + mse;
    loss.backward();
    if (amp) at::autocast::set_autocast_enabled(at::kCUDA, prev_amp);
    return {ce.item<double>(), mse.item<double>()};
}

std::vector<std::pair<std::string, torch::Tensor>> TorchNet::named_tensors() const {
    std::vector<std::pair<std::string, torch::Tensor>> out;
    out.reserve(named_parameters().size() + named_buffers().size());
    for (const auto& kv : named_parameters()) out.emplace_back(us2dot(kv.key()), kv.value().detach());
    for (const auto& kv : named_buffers()) out.emplace_back(us2dot(kv.key()), kv.value().detach());
    return out;
}

bool TorchNet::load_named_tensors(const std::vector<std::pair<std::string, torch::Tensor>>& items,
                                  bool strict, std::string* err) {
    std::unordered_map<std::string, torch::Tensor> m;
    for (const auto& kv : items) m.emplace(kv.first, kv.second);
    auto assign = [&](const std::string& name, torch::Tensor& dst, bool is_param) -> bool {
        (void)is_param;
        auto it = m.find(name);
        if (it == m.end()) {
            if (strict) {
                if (err) *err = "缺少张量: " + name;
                return false;
            }
            return true;
        }
        auto src = it->second;
        if (src.sizes() != dst.sizes()) {
            if (err) *err = "形状不匹配: " + name;
            return false;
        }
        torch::NoGradGuard ng;
        dst.copy_(src.to(dst.device()).to(dst.scalar_type()));
        return true;
    };
    bool ok = true;
    ok = ok && assign("stem.weight", stem_w, true);
    ok = ok && assign("stem_bn.weight", stem_bn.weight, true);
    ok = ok && assign("stem_bn.bias", stem_bn.bias, true);
    ok = ok && assign("stem_bn.running_mean", stem_bn.running_mean, false);
    ok = ok && assign("stem_bn.running_var", stem_bn.running_var, false);
    ok = ok && assign("stem_bn.num_batches_tracked", stem_bn.num_batches_tracked, false);
    for (int i = 0; i < arch_.blocks && ok; ++i) {
        const std::string p = "backbone." + std::to_string(i) + ".";
        ok = ok && assign(p + "conv1.weight", conv1_w[(size_t)i], true);
        ok = ok && assign(p + "bn1.weight", bn1[(size_t)i].weight, true);
        ok = ok && assign(p + "bn1.bias", bn1[(size_t)i].bias, true);
        ok = ok && assign(p + "bn1.running_mean", bn1[(size_t)i].running_mean, false);
        ok = ok && assign(p + "bn1.running_var", bn1[(size_t)i].running_var, false);
        ok = ok && assign(p + "bn1.num_batches_tracked", bn1[(size_t)i].num_batches_tracked, false);
        ok = ok && assign(p + "conv2.weight", conv2_w[(size_t)i], true);
        ok = ok && assign(p + "bn2.weight", bn2[(size_t)i].weight, true);
        ok = ok && assign(p + "bn2.bias", bn2[(size_t)i].bias, true);
        ok = ok && assign(p + "bn2.running_mean", bn2[(size_t)i].running_mean, false);
        ok = ok && assign(p + "bn2.running_var", bn2[(size_t)i].running_var, false);
        ok = ok && assign(p + "bn2.num_batches_tracked", bn2[(size_t)i].num_batches_tracked, false);
    }
    ok = ok && assign("policy_conv.weight", policy_conv_w, true);
    ok = ok && assign("policy_bn.weight", policy_bn.weight, true);
    ok = ok && assign("policy_bn.bias", policy_bn.bias, true);
    ok = ok && assign("policy_bn.running_mean", policy_bn.running_mean, false);
    ok = ok && assign("policy_bn.running_var", policy_bn.running_var, false);
    ok = ok && assign("policy_bn.num_batches_tracked", policy_bn.num_batches_tracked, false);
    ok = ok && assign("policy_out.weight", policy_out_w, true);
    ok = ok && assign("policy_out.bias", policy_out_b, true);
    ok = ok && assign("value_conv.weight", value_conv_w, true);
    ok = ok && assign("value_bn.weight", value_bn.weight, true);
    ok = ok && assign("value_bn.bias", value_bn.bias, true);
    ok = ok && assign("value_bn.running_mean", value_bn.running_mean, false);
    ok = ok && assign("value_bn.running_var", value_bn.running_var, false);
    ok = ok && assign("value_bn.num_batches_tracked", value_bn.num_batches_tracked, false);
    ok = ok && assign("value_fc1.weight", value_fc1_w, true);
    ok = ok && assign("value_fc1.bias", value_fc1_b, true);
    ok = ok && assign("value_fc2.weight", value_fc2_w, true);
    ok = ok && assign("value_fc2.bias", value_fc2_b, true);
    fold_ready_.store(false, std::memory_order_release);
    return ok;
}

void TorchNet::ema_init(torch::TensorOptions opt) {
    (void)opt;
    ema_shadow_.clear();
    for (const auto& kv : named_parameters())
        ema_shadow_.push_back(kv.value().detach().clone());
}

void TorchNet::ema_update(double decay) {
    if (ema_shadow_.empty()) return;
    torch::NoGradGuard ng;
    // 用 foreach 把「每个参数两次小 kernel」合并成两个大 kernel：
    // 96 个张量 × 2 次 launch 在 12GB 卡上约 3-5 ms/步，值得。
    std::vector<torch::Tensor> ps;
    ps.reserve(ema_shadow_.size());
    for (const auto& kv : named_parameters()) ps.push_back(kv.value().detach());
    at::_foreach_mul_(ema_shadow_, decay);
    at::_foreach_add_(ema_shadow_, ps, 1.0 - decay);
}

void TorchNet::ema_copy_to(TorchNet& dst) const {
    torch::NoGradGuard ng;
    size_t i = 0;
    for (const auto& kv : named_parameters())
        kv.value().copy_(ema_shadow_[i++].to(kv.value().device()));
    // BN 的 buffer 直接拷贝（不做 EMA）
    for (const auto& kv : named_buffers()) {
        auto* p = dst.named_buffers().find(kv.key());
        if (p) p->copy_(kv.value());
    }
    fold_ready_.store(false, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// 自定义权重文件
// ---------------------------------------------------------------------------
namespace {
constexpr char kMag[] = {'G', 'K', 'A', 'Z', '1'};

template <typename T>
void wr(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
bool rd(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return f.good();
}
}  // namespace

bool save_named_tensors(const std::string& path, const Arch& arch,
                        const std::vector<std::pair<std::string, torch::Tensor>>& items) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(kMag, sizeof(kMag));
    int32_t a[3] = {arch.blocks, arch.channels, arch.value_hidden};
    f.write(reinterpret_cast<const char*>(a), sizeof(a));
    auto cnt = (int32_t)items.size();
    wr(f, cnt);
    for (const auto& kv : items) {
        auto t = kv.second.detach().to(torch::kCPU).contiguous();
        auto dt = t.scalar_type();
        uint8_t code = (dt == torch::kFloat32) ? 0 : (dt == torch::kInt64 ? 1 : 2);
        if (code == 2) return false;  // 只支持 f32/i64
        auto name = kv.first;
        auto nlen = (int32_t)name.size();
        wr(f, nlen);
        f.write(name.data(), nlen);
        wr(f, code);
        auto nd = (int32_t)t.dim();
        wr(f, nd);
        for (int i = 0; i < nd; ++i) {
            auto d = (int64_t)t.size(i);
            wr(f, d);
        }
        const size_t bytes = (size_t)t.numel() * (code == 0 ? 4 : 8);
        if (bytes) f.write(static_cast<const char*>(t.data_ptr()), (std::streamsize)bytes);
    }
    return f.good();
}

bool load_named_tensors_file(const std::string& path, Arch& arch,
                             std::vector<std::pair<std::string, torch::Tensor>>& items) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char mag[sizeof(kMag)];
    f.read(mag, sizeof(mag));
    if (!f || std::memcmp(mag, kMag, sizeof(kMag)) != 0) return false;
    int32_t a[3];
    f.read(reinterpret_cast<char*>(a), sizeof(a));
    arch.blocks = a[0];
    arch.channels = a[1];
    arch.value_hidden = a[2];
    int32_t cnt = 0;
    if (!rd(f, cnt)) return false;
    items.clear();
    for (int32_t k = 0; k < cnt; ++k) {
        int32_t nlen = 0;
        if (!rd(f, nlen)) return false;
        std::string name((size_t)nlen, '\0');
        f.read(name.data(), nlen);
        uint8_t code = 0;
        if (!rd(f, code)) return false;
        int32_t nd = 0;
        if (!rd(f, nd)) return false;
        std::vector<int64_t> dims((size_t)nd);
        for (int32_t i = 0; i < nd; ++i) {
            int64_t d = 0;
            if (!rd(f, d)) return false;
            dims[(size_t)i] = d;
        }
        auto opts = torch::TensorOptions().dtype(code == 0 ? torch::kFloat32 : torch::kInt64);
        auto t = torch::empty(dims, opts);
        const size_t bytes = (size_t)t.numel() * (code == 0 ? 4 : 8);
        if (bytes) f.read(static_cast<char*>(t.data_ptr()), (std::streamsize)bytes);
        if (!f) return false;
        items.emplace_back(std::move(name), std::move(t));
    }
    return true;
}

}  // namespace gk
