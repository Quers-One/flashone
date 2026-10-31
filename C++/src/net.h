// net.h — GomokuNet（libtorch）：6 块 x 128 通道残差塔 + 策略/价值头
//
// 严格对标 PYTHON/network.py：
//   stem:  Conv2d(3,C,3,pad=1,bias=False) -> BN -> ReLU
//   block: h = relu(bn1(conv1(x))); h = bn2(conv2(h)); x = relu(x + h)
//   policy: relu(bn_p(conv_p(x))) -> conv_p_out(1x1,bias) -> flatten(1) -> softmax
//   value:  relu(bn_v(conv_v(x,1x1))) -> global avg pool -> fc1 -> relu -> fc2 -> tanh
// 参数名与 Python state_dict 完全一致（"stem.weight"、"backbone.0.conv1.weight" …），
// 因此 Python 侧 torch.save 的权重可以逐名搬运。
#pragma once

#include <torch/torch.h>

#include <atomic>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "board.h"

namespace gk {

struct Arch {
    int blocks = 6;
    int channels = 128;
    int value_hidden = 64;
    bool operator==(const Arch& o) const {
        return blocks == o.blocks && channels == o.channels && value_hidden == o.value_hidden;
    }
};

// 一组 BatchNorm 的仿射参数 + 运行统计（train 用 batch 统计并更新 running）
struct BNParams {
    torch::Tensor weight, bias, running_mean, running_var, num_batches_tracked;
};

class TorchNet : public torch::nn::Module {
public:
    explicit TorchNet(const Arch& arch, bool cuda = true, uint64_t seed = 1234);

    // x: float32 [B,3,19,19]（值 0/1）。返回 {logits [B,361], value [B]}。
    // train_mode=true 时 BN 用 batch 统计并更新 running stats；否则用 running stats。
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x, bool train_mode);

    // 推理用的便捷封装：输入 CPU float 缓冲（B*3*361），输出 softmax 策略与 value。
    void infer_batch(const float* states, int batch, float* policies, float* values);

    // 训练：给定已增强的 batch（float32 [B,3,19,19]）与目标，返回 {loss_p, loss_v}
    // amp=true 时用 bf16 autocast（卷积走 bf16，BN/softmax/loss 仍 fp32）
    std::pair<double, double> train_batch(const torch::Tensor& x, const torch::Tensor& target_p,
                                          const torch::Tensor& target_v, bool amp = false);

    // 把 state_dict（按名字）取出/写入，用于 checkpoint 与 Python 互转
    std::vector<std::pair<std::string, torch::Tensor>> named_tensors() const;
    bool load_named_tensors(const std::vector<std::pair<std::string, torch::Tensor>>& items,
                            bool strict, std::string* err);

    // EMA 影子权重（只 EMA 参数，BN 的 buffer 直接 copy，对标 train.py:597）
    void ema_init(torch::TensorOptions opt);
    void ema_update(double decay);
    void ema_copy_to(TorchNet& dst) const;

    void to_device(bool cuda);
    bool on_cuda() const { return cuda_; }

    // 权重缓冲的「代」计数：CUDA Graph 里烘的是张量地址，凡是会换掉张量缓冲
    // 的操作（to_device 走 Module::to）都必须让它 +1，推理侧据此重建图。
    // load_named_tensors / EMA 都是 copy_ 原地更新，不 bump。
    uint64_t device_version() const { return device_version_.load(std::memory_order_acquire); }
    const Arch& arch() const { return arch_; }
    torch::Device device() const { return cuda_ ? torch::kCUDA : torch::kCPU; }

    torch::Tensor policy_logits(const torch::Tensor& x, bool train_mode);
    torch::Tensor value_out(const torch::Tensor& x, bool train_mode);

private:
    void register_all();
    torch::Tensor bn(const torch::Tensor& x, BNParams& b, bool train_mode);
    // 推理专用：把 BN 的 scale/shift 折进前一层卷积（数学等价），省掉 13 个 BN kernel。
    // 训练路径完全不用它；参数被改动后 fold_ready_ 置 false，下次推理重新折叠。
    void fold_bn_inference() const;

    Arch arch_;
    bool cuda_ = true;
    std::atomic<uint64_t> device_version_{0};

    torch::Tensor stem_w;
    BNParams stem_bn;
    std::vector<torch::Tensor> conv1_w, conv2_w;
    std::vector<BNParams> bn1, bn2;
    torch::Tensor policy_conv_w, policy_out_w, policy_out_b;
    BNParams policy_bn;
    torch::Tensor value_conv_w, value_fc1_w, value_fc1_b, value_fc2_w, value_fc2_b;
    BNParams value_bn;

    std::vector<torch::Tensor> ema_shadow_;  // 与 named_tensors 的「参数」顺序一致

    // —— 推理侧 BN 折叠缓存（mutable：forward 是 const 语义的读操作）——
    mutable std::atomic<bool> fold_ready_{false};
    mutable std::mutex fold_mu_;
    mutable torch::Tensor f_stem_w_, f_stem_b_, f_pol_w_, f_pol_b_, f_val_w_, f_val_b_;
    mutable std::vector<torch::Tensor> f_conv1_w_, f_conv2_w_, f_b1_, f_b2_;

    // —— 推理侧 fp16 缓存（GK_HALF=1 才有；权重仍是 fp32 主副本，这里只是拷贝）——
    mutable std::atomic<bool> half_ready_{false};
    mutable std::vector<torch::Tensor> h_conv1_w_, h_conv2_w_, h_b1_, h_b2_;
    mutable torch::Tensor h_stem_w_, h_stem_b_, h_pol_w_, h_pol_b_;
    mutable torch::Tensor h_polout_w_, h_polout_b_, h_val_w_, h_val_b_;
    mutable torch::Tensor h_fc1_w_, h_fc1_b_, h_fc2_w_, h_fc2_b_;
};

// ---------------------------------------------------------------------------
// 自定义权重文件（C++ 原生，避免依赖 torch 的 zip 格式）
//   magic "GKAZ1" + arch + 命名张量表
// ---------------------------------------------------------------------------
bool save_named_tensors(const std::string& path, const Arch& arch,
                        const std::vector<std::pair<std::string, torch::Tensor>>& items);
bool load_named_tensors_file(const std::string& path, Arch& arch,
                             std::vector<std::pair<std::string, torch::Tensor>>& items);

}  // namespace gk
