// trainer.h — 训练步：抽样 -> 增强 -> 前向/反向 -> 全局梯度裁剪 -> Adam -> EMA
//
// 对标 PYTHON/train.py 的 train_step / ema 更新 / lr 调整：
//   * loss = CE(软目标, +1e-8 归一化) + MSE(value)，无 AMP、无 label smoothing
//   * 每步先 zero_grad，反向之后按**全局 L2 范数**裁剪到 clip_norm，再 optimizer.step()
//   * EMA 每 step 更新一次：shadow = decay*shadow + (1-decay)*param（BN buffer 直接 copy）
//   * 每步重新抽样、重新做随机增广（不是把一批数据反复用）
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "dataset.h"
#include "net.h"

namespace gk {

struct TrainOptions {
    int batch_size = 256;
    int min_steps = 320;
    double fresh_ratio = 0.7;   // 每步 batch 里新样本占比
    double lr = 1e-3;
    double weight_decay = 0.0;
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    double clip_norm = 5.0;
    double ema_decay = 0.999;
    // bf16 autocast：只影响卷积的算力路径，FN 与 loss 仍 fp32
    bool amp = false;
};

struct TrainStepResult {
    double loss = 0.0;
    double ce = 0.0;
    double mse = 0.0;
    double grad_norm = 0.0;   // 裁剪前的全局 L2 范数
    int fresh = 0;
    int hist = 0;
    bool ok = false;
    // 分项耗时（毫秒），用来定位训练步的瓶颈
    double ms_data = 0.0;   // 增广 + 组 batch + 拷到显存
    double ms_fwd = 0.0;    // 前向 + 反向
    double ms_clip = 0.0;   // 全局范数 + 裁剪
    double ms_step = 0.0;   // optimizer.step
    double ms_ema = 0.0;    // EMA 影子权重更新
};

// 融合版 Adam：libtorch 的 torch::optim::Adam 是逐参数实现（96 个张量 × 5 个 kernel，
// 每步几百次 launch + 隐式同步），在 batch=256 的小网络上开销比计算本身还大。
// 这里用 at::_foreach_* 把每步压到 ~8 个 kernel，数值与标准 Adam 完全一致。
class FusedAdam {
public:
    FusedAdam(std::vector<torch::Tensor> params, double lr, double beta1, double beta2,
              double eps, double weight_decay);
    void step(const std::vector<torch::Tensor>& grads);
    void set_lr(double lr) { lr_ = lr; }
    double lr() const { return lr_; }
    int64_t t() const { return t_; }

private:
    std::vector<torch::Tensor> params_;
    std::vector<torch::Tensor> m_;
    std::vector<torch::Tensor> v_;
    double lr_, beta1_, beta2_, eps_, wd_;
    int64_t t_ = 0;
};

class Trainer {
public:
    Trainer(TorchNet* net, const TrainOptions& opt, uint64_t seed = 0);

    // 从「当代新样本 + 历史池」抽样一步并训练。
    // fresh 是当代自对弈样本；历史部分从 buf 里抽（排除最后入池的一代）。
    // 凑不满一个 batch 时返回 ok=false 且不改参数。
    TrainStepResult train_step(ReplayBuffer& buf, const std::vector<Sample>& fresh);

    // 用调用方给定的样本训练一步（测试用，行为与 train_step 完全一致）。
    TrainStepResult train_on(const std::vector<Sample>& samples);

    void set_lr(double lr);
    double lr() const;
    int64_t steps_done() const { return steps_; }
    // 连续退化的判据用：最近 N 步的 loss 均值
    double recent_loss() const;

private:
    TorchNet* net_;
    TrainOptions opt_;
    std::unique_ptr<FusedAdam> adam_;
    std::mt19937_64 rng_;
    int64_t steps_ = 0;

    std::vector<Sample> raw_;
    std::vector<Sample> aug_;
    torch::Tensor x_, tp_, tv_;
    double loss_ema_ = -1.0;
};

}  // namespace gk
