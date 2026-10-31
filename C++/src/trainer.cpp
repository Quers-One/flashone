// trainer.cpp — 训练步实现
#include "trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace gk {

namespace {
inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

FusedAdam::FusedAdam(std::vector<torch::Tensor> params, double lr, double beta1, double beta2,
                     double eps, double weight_decay)
    : params_(std::move(params)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      wd_(weight_decay) {
    torch::NoGradGuard ng;
    for (auto& p : params_) {
        m_.push_back(torch::zeros_like(p));
        v_.push_back(torch::zeros_like(p));
    }
}

void FusedAdam::step(const std::vector<torch::Tensor>& grads) {
    if (params_.empty() || grads.size() != params_.size()) return;
    torch::NoGradGuard ng;
    ++t_;
    // 偏差修正合并进步长：p -= (lr*sqrt(1-b2^t)/(1-b1^t)) * m / (sqrt(v) + eps*sqrt(1-b2^t))
    const double bc1 = 1.0 - std::pow(beta1_, (double)t_);
    const double bc2 = 1.0 - std::pow(beta2_, (double)t_);
    const double lr_t = lr_ * std::sqrt(bc2) / bc1;
    const double eps_t = eps_ * std::sqrt(bc2);

    at::_foreach_mul_(m_, beta1_);
    at::_foreach_add_(m_, grads, 1.0 - beta1_);

    auto gg = at::_foreach_mul(grads, grads);
    at::_foreach_mul_(v_, beta2_);
    at::_foreach_add_(v_, gg, 1.0 - beta2_);

    auto d = at::_foreach_add(at::_foreach_sqrt(v_), eps_t);
    auto upd = at::_foreach_div(at::_foreach_mul(m_, lr_t), d);
    if (wd_ != 0.0) at::_foreach_add_(params_, at::_foreach_mul(params_, wd_ * lr_));
    at::_foreach_sub_(params_, upd);
}

Trainer::Trainer(TorchNet* net, const TrainOptions& opt, uint64_t seed)
    : net_(net), opt_(opt), rng_(seed ? seed : 0x243F6A8885A308D3ULL) {
    adam_ = std::make_unique<FusedAdam>(net_->parameters(), opt.lr, opt.beta1, opt.beta2, opt.eps,
                                        opt.weight_decay);
    net_->ema_init(torch::TensorOptions().dtype(torch::kFloat32).device(net_->device()));
    raw_.reserve(512);
    aug_.reserve(512);
}

void Trainer::set_lr(double lr) {
    opt_.lr = lr;
    adam_->set_lr(lr);
}

double Trainer::lr() const { return opt_.lr; }

double Trainer::recent_loss() const { return loss_ema_; }

TrainStepResult Trainer::train_on(const std::vector<Sample>& samples) {
    TrainStepResult r;
    const int n = std::min<int>((int)samples.size(), opt_.batch_size);
    if (n <= 0) return r;
    const double t_data0 = now_ms();

    raw_.clear();
    raw_.insert(raw_.end(), samples.begin(), samples.begin() + n);

    // 每条样本独立抽一个随机变换（等价于 Python 里逐条 np.rot90/fliplr）
    aug_.resize((size_t)n);
    augment_batch(raw_.data(), aug_.data(), n, rng_);

    std::vector<float> xs((size_t)n * 3 * N_CELLS);
    std::vector<float> ps((size_t)n * N_CELLS);
    std::vector<float> vs((size_t)n);
    for (int i = 0; i < n; ++i) {
        const Sample& s = aug_[(size_t)i];
        for (int k = 0; k < 3 * N_CELLS; ++k) xs[(size_t)i * 3 * N_CELLS + k] = (float)s.state[k];
        std::memcpy(ps.data() + (size_t)i * N_CELLS, s.policy, sizeof(float) * N_CELLS);
        vs[(size_t)i] = s.value;
    }

    auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32);
    auto dev = net_->device();
    x_ = torch::from_blob(xs.data(), {n, 3, SIZE, SIZE}, opts_cpu).clone().to(dev);
    tp_ = torch::from_blob(ps.data(), {n, N_CELLS}, opts_cpu).clone().to(dev);
    tv_ = torch::from_blob(vs.data(), {n}, opts_cpu).clone().to(dev);
    r.ms_data = now_ms() - t_data0;

    // train_batch 内部会 zero_grad + backward
    const double t_fwd0 = now_ms();
    auto ce_mse = net_->train_batch(x_, tp_, tv_, opt_.amp);
    r.ce = ce_mse.first;
    r.mse = ce_mse.second;
    r.loss = r.ce + r.mse;
    r.ms_fwd = now_ms() - t_fwd0;

    // 全局 L2 梯度裁剪：同样用 foreach 一次算完 96 个张量的范数（原来逐张量 .norm() 会同步 96 次）
    const double t_clip0 = now_ms();
    std::vector<torch::Tensor> grads;
    for (const auto& p : net_->parameters()) {
        if (p.grad().defined()) grads.push_back(p.grad());
    }
    if (!grads.empty()) {
        auto norms = at::_foreach_norm(grads, 2.0);
        const double total = std::sqrt(at::stack(norms).pow(2).sum().item<double>());
        r.grad_norm = total;
        if (opt_.clip_norm > 0.0 && total > opt_.clip_norm) {
            at::_foreach_mul_(grads, opt_.clip_norm / total);
        }
    }
    r.ms_clip = now_ms() - t_clip0;

    const double t_step0 = now_ms();
    adam_->step(grads);
    r.ms_step = now_ms() - t_step0;

    const double t_ema0 = now_ms();
    net_->ema_update(opt_.ema_decay);
    r.ms_ema = now_ms() - t_ema0;

    ++steps_;
    r.ok = true;
    loss_ema_ = (loss_ema_ < 0.0) ? r.loss : (0.99 * loss_ema_ + 0.01 * r.loss);
    return r;
}

TrainStepResult Trainer::train_step(ReplayBuffer& buf, const std::vector<Sample>& fresh) {
    TrainStepResult r;
    if (fresh.empty()) return r;

    const int batch = opt_.batch_size;
    int n_fresh = (int)std::llround(batch * opt_.fresh_ratio);
    n_fresh = std::max(0, std::min(batch, n_fresh));
    int n_hist = batch - n_fresh;

    std::vector<Sample> mixed;
    mixed.reserve((size_t)batch);
    buf.sample_mixed(fresh, n_fresh, n_hist, mixed, rng_);

    // 历史样本抽不出来（例如第一代还没有"上一代"可抽）时，与 Python 的 train_step 一致：
    // 退回「只用当代新数据」，凑满整个 batch，而不是把这一代白白跳过。
    if ((int)mixed.size() < batch) {
        n_fresh = batch;
        n_hist = 0;
        mixed.clear();
        std::uniform_int_distribution<size_t> ui(0, fresh.size() - 1);
        for (int i = 0; i < batch; ++i) mixed.push_back(fresh[ui(rng_)]);
    }

    TrainStepResult s = train_on(mixed);
    s.fresh = n_fresh;
    s.hist = n_hist;
    return s;
}

}  // namespace gk
