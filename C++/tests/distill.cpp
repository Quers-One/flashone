// distill.cpp — R2：把宽网络（老师）蒸馏成窄网络（学生）
//
// 为什么要有它：现有 `gk loop` 永远是「自己下棋 -> 自己学」，没有「照着老师学」这条路。
// 学生变窄之后如果还从随机权重开始，就要重新爬几十代；这里直接把老师的 MCTS 访问分布
// 当软目标，做有监督的模仿，一代之内把学生的起点拉到老师附近。
//
// 用法（在 C++\ 根目录跑）：
//   build\distill.exe model\run_v2\best.bin _tmp\r2\samples.bin model\r2\s64.bin ^
//                     --channels 64 --steps 3000 --batch 256 --lr 0.0003 --holdout 2000
//
// 输出：
//   [初始] / [训练] / [最终] 三段 —— top1 一致率（学生与老师在同一局面上首选同一点的比例）、
//   蒸馏交叉熵 CE（-sum p_teacher * log p_student）、value 的 MSE。
//   这三个数就是「学生像不像老师」的唯一判据；棋力还得靠 gk gate 另外量。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "dataset.h"
#include "net.h"
#include "trainer.h"

using namespace gk;

static void to_float_states(const std::vector<Sample>& s, std::vector<float>& out) {
    out.resize(s.size() * 3 * N_CELLS);
    for (size_t i = 0; i < s.size(); ++i) {
        for (int k = 0; k < 3 * N_CELLS; ++k) {
            out[i * 3 * N_CELLS + (size_t)k] = (float)s[i].state[k];
        }
    }
}

struct Eval {
    double agree = 0, ce = 0, mse = 0;
};

// 学生 vs 老师：policy 用老师的策略做软目标算 CE，value 直接 MSE，另报 top-1 一致率。
static Eval evaluate(TorchNet& teacher, TorchNet& student, const std::vector<float>& xs, int n) {
    std::vector<float> tp((size_t)n * N_CELLS), tv((size_t)n);
    std::vector<float> sp((size_t)n * N_CELLS), sv((size_t)n);
    teacher.eval();
    student.eval();
    {
        torch::NoGradGuard g;
        teacher.infer_batch(xs.data(), n, tp.data(), tv.data());
        student.infer_batch(xs.data(), n, sp.data(), sv.data());
    }
    long agree = 0;
    double ce = 0, mse = 0;
    for (int i = 0; i < n; ++i) {
        const float* a = &tp[(size_t)i * N_CELLS];
        const float* b = &sp[(size_t)i * N_CELLS];
        int ia = 0, ib = 0;
        for (int k = 1; k < N_CELLS; ++k) {
            if (a[k] > a[ia]) ia = k;
            if (b[k] > b[ib]) ib = k;
        }
        if (ia == ib) ++agree;
        for (int k = 0; k < N_CELLS; ++k) {
            if (a[k] > 0) ce -= (double)a[k] * std::log((double)b[k] + 1e-12);
        }
        const double d = (double)tv[(size_t)i] - (double)sv[(size_t)i];
        mse += d * d;
    }
    Eval e;
    e.agree = (double)agree / (double)n;
    e.ce = ce / n;
    e.mse = mse / n;
    return e;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("用法: distill.exe <teacher.bin> <samples.bin> <out.bin> "
                    "[--channels 64] [--steps 3000] [--batch 256] [--lr 0.0003] "
                    "[--holdout 2000] [--drop-samples N]\n");
        return 2;
    }
    const std::string tpath = argv[1];
    const std::string spath = argv[2];
    const std::string opath = argv[3];
    int channels = 64, steps = 3000, batch_n = 256, holdout = 2000;
    double lr = 3e-4;
    std::string value_src = "teacher";
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        const std::string v = argv[i + 1];
        if (k == "--channels") channels = std::atoi(v.c_str());
        else if (k == "--steps") steps = std::atoi(v.c_str());
        else if (k == "--batch") batch_n = std::atoi(v.c_str());
        else if (k == "--holdout") holdout = std::atoi(v.c_str());
        else if (k == "--lr") lr = std::atof(v.c_str());
        else if (k == "--value-src") value_src = v;
        else std::printf("[警告] 未知参数 %s（忽略）\n", k.c_str());
    }

    // ---- 老师 ----
    Arch ta;
    std::vector<std::pair<std::string, torch::Tensor>> items;
    if (!load_named_tensors_file(tpath, ta, items)) {
        std::printf("[错误] 读不到老师权重: %s\n", tpath.c_str());
        return 1;
    }
    TorchNet teacher(ta, true, 1234);
    std::string err;
    if (!teacher.load_named_tensors(items, true, &err)) {
        std::printf("[错误] 老师权重载入失败: %s\n", err.c_str());
        return 1;
    }
    teacher.to_device(true);
    std::printf("[老师] %dx%dx%d  (%s)\n", ta.blocks, ta.channels, ta.value_hidden, tpath.c_str());

    // ---- 学生 ----
    const Arch sa{ta.blocks, channels, ta.value_hidden};
    TorchNet student(sa, true, 4321);
    student.to_device(true);
    {
        size_t np = 0;
        for (const auto& t : student.named_tensors()) np += (size_t)t.second.numel();
        std::printf("[学生] %dx%dx%d  参数量 %zu\n", sa.blocks, sa.channels, sa.value_hidden, np);
    }

    // ---- 样本 ----
    std::vector<Sample> all;
    if (!load_samples(spath, all)) {
        std::printf("[错误] 读不到样本: %s\n", spath.c_str());
        return 1;
    }
    if ((int)all.size() < holdout + batch_n) {
        std::printf("[错误] 样本太少 (%zu)，至少要 holdout+batch = %d\n", all.size(),
                    holdout + batch_n);
        return 1;
    }
    std::vector<Sample> train(all.begin(), all.end() - holdout);
    std::vector<Sample> hold(all.end() - holdout, all.end());
    std::vector<float> xh;
    to_float_states(hold, xh);
    std::printf("[样本] 训练 %zu 条 / 留出 %zu 条（留出集只用来看像不像，不参与训练）\n",
                train.size(), hold.size());

    // ---- 价值目标：默认改成蒸馏老师的 net_win ----
    // 原因：样本里的 ±1 是「一盘棋的最终结果」，在 600 模拟档位 74% 由先后手决定，
    // 几乎不含局面强弱信息；拿它回归等于拟合噪声。老师的 net_win 是稠密、已校准的评估，
    // 才是学生该对齐的东西。--value-src outcome 可退回旧口径做对照。
    {
        std::vector<float> xa;
        to_float_states(all, xa);
        const size_t n = all.size();
        std::vector<float> tv(n);
        const int chunk = 256;
        std::vector<float> tpb((size_t)chunk * N_CELLS);  // infer_batch 无条件写 policies，不能给 nullptr
        teacher.eval();
        torch::NoGradGuard g;
        for (size_t i = 0; i < n; i += chunk) {
            const int m = (int)std::min<size_t>(chunk, n - i);
            teacher.infer_batch(xa.data() + i * N_CELLS, m, tpb.data(), tv.data() + i);
        }
        int changed = 0;
        double mean_tv = 0.0;
        for (size_t i = 0; i < n; ++i) mean_tv += tv[i];
        mean_tv /= (double)std::max<size_t>(1, n);
        if (value_src != "outcome") {
            for (size_t i = 0; i < n; ++i) {
                all[i].value = (value_src == "blend") ? 0.5f * tv[i] + 0.5f * all[i].value
                                                      : tv[i];
                ++changed;
            }
        }
        std::printf("[价值目标] %s：改写 %d / %zu 条（老师 net_win 均值 %.3f）\n",
                    value_src.c_str(), changed, n, mean_tv);
        train.assign(all.begin(), all.end() - holdout);
        hold.assign(all.end() - holdout, all.end());
        xh.clear();
        to_float_states(hold, xh);
    }

    const Eval e0 = evaluate(teacher, student, xh, holdout);
    std::printf("[初始] top1 一致 %.3f   CE %.4f   value MSE %.5f\n", e0.agree, e0.ce, e0.mse);

    // ---- 训练 ----
    TrainOptions to;
    to.batch_size = batch_n;
    to.lr = lr;
    to.weight_decay = 0.0;
    to.clip_norm = 5.0;
    to.ema_decay = 0.999;
    to.fresh_ratio = 1.0;  // 全用给定样本，池子不参与
    to.amp = false;
    Trainer trainer(&student, to, 777);

    std::mt19937_64 rng(20261029);
    std::vector<int> idx(train.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    size_t pos = 0;
    int done = 0;
    double loss_ema = -1.0;
    const int report_every = std::max(50, steps / 12);
    while (done < steps) {
        if (pos + (size_t)batch_n > idx.size()) {
            std::shuffle(idx.begin(), idx.end(), rng);
            pos = 0;
        }
        std::vector<Sample> batch((size_t)batch_n);
        for (int i = 0; i < batch_n; ++i) batch[(size_t)i] = train[(size_t)idx[pos + (size_t)i]];
        pos += (size_t)batch_n;

        // 余弦退火：前期大步走，后期小步收，避免在最优附近来回振荡
        const double frac = (double)done / (double)std::max(1, steps);
        trainer.set_lr(lr * (0.05 + 0.95 * 0.5 * (1.0 + std::cos(3.14159265358979323846 * frac))));
        const TrainStepResult r = trainer.train_on(batch);
        if (!r.ok) {
            std::printf("[错误] 第 %d 步 train_on 返回 ok=false（batch 不完整？）\n", done);
            break;
        }
        loss_ema = (loss_ema < 0) ? r.loss : (loss_ema * 0.98 + r.loss * 0.02);
        ++done;
        if (done % report_every == 0 || done == steps) {
            std::printf("  步 %5d/%d  loss %.4f (ce %.4f mse %.4f)  %.1f ms/步  [%s]\n", done,
                        steps, loss_ema, r.ce, r.mse, r.ms_fwd + r.ms_data, "训练");
        }
        if (done % std::max(1, steps / 6) == 0 || done == steps) {
            const Eval e = evaluate(teacher, student, xh, holdout);
            std::printf("  └ 留出集: top1 一致 %.3f  CE %.4f  value MSE %.5f\n", e.agree, e.ce,
                        e.mse);
        }
    }

    const Eval e1 = evaluate(teacher, student, xh, holdout);
    std::printf("[最终] top1 一致 %.3f (初始 %.3f)  CE %.4f (初始 %.4f)  value MSE %.5f (初始 %.5f)\n",
                e1.agree, e0.agree, e1.ce, e0.ce, e1.mse, e0.mse);

    if (!save_named_tensors(opath, student.arch(), student.named_tensors())) {
        std::printf("[错误] 写权重失败: %s\n", opath.c_str());
        return 1;
    }
    std::printf("[完成] 学生权重已写入 %s\n", opath.c_str());
    return 0;
}
