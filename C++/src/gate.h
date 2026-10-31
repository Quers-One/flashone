// gate.h — 门控对局：候选网络（通常是 EMA 权重）vs 当前 best 网络
//
// 严格对标 PYTHON/train.py 的 play_match / gate_verdict：
//   * 双方各持一棵独立搜索树，每局开始 reset_tree()，落子后 set_root_move 复用
//   * 对局里走温度 0（贪心），但根的 Dirichlet 噪声保持打开（eval_noise_alpha=0.3,
//     eps=0.25）—— 否则同一先后手会下出一模一样的对局，20 局只有 2 个独立样本
//   * 局号偶数 → 候选执黑；奇数 → 候选执白（黑白各半，记账按颜色分开）
//   * win_rate = 候选胜局 / 总局数（和棋不算胜、但计入分母）
//   * 判定：win_rate >= threshold（2026-10-29 起由严格 > 改成含等号）；gate_ci_ll>0 时还要求
//     Wilson 下界 > ci_ll。改等号的理由：60 局下 33/60 恰好 = 0.5500，严格大于把「刚好达标」
//     判成未达标，而连续两次门控都是 33/60（gen 58、60）——一局之差决定的不是实力是硬币。
//   * 门控对局**没有认输逻辑**（与 Python 一致），一直下到五连或满盘
#pragma once

#include <cstdint>
#include <vector>

#include "mcts.h"

namespace gk {

struct GateOptions {
    int games = 20;                 // 奇数会自动 +1
    double threshold = 0.55;
    double ci_ll = 0.0;             // <=0 表示关闭第二道条件
    // 注意：n_sim 必须明显大于 MCTS 的 batch_size（默认 16）。等于或小于时，
    // 第一次 flush 之前所有模拟都落在根上、子节点访问数全 0（Python 同此行为），
    // 走子退化为「第一个合法点」。正式对局一律用 400。
    int n_sim = 400;
    double c_puct = 1.0;
    double noise_alpha = 0.3;
    double noise_eps = 0.25;
    double heuristic_weight = 0.5;
    double policy_weight = 1.0;
    int max_children = 6;
    int root_max_children = 16;
    // VCF 静态搜索（见 vcf.h）。默认关闭，理由见 config.h 实测注记。
    bool vcf = false;
    int vcf_max_ply = 6;
    int vcf_max_nodes = 128;
    bool guarantee_rush = false;
    // 【2026-10-31 加】根节点 VCF：落子前先在根算一次连续冲四必胜，命中就跳过 MCTS 直接落那一手。
    // 与上面的 vcf 是**两种接法**（详见 mcts.h 注释）：vcf 在叶子替换网络评估，vcf_root 不进
    // MCTS 内部。同样给候选侧留一个开关，方便做「同一权重、只开一侧」的强度 A/B。
    bool vcf_root = false;
    bool cand_vcf_root = false;
    // 只作用于候选侧的开关：用于「同一个网络、只改搜索配置」的强度 A/B。
    // 默认与 best 侧相同，所以不影响正常门控语义。
    bool cand_vcf = false;
    bool cand_guarantee_rush = false;
    // 候选侧模拟数；>0 时覆盖 n_sim。用于验证「这套对局能否测出已知强度差」。
    int cand_sims = 0;
    // 【配方 3 起】配对开局：每两局（偶=候选执黑、奇=候选执白）用**完全相同**的开局落子，
    // 只把黑白互换。理由：19 路无禁手下先手优势极大（实测候选执黑转化率 71% vs 执白 36%），
    // 开局不同时两局胜负大半由"抽到哪个开局"决定，方差把 1-2pp 的真实进步埋掉。
    // 同开局正反各一局是引擎测试的标准做法，能把颜色优势从估计里扣干净。
    // 0 = 关闭（与历史 43 代的门控口径一致）。
    int opening_plies = 0;
};

struct GateResult {
    int games = 0;
    int wins = 0;         // 候选胜
    int losses = 0;
    int draws = 0;
    int wins_black = 0, games_black = 0;
    int wins_white = 0, games_white = 0;
    double win_rate = 0.0;
    double wilson_lo = 0.0, wilson_hi = 0.0;
    double elapsed_ms = 0.0;
    bool pass_score = false;
    bool pass_ci = false;
    bool accepted = false;
};

// 单局。cand_black=true 时候选执黑先手。
// opening != nullptr 时先把这些点按「黑、白、黑、白…」摆上去，再开始搜索（配对开局用）。
// 返回 1=黑胜、2=白胜、0=平。
int play_gate_game(Evaluator* candidate, Evaluator* best, const GateOptions& opt, bool cand_black,
                   uint64_t seed, const std::vector<int>* opening = nullptr);

GateResult run_gate(Evaluator* candidate, Evaluator* best, const GateOptions& opt, uint64_t seed);

// 多线程版：每局互相独立（各自的树、各自的种子），所以并行不改变任何一局的结果，
// 只是把「20 局串行 ~45 s」压到 GPU 能同时吞下的吞吐。结果按局号汇总，与完成顺序无关。
GateResult run_gate_threads(Evaluator* candidate, Evaluator* best, const GateOptions& opt,
                            uint64_t seed, int nthreads);

// Wilson 95% 区间（与 PYTHON/ladder.py 同式，z=1.96）
void wilson_interval(int k, int n, double* lo, double* hi, double z = 1.96);

}  // namespace gk
