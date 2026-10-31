// config.h — 训练配置（默认值 = PYTHON V4 的实跑配方）
#pragma once

#include <cstdint>
#include <string>

#include "net.h"

namespace gk {

struct TrainConfig {
    // ---- 网络 ----
    Arch arch{6, 128, 64};
    bool cuda = true;

    // ---- 自对弈 ----
    int workers = 16;           // 并行自对弈线程数（V4 用 16 个进程；实测 16 线程最佳）
    int games = 320;            // 每代局数（160 时单代约 9k 样本，方差偏大）
    int n_sim = 600;            // 每手模拟数（400 时策略目标偏噪；代价是自对弈时间近似线性）
    int batch_size_mcts = 16;
    double c_puct = 1.0;
    double virtual_loss = 3.0;
    int max_children = 6;
    int root_max_children = 16;
    double heuristic_weight = 0.5;
    double policy_weight = 1.0;
    double temp_open = 1.0;
    double temp_mid = 0.5;
    int temp_threshold1 = 12;
    int temp_threshold2 = 30;
    double noise_alpha = 0.15;
    double noise_eps = 0.25;
    // 【2026-10-31 加】自对弈随机开局手数（0 = 关闭，与旧行为完全一致）。
    // 动机：无禁手下黑方先手优势巨大（统计上黑胜约 89%），而现在每局都从空盘开始、
    // 由 600 模拟搜出黑方最强起手 ⇒ 白方的样本几乎全是劣势局面，练不出防守。
    // 先随机走 N 手再开始正式搜索，可以削弱先手的稳定性、并让开局分散。
    // 这几手**不产生训练样本**（没有搜索访问数，policy 目标是随机的，学不到东西）。
    // 注意：做不到"给白方补一子"——board.cpp 的 place() 假定 who == player，
    // 在非行棋方落子会把 hash 的行棋方分量搞乱，而重算哈希的 zobrist() 是文件内静态。
    int sp_open_plies = 0;
    double resign_threshold = -0.95;
    int resign_plies = 3;
    int max_plies = 0;          // 0 = 直到终局
    int eval_batch = 32;        // 共享评估器的最大合并状态数（实测 24–32 最佳；GPU 是延迟瓶颈，凑大反而慢）
    double eval_window_ms = 0.4;  // 等更多请求的合并窗口
    int eval_threads = 3;       // 评估线程数（各自独立 CUDA stream，实测 3 条能填满单 stream 的同步空转）
    int mp_procs = 0;           // >1 = 多进程自对弈（每进程独立 CUDA 上下文）；0 = 单进程多线程

    // ---- VCF 静态搜索（连续冲四必胜的廉价证明，命中则跳过网络） ----
    // 实测：400 局自战（400 sims）开启后胜率 0.390，Wilson 上界 0.439 < 0.5，
    // 即显著变弱；速度无差异。故默认关闭，保留代码与开关供以后在更大 sims 下复测。
    bool vcf = false;
    int vcf_max_ply = 6;
    int vcf_max_nodes = 128;
    // 【2026-10-31 加】根节点 VCF（语义见 mcts.h）：落子前先在根算一次连续冲四必胜，
    // 命中就跳过 MCTS 直接落那一手。与上面 vcf 的不同是它**不进 MCTS 叶子**，
    // 所以不会破坏探索与价值回传，也不改变自对弈的训练数据分布。
    // 实测（HANDOFF §16.1）：同权重自打 400 局 0.4950，Elo −3.5，95%CI[−37.5,+30.6] ⇒
    // 600 sims 下中性（MCTS+网络本来就能找到这些强制胜）。默认关闭；只在低 sims 快棋
    // 场景还有一点正向苗头（120 sims 400 局 0.535，未坐实）。
    bool vcf_root = false;
    // 实测：400 局自战 0.450（Wilson [0.402, 0.499]），无收益。默认关闭。
    bool guarantee_rush = false;  // 宽度剪枝时无条件保留「四的争夺」点

    // ---- 回放池 ----
    size_t buffer_size = 500000;
    int buffer_gens = 60;
    double half_life = 20.0;
    // 0.5：每代只有 320 局新数据，太偏新数据会让策略随单代噪声来回摆（V4 用 0.7 时
    // 价值 mse 逐代从 0.32 涨到 0.41、7 代后强度与初始持平）
    double fresh_ratio = 0.5;

    // ---- 训练 ----
    int batch_size = 256;
    // 160 步 × 256 = 40k 次样本更新，对新数据（320 局 ≈ 17k 条）约 2.4 遍；
    // 原来是 320 步 = 过 9 遍，lr 1e-3 下每代都把上一代拽走
    int min_steps = 160;
    double samples_per_gen = 1.0;   // steps = max(min_steps, int(N*samples_per_gen/batch))
    // 3e-4：网络已接近收敛，1e-3 每代都会把好权重拽走
    double lr = 3e-4;
    double weight_decay = 0.0;
    double clip_norm = 5.0;
    double ema_decay = 0.999;
    // bf16 autocast 训练（卷积走 bf16，BN/softmax/loss 仍 fp32）。
    // 实测 batch=256 训练步 84 ms → 47 ms；数值与 fp32 的差异在 1e-3 量级，远小于自对弈噪声。
    bool amp = true;
    // 门控并行局数（对应 Python 的 --eval-workers 8）：每局独立树/种子，并行不改变结果
    int eval_workers = 8;
    bool ema_enabled = true;
    // 门控候选是否用 EMA 影子权重。默认 false：EMA(0.999) 落后约 3 代，会让门控
    // 结构性看不到最新进度（实测连续两代都测出同一个 7/20）
    bool gate_use_ema = false;
    // 权重快照保留代數：0 = 全留（默认，便于事后逐代对比）；>0 只留最近 N 代
    int keep_gens = 0;
    // 回放池样本文件保留代數（每代约 20 MB）：只需覆盖回放窗口，默认 10
    int keep_buffer_gens = 10;

    // ---- 门控 ----
    int eval_games = 20;
    int eval_every = 2;
    double gate_threshold = 0.55;
    double gate_ci_ll = 0.0;
    double lr_decay = 0.8;
    double lr_decay_min_win_rate = 0.30;
    double lr_floor = 1e-4;
    int max_lr_decay = 20;

    // ---- Elo 曲线（边库模式，见 elo_store.h 的动机说明）----
    // elo_every：训练中每多少代自动跑一次 Elo。默认 0 = 关。
    //   开着会让训练暂停（Elo 是真下棋 600 模拟，与训练抢同一块 GPU，只能串行），
    //   5 代一跑 ≈ 每 35 分钟多花 8 分钟。想要曲线自动更新就 --elo-every 5。
    int elo_every = 0;
    int elo_window = 5;          // 滚动窗口代数（会再往前延到「最新已评代-2」保证重叠）
    int elo_games = 30;          // 滚动边每对局数
    // 跨段标定：每 calib_every 代拿「calib_every 代前的老版本」直打当前代 400 局。
    // 链式拼接每接一段累积一次误差，这条直边是**重新标定尺子**——加进边库后
    // 全局拟合会自动把中间各代的位置校正。0 = 不做。
    int elo_calib_every = 15;
    // 采纳补打的局数。400 局 ±17 Elo / 11.5 分钟，150 局 ±28 Elo / 4.3 分钟。
    // 选 150 的原因：+10~+20 的进步**400 局也测不出来**（历史 075→080 实测 +22.6，
    // 已贴着噪声边），所以 400 局买到的主要是"发现配方崩坏"（配方 5 那次 −88）——
    // 150 局的 ±28 抓 −50 以上的退化绰绰有余。要对外结论时再单独打 600 局。
    int elo_calib_games = 150;
    int elo_max_ref_span = 30;   // 参照边跨度上限，超过就跳（那种边必饱和，纯浪费局数）
    // 【采纳驱动】门控一采纳，就把「这一代 vs 上一个被采纳的代」补打 elo_games 局进边库。
    // 这是默认推荐的口径，理由（来自实测教训）：
    //   · 门控回答"有没有更强"（二元），Elo 回答"强多少"（幅度）——两者测**同一对**模型，
    //     于是曲线上的每一段都是**直接对打**的结果，不是链式推的，没有累积误差；
    //   · 只在采纳时才跑：43 代里 19 代被拒，那些代本来也不该在曲线上占点；
    //   · 每段的局数用 elo_calib_games（400）而不是 30：30 局 ±45 Elo，看不出 +35 以内的进步，
    //     等于用噪声验证噪声。400 局 ±17 Elo，才配得上"正式结论"。
    // 代价：每次采纳约 4 分钟（训练暂停）。不想要就 --elo-on-accept 0。
    bool elo_on_accept = true;
    // 每**几次采纳**才补打一条 Elo 边（默认 2）。
    // 边是直接对打，跨 2 代打一条仍然是直接测量，误差不累积——只是曲线上的点稀一些。
    // 代价对比（每次采纳都测 vs 每 2 次测一次）：Elo 开销从 +43% 降到 +20%。
    int elo_every_accepts = 2;

    // ---- 配方编号与按代退火（配方 3 起）----
    // recipe_id：写进 train_log 每一行和 run.json。看板据此在损失曲线上标注「配方 N」，
    // 换配方的地方画一条竖虚线——否则跨配方的损失曲线根本不可比（模拟数/局数/步数一变，
    // 目标就变了），却会被误读成"模型退步"。
    int recipe_id = 0;
    // lr_gen_decay：每代把 lr 乘一次（含被门控拒绝的代），地板 lr_floor。1.0 = 关闭。
    // 与上面的 lr_decay 是两回事——那个只在「显著退化（胜率<0.30）」时触发，
    // 43 代实测一次都没触发过（胜率一直在 0.45–0.55），所以 lr 从未退火过。
    double lr_gen_decay = 1.0;
    // lr_reset：续跑时**不**接上一代的 lr，强制回到 --lr。默认 false（接着退火）。
    // 历史 bug：此前无论续跑与否都从 --lr 起算，于是配方 3 的按代退火一次续跑就断
    // （train_log 里 gen 49、59 复位到 0.000291）。要主动重新起算退火时加 --lr-reset 1。
    bool lr_reset = false;
    // 门控配对开局子数（0 = 关）。默认 2：同一对的正反两局用同一盘面，只互换黑白，
    // 把 19 路无禁手的巨大先手优势从估计里扣掉（实测执黑 71% vs 执白 36%）。
    int gate_opening_plies = 2;

    uint64_t seed = 12345;
    int log_every_steps = 0;   // 0 = 自动（steps/20）

    std::string weights_path() const;
};

}  // namespace gk
