// main.cpp — 命令行入口
//
// 子命令：
//   bench     单局搜索基准（ms/手），用于和 PYTHON/bench.py 的 119 ms/手 对比
//   selfplay  并行自对弈，输出样本与吞吐
//   gate      两个权重做门控对局
//   loop      完整训练循环（自对弈 → 训练 → 门控 → 落盘）
//   elo       版本间 Elo 估计（Bradley–Terry，输出 elo.json 给看板画曲线）
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "batcher.h"
#include "config.h"
#include "dataset.h"
#include "elo.h"
#include "elo_run.h"
#include "elo_store.h"
#include "eval_torch.h"
#include "gate.h"
#include "mcts.h"
#include "net.h"
#include "pipeline.h"
#include "sysutil.h"
#include "selfplay.h"
#include "trainer.h"
#include "vcf.h"

using namespace gk;

namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}


// 本地时间戳（写 elo.json / 边库用）
std::string stamp_now() {
    std::time_t t = std::time(nullptr);
    std::tm lt{};
#ifdef _WIN32
    localtime_s(&lt, &t);
#else
    localtime_r(&t, &lt);
#endif
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%S", &lt);
    return std::string(b);
}

struct Args {
    std::map<std::string, std::string> kv;
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    std::string str(const std::string& k, const std::string& d = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    int num(const std::string& k, int d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : std::atoi(it->second.c_str());
    }
    double dbl(const std::string& k, double d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : std::atof(it->second.c_str());
    }
    unsigned long long u64(const std::string& k, unsigned long long d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10);
    }
};

// --key value / --flag 形式
Args parse(int argc, char** argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            a.kv[k] = argv[++i];
        } else {
            a.kv[k] = "1";
        }
    }
    return a;
}

void apply_common(const Args& a, TrainConfig& cfg) {
    cfg.cuda = !a.has("cpu");
    cfg.arch.blocks = a.num("blocks", cfg.arch.blocks);
    cfg.arch.channels = a.num("channels", cfg.arch.channels);
    cfg.workers = a.num("workers", cfg.workers);
    cfg.games = a.num("games", cfg.games);
    cfg.n_sim = a.num("sims", cfg.n_sim);
    cfg.batch_size_mcts = a.num("batch-mcts", cfg.batch_size_mcts);
    cfg.c_puct = a.dbl("c-puct", cfg.c_puct);
    cfg.max_children = a.num("max-children", cfg.max_children);
    cfg.root_max_children = a.num("root-max-children", cfg.root_max_children);
    cfg.heuristic_weight = a.dbl("hw", cfg.heuristic_weight);
    cfg.policy_weight = a.dbl("pw", cfg.policy_weight);
    cfg.noise_alpha = a.dbl("noise-alpha", cfg.noise_alpha);
    cfg.noise_eps = a.dbl("noise-eps", cfg.noise_eps);
    cfg.sp_open_plies = a.num("sp-open-plies", cfg.sp_open_plies);
    cfg.resign_threshold = a.dbl("resign-threshold", cfg.resign_threshold);
    cfg.resign_plies = a.num("resign-plies", cfg.resign_plies);
    cfg.buffer_gens = a.num("buffer-gens", cfg.buffer_gens);
    cfg.buffer_size = (size_t)a.u64("buffer-size", cfg.buffer_size);
    cfg.half_life = a.dbl("half-life", cfg.half_life);
    cfg.fresh_ratio = a.dbl("fresh-ratio", cfg.fresh_ratio);
    cfg.batch_size = a.num("batch", cfg.batch_size);
    cfg.min_steps = a.num("min-steps", cfg.min_steps);
    cfg.lr = a.dbl("lr", cfg.lr);
    cfg.clip_norm = a.dbl("clip", cfg.clip_norm);
    cfg.ema_decay = a.dbl("ema-decay", cfg.ema_decay);
    if (a.has("no-amp")) cfg.amp = false;
    if (a.has("amp")) cfg.amp = true;
    cfg.eval_workers = a.num("eval-workers", cfg.eval_workers);
    if (a.has("gate-ema")) cfg.gate_use_ema = true;
    cfg.keep_gens = a.num("keep-gens", cfg.keep_gens);
    cfg.keep_buffer_gens = a.num("keep-buffer-gens", cfg.keep_buffer_gens);
    cfg.eval_games = a.num("eval-games", cfg.eval_games);
    cfg.eval_every = a.num("eval-every", cfg.eval_every);
    cfg.gate_threshold = a.dbl("gate-threshold", cfg.gate_threshold);
    cfg.gate_ci_ll = a.dbl("gate-ci-ll", cfg.gate_ci_ll);
    cfg.eval_batch = a.num("eval-batch", cfg.eval_batch);
    cfg.eval_window_ms = a.dbl("window-ms", cfg.eval_window_ms);
    cfg.eval_threads = a.num("eval-threads", cfg.eval_threads);
    cfg.mp_procs = a.num("procs", cfg.mp_procs);
    // VCF / guarantee_rush 默认关闭（实测无收益，见 config.h），这里只保留显式开启。
    if (a.has("vcf")) cfg.vcf = true;
    if (a.has("no-vcf")) cfg.vcf = false;
    cfg.vcf_max_ply = a.num("vcf-max-ply", cfg.vcf_max_ply);
    cfg.vcf_max_nodes = a.num("vcf-max-nodes", cfg.vcf_max_nodes);
    if (a.has("guarantee-rush")) cfg.guarantee_rush = true;
    if (a.has("no-guarantee-rush")) cfg.guarantee_rush = false;
    // 配方编号与按代退火（见 config.h 注释）
    cfg.recipe_id = a.num("recipe", cfg.recipe_id);
    cfg.lr_gen_decay = a.dbl("lr-gen-decay", cfg.lr_gen_decay);
    cfg.lr_floor = a.dbl("lr-floor", cfg.lr_floor);
    cfg.gate_opening_plies = a.num("opening-plies", cfg.gate_opening_plies);
    // Elo 曲线（边库模式）：训练中自动跑 + 跨段标定
    cfg.elo_every = a.num("elo-every", cfg.elo_every);
    cfg.elo_window = a.num("elo-window", cfg.elo_window);
    cfg.elo_games = a.num("elo-games", cfg.elo_games);
    cfg.elo_calib_every = a.num("elo-calib-every", cfg.elo_calib_every);
    cfg.elo_calib_games = a.num("elo-calib-games", cfg.elo_calib_games);
    cfg.elo_max_ref_span = a.num("elo-max-ref-span", cfg.elo_max_ref_span);
    cfg.elo_on_accept = a.num("elo-on-accept", cfg.elo_on_accept ? 1 : 0) != 0;
    cfg.elo_every_accepts = a.num("elo-every-accepts", cfg.elo_every_accepts);
    if (a.has("lr-reset")) cfg.lr_reset = true;
    cfg.seed = a.u64("seed", cfg.seed);
}

void usage() {
    printf(
        "用法: gk <命令> [选项]\n"
        "\n"
        "命令:\n"
        "  bench     单局搜索基准      --net F [--moves 20] [--sims 400] [--cpu]\n"
        "  selfplay  并行自对弈        --net F --games 16 [--workers 8] [--out DIR]\n"
        "  gate      门控对局          --cand F --best F [--games 20] [--sims 400]\n"
        "  loop      完整训练循环      --dir DIR [--init F] [--gens 10] [--games 160]\n"
        "  worker    多进程子进程      --net F --games N --out F --stats F\n"
        "  play      对战服务（stdin/stdout 行协议，网页用它下棋）  --net F\n"
        "  analyze   分析台服务（stdin/stdout 行协议，JSON 快照）  --net F\n"
        "  elo       版本间 Elo 曲线    --dir DIR --gens 4,12,20,24,34,43 [--eval-games 24]\n"
        "\n"
        "通用选项: --blocks 6 --channels 128 --sims 400 --workers 8 --batch-mcts 16\n"
        "          --c-puct 1.0 --max-children 6 --root-max-children 16 --hw 0.5 --pw 1.0\n"
        "          --eval-batch 256 --window-ms 0.4 --eval-threads 2 --procs 0\n"
        "          --eval-games 20 --eval-every 2 --eval-workers 8\n"
        "          --min-steps 320 --batch 256 --lr 0.001 --buffer-gens 60 --half-life 20\n"
        "          --fresh-ratio 0.5 --seed 12345 --cpu --no-amp --gate-ema --nice\n"
        "          --vcf（开启 VCF 静态搜索，默认关） --vcf-max-ply 6 --vcf-max-nodes 128\n"
        "          --guarantee-rush（保护冲四点，默认关）\n"
        "          gate 专用: --cand-vcf / --cand-no-vcf / --cand-guarantee-rush /"
        " --cand-no-guarantee-rush\n"
        "          --cand-sims N（只改候选侧模拟数，用于校准测试的分辨力）\n"
        "          （只改候选侧搜索配置；配 --no-vcf 用同一个网络做强度 A/B）\n"
        "          --keep-gens 0（0=保留全部代权重） --keep-buffer-gens 10\n"
        "          --recipe N（配方编号，写进 train_log） --lr-gen-decay 0.97 --lr-floor 1e-4\n"
        "          （按代退火；续跑时接着 train_log 末行的 lr，换配方才重设。--lr-reset 1 强制重设）\n"
        "          --opening-plies 2（门控配对开局，正反两局同盘面只换色，0=关）\n"
        "          （训练默认 bf16 autocast；门控默认 8 局并行、候选用原始 net；判据 win_rate >= 阈值）\n");
}

bool load_net(TorchNet& net, const std::string& path, std::string* err) {
    Arch a;
    std::vector<std::pair<std::string, torch::Tensor>> items;
    if (!load_named_tensors_file(path, a, items)) return false;
    if (!(a == net.arch())) {
        printf("[错误] %s 的架构 %d×%d×%d 与请求的 %d×%d×%d 不一致\n", path.c_str(), a.blocks,
               a.channels, a.value_hidden, net.arch().blocks, net.arch().channels,
               net.arch().value_hidden);
        return false;
    }
    return net.load_named_tensors(items, true, err);
}

int cmd_bench(const Args& a) {
    TrainConfig cfg;
    apply_common(a, cfg);
    const std::string netpath = a.str("net");
    if (netpath.empty()) {
        printf("需要 --net\n");
        return 2;
    }
    TorchNet net(cfg.arch, cfg.cuda, cfg.seed);
    std::string err;
    if (!load_net(net, netpath, &err)) {
        printf("[错误] 载入权重失败: %s\n", err.c_str());
        return 1;
    }
    TorchEvaluator ev(&net);
    const int moves = a.num("moves", 20);
    MCTSOptions mo = make_mcts_options(cfg);
    MCTS mcts(&ev, mo, cfg.seed);

    Pos board;
    std::vector<float> pol(N_CELLS);
    std::vector<double> times;
    int64_t evals = 0;
    const double t_all = now_ms();
    std::mt19937_64 rng(cfg.seed);
    for (int i = 0; i < moves; ++i) {
        if (board.stones >= N_CELLS) break;
        const double t0 = now_ms();
        const int mv = mcts.sample_move_with_policy(board, pol, 0.5, cfg.n_sim, cfg.noise_alpha,
                                                    cfg.noise_eps);
        times.push_back(now_ms() - t0);
        const MCTSStats st = mcts.collect_stats();
        evals = st.states_evaluated;
        if (a.has("verbose")) {
            printf("  手 %2d: %.1f ms  sims=%lld fwd=%lld states=%lld cache=%lld terminal=%lld "
                   "vcf=%lld nodes=%lld 根子=%d\n",
                   i, times.back(), (long long)st.simulations, (long long)st.network_calls,
                   (long long)st.states_evaluated, (long long)st.cache_hits,
                   (long long)st.terminal_hits, (long long)st.vcf_hits,
                   (long long)st.nodes_created, st.root_children);
        }
        if (mv < 0) break;
        board.place(mv % SIZE, mv / SIZE, board.player);
        mcts.set_root_move((int16_t)mv);
    }
    const double total = now_ms() - t_all;
    std::sort(times.begin(), times.end());
    const double sum = std::accumulate(times.begin(), times.end(), 0.0);
    const MCTSStats st = mcts.collect_stats();
    printf("bench: %zu 手，总 %.1f ms，平均 %.1f ms/手，中位 %.1f ms/手\n", times.size(), total,
           times.empty() ? 0.0 : sum / (double)times.size(),
           times.empty() ? 0.0 : times[times.size() / 2]);
    printf("       前向 %llu 次（%.1f 次/手），状态 %llu，VCF 命中 %lld，%.2f 手/s\n",
           (unsigned long long)st.network_calls,
           times.empty() ? 0.0 : (double)st.network_calls / (double)times.size(),
           (unsigned long long)evals, (long long)st.vcf_hits,
           total > 0 ? (double)times.size() * 1000.0 / total : 0.0);
    return 0;
}

// ------------------------------------------------------------------ 对战服务
// 供 PYTHON/web_play.py 驱动的**单行文本协议**（一行请求 / 一行应答，好解析）：
//   GO <n_sim> <hw> <pw> <手数> <x1> <y1> ...   → 算这个局面（从空盘黑白交替）的最佳手
//   →  MOVE <x> <y> <胜率> <ms> <候选数> [<x> <y> <n> <胜率> <先验>]*
//   INFO                                        →  INFO <b>x<c>x<v> <参数量> <cpu|cuda>
//   QUIT                                        →  BYE
//   ERR <原因>                                  ← 局面非法 / 已终局
// 胜率一律是「**轮到的这一方**」视角（Python 那边按人/AI 再翻一次）。
//
// 树复用：用 set_root_move 沿手顺把根推进下去；推不动（比如人类那一手搜索没访问过）
// 就整体重建 —— 语义与 Python 版一致，只是少一点复用。
// ------------------------------------------------------------------ VCF 战术基准
// Elo 只能回答"它比上一代强多少"，回答不了"它算杀能力如何"。这个命令把 VCF
// （连续冲四）的**穷举证明**当成真值 —— 不是启发式判断，是"这个局面确实存在
// 强制胜"。于是"在有限算力下能否找到已被证明的胜手"就变成一个可对外比较的数字。
//
// 用法：
//   gk vcfsuite < pos.txt                         每行一个局面，逐行输出是否存在必胜 + 首手
//   gk vcfsuite --check-x 8 --check-y 7 < pos.txt 判定指定着法是否为必胜着
// 输入：每行 "x1 y1 x2 y2 ..."，黑先、黑白交替，坐标 0..18（同 play 协议的约定）。
static bool vcf_move_wins(Pos pos, int mx, int my, const VCFOptions& opt) {
    const int who = pos.player;
    if (mx < 0 || my < 0 || mx >= SIZE || my >= SIZE || pos.at(mx, my) != 0) return false;
    Pos after = pos;
    if (after.place(mx, my, who)) return true;   // 直接成五
    if (!after.has_legal()) return false;        // 棋盘填满是平局，不算胜
    // 我方这一步落下后，对方**任何**应手都必须仍然挡不住 VCF 才算必胜着。
    // 这比 VCF 本身更严（VCF 只要求对方走唯一挡点），所以只会低估、不会高估。
    for (int y = 0; y < SIZE; ++y)
        for (int x = 0; x < SIZE; ++x) {
            if (after.at(x, y) != 0) continue;
            Pos t = after;
            if (t.place(x, y, after.player)) return false;   // 对方成五，我们输了
            if (vcf_win(t, opt) == 0) return false;
        }
    return true;
}

int cmd_vcfsuite(const Args& a) {
    VCFOptions vo;
    vo.max_ply = a.num("vcf-max-ply", vo.max_ply);
    vo.max_nodes = a.num("vcf-max-nodes", vo.max_nodes);
    const bool do_check = a.has("check-x") && a.has("check-y");
    const int qx = a.num("check-x", -1), qy = a.num("check-y", -1);

    std::string line;
    int idx = 0;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::vector<int> mv;
        int x = 0, y = 0;
        while (is >> x >> y) mv.push_back(y * SIZE + x);
        if (mv.empty()) continue;
        Pos pos;
        bool bad = false;
        for (int m : mv) {
            if (m < 0 || m >= N_CELLS || pos.at(m % SIZE, m / SIZE) != 0) { bad = true; break; }
            pos.place(m % SIZE, m / SIZE, pos.player);
        }
        const double t0 = now_ms();
        if (bad) {
            printf("{\"idx\":%d,\"err\":\"badmove\"}\n", idx++);
        } else if (do_check) {
            const bool ok = vcf_move_wins(pos, qx, qy, vo);
            printf("{\"idx\":%d,\"ply\":%d,\"winning\":%d,\"ms\":%.1f}\n",
                   idx++, (int)mv.size(), ok ? 1 : 0, now_ms() - t0);
        } else {
            int cell = -1;
            const int win = vcf_find_move(pos, &cell, vo);
            if (win && cell >= 0)
                printf("{\"idx\":%d,\"ply\":%d,\"vcf\":1,\"x\":%d,\"y\":%d,\"ms\":%.1f}\n",
                       idx++, (int)mv.size(), cell % SIZE, cell / SIZE, now_ms() - t0);
            else
                printf("{\"idx\":%d,\"ply\":%d,\"vcf\":0,\"ms\":%.1f}\n",
                       idx++, (int)mv.size(), now_ms() - t0);
        }
        fflush(stdout);
    }
    return 0;
}

int cmd_play(const Args& a) {
    TrainConfig cfg;
    apply_common(a, cfg);
    const std::string netpath = a.str("net");
    if (netpath.empty()) {
        printf("需要 --net\n");
        return 2;
    }
    TorchNet net(cfg.arch, cfg.cuda, cfg.seed);
    std::string err;
    if (!load_net(net, netpath, &err)) {
        printf("[错误] 载入权重失败: %s\n", err.c_str());
        return 1;
    }
    TorchEvaluator ev(&net);
    MCTSOptions mo = make_mcts_options(cfg);
    MCTS mcts(&ev, mo, cfg.seed);
    mcts.heuristic_weight = cfg.heuristic_weight;
    mcts.policy_weight = cfg.policy_weight;
    std::vector<int> applied;                 // 已经推进到树根上手的落子
    printf("READY %dx%dx%d %s\n", cfg.arch.blocks, cfg.arch.channels, cfg.arch.value_hidden,
           cfg.cuda ? "cuda" : "cpu");
    fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        if (!(is >> cmd)) continue;
        if (cmd == "QUIT") {
            printf("BYE\n");
            fflush(stdout);
            break;
        }
        if (cmd == "INFO") {
            printf("INFO %dx%dx%d %lld %s\n", cfg.arch.blocks, cfg.arch.channels,
                   cfg.arch.value_hidden, 0LL, cfg.cuda ? "cuda" : "cpu");
            fflush(stdout);
            continue;
        }
        if (cmd != "GO") {
            printf("ERR unknown\n");
            fflush(stdout);
            continue;
        }
        int nsim = cfg.n_sim, n = 0;
        double hw = cfg.heuristic_weight, pw = cfg.policy_weight;
        is >> nsim >> hw >> pw >> n;
        std::vector<int> moves;
        for (int i = 0; i < n; ++i) {
            int x = -1, y = -1;
            is >> x >> y;
            moves.push_back(y * SIZE + x);
        }
        const double t0 = now_ms();
        Pos pos;
        bool bad = false;
        for (size_t i = 0; i < moves.size(); ++i) {
            const int mv = moves[i];
            // 注意：pos.place() 的返回值是「这一手是否成五」，不是「是否成功」。
            // 合法性只看该点是否已有子。
            if (mv < 0 || mv >= N_CELLS || pos.at(mv % SIZE, mv / SIZE) != 0) {
                bad = true;
                printf("ERR badmove %d\n", (int)i);
                fflush(stdout);
                break;
            }
            pos.place(mv % SIZE, mv / SIZE, pos.player);
        }
        if (bad) {
            applied.clear();
            mcts.reset_tree();
            continue;
        }
        if (pos.has_legal() == false) {
            printf("ERR terminal\n");
            fflush(stdout);
            continue;
        }
        // 沿手顺把树根推下去；推不动就重建（get_action_probs 里会 reset）。
        bool aligned = applied.size() <= moves.size();
        if (aligned) {
            for (size_t i = 0; i < applied.size(); ++i)
                if (applied[i] != moves[i]) {
                    aligned = false;
                    break;
                }
        }
        if (aligned) {
            for (size_t i = applied.size(); i < moves.size(); ++i) {
                if (!mcts.set_root_move((int16_t)moves[i])) {
                    mcts.reset_tree();
                    applied.clear();
                    break;
                }
                applied.push_back(moves[i]);
            }
        } else {
            mcts.reset_tree();
            applied.clear();
        }
        mcts.heuristic_weight = hw;
        mcts.policy_weight = pw;
        mcts.set_n_sim(nsim);
        const int best = mcts.get_best_action(pos, nsim);
        if (best < 0) {
            printf("ERR nomove\n");
            fflush(stdout);
            continue;
        }
        // 候选统计：子节点 q 是「落子后轮到的那一方」视角 ⇒ 落子方胜率 = (1-q)/2
        struct Row {
            int mv;
            int n;
            double win;
            double p;
        };
        std::vector<Row> rows;
        Node* root = mcts.root();
        if (root) {
            for (const auto& c : root->children) {
                if (c.node->n <= 0) continue;
                const double q = c.node->w / (double)c.node->n;
                rows.push_back(Row{c.move, c.node->n, (1.0 - q) / 2.0, c.node->prior});
            }
        }
        std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) { return x.n > y.n; });
        const double stm_win = root ? (root->q + 1.0) / 2.0 : 0.5;
        const size_t nk = rows.size() < 8 ? rows.size() : 8;
        printf("MOVE %d %d %.4f %.1f %zu", best % SIZE, best / SIZE, stm_win, now_ms() - t0, nk);
        for (size_t i = 0; i < nk; ++i)
            printf(" %d %d %d %.4f %.4f", rows[i].mv % SIZE, rows[i].mv / SIZE, rows[i].n,
                   rows[i].win, rows[i].p);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}

// 往 std::string 里拼一行 JSON 的小工具：比逐个 printf 少了每格一次 flush
// （setvbuf 是 _IONBF，一段 361 格的分析要 printf 上千次，摊到管道上就是几十毫秒）。
void appendf(std::string& s, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(buf)) {
        s.append(buf, (size_t)n);
    } else {
        s.append(buf, sizeof(buf) - 1);
    }
}

// ------------------------------------------------------------------ 分析服务
// 供 PYTHON/web_play.py 的「全盘分析」用，输出契约逐键对齐 PYTHON/analysis.py
// 的 analyze_board()。为什么单开一个子命令、而不是塞进 cmd_play：
//   1. 分析要 root_max_children = 361（根全展开），对局要 16，同进程会互相改配置；
//   2. 分析是长跑、对局是短请求，共用一条 stdin 会让搜索树在两者间跳变
//      （每批都得 reset_tree，复用失效、batch 变小，速度掉回 Python 水平）；
//   3. 分析崩了不该带崩对局（反之亦然）。
//
// 行协议（一行请求 / 一行应答；细节见 C++/ANALYZE_HANDOFF.md §5）：
//   AN <n_sim> <hw> <pw> <full_root> <手数> (<x> <y> <p>)*
//       ⇒ 在当前局面上**追加** n_sim 次模拟，回一行 JSON 快照；
//         手顺与树根不一致时内部 reset_tree() 重建。
//         一次性分析 = 只发一条 AN；持续分析 = 循环发（同一棵树，访问只增不减）。
//   INFO  ⇒  INFO <b>x<c>x<v> <参数量> <cpu|cuda>
//   QUIT  ⇒  BYE 后退出
// 手顺带 <p>（1 黑 / 2 白）是因为网页的「摆棋」允许不按黑白交替落子，
// 只靠交替推不出真实局面；p 缺失或非 1/2 时按交替处理（向后兼容）。
int cmd_analyze(const Args& a) {
    TrainConfig cfg;
    apply_common(a, cfg);
    const std::string netpath = a.str("net");
    if (netpath.empty()) {
        printf("需要 --net\n");
        return 2;
    }
    if (cfg.n_sim <= 0) cfg.n_sim = 800;   // 分析口径的默认批大小（web_play 也会显式给）
    TorchNet net(cfg.arch, cfg.cuda, cfg.seed);
    std::string err;
    if (!load_net(net, netpath, &err)) {
        printf("[错误] 载入权重失败: %s\n", err.c_str());
        return 1;
    }
    TorchEvaluator ev(&net);
    MCTSOptions mo = make_mcts_options(cfg);
    mo.root_max_children = N_CELLS;         // 分析口径：根全展开（AN 里可改为 16）
    MCTS mcts(&ev, mo, cfg.seed);
    mcts.root_noise_alpha = 0.0;            // 分析不要根噪声：访问数要能解释
    mcts.heuristic_weight = cfg.heuristic_weight;
    mcts.policy_weight = cfg.policy_weight;

    // 预热：把首次前向的 kernel 初始化 / CUDA Graph 捕获挪到 READY 之前。
    // 分析台是常驻进程，这笔钱只该付一次；否则第一条 AN 平白多 ~400 ms。
    {
        Pos fresh;
        std::vector<float> one(3 * N_CELLS), pol16(16 * N_CELLS), val16(16);
        std::vector<float> many(16 * 3 * N_CELLS);
        fresh.to_tensor(one.data());
        for (int b = 0; b < 16; ++b)
            std::memcpy(many.data() + (size_t)b * 3 * N_CELLS, one.data(),
                        3 * N_CELLS * sizeof(float));
        ev.evaluate_batch(many.data(), 16, pol16.data(), val16.data());   // MCTS 的批形状
        ev.evaluate_batch(one.data(), 1, pol16.data(), val16.data());     // netp/net_win 形状
    }
    printf("READY %dx%dx%d %s\n", cfg.arch.blocks, cfg.arch.channels, cfg.arch.value_hidden,
           cfg.cuda ? "cuda" : "cpu");
    fflush(stdout);

    std::vector<int> applied;               // 已推进到树根上的落子（扁平下标）
    std::vector<int> moves;                 // 本条 AN 的手顺
    std::vector<int> legal;
    std::vector<uint8_t> is_legal(N_CELLS, 0);
    std::vector<uint8_t> w5(N_CELLS, 0), b5(N_CELLS, 0);
    std::vector<int> cn(N_CELLS, 0);
    std::vector<double> cw(N_CELLS, 0.0);
    std::vector<float> cprior(N_CELLS, -1.f);
    std::vector<uint8_t> cterm(N_CELLS, 0);
    std::vector<float> pol(N_CELLS), polbuf(N_CELLS), st(3 * N_CELLS);
    std::string json;
    json.reserve(64 * 1024);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        if (!(is >> cmd)) continue;
        if (cmd == "QUIT") {
            printf("BYE\n");
            fflush(stdout);
            break;
        }
        if (cmd == "INFO") {
            printf("INFO %dx%dx%d %lld %s\n", cfg.arch.blocks, cfg.arch.channels,
                   cfg.arch.value_hidden, 0LL, cfg.cuda ? "cuda" : "cpu");
            fflush(stdout);
            continue;
        }
        if (cmd != "AN") {
            printf("ERR unknown\n");
            fflush(stdout);
            continue;
        }

        int nsim = cfg.n_sim, full_root = 1, n = 0;
        double hw = cfg.heuristic_weight, pw = cfg.policy_weight;
        if (!(is >> nsim >> hw >> pw >> full_root >> n) || n < 0 || n > N_CELLS) {
            printf("ERR badargs\n");
            fflush(stdout);
            continue;
        }
        if (nsim < 1) nsim = 1;

        Pos pos;
        moves.clear();
        moves.reserve((size_t)n);
        bool bad = false, over = false;
        for (int i = 0; i < n; ++i) {
            int x = -1, y = -1, p = 0;
            if (!(is >> x >> y >> p)) bad = true;
            if (!bad && (x < 0 || x >= SIZE || y < 0 || y >= SIZE)) bad = true;
            if (!bad && pos.at(x, y) != 0) bad = true;
            if (bad) {
                printf("ERR badmove %d\n", i);
                fflush(stdout);
                break;
            }
            if (p != BLACK && p != WHITE) p = pos.player;   // 没给 ⇒ 按黑白交替
            // 摆棋允许不按交替落子：hash 里的行棋方（side[]）与 player 一起改成记录值，
            // 否则置换表会把"同一堆子、不同执子方"当成同一个局面。
            if (pos.player != p) {
                const Zobrist& z = zobrist();
                pos.hash ^= z.side[pos.player - 1] ^ z.side[p - 1];
                pos.player = (int8_t)p;
            }
            if (pos.place(x, y, p)) over = true;
            moves.push_back(y * SIZE + x);
        }
        if (bad) continue;
        if (over || pos.stones >= N_CELLS || !pos.has_legal()) {
            printf("ERR terminal\n");
            fflush(stdout);
            continue;
        }

        // 根宽：全展开 361 / 对局口径 16。已经展开的根改不回来，只能重建树。
        const int want_width = full_root ? N_CELLS : 16;
        bool rebuilt = false;
        if (mcts.options().root_max_children != want_width) {
            mcts.reset_tree();
            mcts.options().root_max_children = want_width;
            rebuilt = true;
        }
        if (!rebuilt) {
            // 照抄 cmd_play 的前缀比对：推不动 / 手顺与树根分岔就整体重建。
            bool aligned = applied.size() <= moves.size();
            if (aligned) {
                for (size_t i = 0; i < applied.size(); ++i)
                    if (applied[i] != moves[i]) {
                        aligned = false;
                        break;
                    }
            }
            if (aligned) {
                for (size_t i = applied.size(); i < moves.size(); ++i) {
                    if (!mcts.set_root_move((int16_t)moves[i])) {
                        rebuilt = true;
                        break;
                    }
                }
            } else {
                rebuilt = true;
            }
            if (rebuilt) mcts.reset_tree();
        }
        // 本条 AN 结束后树根一定落在 pos 上：要么 set_root_move 推到了这里，
        // 要么 reset 后由 get_action_probs 里的 set_root(pos) 重建。
        applied = moves;

        mcts.heuristic_weight = hw;
        mcts.policy_weight = pw;
        const double t0 = now_ms();
        mcts.get_action_probs(pos, pol, 0.0, nsim);
        const double dt_ms = now_ms() - t0;

        // 网络直觉（netp / net_win）：一次单状态前向 ≈1.7 ms，占 800 模拟的 1.4%，
        // 每批直接算最省事（analysis.py 复用 net_pol 只是为了省这 1.4%）。
        pos.to_tensor(st.data());
        float val = 0.f;
        ev.evaluate_batch(st.data(), 1, polbuf.data(), &val);
        const double net_win = (val + 1.0) / 2.0;

        pos.legal_moves(legal);
        std::fill(is_legal.begin(), is_legal.end(), 0);
        for (int mv : legal) is_legal[(size_t)mv] = 1;
        const int mover = pos.player;
        for (int i = 0; i < N_CELLS; ++i) {
            // 只在「合法且未终局」时算，否则会误报（analysis.py:167-172 同口径）
            if (!is_legal[(size_t)i]) continue;
            if (pos.check_win_after(i, mover)) {
                w5[(size_t)i] = 1;
            } else if (pos.check_win_after(i, 3 - mover)) {
                b5[(size_t)i] = 1;
            }
        }

        std::fill(cn.begin(), cn.end(), 0);
        std::fill(cw.begin(), cw.end(), 0.0);
        std::fill(cprior.begin(), cprior.end(), -1.f);
        std::fill(cterm.begin(), cterm.end(), 0);
        Node* root = mcts.root();
        if (root) {
            for (const auto& c : root->children) {
                const int mv = c.move;
                if (mv < 0 || mv >= N_CELLS) continue;
                cn[(size_t)mv] = c.node->n;
                cw[(size_t)mv] = c.node->w;
                cprior[(size_t)mv] = (float)c.node->prior;
                cterm[(size_t)mv] = c.node->is_terminal ? 1 : 0;
            }
        }

        // 候选排行：按访问降序、同访问按落子方胜率降序（与 analysis.py:183-184 同口径）
        struct Row {
            int x, y, n;
            double win, prior;
        };
        std::vector<Row> rows;
        rows.reserve(64);
        for (int i = 0; i < N_CELLS; ++i) {
            if (!is_legal[(size_t)i] || cn[(size_t)i] <= 0) continue;
            Row r;
            r.x = i % SIZE;
            r.y = i / SIZE;
            r.n = cn[(size_t)i];
            // 子节点 q 是「落子后轮到的那一方」视角 ⇒ 落子方胜率要取反（HANDOFF §7.1）
            r.win = (1.0 - cw[(size_t)i] / (double)r.n) / 2.0;
            r.prior = (double)cprior[(size_t)i];
            rows.push_back(r);
        }
        std::sort(rows.begin(), rows.end(), [](const Row& p, const Row& q) {
            if (p.n != q.n) return p.n > q.n;
            return p.win > q.win;
        });
        int biw = -1;
        for (size_t i = 0; i < rows.size(); ++i)
            if (biw < 0 || rows[i].win > rows[(size_t)biw].win) biw = (int)i;

        json.clear();
        appendf(json, "{\"ply\":%d,\"mover\":%d,\"over\":false,\"winner\":null",
                (int)pos.stones, mover);
        appendf(json, ",\"legal_count\":%d,\"n_sim\":%d,\"full_root\":%s", (int)legal.size(), nsim,
                full_root ? "true" : "false");
        appendf(json, ",\"root_n\":%d,\"root_children\":%d", root ? root->n : 0,
                root ? (int)root->children.size() : 0);
        if (root) {
            appendf(json, ",\"root_win\":%.4f", (root->q + 1.0) / 2.0);   // 根 q = 执子方视角
        } else {
            appendf(json, ",\"root_win\":null");
        }
        appendf(json, ",\"net_win\":%.4f", net_win);
        if (!rows.empty()) {
            const Row& b0 = rows[0];
            const Row& bw = rows[(size_t)biw];
            appendf(json, ",\"best\":{\"x\":%d,\"y\":%d,\"n\":%d,\"win\":%.4f}", b0.x, b0.y, b0.n,
                    b0.win);
            appendf(json, ",\"best_by_win\":{\"x\":%d,\"y\":%d,\"n\":%d,\"win\":%.4f}", bw.x, bw.y,
                    bw.n, bw.win);
        } else {
            appendf(json, ",\"best\":null,\"best_by_win\":null");
        }
        appendf(json, ",\"cands\":[");
        const size_t nk = rows.size() < 10 ? rows.size() : 10;
        for (size_t i = 0; i < nk; ++i) {
            const Row& r = rows[i];
            const int idx = r.y * SIZE + r.x;
            const double dw = r.win - rows[0].win;   // Δ 相对最佳点（analysis.py:194-195）
            appendf(json,
                    "%s{\"x\":%d,\"y\":%d,\"n\":%d,\"win\":%.4f,\"prior\":%.5f,\"netp\":%.5f,"
                    "\"w5\":%s,\"b5\":%s,\"dwin\":%.4f}",
                    i ? "," : "", r.x, r.y, r.n, r.win, r.prior, polbuf[(size_t)idx],
                    w5[(size_t)idx] ? "true" : "false", b5[(size_t)idx] ? "true" : "false", dw);
        }
        appendf(json, "],\"points\":[");
        for (int i = 0; i < N_CELLS; ++i) {
            const int x = i % SIZE, y = i / SIZE;
            if (i) json.push_back(',');
            appendf(json, "{\"x\":%d,\"y\":%d,\"occ\":%d,\"legal\":%s,\"n\":%d,\"win\":", x, y,
                    (int)pos.at(x, y), is_legal[(size_t)i] ? "true" : "false", cn[(size_t)i]);
            if (cn[(size_t)i] > 0) {
                appendf(json, "%.4f", (1.0 - cw[(size_t)i] / (double)cn[(size_t)i]) / 2.0);
            } else {
                appendf(json, "null");   // n==0 时不要算 w/n（会打印 NaN）
            }
            if (cprior[(size_t)i] >= 0.f) {
                appendf(json, ",\"prior\":%.5f", cprior[(size_t)i]);
            } else {
                appendf(json, ",\"prior\":null");
            }
            appendf(json, ",\"netp\":%.5f,\"w5\":%s,\"b5\":%s,\"terminal\":%s}",
                    polbuf[(size_t)i], w5[(size_t)i] ? "true" : "false",
                    b5[(size_t)i] ? "true" : "false", cterm[(size_t)i] ? "true" : "false");
        }
        appendf(json, "],\"elapsed\":%.3f}\n", dt_ms / 1000.0);
        std::fwrite(json.data(), 1, json.size(), stdout);
        std::fflush(stdout);
    }
    return 0;
}

int cmd_selfplay(const Args& a) {
    TrainConfig cfg;
    apply_common(a, cfg);
    const std::string netpath = a.str("net");
    if (netpath.empty()) {
        printf("需要 --net\n");
        return 2;
    }
    TorchNet net(cfg.arch, cfg.cuda, cfg.seed);
    std::string err;
    if (!load_net(net, netpath, &err)) {
        printf("[错误] 载入权重失败: %s\n", err.c_str());
        return 1;
    }
    TorchEvaluator ev(&net);
    std::vector<Sample> samples;
    const double t0 = now_ms();

    // 多进程路径：每个子进程独立 CUDA 上下文、一次只跑 1 局（与 Python 16 进程同构）
    if (cfg.mp_procs > 1) {
        const std::string tmp = a.str("tmpdir").empty() ? std::string("..\\_tmp\\sp_tmp") : a.str("tmpdir");
        SelfPlayStats sp = collect_games_mp(cfg, netpath, cfg.games, cfg.seed, tmp, samples);
        const double dt = now_ms() - t0;
        printf("selfplay(mp %d 进程): %d 局 %d 样本，%.1f s，%.2f 局/s，%.1f 手/s，%.1f 手/局\n",
               cfg.mp_procs, sp.games, sp.samples, dt / 1000.0,
               dt > 0 ? (double)sp.games * 1000.0 / dt : 0.0,
               dt > 0 ? (double)sp.plies * 1000.0 / dt : 0.0,
               sp.games ? (double)sp.plies / sp.games : 0.0);
        printf("          黑 %d / 白 %d / 平 %d / 认输 %d / 截断 %d\n", sp.black_wins, sp.white_wins,
               sp.draws, sp.resigned, sp.truncated);
        const std::string mpout = a.str("out");
        if (!mpout.empty() && save_samples(mpout, samples)) printf("          样本已写入 %s\n", mpout.c_str());
        return sp.samples > 0 ? 0 : 1;
    }

    NetService svc(&ev, cfg.eval_batch, cfg.eval_window_ms, cfg.eval_threads);
    if (cfg.workers > 1) svc.start();
    SelfPlayStats sp = collect_games(cfg.workers > 1 ? (Evaluator*)&svc : (Evaluator*)&ev, cfg,
                                     cfg.games, cfg.seed, samples);
    const double dt = now_ms() - t0;
    if (cfg.workers > 1) svc.stop();
    const NetService::Stats ns = svc.stats();
    printf("selfplay: %d 局 %d 样本，%.1f s，%.2f 局/s，%.1f 手/s，%.1f 手/局\n", sp.games,
           sp.samples, dt / 1000.0, dt > 0 ? (double)sp.games * 1000.0 / dt : 0.0,
           dt > 0 ? (double)sp.plies * 1000.0 / dt : 0.0,
           sp.games ? (double)sp.plies / sp.games : 0.0);
    printf("          前向 %llu 次，状态 %llu，平均 batch %.1f（%.1f ms/手）\n",
           (unsigned long long)ns.batches, (unsigned long long)ns.states,
           ns.batches ? (double)ns.states / (double)ns.batches : 0.0,
           sp.plies ? dt / (double)sp.plies : 0.0);
    printf("          黑 %d / 白 %d / 平 %d / 认输 %d / 截断 %d\n", sp.black_wins, sp.white_wins,
           sp.draws, sp.resigned, sp.truncated);
    printf("          [net] service %.1f s (%.1f%% wall) / sum-wait %.1f s / batches/s %.1f\n",
           ns.service_ms / 1000.0, dt > 0 ? ns.service_ms / dt * 100.0 : 0.0,
           ns.wait_ms / 1000.0, dt > 0 ? (double)ns.batches * 1000.0 / dt : 0.0);
    if (!ns.fwd_ms.empty()) {
        std::vector<double> q = ns.fwd_ms;
        std::sort(q.begin(), q.end());
        const size_t n = q.size();
        printf("          [net] 单批耗时 p50 %.2f / p95 %.2f / max %.2f ms\n", q[n / 2],
               q[(size_t)((double)n * 0.95)], q[n - 1]);
    }
    const std::string out = a.str("out");
    if (!out.empty()) {
        std::string path = out;
        if (path.size() > 4 && path.substr(path.size() - 4) == ".bin") {
            // 直接用
        } else {
            path += "/samples.bin";
        }
        if (save_samples(path, samples)) printf("          样本已写入 %s\n", path.c_str());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// worker：多进程自对弈的子进程。一次只做 --games 局，样本写 --out，统计写 --stats。
// 故意保持安静（stdout 与父进程共用控制台，只在出错时打印）。
// ---------------------------------------------------------------------------
int cmd_worker(const Args& a) {
    TrainConfig cfg;
    apply_common(a, cfg);
    cfg.workers = 1;  // 子进程内不再开线程池，并发靠多进程
    const std::string netpath = a.str("net");
    const std::string out = a.str("out");
    if (netpath.empty() || out.empty()) {
        fprintf(stderr, "[错误] worker 需要 --net 与 --out\n");
        return 2;
    }
    TorchNet net(cfg.arch, cfg.cuda, cfg.seed);
    std::string err;
    if (!load_net(net, netpath, &err)) {
        fprintf(stderr, "[错误] worker 载入权重失败: %s\n", err.c_str());
        return 1;
    }
    TorchEvaluator ev(&net);
    std::vector<Sample> samples;
    const SelfPlayStats sp = collect_games(&ev, cfg, cfg.games, cfg.seed, samples);
    if (!save_samples(out, samples)) {
        fprintf(stderr, "[错误] worker 写样本失败: %s\n", out.c_str());
        return 1;
    }
    const std::string stf = a.str("stats");
    if (!stf.empty()) {
        FILE* f = fopen(stf.c_str(), "w");
        if (!f) {
            fprintf(stderr, "[错误] worker 写统计失败: %s\n", stf.c_str());
            return 1;
        }
        fprintf(f, "%d %d %d %d %d %d %lld\n", sp.games, sp.black_wins, sp.white_wins, sp.draws,
                sp.resigned, sp.truncated, (long long)sp.plies);
        fclose(f);
    }
    return 0;
}

// 与 PYTHON/probe_workers_cpu.py 完全同口径的并行强度测试：
// 每个线程固定走 --moves 手（同一组 5 手开局、temp=1、n_sim 固定），
// 统计聚合手/秒，用来和 Python 的 9.2 / 14.3 / 29.6 / 41.5 / 46.1 手/秒对比。
static const int kOpenings[8][5][2] = {
    {{9, 9}, {9, 10}, {10, 9}, {8, 8}, {10, 10}},   {{9, 9}, {10, 10}, {9, 10}, {10, 9}, {8, 9}},
    {{9, 9}, {9, 11}, {10, 10}, {8, 9}, {11, 9}},   {{9, 9}, {8, 10}, {10, 9}, {9, 11}, {11, 10}},
    {{9, 9}, {11, 9}, {9, 10}, {10, 11}, {8, 8}},   {{9, 9}, {10, 8}, {9, 11}, {11, 10}, {8, 9}},
    {{9, 9}, {8, 9}, {10, 10}, {9, 8}, {11, 11}},   {{9, 9}, {10, 11}, {8, 10}, {11, 9}, {9, 12}},
};

// 纯 CPU 的假评估器：用于把「搜索/线程」问题与「网络/评估器」问题分开定位
struct DummyEval : Evaluator {
    void evaluate_batch(const float* states, int batch, float* policies, float* values) override {
        (void)states;
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < N_CELLS; ++j) policies[(size_t)i * N_CELLS + j] = 1.0f / N_CELLS;
            values[i] = 0.0f;
        }
    }
};

int cmd_probe(const Args& a) {
    TrainConfig cfg;
    apply_common(a, cfg);
    const std::string netpath = a.str("net");
    const bool dummy = a.has("dummy");
    if (netpath.empty() && !dummy) {
        printf("需要 --net（或用 --dummy 跑纯 CPU 假网络）\n");
        return 2;
    }
    TorchNet net(cfg.arch, cfg.cuda, cfg.seed);
    std::string err;
    if (!dummy && !load_net(net, netpath, &err)) {
        printf("[错误] 载入权重失败: %s\n", err.c_str());
        return 1;
    }
    TorchEvaluator ev(&net);
    DummyEval dummy_ev;
    Evaluator* base = dummy ? (Evaluator*)&dummy_ev : (Evaluator*)&ev;
    NetService svc(base, cfg.eval_batch, cfg.eval_window_ms, a.num("eval-threads", 2));
    const bool use_svc = cfg.workers > 1 && !a.has("no-service");
    if (use_svc) svc.start();
    Evaluator* e = use_svc ? (Evaluator*)&svc : base;

    const int moves_per = a.num("moves", 240);
    const int threads = a.num("workers", 1);
    std::atomic<int64_t> done{0}, states{0}, games{0}, bad{0};
    const double t0 = now_ms();
    std::vector<std::thread> ths;
    for (int t = 0; t < threads; ++t) {
        ths.emplace_back([&, t]() {
            MCTS m(e, make_mcts_options(cfg), cfg.seed + (uint64_t)t * 7919);
            std::vector<float> pol(N_CELLS);
            auto fresh = [&](Pos& b) {
                b = Pos();
                for (int k = 0; k < 5; ++k) b.place(kOpenings[t % 8][k][0], kOpenings[t % 8][k][1], b.player);
                m.reset_tree();
            };
            auto play = [&](Pos& b, int* st_prev, int64_t* st_sum) {
                const int mv = m.sample_move_with_policy(b, pol, 1.0, cfg.n_sim, cfg.noise_alpha,
                                                         cfg.noise_eps);
                const int64_t cur = (int64_t)m.collect_stats().states_evaluated;
                *st_sum += cur - *st_prev;
                *st_prev = (int)cur;
                if (mv < 0) {
                    ++bad;
                    return false;
                }
                const bool win = b.place(mv % SIZE, mv / SIZE, b.player);
                m.set_root_move((int16_t)mv);
                if (win || b.stones >= N_CELLS) {
                    ++games;
                    fresh(b);
                    return false;
                }
                return true;
            };
            Pos b;
            fresh(b);
            int st_prev = 0;
            int64_t st_sum = 0;
            for (int w = 0; w < 3; ++w) play(b, &st_prev, &st_sum);  // 与 Python 的 WARMUP=3 对齐
            int n = 0;
            while (n < moves_per) {
                play(b, &st_prev, &st_sum);
                ++n;
                ++done;
            }
            states += st_sum;
        });
    }
    for (auto& th : ths) th.join();
    const double dt = now_ms() - t0;
    const NetService::Stats ns = svc.stats();
    if (use_svc) svc.stop();
    printf("probe: %d 线程，%lld 局，%lld 手（无着 %lld），%.1f s → 聚合 %.2f 手/s（单线程 %.2f 手/s）\n",
           threads, (long long)games.load(), (long long)done.load(), (long long)bad.load(),
           dt / 1000.0,
           dt > 0 ? (double)done.load() * 1000.0 / dt : 0.0,
           dt > 0 ? (double)done.load() * 1000.0 / dt / (double)threads : 0.0);
    printf("       状态 %lld（%.0f 状态/手），%.0f 状态/s，前向 %llu 次（%.1f 状态/次）\n",
           (long long)states.load(),
           done.load() ? (double)states.load() / (double)done.load() : 0.0,
           dt > 0 ? (double)states.load() * 1000.0 / dt : 0.0, (unsigned long long)ns.batches,
           ns.batches ? (double)ns.states / (double)ns.batches : 0.0);
    return 0;
}

int cmd_gate(const Args& a) {
    TrainConfig cfg;
    apply_common(a, cfg);
    const std::string cand = a.str("cand"), base = a.str("best");
    if (cand.empty() || base.empty()) {
        printf("需要 --cand 与 --best\n");
        return 2;
    }
    TorchNet nc(cfg.arch, cfg.cuda, cfg.seed), nb(cfg.arch, cfg.cuda, cfg.seed + 1);
    std::string err;
    if (!load_net(nc, cand, &err) || !load_net(nb, base, &err)) {
        printf("[错误] 载入权重失败: %s\n", err.c_str());
        return 1;
    }
    TorchEvaluator ec(&nc), eb(&nb);
    GateOptions go;
    go.games = a.num("eval-games", 20);
    go.threshold = a.dbl("gate-threshold", 0.55);
    go.ci_ll = a.dbl("gate-ci-ll", 0.0);
    go.n_sim = cfg.n_sim;
    go.c_puct = cfg.c_puct;
    go.heuristic_weight = cfg.heuristic_weight;
    go.policy_weight = cfg.policy_weight;
    go.max_children = cfg.max_children;
    go.root_max_children = cfg.root_max_children;
    go.vcf = cfg.vcf;
    go.vcf_root = a.has("vcf-root") ? true : cfg.vcf_root;
    go.vcf_max_ply = cfg.vcf_max_ply;
    go.vcf_max_nodes = cfg.vcf_max_nodes;
    go.guarantee_rush = cfg.guarantee_rush;
    // 强度 A/B 用：同一个 --best/--cand 网络，只让候选侧换搜索配置。
    // 组合 --no-vcf --cand-vcf 即「候选开 VCF、对手关 VCF」。
    go.cand_vcf = a.has("cand-no-vcf") ? false : (a.has("cand-vcf") ? true : cfg.vcf);
    // 根节点 VCF 的候选侧开关：--cand-vcf-root 只让候选开，对手保持关闭 ⇒ 同一份权重、
    // 只差这个开关的强度 A/B（这正是判定「VCF 到底有没有用」最干净的测法）。
    go.cand_vcf_root = a.has("cand-no-vcf-root") ? false
                     : (a.has("cand-vcf-root") ? true : cfg.vcf_root);
    go.cand_guarantee_rush =
        a.has("cand-no-guarantee-rush") ? false : (a.has("cand-guarantee-rush") ? true : cfg.guarantee_rush);
    go.cand_sims = a.num("cand-sims", 0);
    GateResult r;
    {
        NetService sc(&ec, cfg.eval_batch, cfg.eval_window_ms, cfg.eval_threads);
        NetService sb(&eb, cfg.eval_batch, cfg.eval_window_ms, cfg.eval_threads);
        // 必须启动：门控线程再少，前向也要走评估线程
        sc.start();
        sb.start();
        r = run_gate_threads(&sc, &sb, go, cfg.seed, std::max(1, cfg.eval_workers));
        sc.stop();
        sb.stop();
    }
    printf("gate: %d 局 %.1f s，候选 %d 胜 %d 负 %d 平（黑 %d/%d，白 %d/%d），胜率 %.3f\n",
           r.games, r.elapsed_ms / 1000.0, r.wins, r.losses, r.draws, r.wins_black, r.games_black,
           r.wins_white, r.games_white, r.win_rate);
    printf("      Wilson[%.3f, %.3f] → %s\n", r.wilson_lo, r.wilson_hi,
           r.accepted ? "采纳" : "拒绝");
    return 0;
}

// elo：把「某一代比某一代强多少」量成一条曲线。
// 连边策略（保证图连通且成本可控）：
//   ① 链式相邻：g_i vs g_{i+1}（局部斜率，误差不累积）
//   ② 每代 vs 锚点：g_i vs g_anchor（全局位置，把锚点钉死在 0）
// 只用 play_gate_game 同一套对局逻辑，所以和门控口径一致（同 sims、同配对开局）。
int cmd_elo(const Args& a) {
    TrainConfig cfg;
    apply_common(a, cfg);
    const std::string dir = a.str("dir", "model/run_v2");
    const std::string gens_s = a.str("gens", "");
    const int games = a.num("eval-games", 24);
    const bool with_ref = !a.has("no-ref-edge");

    std::vector<int> gens;
    {
        std::stringstream ss(gens_s);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            const int g = std::atoi(tok.c_str());
            if (g > 0) gens.push_back(g);
        }
    }
    // --refit-only 不需要 --gens（它不打对局，只是重拟合边库），所以这里放行
    if (gens.empty() && !a.has("refit-only")) {
        printf("[错误] 需要 --gens 4,12,20,24,34,43\n");
        return 2;
    }
    std::sort(gens.begin(), gens.end());
    gens.erase(std::unique(gens.begin(), gens.end()), gens.end());

    // 【2026-10-31 重构】边库模式：
    //   打完的边**永久保存**到 <dir>/elo_edges.jsonl，然后用**全部边**重新全局拟合。
    //   旧行为是「只拟合本次窗口、再平移接到旧曲线上」，实测新旧差值散在 +60~+218，
    //   根本不存在一个常数偏移，等于在猜；而且每接一段累积一次误差。
    //   边库模式下没有对齐这一步 —— 后打的边会反过来校正老点的位置。
    //   于是「跨段标定」= 打一条长跨边（如 --gens 85,100 --eval-games 400），
    //   加进去全局拟合自动把中间那些链式累积误差压下去，不需要任何特殊代码。
    const std::string store_path = a.str("store", dir + "/elo_edges.jsonl");
    const bool use_store = !a.has("no-store");
    std::vector<EloStoreEdge> store = use_store ? elo_store_load(store_path)
                                                : std::vector<EloStoreEdge>();

    // --refit-only：不打任何对局，只用边库重拟合一遍并重写 elo.json。
    // 三个用途：① 把旧 elo.json 的边导入边库后重建曲线；② 手改边库（删掉可疑边）后重算；
    // ③ 换一个 bootstrap 精度重算置信区间。
    if (a.has("refit-only")) {
        if (store.empty()) {
            printf("[错误] 边库 %s 是空的，没有可拟合的边\n", store_path.c_str());
            return 2;
        }
        const int ag = a.num("anchor", 0);
        EloCurve cur = elo_curve_from_store(dir, store, ag, a.num("bootstrap", 200), cfg.seed);
        if (cur.models.empty()) {
            printf("[错误] 拟合后没有可用版本（权重文件都缺了？）\n");
            return 2;
        }
        printf("\n%-10s %8s %18s %6s\n", "版本", "Elo", "95% 区间", "对局");
        for (const auto& m : cur.models)
            printf("%-10s %8.1f  [%7.1f, %7.1f] %6d\n", m.name.c_str(), m.elo, m.lo, m.hi, m.games);
        printf("锚点：%s = 0 Elo，%zu 个版本 / %d 条边\n",
               cur.models[(size_t)cur.anchor].name.c_str(), cur.models.size(), cur.store_edges);
        {
            const EloStoreEdge* best = nullptr;
            int best_span = -1;
            for (const auto& e : store) {
                const int ga = elo_gen_of_name(e.a), gb = elo_gen_of_name(e.b);
                if (ga < 0 || gb < 0) continue;
                if (std::abs(ga - gb) > best_span) {
                    best_span = std::abs(ga - gb);
                    best = &e;
                }
            }
            const int newest = cur.models.back().gen;
            if (best)
                printf("最长直边：%s–%s（跨 %d 代，%d 局）。当前最新已评 gen_%03d；"
                       "跨段标定建议每 15 代一条（400 局），否则链式累积误差会越接越长。\n",
                       best->a.c_str(), best->b.c_str(), best_span, best->games, newest);
        }
        const std::string out = a.str("out", dir + "/elo.json");
        std::ofstream of(out, std::ios::binary);
        if (!of.good()) {
            printf("[警告] 写不进 %s\n", out.c_str());
            return 1;
        }
        const std::string js = elo_json(cur.models, cur.edges, cur.anchor, stamp_now());
        of.write(js.data(), (std::streamsize)js.size());
        of.close();
        printf("已写入 %s\n", out.c_str());
        return 0;
    }

    // 锚点：显式 --anchor 优先；否则 0 = 「边库里最老的版本」当 0 点。
    // 为什么不再用"最新已评代"：那会让曲线整体为负、越往后越接近 0，看着像在退步。
    // 也不再用固定的 gen_010：远锚点在**旧的分段拼接**里才会饱和（它要靠对局提供刻度），
    // 边库模式是全量拟合，锚点只是一个平移自由度，用最老代最直观 —— 向上就是变强。
    int anchor_gen = a.num("anchor", 0);
    if (anchor_gen <= 0 && gens.empty()) anchor_gen = 0;

    EloRunOptions ro;
    ro.games = games;
    ro.opening_plies = a.num("opening-plies", 2);
    ro.max_ref_span = a.num("max-ref-span", 30);
    ro.with_ref = with_ref;
    ro.verbose = true;
    // --skip-existing N：边库里这一对已经打过 N 局以上就跳过。看板按钮每次都送同一个
    // 窗口（滚动更新），不跳过会反复重打同样的对局——局数白白翻倍，曲线却没更准。
    ro.skip_if_games = a.num("skip-existing", 0);
    ro.existing = &store;

    const double t_all = now_ms();
    std::vector<EloStoreEdge> fresh = elo_play(cfg, dir, gens, anchor_gen, ro);
    if (fresh.empty()) {
        printf("[elo] 本次没有打出任何对局（权重缺失或版本不足 2 个）\n");
        if (!use_store) return 2;
    }

    const int n_new = elo_store_merge(store, fresh);
    if (use_store) {
        if (!elo_store_save(store_path, store)) {
            printf("[警告] 写不进边库 %s，本次对局不会进入曲线\n", store_path.c_str());
        } else {
            printf("[elo] 边库 %s：新增 %d 条，共 %zu 条边\n", store_path.c_str(), n_new,
                   store.size());
        }
    } else {
        printf("[elo] --no-store：本次 %zu 条边只用于当场拟合，不写入边库\n", fresh.size());
    }

    // 全量重拟合（边库里所有边一起）
    EloCurve cur = elo_curve_from_store(dir, store, anchor_gen, a.num("bootstrap", 200), cfg.seed);
    if (cur.models.empty()) {
        printf("[错误] 拟合后没有可用版本\n");
        return 2;
    }
    if (cur.dropped_missing > 0)
        printf("[elo] 边库里 %d 个版本的权重文件已不存在，已剔除\n", cur.dropped_missing);
    if (cur.dropped_disconnected > 0)
        printf("[elo] 边库里 %d 个版本与主体不连通，已剔除\n", cur.dropped_disconnected);

    printf("\n%-10s %8s %18s %6s\n", "版本", "Elo", "95% 区间", "对局");
    for (const auto& m : cur.models) {
        printf("%-10s %8.1f  [%7.1f, %7.1f] %6d\n", m.name.c_str(), m.elo, m.lo, m.hi, m.games);
    }
    printf("锚点：%s = 0 Elo，%zu 个版本 / %d 条边，总耗时 %.1f s\n",
           cur.models[(size_t)cur.anchor].name.c_str(), cur.models.size(), cur.store_edges,
           (now_ms() - t_all) / 1000.0);

    // --no-store 是一次性测量（比如临时验证某两代），默认**不**覆盖曲线：
    // 那份拟合只含本次几条边，写回去会把整条曲线冲掉。要写必须显式给 --out。
    const std::string out = a.str("out", use_store ? (dir + "/elo.json") : "");
    if (out.empty()) {
        printf("[elo] 未写 elo.json（--no-store 且未给 --out）\n");
        return 0;
    }
    std::ofstream of(out, std::ios::binary);
    if (!of.good()) {
        printf("[警告] 写不进 %s\n", out.c_str());
        return 1;
    }
    const std::string js = elo_json(cur.models, cur.edges, cur.anchor, stamp_now());
    of.write(js.data(), (std::streamsize)js.size());
    of.close();
    printf("已写入 %s\n", out.c_str());
    return 0;
}
int cmd_loop(const Args& a) {
    LoopOptions lo;
    apply_common(a, lo.cfg);
    // 默认把 C++ 侧的模型写在 C++\model 下（与 Python 的 models/ 分开，两套版本不混用）
    lo.dir = a.str("dir", "model/run");
    lo.init_weights = a.str("init");
    lo.gens = a.num("gens", 10);
    if (a.has("resume")) lo.resume_dir = a.str("resume");
    if (lo.cfg.n_sim < 4 * lo.cfg.batch_size_mcts) {
        printf("[警告] --sims %d 相对 --batch-mcts %d 太小：根的子节点会被少数几次模拟分摊，"
               "策略目标会退化（CE 卡在 log(361)）\n",
               lo.cfg.n_sim, lo.cfg.batch_size_mcts);
    }
    return run_training_loop(lo);
}

// 把默认配方以 JSON 打给网页控制面板。
// 为什么要这个命令：网页上要显示「C++ 引擎的参数」，而那些默认值写在 config.h 里。
// 让网页自己抄一份数字，迟早会和 config.h 对不上（改了一处忘另一处），
// 所以默认值只留一个权威来源：这里直接读 TrainConfig 的默认构造。
int cmd_defaults(const Args&) {
    const TrainConfig c;
    printf("{");
    printf("\"arch_blocks\":%d,\"arch_channels\":%d,\"arch_value_hidden\":%d,",
           c.arch.blocks, c.arch.channels, c.arch.value_hidden);
    printf("\"workers\":%d,\"games\":%d,\"sims\":%d,\"batch_mcts\":%d,",
           c.workers, c.games, c.n_sim, c.batch_size_mcts);
    printf("\"c_puct\":%.6g,\"max_children\":%d,\"root_max_children\":%d,",
           c.c_puct, c.max_children, c.root_max_children);
    printf("\"hw\":%.6g,\"pw\":%.6g,\"eval_batch\":%d,\"eval_threads\":%d,\"procs\":%d,",
           c.heuristic_weight, c.policy_weight, c.eval_batch, c.eval_threads, c.mp_procs);
    printf("\"buffer_gens\":%d,\"buffer_size\":%llu,\"half_life\":%.6g,\"fresh_ratio\":%.6g,",
           c.buffer_gens, (unsigned long long)c.buffer_size, c.half_life, c.fresh_ratio);
    printf("\"batch\":%d,\"min_steps\":%d,\"lr\":%.6g,\"clip\":%.6g,\"ema_decay\":%.6g,",
           c.batch_size, c.min_steps, c.lr, c.clip_norm, c.ema_decay);
    printf("\"amp\":%s,\"eval_workers\":%d,\"keep_gens\":%d,\"keep_buffer_gens\":%d,",
           c.amp ? "true" : "false", c.eval_workers, c.keep_gens, c.keep_buffer_gens);
    printf("\"eval_games\":%d,\"eval_every\":%d,\"gate_threshold\":%.6g,\"seed\":%llu,",
           c.eval_games, c.eval_every, c.gate_threshold, (unsigned long long)c.seed);
    // 配方编号与可测量性相关的新字段：看板切到 C++ 时用这份 JSON 填表，缺了它们
    // 就会把「配方 3」悄悄退回配方 1（和 160/320 那次事故同一个形状）。
    printf("\"recipe\":%d,\"lr_gen_decay\":%.6g,\"lr_floor\":%.6g,\"opening_plies\":%d,",
           c.recipe_id, c.lr_gen_decay, c.lr_floor, c.gate_opening_plies);
    // Elo 曲线自动更新（边库模式）。看板切到 C++ 时用这份 JSON 填表；缺了它们
    // 就会把 --elo-every 悄悄退回 0（= 不自动跑），又变成"得手点按钮"。
    printf("\"elo_every\":%d,\"elo_window\":%d,\"elo_games\":%d,",
           c.elo_every, c.elo_window, c.elo_games);
    printf("\"elo_calib_every\":%d,\"elo_calib_games\":%d,\"elo_max_ref_span\":%d,",
           c.elo_calib_every, c.elo_calib_games, c.elo_max_ref_span);
    printf("\"elo_on_accept\":%d,\"elo_every_accepts\":%d,",
           c.elo_on_accept ? 1 : 0, c.elo_every_accepts);
    printf("\"gens\":10}");
    printf("\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // stdout 被重定向到文件（看板用 Popen 把 loop 的输出写进 loop_stdout.log）时，
    // C 运行库默认用 4KB 全缓冲 ⇒ 面板的「日志尾巴」要等好几代才更新一次。
    // 这里改成**不缓冲**：每行立刻落盘，看板才能实时看到「=== gen k/N === / [自对弈] …」。
    // 注意别用 _IOLBF + size 0：MSVC 在这个组合下会走 invalid parameter 直接终止
    // （实测退出码 0xC0000409，且没有任何输出，看起来像"编译好的 exe 坏了"）。
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    const Args a = parse(argc, argv, 2);
    if (a.has("nice")) {
        // 与游戏/其它 GPU 程序共存：降到 BelowNormal，只抢空闲算力
        if (set_below_normal_priority()) printf("[nice] 进程优先级已降为 BelowNormal\n");
    }
    if (cmd == "bench") return cmd_bench(a);
    if (cmd == "vcfsuite") return cmd_vcfsuite(a);
    if (cmd == "selfplay") return cmd_selfplay(a);
    if (cmd == "worker") return cmd_worker(a);
    if (cmd == "gate") return cmd_gate(a);
    if (cmd == "play") return cmd_play(a);
    if (cmd == "analyze") return cmd_analyze(a);
    if (cmd == "probe") return cmd_probe(a);
    if (cmd == "elo") return cmd_elo(a);
    if (cmd == "loop") return cmd_loop(a);
    if (cmd == "defaults") return cmd_defaults(a);
    usage();
    return 2;
}
