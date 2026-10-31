// pipeline.cpp — 一代的完整流程
#include "pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "batcher.h"
#include "eval_torch.h"
#include "elo_run.h"
#include "gate.h"
#include "sysutil.h"
#include "trainer.h"

namespace fs = std::filesystem;

namespace gk {
namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

// 本地时间戳（写 run.json 用；只给人看，不参与算法）
std::string stamp_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

// 从 "…/gen_007.bin" 里取出 7（用于记录"这个目录是从哪个全局代数接过来的"）
int gen_from_path(const std::string& p) {
    const size_t slash = p.find_last_of("/\\");
    const std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
    if (base.rfind("gen_", 0) != 0) return 0;
    try {
        return std::stoi(base.substr(4));
    } catch (...) {
        return 0;
    }
}

// 给每个运行目录写一份 run.json（人类可读的"这个目录是什么"）。
// 为什么需要：目录名（run_v2 / run_v4amp / broken_amp_do_not_use）本身说不清
// 谁该继续训、哪一代是接过来的；而 gen 编号在每个目录里都从 1 开始，
// 只看 gen_004 无法知道它是项目的第 4 代还是第 11 代。
void write_run_manifest(const std::string& dir, const TrainConfig& cfg, const LoopOptions& opt,
                        int gen_offset) {
    const std::string path = dir + "/run.json";
    std::error_code ec;
    if (fs::exists(path, ec)) return;  // 只在第一次创建时写，之后不覆盖（人可能手工改过）
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f,
                 "{\"created\":\"%s\",\"arch\":\"%dx%dx%d\","
                 "\"init_from\":\"%s\",\"gen_offset\":%d,"
                 "\"recipe_id\":%d,"
                 "\"recipe\":{\"games\":%d,\"sims\":%d,\"min_steps\":%d,\"lr\":%.6g,"
                 "\"fresh_ratio\":%.6g,\"buffer_gens\":%d,\"half_life\":%.6g,"
                 "\"eval_games\":%d,\"lr_gen_decay\":%.6g,\"opening_plies\":%d}}\n",
                 stamp_now().c_str(), cfg.arch.blocks, cfg.arch.channels, cfg.arch.value_hidden,
                 opt.init_weights.empty() ? "" : opt.init_weights.c_str(), gen_offset, cfg.recipe_id,
                 cfg.games, cfg.n_sim, cfg.min_steps, cfg.lr, cfg.fresh_ratio, cfg.buffer_gens,
                 cfg.half_life, cfg.eval_games, cfg.lr_gen_decay, cfg.gate_opening_plies);
    std::fclose(f);
}

// 把 dir 里形如 gen_007.bin 的最大编号找出来
int find_last_gen(const std::string& dir) {
    int best = 0;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return 0;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("gen_", 0) != 0 || e.path().extension() != ".bin") continue;
        try {
            const int g = std::stoi(name.substr(4, name.size() - 8));
            if (g > best) best = g;
        } catch (...) {
        }
    }
    return best;
}

// 读 train_log.jsonl 最后一行的 lr / recipe_id。用途：同一个配方内续跑时接着上一代退火后的
// lr，而不是把 --lr 重新套上去。没有这一条，配方 3 的「每代 ×0.97」只要重启一次就断，
// 表现为 train_log 里 lr 每隔十来代跳回 0.000291（gen 49、59 就是这样）。
bool read_last_log_lr(const std::string& dir, double* out_lr, int* out_recipe) {
    std::ifstream f((dir + "/train_log.jsonl").c_str());
    if (!f) return false;
    std::string line, last;
    while (std::getline(f, line)) {
        if (!line.empty()) last = line;
    }
    if (last.empty()) return false;
    double lr = 0.0;
    int rid = 0;
    const size_t pl = last.find("\"lr\":");
    if (pl == std::string::npos) return false;
    lr = std::strtod(last.c_str() + pl + 5, nullptr);
    const size_t pr = last.find("\"recipe_id\":");
    if (pr != std::string::npos) rid = std::atoi(last.c_str() + pr + 12);
    if (lr <= 0.0) return false;
    if (out_lr) *out_lr = lr;
    if (out_recipe) *out_recipe = rid;
    return true;
}

}  // namespace

MCTSOptions make_mcts_options(const TrainConfig& cfg) {
    MCTSOptions o;
    o.n_sim = cfg.n_sim;
    o.c_puct = cfg.c_puct;
    o.virtual_loss = cfg.virtual_loss;
    o.batch_size = cfg.batch_size_mcts;
    o.max_children = cfg.max_children;
    o.root_max_children = cfg.root_max_children;
    o.heuristic_weight = cfg.heuristic_weight;
    o.policy_weight = cfg.policy_weight;
    o.use_heuristics = true;
    o.vcf = cfg.vcf;
    o.vcf_max_ply = cfg.vcf_max_ply;
    o.vcf_max_nodes = cfg.vcf_max_nodes;
    o.guarantee_rush = cfg.guarantee_rush;
    return o;
}

PlayOptions make_play_options(const TrainConfig& cfg) {
    PlayOptions o;
    o.n_sim = cfg.n_sim;
    o.c_puct = cfg.c_puct;
    o.max_children = cfg.max_children;
    o.root_max_children = cfg.root_max_children;
    o.heuristic_weight = cfg.heuristic_weight;
    o.policy_weight = cfg.policy_weight;
    o.use_heuristics = true;
    o.vcf = cfg.vcf;
    o.vcf_max_ply = cfg.vcf_max_ply;
    o.vcf_max_nodes = cfg.vcf_max_nodes;
    o.guarantee_rush = cfg.guarantee_rush;
    o.temp_open = cfg.temp_open;
    o.temp_mid = cfg.temp_mid;
    o.temp_threshold1 = cfg.temp_threshold1;
    o.temp_threshold2 = cfg.temp_threshold2;
    o.noise_alpha = cfg.noise_alpha;
    o.noise_eps = cfg.noise_eps;
    o.sp_open_plies = cfg.sp_open_plies;
    o.resign_threshold = cfg.resign_threshold;
    o.resign_plies = cfg.resign_plies;
    o.max_plies = cfg.max_plies;
    return o;
}

SelfPlayStats collect_games(Evaluator* ev, const TrainConfig& cfg, int games, uint64_t seed,
                            std::vector<Sample>& out) {
    SelfPlayStats st;
    if (games <= 0) return st;
    const int nthreads = std::max(1, std::min(cfg.workers, games));

    std::vector<std::vector<Sample>> per_game((size_t)games);
    std::vector<GameStats> per_stats((size_t)games);
    std::atomic<int> next_game{0};
    std::mutex stat_mu;

    const double t0 = now_ms();
    auto worker = [&](int tid) {
        const PlayOptions po = make_play_options(cfg);
        while (true) {
            const int g = next_game.fetch_add(1);
            if (g >= games) break;
            std::vector<Sample> sv;
            GameStats gs;
            const uint64_t gseed = seed * 1000003ULL + (uint64_t)g * 7919ULL + (uint64_t)tid * 13ULL;
            play_game(ev, po, sv, &gs, gseed);
            per_game[(size_t)g] = std::move(sv);
            per_stats[(size_t)g] = gs;
        }
    };

    std::vector<std::thread> ths;
    ths.reserve((size_t)nthreads);
    for (int i = 0; i < nthreads; ++i) ths.emplace_back(worker, i);
    for (auto& t : ths) t.join();
    st.elapsed_ms = now_ms() - t0;

    for (int g = 0; g < games; ++g) {
        const GameStats& gs = per_stats[(size_t)g];
        ++st.games;
        if (gs.winner == 1) ++st.black_wins;
        else if (gs.winner == 2) ++st.white_wins;
        else ++st.draws;
        if (gs.resigned) ++st.resigned;
        if (gs.truncated) ++st.truncated;
        st.plies += gs.plies;
        st.evaluations += gs.evaluations;
        for (auto& s : per_game[(size_t)g]) out.push_back(std::move(s));
    }
    st.samples = (int)out.size();
    return st;
}

// ---------------------------------------------------------------------------
// 多进程自对弈
// ---------------------------------------------------------------------------
SelfPlayStats collect_games_mp(const TrainConfig& cfg, const std::string& weights_path, int games,
                               uint64_t seed, const std::string& tmpdir,
                               std::vector<Sample>& out) {
    SelfPlayStats st;
    if (games <= 0) return st;
    std::error_code ec;
    fs::create_directories(tmpdir, ec);

    const int procs = std::max(1, std::min(cfg.mp_procs > 0 ? cfg.mp_procs : 8, games));
    const int per = games / procs;
    const int extra = games % procs;

    std::vector<std::string> cmds;
    std::vector<std::string> sample_files;
    cmds.reserve((size_t)procs);
    for (int i = 0; i < procs; ++i) {
        const int n = per + (i < extra ? 1 : 0);
        if (n <= 0) continue;
        const std::string sf = fmt("%s/w%d.bin", tmpdir.c_str(), i);
        const std::string tf = fmt("%s/w%d.txt", tmpdir.c_str(), i);
        sample_files.push_back(sf);
        cmds.push_back(fmt(
            "worker --net \"%s\" --games %d --seed %llu --workers 1 --out \"%s\" --stats \"%s\" "
            "--blocks %d --channels %d --sims %d --batch-mcts %d --c-puct %.4f --max-children %d "
            "--root-max-children %d --hw %.4f --pw %.4f --noise-alpha %.4f --noise-eps %.4f "
            "--resign-threshold %.4f --resign-plies %d --eval-batch %d%s",
            weights_path.c_str(), n, (unsigned long long)(seed + (uint64_t)i * 1013ULL),
            sf.c_str(), tf.c_str(), cfg.arch.blocks, cfg.arch.channels, cfg.n_sim,
            cfg.batch_size_mcts, cfg.c_puct, cfg.max_children, cfg.root_max_children,
            cfg.heuristic_weight, cfg.policy_weight, cfg.noise_alpha, cfg.noise_eps,
            cfg.resign_threshold, cfg.resign_plies, cfg.eval_batch, cfg.cuda ? "" : " --cpu"));
    }

    const double t0 = now_ms();
    std::vector<int> codes;
    const bool ok = run_parallel(cmds, &codes);
    st.elapsed_ms = now_ms() - t0;
    if (!ok) {
        std::fprintf(stderr, "[错误] 有子进程失败，退出码：");
        for (int c : codes) std::fprintf(stderr, "%d ", c);
        std::fprintf(stderr, "\n");
        return st;  // games=0 表示这一代自对弈失败
    }

    for (const auto& sf : sample_files) {
        std::vector<Sample> part;
        if (!load_samples(sf, part)) {
            std::fprintf(stderr, "[错误] 读取子进程样本失败: %s\n", sf.c_str());
            continue;
        }
        for (auto& s : part) out.push_back(std::move(s));
    }
    // 合并子进程的统计行：games black white draws resigned truncated plies
    for (int i = 0; i < (int)sample_files.size(); ++i) {
        const std::string tf = fmt("%s/w%d.txt", tmpdir.c_str(), i);
        std::ifstream f(tf);
        if (!f) continue;
        int g = 0, b = 0, w = 0, d = 0, r = 0, tr = 0;
        long long p = 0;
        if (f >> g >> b >> w >> d >> r >> tr >> p) {
            st.games += g;
            st.black_wins += b;
            st.white_wins += w;
            st.draws += d;
            st.resigned += r;
            st.truncated += tr;
            st.plies += p;
        }
    }
    st.samples = (int)out.size();

    // 清理子进程的临时文件（样本已读进内存）
    for (const auto& sf : sample_files) {
        std::error_code ec2;
        fs::remove(sf, ec2);
    }
    for (int i = 0; i < (int)sample_files.size(); ++i) {
        std::error_code ec2;
        fs::remove(fmt("%s/w%d.txt", tmpdir.c_str(), i), ec2);
    }
    std::error_code ec3;
    fs::remove(tmpdir, ec3);  // 目录空了才删得掉；非空会失败，无害
    return st;
}

int run_training_loop(const LoopOptions& opt_in) {
    LoopOptions opt = opt_in;
    TrainConfig& cfg = opt.cfg;

    std::error_code ec;
    fs::create_directories(opt.dir, ec);
    fs::create_directories(opt.dir + "/buffer", ec);

    // ---- 网络 ----
    TorchNet net(cfg.arch, cfg.cuda, cfg.seed);
    TorchNet ema_net(cfg.arch, cfg.cuda, cfg.seed + 1);
    TorchNet best_net(cfg.arch, cfg.cuda, cfg.seed + 2);
    TorchEvaluator ev_net(&net), ev_ema(&ema_net), ev_best(&best_net);

    std::string err;
    const std::string resume_dir = opt.resume_dir.empty() ? opt.dir : opt.resume_dir;
    int start_gen = 1;
    int last_gen = find_last_gen(resume_dir);
    if (last_gen > 0) {
        const std::string p = resume_dir + fmt("/gen_%03d.bin", last_gen);
        Arch a;
        std::vector<std::pair<std::string, torch::Tensor>> items;
        if (load_named_tensors_file(p, a, items)) {
            net.load_named_tensors(items, true, &err);
            printf("[续训] 载入 %s（arch %d×%d）\n", p.c_str(), a.blocks, a.channels);
            start_gen = last_gen + 1;
        }
    } else if (!opt.init_weights.empty()) {
        Arch a;
        std::vector<std::pair<std::string, torch::Tensor>> items;
        if (!load_named_tensors_file(opt.init_weights, a, items)) {
            printf("[错误] 载入初始权重失败: %s\n", opt.init_weights.c_str());
            return 1;
        }
        if (!(a == cfg.arch)) {
            printf("[错误] 初始权重架构 %d×%d 与配置 %d×%d 不一致\n", a.blocks, a.channels,
                   cfg.arch.blocks, cfg.arch.channels);
            return 1;
        }
        net.load_named_tensors(items, true, &err);
        printf("[初始化] 载入 %s（arch %d×%d×%d）\n", opt.init_weights.c_str(), a.blocks,
               a.channels, a.value_hidden);
    } else {
        printf("[初始化] 随机权重 %d×%d×%d\n", cfg.arch.blocks, cfg.arch.channels,
               cfg.arch.value_hidden);
        // 找不到 gen_*.bin/best.bin 时是**从零**开始：这条提示很关键，否则很容易以为
        // 自己接着主线在练，其实把之前十几代的成果扔掉了。
        printf("[提示] 目录 %s 里没有可续训的权重（gen_*.bin/best.bin），本次从**随机权重**开始。\n"
               "       要续训主线请用 --dir %s。\n",
               opt.dir.c_str(), "model/run_v2");
    }

    // best 基线：优先 best.bin，否则用当前权重
    bool have_best = false;
    {
        Arch a;
        std::vector<std::pair<std::string, torch::Tensor>> items;
        const std::string bp = resume_dir + "/best.bin";
        if (fs::exists(bp, ec) && load_named_tensors_file(bp, a, items)) {
            best_net.load_named_tensors(items, true, &err);
            best_net.to_device(cfg.cuda);
            have_best = true;
            printf("[初始化] best.bin 已载入\n");
        }
    }
    net.to_device(cfg.cuda);
    ema_net.to_device(cfg.cuda);
    best_net.to_device(cfg.cuda);

    // 目录自述文件：只在第一次（目录里还没有 run.json）时写
    {
        int gen_offset = 0;
        if (!opt.init_weights.empty()) {
            gen_offset = gen_from_path(opt.init_weights);
        } else if (resume_dir != opt.dir) {
            gen_offset = find_last_gen(resume_dir);
        }
        write_run_manifest(opt.dir, cfg, opt, gen_offset);
    }

    // ---- 回放池 ----
    ReplayBuffer buf(cfg.buffer_size, cfg.buffer_gens, cfg.half_life);
    {
        std::vector<std::string> files;
        if (fs::is_directory(resume_dir + "/buffer", ec)) {
            for (const auto& e : fs::directory_iterator(resume_dir + "/buffer", ec)) {
                if (e.path().extension() == ".bin") files.push_back(e.path().string());
            }
        }
        std::sort(files.begin(), files.end());
        for (const auto& f : files) {
            std::vector<Sample> sv;
            if (load_samples(f, sv) && !sv.empty()) {
                const std::string name = fs::path(f).stem().string();
                int g = 0;
                try {
                    g = std::stoi(name.substr(name.find_last_of('_') + 1));
                } catch (...) {
                }
                buf.add_generation(g, std::move(sv));
            }
        }
        printf("[回放池] %zu 条 / %d 代\n", buf.total(), buf.generations());
    }

    Trainer trainer(&net, [&] {
        TrainOptions to;
        to.batch_size = cfg.batch_size;
        to.lr = cfg.lr;
        to.weight_decay = cfg.weight_decay;
        to.clip_norm = cfg.clip_norm;
        to.ema_decay = cfg.ema_decay;
        to.fresh_ratio = cfg.fresh_ratio;
        to.amp = cfg.amp;
        return to;
    }(), cfg.seed + 99);

    std::mt19937_64 rng(cfg.seed + 7);
    // ---- lr 起点：同一配方内续跑就接上一代的退火值 ----
    // 换配方（recipe_id 变了）才从 --lr 重设；要主动重新起算退火用 --lr-reset 1。
    double lr = cfg.lr;
    {
        double prev_lr = 0.0;
        int prev_recipe = 0;
        if (!cfg.lr_reset && last_gen > 0 && read_last_log_lr(resume_dir, &prev_lr, &prev_recipe)) {
            if (prev_recipe != cfg.recipe_id) {
                printf("[学习率] 换配方 %d → %d，lr 从 --lr 重设 %.6f\n", prev_recipe,
                       cfg.recipe_id, cfg.lr);
            } else {
                lr = std::max(cfg.lr_floor, prev_lr);
                printf("[学习率] 续跑接着配方 %d 的上一代 lr %.6f（不回到 --lr %.6f）\n",
                       cfg.recipe_id, lr, cfg.lr);
            }
        } else if (cfg.lr_reset) {
            printf("[学习率] --lr-reset：lr 重设为 %.6f\n", cfg.lr);
        }
    }
    int lr_decay_count = 0;
    // 「上一个被测量的代」= 边库里最大的代，而不是 best.bin 推的 last_gen。
    // 续训时 best.bin 不带代数信息，用 last_gen 会把一个**从没测过的代**当左端点，
    // 那条边落库后与主体不连通（拟合成孤立分量，直接剔除，白打一场）。
    int last_accepted = 0;
    {
        const auto st = elo_store_load(opt.dir + "/elo_edges.jsonl");
        for (const auto& e : st) {
            last_accepted = std::max(last_accepted, elo_gen_of_name(e.a));
            last_accepted = std::max(last_accepted, elo_gen_of_name(e.b));
        }
    }
    if (last_accepted <= 0 && have_best && last_gen > 0) last_accepted = last_gen;
    // 自上次补打 Elo 以来又采纳了几代（--elo-every-accepts，默认 2）。
    int accepts_since_elo = 0;
    // 「上一次真正打过边的那一代」——收尾 flush 用它判断有没有欠账。
    int last_elo_gen = last_accepted;

    // 打一条「from vs to」的直边并落库。返回是否成功。
    // 抽成 lambda 是因为有两个触发点：攒够 elo_every_accepts 次采纳时、以及本轮跑完时。
    auto run_accept_elo = [&](int from, int to, const char* why) -> bool {
        try {
            EloRunOptions ro;
            ro.games = cfg.elo_calib_games;
            ro.opening_plies = cfg.gate_opening_plies;
            ro.max_ref_span = 0;   // 标定边就是要长跨，别被跨度上限跳过
            ro.with_ref = false;   // 只打这一条直边
            ro.verbose = true;
            printf("  [Elo] %s → 补打 gen_%03d vs gen_%03d × %d 局\n", why, from, to, ro.games);
            auto edges = elo_play(cfg, opt.dir, {from, to}, from, ro);
            EloCommitOptions co;
            co.verbose = true;
            elo_commit(cfg, opt.dir, edges, co);
            return true;
        } catch (const std::exception& e) {
            printf("  [警告] 补打 Elo 失败，训练继续：%s\n", e.what());
        } catch (...) {
            printf("  [警告] 补打 Elo 失败（未知异常），训练继续\n");
        }
        return false;
    };
    FILE* logf = fopen((opt.dir + "/train_log.jsonl").c_str(), "a");
    const int end_gen = start_gen + opt.gens - 1;

    // 预热：第一次反向传播会触发 cuDNN 的一次性算法选择（本机实测 ~16 s）。
    // 不预热的话这笔开销会算到第 1 代的「训练」时间里，看起来像 300 ms/步。
    // 做法是拿一批假样本走一遍完整的 train_on，然后把权重还原（warmup 会动参数）。
    if (cfg.cuda && opt.gens > 0) {
        std::vector<std::pair<std::string, torch::Tensor>> snap = net.named_tensors();
        for (auto& kv : snap) kv.second = kv.second.clone();
        std::vector<Sample> dummy((size_t)cfg.batch_size);
        for (auto& s : dummy) {
            std::memset(s.state, 0, sizeof(s.state));
            s.state[0] = 1;            // ch0 在 (0,0) 放一个黑子（内容不重要，只为触发反向）
            s.state[2 * SIZE * SIZE] = 1;  // ch2 = 轮到黑走
            std::memset(s.policy, 0, sizeof(s.policy));
            s.policy[9 * SIZE + 9] = 1.0f;
            s.value = 0.0f;
        }
        const double tw = now_ms();
        trainer.train_on(dummy);
        net.load_named_tensors(snap, true, &err);
        net.ema_init(torch::TensorOptions().dtype(torch::kFloat32));
        printf("[预热] 首步初始化 %.1f s（一次性，已还原权重）\n", (now_ms() - tw) / 1000.0);
    }

    for (int gen = start_gen; gen <= end_gen; ++gen) {
        printf("\n=== gen %d/%d ===\n", gen, end_gen);

        // ---- 1) 自对弈（用当前原始权重，与 Python 一致）----
        // 两条路径：
        //   mp_procs > 1 → 多进程（每进程独立 CUDA 上下文，和 Python 版同构）
        //   否则         → 单进程多线程 + 共享批量评估器
        std::vector<Sample> fresh;
        const double ts = now_ms();
        SelfPlayStats sp;
        NetService::Stats nstat;
        if (cfg.mp_procs > 1) {
            const std::string wpath = fmt("%s/tmp/mp_gen%03d.bin", opt.dir.c_str(), gen);
            std::error_code mec;
            fs::create_directories(opt.dir + "/tmp", mec);
            if (!save_named_tensors(wpath, net.arch(), net.named_tensors())) {
                printf("  [错误] 写不了临时权重 %s，本代跳过\n", wpath.c_str());
                continue;
            }
            sp = collect_games_mp(cfg, wpath, cfg.games, (uint64_t)cfg.seed + (uint64_t)gen,
                                  fmt("%s/tmp/mp_gen%03d", opt.dir.c_str(), gen), fresh);
            std::error_code wec;
            fs::remove(wpath, wec);  // 临时权重用完就删（每代 7 MB）
        } else {
            NetService svc(&ev_net, cfg.eval_batch, cfg.eval_window_ms, cfg.eval_threads);
            if (cfg.workers > 1) svc.start();
            sp = collect_games(cfg.workers > 1 ? (Evaluator*)&svc : (Evaluator*)&ev_net, cfg,
                               cfg.games, (uint64_t)cfg.seed + (uint64_t)gen, fresh);
            nstat = svc.stats();
            if (cfg.workers > 1) svc.stop();
        }
        const double t_selfplay = now_ms() - ts;
        if (fresh.empty()) {
            printf("  [警告] 本代没有可用样本，跳过\n");
            continue;
        }
        printf("  [自对弈] %d 局 %.1fs（%.0f 手/局，黑 %d 白 %d 平 %d 认输 %d）样本 %d\n",
               sp.games, t_selfplay / 1000.0,
               sp.games ? (double)sp.plies / sp.games : 0.0, sp.black_wins, sp.white_wins,
               sp.draws, sp.resigned, sp.samples);
        if (nstat.batches) {
            printf("  [评估器] 前向 %llu 次 / 状态 %llu / 平均 batch %.1f\n",
                   (unsigned long long)nstat.batches, (unsigned long long)nstat.states,
                   (double)nstat.states / (double)nstat.batches);
        }

        // ---- 2) 入库 ----
        buf.add_generation(gen, std::vector<Sample>(fresh.begin(), fresh.end()));

        // ---- 3) 训练 ----
        const int steps = std::max(
            cfg.min_steps, (int)((double)fresh.size() * cfg.samples_per_gen / cfg.batch_size));
        trainer.set_lr(lr);
        const double tt = now_ms();
        double loss_sum = 0, ce_sum = 0, mse_sum = 0;
        int ok_steps = 0;
        const int report = cfg.log_every_steps > 0 ? cfg.log_every_steps : std::max(1, steps / 10);
        for (int s = 0; s < steps; ++s) {
            TrainStepResult r = trainer.train_step(buf, fresh);
            if (!r.ok) break;
            loss_sum += r.loss;
            ce_sum += r.ce;
            mse_sum += r.mse;
            ++ok_steps;
            if ((s + 1) % report == 0) {
                printf("    步 %d/%d  loss %.4f (ce %.4f mse %.5f)  lr %.6f\n", s + 1, steps,
                       loss_sum / ok_steps, ce_sum / ok_steps, mse_sum / ok_steps, lr);
            }
        }
        const double t_train = now_ms() - tt;
        const double m_loss = ok_steps ? loss_sum / ok_steps : 0.0;
        const double m_ce = ok_steps ? ce_sum / ok_steps : 0.0;
        const double m_mse = ok_steps ? mse_sum / ok_steps : 0.0;
        printf("  [训练] %d 步 %.1fs（%.1f ms/步）loss %.4f ce %.4f mse %.5f\n", ok_steps,
               t_train / 1000.0, ok_steps ? t_train / ok_steps : 0.0, m_loss, m_ce, m_mse);

        // ---- 4) 门控 ----
        bool gated = false;
        GateResult gr;
        bool accepted_now = false;   // 本代是否被采纳（供 5.5 的 Elo 触发用）
        int prev_accepted = last_accepted;
        if (cfg.eval_every > 0 && (gen % cfg.eval_every == 0)) {
            // 候选权重：默认用「刚训完的原始 net」，因为 EMA(0.999) 的时间常数约 1000 步
            // ≈ 3 代，拿它去挑战已冻结的 best，测到的是三代前的平均而不是当前进度
            // （实测连续两代都是 7/20）。需要老行为时加 --gate-ema。
            if (cfg.gate_use_ema) {
                net.ema_copy_to(ema_net);
            } else {
                ema_net.load_named_tensors(net.named_tensors(), true, &err);
                ema_net.to_device(cfg.cuda);
            }
            if (!have_best) {
                // 第一代没有基线：直接把候选当作 best
                auto items = ema_net.named_tensors();
                best_net.load_named_tensors(items, true, &err);
                best_net.to_device(cfg.cuda);
                have_best = true;
                last_accepted = gen;
                printf("  [门控] 建立基线 best = 当前候选\n");
            } else {
                GateOptions go;
                go.games = cfg.eval_games;
                go.threshold = cfg.gate_threshold;
                go.ci_ll = cfg.gate_ci_ll;
                go.n_sim = cfg.n_sim;
                go.c_puct = cfg.c_puct;
                go.heuristic_weight = cfg.heuristic_weight;
                go.policy_weight = cfg.policy_weight;
                go.max_children = cfg.max_children;
                go.root_max_children = cfg.root_max_children;
                go.opening_plies = cfg.gate_opening_plies;
                const double tg = now_ms();
                // 门控对局并行：每个评估器包一层共享批量评估器，让 8 局的前向合并成大批
                NetService svc_cand(&ev_ema, cfg.eval_batch, cfg.eval_window_ms, cfg.eval_threads);
                NetService svc_best(&ev_best, cfg.eval_batch, cfg.eval_window_ms, cfg.eval_threads);
                // 必须启动：即使只开 1 个门控线程，前向也要走评估线程（否则请求会一直排队）
                svc_cand.start();
                svc_best.start();
                gr = run_gate_threads(&svc_cand, &svc_best, go,
                                      (uint64_t)cfg.seed * 31ULL + (uint64_t)gen,
                                      std::max(1, cfg.eval_workers));
                svc_cand.stop();
                svc_best.stop();
                gated = true;
                printf("  [门控] %.1fs  %d/%d 胜（黑 %d/%d 白 %d/%d 平 %d）胜率 %.3f "
                       "Wilson[%.3f,%.3f] %s", 
                       (now_ms() - tg) / 1000.0, gr.wins, gr.games, gr.wins_black,
                       gr.games_black, gr.wins_white, gr.games_white, gr.draws, gr.win_rate,
                       gr.wilson_lo, gr.wilson_hi, gr.accepted ? "采纳 → best 更新"
                                                             : "拒绝（best 不变）");
                if (cfg.gate_opening_plies > 0) {
                    // 配对开局生效时必须写出来：否则同一份胜率无法区分「模型变强」和「开局变了」
                    printf("（配对开局 %d 手）", cfg.gate_opening_plies);
                }
                printf("\n");
                if (gr.accepted) {
                    auto items = ema_net.named_tensors();
                    best_net.load_named_tensors(items, true, &err);
                    best_net.to_device(cfg.cuda);
                    have_best = true;
                    accepted_now = true;
                    last_accepted = gen;
                    ++accepts_since_elo;
                } else if (gr.win_rate < cfg.lr_decay_min_win_rate &&
                           lr_decay_count < cfg.max_lr_decay) {
                    lr = std::max(cfg.lr_floor, lr * cfg.lr_decay);
                    ++lr_decay_count;
                    printf("  [学习率] 显著退化 → lr %.6f（第 %d 次）\n", lr, lr_decay_count);
                }
            }
        }

        // ---- 4.5) 按代退火（配方 3 起）----
        // 注意：这一步**无论门控采纳还是拒绝**都执行，每代一次。历史教训：43 代里
        // 上面那个「显著退化（<0.30）才降 lr」一次都没触发，因为胜率一直在 0.45–0.55，
        // 于是 lr 从第 1 代到第 43 代恒为 3e-4，网络后期只是在最优点附近随机游走。
        if (cfg.lr_gen_decay > 0.0 && cfg.lr_gen_decay < 1.0) {
            const double before = lr;
            lr = std::max(cfg.lr_floor, lr * cfg.lr_gen_decay);
            if (lr < before) {
                printf("  [学习率] 按代退火 %.6f → %.6f（配方 %d，衰减 %.4g/代）\n", before, lr,
                       cfg.recipe_id, cfg.lr_gen_decay);
            }
        }

        // ---- 5) 落盘 ----
        {
            Arch a = cfg.arch;
            const std::string genp = opt.dir + fmt("/gen_%03d.bin", gen);
            save_named_tensors(genp, a, net.named_tensors());
            save_named_tensors(opt.dir + "/best.bin", a, best_net.named_tensors());
            save_samples(opt.dir + fmt("/buffer/gen_%03d.bin", gen), fresh);
            // 权重快照：keep_gens=0 表示全留（默认），否则只留最近 keep_gens 代
            if (cfg.keep_gens > 0) {
                for (int old = gen - cfg.keep_gens; old >= start_gen; --old) {
                    std::error_code e2;
                    fs::remove(opt.dir + fmt("/gen_%03d.bin", old), e2);
                }
            }
            // 回放池样本文件按「回放窗口 + 余量」保留，避免每代 20 MB 无限堆积
            {
                const int keep_buf = std::max(3, cfg.keep_buffer_gens);
                for (int old = gen - keep_buf; old >= start_gen; --old) {
                    std::error_code e2;
                    fs::remove(opt.dir + fmt("/buffer/gen_%03d.bin", old), e2);
                }
            }
        }

        // ---- 5.5) 采纳驱动 Elo（--elo-on-accept，默认开）----
        // 必须放在**落盘之后**：Elo 要读 gen_%03d.bin，而门控在落盘之前就跑完了，
        // 那时本代权重还没写盘，只剩"上一个被采纳的代"一个版本 → elo_play 会直接跳过
        // （实测过一次：打出"[elo] 可用版本不足 2 个"）。
        // 门控说"这一代比 best 强"，但它只有 eval_games 局、且是二元判定（>0.55 就过），
        // 给不出"强多少"。这里把**同一对**模型补打 elo_calib_games 局（默认 150，±28 Elo）：
        //   · 曲线上每一段都是直接对打的结果，不是链式推的 → 没有累积误差，
        //     于是"跨段标定"这件事本身就不存在了；
        //   · 只在采纳时才跑，被拒的代不浪费 GPU（43 代里 19 代被拒）；
        //   · 与门控测的是同一对模型，"采纳"与"涨了多少"是同一事件的两个视角，
        //     不会互相矛盾（旧曲线 gen_076 比 gen_071 低 13 Elo 那种矛盾就是这么来的）。
        // 触发条件里多一个「攒够 elo_every_accepts 次采纳」：边是直接对打，跨 2 代打一条
        // 仍然是直接测量，误差不累积，只是曲线上的点稀一些 —— 换来 Elo 开销减半。
        // 但**光靠它会在低采纳期饿死**：实测 88–96 九代只采纳了 gen_092 一代（32/60=0.533），
        // 计数器始终没到 2，整轮一条边都没打，曲线停在 gen_086。所以末尾补了 flush。
        if (cfg.elo_on_accept && accepted_now && prev_accepted > 0 && prev_accepted != gen &&
            accepts_since_elo >= std::max(1, cfg.elo_every_accepts)) {
            if (run_accept_elo(prev_accepted, gen, "采纳")) {
                accepts_since_elo = 0;   // 只有真正落库了才清零，失败就留到下次采纳再试
                last_elo_gen = gen;
            }
        }

        // ---- 6) 日志 ----
        if (logf) {
            // 回放池诊断：pool_gens=池中实际代数，hist_mean_age=历史样本加权平均代龄
            // （age 0 是 exclude_latest 后的次新代，真实代龄要 +1），hist_share20=近 20 代占比
            const auto bst = buf.last_stats();
            fprintf(logf,
                    "{\"gen\":%d,\"games\":%d,\"samples\":%d,\"steps\":%d,\"loss\":%.5f,"
                    "\"ce\":%.5f,\"mse\":%.5f,\"lr\":%.7f,\"selfplay_s\":%.2f,\"train_s\":%.2f,"
                    "\"buffer_total\":%zu,\"gated\":%d,\"gate_wins\":%d,\"gate_games\":%d,"
                    "\"gate_wr\":%.4f,\"gate_accepted\":%d,\"recipe_id\":%d,"
                    "\"pool_gens\":%d,\"hist_mean_age\":%.2f,\"hist_share20\":%.3f,"
                    "\"hist_gens\":%d}\n",
                    gen, sp.games, sp.samples, ok_steps, m_loss, m_ce, m_mse, lr,
                    t_selfplay / 1000.0, t_train / 1000.0, buf.total(), gated ? 1 : 0,
                    gated ? gr.wins : 0, gated ? gr.games : 0, gated ? gr.win_rate : 0.0,
                    gated && gr.accepted ? 1 : 0, cfg.recipe_id,
                    buf.generations(), bst.mean_age_seen, bst.share_recent20_seen,
                    bst.gens_drawn);
            fflush(logf);
        }

        // ---- 7) 自动 Elo（--elo-every N）----
        // 放在一代的最后：权重已落盘、日志已 flush，此时跑 Elo 看到的是完整状态。
        // 串行执行（与训练同一进程、同一块 GPU，不能并行），所以这段时间训练是停的：
        // --elo-every 5 --elo-games 30 ≈ 每 5 代多花 8 分钟。
        // 失败只警告、不中断训练：Elo 是**事后测量**，它挂了不该连累正在跑的这十几代。
        if (cfg.elo_every > 0 && (gen % cfg.elo_every == 0)) {
            EloAutoOptions eo;
            eo.window = cfg.elo_window;
            eo.games = cfg.elo_games;
            eo.calib_every = cfg.elo_calib_every;
            eo.calib_games = cfg.elo_calib_games;
            eo.max_ref_span = cfg.elo_max_ref_span;
            eo.verbose = true;
            try {
                elo_auto_update(cfg, opt.dir, gen, eo);
            } catch (const std::exception& e) {
                printf("  [警告] 自动 Elo 失败，训练继续：%s\n", e.what());
            } catch (...) {
                printf("  [警告] 自动 Elo 失败（未知异常），训练继续\n");
            }
        }
    }
    // ---- 收尾 flush：本轮攒下的采纳还没测，就在退出前补上 ----
    // 「每 2 次采纳测一次」在低采纳期会饿死（88–96 九代只采纳 1 代 → 一条边都没打），
    // 而曲线是事后的账，欠着就等于这段训练没有刻度。这里做一次结账：
    // 只要最后一次采纳晚于最后一次测量，就打一条「上次测量代 → 最后采纳代」的直边
    // （跨度可能 2~4 代，仍是直接对打，不受影响）。
    if (cfg.elo_on_accept && last_accepted > last_elo_gen && last_elo_gen > 0) {
        if (run_accept_elo(last_elo_gen, last_accepted, "本轮收尾")) {
            printf("  [Elo] 本轮欠的账已结清（gen_%03d）\n", last_accepted);
        }
    }
    if (logf) fclose(logf);
    printf("\n完成：权重与日志在 %s\n", opt.dir.c_str());
    return 0;
}

}  // namespace gk
