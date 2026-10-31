// elo_run.cpp — 打 Elo 对局 + 自动更新曲线（依赖 torch / gate）
#include "elo_run.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "batcher.h"
#include "dataset.h"
#include "elo.h"
#include "eval_torch.h"
#include "gate.h"
#include "net.h"

namespace gk {
namespace {

namespace fs = std::filesystem;

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

std::string stamp_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&tm, &t);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return std::string(buf);
}

bool load_net_into(TorchNet& net, const std::string& path, std::string* err) {
    Arch a;
    std::vector<std::pair<std::string, torch::Tensor>> items;
    if (!load_named_tensors_file(path, a, items)) {
        if (err) *err = "读不出权重 " + path;
        return false;
    }
    if (!(a == net.arch())) {
        if (err) *err = "架构不一致 " + path;
        return false;
    }
    return net.load_named_tensors(items, true, err);
}

bool weight_exists(const std::string& dir, int gen) {
    std::error_code ec;
    return fs::exists(dir + "/" + elo_name_of_gen(gen) + ".bin", ec);
}

}  // namespace

std::vector<EloStoreEdge> elo_play(const TrainConfig& cfg, const std::string& dir,
                                   const std::vector<int>& gens_in, int anchor_gen,
                                   const EloRunOptions& o) {
    std::vector<int> gens;
    for (int g : gens_in)
        if (g > 0 && weight_exists(dir, g)) gens.push_back(g);
    std::sort(gens.begin(), gens.end());
    gens.erase(std::unique(gens.begin(), gens.end()), gens.end());

    std::vector<EloStoreEdge> out;
    if (gens.size() < 2) {
        if (o.verbose) printf("[elo] 可用版本不足 2 个，跳过\n");
        return out;
    }

    // 锚点：给 0 或不在窗口里时，退回窗口内最早一代
    int anchor = anchor_gen;
    if (anchor <= 0 || std::find(gens.begin(), gens.end(), anchor) == gens.end()) anchor = gens.front();

    // 对局对：链式相邻 + 每代对锚点。用规范化 key 去重（链式边和参照边可能撞车）。
    std::vector<std::pair<int, int>> pairs;
    std::vector<std::string> keys;
    auto add_pair = [&](int x, int y) {
        if (x == y) return;
        const std::string k = std::to_string(std::min(x, y)) + "-" + std::to_string(std::max(x, y));
        if (std::find(keys.begin(), keys.end(), k) != keys.end()) return;
        keys.push_back(k);
        pairs.push_back({std::min(x, y), std::max(x, y)});
    };
    for (size_t i = 0; i + 1 < gens.size(); ++i) add_pair(gens[i], gens[i + 1]);
    if (o.with_ref) {
        for (int g : gens) {
            if (g == anchor) continue;
            if (o.max_ref_span > 0 && std::abs(g - anchor) > o.max_ref_span) {
                if (o.verbose)
                    printf("[elo] 跳过参照边 gen_%03d vs 锚点 gen_%03d（跨 %d 代，预期饱和）\n", g,
                           anchor, std::abs(g - anchor));
                continue;
            }
            add_pair(g, anchor);
        }
    }
    if (pairs.empty()) return out;

    if (o.verbose)
        printf("[elo] %zu 个版本，锚点 gen_%03d，%zu 对 × %d 局（sims %d，配对开局 %d）\n",
               gens.size(), anchor, pairs.size(), o.games, cfg.n_sim, o.opening_plies);

    const double t_all = now_ms();
    int saturated = 0, skipped = 0;
    for (auto pr : pairs) {
        if (o.skip_if_games > 0 && o.existing) {
            const int have = elo_store_games(*o.existing, elo_name_of_gen(pr.first),
                                             elo_name_of_gen(pr.second));
            if (have >= o.skip_if_games) {
                ++skipped;
                continue;
            }
        }
        TorchNet na(cfg.arch, cfg.cuda, cfg.seed);
        TorchNet nb(cfg.arch, cfg.cuda, cfg.seed + 1);
        std::string err;
        const std::string pa = dir + "/" + elo_name_of_gen(pr.first) + ".bin";
        const std::string pb = dir + "/" + elo_name_of_gen(pr.second) + ".bin";
        if (!load_net_into(na, pa, &err) || !load_net_into(nb, pb, &err)) {
            printf("[elo] 跳过 gen_%03d vs gen_%03d：%s\n", pr.first, pr.second, err.c_str());
            continue;
        }
        TorchEvaluator ea(&na), eb(&nb);
        GateOptions go;
        go.games = o.games;
        go.threshold = 0.5;
        go.n_sim = cfg.n_sim;
        go.c_puct = cfg.c_puct;
        go.heuristic_weight = cfg.heuristic_weight;
        go.policy_weight = cfg.policy_weight;
        go.max_children = cfg.max_children;
        go.root_max_children = cfg.root_max_children;
        go.vcf = cfg.vcf;
        go.vcf_max_ply = cfg.vcf_max_ply;
        go.vcf_max_nodes = cfg.vcf_max_nodes;
        go.guarantee_rush = cfg.guarantee_rush;
        go.opening_plies = o.opening_plies;
        GateResult r;
        {
            NetService sa(&ea, cfg.eval_batch, cfg.eval_window_ms, cfg.eval_threads);
            NetService sb(&eb, cfg.eval_batch, cfg.eval_window_ms, cfg.eval_threads);
            sa.start();
            sb.start();
            r = run_gate_threads(&sa, &sb, go, cfg.seed + (uint64_t)pr.first * 131ULL,
                                 std::max(1, cfg.eval_workers));
            sa.stop();
            sb.stop();
        }
        const double score_a = (double)r.wins + 0.5 * (double)r.draws;
        out.push_back(elo_store_edge(elo_name_of_gen(pr.first), elo_name_of_gen(pr.second), r.games,
                                     score_a, r.wins, r.draws, stamp_now()));
        if (score_a <= 0.001 || score_a >= (double)r.games - 0.001) ++saturated;
        if (o.verbose)
            printf("  gen_%03d vs gen_%03d：%d 胜 %d 负 %d 平  得分 %.3f  %.1fs\n", pr.first,
                   pr.second, r.wins, r.losses, r.draws,
                   score_a / (double)std::max(1, r.games), r.elapsed_ms / 1000.0);
    }
    if (saturated > 0 && o.verbose)
        printf("[elo] 警告：%d/%zu 条边全胜或全负（饱和），不贡献区分度。\n", saturated,
               pairs.size());
    if (skipped > 0 && o.verbose)
        printf("[elo] %d 对在边库里已有 ≥%d 局，跳过（不重复打）\n", skipped, o.skip_if_games);
    if (o.verbose) printf("[elo] 本批 %zu 条边，耗时 %.1f s\n", out.size(), (now_ms() - t_all) / 1000.0);
    return out;
}

int elo_auto_update(const TrainConfig& cfg, const std::string& dir, int cur_gen,
                    const EloAutoOptions& o) {
    const std::string store_path = o.store.empty() ? (dir + "/elo_edges.jsonl") : o.store;
    const std::string out_path = o.out.empty() ? (dir + "/elo.json") : o.out;

    std::vector<EloStoreEdge> store = elo_store_load(store_path);

    // 边库里已经评过的最大代数（决定窗口要往前延多少才接得上）
    int last_eval = 0;
    for (const auto& e : store) {
        last_eval = std::max(last_eval, elo_gen_of_name(e.a));
        last_eval = std::max(last_eval, elo_gen_of_name(e.b));
    }

    // ---- 窗口：最近 window 代，且必须覆盖「最新已评代 - 2」保证有重叠 ----
    int start = std::max(1, cur_gen - o.window + 1);
    if (last_eval > 0) start = std::min(start, std::max(1, last_eval - 2));
    std::vector<int> gens;
    for (int g = start; g <= cur_gen; ++g)
        if (weight_exists(dir, g)) gens.push_back(g);

    const int anchor = last_eval > 0 ? last_eval : (gens.empty() ? 0 : gens.front());
    if (o.verbose)
        printf("\n[elo] 自动更新：窗口 gen_%03d..%03d，锚点 %s\n", start, cur_gen,
               anchor > 0 ? elo_name_of_gen(anchor).c_str() : "（冷启动）");

    // ---- 跳过边库里已经打够局数的对（滚动窗口有重叠，别重复烧 GPU）----
    EloRunOptions ro;
    ro.games = o.games;
    ro.max_ref_span = o.max_ref_span;
    ro.opening_plies = cfg.gate_opening_plies;
    ro.verbose = o.verbose;

    std::vector<EloStoreEdge> fresh;
    {
        // 先按与 elo_play 相同的规则算出「这次想打哪些对」，然后：
        //   * 边库里已经打够局数的对 → 跳过（滚动窗口有重叠，重复打纯属浪费 GPU）
        //   * 剩下的逐对打（每对局数不同时无法合批，所以这里一对一调 elo_play）
        std::vector<std::pair<int, int>> want;
        std::vector<std::string> keys;
        auto add_pair = [&](int x, int y) {
            if (x == y) return;
            const std::string k =
                std::to_string(std::min(x, y)) + "-" + std::to_string(std::max(x, y));
            if (std::find(keys.begin(), keys.end(), k) != keys.end()) return;
            keys.push_back(k);
            want.push_back({std::min(x, y), std::max(x, y)});
        };
        for (size_t i = 0; i + 1 < gens.size(); ++i) add_pair(gens[i], gens[i + 1]);
        for (int g : gens) {
            if (g == anchor) continue;
            if (o.max_ref_span > 0 && std::abs(g - anchor) > o.max_ref_span) continue;
            add_pair(g, anchor);
        }
        // ---- 跨段标定边：老代直打当前代，400 局 ----
        if (o.calib_every > 0 && cur_gen >= o.calib_every && (cur_gen % o.calib_every == 0)) {
            const int old = cur_gen - o.calib_every;
            bool exists = false;
            std::error_code ec;
            if (fs::exists(dir + "/" + elo_name_of_gen(old) + ".bin", ec)) {
                // 只有当边库里还没有这条长跨边（或局数不足 1/4）时才打
                for (const auto& e : store)
                    if ((e.a == elo_name_of_gen(old) && e.b == elo_name_of_gen(cur_gen)) ||
                        (e.b == elo_name_of_gen(old) && e.a == elo_name_of_gen(cur_gen))) {
                        if (e.games >= o.calib_games / 2) exists = true;
                    }
                if (!exists) {
                    if (o.verbose)
                        printf("[elo] 跨段标定：gen_%03d vs gen_%03d × %d 局（重新标定尺子，"
                               "把链式累积误差压下去）\n",
                               old, cur_gen, o.calib_games);
                    // 单独打这条边：局数与滚动边不同，不能混进同一批
                    EloRunOptions ro2 = ro;
                    ro2.games = o.calib_games;
                    ro2.with_ref = false;
                    for (auto& e : elo_play(cfg, dir, {old, cur_gen}, old, ro2))
                        fresh.push_back(e);
                }
            }
        }
        // 滚动边：过滤已够局数的对，剩下的逐对打
        int skipped = 0;
        for (auto pr : want) {
            const int have = elo_store_games(store, elo_name_of_gen(pr.first),
                                             elo_name_of_gen(pr.second));
            if (have >= o.games) {
                ++skipped;
                continue;
            }
            EloRunOptions ro1 = ro;
            ro1.with_ref = false;
            auto got = elo_play(cfg, dir, {pr.first, pr.second}, pr.first, ro1);
            for (auto& e : got) fresh.push_back(e);
        }
        if (skipped > 0 && o.verbose)
            printf("[elo] 已有 %d 对达到 %d 局，跳过（不重复打）\n", skipped, o.games);
    }

    EloCommitOptions co;
    co.bootstrap = o.bootstrap;
    co.anchor_gen = 0;              // 0 = 最老代当 0 点（曲线向上 = 变强）
    co.out = out_path;
    co.store = store_path;
    co.verbose = o.verbose;
    return elo_commit(cfg, dir, fresh, co);
}

int elo_commit(const TrainConfig& cfg, const std::string& dir,
               const std::vector<EloStoreEdge>& fresh, const EloCommitOptions& o) {
    const std::string store_path = o.store.empty() ? (dir + "/elo_edges.jsonl") : o.store;
    const std::string out_path = o.out.empty() ? (dir + "/elo.json") : o.out;

    std::vector<EloStoreEdge> store = elo_store_load(store_path);
    if (fresh.empty()) {
        if (o.verbose) printf("[elo] 没有新对局，曲线不变（边库 %zu 条边）\n", store.size());
    } else {
        const int n_new = elo_store_merge(store, fresh);
        if (!elo_store_save(store_path, store)) {
            printf("[elo] 警告：写不进边库 %s，本次结果丢弃\n", store_path.c_str());
            return 1;
        }
        if (o.verbose)
            printf("[elo] 边库：新增 %d 条（本次 %zu 条），共 %zu 条边\n", n_new, fresh.size(),
                   store.size());
    }

    // 锚点：给了就用给的；没给（0）交给 elo_curve_from_store → **最老的那个版本**。
    // 不取"最新已评代"：那会让整条曲线变成负数、越往后越接近 0，读起来像在退步。
    // 边库模式下远锚点**不会饱和**（拟合是全量的，锚点只是一个平移自由度，
    // 不需要它参与对局），所以放心用最老代当 0 点 —— 曲线向上 = 变强。
    EloCurve cur = elo_curve_from_store(dir, store, o.anchor_gen, o.bootstrap, cfg.seed);
    if (cur.models.empty()) {
        printf("[elo] 警告：拟合后没有可用版本，不写 elo.json\n");
        return 1;
    }
    if (cur.dropped_missing > 0 && o.verbose)
        printf("[elo] 边库里有 %d 个版本的权重文件已不存在，已剔除\n", cur.dropped_missing);
    if (cur.dropped_disconnected > 0 && o.verbose)
        printf("[elo] 边库里有 %d 个版本与主体不连通，已剔除\n", cur.dropped_disconnected);

    if (o.verbose) {
        printf("[elo] %-10s %8s %18s %6s\n", "版本", "Elo", "95% 区间", "对局");
        for (const auto& m : cur.models)
            printf("[elo] %-10s %8.1f  [%7.1f, %7.1f] %6d\n", m.name.c_str(), m.elo, m.lo, m.hi,
                   m.games);
        printf("[elo] 锚点 %s = 0 Elo，%d 个版本 / %d 条边\n",
               cur.models[(size_t)cur.anchor].name.c_str(), (int)cur.models.size(), cur.store_edges);
    }

    std::ofstream of(out_path, std::ios::binary);
    if (!of.good()) {
        printf("[elo] 警告：写不进 %s\n", out_path.c_str());
        return 1;
    }
    const std::string js = elo_json(cur.models, cur.edges, cur.anchor, stamp_now());
    of.write(js.data(), (std::streamsize)js.size());
    of.close();
    if (o.verbose) printf("[elo] 已写入 %s\n", out_path.c_str());
    return 0;
}

}  // namespace gk
