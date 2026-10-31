// elo.cpp — Bradley–Terry 拟合（MM 算法）+ JSON 输出
#include "elo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>

namespace gk {
namespace {

constexpr double kLn10 = 2.302585092994046;
constexpr double kScale = 400.0 / kLn10;  // gamma 的对数 → Elo

// MM 一步：γ_i ← S_i / Σ_{j~i} n_ij / (γ_i + γ_j)
// S_i = Σ_j (胜_A + 0.5*平) —— 用「得分」而不是「胜场」，平局按半分计。
std::vector<double> mm_fit(int n, const std::vector<EloEdge>& edges, int iters = 400) {
    std::vector<double> g((size_t)n, 1.0);
    std::vector<double> S((size_t)n, 0.0);
    for (const auto& e : edges) {
        if (e.games <= 0) continue;
        S[(size_t)e.a] += e.score_a;
        S[(size_t)e.b] += (double)e.games - e.score_a;
    }
    for (int it = 0; it < iters; ++it) {
        std::vector<double> den((size_t)n, 0.0);
        for (const auto& e : edges) {
            if (e.games <= 0) continue;
            const double s = g[(size_t)e.a] + g[(size_t)e.b];
            if (s <= 0.0) continue;
            den[(size_t)e.a] += (double)e.games / s;
            den[(size_t)e.b] += (double)e.games / s;
        }
        double max_delta = 0.0;
        for (int i = 0; i < n; ++i) {
            // 孤立点（没打过任何对局）：保持 1.0，不要 NaN
            if (den[(size_t)i] <= 0.0) continue;
            // 全胜/全负时 S 可能为 0（全负）——加 0.5 的 Laplace 修正避免 γ→0
            const double num = S[(size_t)i] + 0.5;
            const double ng = num / den[(size_t)i];
            if (ng > 0.0 && std::isfinite(ng)) {
                max_delta = std::max(max_delta, std::fabs(std::log(ng / g[(size_t)i])));
                g[(size_t)i] = ng;
            }
        }
        if (max_delta < 1e-9) break;
    }
    return g;
}

// 把 γ 归一到 anchor=0 的 Elo
std::vector<double> to_elo(const std::vector<double>& g, int anchor) {
    std::vector<double> elo(g.size(), 0.0);
    const double base = (anchor >= 0 && (size_t)anchor < g.size()) ? g[(size_t)anchor] : 1.0;
    for (size_t i = 0; i < g.size(); ++i) {
        elo[i] = (g[i] > 0.0 && base > 0.0) ? kScale * std::log(g[i] / base) : 0.0;
    }
    return elo;
}
}  // namespace

void fit_elo(std::vector<EloModel>& models, const std::vector<EloEdge>& edges, int anchor,
             int bootstrap, uint64_t seed) {
    const int n = (int)models.size();
    if (n == 0) return;
    if (anchor < 0 || anchor >= n) anchor = 0;

    const std::vector<double> g = mm_fit(n, edges);
    const std::vector<double> elo = to_elo(g, anchor);
    for (int i = 0; i < n; ++i) models[(size_t)i].elo = elo[(size_t)i];

    if (bootstrap <= 0) return;

    // 参数化 bootstrap：每对按拟合概率重采样胜负，重拟合，取 Elo 的分位
    std::mt19937_64 rng(seed);
    std::vector<std::vector<double>> samples((size_t)n);
    for (int b = 0; b < bootstrap; ++b) {
        std::vector<EloEdge> re = edges;
        for (auto& e : re) {
            if (e.games <= 0) continue;
            const double s = g[(size_t)e.a] + g[(size_t)e.b];
            const double p = (s > 0.0) ? g[(size_t)e.a] / s : 0.5;
            std::binomial_distribution<int> bd(e.games, std::min(1.0, std::max(0.0, p)));
            const int wins = bd(rng);
            e.score_a = (double)wins;  // 重采样时把原平局当整场处理（近似，偏差 <1 局）
        }
        const std::vector<double> gb = mm_fit(n, re, 120);
        const std::vector<double> eb = to_elo(gb, anchor);
        for (int i = 0; i < n; ++i) samples[(size_t)i].push_back(eb[(size_t)i]);
    }
    for (int i = 0; i < n; ++i) {
        auto& v = samples[(size_t)i];
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        const size_t lo_i = (size_t)(0.05 * (double)(v.size() - 1));
        const size_t hi_i = (size_t)(0.95 * (double)(v.size() - 1));
        models[(size_t)i].lo = v[lo_i];
        models[(size_t)i].hi = v[hi_i];
    }
}

std::string elo_json(const std::vector<EloModel>& models, const std::vector<EloEdge>& edges,
                     int anchor, const std::string& generated_at) {
    std::ostringstream o;
    o.precision(3);
    o << std::fixed;
    // 键名与看板约定一致：versions（每代一项）/ anchor_name / generated。
    // 注意 anchor_name 的引号是解析端唯一依赖的东西——少一个引号整份 JSON 就废了，
    // 而看板拿不到 elo 时会安静地显示「还没有 Elo 数据」，所以这里必须写全。
    o << "{\"generated\":\"" << generated_at << "\",\"anchor_name\":\""
      << (anchor >= 0 && (size_t)anchor < models.size() ? models[(size_t)anchor].name : std::string())
      << "\",\"anchor_gen\":"
      << (anchor >= 0 && (size_t)anchor < models.size() ? models[(size_t)anchor].gen : 0) << ",";
    o << "\"versions\":[";
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& m = models[i];
        if (i) o << ",";
        o << "{\"name\":\"" << m.name << "\",\"gen\":" << m.gen << ",\"elo\":" << m.elo
          << ",\"lo\":" << m.lo << ",\"hi\":" << m.hi << ",\"games\":" << m.games << "}";
    }
    o << "],\"edges\":[";
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& e = edges[i];
        if (i) o << ",";
        o << "{\"a\":\"" << models[(size_t)e.a].name << "\",\"b\":\"" << models[(size_t)e.b].name
          << "\",\"games\":" << e.games << ",\"wins_a\":" << e.wins_a << ",\"draws\":" << e.draws
          << ",\"score_a\":" << e.score_a << "}";
    }
    o << "]}";
    return o.str();
}

}  // namespace gk
