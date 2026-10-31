// elo_store.cpp — 边库读写与全局拟合（不依赖 torch，纯算法）
#include "elo_store.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

namespace gk {
namespace {

// 手写 JSON 行解析：只认自家写的那一行的字段，不追求通用。
// 之所以不引第三方 JSON：整个项目只有这一个地方需要读 JSON，为它引入依赖不值当。
std::string json_str(const std::string& s, const std::string& key) {
    const std::string k = "\"" + key + "\":";
    const size_t p = s.find(k);
    if (p == std::string::npos) return std::string();
    size_t i = p + k.size();
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    if (i >= s.size() || s[i] != '"') return std::string();
    const size_t q = s.find('"', i + 1);
    if (q == std::string::npos) return std::string();
    return s.substr(i + 1, q - i - 1);
}

double json_num(const std::string& s, const std::string& key, double def) {
    const std::string k = "\"" + key + "\":";
    const size_t p = s.find(k);
    if (p == std::string::npos) return def;
    size_t i = p + k.size();
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    if (i >= s.size()) return def;
    return std::atof(s.c_str() + i);
}

int json_int(const std::string& s, const std::string& key, int def) {
    return (int)json_num(s, key, (double)def);
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

}  // namespace

int elo_gen_of_name(const std::string& name) {
    if (name.rfind("gen_", 0) != 0) return -1;
    return std::atoi(name.c_str() + 4);
}

std::string elo_name_of_gen(int gen) {
    char b[32];
    std::snprintf(b, sizeof(b), "gen_%03d", gen);
    return std::string(b);
}

EloStoreEdge elo_store_edge(std::string a, std::string b, int games, double score_a, int wins_a,
                            int draws, const std::string& ts) {
    EloStoreEdge e;
    if (a <= b) {
        e.a = a;
        e.b = b;
        e.score_a = score_a;
        e.wins_a = wins_a;
    } else {
        e.a = b;
        e.b = a;
        e.score_a = (double)games - score_a;
        e.wins_a = games - wins_a - draws;
    }
    e.games = games;
    e.draws = draws;
    e.ts = ts;
    return e;
}

std::vector<EloStoreEdge> elo_store_load(const std::string& path) {
    std::vector<EloStoreEdge> out;
    std::ifstream in(path);
    if (!in.good()) return out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] != '{') continue;
        const std::string a = json_str(line, "a");
        const std::string b = json_str(line, "b");
        const int games = json_int(line, "games", 0);
        if (a.empty() || b.empty() || games <= 0) continue;
        EloStoreEdge e;
        e.a = a;
        e.b = b;
        e.games = games;
        e.score_a = json_num(line, "score_a", 0.0);
        e.wins_a = json_int(line, "wins_a", 0);
        e.draws = json_int(line, "draws", 0);
        e.ts = json_str(line, "ts");
        out.push_back(e);
    }
    return out;
}

int elo_store_merge(std::vector<EloStoreEdge>& all, const std::vector<EloStoreEdge>& add) {
    int fresh = 0;
    for (const auto& e : add) {
        if (e.games <= 0 || e.a.empty() || e.b.empty()) continue;
        bool hit = false;
        for (auto& o : all) {
            if (o.a == e.a && o.b == e.b) {
                o.games += e.games;
                o.score_a += e.score_a;
                o.wins_a += e.wins_a;
                o.draws += e.draws;
                o.ts = e.ts;
                hit = true;
                break;
            }
        }
        if (!hit) {
            all.push_back(e);
            ++fresh;
        }
    }
    return fresh;
}

bool elo_store_save(const std::string& path, const std::vector<EloStoreEdge>& all) {
    std::ostringstream o;
    o.precision(3);
    o << std::fixed;
    for (const auto& e : all) {
        o << "{\"a\":\"" << json_escape(e.a) << "\",\"b\":\"" << json_escape(e.b)
          << "\",\"games\":" << e.games << ",\"score_a\":" << e.score_a
          << ",\"wins_a\":" << e.wins_a << ",\"draws\":" << e.draws << ",\"ts\":\""
          << json_escape(e.ts) << "\"}\n";
    }
    std::ofstream of(path, std::ios::binary | std::ios::trunc);
    if (!of.good()) return false;
    const std::string s = o.str();
    of.write(s.data(), (std::streamsize)s.size());
    return of.good();
}

int elo_store_games(const std::vector<EloStoreEdge>& all, const std::string& a,
                    const std::string& b) {
    const std::string lo = std::min(a, b), hi = std::max(a, b);
    for (const auto& e : all)
        if (e.a == lo && e.b == hi) return e.games;
    return 0;
}

EloCurve elo_curve_from_store(const std::string& dir, const std::vector<EloStoreEdge>& store,
                              int anchor_gen, int bootstrap, uint64_t seed) {
    EloCurve cur;

    // ---- 1) 收集版本 ----
    std::vector<std::string> names;
    for (const auto& e : store) {
        names.push_back(e.a);
        names.push_back(e.b);
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());

    // ---- 2) 去掉权重文件已不存在的版本 ----
    std::vector<std::string> keep;
    for (const auto& n : names) {
        std::ifstream probe(dir + "/" + n + ".bin", std::ios::binary);
        if (probe.good())
            keep.push_back(n);
        else
            ++cur.dropped_missing;
    }
    std::vector<std::string> alive = keep;
    std::sort(alive.begin(), alive.end(), [](const std::string& x, const std::string& y) {
        return elo_gen_of_name(x) < elo_gen_of_name(y);
    });

    // ---- 3) 并查集：只留最大连通分量 ----
    std::vector<int> idx_of_name(names.size(), -1);
    for (size_t i = 0; i < alive.size(); ++i) {
        auto it = std::lower_bound(names.begin(), names.end(), alive[(size_t)i]);
        if (it != names.end() && *it == alive[(size_t)i])
            idx_of_name[(size_t)(it - names.begin())] = (int)i;
    }
    std::vector<int> parent(alive.size());
    for (size_t i = 0; i < parent.size(); ++i) parent[i] = (int)i;
    std::function<int(int)> find = [&](int x) {
        while (parent[(size_t)x] != x) x = parent[(size_t)x] = parent[(size_t)parent[(size_t)x]];
        return x;
    };
    for (const auto& e : store) {
        auto ia = std::lower_bound(names.begin(), names.end(), e.a);
        auto ib = std::lower_bound(names.begin(), names.end(), e.b);
        if (ia == names.end() || ib == names.end() || *ia != e.a || *ib != e.b) continue;
        const int u = idx_of_name[(size_t)(ia - names.begin())];
        const int v = idx_of_name[(size_t)(ib - names.begin())];
        if (u < 0 || v < 0) continue;
        const int ru = find(u), rv = find(v);
        if (ru != rv) parent[(size_t)ru] = rv;
    }
    std::vector<int> comp_size(alive.size(), 0);
    for (size_t i = 0; i < alive.size(); ++i) ++comp_size[(size_t)find((int)i)];
    int best_root = -1, best_n = 0;
    for (size_t i = 0; i < alive.size(); ++i) {
        if (comp_size[(size_t)i] > best_n) {
            best_n = comp_size[(size_t)i];
            best_root = (int)i;
        }
    }
    if (best_root < 0) return cur;  // 没有任何可用版本
    best_root = find(best_root);
    std::vector<int> keep_local;
    for (size_t i = 0; i < alive.size(); ++i)
        if (find((int)i) == best_root) keep_local.push_back((int)i);
    cur.dropped_disconnected = (int)alive.size() - (int)keep_local.size();

    // ---- 4) 建 models / edges ----
    std::vector<int> local_of(alive.size(), -1);
    for (size_t k = 0; k < keep_local.size(); ++k) local_of[(size_t)keep_local[(size_t)k]] = (int)k;
    cur.models.resize(keep_local.size());
    for (size_t k = 0; k < keep_local.size(); ++k) {
        const std::string& n = alive[(size_t)keep_local[(size_t)k]];
        cur.models[k].name = n;
        cur.models[k].gen = elo_gen_of_name(n);
        cur.models[k].path = dir + "/" + n + ".bin";
    }
    for (const auto& e : store) {
        auto ia = std::lower_bound(names.begin(), names.end(), e.a);
        auto ib = std::lower_bound(names.begin(), names.end(), e.b);
        if (ia == names.end() || ib == names.end() || *ia != e.a || *ib != e.b) continue;
        const int u = idx_of_name[(size_t)(ia - names.begin())];
        const int v = idx_of_name[(size_t)(ib - names.begin())];
        if (u < 0 || v < 0) continue;
        const int lu = local_of[(size_t)u], lv = local_of[(size_t)v];
        if (lu < 0 || lv < 0) continue;  // 不在最大分量里
        EloEdge ee;
        ee.a = lu;
        ee.b = lv;
        ee.games = e.games;
        ee.wins_a = e.wins_a;
        ee.draws = e.draws;
        ee.score_a = e.score_a;
        cur.edges.push_back(ee);
        cur.models[(size_t)lu].games += e.games;
        cur.models[(size_t)lv].games += e.games;
    }
    cur.store_edges = (int)cur.edges.size();

    // ---- 5) 锚点 ----
    int anchor = 0;
    if (anchor_gen > 0) {
        for (size_t i = 0; i < cur.models.size(); ++i)
            if (cur.models[i].gen == anchor_gen) anchor = (int)i;
    }
    cur.anchor = anchor;

    fit_elo(cur.models, cur.edges, anchor, bootstrap, seed);
    return cur;
}

}  // namespace gk
