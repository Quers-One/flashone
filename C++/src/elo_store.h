// elo_store.h — Elo 对局「边库」：持久化 + 全局拟合
//
// 为什么要有边库（这是 2026-10-31 重构的核心）：
//   旧做法是「每次 elo 只打一个窗口 → 拟合出一段 → 用平移把它接到旧曲线上」。
//   平移对齐有个致命前提：新旧两段要有一个常数偏移，而实测差值散在 +60~+218，
//   并不存在这样的常数——于是拼出来的曲线形状取决于选哪个共有代做锚，等于在猜。
//   而且每接一段累积一次 ±45 Elo 的误差，接得越长越歪。
//
//   边库做法：**所有打过的对局边永久保存**，每次 elo 只往里追加新边，然后用
//   **全部边**做一次 Bradley–Terry 全局拟合。好处：
//     * 没有「分段对齐」这一步，也就没有对齐误差；
//     * 任意两个版本只要打过一局就永久留着信息，后打的边会反过来校正老点的位置；
//     * 跨段标定（老版本直打新版本）只是一条普通的边，加进去全局拟合自动把
//       中间那些链式累积的误差压下去——不需要任何特殊代码路径。
//
// 文件格式：<run>/elo_edges.jsonl，一行一条边（我们自己的格式，见 elo_store_line）：
//   {"a":"gen_081","b":"gen_085","games":30,"score_a":18.5,"wins_a":16,"draws":5,"ts":"..."}
//   规范化：存储时 a 恒为字典序较小的那个名字，score_a 相应换算。
//   → 于是 "085 vs 081" 和 "081 vs 085" 会被合并成同一条边，局数累加。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "elo.h"

namespace gk {

struct EloStoreEdge {
    std::string a, b;       // 版本名（规范化后 a < b）
    int games = 0;
    double score_a = 0.0;   // A 视角得分：胜=1、平=0.5
    int wins_a = 0, draws = 0;
    std::string ts;
};

// 规范化一条边：按字典序定方向，score/wins 跟着换手
EloStoreEdge elo_store_edge(std::string a, std::string b, int games, double score_a, int wins_a,
                            int draws, const std::string& ts);

// 读边库。坏行/缺字段的行跳过（手写解析，只对自家格式负责）。
std::vector<EloStoreEdge> elo_store_load(const std::string& path);

// 把 add 合并进 all：同一对 (a,b) 累加局数与得分（重打同一对 = 增加样本量，
// 统计上就是应该累加，而不是覆盖）。返回「全新边」的条数（用于打印"新增 N 条"）。
int elo_store_merge(std::vector<EloStoreEdge>& all, const std::vector<EloStoreEdge>& add);

// 全量写回（覆盖）。边库只有几 KB~几十 KB，全量重写最省心且天然去重。
bool elo_store_save(const std::string& path, const std::vector<EloStoreEdge>& all);

// 查询：这对版本已经打过多少局（0 = 没打过）。用于避免重复打已有边。
int elo_store_games(const std::vector<EloStoreEdge>& all, const std::string& a,
                    const std::string& b);

// ---- 全局拟合 ----
struct EloCurve {
    std::vector<EloModel> models;   // 升序（按 gen），已填 elo/lo/hi/games
    std::vector<EloEdge> edges;     // 下标引用 models
    int anchor = 0;
    // 诊断：边库里有多少版本被剔除了、为什么
    int dropped_missing = 0;        // 权重文件已不在（keep_gens 清掉的旧代）
    int dropped_disconnected = 0;   // 与主体不连通（孤立的一小段）
    int store_edges = 0;            // 参与拟合的边数
};

// 从边库构造曲线：
//   1) 收集所有版本 → 按 gen 升序；
//   2) 剔掉权重文件已不存在的版本（无法复现，且看板画出来也没意义）；
//   3) 并查集求最大连通分量，只保留它 —— BT 模型对不同分量之间没有可比刻度，
//      混在一起拟合会让两个分量各自漂到 ±∞ 之外还互相拉扯；
//   4) 全局拟合 + bootstrap 置信区间。
// anchor_gen <= 0 时锚点取「保留下来最老的一代」（曲线起点恒为 0，最直观）。
EloCurve elo_curve_from_store(const std::string& dir, const std::vector<EloStoreEdge>& store,
                              int anchor_gen, int bootstrap, uint64_t seed);

// 版本名 → 代数。"gen_085" → 85；解析不出返回 -1。
int elo_gen_of_name(const std::string& name);
std::string elo_name_of_gen(int gen);

}  // namespace gk
